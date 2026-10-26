"""Ablation grids.

(a) BM25 k1 x b                          quality + mean BM25 latency
(b) chunk size {128, 256, 512, none}     BM25 only (index rebuilt per size; dense embeddings are
                                         computed once for the default chunking, see notes)
(c) fusion: RRF k, weighted alpha x norm  candidates retrieved once per query, fused many ways
(d) HNSW M x ef                          recall@10 vs exact search, nDCG@10, ANN latency
(e) rerank depth                         cross-encoder scores for the top 100 computed once
                                         (scores are per pair, so reranking top-k with them is
                                         identical to rerunning); latency measured for real on
                                         ``ABLATION_LATENCY_SAMPLE`` queries per depth

All numbers come from running the system; nothing is interpolated or estimated.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.core.bm25 import BM25Scorer
from app.core.custom_hnsw import CustomHNSW
from app.core.fusion import rrf, weighted_fusion
from app.core.inverted_index import InvertedIndex, build_index, iter_chunks
from app.core.pipeline import SearchPipeline
from app.core.vector_store import CustomHNSWStore, DenseSearcher, ExactStore, QdrantStore
from app.eval.datasets import Dataset
from app.eval.latency import summarize
from app.eval.metrics import ndcg_at_k, recall_at_k
from app.eval.runner import select_queries

log = logging.getLogger(__name__)
Progress = Callable[[str], None]

BM25_K1 = (0.6, 0.9, 1.2, 1.5, 2.0)
BM25_B = (0.25, 0.5, 0.75, 0.9)
CHUNK_SIZES: tuple[int | None, ...] = (128, 256, 512, None)
RRF_KS = (10, 30, 60, 100)
ALPHAS = tuple(round(a / 10, 1) for a in range(11))
NORMS = ("minmax", "zscore")
HNSW_MS = (8, 16, 32)
HNSW_EFS = (16, 32, 64, 128, 256)
RERANK_KS = (10, 20, 50, 100)
ALL = ("bm25_grid", "chunk_size", "fusion", "hnsw", "rerank_top_k")


def _say(progress: Progress | None, msg: str) -> None:
    log.info(msg)
    if progress:
        progress(msg)


def _quality(rankings: Sequence[Sequence[str]], qrels: Sequence[dict[str, int]]) -> tuple[float, float]:
    n = max(1, len(rankings))
    nd = sum(ndcg_at_k(r, q, 10) for r, q in zip(rankings, qrels, strict=True)) / n
    rc = sum(recall_at_k(r, q, 100) for r, q in zip(rankings, qrels, strict=True)) / n
    return nd, rc


# ----------------------------------------------------------------------------- (a) BM25
def bm25_grid(
    pipeline: SearchPipeline,
    queries: list[tuple[str, str]],
    qrels: list[dict[str, int]],
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    """nDCG@10 / Recall@100 / mean latency for every (k1, b)."""
    parsed = [pipeline.index.tokenizer.parse_query(t) for _, t in queries]
    rows = []
    for k1 in BM25_K1:
        for b in BM25_B:
            params = replace(pipeline.defaults, k1=k1, b=b)
            rankings, times = [], []
            for pq in parsed:
                t0 = time.perf_counter()
                cand = pipeline.bm25_candidates(pq, params, 100)
                times.append((time.perf_counter() - t0) * 1000)
                rankings.append([d for d, _ in cand.ranked])
            nd, rc = _quality(rankings, qrels)
            rows.append(
                {
                    "k1": k1,
                    "b": b,
                    "ndcg_at_10": nd,
                    "recall_at_100": rc,
                    "mean_ms": float(np.mean(times)) if times else 0.0,
                }
            )
            _say(progress, f"ablation bm25 k1={k1} b={b}: nDCG@10={nd:.4f}")
    return rows


# ----------------------------------------------------------------------------- (b) chunking
def chunk_size_ablation(
    pipeline: SearchPipeline,
    dataset: Dataset,
    settings: Settings,
    queries: list[tuple[str, str]],
    qrels: list[dict[str, int]],
    work_dir: Path,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    """Rebuild the BM25 index per chunk size and evaluate doc-level (max-aggregated) BM25."""
    rows = []
    tok = pipeline.index.tokenizer
    parsed = [tok.parse_query(t) for _, t in queries]
    for size in CHUNK_SIZES:
        cs = size or 0
        out = work_dir / f"chunk_{cs}"
        if out.exists():
            shutil.rmtree(out)
        docs = ((d.doc_id, d.title, d.text) for d in dataset.corpus.values())
        stats = build_index(
            iter_chunks(docs, cs, settings.chunk_overlap),
            out,
            tok,
            chunk_size=cs,
            chunk_overlap=settings.chunk_overlap,
            skip_interval=settings.skip_interval,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
        idx = InvertedIndex(out)
        scorer = BM25Scorer(idx, settings.bm25_k1, settings.bm25_b, settings.phrase_boost)
        rankings, times = [], []
        for pq in parsed:
            t0 = time.perf_counter()
            res = scorer.search(pq, 100)
            times.append((time.perf_counter() - t0) * 1000)
            rankings.append([h.doc_id for h in res.hits])
        idx.close()
        shutil.rmtree(out, ignore_errors=True)
        nd, rc = _quality(rankings, qrels)
        rows.append(
            {
                "chunk_size": size,
                "ndcg_at_10": nd,
                "recall_at_100": rc,
                "n_chunks": stats.n_chunks,
                "index_bytes": stats.index_bytes,
                "build_seconds": stats.build_seconds,
                "mean_ms": float(np.mean(times)) if times else 0.0,
            }
        )
        _say(progress, f"ablation chunk_size={size}: nDCG@10={nd:.4f} chunks={stats.n_chunks}")
    return rows


# ----------------------------------------------------------------------------- (c) fusion
def fusion_ablation(
    pipeline: SearchPipeline,
    queries: list[tuple[str, str]],
    qrels: list[dict[str, int]],
    progress: Progress | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """RRF k sweep and weighted alpha x norm sweep over the same candidate lists."""
    params = pipeline.defaults
    sparse_lists, dense_lists = [], []
    for _, text in queries:
        pq = pipeline.index.tokenizer.parse_query(text)
        sparse_lists.append(pipeline.bm25_candidates(pq, params, params.candidate_depth).ranked)
        vec, _ = pipeline.embed_query(text, use_cache=True)
        dense_lists.append(pipeline.dense_candidates(vec, params, params.candidate_depth).ranked)
    rrf_rows = []
    for k in RRF_KS:
        rankings = [[d for d, _ in rrf([s, d], k=k)] for s, d in zip(sparse_lists, dense_lists, strict=True)]
        nd, rc = _quality(rankings, qrels)
        rrf_rows.append({"rrf_k": k, "ndcg_at_10": nd, "recall_at_100": rc})
    _say(progress, "ablation fusion: rrf k sweep done")
    alpha_rows = []
    for norm in NORMS:
        for a in ALPHAS:
            rankings = [
                [d for d, _ in weighted_fusion(s, d, a, norm)]  # type: ignore[arg-type]
                for s, d in zip(sparse_lists, dense_lists, strict=True)
            ]
            nd, rc = _quality(rankings, qrels)
            alpha_rows.append({"norm": norm, "alpha": a, "ndcg_at_10": nd, "recall_at_100": rc})
    _say(progress, "ablation fusion: alpha sweep done")
    return rrf_rows, alpha_rows


# ----------------------------------------------------------------------------- (d) HNSW
HnswFactory = Callable[[int], DenseSearcher]


def default_hnsw_factory(
    pipeline: SearchPipeline, settings: Settings, embeddings: np.ndarray, progress: Progress | None = None
) -> HnswFactory:
    """Qdrant collections per M when the backend is Qdrant, otherwise the custom HNSW."""
    payloads = [
        {"doc_id": d, "chunk_id": f"{d}#{n}"}
        for d, n in zip(pipeline.index.chunk_doc, pipeline.index.chunk_no, strict=True)
    ]

    def factory(m: int) -> DenseSearcher:
        if isinstance(pipeline.dense, QdrantStore):
            store = pipeline.dense.with_collection(settings.collection_name(m=m))
            if store.count() != embeddings.shape[0]:
                _say(progress, f"ablation hnsw: building qdrant collection M={m}")
                store.ensure_collection(embeddings.shape[1], m, settings.hnsw_ef_construct, recreate=True)
                store.upsert(embeddings, payloads)
            store.wait_indexed()
            return store
        _say(progress, f"ablation hnsw: building custom HNSW M={m}")
        g = CustomHNSW(
            dim=embeddings.shape[1],
            m=m,
            ef_construction=settings.hnsw_ef_construct,
            seed=settings.random_seed,
            capacity=embeddings.shape[0],
        )
        g.add_items(np.asarray(embeddings))
        return CustomHNSWStore(g)

    return factory


def hnsw_ablation(
    pipeline: SearchPipeline,
    settings: Settings,
    queries: list[tuple[str, str]],
    qrels: list[dict[str, int]],
    factory: HnswFactory,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    """Recall@10 vs exact search, doc-level nDCG@10 and ANN latency for every (M, ef)."""
    if pipeline.embeddings is None:
        raise RuntimeError("HNSW ablation needs the chunk embedding matrix (embeddings.npy)")
    emb = np.asarray(pipeline.embeddings, dtype=np.float32)
    exact = ExactStore(emb)
    vecs = [pipeline.embed_query(t, use_cache=True)[0] for _, t in queries]
    truth = [{c for c, _ in exact.search(v, 10)} for v in vecs]
    chunk_doc = pipeline.index.chunk_doc
    k_chunks = 10 * max(1, int(np.ceil(pipeline.index.n_chunks / max(1, pipeline.index.n_docs))))
    rows = []
    default_collection = settings.collection_name()
    for m in HNSW_MS:
        store = factory(m)
        for ef in HNSW_EFS:
            times, recalls, rankings = [], [], []
            for v, tr in zip(vecs, truth, strict=True):
                t0 = time.perf_counter()
                hits = store.search(v, max(k_chunks, 10), ef)
                times.append((time.perf_counter() - t0) * 1000)
                recalls.append(len({c for c, _ in hits[:10]} & tr) / max(1, len(tr)))
                seen: list[str] = []
                for c, _ in hits:
                    d = chunk_doc[c]
                    if d not in seen:
                        seen.append(d)
                rankings.append(seen)
            nd = sum(ndcg_at_k(r, q, 10) for r, q in zip(rankings, qrels, strict=True)) / max(1, len(qrels))
            lat = summarize(times)
            rows.append(
                {
                    "m": m,
                    "ef_construct": settings.hnsw_ef_construct,
                    "ef": ef,
                    "recall_at_10_vs_exact": float(np.mean(recalls)) if recalls else 0.0,
                    "ndcg_at_10": nd,
                    "mean_ms": lat["mean"],
                    "p50_ms": lat["p50"],
                    "p95_ms": lat["p95"],
                }
            )
            _say(progress, f"ablation hnsw M={m} ef={ef}: recall@10={rows[-1]['recall_at_10_vs_exact']:.4f}")
        if isinstance(store, QdrantStore) and store.collection != default_collection:
            try:
                store.client.delete_collection(store.collection)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not delete %s: %s", store.collection, exc)
    return rows


# ----------------------------------------------------------------------------- (e) rerank depth
def rerank_ablation(
    pipeline: SearchPipeline,
    queries: list[tuple[str, str]],
    qrels: list[dict[str, int]],
    latency_sample: int,
    progress: Progress | None = None,
) -> list[dict[str, Any]]:
    """Quality and latency of reranking the top-k of hybrid_rrf for k in RERANK_KS."""
    max_k = max(RERANK_KS)
    params = pipeline.defaults
    fused_lists: list[list[str]] = []
    score_maps: list[dict[str, float]] = []
    for i, (_, text) in enumerate(queries, 1):
        out = pipeline.search(text, "hybrid_rrf", params, use_result_cache=False, use_embedding_cache=True)
        fused = [r.doc_id for r in out.ranking]
        head = fused[:max_k]
        score_maps.append(dict(zip(head, pipeline.rerank_scores(text, head), strict=True)))
        fused_lists.append(fused)
        if i % 25 == 0:
            _say(progress, f"ablation rerank: scored {i}/{len(queries)} queries")
    rows = []
    sample = [t for _, t in queries[:latency_sample]]
    for k in RERANK_KS:
        rankings = []
        for fused, sm in zip(fused_lists, score_maps, strict=True):
            head = sorted(fused[:k], key=lambda d: (-sm[d], d))
            rankings.append(head + fused[k:])
        nd, rc = _quality(rankings, qrels)
        stage, total = [], []
        p = replace(params, rerank_top_k=k)
        for text in sample:
            out = pipeline.search(
                text, "hybrid_rrf_rerank", p, use_result_cache=False, use_embedding_cache=False
            )
            stage.append(out.timings.rerank_ms)
            total.append(out.timings.total_ms)
        s, t = summarize(stage), summarize(total)
        rows.append(
            {
                "rerank_top_k": k,
                "ndcg_at_10": nd,
                "recall_at_100": rc,
                "mean_ms": s["mean"],
                "p50_ms": s["p50"],
                "p95_ms": s["p95"],
                "total_mean_ms": t["mean"],
                "total_p95_ms": t["p95"],
            }
        )
        _say(progress, f"ablation rerank top_k={k}: nDCG@10={nd:.4f} p95={s['p95']:.1f}ms")
    return rows


# ----------------------------------------------------------------------------- driver
def run_ablations(
    pipeline: SearchPipeline,
    dataset: Dataset,
    settings: Settings,
    *,
    which: Sequence[str] = ALL,
    max_queries: int | None = None,
    progress: Progress | None = None,
    hnsw_factory: HnswFactory | None = None,
) -> dict[str, Any]:
    """Run the selected ablations; stages whose dependencies are missing are skipped with a note."""
    n = settings.ablation_max_queries if max_queries is None else max_queries
    queries = select_queries(dataset, n)
    qrels = [dataset.qrels.get(qid, {}) for qid, _ in queries]
    out: dict[str, Any] = {
        "dataset": settings.dataset,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_queries": len(queries),
        "bm25_grid": [],
        "chunk_size": [],
        "rrf_k": [],
        "alpha_sweep": [],
        "hnsw": [],
        "rerank_top_k": [],
        "notes": [
            "Chunk-size ablation evaluates BM25 only (the dense index is embedded once at the "
            "default CHUNK_SIZE to keep first-run time bounded).",
            "HNSW recall@10 is measured at chunk level against an exact NumPy scan of the same "
            "embeddings; latency is the ANN call only.",
            "Rerank latency is measured on the first ABLATION_LATENCY_SAMPLE queries per depth; "
            "quality uses all queries.",
        ],
    }
    dense_ok = pipeline.embedder is not None and pipeline.dense is not None
    if "bm25_grid" in which:
        out["bm25_grid"] = bm25_grid(pipeline, queries, qrels, progress)
    if "chunk_size" in which:
        # temp dir, not DATA_DIR: the api container mounts the data volume read-only
        work = Path(tempfile.mkdtemp(prefix="hybridir-chunks-"))
        out["chunk_size"] = chunk_size_ablation(pipeline, dataset, settings, queries, qrels, work, progress)
        shutil.rmtree(work, ignore_errors=True)
    if "fusion" in which:
        if dense_ok:
            out["rrf_k"], out["alpha_sweep"] = fusion_ablation(pipeline, queries, qrels, progress)
        else:
            out["notes"].append("Fusion ablation skipped: dense stage unavailable.")
    if "hnsw" in which:
        if dense_ok and pipeline.embeddings is not None:
            factory = hnsw_factory or default_hnsw_factory(
                pipeline, settings, np.asarray(pipeline.embeddings, dtype=np.float32), progress
            )
            out["hnsw"] = hnsw_ablation(pipeline, settings, queries, qrels, factory, progress)
        else:
            out["notes"].append("HNSW ablation skipped: dense stage or embeddings unavailable.")
    if "rerank_top_k" in which:
        if dense_ok and pipeline.reranker is not None:
            out["rerank_top_k"] = rerank_ablation(
                pipeline, queries, qrels, settings.ablation_latency_sample, progress
            )
        else:
            out["notes"].append("Rerank ablation skipped: reranker or dense stage unavailable.")
    return out
