"""Idempotent bootstrap: models -> dataset -> inverted index -> embeddings -> vector store -> eval.

Usage::

    python -m scripts.seed            # inside the backend image (docker compose runs this)
    FORCE_RESEED=1 python -m scripts.seed

Idempotency: a marker file (``<DATA_DIR>/<DATASET>/.seed_complete.json``) stores a fingerprint
of every setting that affects the built artifacts. If the fingerprint matches and the files
exist, the build steps are skipped; the Qdrant collection is re-checked (its volume may have
been reset independently) and the baseline evaluation runs only if ``results/summary.json``
is missing or belongs to another dataset. ``FORCE_RESEED=1`` rebuilds everything.

Model weights are downloaded into ``HF_HOME`` (the ``hf_cache`` named volume) by this
service; the API container then runs with ``HF_HUB_OFFLINE=1`` and reads them from there.
"""

from __future__ import annotations

import logging
import random
import sys
import time
import urllib.request
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.core.cache import Cache
from app.core.custom_hnsw import CustomHNSW
from app.core.dense import Embedder
from app.core.inverted_index import InvertedIndex, build_index, iter_chunks
from app.core.tokenizer import Tokenizer, TokenizerConfig
from app.core.vector_store import QdrantStore
from app.deps import build_state
from app.eval import report
from app.eval.datasets import download_dataset, iter_corpus, load_dataset
from app.eval.runner import run_evaluation
from app.logging_setup import setup_logging

log = logging.getLogger("seed")


def step(n: int, total: int, msg: str) -> None:
    """Visible progress banner."""
    log.info("[%d/%d] %s", n, total, msg)


def wait_for_services(settings: Settings) -> None:
    """Block until Redis (and Qdrant when it is the dense backend) answer, or exit 1."""
    deadline = time.monotonic() + settings.seed_wait_timeout_seconds
    cache = Cache(settings.redis_url, timeout_seconds=2.0)
    need_qdrant = settings.dense_backend == "qdrant"
    while True:
        redis_ok, redis_detail = cache.ping()
        qdrant_ok = True
        if need_qdrant:
            try:
                with urllib.request.urlopen(f"{settings.qdrant_url}/readyz", timeout=3) as r:
                    qdrant_ok = r.status == 200
            except Exception:  # noqa: BLE001
                qdrant_ok = False
        if redis_ok and qdrant_ok:
            log.info("services ready (redis=%s, qdrant=%s)", redis_detail, "ok" if need_qdrant else "unused")
            return
        if time.monotonic() > deadline:
            log.error("timed out waiting for services (redis_ok=%s qdrant_ok=%s)", redis_ok, qdrant_ok)
            sys.exit(1)
        time.sleep(2)


def ensure_models(settings: Settings) -> None:
    """Make sure both model snapshots are in HF_HOME (download once, offline afterwards)."""
    from huggingface_hub import snapshot_download

    # skip export formats sentence-transformers never loads with torch on CPU (smaller download)
    ignore = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "coreml/*"]
    for repo in (settings.embed_model, settings.rerank_model):
        try:
            path = snapshot_download(repo, local_files_only=True, ignore_patterns=ignore)
            log.info("model %s cached at %s", repo, path)
        except Exception:  # noqa: BLE001
            log.info("downloading model %s into HF_HOME (first run only)", repo)
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    # snapshot_download resumes completed files, so retrying is safe after
                    # interrupted connections and avoids restarting large model downloads.
                    path = snapshot_download(
                        repo,
                        ignore_patterns=ignore,
                        max_workers=1,
                    )
                    log.info("model %s downloaded to %s", repo, path)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 3:
                        raise RuntimeError(
                            f"unable to download model {repo} after {attempt} attempts; "
                            "check outbound HTTPS access to huggingface.co and retry"
                        ) from exc
                    delay = 2**attempt
                    log.warning(
                        "download of model %s failed (attempt %d/3): %s; retrying in %ds",
                        repo,
                        attempt,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            else:
                # The loop either succeeds or raises on its final attempt.
                raise RuntimeError(f"unable to download model {repo}") from last_error


def fingerprint(settings: Settings) -> dict[str, Any]:
    """Every setting that changes the built artifacts."""
    return {
        "dataset": settings.dataset,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "skip_interval": settings.skip_interval,
        "tokenizer": TokenizerConfig(
            settings.use_stemmer, settings.use_stopwords, settings.min_token_len
        ).to_dict(),
        "bm25_build": [settings.bm25_k1, settings.bm25_b],
        "embed_model": settings.embed_model,
        "dense_backend": settings.dense_backend,
        "hnsw": [settings.hnsw_m, settings.hnsw_ef_construct],
    }


def eta_logger(label: str) -> Any:
    """Progress callback printing done/total, rate and ETA at most every 5 seconds."""
    state = {"next": 0.0}

    def cb(done: int, total: int, elapsed: float) -> None:
        if elapsed < state["next"] and done < total:
            return
        state["next"] = elapsed + 5.0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        log.info("  %s %d/%d (%.1f/s, ETA %.0fs)", label, done, total, rate, eta)

    return cb


def export_k6_queries(settings: Settings) -> None:
    """Write a sample of test queries for the k6 load test to RESULTS_DIR/queries.json."""
    ds = load_dataset(settings.dataset, settings.raw_dir)
    texts = [ds.queries[q] for q in sorted(ds.queries)]
    rng = random.Random(settings.random_seed)
    sample = rng.sample(texts, min(settings.k6_query_sample, len(texts)))
    report.write_json(
        settings.results_dir / "queries.json",
        {"dataset": settings.dataset, "count": len(sample), "queries": sample},
    )
    log.info("exported %d queries for k6 to %s", len(sample), settings.results_dir / "queries.json")


def main() -> int:
    """Entry point; returns the process exit code."""
    settings = get_settings()
    setup_logging(settings.log_level)
    t_start = time.perf_counter()
    total = 8
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    step(1, total, "waiting for Redis / Qdrant")
    wait_for_services(settings)

    step(2, total, f"ensuring model weights in HF cache ({settings.embed_model}, {settings.rerank_model})")
    ensure_models(settings)

    fp = fingerprint(settings)
    marker = report.read_json(settings.marker_path)
    built = (
        not settings.force_reseed
        and marker is not None
        and marker.get("fingerprint") == fp
        and (settings.index_dir / "postings.bin").exists()
        and settings.embeddings_path.exists()
    )
    if built:
        log.info("marker %s matches; skipping download/index/embedding", settings.marker_path)

    step(3, total, f"dataset {settings.dataset}")
    download_dataset(settings.dataset, settings.raw_dir, settings.beir_base_url)

    tok = Tokenizer(TokenizerConfig(settings.use_stemmer, settings.use_stopwords, settings.min_token_len))

    def chunks() -> Any:
        return iter_chunks(
            ((d.doc_id, d.title, d.text) for d in iter_corpus(settings.raw_dir)),
            settings.chunk_size,
            settings.chunk_overlap,
        )

    step(4, total, "building inverted index")
    index_stats: dict[str, Any]
    if built:
        index_stats = InvertedIndex(settings.index_dir).stats()
    else:
        stats = build_index(
            chunks(),
            settings.index_dir,
            tok,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            skip_interval=settings.skip_interval,
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
        index_stats = stats.to_dict()
        log.info(
            "index: %d docs, %d chunks, %d terms, %.1f MB, %.1fs",
            stats.n_docs,
            stats.n_chunks,
            stats.vocab_size,
            stats.index_bytes / 1e6,
            stats.build_seconds,
        )

    step(5, total, "embedding chunks")
    if built:
        emb = np.load(settings.embeddings_path, mmap_mode="r")
        prev = report.read_json(settings.results_dir / "index_stats.json") or {}
        emb_seconds = prev.get("embedding_seconds")
    else:
        texts = [c.text for c in chunks()]
        embedder = Embedder(
            settings.embed_model, batch_size=settings.embed_batch_size, torch_threads=settings.torch_threads
        )
        t0 = time.perf_counter()
        emb = embedder.encode_documents(texts, progress=eta_logger("embedded"))
        emb_seconds = time.perf_counter() - t0
        tmp = settings.embeddings_path.with_suffix(".tmp.npy")
        np.save(tmp, emb.astype(np.float32))
        tmp.replace(settings.embeddings_path)
        log.info("embedded %d chunks in %.1fs", emb.shape[0], emb_seconds)
    index_stats.update(dataset=settings.dataset, embedding_seconds=emb_seconds, n_vectors=int(emb.shape[0]))
    report.write_index_stats(settings.results_dir, index_stats)

    step(6, total, f"dense backend: {settings.dense_backend}")
    if settings.dense_backend == "qdrant":
        store = QdrantStore(settings.qdrant_url, settings.collection_name(), settings.qdrant_timeout_seconds)
        if settings.force_reseed or not built or store.count() != emb.shape[0]:
            idx = InvertedIndex(settings.index_dir)
            payloads = [
                {"doc_id": d, "chunk_id": f"{d}#{n}"}
                for d, n in zip(idx.chunk_doc, idx.chunk_no, strict=True)
            ]
            store.ensure_collection(
                int(emb.shape[1]), settings.hnsw_m, settings.hnsw_ef_construct, recreate=True
            )
            store.upsert(np.asarray(emb), payloads, progress=eta_logger("upserted"))
        log.info("waiting for qdrant HNSW indexing of %s", store.collection)
        store.wait_indexed()
    else:
        if settings.force_reseed or not built or not settings.custom_hnsw_path.exists():
            g = CustomHNSW(
                dim=int(emb.shape[1]),
                m=settings.hnsw_m,
                ef_construction=settings.hnsw_ef_construct,
                ef_search=settings.hnsw_ef,
                seed=settings.random_seed,
                capacity=int(emb.shape[0]),
            )
            arr = np.asarray(emb, dtype=np.float32)
            cb = eta_logger("hnsw inserted")
            t0 = time.perf_counter()
            for start in range(0, arr.shape[0], 1000):
                g.add_items(arr[start : start + 1000])
                cb(min(start + 1000, arr.shape[0]), arr.shape[0], time.perf_counter() - t0)
            g.save(settings.custom_hnsw_path)
            log.info("custom HNSW saved to %s", settings.custom_hnsw_path)

    report.write_json(settings.marker_path, {"fingerprint": fp, "completed_at": time.time()})

    step(7, total, "exporting k6 query sample")
    export_k6_queries(settings)

    step(8, total, "baseline evaluation")
    summary = report.read_json(settings.results_dir / "summary.json")
    if settings.force_reseed or not built or not summary or summary.get("dataset") != settings.dataset:
        state = build_state(settings)
        if state.pipeline is None or state.load_errors:
            log.error("cannot evaluate, components failed to load: %s", state.load_errors)
            return 1
        run_evaluation(state.pipeline, settings, include_ablations=settings.ablations_on_seed, progress=None)
        state.close()
        log.info("results written to %s", settings.results_dir)
    else:
        log.info("results for %s already present; skipping evaluation (make eval to rerun)", settings.dataset)

    log.info("seed complete in %.1f min", (time.perf_counter() - t_start) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
