"""Score explanations: why did document D get its rank for query Q under each stage?

Produces the per-term BM25 decomposition (tf, idf, length norm, term score) for the
document's best chunk, the dense cosine similarity, the exact numbers RRF and weighted fusion
used, the cross-encoder score, and the document's rank under every strategy (rank shift
relative to BM25).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from app.core.bm25 import BM25Scorer
from app.core.fusion import list_stats, missing_floor, normalize_scores
from app.core.inverted_index import bm25_term_weight
from app.core.pipeline import EffectiveParams, SearchPipeline, StageUnavailable

RANK_STRATEGIES = (
    "bm25",
    "dense",
    "hybrid_rrf",
    "hybrid_weighted",
    "hybrid_rrf_rerank",
    "hybrid_weighted_rerank",
)


class DocumentNotFound(KeyError):
    """The requested doc id is not in the corpus."""


def _stats_dict(scores: list[float]) -> dict[str, float | int]:
    st = list_stats(scores)
    return {"n": st.n, "min": st.min, "max": st.max, "mean": st.mean, "std": st.std}


def explain(pipeline: SearchPipeline, query: str, doc_id: str, params: EffectiveParams) -> dict[str, Any]:
    """Build the full explanation payload (shape of :class:`app.schemas.ExplainResponse`)."""
    if doc_id not in pipeline.docs:
        raise DocumentNotFound(doc_id)
    index = pipeline.index
    tok = index.tokenizer
    doc = pipeline.docs[doc_id]
    parsed = tok.parse_query(query)
    qtf = Counter(parsed.terms)
    scorer = BM25Scorer(index, params.k1, params.b, pipeline.phrase_boost)
    weights = scorer.query_weights(parsed.terms)

    # ---------------------------------------------------------------- BM25 breakdown
    chunk_ids = index.doc_chunks.get(doc_id, [])
    best_chunk: int | None = None
    best_score = float("-inf")
    for cid in chunk_ids:
        s = scorer.score_chunk(weights, cid)
        if s > best_score:
            best_chunk, best_score = cid, s
    terms_out: list[dict[str, Any]] = []
    bm25_total = 0.0
    if best_chunk is not None:
        dl = int(index.doc_lengths[best_chunk])
        for term in sorted(qtf):
            tf = index.tf(term, best_chunk) if index.has_term(term) else 0
            df = index.df(term)
            idf = index.idf(term) if df else 0.0
            length_norm = 1.0 - params.b + params.b * dl / index.avgdl if index.avgdl else 1.0
            tf_comp = bm25_term_weight(tf, dl, index.avgdl, params.k1, params.b)
            score = qtf[term] * idf * tf_comp
            bm25_total += score
            terms_out.append(
                {
                    "term": term,
                    "qtf": qtf[term],
                    "tf": tf,
                    "df": df,
                    "idf": idf,
                    "doc_len": dl,
                    "avg_doc_len": index.avgdl,
                    "length_norm": length_norm,
                    "tf_component": tf_comp,
                    "score": score,
                }
            )
    phrase_boost = 0.0
    if best_chunk is not None and pipeline.phrase_boost > 0:
        for phrase in parsed.phrases:
            if index.phrase_match(phrase, best_chunk):
                phrase_boost += pipeline.phrase_boost * sum(index.idf(t) for t in phrase)
    bm25_total += phrase_boost

    # ---------------------------------------------------------------- dense cosine
    cosine: float | None = None
    dense_error: str | None = None
    qvec: np.ndarray | None = None
    try:
        qvec, _ = pipeline.embed_query(query, use_cache=True)
        if pipeline.embeddings is not None and chunk_ids:
            sims = np.asarray(pipeline.embeddings[chunk_ids], dtype=np.float32) @ qvec
            cosine = float(sims.max())
    except StageUnavailable as exc:
        dense_error = exc.detail

    # ---------------------------------------------------------------- fusion math
    depth = params.candidate_depth
    sparse = pipeline.bm25_candidates(parsed, params, depth)
    fusion: dict[str, Any] | None = None
    if qvec is not None:
        try:
            dense = pipeline.dense_candidates(qvec, params, depth)
        except StageUnavailable as exc:
            dense_error = exc.detail
            dense = None
        if dense is not None:
            b_rank = next((i for i, (d, _) in enumerate(sparse.ranked, 1) if d == doc_id), None)
            d_rank = next((i for i, (d, _) in enumerate(dense.ranked, 1) if d == doc_id), None)
            rrf_b = 1.0 / (params.rrf_k + b_rank) if b_rank else 0.0
            rrf_d = 1.0 / (params.rrf_k + d_rank) if d_rank else 0.0
            ns = normalize_scores(sparse.ranked, params.norm)
            nd = normalize_scores(dense.ranked, params.norm)
            b_norm = ns.get(doc_id, missing_floor(ns, params.norm))
            d_norm = nd.get(doc_id, missing_floor(nd, params.norm))
            fusion = {
                "candidate_depth": depth,
                "rrf_k": params.rrf_k,
                "bm25_rank": b_rank,
                "dense_rank": d_rank,
                "rrf_bm25_term": rrf_b,
                "rrf_dense_term": rrf_d,
                "rrf_score": rrf_b + rrf_d,
                "alpha": params.alpha,
                "norm": params.norm,
                "bm25_raw": dict(sparse.ranked).get(doc_id),
                "dense_raw": dict(dense.ranked).get(doc_id),
                "bm25_norm": b_norm,
                "dense_norm": d_norm,
                "bm25_list": _stats_dict([s for _, s in sparse.ranked]),
                "dense_list": _stats_dict([s for _, s in dense.ranked]),
                "weighted_score": params.alpha * d_norm + (1 - params.alpha) * b_norm,
            }

    # ---------------------------------------------------------------- reranker
    rerank_score: float | None = None
    rerank_error: str | None = None
    try:
        rerank_score = pipeline.rerank_scores(query, [doc_id])[0]
    except StageUnavailable as exc:
        rerank_error = exc.detail

    # ---------------------------------------------------------------- ranks everywhere
    ranks: dict[str, int | None] = {}
    for strategy in RANK_STRATEGIES:
        try:
            outcome = pipeline.search(query, strategy, params)
            ranks[strategy] = next((i for i, r in enumerate(outcome.ranking, 1) if r.doc_id == doc_id), None)
        except StageUnavailable:
            ranks[strategy] = None
    base = ranks.get("bm25")
    delta = {s: (base - r) if (base is not None and r is not None) else None for s, r in ranks.items()}

    term_set = set(parsed.terms)
    return {
        "query": query,
        "doc_id": doc_id,
        "title": doc.title,
        "text": doc.text,
        "title_highlights": [{"start": a, "end": b} for a, b in tok.highlight_spans(doc.title, term_set)],
        "text_highlights": [{"start": a, "end": b} for a, b in tok.highlight_spans(doc.text, term_set)],
        "chunk_id": best_chunk,
        "query_terms": sorted(term_set),
        "phrases": parsed.phrases,
        "params": params.to_dict(),
        "bm25_score": bm25_total,
        "phrase_boost": phrase_boost,
        "terms": terms_out,
        "cosine": cosine,
        "dense_error": dense_error,
        "fusion": fusion,
        "rerank_score": rerank_score,
        "rerank_error": rerank_error,
        "ranks": ranks,
        "delta_vs_bm25": delta,
    }
