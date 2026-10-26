"""POST /search and POST /compare."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.api.errors import require_pipeline, run_blocking
from app.core.pipeline import EffectiveParams, SearchOutcome, SearchPipeline, StageUnavailable
from app.deps import AppState, get_state
from app.logging_setup import request_id_var
from app.schemas import CompareRequest, CompareResponse, ErrorInfo, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])
ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorInfo},
    422: {"model": ErrorInfo},
    503: {"model": ErrorInfo},
    504: {"model": ErrorInfo},
}
BASELINE = "bm25"


def outcome_to_response(pipeline: SearchPipeline, outcome: SearchOutcome, top_k: int) -> dict[str, Any]:
    """Convert a pipeline outcome into the SearchResponse payload (top_k hits)."""
    hits = []
    for rank, r in enumerate(outcome.ranking[:top_k], start=1):
        doc = pipeline.docs.get(r.doc_id)
        hits.append(
            {
                "rank": rank,
                "doc_id": r.doc_id,
                "title": doc.title if doc else "",
                "snippet": pipeline.snippet(r.doc_id, outcome.parsed_terms),
                "score": r.score,
                "scores": {"bm25": r.bm25, "dense": r.dense, "fusion": r.fusion, "rerank": r.rerank},
                "ranks": {k: r.ranks.get(k) for k in ("bm25", "dense", "fusion", "rerank")},
                "chunk_id": r.chunk_id,
            }
        )
    return {
        "query": outcome.query,
        "strategy": outcome.strategy,
        "top_k": top_k,
        "params": outcome.params.to_dict(),
        "hits": hits,
        "total_candidates": len(outcome.ranking),
        "query_terms": outcome.parsed_terms,
        "timings": outcome.timings.to_dict(),
        "cache": {
            "result_cache_hit": outcome.result_cache_hit,
            "embedding_cache_hit": outcome.embedding_cache_hit,
        },
        "dense_backend": pipeline.dense.name if pipeline.dense else "none",
    }


@router.post("/search", response_model=SearchResponse, responses=ERRORS)
async def search(req: SearchRequest, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Run one strategy and return the top_k hits with per-stage scores and timings."""
    pipeline = require_pipeline(state)
    params = pipeline.resolve(req.params.model_dump())
    outcome = await run_blocking(
        state,
        pipeline.search,
        req.query,
        req.strategy,
        params,
        use_result_cache=req.use_cache,
        use_embedding_cache=req.use_cache,
    )
    return outcome_to_response(pipeline, outcome, req.top_k)


def _compare(pipeline: SearchPipeline, req: CompareRequest, params: EffectiveParams) -> dict[str, Any]:
    outcomes: dict[str, SearchOutcome] = {}
    results: list[dict[str, Any]] = []
    for strategy in req.strategies:
        try:
            outcomes[strategy] = pipeline.search(req.query, strategy, params)
            results.append(
                {
                    "strategy": strategy,
                    "response": outcome_to_response(pipeline, outcomes[strategy], req.top_k),
                    "error": None,
                }
            )
        except StageUnavailable as exc:
            results.append(
                {
                    "strategy": strategy,
                    "response": None,
                    "error": {
                        "error": "stage_unavailable",
                        "detail": exc.detail,
                        "stage": exc.stage,
                        "request_id": request_id_var.get(),
                    },
                }
            )
    baseline = outcomes.get(BASELINE) or pipeline.search(req.query, BASELINE, params)
    full_rank: dict[str, dict[str, int]] = {
        s: {r.doc_id: i for i, r in enumerate(o.ranking, 1)} for s, o in outcomes.items()
    }
    base_rank = {r.doc_id: i for i, r in enumerate(baseline.ranking, 1)}
    seen: list[str] = []
    for o in outcomes.values():
        for r in o.ranking[: req.top_k]:
            if r.doc_id not in seen:
                seen.append(r.doc_id)
    rows = []
    for doc_id in seen:
        ranks = {s: full_rank[s].get(doc_id) for s in req.strategies if s in full_rank}
        b = base_rank.get(doc_id)
        rows.append(
            {
                "doc_id": doc_id,
                "title": pipeline.docs[doc_id].title if doc_id in pipeline.docs else "",
                "ranks": ranks,
                "delta_vs_bm25": {
                    s: (b - r) if (b is not None and r is not None) else None for s, r in ranks.items()
                },
                "is_new": {s: (r is not None and b is None) for s, r in ranks.items()},
            }
        )
    return {
        "query": req.query,
        "strategies": list(req.strategies),
        "top_k": req.top_k,
        "baseline": BASELINE,
        "results": results,
        "rank_shift": rows,
    }


@router.post("/compare", response_model=CompareResponse, responses=ERRORS)
async def compare(req: CompareRequest, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Run several strategies on the same query; include rank-shift data relative to BM25.

    A strategy whose stage is unavailable yields an ``error`` entry instead of failing the call.
    """
    pipeline = require_pipeline(state)
    params = pipeline.resolve(req.params.model_dump())
    return await run_blocking(state, _compare, pipeline, req, params)
