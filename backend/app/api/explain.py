"""GET /explain."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.errors import APIError, require_pipeline, run_blocking
from app.core.explain import DocumentNotFound, explain
from app.deps import AppState, get_state
from app.schemas import ErrorInfo, ExplainResponse, Norm

router = APIRouter(tags=["explain"])


@router.get(
    "/explain",
    response_model=ExplainResponse,
    responses={404: {"model": ErrorInfo}, 422: {"model": ErrorInfo}, 503: {"model": ErrorInfo}},
)
async def explain_endpoint(
    query: str = Query(min_length=1, max_length=512),
    doc_id: str = Query(min_length=1, max_length=256),
    k1: float | None = Query(default=None, ge=0.0, le=5.0),
    b: float | None = Query(default=None, ge=0.0, le=1.0),
    rrf_k: int | None = Query(default=None, ge=1, le=1000),
    alpha: float | None = Query(default=None, ge=0.0, le=1.0),
    norm: Norm | None = Query(default=None),
    rerank_top_k: int | None = Query(default=None, ge=1, le=200),
    ef: int | None = Query(default=None, ge=1, le=1024),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """Per-term BM25 contributions, cosine, fusion math, reranker score and rank shifts."""
    pipeline = require_pipeline(state)
    params = pipeline.resolve(
        {
            "k1": k1,
            "b": b,
            "rrf_k": rrf_k,
            "alpha": alpha,
            "norm": norm,
            "rerank_top_k": rerank_top_k,
            "ef": ef,
        }
    )
    try:
        return await run_blocking(state, explain, pipeline, query, doc_id, params)
    except DocumentNotFound as exc:
        raise APIError(404, "not_found", f"unknown doc_id {doc_id!r}") from exc
