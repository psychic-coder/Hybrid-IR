"""GET /health (liveness) and GET /ready (readiness)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.deps import AppState, get_state
from app.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    """The process is up and serving HTTP."""
    return {"status": "ok"}


def readiness(state: AppState) -> dict[str, Any]:
    """Evaluate every dependency check."""
    p = state.pipeline
    checks: dict[str, dict[str, Any]] = {}
    if p is None:
        checks["index"] = {"ok": False, "detail": state.load_errors.get("index", "not loaded")}
    else:
        checks["index"] = {"ok": True, "detail": f"{p.index.n_chunks} chunks, {len(p.index.terms)} terms"}
    emb_ok = p is not None and p.embedder is not None and p.embedder.loaded
    checks["embedder"] = {
        "ok": emb_ok,
        "detail": "loaded" if emb_ok else state.load_errors.get("embedder", "not loaded"),
    }
    rr_ok = p is not None and p.reranker is not None and p.reranker.loaded
    checks["reranker"] = {
        "ok": rr_ok,
        "detail": "loaded" if rr_ok else state.load_errors.get("reranker", "not loaded"),
    }
    ok, detail = state.cache.ping()
    checks["redis"] = {"ok": ok, "detail": detail}
    if p is not None and p.dense is not None:
        ok, detail = p.dense.health()
        checks[p.dense.name] = {"ok": ok, "detail": detail}
    else:
        checks[state.settings.dense_backend] = {
            "ok": False,
            "detail": state.load_errors.get("dense", "dense backend not loaded"),
        }
    return {
        "ready": all(c["ok"] for c in checks.values()),
        "dataset": state.settings.dataset,
        "dense_backend": state.settings.dense_backend,
        "checks": checks,
    }


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
async def ready(state: AppState = Depends(get_state)) -> Any:
    """200 when index, models, Redis and the vector store are all available, else 503."""
    body = await run_in_threadpool(readiness, state)
    if not body["ready"]:
        return JSONResponse(status_code=503, content=body)
    return body
