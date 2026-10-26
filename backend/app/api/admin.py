"""POST /admin/eval and GET /admin/eval/status (guarded by the ADMIN_TOKEN header)."""

from __future__ import annotations

import hmac
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, Header

from app.api.errors import APIError, require_pipeline
from app.deps import AppState, get_state, utcnow
from app.schemas import ErrorInfo, EvalStatus, EvalTriggerRequest

router = APIRouter(prefix="/admin", tags=["admin"])
log = logging.getLogger(__name__)
GUARD_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorInfo},
    403: {"model": ErrorInfo},
    409: {"model": ErrorInfo},
    503: {"model": ErrorInfo},
}


def require_admin(
    state: AppState = Depends(get_state), x_admin_token: str | None = Header(default=None)
) -> AppState:
    """Reject the request unless ADMIN_TOKEN is configured and matches X-Admin-Token."""
    expected = state.settings.admin_token
    if not expected:
        raise APIError(403, "admin_disabled", "set ADMIN_TOKEN to enable admin endpoints")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise APIError(401, "unauthorized", "missing or invalid X-Admin-Token header")
    return state


def _run(state: AppState, include_ablations: bool, max_queries: int | None) -> None:
    from app.eval.runner import run_evaluation

    tracker = state.eval
    pipeline = state.pipeline
    assert pipeline is not None

    def progress(msg: str) -> None:
        with tracker.lock:
            tracker.message = msg

    try:
        run_evaluation(
            pipeline,
            state.settings,
            include_ablations=include_ablations,
            max_queries=max_queries,
            progress=progress,
        )
        with tracker.lock:
            tracker.state = "succeeded"
            tracker.message = "results written to RESULTS_DIR"
    except Exception as exc:
        log.exception("background eval failed")
        with tracker.lock:
            tracker.state = "failed"
            tracker.message = f"{type(exc).__name__}: {exc}"
    finally:
        with tracker.lock:
            tracker.finished_at = utcnow()


@router.post("/eval", response_model=EvalStatus, status_code=202, responses=GUARD_ERRORS)
async def trigger_eval(body: EvalTriggerRequest, state: AppState = Depends(require_admin)) -> dict[str, Any]:
    """Start a background evaluation (409 if one is already running)."""
    require_pipeline(state)
    tracker = state.eval
    with tracker.lock:
        if tracker.state == "running":
            raise APIError(409, "conflict", "an evaluation is already running")
        tracker.state = "running"
        tracker.started_at = utcnow()
        tracker.finished_at = None
        tracker.message = "starting"
        tracker.include_ablations = body.include_ablations
    accepted = tracker.snapshot()  # taken before the worker can finish
    threading.Thread(
        target=_run,
        args=(state, body.include_ablations, body.max_queries),
        daemon=True,
        name="hybridir-eval",
    ).start()
    return accepted


@router.get("/eval/status", response_model=EvalStatus, responses=GUARD_ERRORS)
async def eval_status(state: AppState = Depends(require_admin)) -> dict[str, Any]:
    """Current background evaluation status."""
    return state.eval.snapshot()
