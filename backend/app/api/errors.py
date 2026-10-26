"""Uniform error handling: every non-2xx response body is an :class:`app.schemas.ErrorInfo`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.pipeline import SearchPipeline, StageUnavailable
from app.deps import AppState
from app.logging_setup import request_id_var

log = logging.getLogger(__name__)
T = TypeVar("T")


class APIError(Exception):
    """An error with an HTTP status and a machine-readable code."""

    def __init__(self, status: int, error: str, detail: str, stage: str | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.error = error
        self.detail = detail
        self.stage = stage


def _body(error: str, detail: str, stage: str | None = None) -> dict[str, Any]:
    return {"error": error, "detail": detail, "stage": stage, "request_id": request_id_var.get()}


def require_pipeline(state: AppState) -> SearchPipeline:
    """The loaded pipeline or a 503 explaining what is missing."""
    if state.pipeline is None:
        detail = state.load_errors.get("index", "index not loaded; has the seed service completed?")
        raise APIError(503, "index_not_loaded", detail, stage="bm25")
    return state.pipeline


async def run_blocking(state: AppState, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run CPU-bound work in the threadpool with the configured request timeout."""
    try:
        return await asyncio.wait_for(
            run_in_threadpool(fn, *args, **kwargs), timeout=state.settings.search_timeout_seconds
        )
    except TimeoutError as exc:
        raise APIError(
            504, "timeout", f"request exceeded {state.settings.search_timeout_seconds:.0f}s"
        ) from exc


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers mapping exceptions to ErrorInfo JSON."""

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=_body(exc.error, exc.detail, exc.stage))

    @app.exception_handler(StageUnavailable)
    async def _stage(_: Request, exc: StageUnavailable) -> JSONResponse:
        log.warning("stage unavailable", extra={"stage": exc.stage, "detail": exc.detail})
        return JSONResponse(status_code=503, content=_body("stage_unavailable", exc.detail, exc.stage))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        msgs = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg')}" for e in exc.errors()
        )
        return JSONResponse(status_code=422, content=_body("validation_error", msgs))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_body("http_error", str(exc.detail)))

    @app.exception_handler(ValueError)
    async def _value(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_body("bad_request", str(exc)))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content=_body("internal_error", type(exc).__name__))
