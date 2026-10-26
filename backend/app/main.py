"""FastAPI application factory.

Run with ``uvicorn app.main:app``. Tests call :func:`create_app` with a prebuilt state.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, explain, health, metrics, search
from app.api.errors import install_error_handlers
from app.config import Settings, get_settings
from app.deps import AppState, build_state
from app.logging_setup import request_id_var, setup_logging

log = logging.getLogger("hybridir.api")


def create_app(state: AppState | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. When ``state`` is None, components are loaded during startup."""
    settings = settings or (state.settings if state else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state is None:
            setup_logging(settings.log_level)
            loaded = await run_in_threadpool(build_state, settings)
            app.state.hybridir = loaded
        else:
            app.state.hybridir = state
        yield
        app.state.hybridir.close()

    app = FastAPI(
        title="HybridIR",
        version="1.0.0",
        description=(
            "Hybrid retrieval engine (hand-built BM25 + HNSW dense + fusion + cross-encoder "
            "reranking) with a built-in evaluation harness."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID", "X-Admin-Token"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            request_id_var.reset(token)
            raise
        elapsed = (time.perf_counter() - t0) * 1000
        request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Response-Time-ms"] = f"{elapsed:.2f}"
        if request.url.path not in ("/health", "/ready"):
            log.info(
                "request",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed, 2),
                },
            )
        return response

    install_error_handlers(app)
    for r in (health.router, search.router, explain.router, metrics.router, admin.router):
        app.include_router(r)
    return app


app = create_app()
