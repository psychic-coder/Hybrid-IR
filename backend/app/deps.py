"""Application state: loads every component once at startup and hands it to the routers.

Loading is failure-tolerant: a missing index, an unreachable Qdrant or a model that cannot be
loaded is recorded and reported by ``/ready``, and only the strategies that need the missing
component fail (with an explicit 503), everything else keeps working.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import Request

from app.config import Settings
from app.core.cache import Cache
from app.core.dense import Embedder
from app.core.inverted_index import InvertedIndex
from app.core.pipeline import EffectiveParams, SearchPipeline
from app.core.reranker import Reranker
from app.core.vector_store import CustomHNSWStore, DenseSearcher, QdrantStore
from app.eval.datasets import load_corpus

log = logging.getLogger(__name__)


def utcnow() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class EvalTracker:
    """Thread-safe status of the background evaluation job."""

    state: str = "idle"
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    include_ablations: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        """Copy of the public fields."""
        with self.lock:
            return {
                "state": self.state,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "message": self.message,
                "include_ablations": self.include_ablations,
            }


@dataclass
class AppState:
    """Everything a request handler needs."""

    settings: Settings
    pipeline: SearchPipeline | None
    cache: Cache
    load_errors: dict[str, str] = field(default_factory=dict)
    eval: EvalTracker = field(default_factory=EvalTracker)

    def close(self) -> None:
        """Release mmaps."""
        if self.pipeline is not None:
            self.pipeline.index.close()


def default_params(settings: Settings) -> EffectiveParams:
    """Server-side defaults from the environment."""
    return EffectiveParams(
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        rrf_k=settings.rrf_k,
        alpha=settings.fusion_alpha,
        norm=settings.fusion_norm,
        rerank_top_k=settings.rerank_top_k,
        ef=settings.hnsw_ef,
        candidate_depth=settings.candidate_depth,
    )


def build_dense_searcher(settings: Settings) -> DenseSearcher:
    """Qdrant (default) or the from-scratch HNSW, per ``DENSE_BACKEND``."""
    if settings.dense_backend == "custom":
        return CustomHNSWStore.load(settings.custom_hnsw_path)
    return QdrantStore(settings.qdrant_url, settings.collection_name(), settings.qdrant_timeout_seconds)


def build_state(settings: Settings, *, load_models: bool = True) -> AppState:
    """Load index, corpus, models and stores. Never raises; errors land in ``load_errors``."""
    errors: dict[str, str] = {}
    cache = Cache(
        settings.redis_url,
        ttl_seconds=settings.cache_ttl_seconds,
        enabled=settings.cache_enabled,
        timeout_seconds=settings.redis_timeout_seconds,
    )
    try:
        index = InvertedIndex(settings.index_dir)
        docs = load_corpus(settings.raw_dir)
    except Exception as exc:  # noqa: BLE001
        log.error("index or corpus not available: %s", exc)
        errors["index"] = f"{type(exc).__name__}: {exc}"
        return AppState(settings=settings, pipeline=None, cache=cache, load_errors=errors)

    embedder: Embedder | None = Embedder(
        settings.embed_model,
        cache=cache,
        batch_size=settings.embed_batch_size,
        torch_threads=settings.torch_threads,
    )
    reranker: Reranker | None = Reranker(
        settings.rerank_model,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
    )
    if load_models:
        try:
            assert embedder is not None
            _ = embedder.model
        except Exception as exc:  # noqa: BLE001
            log.error("embedding model failed to load: %s", exc)
            errors["embedder"] = f"{type(exc).__name__}: {exc}"
            embedder = None
        try:
            assert reranker is not None
            _ = reranker.model
        except Exception as exc:  # noqa: BLE001
            log.error("reranker failed to load: %s", exc)
            errors["reranker"] = f"{type(exc).__name__}: {exc}"
            reranker = None

    dense: DenseSearcher | None
    try:
        dense = build_dense_searcher(settings)
    except Exception as exc:  # noqa: BLE001
        log.error("dense backend unavailable: %s", exc)
        errors["dense"] = f"{type(exc).__name__}: {exc}"
        dense = None

    embeddings: np.ndarray | None = None
    if settings.embeddings_path.exists():
        embeddings = np.load(settings.embeddings_path, mmap_mode="r")

    pipeline = SearchPipeline(
        index=index,
        docs=docs,
        cache=cache,
        defaults=default_params(settings),
        embedder=embedder,
        dense=dense,
        reranker=reranker,
        embeddings=embeddings,
        phrase_boost=settings.phrase_boost,
        eval_depth=settings.eval_depth,
        dataset=settings.dataset,
    )
    log.info(
        "state loaded", extra={"dataset": settings.dataset, "chunks": index.n_chunks, "errors": list(errors)}
    )
    return AppState(settings=settings, pipeline=pipeline, cache=cache, load_errors=errors)


def get_state(request: Request) -> AppState:
    """FastAPI dependency returning the application state."""
    state: AppState = request.app.state.hybridir
    return state
