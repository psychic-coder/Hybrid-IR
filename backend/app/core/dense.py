"""Dense bi-encoder wrapper (BAAI/bge-small-en-v1.5 by default) with a Redis embedding cache.

* Queries get the BGE instruction prefix; documents do not (as recommended by the model card).
* All embeddings are L2-normalized float32, so cosine similarity == dot product.
* Query embeddings are cached under ``sha256(model + text)`` as raw float32 bytes.
* ``sentence_transformers`` (and therefore torch) is imported lazily so the rest of the
  system, and the test suite, can run without it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from app.config import QUERY_PREFIX
from app.core.cache import Cache

log = logging.getLogger(__name__)


class EncoderModel(Protocol):
    """The subset of ``SentenceTransformer`` used here."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...
    def get_sentence_embedding_dimension(self) -> int | None: ...


def load_sentence_transformer(model_name: str, torch_threads: int = 0) -> EncoderModel:
    """Load a SentenceTransformer on CPU (weights resolved from ``HF_HOME``)."""
    import torch
    from sentence_transformers import SentenceTransformer

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    model: EncoderModel = SentenceTransformer(model_name, device="cpu")
    return model


class Embedder:
    """Encodes queries (cached) and documents (batched) into normalized vectors."""

    def __init__(
        self,
        model_name: str,
        *,
        cache: Cache | None = None,
        batch_size: int = 32,
        model: EncoderModel | None = None,
        query_prefix: str = QUERY_PREFIX,
        torch_threads: int = 0,
    ) -> None:
        self.model_name = model_name
        self.cache = cache
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self._torch_threads = torch_threads
        self._model = model
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    @property
    def model(self) -> EncoderModel:
        """The underlying encoder, loaded on first use."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info("loading embedding model %s", self.model_name)
                    self._model = load_sentence_transformer(self.model_name, self._torch_threads)
        return self._model

    @property
    def loaded(self) -> bool:
        """True once the model is in memory."""
        return self._model is not None

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        d = self.model.get_sentence_embedding_dimension()
        if d is None:
            raise RuntimeError("model does not report an embedding dimension")
        return int(d)

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        out = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    # ------------------------------------------------------------------ queries
    def encode_query(self, text: str, use_cache: bool = True) -> tuple[np.ndarray, bool | None]:
        """Embed one query. Returns (vector, cache_hit) where cache_hit is None if not consulted."""
        prefixed = self.query_prefix + text
        if use_cache and self.cache is not None and self.cache.enabled:
            key = self.cache.key("embedding", self.model_name, prefixed)
            raw = self.cache.get_bytes("embedding", key)
            if raw is not None:
                return np.frombuffer(raw, dtype=np.float32).copy(), True
            vec = self._encode([prefixed], 1)[0]
            self.cache.set_bytes(key, vec.astype(np.float32).tobytes())
            return vec, False
        return self._encode([prefixed], 1)[0], None

    # ------------------------------------------------------------------ documents
    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> np.ndarray:
        """Embed documents in batches; ``progress(done, total, elapsed_s)`` is called per batch."""
        bs = batch_size or self.batch_size
        total = len(texts)
        if total == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, total, bs):
            parts.append(self._encode(texts[start : start + bs], bs))
            if progress is not None:
                progress(min(start + bs, total), total, time.perf_counter() - t0)
        return np.vstack(parts)
