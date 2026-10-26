"""Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2 by default).

A cross-encoder reads ``[CLS] query [SEP] passage [SEP]`` jointly, so its cost is one
transformer forward pass per (query, candidate) pair: O(top_k * L^2) per query, where L is
the truncated pair length (``RERANK_MAX_LENGTH``). That cost is why it is only applied to the
top-k fused candidates.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any, Protocol

log = logging.getLogger(__name__)


class PairScorer(Protocol):
    """The subset of ``sentence_transformers.CrossEncoder`` used here."""

    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: Any) -> Any: ...


def load_cross_encoder(model_name: str, max_length: int) -> PairScorer:
    """Load a CrossEncoder on CPU (weights resolved from ``HF_HOME``)."""
    from sentence_transformers import CrossEncoder

    model: PairScorer = CrossEncoder(model_name, max_length=max_length, device="cpu")
    return model


class Reranker:
    """Batch scorer + top-k reordering."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        max_length: int = 256,
        model: PairScorer | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model
        self._lock = threading.Lock()

    @property
    def model(self) -> PairScorer:
        """The cross-encoder, loaded on first use."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info("loading reranker %s", self.model_name)
                    self._model = load_cross_encoder(self.model_name, self.max_length)
        return self._model

    @property
    def loaded(self) -> bool:
        """True once the model is in memory."""
        return self._model is not None

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance logits for each (query, text) pair, in input order."""
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        out = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in list(out)]

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        """Score the first ``top_k`` (doc_id, text) candidates; return them sorted by score.

        Ties are broken by doc id for determinism.
        """
        head = list(candidates[: max(0, top_k)])
        scores = self.score(query, [t for _, t in head])
        ranked = [(doc, s) for (doc, _), s in zip(head, scores, strict=True)]
        ranked.sort(key=lambda kv: (-kv[1], kv[0]))
        return ranked
