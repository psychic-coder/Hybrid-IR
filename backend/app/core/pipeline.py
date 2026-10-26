"""Retrieval strategies and per-stage instrumentation.

Strategies
----------
``bm25``                    sparse only (WAND over the hand-built index)
``dense``                   bi-encoder + ANN only
``hybrid_rrf``              RRF(bm25, dense)
``hybrid_weighted``         alpha * norm(dense) + (1 - alpha) * norm(bm25)
``hybrid_rrf_rerank``       hybrid_rrf, then cross-encoder over the top ``rerank_top_k``
``hybrid_weighted_rerank``  hybrid_weighted, then cross-encoder over the top ``rerank_top_k``

Every call returns a full ranking (``eval_depth`` docs, default 100, used by the evaluation
harness) with per-stage scores and ranks, plus a timing breakdown:
tokenize_ms, bm25_ms, embed_ms, ann_ms, fusion_ms, rerank_ms, total_ms.

Failure isolation: a stage that cannot run (Qdrant down, model missing) raises
:class:`StageUnavailable` naming the stage. Strategies that do not need that stage are
unaffected, which is how the API degrades gracefully.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from app.core.bm25 import BM25Scorer
from app.core.cache import Cache
from app.core.dense import Embedder
from app.core.fusion import Norm, rrf, weighted_fusion
from app.core.inverted_index import InvertedIndex
from app.core.reranker import Reranker
from app.core.tokenizer import ParsedQuery
from app.core.vector_store import DenseSearcher
from app.eval.datasets import Document

log = logging.getLogger(__name__)

Strategy = Literal[
    "bm25", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_rrf_rerank", "hybrid_weighted_rerank"
]
STRATEGIES: tuple[str, ...] = get_args(Strategy)
STAGES = ("bm25", "dense", "fusion", "rerank")


def needs_sparse(strategy: str) -> bool:
    """True if the strategy uses BM25."""
    return strategy != "dense"


def needs_dense(strategy: str) -> bool:
    """True if the strategy uses the dense retriever."""
    return strategy != "bm25"


def needs_rerank(strategy: str) -> bool:
    """True if the strategy applies the cross-encoder."""
    return strategy.endswith("_rerank")


def fusion_kind(strategy: str) -> str | None:
    """'rrf', 'weighted' or None."""
    if strategy.startswith("hybrid_rrf"):
        return "rrf"
    if strategy.startswith("hybrid_weighted"):
        return "weighted"
    return None


class StageUnavailable(RuntimeError):
    """A pipeline stage could not run (dependency down or not loaded)."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage} stage unavailable: {detail}")
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class EffectiveParams:
    """Fully resolved retrieval parameters (request overrides applied over defaults)."""

    k1: float
    b: float
    rrf_k: int
    alpha: float
    norm: Norm
    rerank_top_k: int
    ef: int
    candidate_depth: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict."""
        return asdict(self)


@dataclass
class Timings:
    """Per-stage wall-clock milliseconds."""

    tokenize_ms: float = 0.0
    bm25_ms: float = 0.0
    embed_ms: float = 0.0
    ann_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """JSON-friendly dict."""
        return asdict(self)


@dataclass
class RankedDoc:
    """One document in a strategy's final ranking, with every stage's view of it."""

    doc_id: str
    score: float
    bm25: float | None = None
    dense: float | None = None
    fusion: float | None = None
    rerank: float | None = None
    ranks: dict[str, int | None] = field(default_factory=dict)
    chunk_id: int | None = None


@dataclass
class SearchOutcome:
    """Pipeline output for one query/strategy."""

    query: str
    strategy: str
    params: EffectiveParams
    ranking: list[RankedDoc]
    timings: Timings
    result_cache_hit: bool = False
    embedding_cache_hit: bool | None = None
    parsed_terms: list[str] = field(default_factory=list)

    def to_cache(self) -> dict[str, Any]:
        """Serializable form stored in the result cache."""
        return {
            "ranking": [asdict(r) for r in self.ranking],
            "parsed_terms": self.parsed_terms,
        }


@dataclass
class Candidates:
    """Doc-level candidate list from one retriever."""

    ranked: list[tuple[str, float]]
    best_chunk: dict[str, int]


class SearchPipeline:
    """Owns all retrieval components and executes strategies."""

    def __init__(
        self,
        *,
        index: InvertedIndex,
        docs: Mapping[str, Document],
        cache: Cache,
        defaults: EffectiveParams,
        embedder: Embedder | None = None,
        dense: DenseSearcher | None = None,
        reranker: Reranker | None = None,
        embeddings: np.ndarray | None = None,
        phrase_boost: float = 1.0,
        eval_depth: int = 100,
        dataset: str = "unknown",
    ) -> None:
        self.index = index
        self.docs = docs
        self.cache = cache
        self.defaults = defaults
        self.embedder = embedder
        self.dense = dense
        self.reranker = reranker
        self.embeddings = embeddings
        self.phrase_boost = phrase_boost
        self.eval_depth = eval_depth
        self.dataset = dataset
        ratio = index.n_chunks / max(1, index.n_docs)
        self._chunk_factor = max(1, min(4, math.ceil(ratio)))

    # ------------------------------------------------------------------ params
    def resolve(self, overrides: Mapping[str, Any] | None = None) -> EffectiveParams:
        """Apply non-null overrides to the configured defaults."""
        base = self.defaults.to_dict()
        for key, value in (overrides or {}).items():
            if value is not None and key in base:
                base[key] = value
        return EffectiveParams(**base)

    # ------------------------------------------------------------------ text helpers
    def doc_text(self, doc_id: str) -> str:
        """Title + body for reranking and display."""
        doc = self.docs.get(doc_id)
        if doc is None:
            return ""
        return f"{doc.title}\n{doc.text}" if doc.title else doc.text

    def snippet(self, doc_id: str, terms: list[str], width: int = 240) -> str:
        """A window of ``width`` chars around the first query-term occurrence."""
        doc = self.docs.get(doc_id)
        if doc is None:
            return ""
        text = doc.text
        spans = self.index.tokenizer.highlight_spans(text, set(terms)) if terms else []
        start = max(0, spans[0][0] - width // 4) if spans else 0
        end = min(len(text), start + width)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        return prefix + text[start:end].strip() + suffix

    # ------------------------------------------------------------------ stages
    def bm25_candidates(self, parsed: ParsedQuery, params: EffectiveParams, depth: int) -> Candidates:
        """Top-``depth`` documents by BM25 (chunk scores max-aggregated)."""
        scorer = BM25Scorer(self.index, params.k1, params.b, self.phrase_boost)
        result = scorer.search(parsed, depth)
        return Candidates(
            ranked=[(h.doc_id, h.score) for h in result.hits],
            best_chunk={h.doc_id: h.chunk_id for h in result.hits},
        )

    def embed_query(self, query: str, use_cache: bool) -> tuple[np.ndarray, bool | None]:
        """Query embedding (cached). Raises StageUnavailable('dense') if no embedder."""
        if self.embedder is None:
            raise StageUnavailable("dense", "embedding model not loaded")
        try:
            return self.embedder.encode_query(query, use_cache=use_cache)
        except StageUnavailable:
            raise
        except Exception as exc:
            raise StageUnavailable("dense", f"embedding failed: {exc}") from exc

    def dense_candidates(self, vector: np.ndarray, params: EffectiveParams, depth: int) -> Candidates:
        """Top-``depth`` documents by ANN (chunk similarities max-aggregated)."""
        if self.dense is None:
            raise StageUnavailable("dense", "vector store not configured")
        k = min(depth * self._chunk_factor, max(1, self.index.n_chunks))
        try:
            hits = self.dense.search(vector, k, params.ef)
        except Exception as exc:
            raise StageUnavailable("dense", f"{self.dense.name} search failed: {exc}") from exc
        best: dict[str, tuple[float, int]] = {}
        for cid, score in hits:
            if cid < 0 or cid >= self.index.n_chunks:
                continue
            did = self.index.chunk_doc[cid]
            if did not in best or score > best[did][0]:
                best[did] = (score, cid)
        ranked = sorted(((d, s) for d, (s, _) in best.items()), key=lambda kv: (-kv[1], kv[0]))
        ranked = ranked[:depth]
        return Candidates(ranked=ranked, best_chunk={d: best[d][1] for d, _ in ranked})

    def rerank_scores(self, query: str, doc_ids: list[str]) -> list[float]:
        """Cross-encoder scores for ``doc_ids`` in order."""
        if self.reranker is None:
            raise StageUnavailable("rerank", "reranker model not loaded")
        try:
            return self.reranker.score(query, [self.doc_text(d) for d in doc_ids])
        except Exception as exc:
            raise StageUnavailable("rerank", f"cross-encoder failed: {exc}") from exc

    # ------------------------------------------------------------------ main entry
    def _cache_key(self, query: str, strategy: str, params: EffectiveParams, depth: int) -> str:
        payload = json.dumps(
            {
                "q": query,
                "s": strategy,
                "p": params.to_dict(),
                "d": depth,
                "ds": self.dataset,
                "be": self.dense.name if self.dense else None,
                "pb": self.phrase_boost,
            },
            sort_keys=True,
        )
        return self.cache.key("result", payload)

    def search(
        self,
        query: str,
        strategy: str,
        params: EffectiveParams | None = None,
        *,
        use_result_cache: bool = True,
        use_embedding_cache: bool = True,
        depth: int | None = None,
    ) -> SearchOutcome:
        """Run ``strategy`` for ``query`` and return a ranking of ``depth`` docs (default eval_depth)."""
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        params = params or self.defaults
        depth = depth or self.eval_depth
        t_start = time.perf_counter()
        timings = Timings()

        cache_key = self._cache_key(query, strategy, params, depth)
        if use_result_cache:
            cached = self.cache.get_json("result", cache_key)
            if cached is not None:
                cached_ranking = [RankedDoc(**r) for r in cached["ranking"]]
                timings.total_ms = (time.perf_counter() - t_start) * 1000
                return SearchOutcome(
                    query, strategy, params, cached_ranking, timings, True, None, cached["parsed_terms"]
                )

        t = time.perf_counter()
        parsed = self.index.tokenizer.parse_query(query)
        timings.tokenize_ms = (time.perf_counter() - t) * 1000
        cand_depth = max(params.candidate_depth, depth if strategy in ("bm25", "dense") else 0)

        sparse: Candidates | None = None
        dense: Candidates | None = None
        emb_hit: bool | None = None
        if needs_sparse(strategy):
            t = time.perf_counter()
            sparse = self.bm25_candidates(parsed, params, cand_depth)
            timings.bm25_ms = (time.perf_counter() - t) * 1000
        if needs_dense(strategy):
            t = time.perf_counter()
            vec, emb_hit = self.embed_query(query, use_embedding_cache)
            timings.embed_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            dense = self.dense_candidates(vec, params, cand_depth)
            timings.ann_ms = (time.perf_counter() - t) * 1000

        sparse_map = dict(sparse.ranked) if sparse else {}
        dense_map = dict(dense.ranked) if dense else {}
        sparse_rank = {d: i for i, (d, _) in enumerate(sparse.ranked, 1)} if sparse else {}
        dense_rank = {d: i for i, (d, _) in enumerate(dense.ranked, 1)} if dense else {}

        kind = fusion_kind(strategy)
        fused: list[tuple[str, float]]
        if kind is None:
            single = sparse if sparse is not None else dense
            fused = single.ranked if single is not None else []
        else:
            assert sparse is not None and dense is not None
            t = time.perf_counter()
            if kind == "rrf":
                fused = rrf([sparse.ranked, dense.ranked], k=params.rrf_k)
            else:
                fused = weighted_fusion(sparse.ranked, dense.ranked, params.alpha, params.norm)
            timings.fusion_ms = (time.perf_counter() - t) * 1000
        fusion_map = dict(fused) if kind else {}
        fusion_rank = {d: i for i, (d, _) in enumerate(fused, 1)} if kind else {}

        rerank_map: dict[str, float] = {}
        final: list[tuple[str, float]] = fused
        if needs_rerank(strategy):
            head = [d for d, _ in fused[: params.rerank_top_k]]
            t = time.perf_counter()
            scores = self.rerank_scores(query, head)
            timings.rerank_ms = (time.perf_counter() - t) * 1000
            rerank_map = dict(zip(head, scores, strict=True))
            reranked = sorted(rerank_map.items(), key=lambda kv: (-kv[1], kv[0]))
            final = reranked + fused[params.rerank_top_k :]
        rerank_rank = (
            {d: i for i, (d, _) in enumerate(sorted(rerank_map.items(), key=lambda kv: (-kv[1], kv[0])), 1)}
            if rerank_map
            else {}
        )

        ranking: list[RankedDoc] = []
        for doc_id, score in final[:depth]:
            chunk = None
            if sparse and doc_id in sparse.best_chunk:
                chunk = sparse.best_chunk[doc_id]
            elif dense and doc_id in dense.best_chunk:
                chunk = dense.best_chunk[doc_id]
            ranking.append(
                RankedDoc(
                    doc_id=doc_id,
                    score=float(score),
                    bm25=sparse_map.get(doc_id),
                    dense=dense_map.get(doc_id),
                    fusion=fusion_map.get(doc_id),
                    rerank=rerank_map.get(doc_id),
                    ranks={
                        "bm25": sparse_rank.get(doc_id),
                        "dense": dense_rank.get(doc_id),
                        "fusion": fusion_rank.get(doc_id),
                        "rerank": rerank_rank.get(doc_id),
                    },
                    chunk_id=chunk,
                )
            )
        timings.total_ms = (time.perf_counter() - t_start) * 1000
        outcome = SearchOutcome(query, strategy, params, ranking, timings, False, emb_hit, list(parsed.terms))
        if use_result_cache:
            self.cache.set_json(cache_key, outcome.to_cache())
        return outcome
