"""Rank fusion: Reciprocal Rank Fusion and normalized weighted score fusion.

RRF (Cormack, Clarke & Buettcher, 2009)::

    RRF(d) = sum_{r in rankers} 1 / (k + rank_r(d))      rank is 1-based; absent => no term

Weighted fusion::

    S(d) = alpha * norm(dense)(d) + (1 - alpha) * norm(sparse)(d)

where ``norm`` is min-max or z-score over each candidate list. A document missing from one
list receives that list's normalized *floor* (0 for min-max; the minimum z-score for
z-score), i.e. it is treated as no better than the worst retrieved candidate.
Ties are broken deterministically by document id.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Norm = Literal["minmax", "zscore"]
Ranked = Sequence[tuple[str, float]]


@dataclass(frozen=True)
class ListStats:
    """Summary of one score list used for normalization (exposed by /explain)."""

    n: int
    min: float
    max: float
    mean: float
    std: float


def list_stats(scores: Sequence[float]) -> ListStats:
    """Population statistics of a score list (zeros for an empty list)."""
    if not scores:
        return ListStats(0, 0.0, 0.0, 0.0, 0.0)
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    return ListStats(n, min(scores), max(scores), mean, math.sqrt(var))


def normalize_scores(ranked: Ranked, norm: Norm) -> dict[str, float]:
    """Normalize a ranked list's scores. Degenerate (constant) lists map to 1.0 / 0.0."""
    if not ranked:
        return {}
    scores = [s for _, s in ranked]
    st = list_stats(scores)
    if norm == "minmax":
        span = st.max - st.min
        if span <= 0:
            return {d: 1.0 for d, _ in ranked}
        return {d: (s - st.min) / span for d, s in ranked}
    if norm == "zscore":
        if st.std <= 0:
            return {d: 0.0 for d, _ in ranked}
        return {d: (s - st.mean) / st.std for d, s in ranked}
    raise ValueError(f"unknown norm {norm!r}")


def missing_floor(normalized: dict[str, float], norm: Norm) -> float:
    """Value assigned to a document absent from a list."""
    if not normalized:
        return 0.0
    return 0.0 if norm == "minmax" else min(normalized.values())


def rrf(rankings: Sequence[Ranked], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of any number of ranked lists (scores ignored)."""
    if k < 0:
        raise ValueError("rrf k must be >= 0")
    fused: dict[str, float] = {}
    for ranked in rankings:
        for rank, (doc, _) in enumerate(ranked, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_fusion(
    sparse: Ranked, dense: Ranked, alpha: float = 0.5, norm: Norm = "minmax"
) -> list[tuple[str, float]]:
    """Convex combination of normalized dense (weight alpha) and sparse (1-alpha) scores."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ns = normalize_scores(sparse, norm)
    nd = normalize_scores(dense, norm)
    fs = missing_floor(ns, norm)
    fd = missing_floor(nd, norm)
    docs = set(ns) | set(nd)
    fused = {d: alpha * nd.get(d, fd) + (1.0 - alpha) * ns.get(d, fs) for d in docs}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
