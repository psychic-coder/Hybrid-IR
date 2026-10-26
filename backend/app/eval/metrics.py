"""Ranking metrics implemented from their definitions (no pytrec_eval dependency).

Conventions (matching trec_eval / BEIR):

* A judged grade > 0 is *relevant*; unjudged documents count as grade 0 (non-relevant).
* nDCG uses linear gain by default: DCG@k = sum_{i=1..k} rel_i / log2(i + 1).
  ``gain="exponential"`` gives the (2^rel - 1) variant.
* Ideal DCG sorts *all* judged grades for the query (not only retrieved ones) descending.
* MAP@k divides by the total number of relevant documents |R| (trec_eval ``map_cut``).
* A query with no relevant documents scores 0 on every metric (callers normally filter them).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

Gain = Literal["linear", "exponential"]


def _gain(rel: int, gain: Gain) -> float:
    return float(rel) if gain == "linear" else float(2**rel - 1)


def dcg_at_k(grades: Sequence[int], k: int, gain: Gain = "linear") -> float:
    """Discounted cumulative gain of a list of grades (in rank order)."""
    return sum(_gain(g, gain) / math.log2(i + 2) for i, g in enumerate(grades[:k]) if g > 0)


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int, gain: Gain = "linear") -> float:
    """Normalized DCG@k with graded relevance and correct ideal ordering."""
    grades = [max(0, qrels.get(d, 0)) for d in ranking[:k]]
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k, gain)
    if idcg == 0:
        return 0.0
    return dcg_at_k(grades, k, gain) / idcg


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Fraction of relevant documents retrieved in the top k."""
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Fraction of the top k that is relevant (denominator is always k)."""
    if k <= 0:
        return 0.0
    return sum(1 for d in ranking[:k] if qrels.get(d, 0) > 0) / k


def mrr_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top k."""
    for i, d in enumerate(ranking[:k], start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """AP@k = (1/|R|) * sum over relevant hits at rank i <= k of precision@i."""
    n_rel = sum(1 for g in qrels.values() if g > 0)
    if n_rel == 0:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranking[:k], start=1):
        if qrels.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


METRIC_NAMES = (
    "ndcg@10",
    "ndcg@100",
    "recall@10",
    "recall@100",
    "mrr@10",
    "map@100",
    "p@10",
)


def per_query_metrics(ranking: Sequence[str], qrels: Mapping[str, int]) -> dict[str, float]:
    """All reported metrics for one query."""
    return {
        "ndcg@10": ndcg_at_k(ranking, qrels, 10),
        "ndcg@100": ndcg_at_k(ranking, qrels, 100),
        "recall@10": recall_at_k(ranking, qrels, 10),
        "recall@100": recall_at_k(ranking, qrels, 100),
        "mrr@10": mrr_at_k(ranking, qrels, 10),
        "map@100": average_precision_at_k(ranking, qrels, 100),
        "p@10": precision_at_k(ranking, qrels, 10),
    }


def aggregate(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Macro-average each metric over queries."""
    if not per_query:
        return {m: 0.0 for m in METRIC_NAMES}
    return {m: sum(q[m] for q in per_query) / len(per_query) for m in METRIC_NAMES}
