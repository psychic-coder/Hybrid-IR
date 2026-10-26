"""Metric fixtures computed by hand (see comments for the arithmetic)."""

import math

import pytest

from app.eval.latency import qps, summarize
from app.eval.metrics import (
    aggregate,
    average_precision_at_k,
    dcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    per_query_metrics,
    precision_at_k,
    recall_at_k,
)
from app.eval.stats import paired_bootstrap

QRELS = {"a": 3, "b": 2, "c": 1, "z": 0}  # z judged non-relevant
RUN = ["b", "x", "a", "y", "c"]  # x, y unjudged


def test_dcg_known_value() -> None:
    # grades [2,0,3,0,1]: 2/log2(2) + 3/log2(4) + 1/log2(6)
    expected = 2 / 1 + 3 / 2 + 1 / math.log2(6)
    assert dcg_at_k([2, 0, 3, 0, 1], 5) == pytest.approx(expected)


def test_ndcg_graded() -> None:
    dcg = 2 + 3 / 2 + 1 / math.log2(6)
    idcg = 3 + 2 / math.log2(3) + 1 / 2
    assert ndcg_at_k(RUN, QRELS, 5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_and_empty() -> None:
    assert ndcg_at_k(["a", "b", "c"], QRELS, 10) == pytest.approx(1.0)
    assert ndcg_at_k([], QRELS, 10) == 0.0
    assert ndcg_at_k(["a"], {"q": 0}, 10) == 0.0


def test_ndcg_ideal_uses_all_judgments_not_only_retrieved() -> None:
    # retrieving only "c" (grade 1) at rank 1 must not score 1.0
    assert ndcg_at_k(["c"], QRELS, 10) < 0.5


def test_ndcg_exponential_gain() -> None:
    dcg = (2**2 - 1) + (2**3 - 1) / 2 + (2**1 - 1) / math.log2(6)
    idcg = (2**3 - 1) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / 2
    assert ndcg_at_k(RUN, QRELS, 5, gain="exponential") == pytest.approx(dcg / idcg)


def test_recall_precision_mrr() -> None:
    assert recall_at_k(RUN, QRELS, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RUN, QRELS, 5) == pytest.approx(1.0)
    assert precision_at_k(RUN, QRELS, 5) == pytest.approx(3 / 5)
    assert precision_at_k(RUN, QRELS, 10) == pytest.approx(3 / 10)  # denominator is k
    assert mrr_at_k(RUN, QRELS, 10) == pytest.approx(1.0)
    assert mrr_at_k(["x", "y", "c"], QRELS, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "c"], QRELS, 2) == 0.0


def test_average_precision() -> None:
    # relevant at ranks 1,3,5 -> (1/1 + 2/3 + 3/5) / 3
    assert average_precision_at_k(RUN, QRELS, 100) == pytest.approx((1 + 2 / 3 + 3 / 5) / 3)
    # cut at 3 still divides by |R| = 3
    assert average_precision_at_k(RUN, QRELS, 3) == pytest.approx((1 + 2 / 3) / 3)


def test_unjudged_counts_as_irrelevant() -> None:
    assert precision_at_k(["x", "y"], QRELS, 2) == 0.0


def test_per_query_and_aggregate() -> None:
    m = per_query_metrics(RUN, QRELS)
    assert set(m) == {"ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10", "map@100", "p@10"}
    agg = aggregate([m, {k: 0.0 for k in m}])
    assert agg["mrr@10"] == pytest.approx(0.5)


def test_latency_summary() -> None:
    s = summarize([float(i) for i in range(1, 101)])
    assert s["n"] == 100
    assert s["p50"] == pytest.approx(50.5)
    assert s["p99"] == pytest.approx(99.01)
    assert s["mean"] == pytest.approx(50.5)
    assert qps(100, 4.0) == 25.0
    assert summarize([])["n"] == 0


def test_bootstrap_detects_clear_improvement() -> None:
    base = [0.2, 0.3, 0.25, 0.4, 0.1] * 20
    sysm = [b + 0.1 for b in base]
    r = paired_bootstrap(sysm, base, n_resamples=2000, seed=1)
    assert r.mean_delta == pytest.approx(0.1)
    assert r.ci_low == pytest.approx(0.1) and r.ci_high == pytest.approx(0.1)
    assert r.significant and r.p_value < 0.01


def test_bootstrap_no_difference() -> None:
    base = [0.1, 0.5, 0.3, 0.7] * 10
    r = paired_bootstrap(base, base, n_resamples=500, seed=1)
    assert r.mean_delta == 0.0 and r.p_value == 1.0 and not r.significant


def test_bootstrap_is_deterministic() -> None:
    a = [0.1, 0.4, 0.3, 0.8, 0.2, 0.5]
    b = [0.2, 0.3, 0.3, 0.6, 0.1, 0.5]
    assert paired_bootstrap(a, b, n_resamples=1000) == paired_bootstrap(a, b, n_resamples=1000)
