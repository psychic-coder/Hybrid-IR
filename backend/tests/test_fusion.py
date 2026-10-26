import pytest

from app.core.fusion import list_stats, missing_floor, normalize_scores, rrf, weighted_fusion

SPARSE = [("a", 12.0), ("b", 9.0), ("c", 3.0)]
DENSE = [("c", 0.9), ("a", 0.8), ("d", 0.5)]


def test_rrf_values() -> None:
    fused = dict(rrf([SPARSE, DENSE], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_order_and_ties() -> None:
    order = [d for d, _ in rrf([SPARSE, DENSE], k=60)]
    assert order == ["a", "c", "b", "d"]
    # symmetric lists produce exact ties, broken by doc id
    tie = rrf([[("x", 1.0), ("y", 0.5)], [("y", 1.0), ("x", 0.5)]], k=10)
    assert [d for d, _ in tie] == ["x", "y"]


def test_rrf_ignores_scores() -> None:
    scaled = [(d, s * 1000) for d, s in SPARSE]
    assert rrf([SPARSE, DENSE]) == rrf([scaled, DENSE])


def test_minmax_and_zscore() -> None:
    mm = normalize_scores(SPARSE, "minmax")
    assert mm == {"a": 1.0, "b": pytest.approx(6 / 9), "c": 0.0}
    z = normalize_scores(SPARSE, "zscore")
    st = list_stats([12.0, 9.0, 3.0])
    assert z["a"] == pytest.approx((12 - st.mean) / st.std)
    assert sum(z.values()) == pytest.approx(0.0, abs=1e-12)


def test_constant_lists() -> None:
    const = [("a", 2.0), ("b", 2.0)]
    assert normalize_scores(const, "minmax") == {"a": 1.0, "b": 1.0}
    assert normalize_scores(const, "zscore") == {"a": 0.0, "b": 0.0}
    assert normalize_scores([], "minmax") == {}


def test_missing_floor() -> None:
    assert missing_floor({"a": 0.3}, "minmax") == 0.0
    assert missing_floor({"a": -1.2, "b": 0.5}, "zscore") == -1.2


@pytest.mark.parametrize("norm", ["minmax", "zscore"])
def test_alpha_extremes_reduce_to_single_list(norm: str) -> None:
    only_dense = [d for d, _ in weighted_fusion(SPARSE, DENSE, alpha=1.0, norm=norm)]  # type: ignore[arg-type]
    # docs missing from a list get that list's floor, so only the top-2 order is defined
    assert only_dense[:2] == ["c", "a"]
    only_sparse = [d for d, _ in weighted_fusion(SPARSE, DENSE, alpha=0.0, norm=norm)]  # type: ignore[arg-type]
    assert only_sparse[:2] == ["a", "b"]


def test_weighted_values_minmax() -> None:
    fused = dict(weighted_fusion(SPARSE, DENSE, alpha=0.5, norm="minmax"))
    # dense minmax: c=1, a=0.75, d=0 ; sparse: a=1, b=2/3, c=0
    assert fused["a"] == pytest.approx(0.5 * 0.75 + 0.5 * 1.0)
    assert fused["c"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0)
    assert fused["b"] == pytest.approx(0.5 * 0.0 + 0.5 * (6 / 9))
    assert fused["d"] == pytest.approx(0.0)


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        weighted_fusion(SPARSE, DENSE, alpha=1.5)
    with pytest.raises(ValueError):
        rrf([SPARSE], k=-1)
