"""The from-scratch HNSW must reach recall@10 >= 0.9 against brute force on 5k vectors."""

from pathlib import Path

import numpy as np
import pytest

from app.core.custom_hnsw import CustomHNSW
from app.core.vector_store import CustomHNSWStore, ExactStore


@pytest.fixture(scope="module")
def built() -> tuple[CustomHNSW, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    # clustered data is harder (and more realistic) than isotropic noise
    centers = rng.normal(size=(50, 32))
    data = centers[rng.integers(0, 50, 5000)] + 0.35 * rng.normal(size=(5000, 32))
    queries = centers[rng.integers(0, 50, 100)] + 0.35 * rng.normal(size=(100, 32))
    index = CustomHNSW(dim=32, m=12, ef_construction=64, ef_search=64, seed=42)
    index.add_items(data.astype(np.float32))
    return index, data.astype(np.float32), queries.astype(np.float32)


def recall_at_10(index: CustomHNSW, queries: np.ndarray, ef: int) -> float:
    hits = 0
    for q in queries:
        approx = {i for i, _ in index.search(q, 10, ef=ef)}
        exact = {i for i, _ in index.brute_force(q, 10)}
        hits += len(approx & exact)
    return hits / (10 * len(queries))


def test_recall_at_least_090(built: tuple[CustomHNSW, np.ndarray, np.ndarray]) -> None:
    index, _, queries = built
    assert index.count == 5000
    assert recall_at_10(index, queries, ef=64) >= 0.9


def test_recall_increases_with_ef(built: tuple[CustomHNSW, np.ndarray, np.ndarray]) -> None:
    index, _, queries = built
    assert recall_at_10(index, queries[:40], ef=128) >= recall_at_10(index, queries[:40], ef=10) - 1e-9


def test_level_distribution_and_degree_bounds(built: tuple[CustomHNSW, np.ndarray, np.ndarray]) -> None:
    index, _, _ = built
    levels = np.asarray(index.levels)
    # with mL = 1/ln(M), P(level >= 1) = 1/M
    assert abs((levels >= 1).mean() - 1 / index.m) < 0.03
    for node in range(index.count):
        assert len(index.neighbors[node][0]) <= index.m0
        for lvl in range(1, index.levels[node] + 1):
            assert len(index.neighbors[node][lvl]) <= index.m


def test_save_load_roundtrip(tmp_path: Path, built: tuple[CustomHNSW, np.ndarray, np.ndarray]) -> None:
    index, _, queries = built
    path = tmp_path / "g.npz"
    index.save(path)
    loaded = CustomHNSW.load(path)
    for q in queries[:10]:
        assert loaded.search(q, 10) == index.search(q, 10)


def test_store_adapters_agree_with_exact(built: tuple[CustomHNSW, np.ndarray, np.ndarray]) -> None:
    index, _, queries = built
    exact = ExactStore(index.vectors)
    store = CustomHNSWStore(index)
    assert store.health()[0]
    q = queries[0] / np.linalg.norm(queries[0])
    assert exact.search(q, 5) == index.brute_force(q, 5)
    assert len(store.search(q, 5, ef=200)) == 5


def test_empty_index_search() -> None:
    assert CustomHNSW(dim=4).search(np.ones(4), 3) == []
