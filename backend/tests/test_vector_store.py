"""QdrantStore against a fake client: request shapes, batching, readiness and search parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import models

from app.core.vector_store import QdrantStore


@dataclass
class _Point:
    id: int
    score: float


@dataclass
class _Resp:
    points: list[_Point]


@dataclass
class _Info:
    status: str
    points_count: int
    indexed_vectors_count: int


@dataclass
class _Count:
    count: int


class FakeQdrant:
    def __init__(self, infos: list[_Info] | None = None) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.upserts: list[models.Batch] = []
        self.queries: list[dict[str, Any]] = []
        self.infos = infos or []

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, **kw: Any) -> None:
        self.collections[kw["collection_name"]] = kw

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    def count(self, name: str, exact: bool = True) -> _Count:
        return _Count(sum(len(b.ids) for b in self.upserts))

    def upsert(self, collection_name: str, points: models.Batch, wait: bool) -> None:
        assert wait
        self.upserts.append(points)

    def get_collection(self, name: str) -> _Info:
        return self.infos.pop(0) if len(self.infos) > 1 else self.infos[0]

    def query_points(self, **kw: Any) -> _Resp:
        self.queries.append(kw)
        return _Resp([_Point(3, 0.9), _Point(1, 0.5)])


def test_create_upsert_and_search() -> None:
    fake = FakeQdrant(infos=[_Info("green", 10, 10)])
    store = QdrantStore("http://x", "ds_m16_efc100", client=fake)
    assert not store.exists() and store.count() == 0
    store.ensure_collection(dim=4, m=16, ef_construct=100)
    cfg = fake.collections["ds_m16_efc100"]
    assert cfg["hnsw_config"].m == 16 and cfg["hnsw_config"].ef_construct == 100
    assert cfg["vectors_config"].distance == models.Distance.COSINE
    vecs = np.eye(10, 4, dtype=np.float32)
    seen: list[int] = []
    store.upsert(
        vecs,
        [{"doc_id": f"d{i}", "chunk_id": f"d{i}#0"} for i in range(10)],
        batch_size=4,
        progress=lambda done, total, el: seen.append(done),
    )
    assert [len(b.ids) for b in fake.upserts] == [4, 4, 2]
    assert fake.upserts[2].ids == [8, 9] and seen == [4, 8, 10]
    assert store.count() == 10 and store.health()[0]
    assert store.wait_indexed(timeout_s=2, poll_s=0)
    hits = store.search(vecs[0], 2, ef=77)
    assert hits == [(3, 0.9), (1, 0.5)]
    params = fake.queries[-1]["search_params"]
    assert params.hnsw_ef == 77 and params.exact is False
    store.exact_search(vecs[0], 2)
    assert fake.queries[-1]["search_params"].exact is True


def test_wait_indexed_settles_when_tail_unindexed() -> None:
    infos = [_Info("yellow", 10, 0)] + [_Info("green", 10, 8)] * 7
    store = QdrantStore("http://x", "c", client=FakeQdrant(infos=infos))
    assert store.wait_indexed(timeout_s=2, poll_s=0, settle_polls=3)


def test_health_reports_missing_collection() -> None:
    ok, detail = QdrantStore("http://x", "nope", client=FakeQdrant()).health()
    assert not ok and "missing" in detail
