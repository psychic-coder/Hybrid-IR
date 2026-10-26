"""End-to-end API tests over the tiny corpus with fake (offline) models."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core.pipeline import STRATEGIES
from app.deps import AppState
from app.eval import report
from app.eval.datasets import Dataset
from app.eval.runner import run_evaluation
from app.main import create_app
from tests.conftest import TINY_DOCS, make_pipeline


class BrokenDense:
    """A dense backend that is down (simulates Qdrant being unreachable)."""

    name = "qdrant"

    def search(self, vector: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        raise ConnectionError("connection refused")

    def health(self) -> tuple[bool, str]:
        return False, "qdrant unreachable: connection refused"


def tiny_dataset() -> Dataset:
    return Dataset(
        name="tiny",
        corpus={d.doc_id: d for d in TINY_DOCS},
        queries={
            "q1": "lung cancer smoking",
            "q2": "heart disease exercise",
            "q3": "vitamin d bone",
            "q4": "cancer risk genetics",
        },
        qrels={"q1": {"d1": 2, "d6": 1}, "q2": {"d2": 1, "d7": 1}, "q3": {"d3": 2}, "q4": {"d8": 2, "d4": 1}},
    )


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        dataset="scifact",
        admin_token="secret",
        cache_enabled=True,
        bootstrap_resamples=200,
        eval_qps_queries=4,
        eval_qps_threads=2,
        ablation_latency_sample=2,
        dense_backend="custom",
    )


@pytest.fixture()
def state(tmp_path: Path, settings: Settings) -> AppState:
    pipeline = make_pipeline(tmp_path)
    return AppState(settings=settings, pipeline=pipeline, cache=pipeline.cache)


@pytest.fixture()
def client(state: AppState) -> Iterator[TestClient]:
    with TestClient(create_app(state)) as c:
        yield c


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ready"] is True
    assert set(body["checks"]) >= {"index", "embedder", "reranker", "redis"}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_search_every_strategy(client: TestClient, strategy: str) -> None:
    r = client.post("/search", json={"query": "lung cancer", "strategy": strategy, "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["strategy"] == strategy
    assert 1 <= len(body["hits"]) <= 3
    assert [h["rank"] for h in body["hits"]] == list(range(1, len(body["hits"]) + 1))
    t = body["timings"]
    assert t["total_ms"] >= t["bm25_ms"]
    if strategy.endswith("rerank"):
        assert body["hits"][0]["scores"]["rerank"] is not None
    if strategy == "bm25":
        assert body["hits"][0]["doc_id"] in {"d1", "d6"}
        assert t["embed_ms"] == 0.0


def test_search_params_and_result_cache(client: TestClient) -> None:
    payload = {
        "query": "heart disease",
        "strategy": "hybrid_weighted",
        "top_k": 5,
        "params": {"alpha": 0.3, "norm": "zscore", "k1": 0.9, "b": 0.4},
    }
    first = client.post("/search", json=payload).json()
    assert first["params"]["alpha"] == 0.3 and first["params"]["norm"] == "zscore"
    assert first["cache"]["result_cache_hit"] is False
    second = client.post("/search", json=payload).json()
    assert second["cache"]["result_cache_hit"] is True
    assert [h["doc_id"] for h in first["hits"]] == [h["doc_id"] for h in second["hits"]]
    stats = client.get("/metrics/cache").json()
    assert stats["result_hits"] >= 1 and stats["redis_ok"] is True


def test_embedding_cache_hit(client: TestClient) -> None:
    body = {"query": "sleep obesity", "strategy": "dense", "top_k": 2}
    client.post("/search", json=body)
    second = client.post("/search", json={**body, "params": {"ef": 50}}).json()
    assert second["cache"]["embedding_cache_hit"] is True


def test_validation_errors(client: TestClient) -> None:
    r = client.post("/search", json={"query": "", "strategy": "bm25"})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    r = client.post("/search", json={"query": "x", "strategy": "nope"})
    assert r.status_code == 422
    r = client.post("/search", json={"query": "x", "params": {"alpha": 2}})
    assert r.status_code == 422
    r = client.post("/search", json={"query": "x", "unknown_field": 1})
    assert r.status_code == 422


def test_request_id_header(client: TestClient) -> None:
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_compare_rank_shift(client: TestClient) -> None:
    r = client.post("/compare", json={"query": "cancer risk", "top_k": 4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [c["strategy"] for c in body["results"]] == ["bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"]
    assert body["rank_shift"]
    for row in body["rank_shift"]:
        assert row["delta_vs_bm25"]["bm25"] in (0, None)
        b = row["ranks"]["bm25"]
        for s, rnk in row["ranks"].items():
            if b is not None and rnk is not None:
                assert row["delta_vs_bm25"][s] == b - rnk
            assert row["is_new"][s] == (rnk is not None and b is None)


def test_explain(client: TestClient) -> None:
    r = client.get("/explain", params={"query": '"lung cancer" smoking', "doc_id": "d1"})
    assert r.status_code == 200, r.text
    body = r.json()
    terms = {t["term"]: t for t in body["terms"]}
    assert terms["cancer"]["tf"] == 2 and terms["smoke"]["tf"] == 1
    assert body["phrase_boost"] > 0
    assert body["bm25_score"] == pytest.approx(sum(t["score"] for t in body["terms"]) + body["phrase_boost"])
    assert body["cosine"] is not None
    f = body["fusion"]
    assert f["rrf_score"] == pytest.approx(f["rrf_bm25_term"] + f["rrf_dense_term"])
    assert body["rerank_score"] is not None
    assert body["ranks"]["bm25"] == 1
    assert body["text_highlights"]
    assert client.get("/explain", params={"query": "x", "doc_id": "missing"}).status_code == 404


def test_graceful_degradation_when_dense_down(state: AppState) -> None:
    assert state.pipeline is not None
    state.pipeline.dense = BrokenDense()
    with TestClient(create_app(state)) as c:
        assert c.post("/search", json={"query": "lung cancer", "strategy": "bm25"}).status_code == 200
        r = c.post("/search", json={"query": "lung cancer", "strategy": "hybrid_rrf"})
        assert r.status_code == 503
        assert r.json()["stage"] == "dense" and r.json()["error"] == "stage_unavailable"
        cmp_ = c.post("/compare", json={"query": "lung cancer"}).json()
        by = {x["strategy"]: x for x in cmp_["results"]}
        assert by["bm25"]["response"] is not None
        assert by["dense"]["error"]["stage"] == "dense"
        ready = c.get("/ready")
        assert ready.status_code == 503 and ready.json()["checks"]["qdrant"]["ok"] is False


def test_reranker_down(state: AppState) -> None:
    assert state.pipeline is not None
    state.pipeline.reranker = None
    with TestClient(create_app(state)) as c:
        assert c.post("/search", json={"query": "cancer", "strategy": "hybrid_rrf"}).status_code == 200
        r = c.post("/search", json={"query": "cancer", "strategy": "hybrid_rrf_rerank"})
        assert r.status_code == 503 and r.json()["stage"] == "rerank"


def test_index_not_loaded(settings: Settings) -> None:
    from app.core.cache import Cache

    st = AppState(
        settings=settings, pipeline=None, cache=Cache(enabled=False), load_errors={"index": "missing"}
    )
    with TestClient(create_app(st)) as c:
        r = c.post("/search", json={"query": "x", "strategy": "bm25"})
        assert r.status_code == 503 and r.json()["error"] == "index_not_loaded"
        assert c.get("/ready").status_code == 503


def test_metrics_empty_then_populated(client: TestClient, state: AppState, settings: Settings) -> None:
    empty = client.get("/metrics/summary").json()
    assert empty["available"] is False and "make eval" in empty["message"]
    assert client.get("/metrics/ablations").json()["available"] is False
    assert state.pipeline is not None
    run_evaluation(state.pipeline, settings, include_ablations=True, dataset=tiny_dataset())
    summary = client.get("/metrics/summary").json()
    assert summary["available"] is True
    names = {s["name"] for s in summary["strategies"]}
    assert names == set(STRATEGIES)
    for s in summary["strategies"]:
        assert 0.0 <= s["metrics"]["ndcg@10"] <= 1.0
        assert s["latency_cold"]["n"] == 4
        assert (s["significance"] is None) == (s["name"] == "bm25")
    assert summary["index_stats"]["n_docs"] == len(TINY_DOCS)
    assert any(p["on_frontier"] for p in summary["pareto"])
    ab = client.get("/metrics/ablations").json()
    assert ab["available"] is True
    assert len(ab["bm25_grid"]) == 20
    assert len(ab["chunk_size"]) == 4
    assert len(ab["alpha_sweep"]) == 22 and len(ab["rrf_k"]) == 4
    assert len(ab["hnsw"]) == 15 and len(ab["rerank_top_k"]) == 4
    md = (settings.results_dir / "report.md").read_text()
    assert "nDCG@10" in md and "Significance vs BM25" in md
    lines = (settings.results_dir / "per_query.jsonl").read_text().splitlines()
    assert len(lines) == 4 * len(STRATEGIES)


def test_load_test_summary_is_surfaced(client: TestClient, settings: Settings, state: AppState) -> None:
    assert state.pipeline is not None
    run_evaluation(state.pipeline, settings, dataset=tiny_dataset(), max_queries=2)
    report.write_json(
        settings.results_dir / "k6_summary.json",
        {
            "script": "k6_search.js",
            "generated_at": "2026-01-01T00:00:00Z",
            "duration_s": 10.0,
            "total_requests": 5,
            "rps": 0.5,
            "error_rate": 0.0,
            "strategies": [
                {"strategy": "bm25", "count": 5, "avg_ms": 1.0, "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 2.0}
            ],
        },
    )
    lt = client.get("/metrics/summary").json()["load_test"]
    assert lt["strategies"][0]["strategy"] == "bm25"


def test_admin_eval_auth_and_run(client: TestClient) -> None:
    assert client.post("/admin/eval", json={}).status_code == 401
    assert client.post("/admin/eval", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/admin/eval/status", headers={"X-Admin-Token": "secret"}).json()["state"] == "idle"


def test_admin_disabled_without_token(state: AppState) -> None:
    state.settings = state.settings.model_copy(update={"admin_token": ""})
    with TestClient(create_app(state)) as c:
        assert c.post("/admin/eval", json={}, headers={"X-Admin-Token": "x"}).status_code == 403


def test_admin_background_eval(state: AppState, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.eval.runner as runner

    calls: list[bool] = []

    def fake_run(pipeline: object, settings: Settings, **kw: object) -> dict[str, object]:
        calls.append(bool(kw.get("include_ablations")))
        return {}

    monkeypatch.setattr(runner, "run_evaluation", fake_run)
    with TestClient(create_app(state)) as c:
        h = {"X-Admin-Token": "secret"}
        r = c.post("/admin/eval", json={"include_ablations": True}, headers=h)
        assert r.status_code == 202 and r.json()["state"] == "running"
        for _ in range(100):
            st = c.get("/admin/eval/status", headers=h).json()
            if st["state"] != "running":
                break
            time.sleep(0.02)
        assert st["state"] == "succeeded"
        assert calls == [True]


def test_openapi_documents_all_routes(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert {
        "/health",
        "/ready",
        "/search",
        "/compare",
        "/explain",
        "/metrics/summary",
        "/metrics/ablations",
        "/metrics/cache",
        "/admin/eval",
        "/admin/eval/status",
    } <= paths
