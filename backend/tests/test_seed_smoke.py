"""Offline smoke test of `python -m scripts.seed` on a synthetic BEIR-format dataset.

The network-facing steps (service wait, model download, dataset download) and the model
loaders are replaced with fakes; everything else (index build, embeddings, custom HNSW,
evaluation, ablations, result files, idempotency marker) runs for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings
from tests.conftest import TINY_DOCS, HashingModel, OverlapCrossEncoder


def write_beir(raw: Path) -> None:
    (raw / "qrels").mkdir(parents=True)
    with open(raw / "corpus.jsonl", "w") as fh:
        for d in TINY_DOCS:
            fh.write(json.dumps({"_id": d.doc_id, "title": d.title, "text": d.text}) + "\n")
    queries = {"q1": "lung cancer smoking", "q2": "heart disease", "q3": "vitamin d", "q9": "unjudged"}
    with open(raw / "queries.jsonl", "w") as fh:
        for qid, text in queries.items():
            fh.write(json.dumps({"_id": qid, "text": text}) + "\n")
    (raw / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td6\t1\nq2\td2\t1\nq3\td3\t1\n"
    )


def test_seed_end_to_end_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.core.dense as dense
    import app.core.reranker as reranker
    import scripts.seed as seed

    env = {
        "DATA_DIR": str(tmp_path / "data"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "DATASET": "scifact",
        "DENSE_BACKEND": "custom",
        "CACHE_ENABLED": "false",
        "BOOTSTRAP_RESAMPLES": "200",
        "EVAL_QPS_QUERIES": "3",
        "ABLATIONS_ON_SEED": "true",
        "ABLATION_LATENCY_SAMPLE": "2",
        "HNSW_M": "8",
        "LOG_LEVEL": "WARNING",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    write_beir(tmp_path / "data" / "scifact" / "raw")

    monkeypatch.setattr(seed, "wait_for_services", lambda s: None)
    monkeypatch.setattr(seed, "ensure_models", lambda s: None)
    monkeypatch.setattr(dense, "load_sentence_transformer", lambda name, threads=0: HashingModel())
    monkeypatch.setattr(reranker, "load_cross_encoder", lambda name, max_length: OverlapCrossEncoder())

    try:
        assert seed.main() == 0
        results = tmp_path / "results"
        for f in (
            "summary.json",
            "per_query.jsonl",
            "ablations.json",
            "pareto.json",
            "index_stats.json",
            "report.md",
            "queries.json",
            "summary.csv",
        ):
            assert (results / f).exists(), f
        summary = json.loads((results / "summary.json").read_text())
        assert summary["n_queries"] == 3  # q9 has no judgments and is dropped
        assert len(summary["strategies"]) == 6
        stats = json.loads((results / "index_stats.json").read_text())
        assert stats["n_vectors"] == len(TINY_DOCS) and stats["embedding_seconds"] is not None
        assert (tmp_path / "data" / "scifact" / "custom_hnsw.npz").exists()

        mtime = (results / "summary.json").stat().st_mtime_ns
        index_mtime = (tmp_path / "data" / "scifact" / "index" / "postings.bin").stat().st_mtime_ns
        assert seed.main() == 0  # idempotent rerun: nothing rebuilt, eval skipped
        assert (results / "summary.json").stat().st_mtime_ns == mtime
        assert (tmp_path / "data" / "scifact" / "index" / "postings.bin").stat().st_mtime_ns == index_mtime
    finally:
        get_settings.cache_clear()
