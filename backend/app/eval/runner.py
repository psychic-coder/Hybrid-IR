"""Evaluation runner: every strategy over every test query, with latency and significance.

For each configuration the runner performs:

1. **Cold pass** (result cache *and* embedding cache bypassed): rankings -> metrics, and
   per-request / per-stage latency. These are the headline numbers.
2. **Warm pass** (dense strategies only): the embedding cache is primed for every query,
   then the pass is repeated with the embedding cache on (result cache still off). The
   difference to the cold pass is the value of caching query embeddings.
3. **Concurrent pass**: ``EVAL_QPS_QUERIES`` queries through a thread pool of
   ``EVAL_QPS_THREADS`` workers, caches off -> QPS = completed / wall time.

Then a paired bootstrap compares each configuration with BM25 on per-query nDCG@10.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.pipeline import STRATEGIES, EffectiveParams, SearchPipeline, needs_dense
from app.eval import report
from app.eval.datasets import Dataset, load_dataset, sha256_file
from app.eval.latency import qps, stage_means, summarize
from app.eval.metrics import aggregate, per_query_metrics
from app.eval.stats import paired_bootstrap

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


@dataclass(frozen=True)
class RunConfig:
    """A named (strategy, params) configuration to evaluate."""

    name: str
    strategy: str
    params: EffectiveParams


def default_configs(defaults: EffectiveParams, strategies: Sequence[str] = STRATEGIES) -> list[RunConfig]:
    """One configuration per strategy at the server defaults."""
    return [RunConfig(s, s, defaults) for s in strategies]


def select_queries(dataset: Dataset, max_queries: int) -> list[tuple[str, str]]:
    """Deterministic query subset: the first ``max_queries`` by sorted id (0 = all)."""
    items = sorted(dataset.queries.items())
    return items[:max_queries] if max_queries and max_queries > 0 else items


class _Eta:
    def __init__(self, label: str, total: int, progress: Progress | None) -> None:
        self.label, self.total, self.progress = label, total, progress
        self.t0 = time.perf_counter()
        self.next = 0.0

    def tick(self, done: int) -> None:
        elapsed = time.perf_counter() - self.t0
        if elapsed < self.next and done < self.total:
            return
        self.next = elapsed + 5.0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        msg = f"{self.label}: {done}/{self.total} queries, {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
        log.info(msg)
        if self.progress:
            self.progress(msg)


def evaluate_config(
    pipeline: SearchPipeline,
    dataset: Dataset,
    queries: list[tuple[str, str]],
    cfg: RunConfig,
    *,
    warm_pass: bool = True,
    qps_queries: int = 100,
    qps_threads: int = 4,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one configuration. Returns (summary entry, per-query rows)."""
    per_query: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, float]] = []
    lat_cold: list[float] = []
    timings: list[dict[str, float]] = []
    eta = _Eta(f"[{cfg.name}] cold", len(queries), progress)
    for i, (qid, text) in enumerate(queries, 1):
        out = pipeline.search(
            text, cfg.strategy, cfg.params, use_result_cache=False, use_embedding_cache=False
        )
        ranking = [r.doc_id for r in out.ranking]
        m = per_query_metrics(ranking, dataset.qrels.get(qid, {}))
        metrics_rows.append(m)
        lat_cold.append(out.timings.total_ms)
        timings.append(out.timings.to_dict())
        per_query.append(
            {
                "config": cfg.name,
                "query_id": qid,
                "metrics": m,
                "latency_ms": out.timings.total_ms,
                "ranking": ranking,
            }
        )
        eta.tick(i)

    lat_warm: list[float] | None = None
    if warm_pass and needs_dense(cfg.strategy) and pipeline.embedder is not None:
        for _, text in queries:
            pipeline.embed_query(text, use_cache=True)
        lat_warm = []
        eta = _Eta(f"[{cfg.name}] warm", len(queries), progress)
        for i, (_, text) in enumerate(queries, 1):
            out = pipeline.search(
                text, cfg.strategy, cfg.params, use_result_cache=False, use_embedding_cache=True
            )
            lat_warm.append(out.timings.total_ms)
            eta.tick(i)

    qps_value: float | None = None
    sample = [t for _, t in queries[:qps_queries]]
    if sample:

        def one(q: str) -> None:
            pipeline.search(q, cfg.strategy, cfg.params, use_result_cache=False, use_embedding_cache=False)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=qps_threads) as pool:
            list(pool.map(one, sample))
        qps_value = qps(len(sample), time.perf_counter() - t0)

    entry = {
        "name": cfg.name,
        "strategy": cfg.strategy,
        "params": cfg.params.to_dict(),
        "n_queries": len(queries),
        "metrics": aggregate(metrics_rows),
        "latency_cold": summarize(lat_cold),
        "latency_warm": summarize(lat_warm) if lat_warm else None,
        "stage_means_ms": stage_means(timings),
        "qps": qps_value,
        "qps_threads": qps_threads if sample else None,
        "significance": None,
    }
    return entry, per_query


def environment_info(
    pipeline: SearchPipeline, settings: Settings, dataset_hash: str | None
) -> dict[str, str]:
    """Hardware/software facts that must accompany any reported number."""
    info: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "dataset": settings.dataset,
        "dataset_zip_sha256": dataset_hash or "unknown",
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "dense_backend": pipeline.dense.name if pipeline.dense else "none",
        "hnsw": f"M={settings.hnsw_m}, ef_construct={settings.hnsw_ef_construct}, ef={settings.hnsw_ef}",
        "chunking": f"size={settings.chunk_size}, overlap={settings.chunk_overlap}",
        "random_seed": str(settings.random_seed),
    }
    try:
        import psutil

        info["ram_gb"] = f"{psutil.virtual_memory().total / 1e9:.1f}"
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = str(torch.get_num_threads())
    except Exception:  # noqa: BLE001
        pass
    return info


def run_baseline(
    pipeline: SearchPipeline,
    dataset: Dataset,
    settings: Settings,
    *,
    configs: list[RunConfig] | None = None,
    max_queries: int | None = None,
    progress: Progress | None = None,
    dataset_hash: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all configurations; returns (summary, per-query rows)."""
    configs = configs or default_configs(pipeline.defaults)
    n = settings.eval_max_queries if max_queries is None else max_queries
    queries = select_queries(dataset, n)
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    per_cfg_ndcg: dict[str, list[float]] = {}
    for cfg in configs:
        if needs_dense(cfg.strategy) and (pipeline.embedder is None or pipeline.dense is None):
            log.warning("skipping %s: dense stage unavailable", cfg.name)
            continue
        if cfg.strategy.endswith("_rerank") and pipeline.reranker is None:
            log.warning("skipping %s: reranker unavailable", cfg.name)
            continue
        entry, pq = evaluate_config(
            pipeline,
            dataset,
            queries,
            cfg,
            warm_pass=settings.eval_warm_pass,
            qps_queries=settings.eval_qps_queries,
            qps_threads=settings.eval_qps_threads,
            progress=progress,
        )
        entries.append(entry)
        rows.extend(pq)
        per_cfg_ndcg[cfg.name] = [r["metrics"]["ndcg@10"] for r in pq]
    base = per_cfg_ndcg.get("bm25")
    if base is not None:
        for e in entries:
            if e["name"] == "bm25":
                continue
            e["significance"] = paired_bootstrap(
                per_cfg_ndcg[e["name"]],
                base,
                system_name=e["name"],
                baseline_name="bm25",
                metric="ndcg@10",
                n_resamples=settings.bootstrap_resamples,
                seed=settings.random_seed,
            ).to_dict()
    summary = {
        "dataset": settings.dataset,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_queries": len(queries),
        "environment": environment_info(pipeline, settings, dataset_hash),
        "strategies": entries,
    }
    return summary, rows


def dataset_hash_for(settings: Settings) -> str | None:
    """sha256 of the downloaded BEIR zip if it is still on disk."""
    z = Path(settings.dataset_dir) / f"{settings.dataset}.zip"
    return sha256_file(z) if z.exists() else None


def run_evaluation(
    pipeline: SearchPipeline,
    settings: Settings,
    *,
    include_ablations: bool = False,
    max_queries: int | None = None,
    progress: Progress | None = None,
    dataset: Dataset | None = None,
) -> dict[str, Any]:
    """Baseline eval (+ optional ablations) and write every result file. Returns the summary."""
    dataset = dataset or load_dataset(settings.dataset, settings.raw_dir)
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = run_baseline(
        pipeline,
        dataset,
        settings,
        max_queries=max_queries,
        progress=progress,
        dataset_hash=dataset_hash_for(settings),
    )
    report.write_summary(results_dir, summary)
    report.write_per_query(results_dir, rows)
    stats = dict(pipeline.index.stats())
    stats["dataset"] = settings.dataset
    old = report.read_json(results_dir / "index_stats.json") or {}
    for k in ("embedding_seconds", "n_vectors"):
        if k in old and k not in stats:
            stats[k] = old[k]
    report.write_index_stats(results_dir, stats)
    if include_ablations:
        from app.eval.ablations import run_ablations

        ab = run_ablations(pipeline, dataset, settings, max_queries=max_queries, progress=progress)
        report.write_ablations(results_dir, ab)
    report.write_pareto(results_dir)
    report.write_report_md(results_dir)
    pipeline.cache.reset_stats()
    return summary
