"""Latency summaries.

Percentiles use the nearest-rank-with-linear-interpolation definition (NumPy's default
``linear`` method). QPS from a concurrent run is ``completed_requests / wall_seconds``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    """mean / p50 / p95 / p99 / max (ms) and n for a list of per-request latencies."""
    if not samples_ms:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(samples_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(arr.max()),
    }


def stage_means(timings: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of every timing field across requests."""
    if not timings:
        return {}
    keys = timings[0].keys()
    return {k: float(np.mean([t[k] for t in timings])) for k in keys}


def qps(n_requests: int, wall_seconds: float) -> float:
    """Throughput; 0 when nothing ran."""
    return n_requests / wall_seconds if wall_seconds > 0 else 0.0


def stage_summaries(timings: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Full percentile summary per timing field."""
    if not timings:
        return {}
    return {k: summarize([t[k] for t in timings]) for k in timings[0]}
