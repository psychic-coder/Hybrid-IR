"""Paired bootstrap significance test on per-query metric differences.

Given per-query scores a_i (system) and b_i (baseline) on the same queries, with
d_i = a_i - b_i:

* mean delta  = mean(d)
* 95% CI      = 2.5th / 97.5th percentiles of mean(d*) over B resamples of queries with
  replacement (percentile bootstrap)
* p-value     = two-sided test of H0: E[d] = 0 using the *shifted* bootstrap: resample
  d - mean(d) (which satisfies H0) and count how often |mean(d*)| >= |mean(d)|;
  reported as (count + 1) / (B + 1) so it is never exactly zero.

The resampling RNG is seeded, so results are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of :func:`paired_bootstrap`."""

    baseline: str
    system: str
    metric: str
    n_queries: int
    n_resamples: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict."""
        return asdict(self)


def paired_bootstrap(
    system: Sequence[float],
    baseline: Sequence[float],
    *,
    system_name: str = "system",
    baseline_name: str = "baseline",
    metric: str = "ndcg@10",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    chunk: int = 1000,
) -> BootstrapResult:
    """Paired percentile bootstrap CI and shifted-bootstrap p-value for mean(system - baseline)."""
    a = np.asarray(system, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("system and baseline must be paired (same length)")
    n = a.size
    if n == 0:
        return BootstrapResult(baseline_name, system_name, metric, 0, n_resamples, 0.0, 0.0, 0.0, 1.0, False)
    d = a - b
    observed = float(d.mean())
    centered = d - observed
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=np.float64)
    boot_null = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        boot[done : done + m] = d[idx].mean(axis=1)
        boot_null[done : done + m] = centered[idx].mean(axis=1)
        done += m
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    extreme = int(np.sum(np.abs(boot_null) >= abs(observed) - 1e-15))
    p = (extreme + 1) / (n_resamples + 1)
    return BootstrapResult(
        baseline=baseline_name,
        system=system_name,
        metric=metric,
        n_queries=int(n),
        n_resamples=n_resamples,
        mean_delta=observed,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=float(p),
        significant=bool(p < alpha and not (lo <= 0.0 <= hi)),
    )
