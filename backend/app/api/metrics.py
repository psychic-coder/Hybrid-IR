"""GET /metrics/summary, /metrics/ablations, /metrics/cache.

Every number served here is read from files the evaluation harness wrote into RESULTS_DIR;
nothing is computed or hard-coded in the API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.deps import AppState, get_state
from app.schemas import AblationsResponse, CacheStats, LoadTestSummary, SummaryResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])
log = logging.getLogger(__name__)
NOT_RUN = "No evaluation results yet. Run `make eval` (or wait for the seed service) to populate."


def _read(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def load_summary(results_dir: Path) -> dict[str, Any]:
    """Assemble the summary payload from results/*.json."""
    summary = _read(results_dir / "summary.json")
    if not summary:
        return {"available": False, "message": NOT_RUN}
    payload: dict[str, Any] = {
        "available": True,
        "message": None,
        "dataset": summary.get("dataset"),
        "generated_at": summary.get("generated_at"),
        "n_queries": summary.get("n_queries"),
        "environment": {k: str(v) for k, v in (summary.get("environment") or {}).items()},
        "strategies": summary.get("strategies", []),
        "index_stats": _read(results_dir / "index_stats.json"),
        "pareto": (_read(results_dir / "pareto.json") or {}).get("points", []),
        "load_test": None,
    }
    k6 = _read(results_dir / "k6_summary.json")
    if k6:
        try:
            payload["load_test"] = LoadTestSummary.model_validate(k6).model_dump()
        except ValidationError as exc:
            log.warning("ignoring malformed k6_summary.json: %s", exc)
    return payload


@router.get("/summary", response_model=SummaryResponse)
async def summary(state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Latest evaluation results for every configuration, plus index stats and Pareto points."""
    payload = await run_in_threadpool(load_summary, state.settings.results_dir)
    try:
        return SummaryResponse.model_validate(payload).model_dump()
    except ValidationError as exc:
        return {
            "available": False,
            "message": f"results/summary.json is malformed: {exc.error_count()} errors",
        }


@router.get("/ablations", response_model=AblationsResponse)
async def ablations(state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Ablation tables written by ``scripts.run_ablations`` (or the seed service)."""
    data = await run_in_threadpool(_read, state.settings.results_dir / "ablations.json")
    if not data:
        return {"available": False, "message": "No ablation results yet. Run `make ablations`."}
    data = {**data, "available": True, "message": None}
    try:
        return AblationsResponse.model_validate(data).model_dump()
    except ValidationError as exc:
        return {
            "available": False,
            "message": f"results/ablations.json is malformed: {exc.error_count()} errors",
        }


@router.get("/cache", response_model=CacheStats)
async def cache_stats(state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Redis hit/miss counters for the embedding and result caches."""
    return await run_in_threadpool(state.cache.stats)
