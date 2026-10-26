"""Run ablation grids and rewrite results/ablations.json, pareto.json and report.md.

python -m scripts.run_ablations [--only bm25_grid,chunk_size,fusion,hnsw,rerank_top_k]
                                [--max-queries N]
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.deps import build_state
from app.eval import report
from app.eval.ablations import ALL, run_ablations
from app.eval.datasets import load_dataset
from app.logging_setup import setup_logging

log = logging.getLogger("run_ablations")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(ALL), help=f"comma separated subset of {ALL}")
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args(argv)
    which = [w.strip() for w in args.only.split(",") if w.strip()]
    unknown = set(which) - set(ALL)
    if unknown:
        ap.error(f"unknown ablations: {sorted(unknown)}")
    settings = get_settings()
    setup_logging(settings.log_level)
    state = build_state(settings)
    if state.pipeline is None:
        log.error("index not loaded: %s", state.load_errors)
        return 1
    dataset = load_dataset(settings.dataset, settings.raw_dir)
    new = run_ablations(state.pipeline, dataset, settings, which=which, max_queries=args.max_queries)
    # merge with previous tables so running one ablation does not erase the others
    old = report.read_json(settings.results_dir / "ablations.json") or {}
    merged = dict(old)
    for key, value in new.items():
        if key in ("bm25_grid", "chunk_size", "rrf_k", "alpha_sweep", "hnsw", "rerank_top_k"):
            if value or key not in merged:
                merged[key] = value
        else:
            merged[key] = value
    report.write_ablations(settings.results_dir, merged)
    report.write_pareto(settings.results_dir)
    report.write_report_md(settings.results_dir)
    print(report.render_markdown(settings.results_dir))
    state.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
