"""Run the baseline evaluation (all strategies) and rewrite every results/* file.

    python -m scripts.run_eval [--max-queries N] [--ablations]

Requires a completed seed (index, embeddings, vector store). ``make eval`` runs this inside
the backend image with access to Qdrant, Redis and the shared volumes.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.deps import build_state
from app.eval.report import render_markdown
from app.eval.runner import run_evaluation
from app.logging_setup import setup_logging

log = logging.getLogger("run_eval")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-queries", type=int, default=None, help="evaluate only the first N queries")
    ap.add_argument("--ablations", action="store_true", help="also run every ablation grid")
    args = ap.parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)
    state = build_state(settings)
    if state.pipeline is None:
        log.error("index not loaded: %s", state.load_errors)
        return 1
    if state.load_errors:
        log.warning("some components failed to load; their strategies will be skipped: %s", state.load_errors)
    run_evaluation(state.pipeline, settings, include_ablations=args.ablations, max_queries=args.max_queries)
    print(render_markdown(settings.results_dir))
    state.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
