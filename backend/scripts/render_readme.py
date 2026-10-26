"""Render results/*.json into README.md between the RESULTS markers.

    python -m scripts.render_readme --results-dir /results --readme /repo/README.md
    # or on the host (standard library only):
    python3 backend/scripts/render_readme.py --results-dir results --readme README.md

Everything between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` is replaced. When no
results exist, the block says "Run `make eval` to populate." No number is ever typed by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # executed as a file on the host
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.report import render_markdown

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def render(readme_text: str, results_dir: Path) -> str:
    """Return README text with the results block replaced."""
    if START not in readme_text or END not in readme_text:
        raise ValueError("README is missing the RESULTS:START/END markers")
    head, rest = readme_text.split(START, 1)
    _, tail = rest.split(END, 1)
    body = render_markdown(results_dir, heading_level=3)
    return f"{head}{START}\n{body}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--check", action="store_true", help="exit 1 if the README is out of date")
    args = ap.parse_args(argv)
    original = args.readme.read_text()
    updated = render(original, args.results_dir)
    if args.check:
        if updated != original:
            print("README results block is out of date; run `make readme`.")
            return 1
        return 0
    args.readme.write_text(updated)
    print(f"updated {args.readme} from {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
