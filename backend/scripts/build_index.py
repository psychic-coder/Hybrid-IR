"""Rebuild only the inverted index (useful when experimenting with tokenizer/chunk settings).

    python -m scripts.build_index [--chunk-size 256] [--chunk-overlap 32] [--skip-interval 64]
                                  [--out /data/scifact/index]

Settings default to the environment (see .env.example). The dataset must already be
downloaded (run the seed once). Note that changing chunking here without re-embedding makes
the dense side inconsistent; use FORCE_RESEED=1 make seed for a full rebuild.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.core.inverted_index import build_index, iter_chunks
from app.core.tokenizer import Tokenizer, TokenizerConfig
from app.eval.datasets import iter_corpus
from app.logging_setup import setup_logging

log = logging.getLogger("build_index")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-size", type=int, default=s.chunk_size)
    ap.add_argument("--chunk-overlap", type=int, default=s.chunk_overlap)
    ap.add_argument("--skip-interval", type=int, default=s.skip_interval)
    ap.add_argument("--out", type=Path, default=s.index_dir)
    args = ap.parse_args(argv)
    setup_logging(s.log_level)
    tok = Tokenizer(TokenizerConfig(s.use_stemmer, s.use_stopwords, s.min_token_len))
    docs = ((d.doc_id, d.title, d.text) for d in iter_corpus(s.raw_dir))
    stats = build_index(
        iter_chunks(docs, args.chunk_size, args.chunk_overlap),
        args.out,
        tok,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        skip_interval=args.skip_interval,
        k1=s.bm25_k1,
        b=s.bm25_b,
    )
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
