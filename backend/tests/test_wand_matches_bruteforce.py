"""WAND must return exactly the brute-force top-k (ids, order and scores) on random queries."""

import random
from pathlib import Path

import pytest

from app.core.bm25 import BM25Scorer, WandStats
from app.core.inverted_index import InvertedIndex, build_index, iter_chunks
from app.core.tokenizer import Tokenizer, TokenizerConfig

VOCAB = [f"w{i}" for i in range(120)]


@pytest.fixture(scope="module")
def random_index(tmp_path_factory: pytest.TempPathFactory) -> InvertedIndex:
    rng = random.Random(1234)
    weights = [1.0 / (i + 1) for i in range(len(VOCAB))]  # Zipf-like
    docs = []
    for i in range(600):
        n = rng.randint(3, 60)
        docs.append((f"d{i}", "", " ".join(rng.choices(VOCAB, weights=weights, k=n))))
    out: Path = tmp_path_factory.mktemp("wand")
    tok = Tokenizer(TokenizerConfig(use_stemmer=False, use_stopwords=False))
    build_index(
        iter_chunks(docs, 0, 0), out, tok, chunk_size=0, chunk_overlap=0, skip_interval=4, k1=1.2, b=0.75
    )
    return InvertedIndex(out)


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.6, 0.25), (2.0, 0.9), (0.9, 0.4)])
def test_wand_equals_bruteforce(random_index: InvertedIndex, k1: float, b: float) -> None:
    rng = random.Random(99)
    scorer = BM25Scorer(random_index, k1, b)
    total_scored = 0
    for _ in range(60):
        q = rng.sample(VOCAB, rng.randint(1, 6))
        if rng.random() < 0.3:
            q.append(q[0])  # repeated term => qtf 2
        for k in (1, 5, 10, 50):
            stats = WandStats()
            got = scorer.wand(q, k, stats)
            ref = scorer.brute_force(q, k)
            assert [d for d, _ in got] == [d for d, _ in ref]
            assert [s for _, s in got] == pytest.approx([s for _, s in ref], rel=1e-12, abs=1e-12)
            total_scored += stats.fully_scored
    assert total_scored > 0


def test_wand_skips_work(random_index: InvertedIndex) -> None:
    scorer = BM25Scorer(random_index)
    q = ["w0", "w1", "w100", "w110"]
    stats = WandStats()
    scorer.wand(q, 5, stats)
    candidates = {d for t in q for d in random_index.postings(t)[0]}
    assert stats.fully_scored < len(candidates)


def test_k_larger_than_matches(random_index: InvertedIndex) -> None:
    scorer = BM25Scorer(random_index)
    got = scorer.wand(["w119"], 10_000)
    assert len(got) == random_index.df("w119")
    assert got == scorer.brute_force(["w119"], 10_000)
