import random
from pathlib import Path

import pytest

from app.core.inverted_index import (
    EXHAUSTED,
    InvertedIndex,
    decode_deltas,
    decode_varint,
    encode_deltas,
    encode_varint,
    iter_chunks,
    split_words,
)
from tests.conftest import TINY_DOCS, build_tiny_index


@pytest.mark.parametrize("value", [0, 1, 127, 128, 255, 300, 16383, 16384, 2**31, 2**40 + 7])
def test_varint_roundtrip(value: int) -> None:
    buf = bytearray()
    encode_varint(value, buf)
    decoded, pos = decode_varint(bytes(buf), 0)
    assert decoded == value and pos == len(buf)


def test_varint_known_bytes() -> None:
    buf = bytearray()
    encode_varint(300, buf)
    assert bytes(buf) == b"\xac\x02"  # 300 = 0b10_0101100


def test_varint_rejects_negative() -> None:
    with pytest.raises(ValueError):
        encode_varint(-1, bytearray())


def test_delta_roundtrip() -> None:
    vals = [3, 7, 8, 200, 100000]
    enc = encode_deltas(vals)
    assert decode_deltas(enc, 0, len(enc)) == vals


def test_chunking_noop_for_short_docs() -> None:
    assert split_words("a b c", 10, 2) == ["a b c"]
    assert split_words("a b c", 0, 2) == ["a b c"]


def test_chunking_windows_overlap_and_cover() -> None:
    text = " ".join(str(i) for i in range(10))
    windows = split_words(text, 4, 1)
    assert windows == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunk_ids_and_doc_mapping() -> None:
    docs = [("x", "T", " ".join(["w"] * 10)), ("y", "", "short")]
    chunks = list(iter_chunks(docs, 4, 0))
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert [c.doc_id for c in chunks] == ["x", "x", "x", "y"]
    assert chunks[0].text.startswith("T\n")


def test_postings_roundtrip(tiny_index: InvertedIndex) -> None:
    docs, tfs = tiny_index.postings("cancer")
    doc_ids = [tiny_index.chunk_doc[d] for d in docs]
    assert doc_ids == ["d1", "d4", "d6", "d8"]
    assert tfs[0] == 2  # title "Lung cancer" + body "lung cancer"
    assert tfs[3] == 2
    assert docs == sorted(docs)


def test_positions_and_phrase(tiny_index: InvertedIndex) -> None:
    cid = tiny_index.doc_chunks["d1"][0]
    pos_lung = tiny_index.positions("lung", cid)
    pos_cancer = tiny_index.positions("cancer", cid)
    assert pos_lung and pos_cancer
    assert tiny_index.phrase_match(["lung", "cancer"], cid)
    assert not tiny_index.phrase_match(["cancer", "lung"], tiny_index.doc_chunks["d8"][0])


def test_doc_lengths_and_stats(tiny_index: InvertedIndex) -> None:
    assert tiny_index.n_chunks == len(TINY_DOCS)
    assert tiny_index.doc_lengths.shape == (len(TINY_DOCS),)
    assert abs(tiny_index.avgdl - float(tiny_index.doc_lengths.mean())) < 1e-9
    st = tiny_index.stats()
    for key in ("vocab_size", "n_docs", "avg_chunk_len", "index_bytes", "build_seconds", "peak_rss_mb"):
        assert key in st


def test_skip_pointers_advance(tmp_path: Path) -> None:
    rng = random.Random(7)
    docs = []
    for i in range(400):
        words = ["common"] + [rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(5)]
        if i % 3 == 0:
            words.append("rare")
        docs.append((f"doc{i:03d}", "", " ".join(words)))
    from app.core.inverted_index import build_index
    from app.core.tokenizer import Tokenizer

    build_index(
        iter_chunks(docs, 0, 0),
        tmp_path,
        Tokenizer(),
        chunk_size=0,
        chunk_overlap=0,
        skip_interval=8,
        k1=1.2,
        b=0.75,
    )
    idx = InvertedIndex(tmp_path)
    full, _ = idx.postings("rare")
    cur = idx.cursor("rare")
    assert cur is not None
    for target in [0, 5, 100, 101, 250, 398]:
        expected = next((d for d in full if d >= target), EXHAUSTED)
        assert cur.advance(target) == expected
    assert cur.blocks_skipped > 0
    assert cur.advance(10_000) == EXHAUSTED
    idx.close()


def test_build_is_deterministic(tmp_path: Path) -> None:
    a = build_tiny_index(tmp_path / "a")
    b = build_tiny_index(tmp_path / "b")
    assert (tmp_path / "a" / "postings.bin").read_bytes() == (tmp_path / "b" / "postings.bin").read_bytes()
    assert (tmp_path / "a" / "meta.json").read_text() == (tmp_path / "b" / "meta.json").read_text()
    a.close()
    b.close()


def test_compression_beats_raw(tmp_path: Path) -> None:
    idx = build_tiny_index(tmp_path / "c")
    st = idx.build_stats
    assert st["raw_bytes_estimate"] > 0
    assert st["compression_ratio"] > 1.0
    idx.close()
