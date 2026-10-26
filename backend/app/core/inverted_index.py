"""Hand-built positional inverted index with delta + varint compression and skip pointers.

On-disk layout (see README §9 for a diagram)::

    postings.bin
    ├── header (16 bytes): b"HYBRIDIX" | u32 version | u32 reserved
    ├── term block for term_0  (terms in lexicographic order)
    │   ├── varint n_blocks
    │   ├── n_blocks x skip entry: varint first_doc | varint last_doc
    │   │                          | varint prev_doc+1 | varint data_offset
    │   └── postings data: for each posting
    │       varint doc_delta | varint tf | varint pos_nbytes | pos_nbytes of varint position deltas
    ├── term block for term_1 ...
    └── doc lengths: u32[n_chunks]   (4-byte aligned, read with np.frombuffer over the mmap)

    meta.json      deterministic: dictionary, chunk map, tokenizer + chunking config
    build_stats.json  non-deterministic stats (build time, peak RSS)

Internal ids are *chunk* ids (0..n_chunks-1) assigned in corpus order; ``chunks[i]`` maps a
chunk id to ``(doc_id, chunk_no)``. A skip entry is written every ``skip_interval`` postings;
``prev_doc+1`` lets a reader start delta-decoding in the middle of the list.

Each dictionary entry stores ``df``, ``offset``, ``length``, ``max_score`` (the exact BM25
upper bound at the build-time k1/b, used by WAND) and ``front``: the Pareto front of
(tf, doc_len) pairs, from which an *exact* upper bound for any other (k1, b) is computed.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import math
import mmap
import os
import resource
import struct
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.tokenizer import Tokenizer, TokenizerConfig

MAGIC = b"HYBRIDIX"
VERSION = 1
HEADER = struct.Struct("<8sII")
EXHAUSTED = 2**62
POSTINGS_FILE = "postings.bin"
META_FILE = "meta.json"
BUILD_STATS_FILE = "build_stats.json"


# --------------------------------------------------------------------------- varint codec
def encode_varint(value: int, out: bytearray) -> None:
    """Append LEB128-style unsigned varint (7 bits per byte, MSB = continuation)."""
    if value < 0:
        raise ValueError(f"varint must be non-negative, got {value}")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def decode_varint(buf: Any, pos: int) -> tuple[int, int]:
    """Decode one varint from ``buf`` (bytes/mmap/memoryview) at ``pos``; returns (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def encode_deltas(values: list[int]) -> bytes:
    """Delta + varint encode a strictly increasing list (first value stored as-is)."""
    out = bytearray()
    prev = 0
    for v in values:
        encode_varint(v - prev, out)
        prev = v
    return bytes(out)


def decode_deltas(buf: Any, pos: int, end: int) -> list[int]:
    """Inverse of :func:`encode_deltas` for the byte range [pos, end)."""
    values: list[int] = []
    acc = 0
    while pos < end:
        delta, pos = decode_varint(buf, pos)
        acc += delta
        values.append(acc)
    return values


# --------------------------------------------------------------------------- chunking
@dataclass(frozen=True)
class Chunk:
    """A unit of indexing. ``chunk_id`` is the dense internal id used everywhere."""

    chunk_id: int
    doc_id: str
    chunk_no: int
    text: str


def split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping windows of ``chunk_size`` whitespace tokens.

    A no-op (single window, original text) when ``chunk_size <= 0`` or the text is not longer
    than ``chunk_size`` words. ``overlap`` is clamped to ``chunk_size - 1``.
    """
    words = text.split()
    if chunk_size <= 0 or len(words) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size - 1))
    step = chunk_size - overlap
    windows: list[str] = []
    start = 0
    while True:
        windows.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return windows


def iter_chunks(docs: Iterable[tuple[str, str, str]], chunk_size: int, chunk_overlap: int) -> Iterator[Chunk]:
    """Yield chunks for ``(doc_id, title, text)`` triples in a deterministic order.

    The title is prepended to every chunk so each chunk is self-describing for both BM25 and
    the dense encoder.
    """
    next_id = 0
    for doc_id, title, text in docs:
        for chunk_no, body in enumerate(split_words(text, chunk_size, chunk_overlap)):
            chunk_text = f"{title}\n{body}" if title else body
            yield Chunk(chunk_id=next_id, doc_id=doc_id, chunk_no=chunk_no, text=chunk_text)
            next_id += 1


# --------------------------------------------------------------------------- BM25 bound helpers
def bm25_term_weight(tf: float, dl: float, avgdl: float, k1: float, b: float) -> float:
    """BM25 saturation component tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl)) (without IDF)."""
    norm = k1 * (1.0 - b + b * dl / avgdl) if avgdl > 0 else k1
    return tf * (k1 + 1.0) / (tf + norm) if tf > 0 else 0.0


def robertson_idf(n_docs: int, df: int) -> float:
    """Robertson/Sparck-Jones IDF with +1 smoothing (Lucene variant), always > 0."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def _pareto_insert(front: list[list[int]], tf: int, dl: int) -> None:
    """Maintain the Pareto front of (high tf, low dl) pairs in place."""
    for f_tf, f_dl in front:
        if f_tf >= tf and f_dl <= dl:
            return
    front[:] = [p for p in front if not (tf >= p[0] and dl <= p[1])]
    front.append([tf, dl])


# --------------------------------------------------------------------------- builder
class _TermAccumulator:
    __slots__ = ("blocks", "data", "df", "front", "last_doc", "n_positions")

    def __init__(self) -> None:
        self.data = bytearray()
        self.df = 0
        self.last_doc = -1
        self.blocks: list[list[int]] = []  # [first_doc, last_doc, prev+1, data_offset]
        self.front: list[list[int]] = []
        self.n_positions = 0

    def add(self, doc: int, positions: list[int], dl: int, skip_interval: int) -> None:
        if self.df % skip_interval == 0:
            self.blocks.append([doc, doc, self.last_doc + 1, len(self.data)])
        else:
            self.blocks[-1][1] = doc
        encode_varint(doc - self.last_doc, self.data)
        tf = len(positions)
        encode_varint(tf, self.data)
        pos_bytes = encode_deltas(positions)
        encode_varint(len(pos_bytes), self.data)
        self.data += pos_bytes
        self.last_doc = doc
        self.df += 1
        self.n_positions += tf
        _pareto_insert(self.front, tf, dl)

    def serialize(self) -> bytes:
        header = bytearray()
        encode_varint(len(self.blocks), header)
        for first, last, prev1, off in self.blocks:
            encode_varint(first, header)
            encode_varint(last, header)
            encode_varint(prev1, header)
            encode_varint(off, header)
        return bytes(header) + bytes(self.data)


@dataclass
class IndexBuildStats:
    """Statistics reported after a build (written to build_stats.json / index_stats.json)."""

    n_docs: int
    n_chunks: int
    vocab_size: int
    avg_chunk_len: float
    total_postings: int
    total_positions: int
    postings_bytes: int
    meta_bytes: int
    index_bytes: int
    raw_bytes_estimate: int
    compression_ratio: float
    build_seconds: float
    peak_rss_mb: float
    chunk_size: int
    chunk_overlap: int
    skip_interval: int
    tokenizer: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return dict(self.__dict__)


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_index(
    chunks: Iterable[Chunk],
    out_dir: Path,
    tokenizer: Tokenizer,
    *,
    chunk_size: int,
    chunk_overlap: int,
    skip_interval: int,
    k1: float,
    b: float,
) -> IndexBuildStats:
    """Build and serialize the index for ``chunks`` (must arrive in chunk_id order).

    The output is byte-for-byte deterministic for identical input and configuration.
    """
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    acc: dict[str, _TermAccumulator] = {}
    doc_lengths: list[int] = []
    chunk_map: list[list[Any]] = []
    doc_ids: list[str] = []
    seen_docs: set[str] = set()

    for expected_id, chunk in enumerate(chunks):
        if chunk.chunk_id != expected_id:
            raise ValueError("chunks must be supplied in increasing chunk_id order")
        terms = tokenizer.tokenize(chunk.text)
        dl = len(terms)
        doc_lengths.append(dl)
        chunk_map.append([chunk.doc_id, chunk.chunk_no])
        if chunk.doc_id not in seen_docs:
            seen_docs.add(chunk.doc_id)
            doc_ids.append(chunk.doc_id)
        positions: dict[str, list[int]] = {}
        for pos, term in enumerate(terms):
            positions.setdefault(term, []).append(pos)
        for term, plist in positions.items():
            ta = acc.get(term)
            if ta is None:
                ta = acc[term] = _TermAccumulator()
            ta.add(chunk.chunk_id, plist, dl, skip_interval)

    n_chunks = len(doc_lengths)
    total_len = sum(doc_lengths)
    avgdl = total_len / n_chunks if n_chunks else 0.0

    postings_path = out_dir / POSTINGS_FILE
    terms_meta: dict[str, list[Any]] = {}
    total_postings = 0
    total_positions = 0
    with open(postings_path, "wb") as fh:
        fh.write(HEADER.pack(MAGIC, VERSION, 0))
        offset = HEADER.size
        for term in sorted(acc):
            ta = acc[term]
            blob = ta.serialize()
            fh.write(blob)
            idf = robertson_idf(n_chunks, ta.df)
            max_score = max(bm25_term_weight(tf, dl, avgdl, k1, b) for tf, dl in ta.front) * idf
            front = sorted(ta.front)
            terms_meta[term] = [ta.df, offset, len(blob), round(max_score, 12), front]
            offset += len(blob)
            total_postings += ta.df
            total_positions += ta.n_positions
        pad = (-offset) % 4
        fh.write(b"\x00" * pad)
        doclen_offset = offset + pad
        fh.write(np.asarray(doc_lengths, dtype="<u4").tobytes())

    meta = {
        "version": VERSION,
        "tokenizer": tokenizer.config.to_dict(),
        "n_chunks": n_chunks,
        "n_docs": len(doc_ids),
        "total_tokens": total_len,
        "avgdl": avgdl,
        "skip_interval": skip_interval,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "build_k1": k1,
        "build_b": b,
        "doclen_offset": doclen_offset,
        "chunks": chunk_map,
        "terms": terms_meta,
    }
    meta_path = out_dir / META_FILE
    meta_path.write_text(json.dumps(meta, sort_keys=True, separators=(",", ":")))

    postings_bytes = postings_path.stat().st_size
    meta_bytes = meta_path.stat().st_size
    raw_estimate = total_postings * 8 + total_positions * 4 + n_chunks * 4
    stats = IndexBuildStats(
        n_docs=len(doc_ids),
        n_chunks=n_chunks,
        vocab_size=len(terms_meta),
        avg_chunk_len=avgdl,
        total_postings=total_postings,
        total_positions=total_positions,
        postings_bytes=postings_bytes,
        meta_bytes=meta_bytes,
        index_bytes=postings_bytes + meta_bytes,
        raw_bytes_estimate=raw_estimate,
        compression_ratio=(raw_estimate / postings_bytes) if postings_bytes else 0.0,
        build_seconds=time.perf_counter() - t0,
        peak_rss_mb=peak_rss_mb(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        skip_interval=skip_interval,
        tokenizer=tokenizer.config.to_dict(),
    )
    (out_dir / BUILD_STATS_FILE).write_text(json.dumps(stats.to_dict(), indent=2))
    return stats


# --------------------------------------------------------------------------- reader
@dataclass(frozen=True)
class TermInfo:
    """Dictionary entry for one term."""

    df: int
    offset: int
    length: int
    max_score: float
    front: tuple[tuple[int, int], ...]


class PostingsCursor:
    """Forward-only iterator over one postings list that decodes lazily, block by block.

    ``advance(target)`` uses the skip table (``last_doc`` of every block) to jump directly to
    the block that may contain ``target`` without decoding the blocks in between.
    """

    __slots__ = (
        "_block",
        "_buf",
        "_data_start",
        "_docs",
        "_end",
        "_firsts",
        "_i",
        "_lasts",
        "_offsets",
        "_pos",
        "_prevs",
        "_tfs",
        "blocks_decoded",
        "blocks_skipped",
        "df",
        "term",
    )

    def __init__(self, buf: Any, term: str, info: TermInfo) -> None:
        self._buf = buf
        self.term = term
        self.df = info.df
        pos = info.offset
        n_blocks, pos = decode_varint(buf, pos)
        firsts: list[int] = []
        lasts: list[int] = []
        prevs: list[int] = []
        offsets: list[int] = []
        for _ in range(n_blocks):
            v, pos = decode_varint(buf, pos)
            firsts.append(v)
            v, pos = decode_varint(buf, pos)
            lasts.append(v)
            v, pos = decode_varint(buf, pos)
            prevs.append(v - 1)
            v, pos = decode_varint(buf, pos)
            offsets.append(v)
        self._firsts = firsts
        self._lasts = lasts
        self._prevs = prevs
        self._offsets = offsets
        self._data_start = pos
        self._end = info.offset + info.length
        self._block = -1
        self._docs: list[int] = []
        self._tfs: list[int] = []
        self._pos: list[tuple[int, int]] = []
        self._i = 0
        self.blocks_decoded = 0
        self.blocks_skipped = 0
        if n_blocks:
            self._load_block(0)

    # -- block decoding ---------------------------------------------------------------
    def _load_block(self, block: int) -> None:
        buf = self._buf
        pos = self._data_start + self._offsets[block]
        end = self._data_start + self._offsets[block + 1] if block + 1 < len(self._offsets) else self._end
        prev = self._prevs[block]
        docs: list[int] = []
        tfs: list[int] = []
        ppos: list[tuple[int, int]] = []
        while pos < end:
            delta, pos = decode_varint(buf, pos)
            prev += delta
            tf, pos = decode_varint(buf, pos)
            nbytes, pos = decode_varint(buf, pos)
            docs.append(prev)
            tfs.append(tf)
            ppos.append((pos, pos + nbytes))
            pos += nbytes
        self._block = block
        self._docs = docs
        self._tfs = tfs
        self._pos = ppos
        self._i = 0
        self.blocks_decoded += 1

    # -- public API -------------------------------------------------------------------
    @property
    def doc(self) -> int:
        """Current chunk id or :data:`EXHAUSTED`."""
        if self._i < len(self._docs):
            return self._docs[self._i]
        return EXHAUSTED

    @property
    def tf(self) -> int:
        """Term frequency at the current posting."""
        return self._tfs[self._i]

    def positions(self) -> list[int]:
        """Decode the positions of the current posting."""
        start, end = self._pos[self._i]
        return decode_deltas(self._buf, start, end)

    def next(self) -> int:
        """Move to the next posting; returns the new current doc."""
        self._i += 1
        if self._i >= len(self._docs) and self._block + 1 < len(self._offsets):
            self._load_block(self._block + 1)
        return self.doc

    def advance(self, target: int) -> int:
        """Move to the first posting with doc >= ``target`` (no-op if already there)."""
        if self.doc >= target:
            return self.doc
        if target > self._lasts[self._block]:
            nb = bisect.bisect_left(self._lasts, target, lo=self._block + 1)
            if nb >= len(self._lasts):
                self._docs = []
                self._i = 0
                self._block = len(self._lasts) - 1
                return EXHAUSTED
            self.blocks_skipped += nb - self._block - 1
            self._load_block(nb)
        self._i = bisect.bisect_left(self._docs, target, lo=self._i)
        return self.doc


class InvertedIndex:
    """Read-only, mmap-backed view over a serialized index directory."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        meta = json.loads((self.index_dir / META_FILE).read_text())
        if meta.get("version") != VERSION:
            raise ValueError(f"unsupported index version {meta.get('version')}")
        self.meta = meta
        self.tokenizer = Tokenizer(TokenizerConfig(**meta["tokenizer"]))
        self.n_chunks: int = meta["n_chunks"]
        self.n_docs: int = meta["n_docs"]
        self.avgdl: float = meta["avgdl"]
        self.skip_interval: int = meta["skip_interval"]
        self.build_k1: float = meta["build_k1"]
        self.build_b: float = meta["build_b"]
        self.terms: dict[str, TermInfo] = {
            t: TermInfo(v[0], v[1], v[2], v[3], tuple((int(a), int(c)) for a, c in v[4]))
            for t, v in meta["terms"].items()
        }
        self.chunk_doc: list[str] = [c[0] for c in meta["chunks"]]
        self.chunk_no: list[int] = [int(c[1]) for c in meta["chunks"]]
        self.doc_chunks: dict[str, list[int]] = {}
        for cid, did in enumerate(self.chunk_doc):
            self.doc_chunks.setdefault(did, []).append(cid)
        self._fh = open(self.index_dir / POSTINGS_FILE, "rb")  # noqa: SIM115 - kept open for mmap
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, _ = HEADER.unpack_from(self._mm, 0)
        if magic != MAGIC or version != VERSION:
            raise ValueError("postings.bin has a bad header")
        self.doc_lengths = np.frombuffer(
            self._mm, dtype="<u4", count=self.n_chunks, offset=meta["doclen_offset"]
        )
        stats_path = self.index_dir / BUILD_STATS_FILE
        self.build_stats: dict[str, Any] = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    # -- lookups ----------------------------------------------------------------------
    def has_term(self, term: str) -> bool:
        """True if the term occurs in the collection."""
        return term in self.terms

    def df(self, term: str) -> int:
        """Document (chunk) frequency of a term (0 if absent)."""
        info = self.terms.get(term)
        return info.df if info else 0

    def idf(self, term: str) -> float:
        """Robertson IDF with +1 smoothing for ``term``."""
        return robertson_idf(self.n_chunks, self.df(term))

    def cursor(self, term: str) -> PostingsCursor | None:
        """A fresh postings cursor, or ``None`` for an unknown term."""
        info = self.terms.get(term)
        return PostingsCursor(self._mm, term, info) if info else None

    def postings(self, term: str) -> tuple[list[int], list[int]]:
        """Fully decode a postings list into (chunk_ids, tfs)."""
        cur = self.cursor(term)
        docs: list[int] = []
        tfs: list[int] = []
        if cur is None:
            return docs, tfs
        while cur.doc != EXHAUSTED:
            docs.append(cur.doc)
            tfs.append(cur.tf)
            cur.next()
        return docs, tfs

    def positions(self, term: str, chunk_id: int) -> list[int]:
        """Positions of ``term`` inside ``chunk_id`` (empty list if absent)."""
        cur = self.cursor(term)
        if cur is None or cur.advance(chunk_id) != chunk_id:
            return []
        return cur.positions()

    def tf(self, term: str, chunk_id: int) -> int:
        """Term frequency of ``term`` in ``chunk_id`` via a skip-pointer seek."""
        cur = self.cursor(term)
        if cur is None or cur.advance(chunk_id) != chunk_id:
            return 0
        return cur.tf

    def upper_bound(self, term: str, k1: float, b: float) -> float:
        """Exact max BM25 contribution of ``term`` over its postings for (k1, b)."""
        info = self.terms[term]
        if k1 == self.build_k1 and b == self.build_b:
            return info.max_score
        idf = robertson_idf(self.n_chunks, info.df)
        return idf * max(bm25_term_weight(tf, dl, self.avgdl, k1, b) for tf, dl in info.front)

    def phrase_match(self, phrase: list[str], chunk_id: int) -> bool:
        """True if ``phrase`` occurs as consecutive analyzed terms in ``chunk_id``."""
        if not phrase:
            return False
        plists = [set(self.positions(t, chunk_id)) for t in phrase]
        if any(not p for p in plists):
            return False
        return any(all((s + i) in plists[i] for i in range(1, len(phrase))) for s in plists[0])

    def stats(self) -> dict[str, Any]:
        """Index statistics (deterministic part + build-time stats if present)."""
        size = os.path.getsize(self.index_dir / POSTINGS_FILE) + os.path.getsize(self.index_dir / META_FILE)
        out = dict(self.build_stats)
        out.update(
            n_docs=self.n_docs,
            n_chunks=self.n_chunks,
            vocab_size=len(self.terms),
            avg_chunk_len=self.avgdl,
            index_bytes=size,
            skip_interval=self.skip_interval,
            tokenizer=self.tokenizer.config.to_dict(),
        )
        return out

    def close(self) -> None:
        """Release the mmap and file handle."""
        # numpy views (doc_lengths) may still reference the buffer; GC will release it then.
        with contextlib.suppress(BufferError):
            self._mm.close()
        self._fh.close()
