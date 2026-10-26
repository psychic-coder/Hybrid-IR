"""BM25 ranking over the hand-built inverted index.

score(q, d) = sum_{t in q} qtf(t) * IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Two top-k strategies are provided:

* :meth:`BM25Scorer.brute_force` - term-at-a-time over fully decoded postings. O(sum df).
  The reference implementation used by tests.
* :meth:`BM25Scorer.wand` - document-at-a-time WAND (Broder et al., 2003) with per-term
  upper bounds and skip-pointer ``advance``; returns exactly the same top-k (ties broken by
  smaller chunk id) while skipping chunks that cannot enter the heap.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field

from app.core.inverted_index import EXHAUSTED, InvertedIndex, PostingsCursor, bm25_term_weight
from app.core.tokenizer import ParsedQuery

_UB_EPS = 1e-9


@dataclass
class WandStats:
    """Counters describing how much work WAND avoided (exposed in /explain and tests)."""

    fully_scored: int = 0
    pivots: int = 0
    blocks_decoded: int = 0
    blocks_skipped: int = 0


@dataclass(frozen=True)
class DocHit:
    """A document-level BM25 hit (chunk scores aggregated with max)."""

    doc_id: str
    score: float
    chunk_id: int
    phrase_matched: bool = False


@dataclass
class BM25Result:
    """Ranked document hits plus work counters."""

    hits: list[DocHit]
    stats: WandStats = field(default_factory=WandStats)


class BM25Scorer:
    """BM25 with configurable k1 / b over an :class:`InvertedIndex`."""

    def __init__(
        self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75, phrase_boost: float = 1.0
    ) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self.phrase_boost = phrase_boost

    # ------------------------------------------------------------------ primitives
    def query_weights(self, terms: list[str]) -> dict[str, int]:
        """Query term frequencies restricted to terms that exist in the index."""
        return {t: c for t, c in sorted(Counter(terms).items()) if self.index.has_term(t)}

    def term_score(self, term: str, tf: int, chunk_id: int) -> float:
        """Contribution of one term occurrence count ``tf`` in ``chunk_id``."""
        dl = float(self.index.doc_lengths[chunk_id])
        return self.index.idf(term) * bm25_term_weight(tf, dl, self.index.avgdl, self.k1, self.b)

    def score_chunk(self, weights: dict[str, int], chunk_id: int) -> float:
        """Exact BM25 score of one chunk (uses skip-pointer seeks per term)."""
        total = 0.0
        for term, qtf in weights.items():
            tf = self.index.tf(term, chunk_id)
            if tf:
                total += qtf * self.term_score(term, tf, chunk_id)
        return total

    # ------------------------------------------------------------------ reference
    def brute_force(self, terms: list[str], k: int) -> list[tuple[int, float]]:
        """Exhaustive term-at-a-time scoring; returns top-k (chunk_id, score)."""
        weights = self.query_weights(terms)
        acc: dict[int, float] = {}
        for term, qtf in weights.items():
            idf = self.index.idf(term)
            docs, tfs = self.index.postings(term)
            for d, tf in zip(docs, tfs, strict=True):
                dl = float(self.index.doc_lengths[d])
                acc[d] = acc.get(d, 0.0) + qtf * idf * bm25_term_weight(
                    tf, dl, self.index.avgdl, self.k1, self.b
                )
        ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    # ------------------------------------------------------------------ WAND
    def wand(self, terms: list[str], k: int, stats: WandStats | None = None) -> list[tuple[int, float]]:
        """WAND top-k over chunks; identical output to :meth:`brute_force`."""
        stats = stats if stats is not None else WandStats()
        weights = self.query_weights(terms)
        if k <= 0 or not weights:
            return []
        cursors: list[PostingsCursor] = []
        ub: dict[str, float] = {}
        idf: dict[str, float] = {}
        for term, qtf in weights.items():
            cur = self.index.cursor(term)
            if cur is None:
                continue
            cursors.append(cur)
            ub[term] = qtf * self.index.upper_bound(term, self.k1, self.b) * (1 + _UB_EPS) + _UB_EPS
            idf[term] = self.index.idf(term)
        avgdl = self.index.avgdl
        k1, b = self.k1, self.b
        doc_lengths = self.index.doc_lengths

        # min-heap of (score, -chunk_id): root is the current k-th best (lowest) entry.
        heap: list[tuple[float, int]] = []
        while True:
            cursors = [c for c in cursors if c.doc != EXHAUSTED]
            if not cursors:
                break
            cursors.sort(key=lambda c: c.doc)
            threshold = heap[0][0] if len(heap) >= k else float("-inf")
            acc_ub = 0.0
            pivot = -1
            for i, c in enumerate(cursors):
                acc_ub += ub[c.term]
                if acc_ub >= threshold:
                    pivot = i
                    break
            if pivot < 0:
                break
            stats.pivots += 1
            pivot_doc = cursors[pivot].doc
            if cursors[0].doc == pivot_doc:
                dl = float(doc_lengths[pivot_doc])
                # sum in term order so floating-point results equal the TAAT reference exactly
                on_pivot = sorted((c for c in cursors if c.doc == pivot_doc), key=lambda c: c.term)
                score = 0.0
                for c in on_pivot:
                    score += weights[c.term] * idf[c.term] * bm25_term_weight(c.tf, dl, avgdl, k1, b)
                stats.fully_scored += 1
                entry = (score, -pivot_doc)
                if len(heap) < k:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
                for c in cursors:
                    if c.doc != pivot_doc:
                        break
                    c.next()
            else:
                # advance the preceding cursor with the largest upper bound to the pivot doc
                # (only cursors strictly behind the pivot can move, otherwise no progress)
                behind = [j for j in range(pivot) if cursors[j].doc < pivot_doc]
                best = max(behind, key=lambda j: (ub[cursors[j].term], -j))
                cursors[best].advance(pivot_doc)
        for c in cursors:
            stats.blocks_decoded += c.blocks_decoded
            stats.blocks_skipped += c.blocks_skipped
        ranked = sorted(((-neg, s) for s, neg in heap), key=lambda kv: (-kv[1], kv[0]))
        return ranked

    # ------------------------------------------------------------------ document level
    def search(self, parsed: ParsedQuery, k: int) -> BM25Result:
        """Top-k *documents*: WAND over chunks, max-aggregate per doc, phrase boost.

        When chunking produced several chunks per doc, the chunk depth is doubled until ``k``
        distinct documents are found (or the candidate set is exhausted).
        """
        stats = WandStats()
        if k <= 0:
            return BM25Result([], stats)
        depth = k
        while True:
            chunk_hits = self.wand(parsed.terms, depth, stats)
            best: dict[str, tuple[float, int]] = {}
            for cid, score in chunk_hits:
                did = self.index.chunk_doc[cid]
                if did not in best or score > best[did][0]:
                    best[did] = (score, cid)
            if len(best) >= k or len(chunk_hits) < depth:
                break
            depth *= 2
        hits: list[DocHit] = []
        for did, (score, cid) in best.items():
            matched = False
            if parsed.phrases and self.phrase_boost > 0:
                for phrase in parsed.phrases:
                    if self.index.phrase_match(phrase, cid):
                        matched = True
                        score += self.phrase_boost * sum(self.index.idf(t) for t in phrase)
            hits.append(DocHit(did, score, cid, matched))
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return BM25Result(hits[:k], stats)
