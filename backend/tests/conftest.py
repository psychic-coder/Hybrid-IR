"""Shared fixtures: a tiny corpus, offline fake models and an in-memory Redis.

Nothing here touches the network or needs torch, so the suite runs in CI on a bare runner.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core.cache import Cache
from app.core.dense import Embedder
from app.core.inverted_index import InvertedIndex, build_index, iter_chunks
from app.core.pipeline import EffectiveParams, SearchPipeline
from app.core.reranker import Reranker
from app.core.tokenizer import Tokenizer, TokenizerConfig
from app.core.vector_store import ExactStore
from app.eval.datasets import Document

TINY_DOCS: list[Document] = [
    Document("d1", "Lung cancer", "Smoking is the leading cause of lung cancer in adults."),
    Document("d2", "Heart disease", "Exercise reduces the risk of heart disease and stroke."),
    Document("d3", "Vitamin D", "Vitamin D deficiency is linked to bone loss and fractures."),
    Document("d4", "Cancer screening", "Early screening improves survival for many cancer types."),
    Document("d5", "Sleep", "Poor sleep quality is associated with obesity and diabetes risk."),
    Document("d6", "Air pollution", "Air pollution increases lung disease and cancer incidence."),
    Document("d7", "Diet", "A Mediterranean diet lowers cardiovascular disease risk."),
    Document("d8", "Genetics", "BRCA mutations raise breast cancer and ovarian cancer risk."),
]


class HashingModel:
    """Deterministic bag-of-words hashing encoder standing in for SentenceTransformer."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.tok = Tokenizer()

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences: Sequence[str], **_: Any) -> np.ndarray:
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            for t in self.tok.tokenize(s):
                h = int(hashlib.md5(t.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            out[i, (i * 7) % self.dim] += 1e-3
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class OverlapCrossEncoder:
    """Scores a (query, text) pair by analyzed-term overlap; deterministic and fast."""

    def __init__(self) -> None:
        self.tok = Tokenizer()

    def predict(self, sentences: Sequence[tuple[str, str]], **_: Any) -> list[float]:
        out = []
        for q, t in sentences:
            qs = set(self.tok.tokenize(q))
            ts = self.tok.tokenize(t)
            out.append(sum(1.0 for x in ts if x in qs) / (1 + len(ts)) + 0.001 * len(qs))
        return out


class FakeRedis:
    """Tiny in-memory subset of redis-py."""

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self.hashes: dict[str, dict[str, int]] = {}

    def get(self, name: str) -> bytes | None:
        return self.kv.get(name)

    def set(self, name: str, value: Any, ex: int | None = None) -> bool:
        self.kv[name] = value if isinstance(value, bytes) else str(value).encode()
        return True

    def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        h = self.hashes.setdefault(name, {})
        h[key] = h.get(key, 0) + amount
        return h[key]

    def hgetall(self, name: str) -> dict[bytes, bytes]:
        return {k.encode(): str(v).encode() for k, v in self.hashes.get(name, {}).items()}

    def delete(self, *names: str) -> int:
        n = 0
        for nm in names:
            n += int(self.kv.pop(nm, None) is not None) + int(self.hashes.pop(nm, None) is not None)
        return n

    def ping(self) -> bool:
        return True

    def info(self, section: str | None = None) -> dict[str, int]:
        return {"used_memory": 1024}

    def dbsize(self) -> int:
        return len(self.kv)


def build_tiny_index(tmp: Path, docs: list[Document] | None = None, **kw: Any) -> InvertedIndex:
    """Build and load an index over ``docs`` (default TINY_DOCS)."""
    docs = docs if docs is not None else TINY_DOCS
    tok = Tokenizer(TokenizerConfig())
    build_index(
        iter_chunks(
            ((d.doc_id, d.title, d.text) for d in docs),
            kw.get("chunk_size", 256),
            kw.get("chunk_overlap", 32),
        ),
        tmp,
        tok,
        chunk_size=kw.get("chunk_size", 256),
        chunk_overlap=kw.get("chunk_overlap", 32),
        skip_interval=kw.get("skip_interval", 2),
        k1=1.2,
        b=0.75,
    )
    return InvertedIndex(tmp)


DEFAULT_PARAMS = EffectiveParams(
    k1=1.2, b=0.75, rrf_k=60, alpha=0.5, norm="minmax", rerank_top_k=5, ef=32, candidate_depth=20
)


def make_pipeline(tmp: Path, docs: list[Document] | None = None) -> SearchPipeline:
    """A complete pipeline over a tiny corpus with fake models and exact dense search."""
    docs = docs if docs is not None else TINY_DOCS
    index = build_tiny_index(tmp / "index", docs)
    cache = Cache(client=FakeRedis(), ttl_seconds=60)
    embedder = Embedder("fake-model", cache=cache, model=HashingModel())
    chunk_texts = [c.text for c in iter_chunks(((d.doc_id, d.title, d.text) for d in docs), 256, 32)]
    emb = embedder.encode_documents(chunk_texts)
    return SearchPipeline(
        index=index,
        docs={d.doc_id: d for d in docs},
        cache=cache,
        defaults=DEFAULT_PARAMS,
        embedder=embedder,
        dense=ExactStore(emb),
        reranker=Reranker("fake-reranker", model=OverlapCrossEncoder()),
        embeddings=emb,
        eval_depth=20,
        dataset="tiny",
    )


@pytest.fixture()
def tiny_index(tmp_path: Path) -> InvertedIndex:
    """Index over TINY_DOCS with a skip interval of 2."""
    return build_tiny_index(tmp_path / "idx")


@pytest.fixture()
def pipeline(tmp_path: Path) -> SearchPipeline:
    """Full fake pipeline."""
    return make_pipeline(tmp_path)


def write_json(path: Path, obj: Any) -> None:
    """Helper for tests writing result fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
