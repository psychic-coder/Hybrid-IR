"""Dense vector search backends.

All backends implement :class:`DenseSearcher` and return ``(chunk_id, cosine_similarity)``
pairs sorted by descending similarity. Chunk ids are the same dense integer ids used by the
inverted index, which is what lets the pipeline join sparse and dense candidates.

* :class:`QdrantStore`  - Qdrant HNSW. One collection per (dataset, M, ef_construct); query-time
  ``ef`` is passed as ``SearchParams.hnsw_ef``. Small collections are forced through the HNSW
  index (``indexing_threshold`` / ``full_scan_threshold`` lowered) so that M / ef ablations
  measure HNSW rather than Qdrant's automatic full-scan fallback.
* :class:`CustomHNSWStore` - the from-scratch graph in :mod:`app.core.custom_hnsw`.
* :class:`ExactStore` - brute-force NumPy scan (ground truth; used by tests).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.core.custom_hnsw import CustomHNSW

log = logging.getLogger(__name__)


class DenseSearcher(Protocol):
    """Common interface for dense retrieval backends."""

    name: str

    def search(self, vector: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        """Approximate top-k chunk ids by cosine similarity."""
        ...

    def health(self) -> tuple[bool, str]:
        """(ok, detail) for readiness checks."""
        ...


class ExactStore:
    """Exhaustive inner-product search over an in-memory normalized matrix."""

    name = "exact"

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, vector: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        """Exact top-k (``ef`` ignored)."""
        n = self.vectors.shape[0]
        if n == 0 or k <= 0:
            return []
        sims = self.vectors @ np.asarray(vector, dtype=np.float32)
        k = min(k, n)
        top = np.argpartition(-sims, k - 1)[:k]
        order = sorted(top.tolist(), key=lambda i: (-float(sims[i]), i))
        return [(int(i), float(sims[i])) for i in order]

    def health(self) -> tuple[bool, str]:
        """Always healthy once constructed."""
        return True, f"exact scan over {self.vectors.shape[0]} vectors"


class CustomHNSWStore:
    """Adapter exposing :class:`CustomHNSW` as a :class:`DenseSearcher`."""

    name = "custom"

    def __init__(self, index: CustomHNSW) -> None:
        self.index = index

    @classmethod
    def load(cls, path: Path) -> CustomHNSWStore:
        """Load a graph saved by :meth:`CustomHNSW.save`."""
        return cls(CustomHNSW.load(path))

    def search(self, vector: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        """HNSW search with query-time ``ef``."""
        return self.index.search(vector, k, ef)

    def health(self) -> tuple[bool, str]:
        """Healthy when the graph has nodes."""
        ok = self.index.count > 0
        return ok, f"custom hnsw with {self.index.count} nodes" if ok else "custom hnsw is empty"


class QdrantStore:
    """Thin wrapper over ``qdrant_client`` for one collection."""

    name = "qdrant"

    def __init__(self, url: str, collection: str, timeout: float = 10.0, client: Any = None) -> None:
        self.url = url
        self.collection = collection
        self.timeout = timeout
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily constructed ``QdrantClient`` (REST)."""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, timeout=int(max(1, self.timeout)))
        return self._client

    def with_collection(self, collection: str) -> QdrantStore:
        """Another store sharing the same client but pointing to ``collection``."""
        return QdrantStore(self.url, collection, self.timeout, client=self._client)

    # ------------------------------------------------------------------ admin
    def exists(self) -> bool:
        """True if the collection exists."""
        return bool(self.client.collection_exists(self.collection))

    def count(self) -> int:
        """Number of points in the collection (0 if it does not exist)."""
        if not self.exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def ensure_collection(self, dim: int, m: int, ef_construct: int, recreate: bool = False) -> None:
        """Create the collection with the requested HNSW parameters (cosine distance)."""
        from qdrant_client import models

        if self.exists():
            if not recreate:
                return
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct, full_scan_threshold=10),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1, default_segment_number=2),
        )
        log.info("created qdrant collection %s (M=%d, ef_construct=%d)", self.collection, m, ef_construct)

    def upsert(
        self,
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]],
        batch_size: int = 256,
        progress: Callable[[int, int, float], None] | None = None,
    ) -> None:
        """Upsert row i of ``vectors`` as point id i with ``payloads[i]``."""
        from qdrant_client import models

        total = int(vectors.shape[0])
        t0 = time.perf_counter()
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.client.upsert(
                collection_name=self.collection,
                points=models.Batch(
                    ids=list(range(start, end)),
                    vectors=np.asarray(vectors[start:end], dtype=np.float32).tolist(),
                    payloads=list(payloads[start:end]),
                ),
                wait=True,
            )
            if progress is not None:
                progress(end, total, time.perf_counter() - t0)

    def wait_indexed(self, timeout_s: float = 300.0, poll_s: float = 1.0, settle_polls: int = 5) -> bool:
        """Block until Qdrant's optimizer has built the HNSW index (or timeout).

        Done when the collection is green and either every point is indexed, or the indexed
        count has been stable for ``settle_polls`` green polls (a small tail of points can stay
        in an appendable segment that Qdrant searches exhaustively, which is still correct).
        """
        deadline = time.monotonic() + timeout_s
        last = -1
        stable = 0
        while time.monotonic() < deadline:
            info = self.client.get_collection(self.collection)
            points = int(info.points_count or 0)
            indexed = int(info.indexed_vectors_count or 0)
            status = str(getattr(info.status, "value", info.status))
            if status == "green":
                if indexed >= points:
                    return True
                stable = stable + 1 if indexed == last else 0
                if stable >= settle_polls:
                    log.info(
                        "collection %s green with %d/%d vectors in HNSW", self.collection, indexed, points
                    )
                    return True
            last = indexed
            time.sleep(poll_s)
        log.warning("collection %s not fully indexed after %.0fs", self.collection, timeout_s)
        return False

    # ------------------------------------------------------------------ search
    def _query(self, vector: np.ndarray, k: int, ef: int | None, exact: bool) -> list[tuple[int, float]]:
        from qdrant_client import models

        res = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=k,
            search_params=models.SearchParams(hnsw_ef=ef, exact=exact),
            with_payload=False,
        )
        return [(int(p.id), float(p.score)) for p in res.points]

    def search(self, vector: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        """HNSW search with query-time ``ef``."""
        return self._query(vector, k, ef, exact=False)

    def exact_search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Qdrant's exhaustive search (ground truth for HNSW recall)."""
        return self._query(vector, k, None, exact=True)

    def health(self) -> tuple[bool, str]:
        """Collection reachable and non-empty."""
        try:
            n = self.count()
        except Exception as exc:  # noqa: BLE001
            return False, f"qdrant unreachable: {exc}"
        if n == 0:
            return False, f"collection {self.collection} missing or empty"
        return True, f"{self.collection}: {n} points"
