"""From-scratch Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016/2018).

NumPy is used only for vector math; graph construction and search are plain Python.
Distances are cosine distances ``1 - <u, v>`` over L2-normalized vectors.

Parameters
----------
M               max neighbors per node on layers >= 1 (layer 0 allows ``2*M``)
ef_construction size of the dynamic candidate list while inserting
ef_search       size of the dynamic candidate list at query time (>= k)
mL = 1/ln(M)    level multiplier: level = floor(-ln(U(0,1)) * mL)

Neighbor selection uses the paper's heuristic (Algorithm 4) with ``keepPrunedConnections``:
a candidate is kept only if it is closer to the new element than to any already-selected
neighbor, which preserves long-range links between clusters.
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class CustomHNSW:
    """In-memory HNSW index with ``add_items``, ``search``, ``save`` and ``load``."""

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        seed: int = 42,
        capacity: int = 1024,
    ) -> None:
        if m < 2:
            raise ValueError("M must be >= 2")
        self.dim = dim
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(m)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._vectors = np.zeros((max(capacity, 1), dim), dtype=np.float32)
        self.count = 0
        self.levels: list[int] = []
        self.neighbors: list[list[list[int]]] = []  # neighbors[node][level]
        self.entry_point = -1
        self.max_level = -1

    # ------------------------------------------------------------------ helpers
    @property
    def vectors(self) -> np.ndarray:
        """The stored (normalized) vectors, shape (count, dim)."""
        return self._vectors[: self.count]

    def _ensure_capacity(self, extra: int) -> None:
        need = self.count + extra
        if need <= self._vectors.shape[0]:
            return
        new_cap = max(need, 2 * self._vectors.shape[0])
        grown = np.zeros((new_cap, self.dim), dtype=np.float32)
        grown[: self.count] = self._vectors[: self.count]
        self._vectors = grown

    def _random_level(self) -> int:
        u = float(self._rng.random())
        return int(-math.log(max(u, 1e-12)) * self.ml)

    def _dist_many(self, q: np.ndarray, nodes: list[int]) -> list[float]:
        dists: list[float] = (1.0 - self._vectors[nodes] @ q).tolist()
        return dists

    def _search_layer(
        self, q: np.ndarray, entry_points: list[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        """Best-first search on one layer; returns up to ``ef`` (distance, node) ascending."""
        visited = set(entry_points)
        d0 = self._dist_many(q, entry_points)
        candidates = list(zip(d0, entry_points, strict=True))
        heapq.heapify(candidates)
        results = [(-d, n) for d, n in candidates]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        neighbors = self.neighbors
        while candidates:
            d, c = heapq.heappop(candidates)
            if d > -results[0][0] and len(results) >= ef:
                break
            fresh = [n for n in neighbors[c][level] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for dn, n in zip(self._dist_many(q, fresh), fresh, strict=True):
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heappush(results, (-dn, n))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-nd, n) for nd, n in results)

    def _select_heuristic(self, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Algorithm 4 (simple variant without extendCandidates) + keepPrunedConnections."""
        selected: list[int] = []
        pruned: list[int] = []
        for d, c in candidates:
            if len(selected) >= m:
                break
            if not selected:
                selected.append(c)
                continue
            dist_to_selected = 1.0 - self._vectors[selected] @ self._vectors[c]
            if d < float(dist_to_selected.min()):
                selected.append(c)
            else:
                pruned.append(c)
        for c in pruned:
            if len(selected) >= m:
                break
            selected.append(c)
        return selected

    # ------------------------------------------------------------------ build
    def add_items(self, data: np.ndarray) -> None:
        """Insert a batch of vectors; ids are assigned sequentially from ``count``."""
        data = _normalize(np.atleast_2d(np.asarray(data, dtype=np.float32)))
        if data.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {data.shape[1]}")
        self._ensure_capacity(data.shape[0])
        for row in data:
            self._insert(row)

    def _insert(self, q: np.ndarray) -> None:
        idx = self.count
        self._vectors[idx] = q
        self.count += 1
        level = self._random_level()
        self.levels.append(level)
        self.neighbors.append([[] for _ in range(level + 1)])
        if self.entry_point < 0:
            self.entry_point = idx
            self.max_level = level
            return
        ep = [self.entry_point]
        for lc in range(self.max_level, level, -1):
            ep = [self._search_layer(q, ep, 1, lc)[0][1]]
        for lc in range(min(level, self.max_level), -1, -1):
            w = self._search_layer(q, ep, self.ef_construction, lc)
            m_max = self.m0 if lc == 0 else self.m
            chosen = self._select_heuristic(w, self.m)
            self.neighbors[idx][lc] = chosen
            for e in chosen:
                lst = self.neighbors[e][lc]
                lst.append(idx)
                if len(lst) > m_max:
                    ev = self._vectors[e]
                    dists = self._dist_many(ev, lst)
                    ordered = sorted(zip(dists, lst, strict=True))
                    self.neighbors[e][lc] = self._select_heuristic(ordered, m_max)
            ep = [n for _, n in w]
        if level > self.max_level:
            self.max_level = level
            self.entry_point = idx

    # ------------------------------------------------------------------ query
    def search(self, query: np.ndarray, k: int, ef: int | None = None) -> list[tuple[int, float]]:
        """Approximate top-k; returns (id, cosine similarity) descending."""
        if self.count == 0 or k <= 0:
            return []
        q = _normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        ef_eff = max(ef if ef is not None else self.ef_search, k)
        ep = [self.entry_point]
        for lc in range(self.max_level, 0, -1):
            ep = [self._search_layer(q, ep, 1, lc)[0][1]]
        w = self._search_layer(q, ep, ef_eff, 0)
        return [(n, 1.0 - d) for d, n in w[:k]]

    def brute_force(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Exact top-k by full scan (ground truth for recall measurements)."""
        q = _normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        sims = self.vectors @ q
        k = min(k, self.count)
        top = np.argpartition(-sims, k - 1)[:k]
        order = sorted(top.tolist(), key=lambda i: (-float(sims[i]), i))
        return [(i, float(sims[i])) for i in order]

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Serialize vectors, levels and adjacency lists to a single ``.npz`` file."""
        flat: list[int] = []
        offsets: list[int] = [0]
        for node in range(self.count):
            for lvl in range(self.levels[node] + 1):
                flat.extend(self.neighbors[node][lvl])
                offsets.append(len(flat))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            np.savez(
                fh,
                vectors=self.vectors,
                levels=np.asarray(self.levels, dtype=np.int32),
                adj=np.asarray(flat, dtype=np.int32),
                offsets=np.asarray(offsets, dtype=np.int64),
                params=np.asarray(
                    [
                        self.dim,
                        self.m,
                        self.ef_construction,
                        self.ef_search,
                        self.seed,
                        self.entry_point,
                        self.max_level,
                    ],
                    dtype=np.int64,
                ),
            )

    @classmethod
    def load(cls, path: Path) -> CustomHNSW:
        """Inverse of :meth:`save`."""
        with np.load(Path(path)) as z:
            dim, m, efc, efs, seed, entry, max_level = (int(x) for x in z["params"])
            vectors = z["vectors"]
            levels = z["levels"].tolist()
            adj = z["adj"].tolist()
            offsets = z["offsets"].tolist()
        index = cls(dim=dim, m=m, ef_construction=efc, ef_search=efs, seed=seed, capacity=max(len(levels), 1))
        index._vectors[: len(levels)] = vectors
        index.count = len(levels)
        index.levels = levels
        cursor = 0
        for node in range(index.count):
            per_level: list[list[int]] = []
            for _ in range(levels[node] + 1):
                per_level.append(adj[offsets[cursor] : offsets[cursor + 1]])
                cursor += 1
            index.neighbors.append(per_level)
        index.entry_point = entry
        index.max_level = max_level
        return index
