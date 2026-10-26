"""Redis-backed caches for query embeddings and full search results.

Keys are ``hybridir:<kind>:<sha256(payload)>``. Hit/miss counters are kept both in Redis
(hash ``hybridir:stats``, shared across API workers and visible to /metrics/cache) and in
process memory (fallback when Redis is unreachable).

Every Redis failure is swallowed and counted as a miss: the cache can make the system
faster but must never make it fail.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Protocol

log = logging.getLogger(__name__)

NAMESPACE = "hybridir"
STATS_KEY = f"{NAMESPACE}:stats"
KINDS = ("embedding", "result")


class RedisLike(Protocol):
    """Subset of the redis-py client API used here (allows an in-memory fake in tests)."""

    def get(self, name: str) -> Any: ...
    def set(self, name: str, value: Any, ex: int | None = None) -> Any: ...
    def hincrby(self, name: str, key: str, amount: int = 1) -> Any: ...
    def hgetall(self, name: str) -> Any: ...
    def delete(self, *names: str) -> Any: ...
    def ping(self) -> Any: ...
    def info(self, section: str | None = None) -> Any: ...
    def dbsize(self) -> Any: ...


def hash_key(*parts: str) -> str:
    """sha256 over the concatenation of ``parts`` (used for cache keys)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class Cache:
    """Fail-open cache with hit/miss accounting."""

    def __init__(
        self,
        url: str | None = None,
        *,
        ttl_seconds: int = 3600,
        enabled: bool = True,
        timeout_seconds: float = 1.0,
        client: RedisLike | None = None,
    ) -> None:
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._lock = threading.Lock()
        self._local: dict[str, int] = {f"{k}_{o}": 0 for k in KINDS for o in ("hits", "misses")}
        self._client: RedisLike | None = client
        if self._client is None and url and enabled:
            import redis

            self._client = redis.Redis.from_url(
                url,
                socket_timeout=timeout_seconds,
                socket_connect_timeout=timeout_seconds,
                health_check_interval=30,
            )

    # ------------------------------------------------------------------ accounting
    def _count(self, kind: str, hit: bool) -> None:
        field = f"{kind}_{'hits' if hit else 'misses'}"
        with self._lock:
            self._local[field] = self._local.get(field, 0) + 1
        if self._client is not None:
            try:
                self._client.hincrby(STATS_KEY, field, 1)
            except Exception as exc:  # noqa: BLE001 - fail open
                log.debug("redis hincrby failed: %s", exc)

    # ------------------------------------------------------------------ primitives
    def key(self, kind: str, *parts: str) -> str:
        """Build a namespaced key from the hash of ``parts``."""
        return f"{NAMESPACE}:{kind}:{hash_key(*parts)}"

    def get_bytes(self, kind: str, key: str) -> bytes | None:
        """Fetch raw bytes; ``None`` on miss, when disabled or when Redis is down."""
        if not self.enabled or self._client is None:
            return None
        try:
            value = self._client.get(key)
        except Exception as exc:  # noqa: BLE001
            log.debug("redis get failed: %s", exc)
            value = None
        self._count(kind, value is not None)
        return bytes(value) if value is not None else None

    def set_bytes(self, key: str, value: bytes) -> None:
        """Store raw bytes with the configured TTL (best effort)."""
        if not self.enabled or self._client is None:
            return
        try:
            self._client.set(key, value, ex=self.ttl)
        except Exception as exc:  # noqa: BLE001
            log.debug("redis set failed: %s", exc)

    def get_json(self, kind: str, key: str) -> Any | None:
        """Fetch and decode a JSON value."""
        raw = self.get_bytes(kind, key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def set_json(self, key: str, value: Any) -> None:
        """Encode and store a JSON value."""
        self.set_bytes(key, json.dumps(value, separators=(",", ":")).encode("utf-8"))

    # ------------------------------------------------------------------ health & stats
    def ping(self) -> tuple[bool, str]:
        """(ok, detail) for readiness checks."""
        if self._client is None:
            return (not self.enabled, "cache disabled" if not self.enabled else "no client")
        try:
            self._client.ping()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, f"redis unreachable: {exc}"

    def stats(self) -> dict[str, Any]:
        """Hit/miss counters (Redis view when available, else process-local) and memory."""
        counters = dict(self._local)
        redis_ok = False
        keys: int | None = None
        used_memory: int | None = None
        if self._client is not None:
            try:
                raw = self._client.hgetall(STATS_KEY)
                remote = {(k.decode() if isinstance(k, bytes) else str(k)): int(v) for k, v in raw.items()}
                for f in counters:
                    counters[f] = remote.get(f, 0)
                keys = int(self._client.dbsize())
                info = self._client.info("memory")
                used_memory = int(info.get("used_memory", 0)) if isinstance(info, dict) else None
                redis_ok = True
            except Exception as exc:  # noqa: BLE001
                log.debug("redis stats failed: %s", exc)

        def rate(kind: str) -> float | None:
            total = counters[f"{kind}_hits"] + counters[f"{kind}_misses"]
            return counters[f"{kind}_hits"] / total if total else None

        return {
            "enabled": self.enabled,
            "redis_ok": redis_ok,
            "embedding_hits": counters["embedding_hits"],
            "embedding_misses": counters["embedding_misses"],
            "result_hits": counters["result_hits"],
            "result_misses": counters["result_misses"],
            "embedding_hit_rate": rate("embedding"),
            "result_hit_rate": rate("result"),
            "keys": keys,
            "used_memory_bytes": used_memory,
        }

    def reset_stats(self) -> None:
        """Zero all counters (used before eval passes)."""
        with self._lock:
            for f in self._local:
                self._local[f] = 0
        if self._client is not None:
            try:
                self._client.delete(STATS_KEY)
            except Exception as exc:  # noqa: BLE001
                log.debug("redis delete failed: %s", exc)
