"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { fmtBytes, fmtInt, fmtPct } from "@/lib/format";
import type { CacheStats } from "@/lib/types";

function Meter({ label, hits, misses, rate }: { label: string; hits: number; misses: number; rate: number | null }) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-sm">
        <span>{label}</span>
        <span className="num font-semibold">{fmtPct(rate)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-sm bg-line/60">
        <div className="h-full bg-accent" style={{ width: `${(rate ?? 0) * 100}%` }} />
      </div>
      <p className="num text-xs text-muted">
        {fmtInt(hits)} hits, {fmtInt(misses)} misses
      </p>
    </div>
  );
}

/** Live Redis hit rates for the query-embedding and result caches. */
export function CacheStatsCard({ initial }: { initial: CacheStats | null }) {
  const [stats, setStats] = useState<CacheStats | null>(initial);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  async function refresh() {
    setBusy(true);
    try {
      setStats(await api.cache());
      setFailed(false);
    } catch {
      setFailed(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">Cache</h2>
        <button type="button" onClick={() => void refresh()} className="btn-quiet" disabled={busy}>
          {busy ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      {failed && <p className="text-sm text-bad">Could not reach the API to refresh cache counters.</p>}
      {!stats ? (
        <p className="text-sm text-muted">Cache counters are unavailable.</p>
      ) : !stats.enabled ? (
        <p className="text-sm text-muted">Caching is disabled (CACHE_ENABLED=false).</p>
      ) : (
        <>
          {!stats.redis_ok && <p className="text-sm text-bad">Redis is unreachable; requests run uncached.</p>}
          <Meter label="Query embeddings" hits={stats.embedding_hits} misses={stats.embedding_misses} rate={stats.embedding_hit_rate} />
          <Meter label="Search results" hits={stats.result_hits} misses={stats.result_misses} rate={stats.result_hit_rate} />
          <p className="num text-xs text-muted">
            {fmtInt(stats.keys)} keys, {fmtBytes(stats.used_memory_bytes)} used. Counters reset after each evaluation run.
          </p>
        </>
      )}
    </section>
  );
}
