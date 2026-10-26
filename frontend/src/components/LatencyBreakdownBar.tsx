import { fmtMs } from "@/lib/format";
import type { CacheFlags, Timings } from "@/lib/types";

const SEGMENTS: { key: keyof Timings; label: string; className: string }[] = [
  { key: "tokenize_ms", label: "tokenize", className: "bg-muted/60" },
  { key: "bm25_ms", label: "BM25", className: "bg-bm25" },
  { key: "embed_ms", label: "embed", className: "bg-dense/55" },
  { key: "ann_ms", label: "ANN", className: "bg-dense" },
  { key: "fusion_ms", label: "fusion", className: "bg-fusion" },
  { key: "rerank_ms", label: "rerank", className: "bg-rerank" },
];

/** Where the request's time went, as one stacked bar (segment width = share of total). */
export function LatencyBreakdownBar({ timings, cache }: { timings: Timings; cache?: CacheFlags }) {
  const total = Math.max(timings.total_ms, 1e-6);
  const accounted = SEGMENTS.reduce((s, seg) => s + timings[seg.key], 0);
  const other = Math.max(0, timings.total_ms - accounted);
  return (
    <figure className="space-y-2">
      <div className="flex items-baseline justify-between text-sm">
        <figcaption className="font-medium">Latency {fmtMs(timings.total_ms)}</figcaption>
        {cache && (
          <span className="text-xs text-muted">
            {cache.result_cache_hit
              ? "served from result cache"
              : cache.embedding_cache_hit === true
                ? "query embedding from cache"
                : cache.embedding_cache_hit === false
                  ? "query embedding computed"
                  : "no dense stage"}
          </span>
        )}
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-sm bg-line/50" role="img" aria-label="Latency by stage">
        {SEGMENTS.map((seg) =>
          timings[seg.key] > 0 ? (
            <div
              key={seg.key}
              className={seg.className}
              style={{ width: `${(timings[seg.key] / total) * 100}%` }}
              title={`${seg.label}: ${fmtMs(timings[seg.key])}`}
            />
          ) : null,
        )}
        {other > 0 && <div className="bg-line" style={{ width: `${(other / total) * 100}%` }} title={`other: ${fmtMs(other)}`} />}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {SEGMENTS.filter((s) => timings[s.key] > 0).map((seg) => (
          <li key={seg.key} className="flex items-center gap-1.5">
            <span className={`inline-block h-2 w-2 rounded-sm ${seg.className}`} aria-hidden />
            {seg.label} <span className="num text-ink">{fmtMs(timings[seg.key])}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}
