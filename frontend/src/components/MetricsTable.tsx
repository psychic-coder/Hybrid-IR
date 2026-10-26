import { STAGE_TEXT, fmtMs, fmtScore, fmtSigned, strategyLabel, strategyStage } from "@/lib/format";
import type { Significance, StrategyMetrics } from "@/lib/types";

function SignificanceBadge({ sig }: { sig: Significance | null }) {
  if (!sig) return <span className="text-xs text-muted">baseline</span>;
  const better = sig.mean_delta > 0;
  const cls = !sig.significant
    ? "border-line text-muted"
    : better
      ? "border-good/40 bg-good/10 text-good"
      : "border-bad/40 bg-bad/10 text-bad";
  return (
    <span
      className={`num inline-block whitespace-nowrap rounded border px-1.5 text-xs ${cls}`}
      title={`Δ nDCG@10 ${fmtSigned(sig.mean_delta)}, 95% CI [${fmtScore(sig.ci_low, 4)}, ${fmtScore(sig.ci_high, 4)}], p = ${fmtScore(sig.p_value, 4)}, ${sig.n_resamples.toLocaleString()} paired bootstrap resamples`}
    >
      {fmtSigned(sig.mean_delta, 3)} {sig.significant ? `p=${sig.p_value < 0.001 ? "<.001" : sig.p_value.toFixed(3)}` : "n.s."}
    </span>
  );
}

/** One row per evaluated configuration; best value per quality column in bold. */
export function MetricsTable({ rows }: { rows: StrategyMetrics[] }) {
  const quality = ["ndcg@10", "recall@100", "mrr@10"] as const;
  const best: Record<string, number> = {};
  for (const q of quality) best[q] = Math.max(...rows.map((r) => r.metrics[q] ?? 0));
  const th = "px-2 py-2 text-right font-normal";
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full min-w-[860px] text-sm">
        <thead className="text-xs text-muted">
          <tr className="border-b border-line">
            <th className="px-3 py-2 text-left font-normal">Configuration</th>
            <th className={th}>nDCG@10</th>
            <th className={th}>Recall@100</th>
            <th className={th}>MRR@10</th>
            <th className={th}>vs BM25</th>
            <th className={th}>p50</th>
            <th className={th}>p95</th>
            <th className={th}>p99</th>
            <th className={th} title="p95 with the query-embedding cache warm">warm p95</th>
            <th className={`${th} pr-3`}>QPS</th>
          </tr>
        </thead>
        <tbody className="num">
          {rows.map((r) => (
            <tr key={r.name} className="border-b border-line last:border-0">
              <th scope="row" className={`px-3 py-2 text-left font-medium ${STAGE_TEXT[strategyStage(r.strategy)]}`}>
                {strategyLabel(r.name)}
              </th>
              {quality.map((q) => (
                <td key={q} className={`px-2 py-2 text-right ${r.metrics[q] === best[q] ? "font-semibold" : ""}`}>
                  {fmtScore(r.metrics[q], 4)}
                </td>
              ))}
              <td className="px-2 py-2 text-right">
                <SignificanceBadge sig={r.significance} />
              </td>
              <td className="px-2 py-2 text-right">{fmtMs(r.latency_cold.p50)}</td>
              <td className="px-2 py-2 text-right">{fmtMs(r.latency_cold.p95)}</td>
              <td className="px-2 py-2 text-right">{fmtMs(r.latency_cold.p99)}</td>
              <td className="px-2 py-2 text-right">{fmtMs(r.latency_warm?.p95)}</td>
              <td className="px-2 py-2 pr-3 text-right">
                {r.qps === null ? "–" : r.qps.toFixed(1)}
                {r.qps_threads && <span className="text-xs text-muted"> /{r.qps_threads}t</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
