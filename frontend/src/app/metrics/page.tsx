import { CacheStatsCard } from "@/components/CacheStatsCard";
import { EmptyState, FailureNotice } from "@/components/ErrorBoundary";
import { MetricsTable } from "@/components/MetricsTable";
import { ParetoChart } from "@/components/ParetoChart";
import { ApiError, api } from "@/lib/api";
import { fmtBytes, fmtInt, fmtMs, fmtScore } from "@/lib/format";
import type { CacheStats, SummaryResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

/** Server component: every number on this page comes from results/*.json via /metrics/summary. */
export default async function MetricsPage() {
  let summary: SummaryResponse | null = null;
  let cache: CacheStats | null = null;
  let failure: string | null = null;
  try {
    [summary, cache] = await Promise.all([api.summary(), api.cache().catch(() => null)]);
  } catch (e) {
    failure = e instanceof ApiError ? e.info.detail : String(e);
  }

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Metrics</h1>
        {summary?.available && (
          <p className="text-sm text-muted">
            {summary.dataset}, {fmtInt(summary.n_queries)} test queries, generated {summary.generated_at}. Latency is in-process
            with caches off unless marked warm.
          </p>
        )}
      </div>
      {failure && <FailureNotice title="Could not load evaluation results" detail={failure} />}
      {summary && !summary.available && (
        <EmptyState title="No evaluation results yet">
          {summary.message} The seed service runs the baseline evaluation on first start.
        </EmptyState>
      )}
      {summary?.available && (
        <>
          <MetricsTable rows={summary.strategies} />
          <div className="grid gap-5 lg:grid-cols-[2fr_1fr]">
            <section className="panel p-4">
              <h2 className="font-semibold">Quality against latency</h2>
              <p className="mb-2 text-sm text-muted">
                Filled points are on the Pareto frontier: no other configuration is both faster and better.
              </p>
              {summary.pareto.length > 0 ? <ParetoChart points={summary.pareto} /> : <EmptyState title="No Pareto points yet" />}
            </section>
            <div className="space-y-5">
              <CacheStatsCard initial={cache} />
              {summary.index_stats && (
                <section className="panel p-4 text-sm">
                  <h2 className="mb-2 font-semibold">Inverted index</h2>
                  <dl className="num grid grid-cols-2 gap-y-1">
                    <dt className="text-muted">Documents</dt>
                    <dd className="text-right">{fmtInt(summary.index_stats.n_docs)}</dd>
                    <dt className="text-muted">Chunks</dt>
                    <dd className="text-right">{fmtInt(summary.index_stats.n_chunks)}</dd>
                    <dt className="text-muted">Vocabulary</dt>
                    <dd className="text-right">{fmtInt(summary.index_stats.vocab_size)}</dd>
                    <dt className="text-muted">Avg chunk length</dt>
                    <dd className="text-right">{fmtScore(summary.index_stats.avg_chunk_len, 1)}</dd>
                    <dt className="text-muted">Postings on disk</dt>
                    <dd className="text-right">{fmtBytes(summary.index_stats.postings_bytes)}</dd>
                    <dt className="text-muted">Compression vs raw</dt>
                    <dd className="text-right">{fmtScore(summary.index_stats.compression_ratio, 2)}×</dd>
                    <dt className="text-muted">Build time</dt>
                    <dd className="text-right">{fmtScore(summary.index_stats.build_seconds, 1)} s</dd>
                    <dt className="text-muted">Peak RSS</dt>
                    <dd className="text-right">{fmtScore(summary.index_stats.peak_rss_mb, 0)} MB</dd>
                  </dl>
                </section>
              )}
            </div>
          </div>
          {summary.load_test && (
            <section className="panel overflow-x-auto p-4">
              <h2 className="font-semibold">Load test</h2>
              <p className="mb-2 text-sm text-muted">
                k6 {summary.load_test.script}, {fmtInt(summary.load_test.total_requests)} requests over{" "}
                {fmtScore(summary.load_test.duration_s, 0)} s, error rate {fmtScore(summary.load_test.error_rate * 100, 2)}%.
              </p>
              <table className="num w-full text-sm">
                <thead className="text-xs text-muted">
                  <tr>
                    <th className="py-1 text-left font-normal">Strategy</th>
                    <th className="py-1 text-right font-normal">Requests</th>
                    <th className="py-1 text-right font-normal">p50</th>
                    <th className="py-1 text-right font-normal">p95</th>
                    <th className="py-1 text-right font-normal">p99</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.load_test.strategies.map((s) => (
                    <tr key={s.strategy} className="border-t border-line">
                      <td className="py-1">{s.strategy}</td>
                      <td className="py-1 text-right">{fmtInt(s.count)}</td>
                      <td className="py-1 text-right">{fmtMs(s.p50_ms)}</td>
                      <td className="py-1 text-right">{fmtMs(s.p95_ms)}</td>
                      <td className="py-1 text-right">{fmtMs(s.p99_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
          {Object.keys(summary.environment).length > 0 && (
            <details className="panel p-4 text-sm">
              <summary className="cursor-pointer font-medium">Environment these numbers were measured on</summary>
              <dl className="mt-2 grid gap-x-4 gap-y-1 sm:grid-cols-[180px_1fr]">
                {Object.entries(summary.environment).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-muted">{k}</dt>
                    <dd className="break-all">{v}</dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
        </>
      )}
    </div>
  );
}
