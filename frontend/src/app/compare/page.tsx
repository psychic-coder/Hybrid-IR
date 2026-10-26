"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { EmptyState, ErrorNotice } from "@/components/ErrorBoundary";
import { ResultColumn } from "@/components/ResultColumn";
import { SearchBar } from "@/components/SearchBar";
import { ResultSkeleton } from "@/components/Skeletons";
import { ApiError, api } from "@/lib/api";
import { STRATEGY_LABELS, fmtMs } from "@/lib/format";
import { indexRows } from "@/lib/rankShift";
import type { CompareResponse, Strategy } from "@/lib/types";

const COLUMNS: Strategy[] = ["bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"];

function ComparePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initial = searchParams.get("q") ?? "";
  const [query, setQuery] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [data, setData] = useState<CompareResponse | null>(null);
  const [hover, setHover] = useState<string | null>(null);

  const run = useCallback(
    async (q: string) => {
      setQuery(q);
      setBusy(true);
      setError(null);
      router.replace(`/compare?q=${encodeURIComponent(q)}`, { scroll: false });
      try {
        setData(await api.compare({ query: q, strategies: COLUMNS, top_k: 10, params: {} }));
      } catch (e) {
        setData(null);
        setError(e);
      } finally {
        setBusy(false);
      }
    },
    [router],
  );

  useEffect(() => {
    if (initial) void run(initial);
    // run only for the URL the page was opened with
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const shiftIndex = useMemo(() => indexRows(data?.rank_shift ?? []), [data]);

  return (
    <div className="space-y-5">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Compare strategies</h1>
        <p className="max-w-prose text-sm text-muted">
          The same query through four pipelines. Badges show how far each document moved relative to its BM25 rank;
          hover a result to find it in the other columns, click its title for the score breakdown.
        </p>
      </div>
      <SearchBar initial={query} busy={busy} onSubmit={(q) => void run(q)} />
      {error !== null && <ErrorNotice error={error} onRetry={query ? () => void run(query) : undefined} />}
      {busy && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {COLUMNS.map((c) => (
            <ResultSkeleton key={c} rows={4} />
          ))}
        </div>
      )}
      {!busy && !error && !data && <EmptyState title="Enter a query to compare BM25, dense, hybrid and reranked results side by side" />}
      {!busy && data && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {data.results.map((col) => (
            <div key={col.strategy} className="min-w-0 space-y-2">
              <div className="flex items-baseline justify-between border-b border-line pb-1">
                <h2 className="text-sm font-semibold">{STRATEGY_LABELS[col.strategy]}</h2>
                {col.response && <span className="num text-xs text-muted">{fmtMs(col.response.timings.total_ms)}</span>}
              </div>
              {col.error ? (
                <ErrorNotice error={new ApiError(503, col.error)} />
              ) : col.response && col.response.hits.length > 0 ? (
                <ResultColumn
                  strategy={col.strategy}
                  hits={col.response.hits}
                  query={data.query}
                  rankShift={shiftIndex}
                  columns={COLUMNS}
                  highlighted={hover}
                  onHover={setHover}
                  compact
                />
              ) : (
                <EmptyState title="No results for this strategy" />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<ResultSkeleton />}>
      <ComparePageInner />
    </Suspense>
  );
}
