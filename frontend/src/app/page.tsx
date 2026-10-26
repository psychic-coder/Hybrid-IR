"use client";

import { useCallback, useState } from "react";
import { EmptyState, ErrorNotice } from "@/components/ErrorBoundary";
import { LatencyBreakdownBar } from "@/components/LatencyBreakdownBar";
import { ParamDrawer } from "@/components/ParamDrawer";
import { ResultColumn } from "@/components/ResultColumn";
import { SearchBar } from "@/components/SearchBar";
import { ResultSkeleton } from "@/components/Skeletons";
import { StrategyToggle } from "@/components/StrategyToggle";
import { api } from "@/lib/api";
import { STRATEGY_LABELS, fmtScore } from "@/lib/format";
import type { SearchParams, SearchResponse, Strategy } from "@/lib/types";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [strategy, setStrategy] = useState<Strategy>("hybrid_rrf_rerank");
  const [params, setParams] = useState<SearchParams>({});
  const [drawer, setDrawer] = useState(false);
  const [topK, setTopK] = useState(10);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);

  const run = useCallback(
    async (q: string, s: Strategy = strategy) => {
      setQuery(q);
      setBusy(true);
      setError(null);
      try {
        setResult(await api.search({ query: q, strategy: s, top_k: topK, params }));
      } catch (e) {
        setResult(null);
        setError(e);
      } finally {
        setBusy(false);
      }
    },
    [strategy, topK, params],
  );

  return (
    <div className="space-y-5">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Search</h1>
        <p className="max-w-prose text-sm text-muted">
          Run one strategy and see each stage&rsquo;s score for every hit and where the time went. Quote two or
          more words to add a phrase boost.
        </p>
      </div>
      <SearchBar initial={query} busy={busy} onSubmit={(q) => void run(q)} />
      <div className="flex flex-wrap items-center gap-3">
        <StrategyToggle
          value={strategy}
          onChange={(s) => {
            setStrategy(s);
            if (query) void run(query, s);
          }}
        />
        <label className="ml-auto flex items-center gap-2 text-sm text-muted">
          Results
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="field">
            {[5, 10, 20, 50].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ParamDrawer open={drawer} onToggle={() => setDrawer((d) => !d)} params={params} onChange={setParams} />

      {error !== null && <ErrorNotice error={error} onRetry={query ? () => void run(query) : undefined} />}
      {busy && <ResultSkeleton />}
      {!busy && !error && !result && (
        <EmptyState title="Type a query to search the corpus">
          Try a claim from the dataset, for example &ldquo;vitamin D supplementation reduces fracture risk&rdquo;.
          Press <kbd className="rounded border border-line px-1">/</kbd> to jump to the search box.
        </EmptyState>
      )}
      {!busy && result && (
        <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
          <div className="space-y-3">
            <p className="text-sm text-muted">
              {STRATEGY_LABELS[result.strategy]} ranked {result.total_candidates} documents; showing{" "}
              {result.hits.length}.
              {result.query_terms.length > 0 && <> Analyzed terms: {result.query_terms.join(", ")}.</>}
            </p>
            {result.hits.length === 0 ? (
              <EmptyState title="No documents matched">
                None of the analyzed query terms occur in the index. Try different wording or the Dense strategy.
              </EmptyState>
            ) : (
              <ResultColumn strategy={result.strategy} hits={result.hits} query={result.query} />
            )}
          </div>
          <aside className="space-y-4 lg:sticky lg:top-4 lg:self-start">
            <div className="panel p-4">
              <LatencyBreakdownBar timings={result.timings} cache={result.cache} />
            </div>
            <div className="panel p-4 text-sm">
              <h2 className="mb-2 font-medium">Parameters used</h2>
              <dl className="grid grid-cols-2 gap-y-1">
                {Object.entries(result.params).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-muted">{k}</dt>
                    <dd className="num text-right">{typeof v === "number" ? fmtScore(v, Number.isInteger(v) ? 0 : 2) : v}</dd>
                  </div>
                ))}
                <dt className="text-muted">dense backend</dt>
                <dd className="text-right">{result.dense_backend}</dd>
              </dl>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
