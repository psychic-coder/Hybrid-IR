import type { ReactNode } from "react";
import { AlphaSweepChart, HnswChart, RerankDepthChart } from "@/components/AblationCharts";
import { AblationHeatmap } from "@/components/AblationHeatmap";
import { EmptyState, FailureNotice } from "@/components/ErrorBoundary";
import { ApiError, api } from "@/lib/api";
import { fmtBytes, fmtInt, fmtScore } from "@/lib/format";
import type { AblationsResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

function Block({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <section className="panel p-4">
      <h2 className="font-semibold">{title}</h2>
      <p className="mb-3 max-w-prose text-sm text-muted">{note}</p>
      {children}
    </section>
  );
}

/** Server component rendering results/ablations.json. */
export default async function AblationsPage() {
  let data: AblationsResponse | null = null;
  let failure: string | null = null;
  try {
    data = await api.ablations();
  } catch (e) {
    failure = e instanceof ApiError ? e.info.detail : String(e);
  }
  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Ablations</h1>
        {data?.available && (
          <p className="text-sm text-muted">
            {data.dataset}, {fmtInt(data.n_queries)} queries, generated {data.generated_at}. One knob changes at a time; the
            others stay at their defaults.
          </p>
        )}
      </div>
      {failure && <FailureNotice title="Could not load ablation results" detail={failure} />}
      {data && !data.available && (
        <EmptyState title="No ablation results yet">
          {data.message} With ABLATIONS_ON_SEED=true the seed service produces them on first start.
        </EmptyState>
      )}
      {data?.available && (
        <>
          <div className="grid gap-5 lg:grid-cols-2">
            {data.bm25_grid.length > 0 && (
              <Block title="BM25 k1 and b" note="nDCG@10 for every (k1, b). k1 sets how fast repeated terms saturate; b how strongly long chunks are penalized.">
                <AblationHeatmap cells={data.bm25_grid} />
              </Block>
            )}
            {data.alpha_sweep.length > 0 && (
              <Block title="Weighted fusion alpha" note="alpha = 0 is BM25 alone, alpha = 1 is dense alone. The dashed line is the best RRF setting for reference.">
                <AlphaSweepChart rows={data.alpha_sweep} rrf={data.rrf_k} />
              </Block>
            )}
            {data.rerank_top_k.length > 0 && (
              <Block title="Rerank depth" note="Quality of reranking the top-k fused candidates against the cross-encoder's p95 cost.">
                <RerankDepthChart rows={data.rerank_top_k} />
              </Block>
            )}
            {data.rrf_k.length > 0 && (
              <Block title="RRF k" note="Small k rewards the top of each list sharply; large k blends ranks more evenly.">
                <table className="num w-full text-sm">
                  <thead className="text-xs text-muted">
                    <tr>
                      <th className="py-1 text-left font-normal">k</th>
                      <th className="py-1 text-right font-normal">nDCG@10</th>
                      <th className="py-1 text-right font-normal">Recall@100</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rrf_k.map((r) => (
                      <tr key={r.rrf_k} className="border-t border-line">
                        <td className="py-1">{r.rrf_k}</td>
                        <td className="py-1 text-right">{fmtScore(r.ndcg_at_10, 4)}</td>
                        <td className="py-1 text-right">{fmtScore(r.recall_at_100, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Block>
            )}
          </div>
          {data.hnsw.length > 0 && (
            <Block title="HNSW M and ef" note="Recall@10 against an exact scan of the same embeddings, and ANN-call latency, as the search breadth ef grows.">
              <div className="grid gap-5 lg:grid-cols-2">
                <div>
                  <h3 className="mb-1 text-sm font-medium">Recall@10 vs exact</h3>
                  <HnswChart rows={data.hnsw} metric="recall" />
                </div>
                <div>
                  <h3 className="mb-1 text-sm font-medium">ANN p95 latency, ms</h3>
                  <HnswChart rows={data.hnsw} metric="latency" />
                </div>
              </div>
            </Block>
          )}
          {data.chunk_size.length > 0 && (
            <Block title="Chunk size (BM25)" note="Documents split into windows of CHUNK_SIZE words, scored per chunk and max-aggregated per document.">
              <div className="overflow-x-auto">
                <table className="num w-full min-w-[560px] text-sm">
                  <thead className="text-xs text-muted">
                    <tr>
                      <th className="py-1 text-left font-normal">CHUNK_SIZE</th>
                      <th className="py-1 text-right font-normal">nDCG@10</th>
                      <th className="py-1 text-right font-normal">Recall@100</th>
                      <th className="py-1 text-right font-normal">Chunks</th>
                      <th className="py-1 text-right font-normal">Index size</th>
                      <th className="py-1 text-right font-normal">Build</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.chunk_size.map((r) => (
                      <tr key={r.chunk_size ?? "none"} className="border-t border-line">
                        <td className="py-1">{r.chunk_size ?? "no chunking"}</td>
                        <td className="py-1 text-right">{fmtScore(r.ndcg_at_10, 4)}</td>
                        <td className="py-1 text-right">{fmtScore(r.recall_at_100, 4)}</td>
                        <td className="py-1 text-right">{fmtInt(r.n_chunks)}</td>
                        <td className="py-1 text-right">{fmtBytes(r.index_bytes)}</td>
                        <td className="py-1 text-right">{fmtScore(r.build_seconds, 1)} s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Block>
          )}
          {data.notes.length > 0 && (
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted">
              {data.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
