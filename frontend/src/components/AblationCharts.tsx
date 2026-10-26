"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { STAGE_COLOR, fmtScore, fmtTick } from "@/lib/format";
import type { AlphaRow, HnswRow, RerankRow, RrfRow } from "@/lib/types";

const AXIS = { stroke: "rgb(var(--muted))", fontSize: 12 };
const TIP = { contentStyle: { background: "rgb(var(--panel))", border: "1px solid rgb(var(--line))", fontSize: 12 } };
const M_COLORS = ["rgb(var(--dense))", "rgb(var(--accent))", "rgb(var(--fusion))"];

/** nDCG@10 as alpha moves from pure BM25 (0) to pure dense (1), per normalization; best RRF as reference. */
export function AlphaSweepChart({ rows, rrf }: { rows: AlphaRow[]; rrf: RrfRow[] }) {
  const alphas = [...new Set(rows.map((r) => r.alpha))].sort((a, b) => a - b);
  const data = alphas.map((a) => ({
    alpha: a,
    minmax: rows.find((r) => r.alpha === a && r.norm === "minmax")?.ndcg_at_10 ?? null,
    zscore: rows.find((r) => r.alpha === a && r.norm === "zscore")?.ndcg_at_10 ?? null,
  }));
  const bestRrf = rrf.reduce<RrfRow | null>((b, r) => (b === null || r.ndcg_at_10 > b.ndcg_at_10 ? r : b), null);
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 20, bottom: 20, left: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" />
          <XAxis dataKey="alpha" {...AXIS} label={{ value: "alpha (dense weight)", position: "insideBottom", offset: -10, fill: "rgb(var(--muted))", fontSize: 12 }} />
          <YAxis {...AXIS} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(3)} />
          <Tooltip {...TIP} formatter={(v: number) => fmtScore(v, 4)} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />
          <Line dataKey="minmax" name="min-max" stroke={STAGE_COLOR.fusion} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
          <Line dataKey="zscore" name="z-score" stroke={STAGE_COLOR.dense} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
          {bestRrf && (
            <ReferenceLine
              y={bestRrf.ndcg_at_10}
              stroke="rgb(var(--muted))"
              strokeDasharray="4 3"
              label={{ value: `best RRF (k=${bestRrf.rrf_k})`, position: "insideBottomRight", fill: "rgb(var(--muted))", fontSize: 11 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Quality (left axis) and latency (right axis) as rerank depth grows. */
export function RerankDepthChart({ rows }: { rows: RerankRow[] }) {
  const data = [...rows].sort((a, b) => a.rerank_top_k - b.rerank_top_k);
  return (
    <div className="h-[280px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 20, left: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" />
          <XAxis dataKey="rerank_top_k" {...AXIS} label={{ value: "candidates reranked", position: "insideBottom", offset: -10, fill: "rgb(var(--muted))", fontSize: 12 }} />
          <YAxis yAxisId="q" {...AXIS} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(3)} />
          <YAxis yAxisId="t" orientation="right" {...AXIS} tickFormatter={fmtTick} />
          <Tooltip {...TIP} formatter={(v: number, name: string) => (name.includes("ms") ? `${v.toFixed(1)} ms` : fmtScore(v, 4))} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />
          <Line yAxisId="q" dataKey="ndcg_at_10" name="nDCG@10" stroke={STAGE_COLOR.rerank} strokeWidth={2} isAnimationActive={false} />
          <Line yAxisId="t" dataKey="p95_ms" name="rerank p95 ms" stroke="rgb(var(--muted))" strokeDasharray="4 3" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** HNSW: recall@10 vs exact search, or ANN p95 latency, against ef, one line per M. */
export function HnswChart({ rows, metric }: { rows: HnswRow[]; metric: "recall" | "latency" }) {
  const ms = [...new Set(rows.map((r) => r.m))].sort((a, b) => a - b);
  const efs = [...new Set(rows.map((r) => r.ef))].sort((a, b) => a - b);
  const data = efs.map((ef) => {
    const point: Record<string, number | null> = { ef };
    for (const m of ms) {
      const r = rows.find((x) => x.m === m && x.ef === ef);
      point[`M=${m}`] = r ? (metric === "recall" ? r.recall_at_10_vs_exact : r.p95_ms) : null;
    }
    return point;
  });
  return (
    <div className="h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 20, left: 4 }}>
          <CartesianGrid stroke="rgb(var(--line))" />
          <XAxis dataKey="ef" scale="log" domain={["auto", "auto"]} type="number" ticks={efs} {...AXIS}
            label={{ value: "ef (log scale)", position: "insideBottom", offset: -10, fill: "rgb(var(--muted))", fontSize: 12 }} />
          <YAxis {...AXIS} domain={metric === "recall" ? [0, 1] : ["auto", "auto"]} tickFormatter={(v: number) => (metric === "recall" ? v.toFixed(2) : fmtTick(v))} />
          <Tooltip {...TIP} formatter={(v: number) => (metric === "recall" ? fmtScore(v, 4) : `${v.toFixed(2)} ms`)} />
          <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 12 }} />
          {ms.map((m, i) => (
            <Line key={m} dataKey={`M=${m}`} stroke={M_COLORS[i % M_COLORS.length]} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
