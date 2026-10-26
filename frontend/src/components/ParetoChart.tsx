"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import { STAGE_COLOR, fmtMs, fmtScore, fmtTick, strategyStage } from "@/lib/format";
import type { ParetoPoint } from "@/lib/types";

interface Datum extends ParetoPoint {
  color: string;
}

function TipBox({ active, payload }: TooltipProps<number, string>) {
  const d = payload?.[0]?.payload as Datum | undefined;
  if (!active || !d) return null;
  return (
    <div className="rounded border border-line bg-panel px-3 py-2 text-xs shadow-sm">
      <p className="font-medium">{d.label}</p>
      <p className="num">nDCG@10 {fmtScore(d.ndcg_at_10, 4)}</p>
      <p className="num">p95 {fmtMs(d.p95_ms)}</p>
      <p className="text-muted">{d.on_frontier ? "On the Pareto frontier" : "Dominated by a faster, better point"}</p>
    </div>
  );
}

/** nDCG@10 against p95 latency; the line joins the non-dominated configurations. */
export function ParetoChart({ points }: { points: ParetoPoint[] }) {
  const data: Datum[] = points.map((p) => ({
    ...p,
    color: STAGE_COLOR[p.source === "rerank_top_k" ? "rerank" : strategyStage(p.label)],
  }));
  const frontier = data.filter((d) => d.on_frontier).sort((a, b) => a.p95_ms - b.p95_ms);
  const useLog = data.length > 1 && Math.max(...data.map((d) => d.p95_ms)) / Math.max(Math.min(...data.map((d) => d.p95_ms)), 1e-3) > 30;
  return (
    <div className="h-[340px]" role="img" aria-label="Quality versus latency Pareto chart">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 24, bottom: 28, left: 8 }}>
          <CartesianGrid stroke="rgb(var(--line))" />
          <XAxis
            type="number"
            dataKey="p95_ms"
            scale={useLog ? "log" : "auto"}
            domain={["auto", "auto"]}
            stroke="rgb(var(--muted))"
            fontSize={12}
            tickFormatter={fmtTick}
            label={{ value: `p95 latency, ms${useLog ? " (log)" : ""}`, position: "insideBottom", offset: -14, fill: "rgb(var(--muted))", fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="ndcg_at_10"
            domain={["auto", "auto"]}
            stroke="rgb(var(--muted))"
            fontSize={12}
            tickFormatter={(v: number) => v.toFixed(3)}
            label={{ value: "nDCG@10", angle: -90, position: "insideLeft", fill: "rgb(var(--muted))", fontSize: 12 }}
          />
          <Tooltip content={<TipBox />} />
          <Scatter
            data={frontier}
            line={{ stroke: "rgb(var(--accent))", strokeDasharray: "4 3", strokeWidth: 1.5 }}
            shape={() => <g />}
            isAnimationActive={false}
            legendType="none"
          />
          <Scatter
            data={data}
            isAnimationActive={false}
            shape={(props: { cx?: number; cy?: number; payload?: Datum }) => {
              const { cx = 0, cy = 0, payload } = props;
              const r = payload?.on_frontier ? 6 : 4.5;
              return (
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={payload?.on_frontier ? payload.color : "rgb(var(--panel))"}
                  stroke={payload?.color ?? "rgb(var(--muted))"}
                  strokeWidth={2}
                />
              );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
