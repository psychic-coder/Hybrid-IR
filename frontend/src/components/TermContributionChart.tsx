"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtScore } from "@/lib/format";
import type { TermContribution } from "@/lib/types";

/** Horizontal bars: how much each query term contributed to the document's BM25 score. */
export function TermContributionChart({ terms }: { terms: TermContribution[] }) {
  const data = [...terms].sort((a, b) => b.score - a.score);
  const height = Math.max(120, data.length * 34 + 40);
  return (
    <div style={{ height }} role="img" aria-label="BM25 contribution per query term">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
          <CartesianGrid horizontal={false} stroke="rgb(var(--line))" />
          <XAxis type="number" stroke="rgb(var(--muted))" fontSize={12} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis type="category" dataKey="term" width={96} stroke="rgb(var(--muted))" fontSize={12} />
          <Tooltip
            cursor={{ fill: "rgb(var(--line) / 0.4)" }}
            contentStyle={{ background: "rgb(var(--panel))", border: "1px solid rgb(var(--line))", fontSize: 12 }}
            formatter={(v: number) => [fmtScore(v), "BM25 contribution"]}
          />
          <Bar dataKey="score" fill="rgb(var(--bm25))" radius={[0, 3, 3, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
