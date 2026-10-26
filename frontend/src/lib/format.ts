import type { Stage, Strategy } from "./types";

export const STRATEGY_LABELS: Record<Strategy, string> = {
  bm25: "BM25",
  dense: "Dense",
  hybrid_rrf: "Hybrid (RRF)",
  hybrid_weighted: "Hybrid (weighted)",
  hybrid_rrf_rerank: "Hybrid RRF + rerank",
  hybrid_weighted_rerank: "Hybrid weighted + rerank",
};

export const STAGE_LABELS: Record<Stage, string> = {
  bm25: "BM25",
  dense: "Dense",
  fusion: "Fusion",
  rerank: "Rerank",
};

/** CSS color for a stage, read from the theme variables (works in SVG fill attributes). */
export const STAGE_COLOR: Record<Stage, string> = {
  bm25: "rgb(var(--bm25))",
  dense: "rgb(var(--dense))",
  fusion: "rgb(var(--fusion))",
  rerank: "rgb(var(--rerank))",
};

/** Text class per stage (Tailwind needs literal class names). */
export const STAGE_TEXT: Record<Stage, string> = {
  bm25: "text-bm25",
  dense: "text-dense",
  fusion: "text-fusion",
  rerank: "text-rerank",
};

/** The dominant stage of a strategy, used to color it in charts and tables. */
export function strategyStage(strategy: string): Stage {
  if (strategy === "bm25") return "bm25";
  if (strategy === "dense") return "dense";
  if (strategy.endsWith("_rerank")) return "rerank";
  return "fusion";
}

export function strategyLabel(name: string): string {
  return (STRATEGY_LABELS as Record<string, string>)[name] ?? name;
}

export function fmtScore(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return x.toFixed(digits);
}

export function fmtMs(x: number | null | undefined): string {
  if (x === null || x === undefined) return "–";
  if (x < 1) return `${x.toFixed(2)} ms`;
  if (x < 100) return `${x.toFixed(1)} ms`;
  return `${Math.round(x).toLocaleString()} ms`;
}

export function fmtInt(x: number | null | undefined): string {
  return x === null || x === undefined ? "–" : Math.round(x).toLocaleString();
}

export function fmtPct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`;
}

export function fmtBytes(x: number | null | undefined): string {
  if (x === null || x === undefined) return "–";
  const units = ["B", "KB", "MB", "GB"];
  let v = x;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u += 1;
  }
  return `${v.toFixed(u === 0 ? 0 : 1)} ${units[u]}`;
}

export function fmtSigned(x: number | null | undefined, digits = 4): string {
  if (x === null || x === undefined) return "–";
  return `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
}

/** Axis tick label with just enough precision for the magnitude. */
export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1) return v.toFixed(2);
  if (a < 10) return v.toFixed(1);
  return Math.round(v).toString();
}
