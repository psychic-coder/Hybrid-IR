/**
 * Rank-shift helpers.
 *
 * The backend computes, for every document shown in any column, its 1-based rank inside each
 * strategy's top-100 list and `delta = bm25_rank - strategy_rank`:
 *   delta > 0  the strategy moved the document UP by delta places relative to BM25
 *   delta < 0  moved DOWN
 *   is_new     the document is absent from BM25's top-100 entirely
 * These helpers turn that into badges, and can recompute it client-side from raw rankings.
 */

import type { RankShiftRow } from "./types";

export type ShiftKind = "up" | "down" | "same" | "new" | "absent";

export interface Shift {
  kind: ShiftKind;
  amount: number;
  label: string;
}

export function describeShift(delta: number | null | undefined, isNew: boolean, rank: number | null | undefined): Shift {
  if (rank === null || rank === undefined) return { kind: "absent", amount: 0, label: "not retrieved" };
  if (isNew) return { kind: "new", amount: 0, label: "new" };
  if (delta === null || delta === undefined || delta === 0) return { kind: "same", amount: 0, label: "=" };
  return delta > 0
    ? { kind: "up", amount: delta, label: `↑${delta}` }
    : { kind: "down", amount: -delta, label: `↓${-delta}` };
}

/** Recompute rank-shift rows from ordered doc-id lists (used for tests and offline views). */
export function computeRankShift(
  rankings: Record<string, string[]>,
  baseline: string,
  topK: number,
  titles: Record<string, string> = {},
): RankShiftRow[] {
  const rankOf: Record<string, Map<string, number>> = {};
  for (const [s, ids] of Object.entries(rankings)) {
    rankOf[s] = new Map(ids.map((id, i) => [id, i + 1]));
  }
  const base = rankOf[baseline] ?? new Map<string, number>();
  const seen: string[] = [];
  for (const ids of Object.values(rankings)) {
    for (const id of ids.slice(0, topK)) if (!seen.includes(id)) seen.push(id);
  }
  return seen.map((docId) => {
    const ranks: Record<string, number | null> = {};
    const delta: Record<string, number | null> = {};
    const isNew: Record<string, boolean> = {};
    const b = base.get(docId) ?? null;
    for (const s of Object.keys(rankings)) {
      const r = rankOf[s]?.get(docId) ?? null;
      ranks[s] = r;
      delta[s] = b !== null && r !== null ? b - r : null;
      isNew[s] = r !== null && b === null;
    }
    return { doc_id: docId, title: titles[docId] ?? "", ranks, delta_vs_bm25: delta, is_new: isNew };
  });
}

export function indexRows(rows: RankShiftRow[]): Map<string, RankShiftRow> {
  return new Map(rows.map((r) => [r.doc_id, r]));
}
