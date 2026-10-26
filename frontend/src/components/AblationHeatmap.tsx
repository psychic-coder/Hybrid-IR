import { fmtMs, fmtScore } from "@/lib/format";
import type { Bm25GridCell } from "@/lib/types";

/** k1 x b grid of BM25 nDCG@10; cell shade scales between the grid's min and max. */
export function AblationHeatmap({ cells }: { cells: Bm25GridCell[] }) {
  const k1s = [...new Set(cells.map((c) => c.k1))].sort((a, b) => a - b);
  const bs = [...new Set(cells.map((c) => c.b))].sort((a, b) => a - b);
  const byKey = new Map(cells.map((c) => [`${c.k1}|${c.b}`, c]));
  const vals = cells.map((c) => c.ndcg_at_10);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const best = cells.reduce<Bm25GridCell | null>((a, c) => (a === null || c.ndcg_at_10 > a.ndcg_at_10 ? c : a), null);
  return (
    <div className="overflow-x-auto">
      <table className="num border-separate border-spacing-1 text-sm">
        <thead>
          <tr>
            <th className="px-2 text-right text-xs font-normal text-muted">k1 \ b</th>
            {bs.map((b) => (
              <th key={b} className="px-2 text-xs font-normal text-muted">
                {b}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {k1s.map((k1) => (
            <tr key={k1}>
              <th scope="row" className="px-2 text-right text-xs font-normal text-muted">
                {k1}
              </th>
              {bs.map((b) => {
                const c = byKey.get(`${k1}|${b}`);
                if (!c) return <td key={b} className="h-11 w-20 rounded bg-line/30" />;
                const t = hi > lo ? (c.ndcg_at_10 - lo) / (hi - lo) : 1;
                const isBest = best !== null && c === best;
                return (
                  <td
                    key={b}
                    className={`h-11 w-20 rounded text-center ${isBest ? "ring-2 ring-ink" : ""}`}
                    style={{ background: `rgb(var(--bm25) / ${0.08 + 0.72 * t})`, color: t > 0.6 ? "white" : undefined }}
                    title={`k1=${k1}, b=${b}: nDCG@10 ${fmtScore(c.ndcg_at_10, 4)}, Recall@100 ${fmtScore(c.recall_at_100, 4)}, mean ${fmtMs(c.mean_ms)}`}
                  >
                    {fmtScore(c.ndcg_at_10, 3)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {best && (
        <p className="mt-2 text-xs text-muted">
          Best cell outlined: k1 = {best.k1}, b = {best.b}. Shading spans {fmtScore(lo, 3)} to {fmtScore(hi, 3)}.
        </p>
      )}
    </div>
  );
}
