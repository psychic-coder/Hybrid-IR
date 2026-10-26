import { fmtScore } from "@/lib/format";
import type { FusionMath } from "@/lib/types";

function Row({ label, formula, value }: { label: string; formula: string; value: string }) {
  return (
    <tr className="border-t border-line">
      <th scope="row" className="py-1.5 pr-3 text-left font-normal text-muted">
        {label}
      </th>
      <td className="py-1.5 pr-3 text-xs text-muted">{formula}</td>
      <td className="num py-1.5 text-right">{value}</td>
    </tr>
  );
}

/** The exact numbers RRF and weighted fusion used for this document. */
export function FusionMathTable({ f }: { f: FusionMath }) {
  const bRank = f.bm25_rank === null ? "absent" : `#${f.bm25_rank}`;
  const dRank = f.dense_rank === null ? "absent" : `#${f.dense_rank}`;
  const nb = f.norm === "minmax" ? "(s − min) / (max − min)" : "(s − mean) / std";
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <table className="w-full text-sm">
        <caption className="mb-1 text-left font-medium text-fusion">Reciprocal Rank Fusion, k = {f.rrf_k}</caption>
        <tbody>
          <Row label="BM25 rank" formula={`within top ${f.candidate_depth}`} value={bRank} />
          <Row label="Dense rank" formula={`within top ${f.candidate_depth}`} value={dRank} />
          <Row label="BM25 term" formula={f.bm25_rank ? `1 / (${f.rrf_k} + ${f.bm25_rank})` : "0 (absent)"} value={fmtScore(f.rrf_bm25_term, 5)} />
          <Row label="Dense term" formula={f.dense_rank ? `1 / (${f.rrf_k} + ${f.dense_rank})` : "0 (absent)"} value={fmtScore(f.rrf_dense_term, 5)} />
          <Row label="RRF score" formula="sum of terms" value={fmtScore(f.rrf_score, 5)} />
        </tbody>
      </table>
      <table className="w-full text-sm">
        <caption className="mb-1 text-left font-medium text-fusion">
          Weighted fusion, alpha = {f.alpha}, {f.norm === "minmax" ? "min-max" : "z-score"}
        </caption>
        <tbody>
          <Row label="BM25 raw" formula={`list min ${fmtScore(f.bm25_list.min, 2)}, max ${fmtScore(f.bm25_list.max, 2)}`} value={fmtScore(f.bm25_raw)} />
          <Row label="BM25 normalized" formula={nb} value={fmtScore(f.bm25_norm)} />
          <Row label="Dense raw" formula={`list min ${fmtScore(f.dense_list.min, 2)}, max ${fmtScore(f.dense_list.max, 2)}`} value={fmtScore(f.dense_raw)} />
          <Row label="Dense normalized" formula={nb} value={fmtScore(f.dense_norm)} />
          <Row
            label="Weighted score"
            formula={`${f.alpha} × ${fmtScore(f.dense_norm)} + ${(1 - f.alpha).toFixed(2)} × ${fmtScore(f.bm25_norm)}`}
            value={fmtScore(f.weighted_score)}
          />
        </tbody>
      </table>
    </div>
  );
}
