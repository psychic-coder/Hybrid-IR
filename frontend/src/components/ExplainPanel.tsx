"use client";

import { Fragment, type ReactNode } from "react";
import { FusionMathTable } from "@/components/FusionMathTable";
import { RankShiftBadge } from "@/components/RankShiftBadge";
import { TermContributionChart } from "@/components/TermContributionChart";
import { STRATEGY_LABELS, fmtScore } from "@/lib/format";
import type { ExplainResponse, HighlightSpan, Strategy } from "@/lib/types";

function highlight(text: string, spans: HighlightSpan[]): ReactNode[] {
  const out: ReactNode[] = [];
  let cursor = 0;
  spans.forEach((s, i) => {
    if (s.start > cursor) out.push(<Fragment key={`t${i}`}>{text.slice(cursor, s.start)}</Fragment>);
    out.push(<mark key={`m${i}`}>{text.slice(s.start, s.end)}</mark>);
    cursor = s.end;
  });
  if (cursor < text.length) out.push(<Fragment key="tail">{text.slice(cursor)}</Fragment>);
  return out;
}

function Section({ title, className, children }: { title: string; className?: string; children: ReactNode }) {
  return (
    <section className="panel p-4">
      <h2 className={`mb-3 font-semibold ${className ?? ""}`}>{title}</h2>
      {children}
    </section>
  );
}

/** Everything that went into this document's rank for this query, stage by stage. */
export function ExplainPanel({ data }: { data: ExplainResponse }) {
  const order = Object.keys(data.ranks) as Strategy[];
  return (
    <div className="space-y-5">
      <header className="space-y-2">
        <p className="text-sm text-muted">
          Why is document <span className="text-ink">{data.doc_id}</span> ranked where it is for &ldquo;{data.query}&rdquo;?
        </p>
        <h1 className="text-2xl font-semibold leading-tight tracking-tight">{highlight(data.title, data.title_highlights)}</h1>
        <ul className="flex flex-wrap gap-2 text-sm">
          {order.map((s) => (
            <li key={s} className="panel flex items-center gap-2 px-2.5 py-1">
              <span className="text-muted">{STRATEGY_LABELS[s] ?? s}</span>
              <span className="num font-medium">{data.ranks[s] ? `#${data.ranks[s]}` : "not in top 100"}</span>
              {s !== "bm25" && (
                <RankShiftBadge
                  delta={data.delta_vs_bm25[s]}
                  isNew={data.ranks[s] !== null && data.ranks.bm25 === null}
                  rank={data.ranks[s]}
                />
              )}
            </li>
          ))}
        </ul>
      </header>

      <div className="grid gap-5 lg:grid-cols-[3fr_2fr]">
        <Section title={`BM25 ${fmtScore(data.bm25_score)}`} className="text-bm25">
          {data.terms.length === 0 ? (
            <p className="text-sm text-muted">No query terms survived analysis (all stopwords?).</p>
          ) : (
            <>
              <TermContributionChart terms={data.terms} />
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr>
                      <th className="py-1 pr-2 font-normal">term</th>
                      <th className="py-1 pr-2 text-right font-normal">tf</th>
                      <th className="py-1 pr-2 text-right font-normal">df</th>
                      <th className="py-1 pr-2 text-right font-normal">idf</th>
                      <th className="py-1 pr-2 text-right font-normal" title="1 − b + b·|d|/avgdl">length norm</th>
                      <th className="py-1 pr-2 text-right font-normal" title="tf·(k1+1)/(tf + k1·norm)">tf part</th>
                      <th className="py-1 text-right font-normal">score</th>
                    </tr>
                  </thead>
                  <tbody className="num">
                    {data.terms.map((t) => (
                      <tr key={t.term} className={`border-t border-line ${t.tf === 0 ? "text-muted" : ""}`}>
                        <td className="py-1 pr-2">
                          {t.term}
                          {t.qtf > 1 && <span className="text-xs text-muted"> ×{t.qtf}</span>}
                        </td>
                        <td className="py-1 pr-2 text-right">{t.tf}</td>
                        <td className="py-1 pr-2 text-right">{t.df}</td>
                        <td className="py-1 pr-2 text-right">{fmtScore(t.idf)}</td>
                        <td className="py-1 pr-2 text-right">{fmtScore(t.length_norm)}</td>
                        <td className="py-1 pr-2 text-right">{fmtScore(t.tf_component)}</td>
                        <td className="py-1 text-right font-medium">{fmtScore(t.score)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-muted">
                k1 = {data.params.k1}, b = {data.params.b}; chunk length {data.terms[0]?.doc_len ?? "–"} terms vs average{" "}
                {fmtScore(data.terms[0]?.avg_doc_len ?? null, 1)}.
                {data.phrase_boost > 0 && <> Phrase boost +{fmtScore(data.phrase_boost)}.</>}
              </p>
            </>
          )}
        </Section>

        <div className="space-y-5">
          <Section title="Dense similarity" className="text-dense">
            {data.cosine !== null ? (
              <p className="text-sm">
                Cosine between the query embedding and the closest chunk:{" "}
                <span className="num text-lg font-semibold text-dense">{fmtScore(data.cosine, 4)}</span>
              </p>
            ) : (
              <p className="text-sm text-muted">Unavailable: {data.dense_error ?? "no embeddings loaded"}</p>
            )}
          </Section>
          <Section title="Cross-encoder" className="text-rerank">
            {data.rerank_score !== null ? (
              <p className="text-sm">
                Relevance logit for (query, document):{" "}
                <span className="num text-lg font-semibold text-rerank">{fmtScore(data.rerank_score, 3)}</span>
                <span className="mt-1 block text-xs text-muted">
                  Applied only when the document is within the top {data.params.rerank_top_k} fused candidates.
                </span>
              </p>
            ) : (
              <p className="text-sm text-muted">Unavailable: {data.rerank_error ?? "reranker not loaded"}</p>
            )}
          </Section>
        </div>
      </div>

      <Section title="Fusion arithmetic" className="text-fusion">
        {data.fusion ? (
          <FusionMathTable f={data.fusion} />
        ) : (
          <p className="text-sm text-muted">Fusion needs the dense stage, which is unavailable: {data.dense_error}</p>
        )}
      </Section>

      <Section title="Document">
        <p className="max-w-prose whitespace-pre-line text-[15px] leading-7">{highlight(data.text, data.text_highlights)}</p>
        <p className="mt-3 text-xs text-muted">
          Highlighted words match an analyzed query term ({data.query_terms.join(", ") || "none"}).
        </p>
      </Section>
    </div>
  );
}
