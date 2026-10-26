"use client";

import Link from "next/link";
import { RankShiftBadge } from "@/components/RankShiftBadge";
import { STAGE_TEXT, STRATEGY_LABELS, fmtScore } from "@/lib/format";
import type { RankShiftRow, SearchHit, Stage, Strategy } from "@/lib/types";

const STAGES: Stage[] = ["bm25", "dense", "fusion", "rerank"];

interface Props {
  strategy: Strategy;
  hits: SearchHit[];
  query: string;
  /** Compare page: rank-shift rows keyed by doc id, and the strategies shown in other columns. */
  rankShift?: Map<string, RankShiftRow>;
  columns?: Strategy[];
  highlighted?: string | null;
  onHover?: (docId: string | null) => void;
  compact?: boolean;
  heading?: boolean;
}

export function explainHref(docId: string, query: string): string {
  return `/explain/${encodeURIComponent(docId)}?q=${encodeURIComponent(query)}`;
}

/** A ranked list of hits with per-stage scores; on the Compare page also rank shifts. */
export function ResultColumn({
  strategy,
  hits,
  query,
  rankShift,
  columns,
  highlighted,
  onHover,
  compact = false,
  heading = false,
}: Props) {
  return (
    <section aria-label={`${STRATEGY_LABELS[strategy]} results`} className="min-w-0">
      {heading && <h2 className="mb-2 text-sm font-semibold">{STRATEGY_LABELS[strategy]}</h2>}
      <ol className="space-y-2">
        {hits.map((h) => {
          const row = rankShift?.get(h.doc_id);
          const active = highlighted === h.doc_id;
          return (
            <li
              key={h.doc_id}
              onMouseEnter={() => onHover?.(h.doc_id)}
              onMouseLeave={() => onHover?.(null)}
              onFocus={() => onHover?.(h.doc_id)}
              onBlur={() => onHover?.(null)}
              className={`panel p-3 transition-colors ${active ? "border-accent ring-1 ring-accent" : ""}`}
            >
              <div className="flex items-start gap-2">
                <span className="num w-6 shrink-0 text-right text-sm font-semibold text-muted">{h.rank}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-start justify-between gap-2">
                    <Link href={explainHref(h.doc_id, query)} className="font-medium leading-snug hover:text-accent hover:underline">
                      {h.title || h.doc_id}
                    </Link>
                    {row && strategy !== "bm25" && (
                      <RankShiftBadge delta={row.delta_vs_bm25[strategy]} isNew={row.is_new[strategy] ?? false} rank={row.ranks[strategy]} />
                    )}
                  </div>
                  {!compact && <p className="mt-1 text-sm text-muted">{h.snippet}</p>}
                  <dl className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
                    {STAGES.filter((s) => h.scores[s] !== null).map((s) => (
                      <div key={s} className="flex gap-1">
                        <dt className={STAGE_TEXT[s]}>{s}</dt>
                        <dd className="num">{fmtScore(h.scores[s])}</dd>
                      </div>
                    ))}
                    <div className="flex gap-1 text-muted">
                      <dt>doc</dt>
                      <dd>{h.doc_id}</dd>
                    </div>
                  </dl>
                  {row && columns && (
                    <p className="mt-1.5 flex flex-wrap gap-x-3 text-xs text-muted">
                      {columns.map((c) => (
                        <span key={c} className={c === strategy ? "font-medium text-ink" : ""}>
                          {STRATEGY_LABELS[c]} <span className="num">{row.ranks[c] ? `#${row.ranks[c]}` : "–"}</span>
                        </span>
                      ))}
                    </p>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
