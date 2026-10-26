import { describeShift } from "@/lib/rankShift";

const STYLE = {
  up: "border-good/40 bg-good/10 text-good",
  down: "border-bad/40 bg-bad/10 text-bad",
  same: "border-line text-muted",
  new: "border-accent/40 bg-accent/10 text-accent",
  absent: "border-line text-muted",
} as const;

const TITLE = {
  up: "moved up relative to BM25",
  down: "moved down relative to BM25",
  same: "same rank as BM25",
  new: "not in BM25's top 100",
  absent: "not retrieved by this strategy",
} as const;

/** ↑n / ↓n / new / = relative to the BM25 rank. */
export function RankShiftBadge({ delta, isNew, rank }: { delta: number | null | undefined; isNew: boolean; rank: number | null | undefined }) {
  const s = describeShift(delta, isNew, rank);
  return (
    <span
      className={`num inline-flex min-w-[2.25rem] justify-center rounded border px-1.5 text-xs font-medium ${STYLE[s.kind]}`}
      title={TITLE[s.kind]}
    >
      {s.label}
    </span>
  );
}
