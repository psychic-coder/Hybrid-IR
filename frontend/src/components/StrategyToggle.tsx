"use client";

import { STRATEGY_LABELS, strategyStage } from "@/lib/format";
import { STRATEGIES, type Stage, type Strategy } from "@/lib/types";

const ACTIVE: Record<Stage, string> = {
  bm25: "border-bm25 bg-bm25/10 text-bm25",
  dense: "border-dense bg-dense/10 text-dense",
  fusion: "border-fusion bg-fusion/10 text-fusion",
  rerank: "border-rerank bg-rerank/10 text-rerank",
};

interface Props {
  value: Strategy;
  onChange: (s: Strategy) => void;
  options?: readonly Strategy[];
}

/** Radio group of retrieval strategies, each tinted with the color of its final stage. */
export function StrategyToggle({ value, onChange, options = STRATEGIES }: Props) {
  return (
    <div role="radiogroup" aria-label="Strategy" className="flex flex-wrap gap-1.5">
      {options.map((s) => {
        const active = s === value;
        return (
          <button
            key={s}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(s)}
            className={`rounded border px-2.5 py-1 text-sm transition-colors ${
              active ? `${ACTIVE[strategyStage(s)]} font-medium` : "border-line text-muted hover:text-ink"
            }`}
          >
            {STRATEGY_LABELS[s]}
          </button>
        );
      })}
    </div>
  );
}
