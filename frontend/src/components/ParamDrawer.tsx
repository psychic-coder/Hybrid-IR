"use client";

import type { Norm, SearchParams } from "@/lib/types";

interface Props {
  open: boolean;
  onToggle: () => void;
  params: SearchParams;
  onChange: (p: SearchParams) => void;
}

interface NumField {
  key: "k1" | "b" | "rrf_k" | "alpha" | "rerank_top_k" | "ef";
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
}

const FIELDS: NumField[] = [
  { key: "k1", label: "k1", hint: "BM25 term-frequency saturation", min: 0, max: 5, step: 0.1 },
  { key: "b", label: "b", hint: "BM25 length normalization (0 = off)", min: 0, max: 1, step: 0.05 },
  { key: "rrf_k", label: "RRF k", hint: "Larger k flattens rank differences", min: 1, max: 1000, step: 1 },
  { key: "alpha", label: "alpha", hint: "Dense weight in weighted fusion", min: 0, max: 1, step: 0.05 },
  { key: "rerank_top_k", label: "Rerank top-k", hint: "Candidates scored by the cross-encoder", min: 1, max: 200, step: 1 },
  { key: "ef", label: "ef", hint: "HNSW search breadth (recall vs latency)", min: 1, max: 1024, step: 1 },
];

/** Collapsible per-request parameter overrides. Empty fields use the server defaults. */
export function ParamDrawer({ open, onToggle, params, onChange }: Props) {
  function set(key: NumField["key"], raw: string) {
    const v = raw === "" ? null : Number(raw);
    onChange({ ...params, [key]: v === null || Number.isNaN(v) ? null : v });
  }

  return (
    <div className="panel">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls="param-drawer"
        className="flex w-full items-center justify-between px-3 py-2 text-sm"
      >
        <span className="font-medium">Parameters</span>
        <span className="text-muted">{open ? "Hide" : "Override defaults"}</span>
      </button>
      {open && (
        <div id="param-drawer" className="grid gap-3 border-t border-line p-3 sm:grid-cols-2 lg:grid-cols-4">
          {FIELDS.map((f) => (
            <label key={f.key} className="flex flex-col gap-1 text-sm">
              <span className="font-medium">{f.label}</span>
              <input
                type="number"
                inputMode="decimal"
                min={f.min}
                max={f.max}
                step={f.step}
                value={params[f.key] ?? ""}
                placeholder="default"
                onChange={(e) => set(f.key, e.target.value)}
                className="field num"
              />
              <span className="text-xs text-muted">{f.hint}</span>
            </label>
          ))}
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-medium">Normalization</span>
            <select
              value={params.norm ?? ""}
              onChange={(e) => onChange({ ...params, norm: e.target.value === "" ? null : (e.target.value as Norm) })}
              className="field"
            >
              <option value="">default</option>
              <option value="minmax">min-max</option>
              <option value="zscore">z-score</option>
            </select>
            <span className="text-xs text-muted">Used by weighted fusion only</span>
          </label>
          <div className="flex items-end">
            <button type="button" className="btn-quiet" onClick={() => onChange({})}>
              Reset to defaults
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
