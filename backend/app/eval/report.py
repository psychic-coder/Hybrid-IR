"""Result files and programmatic Markdown tables.

Writes into RESULTS_DIR:

    summary.json      per-configuration aggregate metrics, latency, QPS, significance
    summary.csv       the same, flattened
    per_query.jsonl   one line per (configuration, query): metrics + top-100 ranking
    ablations.json    every ablation table
    ablations_<name>.csv
    pareto.json       nDCG@10 vs p95 latency points with the Pareto frontier flagged
    index_stats.json  inverted-index statistics
    report.md         Markdown tables rendered from the files above

This module is deliberately standard-library only, so ``scripts/render_readme.py`` can import
it on a host that has nothing but ``python3``. No metric value is ever hard-coded: every
table is rendered from the JSON files.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PLACEHOLDER = "_No results yet. Run `make eval` to populate._"


# ----------------------------------------------------------------------------- io
def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp file + rename) so readers never see partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=float))
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """Parsed JSON or None if missing/invalid."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write dict rows (union of keys, in first-seen order) as CSV."""
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def write_per_query(results_dir: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """per_query.jsonl (atomic)."""
    path = Path(results_dir) / "per_query.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- pareto
def compute_pareto(
    summary: Mapping[str, Any] | None, ablations: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Collect (nDCG@10, end-to-end p95) points and flag non-dominated ones."""
    points: list[dict[str, Any]] = []
    for s in (summary or {}).get("strategies", []):
        points.append(
            {
                "label": s["name"],
                "source": "strategy",
                "ndcg_at_10": float(s["metrics"]["ndcg@10"]),
                "p95_ms": float(s["latency_cold"]["p95"]),
            }
        )
    for r in (ablations or {}).get("rerank_top_k", []):
        if r.get("total_p95_ms") is not None:
            points.append(
                {
                    "label": f"hybrid_rrf_rerank@{r['rerank_top_k']}",
                    "source": "rerank_top_k",
                    "ndcg_at_10": float(r["ndcg_at_10"]),
                    "p95_ms": float(r["total_p95_ms"]),
                }
            )
    for p in points:
        p["on_frontier"] = not any(
            (q["ndcg_at_10"] >= p["ndcg_at_10"] and q["p95_ms"] <= p["p95_ms"])
            and (q["ndcg_at_10"] > p["ndcg_at_10"] or q["p95_ms"] < p["p95_ms"])
            for q in points
        )
    points.sort(key=lambda p: p["p95_ms"])
    return points


# ----------------------------------------------------------------------------- writers
def write_summary(results_dir: Path, summary: Mapping[str, Any]) -> None:
    """summary.json + summary.csv."""
    results_dir = Path(results_dir)
    write_json(results_dir / "summary.json", summary)
    rows = []
    for s in summary.get("strategies", []):
        row: dict[str, Any] = {"name": s["name"], "strategy": s["strategy"], "n_queries": s["n_queries"]}
        row.update(s["metrics"])
        for k in ("mean", "p50", "p95", "p99"):
            row[f"cold_{k}_ms"] = s["latency_cold"][k]
            row[f"warm_{k}_ms"] = (s.get("latency_warm") or {}).get(k)
        row["qps"] = s.get("qps")
        sig = s.get("significance") or {}
        row["delta_ndcg10_vs_bm25"] = sig.get("mean_delta")
        row["p_value"] = sig.get("p_value")
        rows.append(row)
    write_csv(results_dir / "summary.csv", rows)


def write_ablations(results_dir: Path, ablations: Mapping[str, Any]) -> None:
    """ablations.json + one CSV per table."""
    results_dir = Path(results_dir)
    write_json(results_dir / "ablations.json", ablations)
    for key in ("bm25_grid", "chunk_size", "rrf_k", "alpha_sweep", "hnsw", "rerank_top_k"):
        rows = ablations.get(key) or []
        if rows:
            write_csv(results_dir / f"ablations_{key}.csv", rows)


def write_pareto(results_dir: Path) -> list[dict[str, Any]]:
    """Recompute pareto.json from whatever summary/ablations exist."""
    results_dir = Path(results_dir)
    pts = compute_pareto(read_json(results_dir / "summary.json"), read_json(results_dir / "ablations.json"))
    write_json(results_dir / "pareto.json", {"points": pts})
    return pts


def write_index_stats(results_dir: Path, stats: Mapping[str, Any]) -> None:
    """index_stats.json."""
    write_json(Path(results_dir) / "index_stats.json", stats)


# ----------------------------------------------------------------------------- markdown
def _f(x: Any, digits: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(results_dir: Path, heading_level: int = 3) -> str:
    """All result tables as Markdown, or a placeholder when nothing has been generated."""
    results_dir = Path(results_dir)
    h = "#" * heading_level
    summary = read_json(results_dir / "summary.json")
    ablations = read_json(results_dir / "ablations.json")
    index_stats = read_json(results_dir / "index_stats.json")
    pareto = (read_json(results_dir / "pareto.json") or {}).get("points", [])
    k6 = read_json(results_dir / "k6_summary.json")
    if not summary:
        return PLACEHOLDER
    parts: list[str] = []
    parts.append(
        f"_Generated from `results/*.json` at {summary.get('generated_at')} — dataset "
        f"**{summary.get('dataset')}**, {summary.get('n_queries')} test queries. "
        "Latency columns are in-process milliseconds with caches disabled (cold)._"
    )
    rows = []
    for s in summary.get("strategies", []):
        m, lc = s["metrics"], s["latency_cold"]
        rows.append(
            [
                f"`{s['name']}`",
                _f(m["ndcg@10"]),
                _f(m["ndcg@100"]),
                _f(m["recall@10"]),
                _f(m["recall@100"]),
                _f(m["mrr@10"]),
                _f(m["map@100"]),
                _f(m["p@10"]),
                _f(lc["p50"], 1),
                _f(lc["p95"], 1),
                _f(lc["p99"], 1),
                _f((s.get("latency_warm") or {}).get("p95"), 1),
                _f(s.get("qps"), 1),
            ]
        )
    parts.append(
        f"{h} Strategies\n\n"
        + _table(
            [
                "Config",
                "nDCG@10",
                "nDCG@100",
                "R@10",
                "R@100",
                "MRR@10",
                "MAP@100",
                "P@10",
                "p50 ms",
                "p95 ms",
                "p99 ms",
                "warm p95 ms",
                "QPS",
            ],
            rows,
        )
    )

    sig_rows = []
    for s in summary.get("strategies", []):
        sig = s.get("significance")
        if sig:
            sig_rows.append(
                [
                    f"`{s['name']}`",
                    _f(sig["mean_delta"]),
                    f"[{_f(sig['ci_low'])}, {_f(sig['ci_high'])}]",
                    _f(sig["p_value"]),
                    "yes" if sig["significant"] else "no",
                ]
            )
    if sig_rows:
        n_res = next(
            (s["significance"]["n_resamples"] for s in summary["strategies"] if s.get("significance")), 0
        )
        parts.append(
            f"{h} Significance vs BM25 (paired bootstrap, {n_res:,} resamples, nDCG@10)\n\n"
            + _table(["System", "Δ nDCG@10", "95% CI", "p-value", "Significant (α=0.05)"], sig_rows)
        )

    if pareto:
        parts.append(
            f"{h} Quality / latency Pareto frontier\n\n"
            + _table(
                ["Point", "Source", "nDCG@10", "p95 ms", "On frontier"],
                [
                    [
                        f"`{p['label']}`",
                        p["source"],
                        _f(p["ndcg_at_10"]),
                        _f(p["p95_ms"], 1),
                        "**yes**" if p["on_frontier"] else "no",
                    ]
                    for p in pareto
                ],
            )
        )

    if index_stats:
        keys = [
            "n_docs",
            "n_chunks",
            "vocab_size",
            "avg_chunk_len",
            "total_postings",
            "total_positions",
            "postings_bytes",
            "raw_bytes_estimate",
            "compression_ratio",
            "build_seconds",
            "peak_rss_mb",
            "embedding_seconds",
        ]
        parts.append(
            f"{h} Index statistics\n\n"
            + _table(
                ["Statistic", "Value"], [[k, _f(index_stats.get(k), 2)] for k in keys if k in index_stats]
            )
        )

    if ablations:
        parts.append(_render_ablations(ablations, h))

    if k6:
        parts.append(
            f"{h} Load test (k6, `{k6.get('script')}`)\n\n"
            + _table(
                ["Strategy", "Requests", "avg ms", "p50 ms", "p95 ms", "p99 ms"],
                [
                    [
                        s["strategy"],
                        _f(s["count"]),
                        _f(s["avg_ms"], 1),
                        _f(s["p50_ms"], 1),
                        _f(s["p95_ms"], 1),
                        _f(s["p99_ms"], 1),
                    ]
                    for s in k6.get("strategies", [])
                ],
            )
            + f"\n\nTotal {k6.get('total_requests')} requests, error rate {_f(k6.get('error_rate'))}, "
            f"{_f(k6.get('rps'), 2)} req/s over {_f(k6.get('duration_s'), 0)} s."
        )

    env = summary.get("environment") or {}
    if env:
        parts.append(
            f"{h} Environment\n\n" + _table(["Key", "Value"], [[k, f"`{v}`"] for k, v in env.items()])
        )
    return "\n\n".join(parts)


def _render_ablations(ab: Mapping[str, Any], h: str) -> str:
    parts = [f"{h} Ablations (n = {ab.get('n_queries')} queries)"]
    grid = ab.get("bm25_grid") or []
    if grid:
        k1s = sorted({r["k1"] for r in grid})
        bs = sorted({r["b"] for r in grid})
        cell = {(r["k1"], r["b"]): r["ndcg_at_10"] for r in grid}
        best = max(grid, key=lambda r: r["ndcg_at_10"])
        parts.append(
            f"**BM25 nDCG@10 (rows k1, columns b)** — best k1={best['k1']}, b={best['b']}\n\n"
            + _table(
                ["k1 \\ b"] + [str(b) for b in bs],
                [[str(k)] + [_f(cell.get((k, b))) for b in bs] for k in k1s],
            )
        )
    chunk = ab.get("chunk_size") or []
    if chunk:
        parts.append(
            "**Chunk size (BM25)**\n\n"
            + _table(
                ["CHUNK_SIZE", "nDCG@10", "R@100", "chunks", "index bytes", "build s"],
                [
                    [
                        "none" if r["chunk_size"] is None else r["chunk_size"],
                        _f(r["ndcg_at_10"]),
                        _f(r["recall_at_100"]),
                        _f(r["n_chunks"]),
                        _f(r["index_bytes"]),
                        _f(r["build_seconds"], 2),
                    ]
                    for r in chunk
                ],
            )
        )
    rrfk = ab.get("rrf_k") or []
    if rrfk:
        parts.append(
            "**RRF k**\n\n"
            + _table(
                ["k", "nDCG@10", "R@100"],
                [[r["rrf_k"], _f(r["ndcg_at_10"]), _f(r["recall_at_100"])] for r in rrfk],
            )
        )
    alpha = ab.get("alpha_sweep") or []
    if alpha:
        norms = sorted({r["norm"] for r in alpha})
        alphas = sorted({r["alpha"] for r in alpha})
        cell = {(r["norm"], r["alpha"]): r["ndcg_at_10"] for r in alpha}
        parts.append(
            "**Weighted fusion alpha sweep (nDCG@10; alpha = dense weight)**\n\n"
            + _table(["alpha", *norms], [[_f(a, 1)] + [_f(cell.get((n, a))) for n in norms] for a in alphas])
        )
    hn = ab.get("hnsw") or []
    if hn:
        parts.append(
            "**HNSW (recall@10 vs exact search, ANN latency)**\n\n"
            + _table(
                ["M", "ef_construct", "ef", "recall@10", "nDCG@10", "p50 ms", "p95 ms"],
                [
                    [
                        r["m"],
                        r["ef_construct"],
                        r["ef"],
                        _f(r["recall_at_10_vs_exact"]),
                        _f(r["ndcg_at_10"]),
                        _f(r["p50_ms"], 2),
                        _f(r["p95_ms"], 2),
                    ]
                    for r in hn
                ],
            )
        )
    rr = ab.get("rerank_top_k") or []
    if rr:
        parts.append(
            "**Rerank depth (hybrid_rrf + cross-encoder)**\n\n"
            + _table(
                ["top-k", "nDCG@10", "R@100", "rerank p50 ms", "rerank p95 ms", "end-to-end p95 ms"],
                [
                    [
                        r["rerank_top_k"],
                        _f(r["ndcg_at_10"]),
                        _f(r["recall_at_100"]),
                        _f(r["p50_ms"], 1),
                        _f(r["p95_ms"], 1),
                        _f(r.get("total_p95_ms"), 1),
                    ]
                    for r in rr
                ],
            )
        )
    notes = ab.get("notes") or []
    if notes:
        parts.append("\n".join(f"- {n}" for n in notes))
    return "\n\n".join(parts)


def write_report_md(results_dir: Path) -> str:
    """report.md from the current result files."""
    md = "# HybridIR evaluation report\n\n" + render_markdown(results_dir, heading_level=2) + "\n"
    (Path(results_dir) / "report.md").write_text(md)
    return md
