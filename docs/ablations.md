# Ablations

Each ablation changes one knob and holds every other parameter at its default
(`.env.example`). Results are written to `results/ablations.json` (+ one CSV per table) and
shown on the **Ablations** page. Run them with `make ablations` (or `ONLY=hnsw make ablations`
to restrict); the seed service runs all of them on first start when `ABLATIONS_ON_SEED=true`.

| Id | Grid | What is measured | Implementation notes |
|---|---|---|---|
| `bm25_grid` | k1 ∈ {0.6, 0.9, 1.2, 1.5, 2.0} × b ∈ {0.25, 0.5, 0.75, 0.9} | nDCG@10, Recall@100, mean BM25 latency | WAND with the Pareto-front upper bound; no rebuild needed |
| `chunk_size` | CHUNK_SIZE ∈ {128, 256, 512, none} | nDCG@10, Recall@100, #chunks, index bytes, build time | index rebuilt in a temp dir per size; **BM25 only** (the dense side is embedded once at the default size to bound first-run time) |
| `fusion` → `rrf_k` | k ∈ {10, 30, 60, 100} | nDCG@10, Recall@100 | BM25 and dense candidates retrieved once per query, then fused many ways |
| `fusion` → `alpha_sweep` | alpha ∈ {0.0, 0.1, …, 1.0} × norm ∈ {minmax, zscore} | nDCG@10, Recall@100 | alpha = 0 is BM25 alone, 1 is dense alone |
| `hnsw` | M ∈ {8, 16, 32} × ef ∈ {16, 32, 64, 128, 256}, ef_construct = `HNSW_EF_CONSTRUCT` | recall@10 vs exact search (chunk level), doc-level nDCG@10, ANN-call mean/p50/p95 | one Qdrant collection per M (non-default ones deleted afterwards); with `DENSE_BACKEND=custom` the from-scratch HNSW is built per M |
| `rerank_top_k` | k ∈ {10, 20, 50, 100} | nDCG@10, Recall@100, cross-encoder stage and end-to-end latency | cross-encoder scores for the top 100 computed once per query (scores are per pair, so reranking the top k with them equals rerunning); latency measured for real on `ABLATION_LATENCY_SAMPLE` queries per k |

## How to read them

- **BM25 heatmap.** k1 controls term-frequency saturation (low k1: the first occurrence is
  almost all that matters); b controls length normalization (b = 0: none). Scientific claims
  (scifact) are short queries against abstracts, so expect a broad flat optimum; a sharp
  optimum on a small query set is more likely noise than signal — check it against the
  bootstrap CI magnitude in `summary.json`.
- **Alpha sweep.** The endpoints reproduce the single retrievers (up to tie-breaking among
  documents missing from one list). Min-max is sensitive to the extreme scores of each list;
  z-score to the list's spread. A curve that peaks in the interior is the evidence that the
  two retrievers are complementary.
- **RRF k.** Small k strongly favors documents at the very top of either list; large k makes
  RRF approach a Borda-like vote. The canonical k = 60 is a default, not a law.
- **HNSW.** Recall against *exact search over the same embeddings* isolates the ANN error
  from the embedding model's error. Raising ef trades latency for recall with diminishing
  returns; raising M increases memory and build time and helps mostly at high recall targets.
  On small corpora (scifact) an exact scan is already fast, so latency differences can be
  dominated by HTTP/serialization overhead to Qdrant.
- **Rerank depth.** Quality usually rises then plateaus as k grows while cost grows linearly;
  the knee of the curve is the practical setting. The end-to-end points also appear on the
  Metrics page's Pareto chart.

## Adding an ablation

1. Write a function in `backend/app/eval/ablations.py` returning a list of dict rows.
2. Call it from `run_ablations()` under a new key and add the key to `ALL`.
3. Add a row model to `backend/app/schemas.py::AblationsResponse` and the mirror type in
   `frontend/src/lib/types.ts`.
4. Render it in `frontend/src/app/ablations/page.tsx` and in
   `backend/app/eval/report.py::_render_ablations` (README/report tables).
