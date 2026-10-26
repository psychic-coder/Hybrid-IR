# Evaluation protocol

What exactly is measured, how, and what the numbers do and do not mean. Every number shown
anywhere in this project is produced by the procedure below and read from `results/*.json`.

## Data

- **Corpus:** `corpus.jsonl` of the BEIR dataset named by `DATASET`, in file order.
- **Queries:** `queries.jsonl` restricted to queries that appear in `qrels/test.tsv` with at
  least one judgment of grade > 0. Queries are processed in sorted-id order;
  `EVAL_MAX_QUERIES=N` keeps the first N (deterministic, but *not* a random sample — only use
  it for quick iterations).
- **Judgments:** graded qrels from `qrels/test.tsv`. Grade > 0 means relevant. Documents with
  no judgment are treated as grade 0 (see "Unjudged documents" below).
- **Provenance:** the sha256 of the downloaded zip is recorded in
  `results/summary.json -> environment.dataset_zip_sha256`.

## Configurations

`run_baseline()` evaluates one configuration per strategy at the server defaults:
`bm25, dense, hybrid_rrf, hybrid_weighted, hybrid_rrf_rerank, hybrid_weighted_rerank`. A
strategy whose stage is unavailable is skipped with a warning, never silently scored as 0.

## Ranking depth

Each configuration returns a ranking of `EVAL_DEPTH` (default 100) documents per query.
Hybrid strategies fuse `CANDIDATE_DEPTH` documents from each retriever. Rerank strategies
rescore the top `RERANK_TOP_K` fused documents and keep the remaining fused order below them,
so @100 metrics are well defined for every strategy.

## Metrics (`app/eval/metrics.py`)

Let `rel(d)` be the graded judgment (0 if unjudged) and `R = {d : rel(d) > 0}`.

| Metric | Definition |
|---|---|
| DCG@k | `sum_{i=1..k} rel(d_i) / log2(i + 1)` (linear gain, as trec_eval / BEIR) |
| nDCG@k | DCG@k / IDCG@k, where IDCG@k is the DCG of **all** judged grades sorted descending (not just retrieved ones); 0 if IDCG = 0 |
| Recall@k | `|R ∩ top_k| / |R|` |
| P@k | `|R ∩ top_k| / k` (denominator is k even if fewer docs returned) |
| MRR@k | `1 / rank` of the first relevant document within k, else 0 |
| AP@k | `(1/|R|) * sum over relevant d_i with i <= k of P@i` (MAP@100 averages AP@100) |

All metrics are macro-averaged over queries. Unit tests in `tests/test_metrics.py` check
each against hand-computed values. The exponential-gain variant `(2^rel - 1)` is available
(`gain="exponential"`) but not reported by default.

## Unjudged documents

BEIR qrels are sparse: most corpus documents were never shown to an assessor. Scoring an
unjudged document as non-relevant is the standard convention, and it systematically
penalizes systems that retrieve *different* documents from those in the judgment pool (often
dense retrievers on lexically-pooled datasets). Treat absolute differences between lexical and
dense systems with that bias in mind; `per_query.jsonl` lets you count unjudged documents in
each top-10.

## Latency

- **Cold** (headline): every request bypasses both the result cache and the query-embedding
  cache. Measured per query with `time.perf_counter()` inside the pipeline; HTTP and JSON are
  not included.
- **Warm**: for strategies with a dense stage, the embedding cache is primed for every query,
  then the pass is repeated with the embedding cache on and the result cache off. The
  difference to cold is the value of caching query embeddings.
- **QPS**: the first `EVAL_QPS_QUERIES` queries through a `ThreadPoolExecutor` with
  `EVAL_QPS_THREADS` workers, caches off; QPS = completed / wall time.
- Percentiles are NumPy's linear-interpolation percentiles over per-query latencies.

## Significance

For each non-BM25 configuration, a **paired bootstrap** on per-query nDCG@10 differences
`d_i = system_i - bm25_i` with `BOOTSTRAP_RESAMPLES` (default 10,000) resamples of queries
with replacement and a fixed seed:

- mean delta = mean(d);
- 95% CI = 2.5th and 97.5th percentiles of the resampled means;
- p-value (two-sided) = `(#{|mean(d* - mean(d))| >= |mean(d)|} + 1) / (B + 1)` (shifted
  bootstrap under H0: E[d] = 0);
- *significant* when p < 0.05 and the CI excludes 0.

With six configurations compared to one baseline, apply a multiple-comparison correction
(e.g. Holm–Bonferroni) before claiming several wins at once. The raw p-values are reported so
you can.

## Pareto frontier

`results/pareto.json` contains one point per strategy (nDCG@10, cold p95) plus one per rerank
depth from the ablation (nDCG@10, end-to-end p95 on the latency sample). A point is on the
frontier if no other point has both higher-or-equal nDCG@10 and lower-or-equal p95 with at
least one strict inequality.

## Reproducing a run

```bash
make clean          # optional: wipe volumes for a truly cold run
make up             # seed runs baseline eval (+ ablations)
make eval           # rerun baseline only
make ablations      # rerun ablations only
make results        # copy results/ out of the volume
make readme         # render README tables from results/
```

Report alongside any number: dataset, `n_queries`, the `environment` block (CPU, cores, RAM,
torch threads, versions), dense backend and HNSW parameters, and whether caches were cold.
