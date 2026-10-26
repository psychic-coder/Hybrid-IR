# Load testing with k6

Two scripts drive the running `api` service over HTTP:

| Script | Workload | Output |
|---|---|---|
| `k6_search.js` | One `ramping-arrival-rate` scenario per strategy (`bm25`, `dense`, `hybrid_rrf`, `hybrid_rrf_rerank`), run back to back: ramp to the target rate, hold, ramp down | `results/k6_summary.json`, `results/k6_search_raw.json` |
| `k6_mixed.js` | 70% `hybrid_rrf`, 20% `hybrid_rrf_rerank`, 10% `/compare` at one target rate | `results/k6_summary.json`, `results/k6_mixed_raw.json` |

Queries are sampled from the dataset's test split. The seed service exports
`K6_QUERY_SAMPLE` of them to `results/queries.json`, which the k6 container reads from the
shared `results_data` volume.

## Running

```bash
make up                 # stack must be healthy first
make load               # = docker compose --profile loadtest run --rm k6
make load-mixed         # = K6_SCRIPT=k6_mixed.js docker compose --profile loadtest run --rm k6
make readme             # re-render README results (includes the k6 table)
```

Tune the load with environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `K6_BM25_RATE`, `K6_DENSE_RATE`, `K6_HYBRID_RATE`, `K6_RERANK_RATE` | 20, 10, 8, 2 | target requests/second per strategy scenario |
| `K6_MIXED_RATE` | 5 | target requests/second for the mixed workload |
| `K6_STAGE_SECONDS` | 30 | ramp and hold duration |
| `K6_P95_MS`, `K6_RERANK_P95_MS` | 2000, 8000 | p95 thresholds; a breach makes k6 exit non-zero |

The defaults are deliberately modest: the cross-encoder runs on CPU, and an arrival rate the
machine cannot sustain only measures queueing.

## How the numbers reach the Metrics page

`handleSummary` writes `results/k6_summary.json` (per-strategy count, avg, p50, p95, p99 and
overall request rate and error rate). `GET /metrics/summary` includes it as `load_test`, the
Metrics page renders it as the *Load test* table, and `make readme` adds it to the README
results block. The last script run wins; both scripts write the same file.

## Reporting load-test numbers honestly

- **State the hardware.** CPU model and core count, RAM, whether Docker Desktop's VM limits
  apply (macOS/Windows), and `TORCH_THREADS`. The environment block of `results/summary.json`
  records most of this.
- **State dataset and configuration.** Dataset name, corpus size, `DENSE_BACKEND`, HNSW
  parameters and `RERANK_TOP_K` all change latency.
- **Cache state.** The scripts send `use_cache: false`, which bypasses both the result cache
  and the query-embedding cache, so they measure cold compute. If you change that, say so:
  with a 200-query sample and repeated arrivals, a warm result cache turns the test into a
  Redis benchmark.
- **Coordinated omission.** A closed-loop tester (N virtual users each waiting for a
  response before sending the next request) slows down exactly when the server does and
  therefore under-reports tail latency. Arrival-rate executors schedule requests on a clock
  independent of responses, so slow responses show up in the percentiles. If k6 warns that it
  ran out of VUs, the target rate exceeded capacity and the results are not valid at that
  rate; lower the rate or raise `maxVUs`.
- **In-process vs over-the-wire.** The Metrics page's main latency columns come from the
  in-process evaluation harness (no HTTP, no JSON). k6 numbers include HTTP, serialization
  and queueing, so they are higher; report which one you quote.
- **Run long enough.** p99 from a 60-second hold at 2 req/s is computed from ~120 samples;
  treat it as a rough estimate.
