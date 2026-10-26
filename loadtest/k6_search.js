// k6 load test: open-model (ramping-arrival-rate) load against POST /search, one scenario per
// strategy, run back to back. Queries are the test queries exported by the seed service.
//
//   docker compose --profile loadtest run --rm k6            # this script
//   K6_SCRIPT=k6_mixed.js docker compose --profile loadtest run --rm k6
//
// Arrival-rate executors start iterations on a schedule regardless of how slow responses are,
// so they do not suffer from coordinated omission (see loadtest/README.md).

import http from "k6/http";
import { check } from "k6";

const API = __ENV.API_URL || "http://api:8000";
const RESULTS = __ENV.RESULTS_DIR || "/results";
const STAGE = parseInt(__ENV.K6_STAGE_SECONDS || "30", 10);
const P95 = parseInt(__ENV.K6_P95_MS || "2000", 10);
const RERANK_P95 = parseInt(__ENV.K6_RERANK_P95_MS || "8000", 10);
const QUERIES = JSON.parse(open(__ENV.QUERIES_FILE || "/results/queries.json")).queries;

const STRATEGIES = [
  { name: "bm25", rate: parseInt(__ENV.K6_BM25_RATE || "20", 10), p95: P95 },
  { name: "dense", rate: parseInt(__ENV.K6_DENSE_RATE || "10", 10), p95: P95 },
  { name: "hybrid_rrf", rate: parseInt(__ENV.K6_HYBRID_RATE || "8", 10), p95: P95 },
  { name: "hybrid_rrf_rerank", rate: parseInt(__ENV.K6_RERANK_RATE || "2", 10), p95: RERANK_P95 },
];

// ramp up (STAGE s), hold (STAGE s), ramp down (STAGE/3 s); scenarios do not overlap
const SCENARIO_SECONDS = STAGE * 2 + Math.ceil(STAGE / 3) + 5;

function scenario(s, index) {
  return {
    executor: "ramping-arrival-rate",
    exec: "search",
    env: { STRATEGY: s.name },
    tags: { strategy: s.name },
    startTime: `${index * SCENARIO_SECONDS}s`,
    startRate: 1,
    timeUnit: "1s",
    preAllocatedVUs: Math.max(4, s.rate * 2),
    maxVUs: Math.max(20, s.rate * 10),
    stages: [
      { target: s.rate, duration: `${STAGE}s` },
      { target: s.rate, duration: `${STAGE}s` },
      { target: 0, duration: `${Math.ceil(STAGE / 3)}s` },
    ],
  };
}

const scenarios = {};
const thresholds = { http_req_failed: ["rate<0.01"] };
STRATEGIES.forEach((s, i) => {
  scenarios[s.name] = scenario(s, i);
  // thresholds on tagged sub-metrics also make k6 report them in handleSummary
  thresholds[`http_req_duration{strategy:${s.name}}`] = [`p(95)<${s.p95}`];
});

export const options = {
  scenarios,
  thresholds,
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max", "count"],
  discardResponseBodies: true,
};

export function search() {
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const res = http.post(
    `${API}/search`,
    JSON.stringify({ query: q, strategy: __ENV.STRATEGY, top_k: 10, use_cache: false }),
    { headers: { "Content-Type": "application/json" }, tags: { strategy: __ENV.STRATEGY }, timeout: "60s" },
  );
  check(res, { "status is 200": (r) => r.status === 200 });
}

export function summarize(data, script, names) {
  const strategies = [];
  for (const name of names) {
    const m = data.metrics[`http_req_duration{strategy:${name}}`];
    if (!m) continue;
    const v = m.values;
    strategies.push({
      strategy: name,
      count: Math.round(v.count || 0),
      avg_ms: v.avg || 0,
      p50_ms: v.med || 0,
      p95_ms: v["p(95)"] || 0,
      p99_ms: v["p(99)"] || 0,
    });
  }
  const reqs = data.metrics.http_reqs ? data.metrics.http_reqs.values : { count: 0, rate: 0 };
  const failed = data.metrics.http_req_failed ? data.metrics.http_req_failed.values.rate : 0;
  return {
    script,
    generated_at: new Date().toISOString(),
    duration_s: (data.state.testRunDurationMs || 0) / 1000,
    total_requests: reqs.count || 0,
    rps: reqs.rate || 0,
    error_rate: failed || 0,
    strategies,
  };
}

export function handleSummary(data) {
  const out = summarize(data, "k6_search.js", STRATEGIES.map((s) => s.name));
  return {
    [`${RESULTS}/k6_summary.json`]: JSON.stringify(out, null, 2),
    [`${RESULTS}/k6_search_raw.json`]: JSON.stringify(data, null, 2),
    stdout: `\nk6 summary written to ${RESULTS}/k6_summary.json\n${JSON.stringify(out, null, 2)}\n`,
  };
}
