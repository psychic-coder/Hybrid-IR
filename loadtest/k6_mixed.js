// k6 mixed workload: 70% hybrid_rrf, 20% hybrid_rrf_rerank, 10% /compare at a constant
// open-model arrival rate (ramping-arrival-rate executor).
//
//   K6_SCRIPT=k6_mixed.js docker compose --profile loadtest run --rm k6

import http from "k6/http";
import { check } from "k6";
import { summarize } from "./k6_search.js";

const API = __ENV.API_URL || "http://api:8000";
const RESULTS = __ENV.RESULTS_DIR || "/results";
const STAGE = parseInt(__ENV.K6_STAGE_SECONDS || "30", 10);
const RATE = parseInt(__ENV.K6_MIXED_RATE || "5", 10);
const P95 = parseInt(__ENV.K6_P95_MS || "2000", 10);
const RERANK_P95 = parseInt(__ENV.K6_RERANK_P95_MS || "8000", 10);
const QUERIES = JSON.parse(open(__ENV.QUERIES_FILE || "/results/queries.json")).queries;

export const options = {
  scenarios: {
    mixed: {
      executor: "ramping-arrival-rate",
      exec: "mixed",
      startRate: 1,
      timeUnit: "1s",
      preAllocatedVUs: Math.max(5, RATE * 2),
      maxVUs: Math.max(30, RATE * 10),
      stages: [
        { target: RATE, duration: `${STAGE}s` },
        { target: RATE, duration: `${STAGE * 2}s` },
        { target: 0, duration: `${Math.ceil(STAGE / 3)}s` },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{strategy:hybrid_rrf}": [`p(95)<${P95}`],
    "http_req_duration{strategy:hybrid_rrf_rerank}": [`p(95)<${RERANK_P95}`],
    "http_req_duration{strategy:compare}": [`p(95)<${RERANK_P95 * 2}`],
  },
  summaryTrendStats: ["avg", "min", "med", "p(90)", "p(95)", "p(99)", "max", "count"],
  discardResponseBodies: true,
};

const JSON_HEADERS = { "Content-Type": "application/json" };

export function mixed() {
  const q = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const roll = Math.random();
  let res;
  if (roll < 0.7) {
    res = http.post(`${API}/search`, JSON.stringify({ query: q, strategy: "hybrid_rrf", top_k: 10, use_cache: false }),
      { headers: JSON_HEADERS, tags: { strategy: "hybrid_rrf" }, timeout: "60s" });
  } else if (roll < 0.9) {
    res = http.post(`${API}/search`, JSON.stringify({ query: q, strategy: "hybrid_rrf_rerank", top_k: 10, use_cache: false }),
      { headers: JSON_HEADERS, tags: { strategy: "hybrid_rrf_rerank" }, timeout: "60s" });
  } else {
    res = http.post(`${API}/compare`, JSON.stringify({ query: q, top_k: 10 }),
      { headers: JSON_HEADERS, tags: { strategy: "compare" }, timeout: "120s" });
  }
  check(res, { "status is 200": (r) => r.status === 200 });
}

export function handleSummary(data) {
  const out = summarize(data, "k6_mixed.js", ["hybrid_rrf", "hybrid_rrf_rerank", "compare"]);
  return {
    [`${RESULTS}/k6_summary.json`]: JSON.stringify(out, null, 2),
    [`${RESULTS}/k6_mixed_raw.json`]: JSON.stringify(data, null, 2),
    stdout: `\nk6 summary written to ${RESULTS}/k6_summary.json\n${JSON.stringify(out, null, 2)}\n`,
  };
}
