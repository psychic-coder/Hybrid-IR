/**
 * Types mirroring backend/app/schemas.py one-to-one. Keep the two files in sync:
 * a field renamed on one side must be renamed on the other.
 */

export const STRATEGIES = [
  "bm25",
  "dense",
  "hybrid_rrf",
  "hybrid_weighted",
  "hybrid_rrf_rerank",
  "hybrid_weighted_rerank",
] as const;
export type Strategy = (typeof STRATEGIES)[number];
export type Norm = "minmax" | "zscore";
export type Stage = "bm25" | "dense" | "fusion" | "rerank";

export interface SearchParams {
  k1?: number | null;
  b?: number | null;
  rrf_k?: number | null;
  alpha?: number | null;
  norm?: Norm | null;
  rerank_top_k?: number | null;
  ef?: number | null;
  candidate_depth?: number | null;
}

export interface EffectiveParams {
  k1: number;
  b: number;
  rrf_k: number;
  alpha: number;
  norm: Norm;
  rerank_top_k: number;
  ef: number;
  candidate_depth: number;
}

export interface SearchRequest {
  query: string;
  strategy: Strategy;
  top_k: number;
  params: SearchParams;
  use_cache?: boolean;
}

export interface StageScores {
  bm25: number | null;
  dense: number | null;
  fusion: number | null;
  rerank: number | null;
}

export interface StageRanks {
  bm25: number | null;
  dense: number | null;
  fusion: number | null;
  rerank: number | null;
}

export interface SearchHit {
  rank: number;
  doc_id: string;
  title: string;
  snippet: string;
  score: number;
  scores: StageScores;
  ranks: StageRanks;
  chunk_id: number | null;
}

export interface Timings {
  tokenize_ms: number;
  bm25_ms: number;
  embed_ms: number;
  ann_ms: number;
  fusion_ms: number;
  rerank_ms: number;
  total_ms: number;
}

export interface CacheFlags {
  result_cache_hit: boolean;
  embedding_cache_hit: boolean | null;
}

export interface SearchResponse {
  query: string;
  strategy: Strategy;
  top_k: number;
  params: EffectiveParams;
  hits: SearchHit[];
  total_candidates: number;
  query_terms: string[];
  timings: Timings;
  cache: CacheFlags;
  dense_backend: string;
}

export interface CompareRequest {
  query: string;
  strategies: Strategy[];
  top_k: number;
  params: SearchParams;
}

export interface ErrorInfo {
  error: string;
  detail: string;
  stage: string | null;
  request_id: string | null;
}

export interface StrategyResult {
  strategy: Strategy;
  response: SearchResponse | null;
  error: ErrorInfo | null;
}

export interface RankShiftRow {
  doc_id: string;
  title: string;
  ranks: Record<string, number | null>;
  delta_vs_bm25: Record<string, number | null>;
  is_new: Record<string, boolean>;
}

export interface CompareResponse {
  query: string;
  strategies: Strategy[];
  top_k: number;
  baseline: Strategy;
  results: StrategyResult[];
  rank_shift: RankShiftRow[];
}

export interface TermContribution {
  term: string;
  qtf: number;
  tf: number;
  df: number;
  idf: number;
  doc_len: number;
  avg_doc_len: number;
  length_norm: number;
  tf_component: number;
  score: number;
}

export interface ListStats {
  n: number;
  min: number;
  max: number;
  mean: number;
  std: number;
}

export interface FusionMath {
  candidate_depth: number;
  rrf_k: number;
  bm25_rank: number | null;
  dense_rank: number | null;
  rrf_bm25_term: number;
  rrf_dense_term: number;
  rrf_score: number;
  alpha: number;
  norm: Norm;
  bm25_raw: number | null;
  dense_raw: number | null;
  bm25_norm: number;
  dense_norm: number;
  bm25_list: ListStats;
  dense_list: ListStats;
  weighted_score: number;
}

export interface HighlightSpan {
  start: number;
  end: number;
}

export interface ExplainResponse {
  query: string;
  doc_id: string;
  title: string;
  text: string;
  title_highlights: HighlightSpan[];
  text_highlights: HighlightSpan[];
  chunk_id: number | null;
  query_terms: string[];
  phrases: string[][];
  params: EffectiveParams;
  bm25_score: number;
  phrase_boost: number;
  terms: TermContribution[];
  cosine: number | null;
  dense_error: string | null;
  fusion: FusionMath | null;
  rerank_score: number | null;
  rerank_error: string | null;
  ranks: Record<string, number | null>;
  delta_vs_bm25: Record<string, number | null>;
}

export interface LatencyStats {
  n: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

export interface Significance {
  baseline: string;
  system: string;
  metric: string;
  n_queries: number;
  n_resamples: number;
  mean_delta: number;
  ci_low: number;
  ci_high: number;
  p_value: number;
  significant: boolean;
}

export interface StrategyMetrics {
  name: string;
  strategy: Strategy;
  params: EffectiveParams;
  n_queries: number;
  metrics: Record<string, number>;
  latency_cold: LatencyStats;
  latency_warm: LatencyStats | null;
  stage_means_ms: Record<string, number>;
  qps: number | null;
  qps_threads: number | null;
  significance: Significance | null;
}

export interface IndexStats {
  dataset: string;
  n_docs: number;
  n_chunks: number;
  vocab_size: number;
  avg_chunk_len: number;
  total_postings: number;
  total_positions: number;
  postings_bytes: number;
  index_bytes: number;
  raw_bytes_estimate: number;
  compression_ratio: number;
  build_seconds: number;
  peak_rss_mb: number;
  chunk_size: number;
  chunk_overlap: number;
  skip_interval: number;
  embedding_seconds: number | null;
  n_vectors: number | null;
}

export interface ParetoPoint {
  label: string;
  source: string;
  ndcg_at_10: number;
  p95_ms: number;
  on_frontier: boolean;
}

export interface LoadTestStrategy {
  strategy: string;
  count: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface LoadTestSummary {
  script: string;
  generated_at: string;
  duration_s: number;
  total_requests: number;
  rps: number;
  error_rate: number;
  strategies: LoadTestStrategy[];
}

export interface SummaryResponse {
  available: boolean;
  message: string | null;
  dataset: string | null;
  generated_at: string | null;
  n_queries: number | null;
  environment: Record<string, string>;
  strategies: StrategyMetrics[];
  index_stats: IndexStats | null;
  pareto: ParetoPoint[];
  load_test: LoadTestSummary | null;
}

export interface Bm25GridCell {
  k1: number;
  b: number;
  ndcg_at_10: number;
  recall_at_100: number;
  mean_ms: number;
}

export interface ChunkRow {
  chunk_size: number | null;
  ndcg_at_10: number;
  recall_at_100: number;
  n_chunks: number;
  index_bytes: number;
  build_seconds: number;
  mean_ms: number;
}

export interface RrfRow {
  rrf_k: number;
  ndcg_at_10: number;
  recall_at_100: number;
}

export interface AlphaRow {
  norm: Norm;
  alpha: number;
  ndcg_at_10: number;
  recall_at_100: number;
}

export interface HnswRow {
  m: number;
  ef_construct: number;
  ef: number;
  recall_at_10_vs_exact: number;
  ndcg_at_10: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
}

export interface RerankRow {
  rerank_top_k: number;
  ndcg_at_10: number;
  recall_at_100: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  total_mean_ms: number | null;
  total_p95_ms: number | null;
}

export interface AblationsResponse {
  available: boolean;
  message: string | null;
  dataset: string | null;
  generated_at: string | null;
  n_queries: number | null;
  bm25_grid: Bm25GridCell[];
  chunk_size: ChunkRow[];
  rrf_k: RrfRow[];
  alpha_sweep: AlphaRow[];
  hnsw: HnswRow[];
  rerank_top_k: RerankRow[];
  notes: string[];
}

export interface CacheStats {
  enabled: boolean;
  redis_ok: boolean;
  embedding_hits: number;
  embedding_misses: number;
  result_hits: number;
  result_misses: number;
  embedding_hit_rate: number | null;
  result_hit_rate: number | null;
  keys: number | null;
  used_memory_bytes: number | null;
}

export interface CheckResult {
  ok: boolean;
  detail: string;
}

export interface ReadyResponse {
  ready: boolean;
  dataset: string;
  dense_backend: string;
  checks: Record<string, CheckResult>;
}

export interface HealthResponse {
  status: "ok";
}

export interface EvalTriggerRequest {
  include_ablations: boolean;
  max_queries: number | null;
}

export interface EvalStatus {
  state: "idle" | "running" | "succeeded" | "failed";
  started_at: string | null;
  finished_at: string | null;
  message: string | null;
  include_ablations: boolean;
}
