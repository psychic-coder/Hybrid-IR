"""Pydantic v2 request/response models. Mirrored 1:1 by ``frontend/src/lib/types.ts``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal[
    "bm25", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_rrf_rerank", "hybrid_weighted_rerank"
]
Norm = Literal["minmax", "zscore"]
DEFAULT_COMPARE: list[Strategy] = ["bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank"]


class _Model(BaseModel):
    """Strict base for request/response bodies built by the API itself."""

    model_config = ConfigDict(extra="forbid")


class _FileModel(BaseModel):
    """Lenient base for payloads read from results/*.json (unknown keys are dropped)."""

    model_config = ConfigDict(extra="ignore")


# ----------------------------------------------------------------------------- search
class SearchParams(_Model):
    """Optional per-request overrides; omitted fields use the server defaults."""

    k1: float | None = Field(default=None, ge=0.0, le=5.0, description="BM25 term saturation")
    b: float | None = Field(default=None, ge=0.0, le=1.0, description="BM25 length normalization")
    rrf_k: int | None = Field(default=None, ge=1, le=1000, description="RRF rank constant")
    alpha: float | None = Field(default=None, ge=0.0, le=1.0, description="dense weight")
    norm: Norm | None = Field(default=None, description="score normalization for weighted fusion")
    rerank_top_k: int | None = Field(default=None, ge=1, le=200, description="docs to rerank")
    ef: int | None = Field(default=None, ge=1, le=1024, description="HNSW query-time ef")
    candidate_depth: int | None = Field(default=None, ge=10, le=1000, description="per-retriever depth")


class EffectiveParamsModel(_Model):
    """Parameters actually used for the request."""

    k1: float
    b: float
    rrf_k: int
    alpha: float
    norm: Norm
    rerank_top_k: int
    ef: int
    candidate_depth: int


class SearchRequest(_Model):
    """Body of POST /search."""

    query: str = Field(min_length=1, max_length=512)
    strategy: Strategy = "hybrid_rrf"
    top_k: int = Field(default=10, ge=1, le=100)
    params: SearchParams = Field(default_factory=SearchParams)
    use_cache: bool = True


class StageScores(_Model):
    """Score each stage assigned to a document (null if the stage did not see it)."""

    bm25: float | None = None
    dense: float | None = None
    fusion: float | None = None
    rerank: float | None = None


class StageRanks(_Model):
    """1-based rank of a document in each stage's list (null if absent)."""

    bm25: int | None = None
    dense: int | None = None
    fusion: int | None = None
    rerank: int | None = None


class SearchHit(_Model):
    """One ranked result."""

    rank: int
    doc_id: str
    title: str
    snippet: str
    score: float
    scores: StageScores
    ranks: StageRanks
    chunk_id: int | None = None


class Timings(_Model):
    """Per-stage latency in milliseconds."""

    tokenize_ms: float
    bm25_ms: float
    embed_ms: float
    ann_ms: float
    fusion_ms: float
    rerank_ms: float
    total_ms: float


class CacheFlags(_Model):
    """Which caches served this request."""

    result_cache_hit: bool
    embedding_cache_hit: bool | None


class SearchResponse(_Model):
    """Response of POST /search."""

    query: str
    strategy: Strategy
    top_k: int
    params: EffectiveParamsModel
    hits: list[SearchHit]
    total_candidates: int
    query_terms: list[str]
    timings: Timings
    cache: CacheFlags
    dense_backend: str


# ----------------------------------------------------------------------------- compare
class CompareRequest(_Model):
    """Body of POST /compare."""

    query: str = Field(min_length=1, max_length=512)
    strategies: list[Strategy] = Field(
        default_factory=lambda: list(DEFAULT_COMPARE),
        min_length=1,
        max_length=6,
    )
    top_k: int = Field(default=10, ge=1, le=50)
    params: SearchParams = Field(default_factory=SearchParams)


class ErrorInfo(_Model):
    """Structured error (also the body of every non-2xx response)."""

    error: str
    detail: str
    stage: str | None = None
    request_id: str | None = None


class StrategyResult(_Model):
    """One column of a comparison."""

    strategy: Strategy
    response: SearchResponse | None = None
    error: ErrorInfo | None = None


class RankShiftRow(_Model):
    """Ranks of one document across strategies (ranks within each strategy's top-100)."""

    doc_id: str
    title: str
    ranks: dict[str, int | None]
    delta_vs_bm25: dict[str, int | None]
    is_new: dict[str, bool]


class CompareResponse(_Model):
    """Response of POST /compare."""

    query: str
    strategies: list[Strategy]
    top_k: int
    baseline: Strategy
    results: list[StrategyResult]
    rank_shift: list[RankShiftRow]


# ----------------------------------------------------------------------------- explain
class TermContribution(_Model):
    """BM25 contribution of one query term to the explained document's best chunk."""

    term: str
    qtf: int
    tf: int
    df: int
    idf: float
    doc_len: int
    avg_doc_len: float
    length_norm: float
    tf_component: float
    score: float


class ListStatsModel(_Model):
    """Statistics of a candidate score list."""

    n: int
    min: float
    max: float
    mean: float
    std: float


class FusionMath(_Model):
    """Exact numbers used by RRF and weighted fusion for this document."""

    candidate_depth: int
    rrf_k: int
    bm25_rank: int | None
    dense_rank: int | None
    rrf_bm25_term: float
    rrf_dense_term: float
    rrf_score: float
    alpha: float
    norm: Norm
    bm25_raw: float | None
    dense_raw: float | None
    bm25_norm: float
    dense_norm: float
    bm25_list: ListStatsModel
    dense_list: ListStatsModel
    weighted_score: float


class HighlightSpan(_Model):
    """Character span [start, end) to highlight."""

    start: int
    end: int


class ExplainResponse(_Model):
    """Response of GET /explain."""

    query: str
    doc_id: str
    title: str
    text: str
    title_highlights: list[HighlightSpan]
    text_highlights: list[HighlightSpan]
    chunk_id: int | None
    query_terms: list[str]
    phrases: list[list[str]]
    params: EffectiveParamsModel
    bm25_score: float
    phrase_boost: float
    terms: list[TermContribution]
    cosine: float | None
    dense_error: str | None
    fusion: FusionMath | None
    rerank_score: float | None
    rerank_error: str | None
    ranks: dict[str, int | None]
    delta_vs_bm25: dict[str, int | None]


# ----------------------------------------------------------------------------- metrics
class LatencyStats(_FileModel):
    """Latency distribution in milliseconds."""

    n: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float


class Significance(_FileModel):
    """Paired bootstrap result vs the BM25 baseline."""

    baseline: str
    system: str
    metric: str
    n_queries: int
    n_resamples: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool


class StrategyMetrics(_FileModel):
    """Quality + latency for one evaluated configuration."""

    name: str
    strategy: Strategy
    params: EffectiveParamsModel
    n_queries: int
    metrics: dict[str, float]
    latency_cold: LatencyStats
    latency_warm: LatencyStats | None
    stage_means_ms: dict[str, float]
    qps: float | None
    qps_threads: int | None
    significance: Significance | None


class IndexStats(_FileModel):
    """Inverted index statistics (from results/index_stats.json)."""

    dataset: str
    n_docs: int
    n_chunks: int
    vocab_size: int
    avg_chunk_len: float
    total_postings: int
    total_positions: int
    postings_bytes: int
    index_bytes: int
    raw_bytes_estimate: int
    compression_ratio: float
    build_seconds: float
    peak_rss_mb: float
    chunk_size: int
    chunk_overlap: int
    skip_interval: int
    embedding_seconds: float | None = None
    n_vectors: int | None = None


class ParetoPoint(_FileModel):
    """One configuration in nDCG@10 vs p95-latency space."""

    label: str
    source: str
    ndcg_at_10: float
    p95_ms: float
    on_frontier: bool


class LoadTestStrategy(_FileModel):
    """k6 latency summary for one strategy tag."""

    strategy: str
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class LoadTestSummary(_FileModel):
    """Written by the k6 ``handleSummary`` hook."""

    script: str
    generated_at: str
    duration_s: float
    total_requests: int
    rps: float
    error_rate: float
    strategies: list[LoadTestStrategy]


class SummaryResponse(_FileModel):
    """Response of GET /metrics/summary."""

    available: bool
    message: str | None = None
    dataset: str | None = None
    generated_at: str | None = None
    n_queries: int | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    strategies: list[StrategyMetrics] = Field(default_factory=list)
    index_stats: IndexStats | None = None
    pareto: list[ParetoPoint] = Field(default_factory=list)
    load_test: LoadTestSummary | None = None


class Bm25GridCell(_FileModel):
    """One (k1, b) cell of the BM25 ablation."""

    k1: float
    b: float
    ndcg_at_10: float
    recall_at_100: float
    mean_ms: float


class ChunkRow(_FileModel):
    """Chunk-size ablation row (BM25 only); chunk_size null = no chunking."""

    chunk_size: int | None
    ndcg_at_10: float
    recall_at_100: float
    n_chunks: int
    index_bytes: int
    build_seconds: float
    mean_ms: float


class RrfRow(_FileModel):
    """RRF k ablation row."""

    rrf_k: int
    ndcg_at_10: float
    recall_at_100: float


class AlphaRow(_FileModel):
    """Weighted fusion alpha sweep row."""

    norm: Norm
    alpha: float
    ndcg_at_10: float
    recall_at_100: float


class HnswRow(_FileModel):
    """HNSW (M, ef) ablation row."""

    m: int
    ef_construct: int
    ef: int
    recall_at_10_vs_exact: float
    ndcg_at_10: float
    mean_ms: float
    p50_ms: float
    p95_ms: float


class RerankRow(_FileModel):
    """Rerank depth ablation row.

    ``mean_ms/p50_ms/p95_ms`` time the cross-encoder stage only; ``total_*`` time the whole
    hybrid_rrf_rerank request (used for the Pareto chart).
    """

    rerank_top_k: int
    ndcg_at_10: float
    recall_at_100: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    total_mean_ms: float | None = None
    total_p95_ms: float | None = None


class AblationsResponse(_FileModel):
    """Response of GET /metrics/ablations."""

    available: bool
    message: str | None = None
    dataset: str | None = None
    generated_at: str | None = None
    n_queries: int | None = None
    bm25_grid: list[Bm25GridCell] = Field(default_factory=list)
    chunk_size: list[ChunkRow] = Field(default_factory=list)
    rrf_k: list[RrfRow] = Field(default_factory=list)
    alpha_sweep: list[AlphaRow] = Field(default_factory=list)
    hnsw: list[HnswRow] = Field(default_factory=list)
    rerank_top_k: list[RerankRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CacheStats(_Model):
    """Response of GET /metrics/cache."""

    enabled: bool
    redis_ok: bool
    embedding_hits: int
    embedding_misses: int
    result_hits: int
    result_misses: int
    embedding_hit_rate: float | None
    result_hit_rate: float | None
    keys: int | None
    used_memory_bytes: int | None


# ----------------------------------------------------------------------------- health/admin
class HealthResponse(_Model):
    """Liveness."""

    status: Literal["ok"]


class CheckResult(_Model):
    """One readiness check."""

    ok: bool
    detail: str


class ReadyResponse(_Model):
    """Readiness: every dependency the default strategies need."""

    ready: bool
    dataset: str
    dense_backend: str
    checks: dict[str, CheckResult]


class EvalTriggerRequest(_Model):
    """Body of POST /admin/eval."""

    include_ablations: bool = False
    max_queries: int | None = Field(default=None, ge=1)


class EvalStatus(_Model):
    """Background evaluation status."""

    state: Literal["idle", "running", "succeeded", "failed"]
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = None
    include_ablations: bool = False
