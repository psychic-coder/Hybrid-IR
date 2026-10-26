"""Application configuration.

Every setting is read from an environment variable of the same name (upper-cased).
Defaults are chosen so the stack runs on an 8 GB laptop with the ``scifact`` dataset.
See ``.env.example`` and README §12 for the meaning of each variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DatasetName = Literal["scifact", "fiqa", "nfcorpus"]
DenseBackend = Literal["qdrant", "custom"]
NormName = Literal["minmax", "zscore"]

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Settings(BaseSettings):
    """Typed view over the process environment."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    # --- data -------------------------------------------------------------------------
    dataset: DatasetName = "scifact"
    data_dir: Path = Path("/data")
    results_dir: Path = Path("/results")
    beir_base_url: str = BEIR_BASE_URL

    # --- services ---------------------------------------------------------------------
    qdrant_url: str = "http://qdrant:6333"
    qdrant_timeout_seconds: float = 10.0
    redis_url: str = "redis://redis:6379/0"
    redis_timeout_seconds: float = 1.0

    # --- models -----------------------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embed_batch_size: int = Field(default=32, ge=1, le=512)
    rerank_batch_size: int = Field(default=32, ge=1, le=512)
    rerank_max_length: int = Field(default=256, ge=16, le=512)
    torch_threads: int = Field(default=0, ge=0, le=64)

    # --- tokenizer / index ------------------------------------------------------------
    use_stemmer: bool = True
    use_stopwords: bool = True
    min_token_len: int = Field(default=1, ge=1, le=10)
    chunk_size: int = Field(default=256, ge=0, le=8192)
    chunk_overlap: int = Field(default=32, ge=0, le=4096)
    skip_interval: int = Field(default=64, ge=2, le=4096)

    # --- retrieval defaults -----------------------------------------------------------
    dense_backend: DenseBackend = "qdrant"
    bm25_k1: float = Field(default=1.2, ge=0.0, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    phrase_boost: float = Field(default=1.0, ge=0.0, le=100.0)
    hnsw_m: int = Field(default=16, ge=4, le=128)
    hnsw_ef_construct: int = Field(default=100, ge=8, le=1024)
    hnsw_ef: int = Field(default=64, ge=1, le=1024)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    fusion_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    fusion_norm: NormName = "minmax"
    candidate_depth: int = Field(default=100, ge=10, le=1000)
    rerank_top_k: int = Field(default=50, ge=1, le=200)
    eval_depth: int = Field(default=100, ge=10, le=1000)

    # --- cache ------------------------------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = Field(default=3600, ge=1)

    # --- api --------------------------------------------------------------------------
    admin_token: str = ""
    cors_origins: str = "http://localhost:3000"
    search_timeout_seconds: float = Field(default=30.0, gt=0)
    log_level: str = "INFO"

    # --- seed / eval ------------------------------------------------------------------
    force_reseed: bool = False
    seed_wait_timeout_seconds: int = Field(default=180, ge=1)
    eval_max_queries: int = Field(default=0, ge=0)
    eval_qps_queries: int = Field(default=100, ge=1)
    eval_qps_threads: int = Field(default=4, ge=1, le=64)
    eval_warm_pass: bool = True
    ablations_on_seed: bool = True
    ablation_max_queries: int = Field(default=0, ge=0)
    ablation_latency_sample: int = Field(default=50, ge=1)
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    random_seed: int = 42
    k6_query_sample: int = Field(default=200, ge=1)

    # ------------------------------------------------------------------ derived paths
    @property
    def dataset_dir(self) -> Path:
        """Directory holding everything derived from the active dataset."""
        return self.data_dir / self.dataset

    @property
    def raw_dir(self) -> Path:
        """Extracted BEIR files (corpus.jsonl, queries.jsonl, qrels/)."""
        return self.dataset_dir / "raw"

    @property
    def index_dir(self) -> Path:
        """Inverted index directory (postings.bin + meta.json)."""
        return self.dataset_dir / "index"

    @property
    def embeddings_path(self) -> Path:
        """Chunk-level float32 embedding matrix (row i == internal chunk id i)."""
        return self.dataset_dir / "embeddings.npy"

    @property
    def custom_hnsw_path(self) -> Path:
        """Serialized from-scratch HNSW graph."""
        return self.dataset_dir / "custom_hnsw.npz"

    @property
    def marker_path(self) -> Path:
        """Seed completion marker (idempotency)."""
        return self.dataset_dir / ".seed_complete.json"

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins parsed from a comma separated string."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def collection_name(self, m: int | None = None, ef_construct: int | None = None) -> str:
        """Qdrant collection name for (dataset, M, ef_construct)."""
        mm = self.hnsw_m if m is None else m
        efc = self.hnsw_ef_construct if ef_construct is None else ef_construct
        return f"{self.dataset}_m{mm}_efc{efc}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
