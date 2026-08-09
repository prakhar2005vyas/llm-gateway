"""Central, env-driven configuration.

Everything model/endpoint/threshold-related is an env var — never hardcoded
(see CLAUDE.md). Settings are cached so the whole process shares one instance;
tests clear the cache via `get_settings.cache_clear()`.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Upstream provider (OpenAI-compatible) -------------------------------
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    # When set, every forwarded request has its `model` field overwritten with
    # this value. Handles provider migrations (e.g. Fireworks → Groq) without
    # requiring clients to update their model strings. Empty = passthrough.
    upstream_model_id: str = ""

    # --- Reliability knobs ---------------------------------------------------
    upstream_timeout_seconds: float = 60.0
    # Number of *retries* after the first attempt (so total tries = retries + 1).
    upstream_max_retries: int = 2

    # --- Failover (Phase 4) ----------------------------------------------------
    # Secondary OpenAI-compatible provider (same schema = swap base URL + key).
    # Used only after the primary exhausts ALL its retries with transport
    # errors or 5xx. Empty = failover disabled. Cross-schema (Anthropic)
    # failover is DOCUMENT-ONLY in SCALING.md.
    failover_base_url: str = ""
    failover_api_key: str = ""

    # --- Rate limiting (Phase 4) -------------------------------------------------
    # In-memory per-key sliding window (v1, single-process; distributed
    # counters are SCALING.md material). Key = client bearer token.
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0

    # --- Gateway auth (optional in Phase 0) ----------------------------------
    # Empty string => auth disabled (transparent pass-through).
    gateway_api_key: str = ""

    # --- Database (Phase 1) ---------------------------------------------------
    # Async SQLAlchemy URL. Postgres in real deployments (Neon-compatible);
    # tests override with sqlite+aiosqlite:// so they need zero infrastructure.
    database_url: str = "postgresql+asyncpg://gateway:gateway@localhost:5432/gateway"
    # Connection pool knobs — bounded on purpose. The cold path writes traces
    # through this pool; a bounded pool is part of the backpressure story
    # (SPEC.md: shed cold-path work rather than grow unboundedly).
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_seconds: float = 5.0
    # Recycle idle connections (Neon and other serverless PG close idle conns).
    db_pool_recycle_seconds: int = 300
    db_echo: bool = False
    # Hard ceiling on one cold-path trace write (Phase 7 frozen-DB fix). A
    # BackgroundTask runs inside the ASGI cycle: a write wedged on a frozen
    # DB holds that keep-alive connection hostage and the NEXT request queued
    # on it stalls behind the freeze. At the timeout the write is cancelled
    # and shed loudly — bounded cold path, per SPEC's backpressure rule.
    trace_write_timeout_seconds: float = 5.0

    # --- Semantic cache (Phase 3) ---------------------------------------------
    # Dimension of the embedding model's output. 384 = all-MiniLM-L6-v2 (the
    # SPEC-chosen local CPU model). NOTE: changing this after rows exist
    # requires dropping/re-embedding the cache — the column is dimensioned.
    embedding_dim: int = 384
    embedding_model_id: str = "all-MiniLM-L6-v2"
    semantic_cache_enabled: bool = True
    # Cosine similarity threshold for a hit. Tuned empirically (2026-07,
    # all-MiniLM-L6-v2): paraphrases score ~0.92+, the adversarial pairs we
    # must NOT collide ("delete my account"/"recover my account" 0.71,
    # "cancel"/"renew subscription" 0.77) sit well below 0.90. Known
    # limitation: entity/direction flips ("transfer TO"/"FROM savings", 0.99)
    # are inseparable by MiniLM embeddings at ANY threshold — documented in
    # the test suite, mitigated only by raising the threshold toward exact-
    # match territory.
    cache_similarity_threshold: float = 0.90
    # Correctness rule (same argument as SPEC's coalescing temperature rule):
    # a cached answer may only substitute a DETERMINISTIC request. Creative
    # (high-temperature) calls expect variety; freezing one answer for them
    # is a correctness bug. Requests without an explicit temperature default
    # to 1.0 upstream → not cacheable.
    cache_max_temperature: float = 0.2

    # --- Request coalescing (Phase 4) ------------------------------------------
    coalesce_enabled: bool = True
    # Same correctness rule as caching: only deterministic requests may share
    # one result. Explicit temperature required; omitted = 1.0 = bypass.
    coalesce_max_temperature: float = 0.2

    # --- Eval harness (Phase 6) -------------------------------------------------
    # Fraction of successful traffic sampled for LLM-as-judge scoring.
    # 0.0 = off (default: evals cost real tokens; opting in is explicit).
    eval_sample_rate: float = 0.0
    # OpenAI-compatible judge endpoint (e.g. Fireworks). Empty base URL =
    # evaluator unconfigured → sampled evals recorded as status='skipped'.
    eval_base_url: str = ""
    eval_api_key: str = ""
    eval_model_id: str = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    eval_timeout_seconds: float = 30.0
    eval_max_retries: int = 1
    # Max eval tasks in flight; beyond this, samples are SHED (bounded-queue
    # rule: drop cold-path work before it can pile up into an OOM).
    eval_max_concurrent: int = 4

    # --- Server --------------------------------------------------------------
    log_level: str = "INFO"

    @property
    def chat_completions_url(self) -> str:
        """Fully-qualified upstream URL for the chat-completions endpoint."""
        return f"{self.upstream_base_url.rstrip('/')}/chat/completions"


@lru_cache
def get_settings() -> Settings:
    return Settings()
