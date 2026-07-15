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

    # --- Reliability knobs ---------------------------------------------------
    upstream_timeout_seconds: float = 60.0
    # Number of *retries* after the first attempt (so total tries = retries + 1).
    upstream_max_retries: int = 2

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

    # --- Server --------------------------------------------------------------
    log_level: str = "INFO"

    @property
    def chat_completions_url(self) -> str:
        """Fully-qualified upstream URL for the chat-completions endpoint."""
        return f"{self.upstream_base_url.rstrip('/')}/chat/completions"


@lru_cache
def get_settings() -> Settings:
    return Settings()
