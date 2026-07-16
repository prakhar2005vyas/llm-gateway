"""Relational schema — Trace + ModelPrice + SemanticCache.

ER decisions (deliberate, see also the module-level notes in db.py):

* `Trace.model_id` is intentionally NOT a foreign key to `ModelPrice`.
  The gateway is a transparent proxy: it must log every request it forwards,
  including requests for models we have no price row for yet. A hard FK would
  make trace logging fail on unknown models — the observability layer would
  drop data exactly when it's most interesting. Cost is computed at write time
  by *looking up* ModelPrice; a missing price yields cost=NULL (an honest
  "unknown"), never a rejected row. This is a soft reference by value, and the
  join is still cheap because both sides are indexed.

* JSONB (via SQLAlchemy's JSON with a Postgres JSONB variant) for request/
  response bodies — they're variable-shape OpenAI payloads; forcing them into
  columns would be a lie. Aggregatable facts (tokens, cost, latency, status)
  are first-class typed columns so analytics never has to dig into JSON.

* Indexes: `created_at` (time-range dashboards), `model_id` (per-model cost/
  usage), and a composite (model_id, created_at) for the "cost of model X over
  time" query the Grafana dashboards will run constantly.

* Prices are Numeric, not Float — money in floats accumulates rounding error;
  cost figures are a headline resume number, so they must be exact.

* SemanticCache.prompt_embedding is a pgvector `vector(N)` on Postgres and
  degrades to JSON (list of floats) on SQLite so unit tests stay hermetic.
  SQLite obviously can't do ANN similarity search — nearest-neighbor queries
  are exercised against real Postgres; SQLite covers schema/round-trip only.
  The dimension comes from settings (all-MiniLM = 384), resolved at class
  definition time — changing it later means re-embedding the cache.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

from .config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    # JSON columns become JSONB on Postgres, plain JSON elsewhere (tests run
    # on SQLite). Same model code, right type per backend.
    type_annotation_map = {
        dict: JSON().with_variant(JSONB(), "postgresql"),
    }


class Trace(Base):
    """One row per proxied request — the atom of observability.

    Written by the cold path (BackgroundTask), off the hot path. Columns are
    nullable where the fact may legitimately be unknown (e.g. cost when we
    have no price row, usage when the upstream omitted it) — NULL means
    "unknown", never "zero". That distinction keeps aggregates honest.
    """

    __tablename__ = "traces"

    # UUID PK generated app-side: the cold path can know the trace id before
    # the row lands, and ids are globally unique across future replicas.
    # SQLAlchemy's dialect-agnostic Uuid: native uuid on Postgres, CHAR(32)
    # on SQLite — same model code on both.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    # --- what was asked and what came back (variable shape → JSONB) ----------
    request_body: Mapped[dict | None] = mapped_column(nullable=True)
    response_body: Mapped[dict | None] = mapped_column(nullable=True)

    # --- aggregatable facts (typed columns, never buried in JSON) ------------
    model_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Streaming only: time-to-first-token, the Phase 2 headline metric.
    # NULL for non-streamed requests.
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # USD. Numeric(12,8): six decimal places is not enough for per-request
    # costs at $0.0000004/token scale; 8 keeps sub-cent precision exact.
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 8), nullable=True)

    # True when this request was served from the semantic cache: zero
    # upstream calls, cost_usd is an honest 0 (actually free, not unknown).
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # True when this request was a coalesced FOLLOWER sharing the leader's
    # single upstream call. Cost/tokens are 0/NULL on followers — the leader's
    # trace carries the real spend exactly once, so aggregates stay truthful.
    coalesced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Honest failure surface: 'ok' | 'upstream_error' | 'inconclusive' — a
    # trace that failed mid-flight is still a trace (never silently dropped).
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="ok")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # The dashboard's bread-and-butter query: model X over time window Y.
        Index("ix_traces_model_created", "model_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Trace {self.id} model={self.model_id} cost={self.cost_usd}>"


class ModelPrice(Base):
    """Price table: model id → $/1K input tokens, $/1K output tokens.

    `model_id` is the natural primary key — provider model ids are unique,
    stable identifiers and every lookup is by exact id. A surrogate integer
    PK would add a column and an index for nothing.
    """

    __tablename__ = "model_prices"

    model_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    usd_per_1k_input: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    usd_per_1k_output: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    # Where this price came from (provider pricing page / manual entry) and
    # when it was last verified — prices drift, and a wrong price silently
    # corrupts every cost number downstream.
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ModelPrice {self.model_id} in={self.usd_per_1k_input}"
            f" out={self.usd_per_1k_output}>"
        )


class Eval(Base):
    """One LLM-as-judge quality verdict for one sampled Trace (Phase 6).

    `trace_id` is a REAL foreign key — the deliberate opposite of the
    Trace→ModelPrice soft reference: an eval row without its trace is
    meaningless noise, so referential enforcement is correct here.

    Honesty note (SPEC): LLM-as-judge is NOISY. Scores are one sampled
    model's opinion, stored with the evaluator's identity and raw verdict so
    the noise is inspectable, never laundered into false precision.
    """

    __tablename__ = "evals"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # 1-10 judge scores. Nullable: a failed/skipped eval has no scores, and
    # NULL means "no verdict", never "scored zero".
    helpfulness_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), nullable=True)
    tone_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2), nullable=True)

    evaluator_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # pending → completed | failed | skipped (skipped = no evaluator
    # configured / shed under load — visible, not silently dropped).
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The judge's raw JSON verdict, for auditing the noise.
    raw_verdict: Mapped[dict | None] = mapped_column(nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Eval {self.id} trace={self.trace_id} status={self.status}>"


class SemanticCache(Base):
    """One cached completion, addressable by prompt similarity (Phase 3).

    THE PII INVARIANT (SPEC §6, cross-cutting rule): `masked_prompt` and
    `prompt_embedding` are derived from the prompt AFTER `pii.mask_prompt()`
    has run. Raw prompts must never be written here — otherwise the cache IS
    a PII database and the masking feature is defeated before it exists.
    Nothing in this table may bypass that path.

    Cache key design: the embedding is the fuzzy key (similarity search);
    `model_id` is an exact filter — an answer from gpt-4o-mini must never be
    served for a gpt-4o request, however similar the prompts. The cached
    `response_body` is the full OpenAI-shaped JSON so a hit can be returned
    to the client indistinguishably from a real completion.
    """

    __tablename__ = "semantic_cache"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    # Masked prompt text — kept for debugging/eviction and adversarial-miss
    # analysis ("what did this vector actually say?").
    masked_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    # vector(N) on Postgres; JSON float list on SQLite (round-trip only).
    prompt_embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dim).with_variant(JSON(), "sqlite"),
        nullable=False,
    )

    # Exact-match constraint on hits: same model only.
    model_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    # The complete OpenAI-shaped response a hit serves back.
    response_body: Mapped[dict] = mapped_column(nullable=False)

    # Ops facts for the cache-hit-rate story.
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SemanticCache {self.id} model={self.model_id} hits={self.hit_count}>"
