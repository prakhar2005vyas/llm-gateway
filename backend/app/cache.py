"""Semantic cache — lookup on the hot path, population on the cold path.

Pipeline (order is the PII invariant, SPEC §6):
    messages → serialize → pii.mask_prompt() → get_embedding() → search/store
Nothing may embed or store prompt text that hasn't been through mask_prompt.
This module is the single choke point for both directions.

Cacheability rules (documented correctness decisions, not accidents):
* Non-streaming requests only for LOOKUP in Phase 3 — a cache hit is a JSON
  body; serving it to a client that asked for SSE would break the contract.
  Streamed responses still POPULATE the cache (their reassembled text can
  serve future non-streaming duplicates).
* Deterministic requests only: temperature must be explicitly set and
  ≤ CACHE_MAX_TEMPERATURE. OpenAI defaults omitted temperature to 1.0, and
  freezing one answer for a creative call is the same correctness bug SPEC
  names for coalescing — so "no temperature" means "not cacheable".
* Same model_id only, exact match: a gpt-4o-mini answer must never serve a
  gpt-4o request, however similar the prompts.
* Only successful, complete responses are stored: status 200, outcome ok,
  finish_reason == "stop", non-empty content. No tool-call or truncated
  responses in the cache.

Similarity: pgvector cosine distance (`<=>`), hit iff 1 - distance >=
CACHE_SIMILARITY_THRESHOLD (0.90, tuned empirically — see config.py for the
measured pair scores and the documented direction-flip limitation).

On SQLite (unit tests) there is no `<=>`; lookup falls back to a brute-force
Python dot product over same-model rows. Same semantics, O(n) — test-scale
only. The real ANN path is exercised by the pgvector integration tests.

Failure contract: a broken cache must never break proxying. Every public
function catches its own exceptions, logs loudly, and degrades: lookup → miss,
store → skipped. (Same boundary rule as tracing.py.)
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from .config import get_settings
from .db import get_engine, session
from .embeddings import get_embedding
from .models import SemanticCache
from .pii import mask_prompt

logger = logging.getLogger(__name__)


@dataclass
class CacheDecision:
    """What the hot path learned from one cache consultation."""

    cacheable: bool
    masked_prompt: str | None = None
    embedding: list[float] | None = None  # reused by the cold path store
    hit: SemanticCache | None = None
    similarity: float | None = None


def serialize_messages(body: dict) -> str | None:
    """Canonical prompt text for embedding: every message, role-tagged.

    The whole conversation is the cache key — a different system prompt must
    produce a different embedding. Non-string content (multimodal parts)
    makes the request uncacheable rather than half-serialized.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            return None
        lines.append(f"{m.get('role', 'user')}: {m['content']}")
    return "\n".join(lines)


def is_cacheable_request(body: dict) -> bool:
    """Deterministic, plain-text, known-model requests only (see module doc)."""
    settings = get_settings()
    if not settings.semantic_cache_enabled:
        return False
    temperature = body.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        return False  # omitted temperature defaults to 1.0 upstream
    if temperature > settings.cache_max_temperature:
        return False
    if not isinstance(body.get("model"), str):
        return False
    return serialize_messages(body) is not None


async def consult(body: dict) -> CacheDecision:
    """Hot-path lookup: mask → embed → nearest-neighbor → threshold check.

    Returns a CacheDecision; `hit` is set iff a same-model row clears the
    similarity threshold. Never raises — cache failure degrades to a miss.
    """
    if not is_cacheable_request(body):
        return CacheDecision(cacheable=False)

    try:
        masked = mask_prompt(serialize_messages(body))  # type: ignore[arg-type]
        embedding = await get_embedding(masked)
        row, similarity = await _nearest(body["model"], embedding)

        threshold = get_settings().cache_similarity_threshold
        if row is not None and similarity is not None and similarity >= threshold:
            logger.info(
                "cache HIT (similarity=%.4f >= %.2f, entry=%s)",
                similarity, threshold, row.id,
            )
            return CacheDecision(
                cacheable=True, masked_prompt=masked, embedding=embedding,
                hit=row, similarity=similarity,
            )
        logger.debug(
            "cache miss (best similarity=%s, threshold=%.2f)",
            f"{similarity:.4f}" if similarity is not None else "none",
            threshold,
        )
        return CacheDecision(cacheable=True, masked_prompt=masked, embedding=embedding)
    except Exception as exc:  # noqa: BLE001 — cache must never break proxying
        logger.error(
            "CACHE LOOKUP FAILED (degrading to miss, request unaffected): %s: %s",
            type(exc).__name__, exc,
        )
        return CacheDecision(cacheable=False)


async def _nearest(
    model_id: str, embedding: list[float]
) -> tuple[SemanticCache | None, float | None]:
    """Closest same-model entry and its cosine similarity (1 - distance)."""
    if get_engine().dialect.name == "postgresql":
        dist = SemanticCache.prompt_embedding.cosine_distance(embedding).label("dist")
        stmt = (
            select(SemanticCache, dist)
            .where(SemanticCache.model_id == model_id)
            .order_by(dist)
            .limit(1)
        )
        async with session() as s:
            found = (await s.execute(stmt)).first()
        if found is None:
            return None, None
        row, distance = found
        return row, 1.0 - float(distance)

    # SQLite fallback (tests): brute-force dot product; vectors are normalized.
    async with session() as s:
        rows = list(
            (await s.execute(
                select(SemanticCache).where(SemanticCache.model_id == model_id)
            )).scalars().all()
        )
    best_row, best_sim = None, None
    for row in rows:
        sim = sum(a * b for a, b in zip(embedding, row.prompt_embedding))
        if best_sim is None or sim > best_sim:
            best_row, best_sim = row, sim
    return best_row, best_sim


def is_storable_response(response_body: dict | None) -> bool:
    """Only complete, successful text completions enter the cache."""
    if not isinstance(response_body, dict):
        return False
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    c0 = choices[0]
    if not isinstance(c0, dict) or c0.get("finish_reason") != "stop":
        return False
    message = c0.get("message")
    return isinstance(message, dict) and bool(message.get("content"))


async def store(
    *,
    model_id: str,
    masked_prompt: str,
    embedding: list[float],
    response_body: dict,
) -> None:
    """Cold-path population. Upserts on exact (model, masked_prompt) match so
    repeated misses refresh one row instead of accumulating duplicates.
    Never raises (same boundary contract as record_trace)."""
    try:
        async with session() as s:
            existing = (
                await s.execute(
                    select(SemanticCache)
                    .where(
                        SemanticCache.model_id == model_id,
                        SemanticCache.masked_prompt == masked_prompt,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is not None:
                existing.response_body = response_body
                existing.prompt_embedding = embedding
            else:
                s.add(
                    SemanticCache(
                        model_id=model_id,
                        masked_prompt=masked_prompt,
                        prompt_embedding=embedding,
                        response_body=response_body,
                    )
                )
        logger.debug("cache stored (model=%s)", model_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "CACHE STORE FAILED (entry skipped, client unaffected): %s: %s",
            type(exc).__name__, exc,
        )


async def record_hit(entry_id: uuid.UUID) -> None:
    """Cold-path hit bookkeeping: bump hit_count / last_hit_at. Never raises."""
    try:
        async with session() as s:
            row = await s.get(SemanticCache, entry_id)
            if row is not None:
                row.hit_count += 1
                row.last_hit_at = datetime.now(timezone.utc)
    except Exception as exc:  # noqa: BLE001
        logger.error("cache hit-count update failed (non-fatal): %s: %s",
                     type(exc).__name__, exc)
