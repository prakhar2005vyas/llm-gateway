"""Baseline price seeding — runs at startup, idempotent.

Strategy: insert-if-missing, never overwrite. If an operator has corrected a
price in the DB, a container restart must not clobber it with the baseline.
(That's also why this isn't a blind upsert: seed data is a floor, not truth.)

Prices are $/1K tokens, Decimal end-to-end. Source column records where the
number came from and when, because a silently-stale price corrupts every cost
aggregate downstream.
"""
from __future__ import annotations

import logging
from decimal import Decimal

from sqlalchemy import select

from .db import session
from .models import ModelPrice

logger = logging.getLogger(__name__)

# $/1M on the provider pricing pages ÷ 1000 → $/1K here.
#   gpt-4o-mini:   $0.15/1M in,  $0.60/1M out
#   gpt-4o:        $2.50/1M in, $10.00/1M out
#   gpt-3.5-turbo: $0.50/1M in,  $1.50/1M out
# Free-tier providers (all OpenAI-compatible — see .env.example recipes).
# Groq/Gemini free tiers BILL $0, but traces record the provider's list
# price so the "what would this traffic cost" story stays visible; Ollama
# runs locally and is genuinely $0. Rates hand-copied 2026-08 — verify
# against the pricing pages before quoting cost aggregates.
#   llama-3.1-8b-instant (Groq):    $0.05/1M in, $0.08/1M out
#   llama-3.3-70b-versatile (Groq): $0.59/1M in, $0.79/1M out
#   gemini-1.5-flash:               $0.075/1M in, $0.30/1M out
#   gemini-2.0-flash:               $0.10/1M in, $0.40/1M out
#   llama3 (Ollama, local):         $0
BASELINE_PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "gpt-4o": (Decimal("0.0025"), Decimal("0.01")),
    "gpt-3.5-turbo": (Decimal("0.0005"), Decimal("0.0015")),
    "llama-3.1-8b-instant": (Decimal("0.00005"), Decimal("0.00008")),
    "llama-3.3-70b-versatile": (Decimal("0.00059"), Decimal("0.00079")),
    "gemini-1.5-flash": (Decimal("0.000075"), Decimal("0.0003")),
    "gemini-2.0-flash": (Decimal("0.0001"), Decimal("0.0004")),
    "llama3": (Decimal("0"), Decimal("0")),
}

_SEED_SOURCE = "baseline seed (provider pricing pages, 2026-08)"


async def seed_model_prices() -> int:
    """Insert any baseline price rows that don't exist yet.

    Returns the number of rows inserted (0 on an already-seeded DB).
    Existing rows are left untouched — operator edits win over the baseline.
    """
    async with session() as s:
        existing = set(
            (await s.execute(select(ModelPrice.model_id))).scalars().all()
        )
        inserted = 0
        for model_id, (usd_in, usd_out) in BASELINE_PRICES.items():
            if model_id in existing:
                continue
            s.add(
                ModelPrice(
                    model_id=model_id,
                    usd_per_1k_input=usd_in,
                    usd_per_1k_output=usd_out,
                    source=_SEED_SOURCE,
                )
            )
            inserted += 1

    if inserted:
        logger.info("seeded %d baseline model price(s)", inserted)
    else:
        logger.debug("model_prices already seeded — nothing to do")
    return inserted
