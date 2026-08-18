"""Cost calculation — token-based pricing, exact to the token.

Pure business logic: no DB, no I/O. The caller (Phase 1's cold-path trace
writer) fetches the ModelPrice row / price map; these functions only do math.

Money rules:
* Everything is Decimal end-to-end. Floats are rejected loudly rather than
  silently accepted — a float rate like 0.00015 is already inexact before we
  multiply it by a million tokens, and cost is a headline number.
* Results are quantized to 8 decimal places (ROUND_HALF_UP) to match the
  Numeric(12, 8) storage column exactly — the number we compute is the number
  the DB holds, byte for byte.
* Unknown model → None ("cost unknown") + a warning log. NOT zero: the Trace
  schema defines cost_usd NULL as "unknown" and 0 as "actually free".
  Defaulting to zero would silently understate every aggregate the resume
  quotes. Honest NULL over comfortable wrong number.
"""
from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import Mapping, Protocol

logger = logging.getLogger(__name__)

# Matches Numeric(12, 8) in models.py — keep in lockstep.
_CENT_PRECISION = Decimal("0.00000001")

_TOKENS_PER_UNIT = Decimal(1000)  # prices are quoted per 1K tokens

# Flat-rate / GPU-time billed models that do not have a per-token price.
# cost_for_model will return None for these without logging a warning.
UNMETERED_MODELS = frozenset(["gemma4:31b"])


class PriceLike(Protocol):
    """Anything carrying the two per-1K rates (the ORM ModelPrice qualifies)."""

    usd_per_1k_input: Decimal
    usd_per_1k_output: Decimal


def compute_cost_usd(
    usd_per_1k_input: Decimal,
    usd_per_1k_output: Decimal,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """Exact cost of one request given the two per-1K rates.

    cost = prompt/1000 * in_rate + completion/1000 * out_rate,
    quantized to 8 dp (the storage precision).

    Raises:
        TypeError: if a rate arrives as float (silent inexactness — refuse).
        ValueError: on negative token counts (upstream corruption — refuse).
    """
    for name, rate in (
        ("usd_per_1k_input", usd_per_1k_input),
        ("usd_per_1k_output", usd_per_1k_output),
    ):
        if isinstance(rate, float):
            raise TypeError(
                f"{name} must be Decimal, got float ({rate!r}) — "
                "float money is inexact by construction"
            )
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError(
            f"token counts cannot be negative "
            f"(prompt={prompt_tokens}, completion={completion_tokens})"
        )

    raw = (
        Decimal(prompt_tokens) / _TOKENS_PER_UNIT * usd_per_1k_input
        + Decimal(completion_tokens) / _TOKENS_PER_UNIT * usd_per_1k_output
    )
    return raw.quantize(_CENT_PRECISION, rounding=ROUND_HALF_UP)


def cost_for_model(
    model_id: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    prices: Mapping[str, PriceLike],
) -> Decimal | None:
    """Cost for a request, or None when it genuinely cannot be known.

    None (never zero) when:
      * the model has no price row — logged as a warning so a missing price
        is visible in ops, not buried;
      * token counts are missing (upstream omitted `usage`) — the cost of an
        unmetered request is unknown, not free.
    """
    if model_id is None or model_id not in prices:
        if model_id in UNMETERED_MODELS:
            logger.info(
                "cost intentionally unknown: %r is a flat-rate provider, not token-metered",
                model_id,
            )
        else:
            logger.warning(
                "no price row for model %r — recording cost as unknown (NULL). "
                "Add it to model_prices to keep cost aggregates complete.",
                model_id,
            )
        return None

    if prompt_tokens is None or completion_tokens is None:
        logger.warning(
            "usage missing for model %r (prompt=%r, completion=%r) — "
            "cost recorded as unknown (NULL)",
            model_id,
            prompt_tokens,
            completion_tokens,
        )
        return None

    price = prices[model_id]
    return compute_cost_usd(
        price.usd_per_1k_input,
        price.usd_per_1k_output,
        prompt_tokens,
        completion_tokens,
    )
