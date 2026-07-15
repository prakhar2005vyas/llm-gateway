"""Cost calculator tests — every expected value below is hand-computed.

The arithmetic shown in each comment is the ground truth; if a test fails,
the code is wrong, not the comment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

import pytest

from app.costs import compute_cost_usd, cost_for_model


@dataclass
class FakePrice:
    """Stands in for the ORM ModelPrice — the calculator only needs the rates."""

    usd_per_1k_input: Decimal
    usd_per_1k_output: Decimal


# Real-world-shaped price book (rates as of the 2026-07 pricing pages).
PRICES = {
    # gpt-4o-mini: $0.15 / 1M input = $0.00015/1K; $0.60 / 1M output = $0.0006/1K
    "gpt-4o-mini": FakePrice(Decimal("0.00015"), Decimal("0.0006")),
    # a chunkier model with round numbers for easy mental math
    "big-model": FakePrice(Decimal("0.01"), Decimal("0.03")),
    # an actually-free model: cost 0 is CORRECT for it (vs None for unknown)
    "free-model": FakePrice(Decimal("0"), Decimal("0")),
}


# ---------------------------------------------------------------------------
# compute_cost_usd — pure math against hand-computed values
# ---------------------------------------------------------------------------

class TestComputeCost:
    def test_simple_round_numbers(self):
        # 1000 in @ $0.01/1K  = 0.01
        # 2000 out @ $0.03/1K = 0.06
        # total               = 0.07
        got = compute_cost_usd(Decimal("0.01"), Decimal("0.03"), 1000, 2000)
        assert got == Decimal("0.07000000")

    def test_gpt4o_mini_realistic_call(self):
        # 9 in @ $0.00015/1K   = 9/1000 * 0.00015  = 0.00000135
        # 3 out @ $0.0006/1K   = 3/1000 * 0.0006   = 0.0000018
        # total                                    = 0.00000315
        got = compute_cost_usd(Decimal("0.00015"), Decimal("0.0006"), 9, 3)
        assert got == Decimal("0.00000315")

    def test_single_token_each_way(self):
        # 1 in  @ $0.00015/1K = 0.00000015
        # 1 out @ $0.0006/1K  = 0.00000060
        # total               = 0.00000075
        got = compute_cost_usd(Decimal("0.00015"), Decimal("0.0006"), 1, 1)
        assert got == Decimal("0.00000075")

    def test_large_context_no_float_drift(self):
        # 1,000,000 in @ $0.00015/1K = 1000 * 0.00015 = 0.15 exactly
        # 250,000 out @ $0.0006/1K   = 250 * 0.0006   = 0.15 exactly
        # total                                       = 0.30 exactly
        # (in float this arithmetic already drifts; Decimal must not)
        got = compute_cost_usd(Decimal("0.00015"), Decimal("0.0006"), 1_000_000, 250_000)
        assert got == Decimal("0.30000000")

    def test_zero_tokens_costs_zero(self):
        got = compute_cost_usd(Decimal("0.00015"), Decimal("0.0006"), 0, 0)
        assert got == Decimal("0.00000000")

    def test_rounding_is_half_up_at_8dp(self):
        # 1 in @ $0.000005/1K = 0.000000005 → 9th dp is a 5 → rounds UP at 8dp
        # to 0.00000001 (ROUND_HALF_UP, matching Numeric(12,8) storage).
        got = compute_cost_usd(Decimal("0.000005"), Decimal("0"), 1, 0)
        assert got == Decimal("0.00000001")

    def test_result_precision_matches_storage_column(self):
        # Whatever the inputs, the result always has exactly 8 dp — identical
        # to Numeric(12, 8), so DB round-trips can never change the value.
        got = compute_cost_usd(Decimal("0.00015"), Decimal("0.0006"), 7, 13)
        assert got == got.quantize(Decimal("0.00000001"))

    def test_float_rates_are_rejected(self):
        with pytest.raises(TypeError, match="float"):
            compute_cost_usd(0.00015, Decimal("0.0006"), 10, 10)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="float"):
            compute_cost_usd(Decimal("0.00015"), 0.0006, 10, 10)  # type: ignore[arg-type]

    def test_negative_tokens_are_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            compute_cost_usd(Decimal("0.01"), Decimal("0.03"), -1, 10)
        with pytest.raises(ValueError, match="negative"):
            compute_cost_usd(Decimal("0.01"), Decimal("0.03"), 10, -1)


# ---------------------------------------------------------------------------
# cost_for_model — lookup wrapper + the "unknown ≠ zero" contract
# ---------------------------------------------------------------------------

class TestCostForModel:
    def test_known_model_hand_computed(self):
        # 100 in @ $0.00015/1K = 0.000015
        # 50 out @ $0.0006/1K  = 0.00003
        # total                = 0.000045
        got = cost_for_model("gpt-4o-mini", 100, 50, PRICES)
        assert got == Decimal("0.00004500")

    def test_unknown_model_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.costs"):
            got = cost_for_model("mystery-model-9000", 100, 50, PRICES)
        assert got is None  # unknown — NOT zero
        assert any("no price row" in r.message for r in caplog.records)

    def test_none_model_id_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="app.costs"):
            got = cost_for_model(None, 100, 50, PRICES)
        assert got is None

    def test_missing_usage_returns_none_and_warns(self, caplog):
        # Upstream omitted `usage` — cost is unknown, not free.
        with caplog.at_level(logging.WARNING, logger="app.costs"):
            assert cost_for_model("gpt-4o-mini", None, 50, PRICES) is None
            assert cost_for_model("gpt-4o-mini", 100, None, PRICES) is None
        assert sum("usage missing" in r.message for r in caplog.records) == 2

    def test_genuinely_free_model_is_zero_not_none(self):
        # The whole point of the None-vs-zero distinction, from the other side:
        # a model priced AT zero really does cost 0.00000000.
        got = cost_for_model("free-model", 5000, 5000, PRICES)
        assert got == Decimal("0.00000000")
        assert got is not None

    def test_known_model_zero_tokens(self):
        got = cost_for_model("gpt-4o-mini", 0, 0, PRICES)
        assert got == Decimal("0.00000000")
