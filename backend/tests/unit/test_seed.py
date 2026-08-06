"""Seed logic: inserts baselines, is idempotent, never clobbers operator edits."""
from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select

from app.db import dispose_db, init_db, session
from app.models import ModelPrice
from app.seed import BASELINE_PRICES, seed_model_prices


@pytest.fixture(autouse=True)
async def _fresh_db():
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


async def test_seed_inserts_all_baselines_on_empty_db():
    inserted = await seed_model_prices()
    assert inserted == len(BASELINE_PRICES)

    async with session() as s:
        rows = {p.model_id: p for p in (await s.execute(select(ModelPrice))).scalars()}
    assert set(rows) == set(BASELINE_PRICES)
    # Spot-check hand-copied rate pairs, one paid + one free-tier.
    assert rows["gpt-4o"].usd_per_1k_input == Decimal("0.0025")
    assert rows["gpt-4o"].usd_per_1k_output == Decimal("0.01")
    assert rows["llama-3.1-8b-instant"].usd_per_1k_input == Decimal("0.00005")
    assert rows["llama3"].usd_per_1k_output == Decimal("0")


async def test_seed_is_idempotent():
    assert await seed_model_prices() == len(BASELINE_PRICES)
    assert await seed_model_prices() == 0  # second run: nothing to do
    async with session() as s:
        count = len((await s.execute(select(ModelPrice))).scalars().all())
    assert count == len(BASELINE_PRICES)


async def test_seed_never_overwrites_operator_edits():
    # Operator corrected a price by hand...
    async with session() as s:
        s.add(
            ModelPrice(
                model_id="gpt-4o-mini",
                usd_per_1k_input=Decimal("0.00099"),
                usd_per_1k_output=Decimal("0.00199"),
                source="operator manual fix",
            )
        )

    # ...then the container restarts and reseeds.
    inserted = await seed_model_prices()
    assert inserted == len(BASELINE_PRICES) - 1  # only the missing baselines

    async with session() as s:
        row = await s.get(ModelPrice, "gpt-4o-mini")
    assert row.usd_per_1k_input == Decimal("0.00099")  # edit survived
    assert row.source == "operator manual fix"
