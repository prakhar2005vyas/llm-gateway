"""Data access layer tests — schema, unit-of-work atomicity, round-trips.

Run against in-memory SQLite (aiosqlite): zero infrastructure, same model
code. The JSONB/Uuid columns degrade to JSON/CHAR(32) via SQLAlchemy's
dialect variants; Postgres-specific behavior is exercised in later phases'
integration environment where a real PG container exists.
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import inspect, select

from app import db as db_module
from app.config import get_settings
from app.db import dispose_db, init_db, session
from app.models import Base, ModelPrice, Trace


@pytest.fixture(autouse=True)
async def _fresh_db(monkeypatch):
    """Each test gets its own in-memory database and a clean engine."""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    get_settings.cache_clear()
    # Reset any engine a previous test created against a different URL.
    await dispose_db()
    await init_db()
    yield
    await dispose_db()
    get_settings.cache_clear()


async def test_schema_creates_expected_tables_and_indexes():
    engine = db_module.get_engine()
    async with engine.connect() as conn:
        def _inspect(sync_conn):
            insp = inspect(sync_conn)
            tables = set(insp.get_table_names())
            trace_indexes = {ix["name"] for ix in insp.get_indexes("traces")}
            return tables, trace_indexes

        tables, trace_indexes = await conn.run_sync(_inspect)

    assert {"traces", "model_prices"} <= tables
    # The composite index the dashboard queries depend on.
    assert "ix_traces_model_created" in trace_indexes


async def test_trace_round_trip_with_json_bodies():
    trace_id = uuid.uuid4()
    async with session() as s:
        s.add(
            Trace(
                id=trace_id,
                model_id="gpt-4o-mini",
                request_body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                response_body={"choices": [{"message": {"content": "hello"}}]},
                status_code=200,
                latency_ms=142,
                prompt_tokens=9,
                completion_tokens=3,
                total_tokens=12,
                cost_usd=Decimal("0.00001350"),
                outcome="ok",
            )
        )

    async with session() as s:
        row = (await s.execute(select(Trace).where(Trace.id == trace_id))).scalar_one()

    assert row.request_body["messages"][0]["content"] == "hi"
    assert row.response_body["choices"][0]["message"]["content"] == "hello"
    assert row.cost_usd == Decimal("0.00001350")
    assert row.created_at is not None


async def test_unit_of_work_rolls_back_atomically_on_error():
    """The A in ACID, used correctly: a failed block leaves zero rows behind."""
    marker = uuid.uuid4()
    with pytest.raises(RuntimeError):
        async with session() as s:
            s.add(Trace(id=marker, model_id="will-roll-back", outcome="ok"))
            await s.flush()  # row is visible inside the transaction...
            raise RuntimeError("boom mid-transaction")

    async with session() as s:
        row = (
            await s.execute(select(Trace).where(Trace.id == marker))
        ).scalar_one_or_none()
    assert row is None  # ...and gone after rollback. Nothing partial persisted.


async def test_model_price_natural_key_and_upsert_semantics():
    async with session() as s:
        s.add(
            ModelPrice(
                model_id="gpt-4o-mini",
                usd_per_1k_input=Decimal("0.00015"),
                usd_per_1k_output=Decimal("0.0006"),
                source="openai pricing page 2026-07",
            )
        )

    # Same PK again → merge updates rather than duplicating.
    async with session() as s:
        await s.merge(
            ModelPrice(
                model_id="gpt-4o-mini",
                usd_per_1k_input=Decimal("0.00020"),
                usd_per_1k_output=Decimal("0.0006"),
                source="price rev",
            )
        )

    async with session() as s:
        rows = (await s.execute(select(ModelPrice))).scalars().all()
    assert len(rows) == 1
    assert rows[0].usd_per_1k_input == Decimal("0.00020")


async def test_trace_allows_unknown_model_without_price_row():
    """No FK on model_id — logging must never fail because pricing lags."""
    async with session() as s:
        s.add(Trace(model_id="brand-new-model-nobody-priced", outcome="ok"))

    async with session() as s:
        row = (
            await s.execute(
                select(Trace).where(Trace.model_id == "brand-new-model-nobody-priced")
            )
        ).scalar_one()
    assert row.cost_usd is None  # honest NULL: unknown, not zero
