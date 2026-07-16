"""SemanticCache schema + mask_prompt placeholder.

SQLite only proves schema shape and round-trip (the vector column degrades to
JSON there); real similarity search is validated against pgvector Postgres in
the compose environment. mask_prompt is identity until Phase 5 — these tests
pin the seam's CONTRACT, not redaction behavior.
"""
from __future__ import annotations

import pytest
from sqlalchemy import inspect, select

from app import db as db_module
from app.config import get_settings
from app.db import dispose_db, init_db, session
from app.models import SemanticCache
from app.pii import mask_prompt


@pytest.fixture(autouse=True)
async def _fresh_db():
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


async def test_semantic_cache_table_exists_with_expected_columns():
    engine = db_module.get_engine()
    async with engine.connect() as conn:
        def _cols(sync_conn):
            insp = inspect(sync_conn)
            return {c["name"] for c in insp.get_columns("semantic_cache")}

        cols = await conn.run_sync(_cols)
    assert {
        "id", "created_at", "masked_prompt", "prompt_embedding",
        "model_id", "response_body", "hit_count", "last_hit_at",
    } <= cols


async def test_round_trip_embedding_and_response():
    dim = get_settings().embedding_dim
    fake_embedding = [0.01 * i for i in range(dim)]
    async with session() as s:
        s.add(
            SemanticCache(
                masked_prompt=mask_prompt("what is the capital of France?"),
                prompt_embedding=fake_embedding,
                model_id="gpt-4o-mini",
                response_body={"choices": [{"message": {"content": "Paris."}}]},
            )
        )

    async with session() as s:
        row = (await s.execute(select(SemanticCache))).scalar_one()
    assert row.masked_prompt == "what is the capital of France?"
    assert list(row.prompt_embedding) == pytest.approx(fake_embedding)
    assert len(list(row.prompt_embedding)) == dim
    assert row.response_body["choices"][0]["message"]["content"] == "Paris."
    assert row.hit_count == 0
    assert row.last_hit_at is None


def test_mask_prompt_is_identity_until_phase5():
    # Pins the CURRENT placeholder contract loudly. When Phase 5 implements
    # real redaction, this test MUST be replaced by real redaction tests —
    # if it starts failing because masking now works, that's the reminder.
    assert mask_prompt("alice@example.com called 555-0100") == (
        "alice@example.com called 555-0100"
    )
    assert mask_prompt("") == ""


def test_embedding_dim_is_config_driven(monkeypatch):
    # The column dimension binds at class definition (documented in models.py),
    # but the SETTING itself must be env-driven for the Phase 3 cache code.
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    get_settings.cache_clear()
    assert get_settings().embedding_dim == 768
    get_settings.cache_clear()
