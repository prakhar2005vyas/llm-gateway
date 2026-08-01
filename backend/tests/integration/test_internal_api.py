"""Internal trace-browser API (Phase 7 UI): list shape, ordering, limits."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from app.db import dispose_db, init_db, session
from app.main import app
from app.models import Trace


@pytest.fixture(autouse=True)
async def _db():
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gw"
    )


async def _insert_traces(n: int) -> None:
    base = datetime.now(timezone.utc)
    async with session() as s:
        for i in range(n):
            s.add(
                Trace(
                    id=uuid.uuid4(),
                    created_at=base + timedelta(seconds=i),  # i=n-1 is newest
                    model_id="gpt-4o-mini",
                    status_code=200,
                    latency_ms=100 + i,
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost_usd=Decimal("0.00004500"),
                    cache_hit=(i % 2 == 0),
                    outcome="ok",
                )
            )


async def test_returns_200_and_json_list_when_empty():
    async with _client() as c:
        r = await c.get("/internal/traces")
    assert r.status_code == 200
    assert r.json() == []


async def test_lists_traces_newest_first_with_serialized_fields():
    await _insert_traces(3)
    async with _client() as c:
        r = await c.get("/internal/traces")
    assert r.status_code == 200
    traces = r.json()
    assert isinstance(traces, list) and len(traces) == 3

    # Newest first: latency encodes insert order (100 + i, i=2 newest).
    assert [t["latency_ms"] for t in traces] == [102, 101, 100]

    t0 = traces[0]
    assert uuid.UUID(t0["id"])  # id is a parseable uuid string
    assert t0["model"] == "gpt-4o-mini"
    assert t0["status_code"] == 200
    assert t0["total_tokens"] == 15
    # Exact Decimal serialized as string — no float rounding on money.
    assert t0["cost_usd"] == "0.00004500"
    assert t0["cache_hit"] is True
    assert t0["coalesced"] is False
    assert t0["outcome"] == "ok"
    assert "created_at" in t0 and t0["created_at"] is not None


async def test_limit_param_caps_the_page():
    await _insert_traces(5)
    async with _client() as c:
        r = await c.get("/internal/traces", params={"limit": 2})
    assert r.status_code == 200
    assert len(r.json()) == 2
    # Out-of-range limit is a clean 422, not a 500 or an unbounded query.
    async with _client() as c:
        r = await c.get("/internal/traces", params={"limit": 101})
    assert r.status_code == 422
