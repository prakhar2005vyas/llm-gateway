"""End-to-end coalescing through the live app: N concurrent identical
requests → exactly ONE upstream HTTP call; follower traces cost $0."""
from __future__ import annotations

import asyncio
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import select

from app.db import dispose_db, init_db, session
from app.main import app
from app.models import Trace

UPSTREAM = "https://upstream.test/v1/chat/completions"

COMPLETION = {
    "id": "chatcmpl-coal-1",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "shared answer"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@pytest.fixture(autouse=True)
async def _db():
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


@respx.mock
async def test_concurrent_identical_requests_fire_one_upstream_call():
    async def slow_upstream(_request):
        await asyncio.sleep(0.15)  # hold the flight open so all callers pile up
        return httpx.Response(200, json=COMPLETION)

    route = respx.post(UPSTREAM).mock(side_effect=slow_upstream)

    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [{"role": "user", "content": "the same question"}],
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        responses = await asyncio.gather(
            *(c.post("/v1/chat/completions", json=body) for _ in range(20))
        )

    # Every client got a full, correct answer...
    assert all(r.status_code == 200 for r in responses)
    assert all(
        r.json()["choices"][0]["message"]["content"] == "shared answer"
        for r in responses
    )
    # ...from exactly ONE upstream call.
    assert route.call_count == 1, (
        f"coalescing failed end-to-end: {route.call_count} upstream calls for 20 requests"
    )

    # Cost accounting: one leader carries the spend, 19 followers are $0.
    async with session() as s:
        traces = list((await s.execute(select(Trace))).scalars().all())
    assert len(traces) == 20
    followers = [t for t in traces if t.coalesced]
    leaders = [t for t in traces if not t.coalesced]
    assert len(followers) == 19 and len(leaders) == 1
    assert all(t.cost_usd == Decimal("0") for t in followers)
    assert all(t.prompt_tokens is None for t in followers)
    assert leaders[0].total_tokens == 15  # real spend recorded exactly once


@respx.mock
async def test_high_temperature_requests_are_not_coalesced():
    async def slow_upstream(_request):
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=COMPLETION)

    route = respx.post(UPSTREAM).mock(side_effect=slow_upstream)

    body = {
        "model": "gpt-4o-mini",
        "temperature": 1.0,  # creative — every caller deserves a fresh answer
        "messages": [{"role": "user", "content": "write me a poem"}],
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        responses = await asyncio.gather(
            *(c.post("/v1/chat/completions", json=body) for _ in range(5))
        )

    assert all(r.status_code == 200 for r in responses)
    assert route.call_count == 5  # bypassed: five real upstream calls
