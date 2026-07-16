"""Eval pipeline through the live app: sampled trace → judge → Eval row,
with the client response provably NOT waiting on the judge."""
from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import select

from app import evals
from app.config import get_settings
from app.db import dispose_db, init_db, session
from app.main import app
from app.models import Eval, Trace

UPSTREAM = "https://upstream.test/v1/chat/completions"
JUDGE = "https://judge.test/v1/chat/completions"

COMPLETION = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
    "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": "Paris is the capital."},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
}


@pytest.fixture(autouse=True)
async def _env(monkeypatch):
    monkeypatch.setenv("EVAL_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("EVAL_BASE_URL", "https://judge.test/v1")
    monkeypatch.setenv("EVAL_API_KEY", "judge-key")
    get_settings.cache_clear()
    await dispose_db()
    await init_db()
    yield
    await evals.drain_for_tests()
    await dispose_db()
    get_settings.cache_clear()


@respx.mock
async def test_sampled_request_produces_linked_eval_without_delaying_client():
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=COMPLETION))

    JUDGE_DELAY = 0.4
    judge_started = asyncio.Event()

    async def slow_judge(_request):
        judge_started.set()
        await asyncio.sleep(JUDGE_DELAY)  # a slow judge...
        return httpx.Response(200, json={
            "choices": [{"message": {
                "role": "assistant",
                "content": '{"helpfulness": 9, "tone": 8, "rationale": "accurate"}',
            }}]
        })

    respx.post(JUDGE).mock(side_effect=slow_judge)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        t0 = time.perf_counter()
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": "capital of France?"}]},
        )
        client_seconds = time.perf_counter() - t0

    # THE non-blocking proof: the client returned in far less time than the
    # judge takes — the eval task detached from the request lifecycle.
    assert r.status_code == 200
    assert client_seconds < JUDGE_DELAY, (
        f"client waited {client_seconds:.2f}s — eval is blocking the response"
    )

    # The eval finishes AFTER the client is long gone.
    await asyncio.wait_for(judge_started.wait(), timeout=5)
    await evals.drain_for_tests()

    async with session() as s:
        trace = (await s.execute(select(Trace))).scalars().one()
        row = (await s.execute(select(Eval))).scalars().one()
    assert row.trace_id == trace.id  # FK-linked to the sampled trace
    assert row.status == "completed"
    assert row.helpfulness_score == Decimal("9.00")
    assert row.tone_score == Decimal("8.00")


@respx.mock
async def test_cache_hit_is_never_evaluated(monkeypatch):
    """Second identical request hits the semantic cache → no second eval."""
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=COMPLETION))
    respx.post(JUDGE).mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant",
                                 "content": '{"helpfulness": 9, "tone": 8}'}}]
    }))

    body = {"model": "gpt-4o-mini", "temperature": 0,
            "messages": [{"role": "user", "content": "what is the capital of france"}]}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r1 = await c.post("/v1/chat/completions", json=body)
        await evals.drain_for_tests()  # let the first eval land
        r2 = await c.post("/v1/chat/completions", json=body)
        await evals.drain_for_tests()

    assert r1.status_code == r2.status_code == 200
    assert r2.headers.get("x-gateway-cache") == "hit"
    async with session() as s:
        eval_rows = (await s.execute(select(Eval))).scalars().all()
    assert len(eval_rows) == 1  # the upstream answer was judged exactly once
