"""Phase 4 chaos suite — each test injects a failure/load and proves one
resume claim (SPEC chaos matrix). "Don't claim resilience — inject failure
and screenshot the survival."

LIVE GATEWAY, for real: uvicorn on a TCP socket, a real stub upstream on a
second socket (or a genuinely dead port), no respx, no ASGI transport. Both
servers share the test's event loop, so the in-memory DB is queryable for
trace assertions.

Claims proven here:
  1. 500 identical concurrent requests → exactly ONE upstream call.
  2. Over-limit burst → exact N admitted, rest clean 429; no degradation.
  3. Dead primary → automatic failover reroute, client sees a clean 200.
"""
from __future__ import annotations

import asyncio
import socket
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from sqlalchemy import func, select

from app.config import get_settings
from app.db import dispose_db, init_db, session
from app.models import Trace

COMPLETION = {
    "id": "chatcmpl-chaos",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "chaos survivor"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_stub(delay: float) -> tuple[FastAPI, dict]:
    """A counting OpenAI-compatible stub. `delay` holds requests open so
    concurrent callers genuinely overlap in flight."""
    stub = FastAPI()
    state = {"calls": 0}

    @stub.post("/v1/chat/completions")
    async def chat():  # noqa: ANN202
        state["calls"] += 1
        await asyncio.sleep(delay)
        return COMPLETION

    return stub, state


class Server:
    """uvicorn on a real socket, running as a task on the current loop."""

    def __init__(self, app, port: int):
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(200):
            if self._server.started:
                return self
            await asyncio.sleep(0.02)
        raise RuntimeError("uvicorn did not start")

    async def __aexit__(self, *exc):
        self._server.should_exit = True
        await self._task


@pytest.fixture(autouse=True)
async def _db(tmp_path, monkeypatch):
    # FILE-based SQLite, not :memory: (same lesson as test_frozen_db): the
    # shared-memory DB rides ONE StaticPool connection, and under chaos-scale
    # concurrency + a loaded machine its writes get flaky (lost commits,
    # multi-second stalls). A file gives every pooled connection real
    # isolation, like the Postgres these tests stand in for.
    db_file = (tmp_path / "chaos.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    get_settings.cache_clear()
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


async def wait_for_traces(expected: int, timeout: float = 30.0) -> None:
    """Background tasks finish after responses; poll until all traces land."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with session() as s:
            count = (await s.execute(select(func.count()).select_from(Trace))).scalar_one()
        if count >= expected:
            return
        await asyncio.sleep(0.1)
    pytest.fail(f"only {count}/{expected} traces after {timeout}s")


# ---------------------------------------------------------------------------
# 1. Request coalescing: 500 identical concurrent → ONE upstream call
# ---------------------------------------------------------------------------

async def test_chaos_500_identical_requests_one_upstream_call(monkeypatch):
    gw_port, stub_port = free_port(), free_port()
    monkeypatch.setenv("UPSTREAM_BASE_URL", f"http://127.0.0.1:{stub_port}/v1")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10000")  # not under test here
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")  # isolate coalescing
    get_settings.cache_clear()

    # Stub holds the flight open 1.5s — every request overlaps the leader.
    stub_app, stub_state = make_stub(delay=1.5)
    from app.main import app as gateway_app

    body = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [{"role": "user", "content": "the exact same question"}],
    }

    async with Server(stub_app, stub_port), Server(gateway_app, gw_port):
        limits = httpx.Limits(max_connections=600, max_keepalive_connections=600)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{gw_port}", limits=limits, timeout=60
        ) as c:
            t0 = time.perf_counter()
            tasks = [
                asyncio.create_task(c.post("/v1/chat/completions", json=body))
                for _ in range(500)
            ]
            dispatch_seconds = time.perf_counter() - t0
            responses = await asyncio.gather(*tasks)
            total_seconds = time.perf_counter() - t0

        assert dispatch_seconds < 1.0, f"burst not within 1s ({dispatch_seconds:.2f}s)"

        # Every client whole:
        assert all(r.status_code == 200 for r in responses)
        assert all(
            r.json()["choices"][0]["message"]["content"] == "chaos survivor"
            for r in responses
        )
        # THE claim: one upstream call for 500 requests.
        assert stub_state["calls"] == 1, (
            f"coalescing failed under chaos: {stub_state['calls']} upstream calls"
        )
        # Wall-clock sanity: 500 sequential 1.5s calls would take ~12min;
        # one shared flight finishes in ~one stub-delay.
        assert total_seconds < 30

        # Trace accounting: exactly ONE non-coalesced (upstream) trace.
        await wait_for_traces(500)
        async with session() as s:
            upstream_traces = (
                await s.execute(select(Trace).where(Trace.coalesced == False))  # noqa: E712
            ).scalars().all()
            follower_count = (
                await s.execute(
                    select(func.count()).select_from(Trace).where(Trace.coalesced == True)  # noqa: E712
                )
            ).scalar_one()
        assert len(upstream_traces) == 1, (
            f"expected exactly ONE upstream trace, got {len(upstream_traces)}"
        )
        assert follower_count == 499
        assert upstream_traces[0].total_tokens == 15  # real spend logged once

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 2. Rate limit burst: exact admits, clean 429s, zero degradation
# ---------------------------------------------------------------------------

async def test_chaos_rate_limit_burst_exact_split(monkeypatch):
    gw_port, stub_port = free_port(), free_port()
    LIMIT = 25
    monkeypatch.setenv("UPSTREAM_BASE_URL", f"http://127.0.0.1:{stub_port}/v1")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", str(LIMIT))
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()

    stub_app, stub_state = make_stub(delay=0.0)
    from app.main import app as gateway_app

    # No temperature → not coalesceable/cacheable: every admitted request
    # must genuinely reach upstream, making the admit-count independently
    # verifiable at the stub.
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    headers = {"Authorization": "Bearer sk-burst-key"}

    async with Server(stub_app, stub_port), Server(gateway_app, gw_port):
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{gw_port}", timeout=30
        ) as c:
            responses = await asyncio.gather(
                *(
                    c.post("/v1/chat/completions", json=body, headers=headers)
                    for _ in range(100)
                )
            )

        codes = [r.status_code for r in responses]
        # EXACT split: the window admits precisely LIMIT, rejects the rest.
        assert codes.count(200) == LIMIT, f"admitted {codes.count(200)}, want {LIMIT}"
        assert codes.count(429) == 100 - LIMIT
        assert stub_state["calls"] == LIMIT  # upstream saw exactly the admits

        # Rejections are CLEAN: OpenAI-shaped error + usable Retry-After.
        rejected = [r for r in responses if r.status_code == 429]
        assert all(r.json()["error"]["type"] == "rate_limit_error" for r in rejected)
        assert all(int(r.headers["retry-after"]) >= 1 for r in rejected)

        # Within-limit traffic undegraded: a DIFFERENT key sails through
        # immediately after the hostile burst.
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{gw_port}", timeout=30
        ) as c:
            r = await c.post(
                "/v1/chat/completions",
                json=body,
                headers={"Authorization": "Bearer sk-innocent-bystander"},
            )
        assert r.status_code == 200

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 3. Upstream outage: dead primary port → automatic failover → clean 200
# ---------------------------------------------------------------------------

async def test_chaos_dead_primary_fails_over_to_secondary(monkeypatch):
    gw_port, failover_port = free_port(), free_port()
    dead_port = free_port()  # allocated then never bound — genuinely dead

    monkeypatch.setenv("UPSTREAM_BASE_URL", f"http://127.0.0.1:{dead_port}/v1")
    monkeypatch.setenv("FAILOVER_BASE_URL", f"http://127.0.0.1:{failover_port}/v1")
    monkeypatch.setenv("FAILOVER_API_KEY", "fw-chaos-key")
    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "1")  # keep backoff time short
    get_settings.cache_clear()

    stub_app, stub_state = make_stub(delay=0.0)
    from app.main import app as gateway_app

    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello?"}]}

    async with Server(stub_app, failover_port), Server(gateway_app, gw_port):
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{gw_port}", timeout=30
        ) as c:
            r = await c.post("/v1/chat/completions", json=body)

        # Client never learns the primary died: clean 200, real completion.
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "chaos survivor"
        # And it truly came through the failover provider.
        assert stub_state["calls"] == 1

        # Trace records a healthy request (the reroute is an upstream
        # implementation detail, loudly logged, not a client-visible error).
        await wait_for_traces(1)
        async with session() as s:
            trace = (await s.execute(select(Trace))).scalars().one()
        assert trace.outcome == "ok"
        assert trace.status_code == 200

    get_settings.cache_clear()
