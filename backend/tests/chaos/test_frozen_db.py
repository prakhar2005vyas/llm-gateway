"""Phase 7 chaos: the frozen-Postgres test — THE hot/cold seam proof.

SPEC's chaos matrix calls this the crown jewel: "hot-path P99 stays flat
while the log queue sheds." The freeze is injected at the session seam (the
exact boundary a `docker pause postgres` wedges): every trace-write session
hangs until the test "unpauses" the DB, then dies with a connection error —
the same shape asyncpg produces when a paused Postgres finally drops the
connection.

Claims proven, in order:
  1. Hot path untouched: 100 concurrent proxied requests all return 200 with
     P99 within a small factor of the healthy-DB P99 at IDENTICAL load — not
     the freeze duration, not any timeout. The client cannot tell the DB is
     gone.
  2. Event loop alive during the freeze: /health answers instantly while the
     wedged trace writes sit parked in background tasks.
  3. Cold path sheds loudly WHILE THE DB IS STILL FROZEN, never crashes:
     every wedged write is cancelled at TRACE_WRITE_TIMEOUT_SECONDS and
     dropped with its "TRACE WRITE FAILED" log line. The ceiling exists
     because a BackgroundTask runs inside the ASGI cycle — an unbounded
     wedge would hold that keep-alive connection hostage and stall the next
     request pipelined onto it (observed live: httpx ReadError +
     ClientDisconnect before the timeout existed).
  4. Recovery: the first request after the thaw traces normally again.

Honest-scoping note: against real asyncpg two mechanisms compound — the
BOUNDED pool (pool_size + max_overflow + pool_timeout) fails checkouts fast
once wedged writes saturate it, and the write timeout above cancels the
wedged writers themselves. This test wedges the session seam directly so it
runs hermetically on SQLite; the pool bounds are config (db.py).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.db import dispose_db, init_db
from app.db import session as real_session
from app.models import Trace
from tests.chaos.test_chaos import Server, free_port, make_stub

BASELINE_REQUESTS = 100
FROZEN_REQUESTS = 100
STUB_DELAY = 0.05
# The freeze-leak signature is unmistakable: a hot path awaiting a wedged
# session hangs until the thaw (30s+ client timeout), not milliseconds. The
# bound is RELATIVE — frozen P99 within 3x of a healthy-DB P99 measured at
# the SAME concurrency — so machine-speed variance cancels out; the absolute
# floor keeps a very fast baseline from making 3x unreasonably strict.
P99_RELATIVE_BOUND = 3.0
P99_FLOOR_SECONDS = 3.0


def p99(latencies: list[float]) -> float:
    ordered = sorted(latencies)
    return ordered[int(0.99 * (len(ordered) - 1))]


@pytest.fixture(autouse=True)
async def _db(tmp_path, monkeypatch):
    # FILE-based SQLite, not :memory:. The shared-memory DB rides ONE
    # connection (StaticPool); 100+ concurrent trace-write sessions would
    # interleave transactions on it and silently lose commits — an artifact
    # of the test DB, not the seam under test. A file gives every pooled
    # connection real isolation, like the Postgres this test stands in for.
    db_file = (tmp_path / "chaos.db").as_posix()
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    get_settings.cache_clear()
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


async def _trace_count() -> int:
    async with real_session() as s:
        return (await s.execute(select(func.count()).select_from(Trace))).scalar_one()


async def _poll(predicate, timeout: float, message) -> None:
    """Await an async predicate turning true (cold-path work lands async)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.05)
    pytest.fail(message())


async def _fire(client: httpx.AsyncClient, n: int, offset: int) -> tuple[list, list[float]]:
    """N concurrent chat completions with UNIQUE prompts (no temperature →
    neither cacheable nor coalesceable: every request must reach upstream and
    schedule its own trace write). Returns (responses, per-request seconds)."""

    async def one(i: int):
        t0 = time.perf_counter()
        r = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": f"question {offset + i}"}],
            },
        )
        return r, time.perf_counter() - t0

    results = await asyncio.gather(*(one(i) for i in range(n)))
    return [r for r, _ in results], [lat for _, lat in results]


async def test_chaos_frozen_postgres_hot_path_stays_flat(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="app.tracing")
    gw_port, stub_port = free_port(), free_port()
    monkeypatch.setenv("UPSTREAM_BASE_URL", f"http://127.0.0.1:{stub_port}/v1")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "100000")  # not under test here
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")  # isolate the seam
    # Write ceiling low enough to assert the shed in-test, but with enough
    # headroom that HEALTHY-DB baseline writes never trip it: 100 concurrent
    # commits serialize on file-SQLite's write lock, so the tail legitimately
    # takes a few seconds (measured: 1.5s sheds ~10% of a healthy baseline).
    monkeypatch.setenv("TRACE_WRITE_TIMEOUT_SECONDS", "8")
    get_settings.cache_clear()

    # --- the freeze, injected at the session seam --------------------------
    frozen = asyncio.Event()   # set = Postgres is "paused"
    thawed = asyncio.Event()   # set = Postgres "unpaused", wedged conns die
    wedged = {"count": 0, "entered": 0}

    @asynccontextmanager
    async def freezable_session():
        wedged["entered"] += 1
        if frozen.is_set():
            wedged["count"] += 1
            await thawed.wait()  # a query against a paused DB just... hangs
            raise ConnectionError(
                "simulated: connection killed while Postgres was paused"
            )
        async with real_session() as s:
            yield s

    # tracing.py bound `session` at import time — patch ITS reference, which
    # is exactly the seam every cold-path trace write goes through.
    monkeypatch.setattr("app.tracing.session", freezable_session)

    stub_app, stub_state = make_stub(delay=STUB_DELAY)
    from app.main import app as gateway_app

    def shed_log_count() -> int:
        return sum(1 for r in caplog.records if "TRACE WRITE FAILED" in r.getMessage())

    async with Server(stub_app, stub_port), Server(gateway_app, gw_port):
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=200)
        # TWO clients on purpose. A keep-alive connection is only fully free
        # once its ASGI cycle (response + background task) completes, so the
        # frozen phase must not reuse baseline connections: requests pipelined
        # behind a wedged cycle would measure httpx's connection scheduling,
        # not the gateway's hot path. (In production that hostage window is
        # exactly what TRACE_WRITE_TIMEOUT_SECONDS bounds.)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{gw_port}", limits=limits, timeout=30
        ) as c, httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{gw_port}", limits=limits, timeout=30
        ) as c_frozen:
            # --- Phase 1: healthy-DB baseline ------------------------------
            responses, baseline_lat = await _fire(c, BASELINE_REQUESTS, offset=0)
            assert all(r.status_code == 200 for r in responses)

            seen = {"n": -1}

            async def baseline_landed() -> bool:
                seen["n"] = await _trace_count()
                return seen["n"] >= BASELINE_REQUESTS

            await _poll(
                baseline_landed,
                timeout=30,  # 100 serialized file-SQLite commits take a moment
                message=lambda: (
                    f"baseline traces never landed with a healthy DB "
                    f"(count={seen['n']}, entered={wedged['entered']}, errors="
                    f"{[r.getMessage()[:120] for r in caplog.records[:3]]})"
                ),
            )

            # --- Phase 2: freeze the DB, fire the real load ----------------
            # try/finally: the thaw MUST happen even when an assertion fails
            # mid-freeze — wedged background tasks keep their ASGI cycles
            # open, and uvicorn's graceful shutdown waits on those forever.
            # Without the finally, a failed assert here becomes a deadlocked
            # teardown instead of a failure report.
            try:
                frozen.set()
                responses, frozen_lat = await _fire(
                    c_frozen, FROZEN_REQUESTS, offset=BASELINE_REQUESTS
                )

                # Claim 1: every client whole, hot-path latency flat.
                assert all(r.status_code == 200 for r in responses)
                frozen_p99, baseline_p99 = p99(frozen_lat), p99(baseline_lat)
                bound = max(P99_FLOOR_SECONDS, P99_RELATIVE_BOUND * baseline_p99)
                assert frozen_p99 < bound, (
                    f"hot path degraded by the DB freeze: P99 {frozen_p99:.2f}s "
                    f"vs healthy-DB P99 {baseline_p99:.2f}s at identical load "
                    f"(bound {bound:.2f}s) — the seam leaked"
                )

                # Claim 2: event loop alive while wedged writes sit parked.
                for _ in range(3):
                    t0 = time.perf_counter()
                    health = await c.get("/health")
                    assert health.status_code == 200
                    assert time.perf_counter() - t0 < 1.0, "event loop starved"

                # Every frozen-phase request scheduled a trace write; every
                # one hit the wedge — none landed, none crashed anything.
                await _poll(
                    lambda: _is_true(lambda: wedged["count"] >= FROZEN_REQUESTS),
                    timeout=15,
                    message=lambda: f"only {wedged['count']}/{FROZEN_REQUESTS} writes wedged",
                )

                # Claim 3: WHILE THE DB IS STILL FROZEN, every wedged write
                # is cancelled at the write ceiling and shed loudly. This is
                # the bounded cold path — no thaw required to unstick it.
                await _poll(
                    lambda: _is_true(lambda: shed_log_count() >= FROZEN_REQUESTS),
                    timeout=25,  # ceiling is 8s; generous slack for 100 tasks
                    message=lambda: f"only {shed_log_count()}/{FROZEN_REQUESTS} sheds logged",
                )
                assert await _trace_count() == BASELINE_REQUESTS  # dropped, none half-written
            finally:
                # --- Phase 3: thaw (docker unpause) — also the guaranteed
                # unwind path: without it, a failed assert above would leave
                # wedged tasks holding ASGI cycles open and uvicorn's
                # graceful shutdown would deadlock the whole test.
                frozen.clear()
                thawed.set()

            # --- Phase 4: recovery — next request traces normally ----------
            responses, _ = await _fire(c, 1, offset=BASELINE_REQUESTS + FROZEN_REQUESTS)
            assert responses[0].status_code == 200
            await _poll(
                lambda: _landed(BASELINE_REQUESTS + 1),
                timeout=15,
                message=lambda: "post-thaw trace never landed — cold path did not recover",
            )

        # Upstream saw every single request: the proxy never stopped proxying.
        assert stub_state["calls"] == BASELINE_REQUESTS + FROZEN_REQUESTS + 1

    get_settings.cache_clear()


async def _landed(expected: int) -> bool:
    return await _trace_count() >= expected


async def _is_true(check) -> bool:
    return bool(check())
