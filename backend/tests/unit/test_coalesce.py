"""Coalescer tests — the single-flight contract under real concurrency.

The centerpiece is the SPEC chaos-matrix claim at unit scale: 500 identical
concurrent callers → exactly ONE supplier execution, every caller gets the
same result object.
"""
from __future__ import annotations

import asyncio

import pytest

from app.coalesce import Coalescer, is_coalesceable, request_key


def body(prompt: str = "hi", temperature: float | None = 0.0, **extra) -> dict:
    b: dict = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        b["temperature"] = temperature
    b.update(extra)
    return b


# ---------------------------------------------------------------------------
# The mandated proof
# ---------------------------------------------------------------------------

async def test_500_identical_concurrent_callers_one_upstream_call():
    coal = Coalescer()
    calls = 0
    release = asyncio.Event()

    async def supplier():
        nonlocal calls
        calls += 1
        await release.wait()  # hold the flight open so ALL callers pile up
        return {"answer": 42}

    async def one_request():
        return await coal.get_or_run("k", supplier)

    tasks = [asyncio.create_task(one_request()) for _ in range(500)]
    await asyncio.sleep(0)  # let every task reach the CAS
    assert coal.in_flight == 1  # one pending flight, 499 followers attached
    release.set()
    results = await asyncio.gather(*tasks)

    assert calls == 1, f"coalescing failed: {calls} upstream calls for 500 requests"
    leaders = [is_leader for _, is_leader in results]
    assert leaders.count(True) == 1 and leaders.count(False) == 499
    payloads = {id(r) for r, _ in results}
    assert len(payloads) == 1  # everyone shares the leader's result object
    assert coal.in_flight == 0  # pending map fully drained — no leak


# ---------------------------------------------------------------------------
# Flight lifecycle
# ---------------------------------------------------------------------------

async def test_different_keys_run_independently():
    coal = Coalescer()
    calls: list[str] = []
    gate = asyncio.Event()

    def supplier_for(key: str):
        async def s():
            calls.append(key)
            await gate.wait()
            return key.upper()
        return s

    t1 = asyncio.create_task(coal.get_or_run("a", supplier_for("a")))
    t2 = asyncio.create_task(coal.get_or_run("b", supplier_for("b")))
    await asyncio.sleep(0)
    assert coal.in_flight == 2
    gate.set()
    (r1, l1), (r2, l2) = await asyncio.gather(t1, t2)
    assert sorted(calls) == ["a", "b"]  # no cross-key sharing
    assert (r1, r2) == ("A", "B") and l1 and l2


async def test_sequential_requests_do_not_share_stale_results():
    """Coalescing is for CONCURRENT duplicates only. A later identical
    request must trigger a fresh flight (staleness is the cache's domain)."""
    coal = Coalescer()
    calls = 0

    async def supplier():
        nonlocal calls
        calls += 1
        return calls

    r1, _ = await coal.get_or_run("k", supplier)
    r2, _ = await coal.get_or_run("k", supplier)
    assert (r1, r2) == (1, 2)
    assert calls == 2


async def test_leader_failure_is_shared_then_next_flight_is_fresh():
    coal = Coalescer()
    calls = 0
    release = asyncio.Event()

    async def failing_supplier():
        nonlocal calls
        calls += 1
        await release.wait()
        raise RuntimeError("upstream exploded")

    tasks = [
        asyncio.create_task(coal.get_or_run("k", failing_supplier)) for _ in range(10)
    ]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # ALL ten see the same failure — nobody hangs, nobody retries silently.
    assert calls == 1
    assert all(isinstance(r, RuntimeError) for r in results)
    assert coal.in_flight == 0  # cleaned up even on failure

    # And the key is immediately usable for a fresh (successful) flight.
    async def good_supplier():
        nonlocal calls
        calls += 1
        return "ok"

    r, is_leader = await coal.get_or_run("k", good_supplier)
    assert r == "ok" and is_leader and calls == 2


async def test_leader_cancellation_does_not_kill_followers():
    """C-1 audit fix: leader's client disconnects (task cancelled) while the
    upstream flight is open — followers must still get the real result."""
    coal = Coalescer()
    calls = 0
    release = asyncio.Event()

    async def supplier():
        nonlocal calls
        calls += 1
        await release.wait()
        return {"answer": "survived"}

    leader_task = asyncio.create_task(coal.get_or_run("k", supplier))
    await asyncio.sleep(0)  # leader claims the key, flight opens
    follower_tasks = [
        asyncio.create_task(coal.get_or_run("k", supplier)) for _ in range(5)
    ]
    await asyncio.sleep(0)  # followers attach to the leader's future

    leader_task.cancel()  # uvicorn: client went away
    with pytest.raises(asyncio.CancelledError):
        await leader_task

    release.set()  # upstream finally answers
    results = await asyncio.gather(*follower_tasks)

    assert calls == 1  # ONE flight, despite the leader dying
    assert all(r == {"answer": "survived"} for r, _ in results)
    assert all(is_leader is False for _, is_leader in results)
    assert coal.in_flight == 0  # map cleaned by the detached flight


async def test_leader_cancellation_with_zero_followers_still_cleans_up():
    coal = Coalescer()
    release = asyncio.Event()

    async def supplier():
        await release.wait()
        return "unobserved"

    leader_task = asyncio.create_task(coal.get_or_run("k", supplier))
    await asyncio.sleep(0)
    leader_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader_task

    release.set()
    await asyncio.sleep(0.01)  # let the detached flight finish
    assert coal.in_flight == 0  # no leaked pending entry


async def test_cancelled_leader_failed_flight_shares_error_with_followers():
    """Leader gone AND the flight fails: followers get the real upstream
    error (not a hang, not a cancellation)."""
    coal = Coalescer()
    release = asyncio.Event()

    async def failing_supplier():
        await release.wait()
        raise RuntimeError("upstream 503")

    leader_task = asyncio.create_task(coal.get_or_run("k", failing_supplier))
    await asyncio.sleep(0)
    followers = [asyncio.create_task(coal.get_or_run("k", failing_supplier))
                 for _ in range(3)]
    await asyncio.sleep(0)

    leader_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader_task

    release.set()
    results = await asyncio.gather(*followers, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    assert coal.in_flight == 0


# ---------------------------------------------------------------------------
# Guards + keying
# ---------------------------------------------------------------------------

def test_temperature_guard():
    assert is_coalesceable(body(temperature=0.0))
    assert is_coalesceable(body(temperature=0.2))
    assert not is_coalesceable(body(temperature=0.21))
    assert not is_coalesceable(body(temperature=1.0))
    assert not is_coalesceable(body(temperature=None))  # omitted = 1.0 upstream
    assert not is_coalesceable(body(temperature=True))  # bool is not a temperature


def test_stream_and_shape_guards():
    assert not is_coalesceable(body(stream=True))
    assert not is_coalesceable({"model": "m", "temperature": 0, "messages": []})
    no_model = body()
    del no_model["model"]
    assert not is_coalesceable(no_model)


def test_disabled_via_env(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("COALESCE_ENABLED", "false")
    get_settings.cache_clear()
    assert not is_coalesceable(body(temperature=0.0))
    get_settings.cache_clear()


def test_request_key_semantics():
    # Same conversation+model → same key; model or content change → new key.
    assert request_key(body("hi")) == request_key(body("hi"))
    assert request_key(body("hi")) != request_key(body("hi!"))
    other_model = body("hi")
    other_model["model"] = "gpt-4o"
    assert request_key(body("hi")) != request_key(other_model)
