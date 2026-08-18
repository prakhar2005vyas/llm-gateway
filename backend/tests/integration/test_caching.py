"""Phase 3 semantic-cache integration tests — REAL pgvector Postgres.

A throwaway pgvector container is started for this module (docker required;
tests skip cleanly if the daemon is unavailable — set CACHE_TEST_PG_URL to an
existing pgvector Postgres to skip container management). Embeddings are the
real all-MiniLM model. The upstream LLM is respx-mocked as always.

The two mandated proofs:
* ADVERSARIAL: "delete my account" cached, then "recover my account" asked —
  MUST go upstream (a collision here would serve account-deletion advice to a
  recovery request: the worst possible cache bug). Fails on false positive.
* PARAPHRASE: "what is the capital of france" cached, then "what's the
  capital of france?" asked — MUST be served from cache with zero upstream
  calls at $0. Fails on false negative.
Plus the threshold-placement proof making the tuning explicit, and a strict
xfail documenting the known direction-flip limitation.
"""
from __future__ import annotations

import os
import subprocess
import time as time_mod
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import delete, select

from app.config import get_settings
from app.db import dispose_db, init_db, session
from app.main import app
from app.models import ModelPrice, SemanticCache, Trace

UPSTREAM = "https://upstream.test/v1/chat/completions"
CONTAINER = "llmgw-cache-test-pg"
PORT = 55432


# ---------------------------------------------------------------------------
# Real pgvector Postgres for this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pg_url():
    if url := os.environ.get("CACHE_TEST_PG_URL"):
        yield url
        return

    def _docker(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=120
        )

    try:
        if _docker("info").returncode != 0:
            pytest.skip("docker daemon unavailable — set CACHE_TEST_PG_URL to run")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("docker not installed/responding — set CACHE_TEST_PG_URL to run")

    _docker("rm", "-f", CONTAINER)  # stale leftover from an aborted run
    # Retry once: back-to-back suite runs can race the previous container's
    # port release (docker stop is asynchronous with --rm cleanup).
    for attempt in (1, 2):
        started = _docker(
            "run", "--rm", "-d", "--name", CONTAINER,
            "-e", "POSTGRES_PASSWORD=cachetest",
            "-e", "POSTGRES_DB=cachetest",
            "-p", f"{PORT}:5432",
            "pgvector/pgvector:pg15",
        )
        if started.returncode == 0:
            break
        _docker("rm", "-f", CONTAINER)
        time_mod.sleep(3)
    assert started.returncode == 0, f"container failed to start: {started.stderr}"
    try:
        deadline = time_mod.time() + 60
        while time_mod.time() < deadline:
            ready = _docker("exec", CONTAINER, "pg_isready", "-U", "postgres")
            if ready.returncode == 0:
                break
            time_mod.sleep(0.5)
        else:
            pytest.fail("test postgres did not become ready in 60s")
        yield f"postgresql+asyncpg://postgres:cachetest@localhost:{PORT}/cachetest"
    finally:
        _docker("stop", CONTAINER)


@pytest.fixture(autouse=True)
async def _db(pg_url, monkeypatch):
    """Point the app at the pgvector container; clean tables between tests.

    init_db retries: `pg_isready` inside the container reports ready during
    the image's initdb phase (a TEMPORARY server that then restarts), so the
    first host-side asyncpg connection can hit connection_lost. Only a real
    connection through the forwarded port proves readiness.
    """
    monkeypatch.setenv("DATABASE_URL", pg_url)
    get_settings.cache_clear()
    await dispose_db()
    deadline = time_mod.time() + 30
    while True:
        try:
            await init_db()
            break
        except Exception as e:
            if time_mod.time() > deadline:
                raise
            await dispose_db()  # drop the poisoned engine before retrying
            import asyncio

            await asyncio.sleep(0.5)
    async with session() as s:
        await s.execute(delete(Trace))
        await s.execute(delete(SemanticCache))
        await s.merge(
            ModelPrice(
                model_id="gpt-4o-mini",
                usd_per_1k_input=Decimal("0.00015"),
                usd_per_1k_output=Decimal("0.0006"),
                source="test seed",
            )
        )
    yield
    await dispose_db()
    get_settings.cache_clear()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        yield c


def completion(text: str) -> dict:
    return {
        "id": "chatcmpl-real-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def req(prompt: str) -> dict:
    # temperature=0: explicitly deterministic — the cacheability requirement.
    return {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }


async def _post(client, prompt: str):
    return await client.post("/v1/chat/completions", json=req(prompt))


# ---------------------------------------------------------------------------
# The two mandated proofs
# ---------------------------------------------------------------------------

@respx.mock
async def test_adversarial_near_miss_does_NOT_collide(client):
    """'delete my account' cached; 'recover my account' MUST go upstream."""
    route = respx.post(UPSTREAM)
    route.side_effect = [
        httpx.Response(200, json=completion("Account deletion steps: ...")),
        httpx.Response(200, json=completion("Account recovery steps: ...")),
    ]

    r1 = await _post(client, "delete my account")
    assert r1.status_code == 200 and route.call_count == 1

    r2 = await _post(client, "recover my account")
    assert r2.status_code == 200
    # THE assertion: a second upstream call happened — no false positive.
    assert route.call_count == 2, (
        "FALSE POSITIVE: 'recover my account' was served the cached "
        "'delete my account' response — threshold is too loose"
    )
    assert "recovery" in r2.json()["choices"][0]["message"]["content"]
    assert r2.headers["x-gateway-cache"] == "miss"

    # Both traces are full-cost upstream calls.
    async with session() as s:
        hits = (await s.execute(select(Trace).where(Trace.cache_hit))).scalars().all()
    assert hits == []


@respx.mock
async def test_paraphrase_hits_cache_at_zero_cost(client):
    """'what is the capital of france' cached; the paraphrase MUST hit."""
    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, json=completion("Paris."))
    )

    r1 = await _post(client, "what is the capital of france")
    assert r1.status_code == 200 and route.call_count == 1
    assert r1.headers["x-gateway-cache"] == "miss"

    r2 = await _post(client, "what's the capital of france?")
    assert r2.status_code == 200
    # THE assertion: zero additional upstream calls — no false negative.
    assert route.call_count == 1, (
        "FALSE NEGATIVE: the paraphrase went upstream instead of hitting "
        "the cache — threshold is too strict"
    )
    assert r2.headers["x-gateway-cache"] == "hit"
    assert r2.json()["choices"][0]["message"]["content"] == "Paris."

    # Trace: $0 literal, cache_hit=True, no token consumption recorded.
    async with session() as s:
        hit_trace = (
            await s.execute(select(Trace).where(Trace.cache_hit))
        ).scalar_one()
        entry = (await s.execute(select(SemanticCache))).scalar_one()
    assert hit_trace.cost_usd == Decimal("0")
    assert hit_trace.outcome == "ok"
    assert hit_trace.prompt_tokens is None  # nothing consumed upstream
    # And the entry's hit bookkeeping advanced.
    assert entry.hit_count == 1
    assert entry.last_hit_at is not None


# ---------------------------------------------------------------------------
# Threshold placement + guards
# ---------------------------------------------------------------------------

async def test_threshold_sits_between_the_measured_pairs():
    """Makes the tuning explicit: adversarial < threshold <= paraphrase.
    If a model/threshold change breaks this ordering, this fails first with
    the actual numbers, before the behavioral tests confuse anyone."""
    from app.embeddings import get_embedding

    async def sim(a: str, b: str) -> float:
        va, vb = await get_embedding(f"user: {a}"), await get_embedding(f"user: {b}")
        return sum(x * y for x, y in zip(va, vb))

    adversarial = await sim("delete my account", "recover my account")
    paraphrase = await sim(
        "what is the capital of france", "what's the capital of france?"
    )
    threshold = get_settings().cache_similarity_threshold
    assert adversarial < threshold, (
        f"adversarial pair ({adversarial:.4f}) >= threshold ({threshold}) — "
        "false positives WILL happen"
    )
    assert paraphrase >= threshold, (
        f"paraphrase pair ({paraphrase:.4f}) < threshold ({threshold}) — "
        "obvious duplicates won't hit"
    )


@respx.mock
async def test_same_prompt_different_model_does_not_cross_hit(client):
    route = respx.post(UPSTREAM)
    route.side_effect = [
        httpx.Response(200, json=completion("mini answer")),
        httpx.Response(200, json=dict(completion("4o answer"), model="gpt-4o")),
    ]
    await _post(client, "what is the capital of france")
    body = req("what is the capital of france")
    body["model"] = "gpt-4o"
    r2 = await client.post("/v1/chat/completions", json=body)
    # Same prompt, different model: MUST NOT serve the mini answer.
    assert route.call_count == 2
    assert r2.json()["choices"][0]["message"]["content"] == "4o answer"


@respx.mock
async def test_high_temperature_bypasses_cache_entirely(client):
    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, json=completion("creative!"))
    )
    body = req("write me a poem about rivers")
    body["temperature"] = 1.0
    await client.post("/v1/chat/completions", json=body)
    await client.post("/v1/chat/completions", json=body)
    # Both went upstream; nothing was cached for a creative request.
    assert route.call_count == 2
    async with session() as s:
        assert (await s.execute(select(SemanticCache))).scalars().all() == []


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN LIMITATION: MiniLM scores direction flips ('transfer TO' vs "
    "'FROM savings') at ~0.99 — above any threshold that still allows "
    "paraphrase hits. Pure-embedding caching cannot separate these; "
    "documented in config.py. Mitigation: raise the threshold (fewer hits) "
    "or add lexical checks (out of Phase 3 scope).",
)
@respx.mock
async def test_direction_flip_limitation_is_documented(client):
    route = respx.post(UPSTREAM)
    route.side_effect = [
        httpx.Response(200, json=completion("moved TO savings")),
        httpx.Response(200, json=completion("moved FROM savings")),
    ]
    await _post(client, "transfer money to savings")
    await _post(client, "transfer money from savings")
    assert route.call_count == 2  # xfail: today this is 1 — the flip collides
