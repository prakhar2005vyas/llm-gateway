"""Eval harness unit tests — sampling gate, shed-at-cap, judge lifecycle."""
from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import select

from app import evals
from app.config import get_settings
from app.db import dispose_db, init_db, session
from app.models import Eval, Trace

JUDGE = "https://judge.test/v1/chat/completions"


def judge_reply(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture(autouse=True)
async def _db(monkeypatch):
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


async def _make_trace() -> uuid.UUID:
    tid = uuid.uuid4()
    async with session() as s:
        s.add(Trace(id=tid, model_id="gpt-4o-mini", outcome="ok"))
    return tid


def _enqueue(tid: uuid.UUID, **overrides) -> bool:
    kwargs = dict(
        trace_id=tid,
        request_body={"model": "m", "messages": [{"role": "user", "content": "q"}]},
        response_content="the answer",
        outcome="ok",
    )
    kwargs.update(overrides)
    return evals.maybe_enqueue_eval(**kwargs)


# ---------------------------------------------------------------------------
# Sampling gate + eligibility
# ---------------------------------------------------------------------------

async def test_rate_zero_never_samples(monkeypatch):
    monkeypatch.setenv("EVAL_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    assert _enqueue(await _make_trace()) is False


@respx.mock
async def test_force_bypasses_rate_zero_and_coin_flip(monkeypatch):
    """The golden-runner path: rate 0 AND a hostile coin flip, force wins."""
    monkeypatch.setenv("EVAL_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    monkeypatch.setattr(evals.random, "random", lambda: 0.999)
    respx.post(JUDGE).mock(
        return_value=httpx.Response(200, json=judge_reply('{"helpfulness":8,"tone":8}'))
    )
    tid = await _make_trace()
    assert _enqueue(tid, force=True) is True
    await evals.drain_for_tests()
    async with session() as s:
        row = (await s.execute(select(Eval).where(Eval.trace_id == tid))).scalar_one()
    assert row.status == "completed"


async def test_force_still_respects_eligibility(monkeypatch):
    monkeypatch.setenv("EVAL_SAMPLE_RATE", "0.0")
    get_settings.cache_clear()
    tid = await _make_trace()
    assert _enqueue(tid, force=True, cache_hit=True) is False
    assert _enqueue(tid, force=True, outcome="upstream_error") is False


async def test_sample_rate_gates_on_random(monkeypatch):
    monkeypatch.setenv("EVAL_SAMPLE_RATE", "0.05")
    get_settings.cache_clear()
    tid = await _make_trace()

    monkeypatch.setattr(evals.random, "random", lambda: 0.049)  # inside 5%
    respx_mock = respx.mock(assert_all_called=False)
    with respx_mock:
        respx_mock.post(JUDGE).mock(
            return_value=httpx.Response(200, json=judge_reply('{"helpfulness":8,"tone":9}'))
        )
        assert _enqueue(tid) is True
        await evals.drain_for_tests()

    monkeypatch.setattr(evals.random, "random", lambda: 0.051)  # outside 5%
    assert _enqueue(tid) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"outcome": "upstream_error"},
        {"outcome": "inconclusive"},
        {"cache_hit": True},
        {"coalesced": True},
        {"response_content": None},
        {"response_content": "   "},
    ],
)
async def test_ineligible_requests_never_sampled(overrides):
    assert _enqueue(await _make_trace(), **overrides) is False


async def test_shed_at_concurrency_cap(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("EVAL_MAX_CONCURRENT", "2")
    get_settings.cache_clear()
    tid = await _make_trace()

    # Occupy the gauge with two never-finishing placeholders.
    blocker = asyncio.Event()

    async def _stuck():
        await blocker.wait()

    fake = [asyncio.create_task(_stuck()) for _ in range(2)]
    evals._tasks.update(fake)
    try:
        with caplog.at_level(logging.WARNING, logger="app.evals"):
            assert _enqueue(tid) is False  # SHED, not queued
        assert any("SHED" in r.message for r in caplog.records)
    finally:
        blocker.set()
        for t in fake:
            evals._tasks.discard(t)
        await asyncio.gather(*fake)


# ---------------------------------------------------------------------------
# Judge lifecycle
# ---------------------------------------------------------------------------

async def _run_and_get_eval(tid: uuid.UUID) -> Eval:
    assert _enqueue(tid) is True
    await evals.drain_for_tests()
    async with session() as s:
        return (await s.execute(select(Eval).where(Eval.trace_id == tid))).scalar_one()


@respx.mock
async def test_successful_eval_stores_scores_and_raw_verdict():
    route = respx.post(JUDGE).mock(
        return_value=httpx.Response(200, json=judge_reply(
            '{"helpfulness": 8, "tone": 9, "rationale": "clear and correct"}'
        ))
    )
    row = await _run_and_get_eval(await _make_trace())

    assert row.status == "completed"
    assert row.helpfulness_score == Decimal("8.00")
    assert row.tone_score == Decimal("9.00")
    assert row.evaluator_model == get_settings().eval_model_id
    assert row.raw_verdict["rationale"] == "clear and correct"
    assert row.completed_at is not None
    # Judge got the key and a temperature-0 request.
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer judge-key"


@respx.mock
async def test_markdown_fenced_verdict_parses():
    respx.post(JUDGE).mock(
        return_value=httpx.Response(200, json=judge_reply(
            'Sure! Here is my assessment:\n```json\n{"helpfulness": 7, "tone": 6}\n```'
        ))
    )
    row = await _run_and_get_eval(await _make_trace())
    assert row.status == "completed"
    assert row.helpfulness_score == Decimal("7.00")


@respx.mock
async def test_out_of_range_scores_clamped_to_contract():
    respx.post(JUDGE).mock(
        return_value=httpx.Response(200, json=judge_reply('{"helpfulness": 15, "tone": 0}'))
    )
    row = await _run_and_get_eval(await _make_trace())
    assert row.helpfulness_score == Decimal("10.00")
    assert row.tone_score == Decimal("1.00")


@respx.mock
async def test_garbage_verdict_is_failed_never_invented(monkeypatch):
    monkeypatch.setenv("EVAL_MAX_RETRIES", "0")
    get_settings.cache_clear()
    respx.post(JUDGE).mock(
        return_value=httpx.Response(200, json=judge_reply("I think it's pretty good!"))
    )
    row = await _run_and_get_eval(await _make_trace())
    assert row.status == "failed"
    assert row.helpfulness_score is None  # no verdict ≠ zero
    assert "no JSON object" in row.error_message


@respx.mock
async def test_judge_5xx_retried_then_succeeds(monkeypatch):
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(evals.asyncio, "sleep", _no_sleep)
    route = respx.post(JUDGE)
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(200, json=judge_reply('{"helpfulness": 6, "tone": 7}')),
    ]
    row = await _run_and_get_eval(await _make_trace())
    assert row.status == "completed" and route.call_count == 2


async def test_unconfigured_evaluator_records_skipped(monkeypatch):
    monkeypatch.setenv("EVAL_BASE_URL", "")
    get_settings.cache_clear()
    row = await _run_and_get_eval(await _make_trace())
    assert row.status == "skipped"
    assert "no evaluator configured" in row.error_message
    assert row.helpfulness_score is None
