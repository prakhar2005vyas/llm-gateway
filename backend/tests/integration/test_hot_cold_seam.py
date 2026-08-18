"""Hot/cold seam tests.

httpx's ASGITransport awaits background tasks as part of the request cycle,
so after the client call returns, the trace row (written by the BackgroundTask)
is already committed — convenient for asserting on it deterministically.
"""
from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import select

from app.db import dispose_db, init_db, session
from app.main import app
from app.models import ModelPrice, Trace

UPSTREAM = "https://upstream.test/v1/chat/completions"

_UPSTREAM_COMPLETION = {
    "id": "chatcmpl-seam1",
    "object": "chat.completion",
    "model": "gpt-4o-mini-2024-07-18",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
}


@pytest.fixture(autouse=True)
async def _db():
    """Fresh in-memory schema per test, seeded with one price row."""
    await dispose_db()
    await init_db()
    async with session() as s:
        s.add(
            ModelPrice(
                model_id="gpt-4o-mini",
                usd_per_1k_input=Decimal("0.00015"),
                usd_per_1k_output=Decimal("0.0006"),
                source="test seed",
            )
        )
    yield
    await dispose_db()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        yield c


async def _all_traces() -> list[Trace]:
    async with session() as s:
        return list((await s.execute(select(Trace))).scalars().all())


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for c in self._chunks:
            yield c


def _sse(payload: dict) -> bytes:
    import json
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def _delta_event(piece: str, finish: str | None = None, model: str = "gpt-4o-mini-2024-07-18") -> bytes:
    return _sse(
        {
            "id": "c1",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": finish}],
        }
    )


DONE = b"data: [DONE]\n\n"


@respx.mock
async def test_success_writes_trace_with_hand_computed_cost(client):
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=_UPSTREAM_COMPLETION))

    r = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"

    traces = await _all_traces()
    assert len(traces) == 1
    t = traces[0]
    assert t.outcome == "ok"
    assert t.status_code == 200
    # Priced by the REQUESTED id (the price-table key), not the snapshot id
    # the provider echoed back.
    assert t.model_id == "gpt-4o-mini"
    assert (t.prompt_tokens, t.completion_tokens, t.total_tokens) == (100, 50, 150)
    # 100/1000*0.00015 + 50/1000*0.0006 = 0.000015 + 0.00003 = 0.000045
    assert t.cost_usd == Decimal("0.00004500")
    assert t.latency_ms is not None and t.latency_ms >= 0
    assert t.request_body["messages"][0]["content"] == "hello"
    assert t.response_body["id"] == "chatcmpl-seam1"


@respx.mock
async def test_unpriced_model_traces_with_null_cost(client):
    body = dict(_UPSTREAM_COMPLETION, model="mystery-9000")
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=body))

    r = await client.post(
        "/v1/chat/completions", json={"model": "mystery-9000", "messages": []}
    )
    assert r.status_code == 200

    (t,) = await _all_traces()
    assert t.model_id == "mystery-9000"
    assert t.cost_usd is None  # unknown, not zero
    assert t.total_tokens == 150  # usage still captured


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
async def test_rewritten_model_cost_calculation(client, monkeypatch, stream):
    from app.config import get_settings
    monkeypatch.setenv("UPSTREAM_MODEL_ID", "llama-3.3-70b-versatile")
    get_settings.cache_clear()
    
    # We add the price row for the REWRITTEN model.
    async with session() as s:
        s.add(
            ModelPrice(
                model_id="llama-3.3-70b-versatile",
                usd_per_1k_input=Decimal("0.00059"),
                usd_per_1k_output=Decimal("0.00079"),
                source="test seed",
            )
        )

    req_body = {"model": "llama-3.1-8b-instant", "messages": [], "stream": stream}
    if stream:
        # Provider responds with a snapshot ID in the chunks
        chunks = [
            _delta_event("hi", model="llama-3.3-70b-versatile-0102"), 
            _sse({"id": "c1", "model": "llama-3.3-70b-versatile-0102", "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}), 
            DONE
        ]
        respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=ChunkStream(chunks)))
    else:
        resp_body = dict(_UPSTREAM_COMPLETION, model="llama-3.3-70b-versatile-0102")
        respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=resp_body))

    r = await client.post("/v1/chat/completions", json=req_body)
    assert r.status_code == 200

    (t,) = await _all_traces()
    # model_id in the trace MUST be the rewritten string ("llama-3.3-70b-versatile")
    # because that's what we have a price row for, NOT the snapshot ID, and NOT the client's string.
    assert t.model_id == "llama-3.3-70b-versatile"
    assert t.cost_usd == Decimal("0.00009850") # 100*0.00059 + 50*0.00079 = 0.059 + 0.0395 = 0.0985 per 1k = 0.0000985


@respx.mock
async def test_rewritten_model_without_price_warns_with_rewritten_name(client, monkeypatch, caplog):
    from app.config import get_settings
    import logging
    monkeypatch.setenv("UPSTREAM_MODEL_ID", "mystery-rewritten-model")
    get_settings.cache_clear()
    
    resp_body = dict(_UPSTREAM_COMPLETION, model="mystery-rewritten-model-snapshot")
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=resp_body))

    with caplog.at_level(logging.WARNING, logger="app.costs"):
        r = await client.post("/v1/chat/completions", json={"model": "google/gemma-4-26B-A4B-it", "messages": []})
    
    assert r.status_code == 200
    (t,) = await _all_traces()
    assert t.model_id == "mystery-rewritten-model"
    assert t.cost_usd is None
    
    # Must log with the REWRITTEN name ("mystery-rewritten-model"), not the original or snapshot.
    assert any("no price row for model 'mystery-rewritten-model'" in rec.message for rec in caplog.records)


@respx.mock
async def test_upstream_failure_still_leaves_a_trace(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    respx.post(UPSTREAM).mock(side_effect=httpx.ConnectError("refused"))

    r = await client.post(
        "/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []}
    )
    assert r.status_code == 502

    (t,) = await _all_traces()
    assert t.outcome == "upstream_error"
    assert t.status_code == 502
    assert t.response_body is None
    assert t.error_message and "refused" in t.error_message
    assert t.cost_usd is None


@respx.mock
async def test_db_failure_never_touches_the_client_response(client, monkeypatch, caplog):
    """The failure contract: trace write blows up → client still gets its 200."""
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=_UPSTREAM_COMPLETION))

    # Break session acquisition inside the background task only.
    import app.tracing as tracing_mod

    class _BrokenSession:
        def __call__(self):
            return self

        async def __aenter__(self):
            raise RuntimeError("simulated: postgres is on fire")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(tracing_mod, "session", _BrokenSession())

    import logging

    with caplog.at_level(logging.ERROR, logger="app.tracing"):
        r = await client.post(
            "/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": []}
        )

    # Client is whole:
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"
    # ...and the failure was loud, not silent:
    assert any("TRACE WRITE FAILED" in rec.message for rec in caplog.records)
    # ...and no PII/body content leaked into the log line:
    assert not any("hello" in rec.getMessage() for rec in caplog.records)


async def test_real_db_error_does_not_leak_bound_bodies_into_logs(client, caplog):
    """Regression (found in Phase 1 review): a failed INSERT raises a DBAPI
    error whose str() embeds the bound parameters — i.e. the full request/
    response bodies — unless the engine hides them. Force a genuine
    constraint failure during the trace INSERT and assert the PII stays out
    of the log line (hide_parameters=True in db.py is what this pins)."""
    import logging

    from sqlalchemy import text

    from app.db import get_engine
    from app.tracing import record_trace

    # Rebuild traces with a CHECK that every real INSERT violates.
    eng = get_engine()
    async with eng.begin() as conn:
        await conn.execute(text("DROP TABLE traces"))
        await conn.execute(
            text(
                """
                CREATE TABLE traces (
                    id CHAR(32) PRIMARY KEY, created_at TIMESTAMP,
                    request_body JSON, response_body JSON,
                    model_id VARCHAR(200),
                    status_code INTEGER CHECK (status_code < 0),
                    latency_ms INTEGER, prompt_tokens INTEGER,
                    completion_tokens INTEGER, total_tokens INTEGER,
                    cost_usd NUMERIC(12,8),
                    outcome VARCHAR(32) NOT NULL, error_message TEXT
                )
                """
            )
        )

    with caplog.at_level(logging.ERROR, logger="app.tracing"):
        await record_trace(
            request_body={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "SSN 123-45-6789 SECRET_PII"}],
            },
            response_body={"choices": [{"message": {"content": "SECRET_REPLY"}}]},
            status_code=200,
            latency_ms=1,
            outcome="ok",
        )

    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "TRACE WRITE FAILED" in joined  # loud degraded state...
    assert "SECRET_PII" not in joined      # ...with zero body leakage
    assert "SECRET_REPLY" not in joined
    assert "[parameters:" not in joined    # the DBAPI param dump is hidden
