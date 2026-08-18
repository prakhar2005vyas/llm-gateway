"""Phase 2 end-to-end: streaming through the live ASGI app.

Client side: the official openai SDK (unmodified, stream=True) and a raw
httpx SSE consumer. Upstream side: respx streaming a Devanagari completion
split mid-character across network chunks, with real inter-chunk delays so
"the client streams live" is measurable, not assumed.

DB side: after the response drains, the BackgroundTask writes the trace —
httpx's ASGITransport awaits background tasks, so the row is committed and
queryable right after the client call returns.
"""
from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal

import httpx
import pytest
import respx
from openai import AsyncOpenAI
from sqlalchemy import select

from app.db import dispose_db, init_db, session
from app.main import app
from app.models import ModelPrice, Trace

UPSTREAM = "https://upstream.test/v1/chat/completions"


def chunk_event(piece: str | None, finish: str | None = None, usage: dict | None = None) -> bytes:
    """A well-formed chat.completion.chunk SSE event (SDK-parseable)."""
    delta: dict = {}
    if piece is not None:
        delta["content"] = piece
    payload = {
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
        payload["choices"] = []
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


DONE = b"data: [DONE]\n\n"


def devanagari_split_chunks(delay: float = 0.03) -> tuple[list[bytes], "SlowStream"]:
    """The upstream byte plan: नमस्ते's event cut INSIDE the 3-byte न, plus a
    finish event, a usage event, and [DONE] — each arriving after `delay`."""
    ev_text = chunk_event("नमस्ते")
    cut = ev_text.find("न".encode()) + 1  # mid-character
    assert cut > 0
    with pytest.raises(UnicodeDecodeError):
        ev_text[:cut].decode("utf-8")  # prove the cut is hostile
    chunks = [
        ev_text[:cut],
        ev_text[cut:],
        chunk_event("!", finish="stop"),
        chunk_event(None, usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
        DONE,
    ]
    return chunks, SlowStream(chunks, delay)


class SlowStream(httpx.AsyncByteStream):
    """Streams chunks with a real delay between them — makes 'live' testable."""

    def __init__(self, chunks: list[bytes], delay: float):
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self):
        for c in self._chunks:
            yield c
            await asyncio.sleep(self._delay)


@pytest.fixture(autouse=True)
async def _db():
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


async def _sole_trace() -> Trace:
    async with session() as s:
        traces = list((await s.execute(select(Trace))).scalars().all())
    assert len(traces) == 1
    return traces[0]


@respx.mock
async def test_sdk_streams_devanagari_end_to_end_with_trace():
    """The Phase 2 'done when': SDK streams live, नमस्ते reassembles into the
    trace with usage-based exact cost and a measured TTFT."""
    _, slow = devanagari_split_chunks()
    respx.post(UPSTREAM).mock(
        return_value=httpx.Response(
            200, stream=slow, headers={"content-type": "text/event-stream"}
        )
    )

    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://gw")
    sdk = AsyncOpenAI(base_url="http://gw/v1", api_key="unused", http_client=http_client)

    pieces: list[str] = []
    stream = await sdk.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "greet me in Hindi"}],
        stream=True,
    )
    async for chunk in stream:
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                pieces.append(choice.delta.content)
    await http_client.aclose()

    # Client saw the text intact — the mid-character split was invisible.
    # (Liveness is NOT asserted here: ASGITransport buffers the full response,
    # so arrival timing through it is meaningless. The real-socket test below
    # — test_stream_is_live_over_real_tcp — owns that proof.)
    assert "".join(pieces) == "नमस्ते!"

    # The cold path wrote one complete trace.
    t = await _sole_trace()
    assert t.outcome == "ok"
    assert t.status_code == 200
    assert t.model_id == "gpt-4o-mini"
    # Fully reassembled Devanagari in the synthesized response body:
    assert t.response_body["choices"][0]["message"]["content"] == "नमस्ते!"
    assert t.response_body["gateway.reassembled"] is True
    # Usage came through the stream → cost is the exact Phase 1 number:
    assert (t.prompt_tokens, t.completion_tokens) == (100, 50)
    assert t.cost_usd == Decimal("0.00004500")
    # TTFT was measured and is coherent with total latency:
    assert t.ttft_ms is not None and t.ttft_ms >= 0
    assert t.latency_ms is not None and t.latency_ms >= t.ttft_ms
    # Four inter-chunk delays of 30ms mean total latency must exceed TTFT
    # by a real margin — TTFT is genuinely "first token", not "whole stream".
    assert t.latency_ms - t.ttft_ms >= 50


@respx.mock
async def test_raw_bytes_relayed_verbatim():
    """A non-SDK consumer gets byte-identical SSE, including [DONE]."""
    chunks, slow = devanagari_split_chunks(delay=0.0)
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=slow))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [], "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            received = b"".join([chunk async for chunk in resp.aiter_bytes()])

    assert received == b"".join(chunks)


@respx.mock
async def test_midstream_death_traces_inconclusive():
    class DiesAfterOne(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield chunk_event("नम")
            raise httpx.ReadError("reset mid-stream")

    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=DiesAfterOne()))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        async with c.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [], "stream": True},
        ) as resp:
            received = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert "नम".encode() in received  # partial data did reach the client

    t = await _sole_trace()
    assert t.outcome == "inconclusive"  # honest degraded state, not "ok"
    assert t.error_message and "interrupted" in t.error_message
    assert t.response_body["choices"][0]["message"]["content"] == "नम"
    assert t.cost_usd is None  # no usage arrived — unknown, not zero


@respx.mock
async def test_stream_connect_failure_returns_502_and_traces(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    respx.post(UPSTREAM).mock(side_effect=httpx.ConnectError("refused"))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [], "stream": True},
        )
    assert r.status_code == 502
    assert "unavailable" in r.json()["error"]["message"]

    t = await _sole_trace()
    assert t.outcome == "inconclusive"
    assert t.status_code == 502
    assert t.ttft_ms is None  # no first token ever arrived


async def test_stream_is_live_over_real_tcp(unused_tcp_port):
    """Prove tokens reach the client AS THEY ARE PRODUCED, not in one buffered
    burst at the end. ASGITransport can't show this (it buffers), so this test
    boots uvicorn in-process on a real socket. The upstream is still respx-
    mocked (same process/event loop); the client's localhost connection is
    explicitly passed through.
    """
    import uvicorn

    from app.main import app as gateway_app

    delay = 0.08
    _, slow = devanagari_split_chunks(delay=delay)

    server = uvicorn.Server(
        uvicorn.Config(gateway_app, host="127.0.0.1", port=unused_tcp_port, log_level="error")
    )
    server_task = asyncio.create_task(server.serve())
    try:
        with respx.mock:
            respx.route(host="127.0.0.1").pass_through()  # client → gateway: real TCP
            respx.post(UPSTREAM).mock(
                return_value=httpx.Response(200, stream=slow)
            )

            # Wait for the socket to accept. (Increase to 300 to prevent flakiness on slow CI/Windows)
            for _ in range(300):
                if server.started:
                    break
                await asyncio.sleep(0.02)
            assert server.started, "uvicorn did not start"

            arrival_times: list[float] = []
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{unused_tcp_port}") as c:
                async with c.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [], "stream": True},
                ) as resp:
                    assert resp.status_code == 200
                    async for _chunk in resp.aiter_raw():
                        arrival_times.append(time.perf_counter())

            # 5 upstream chunks, each `delay` apart. A buffering proxy would
            # deliver everything in one burst (spread ≈ 0); a live relay's
            # spread must span most of the upstream's production time.
            assert len(arrival_times) >= 3
            spread = arrival_times[-1] - arrival_times[0]
            assert spread >= delay * 2, (
                f"stream arrived in a burst (spread={spread:.3f}s) — "
                "the gateway is buffering instead of relaying live"
            )
    finally:
        server.should_exit = True
        await server_task


@respx.mock
async def test_upstream_4xx_streaming_request_returns_json_error():
    err = json.dumps({"error": {"message": "invalid api key"}}).encode()
    respx.post(UPSTREAM).mock(return_value=httpx.Response(401, content=err))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [], "stream": True},
        )
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["message"] == "invalid api key"

    t = await _sole_trace()
    assert t.outcome == "upstream_error"
    assert t.status_code == 401
