"""forward_stream tests: verbatim relay, reassembly tap, retry boundaries.

respx streams a custom AsyncByteStream, preserving chunk boundaries exactly —
verified before writing these tests — so fragmentation scenarios here are
faithful to real TCP behavior.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.config import get_settings
from app.upstream import StreamResult, UpstreamError, forward_stream

UPSTREAM = "https://upstream.test/v1/chat/completions"


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr("app.upstream.asyncio.sleep", _no_sleep)


class ChunkStream(httpx.AsyncByteStream):
    """Replay exact byte chunks, simulating TCP fragmentation/coalescing."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for c in self._chunks:
            yield c


def sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def delta_event(piece: str, model: str = "gpt-4o-mini", finish: str | None = None) -> bytes:
    return sse(
        {
            "id": "c1",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": finish}],
        }
    )


DONE = b"data: [DONE]\n\n"


async def drain(body: dict, result: StreamResult) -> list[bytes]:
    return [chunk async for chunk in forward_stream(body, result)]


@respx.mock
async def test_relays_bytes_verbatim_and_reassembles_content():
    chunks = [delta_event("Hel"), delta_event("lo"), delta_event("!", finish="stop"), DONE]
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=ChunkStream(chunks)))

    result = StreamResult()
    got = await drain({"model": "gpt-4o-mini", "stream": True}, result)

    assert b"".join(got) == b"".join(chunks)  # byte-for-byte what upstream sent
    assert result.content == "Hello!"
    assert result.status_code == 200
    assert result.model_id == "gpt-4o-mini"
    assert result.finish_reason == "stop"
    assert result.error is None
    assert result.events_seen == 4  # 3 deltas + [DONE]


@respx.mock
async def test_devanagari_split_mid_character_across_chunks():
    """End-to-end fragmentation: an event whose न is split across two reads.
    Client bytes must be identical; reassembled content must be intact."""
    event = delta_event("नमस्ते")
    cut = event.find("न".encode()) + 1  # 1 byte into the 3-byte न
    chunks = [event[:cut], event[cut:], DONE]
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=ChunkStream(chunks)))

    result = StreamResult()
    got = await drain({"model": "m", "stream": True}, result)

    assert b"".join(got) == event + DONE
    assert result.content == "नमस्ते"
    assert result.error is None


@respx.mock
async def test_usage_harvested_when_present():
    usage_event = sse({"id": "c1", "model": "gpt-4o-mini", "choices": [],
                       "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}})
    chunks = [delta_event("hi", finish="stop"), usage_event, DONE]
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=ChunkStream(chunks)))

    result = StreamResult()
    await drain({"model": "m", "stream": True}, result)
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


@respx.mock
async def test_no_usage_stays_none_not_guessed():
    respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, stream=ChunkStream([delta_event("x", finish="stop"), DONE]))
    )
    result = StreamResult()
    await drain({"model": "m", "stream": True}, result)
    assert result.usage is None  # honest unknown — Phase 2 does not estimate


@respx.mock
async def test_connect_error_retries_then_raises():
    get_settings.cache_clear()
    route = respx.post(UPSTREAM).mock(side_effect=httpx.ConnectError("refused"))
    result = StreamResult()
    with pytest.raises(UpstreamError):
        await drain({"model": "m", "stream": True}, result)
    assert route.call_count == get_settings().upstream_max_retries + 1
    assert result.status_code is None  # nothing was ever relayed


@respx.mock
async def test_5xx_on_connect_is_retried_then_succeeds():
    route = respx.post(UPSTREAM)
    route.side_effect = [
        httpx.Response(503, content=b"overloaded"),
        httpx.Response(200, stream=ChunkStream([delta_event("ok", finish="stop"), DONE])),
    ]
    result = StreamResult()
    got = await drain({"model": "m", "stream": True}, result)
    assert route.call_count == 2
    assert result.content == "ok"
    # The 503 body was drained, never relayed to the client:
    assert b"overloaded" not in b"".join(got)


@respx.mock
async def test_4xx_relayed_verbatim_without_retry():
    err = json.dumps({"error": {"message": "bad key"}}).encode()
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(401, content=err))
    result = StreamResult()
    got = await drain({"model": "m", "stream": True}, result)
    assert route.call_count == 1
    assert b"".join(got) == err
    assert result.status_code == 401


@respx.mock
async def test_midstream_death_no_retry_marks_error():
    """Once bytes have flowed, a failure must NOT retry (tokens would replay);
    it ends the stream and records the degraded state."""

    class DiesMidStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield delta_event("par")
            raise httpx.ReadError("connection reset mid-stream")

    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, stream=DiesMidStream())
    )
    result = StreamResult()
    got = await drain({"model": "m", "stream": True}, result)

    assert route.call_count == 1  # crucially: no second attempt
    assert result.content == "par"  # what made it through was still harvested
    assert result.error is not None and "interrupted" in result.error
    assert b"par" in b"".join(got)


@respx.mock
async def test_non_json_data_line_degrades_trace_not_relay():
    chunks = [b"data: this is not json\n\n", delta_event("ok", finish="stop"), DONE]
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=ChunkStream(chunks)))
    result = StreamResult()
    got = await drain({"model": "m", "stream": True}, result)
    assert b"".join(got) == b"".join(chunks)  # relay untouched
    assert result.content == "ok"             # later events still harvested
    assert result.error and "non-JSON" in result.error


@respx.mock
async def test_multiple_choices_only_content_pieces_concatenated():
    # Tool-call deltas / role-only deltas have no content — must not break.
    role_only = sse({"model": "m", "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    chunks = [role_only, delta_event("a"), delta_event("b", finish="stop"), DONE]
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=ChunkStream(chunks)))
    result = StreamResult()
    await drain({"model": "m", "stream": True}, result)
    assert result.content == "ab"
