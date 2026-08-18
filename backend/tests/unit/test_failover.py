"""Failover tests — provider chain semantics, non-streaming and streaming."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.config import get_settings
from app.upstream import (
    StreamResult,
    UpstreamError,
    forward_chat_completion,
    forward_stream,
)

PRIMARY = "https://upstream.test/v1/chat/completions"
FAILOVER = "https://fireworks.test/v1/chat/completions"

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "from failover"},
            "finish_reason": "stop",
        }
    ],
}


@pytest.fixture(autouse=True)
def _failover_env(monkeypatch):
    monkeypatch.setenv("FAILOVER_BASE_URL", "https://fireworks.test/v1")
    monkeypatch.setenv("FAILOVER_API_KEY", "fw-test-key")
    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "1")  # keep retry math small
    get_settings.cache_clear()

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr("app.upstream.asyncio.sleep", _no_sleep)
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------

@respx.mock
async def test_dead_primary_transport_fails_over():
    primary = respx.post(PRIMARY).mock(side_effect=httpx.ConnectError("refused"))
    failover = respx.post(FAILOVER).mock(return_value=httpx.Response(200, json=COMPLETION))

    status, payload, _, _ = await forward_chat_completion({"model": "m", "messages": []})

    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "from failover"
    assert primary.call_count == 2  # full retry budget spent on primary first
    assert failover.call_count == 1
    # The failover call carries the FAILOVER key, not the primary's.
    sent = failover.calls.last.request
    assert sent.headers["authorization"] == "Bearer fw-test-key"


@respx.mock
async def test_persistent_5xx_fails_over():
    primary = respx.post(PRIMARY).mock(return_value=httpx.Response(503))
    failover = respx.post(FAILOVER).mock(return_value=httpx.Response(200, json=COMPLETION))

    status, _, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    assert status == 200
    assert primary.call_count == 2  # 503, retried 503, then chain advance
    assert failover.call_count == 1


@respx.mock
async def test_4xx_never_fails_over():
    primary = respx.post(PRIMARY).mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    failover = respx.post(FAILOVER).mock(return_value=httpx.Response(200, json=COMPLETION))

    status, payload, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    assert status == 401
    assert payload["error"]["message"] == "bad key"
    assert failover.call_count == 0  # client errors are not retargeted


@respx.mock
async def test_both_providers_dead_raises_upstream_error():
    primary = respx.post(PRIMARY).mock(side_effect=httpx.ConnectError("down"))
    failover = respx.post(FAILOVER).mock(side_effect=httpx.ConnectError("also down"))

    with pytest.raises(UpstreamError, match="all 2 provider"):
        await forward_chat_completion({"model": "m", "messages": []})
    assert primary.call_count == 2 and failover.call_count == 2


@respx.mock
async def test_both_providers_5xx_returns_last_body_transparently():
    respx.post(PRIMARY).mock(return_value=httpx.Response(503, json={"error": "p"}))
    respx.post(FAILOVER).mock(return_value=httpx.Response(502, json={"error": "f"}))

    status, payload, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    assert status == 502  # the last provider's honest answer, not a synthetic
    assert payload == {"error": "f"}


@respx.mock
async def test_no_failover_configured_preserves_phase0_behavior(monkeypatch):
    monkeypatch.setenv("FAILOVER_BASE_URL", "")
    get_settings.cache_clear()
    primary = respx.post(PRIMARY).mock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(UpstreamError):
        await forward_chat_completion({"model": "m", "messages": []})
    assert primary.call_count == 2


@respx.mock
async def test_failover_model_id_rewrites_body_and_returns_rewritten_model(monkeypatch):
    monkeypatch.setenv("UPSTREAM_MODEL_ID", "primary-model")
    monkeypatch.setenv("FAILOVER_MODEL_ID", "failover-model")
    get_settings.cache_clear()
    
    primary = respx.post(PRIMARY).mock(side_effect=httpx.ConnectError("refused"))
    failover = respx.post(FAILOVER).mock(return_value=httpx.Response(200, json=COMPLETION))

    status, payload, provider_name, rewritten_model = await forward_chat_completion(
        {"model": "client-model", "messages": []}
    )

    assert status == 200
    assert provider_name == "failover"
    assert rewritten_model == "failover-model"  # Must match the failover's target model
    
    # Assert actual requests
    assert json.loads(primary.calls.last.request.content)["model"] == "primary-model"
    assert json.loads(failover.calls.last.request.content)["model"] == "failover-model"


# ---------------------------------------------------------------------------
# Streaming (connect phase only)
# ---------------------------------------------------------------------------

class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self):
        for c in self._chunks:
            yield c


def sse_event(content: str) -> bytes:
    payload = {"model": "m", "choices": [{"index": 0, "delta": {"content": content},
                                          "finish_reason": "stop"}]}
    return b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"


@respx.mock
async def test_stream_connect_failure_fails_over():
    primary = respx.post(PRIMARY).mock(side_effect=httpx.ConnectError("refused"))
    failover = respx.post(FAILOVER).mock(
        return_value=httpx.Response(200, stream=ChunkStream([sse_event("hi")]))
    )

    result = StreamResult()
    chunks = [c async for c in forward_stream({"model": "m", "stream": True}, result)]

    assert primary.call_count == 2 and failover.call_count == 1
    assert result.content == "hi"
    assert result.status_code == 200
    assert b"hi" in b"".join(chunks)


@respx.mock
async def test_midstream_death_does_NOT_fail_over():
    """Replay safety: once bytes reached the client, a provider switch would
    restart the completion into a half-consumed stream. Mid-stream death ends
    the stream with a degraded trace — failover must NOT trigger."""

    class DiesMidStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"par"}}]}\n\n'
            raise httpx.ReadError("reset")

    primary = respx.post(PRIMARY).mock(
        return_value=httpx.Response(200, stream=DiesMidStream())
    )
    failover = respx.post(FAILOVER).mock(
        return_value=httpx.Response(200, stream=ChunkStream([sse_event("nope")]))
    )

    result = StreamResult()
    _ = [c async for c in forward_stream({"model": "m", "stream": True}, result)]

    assert primary.call_count == 1
    assert failover.call_count == 0  # THE assertion
    assert result.error and "interrupted" in result.error
    assert result.content == "par"


@respx.mock
async def test_stream_failover_model_id_rewrites_body_and_sets_rewritten_model(monkeypatch):
    monkeypatch.setenv("UPSTREAM_MODEL_ID", "primary-model")
    monkeypatch.setenv("FAILOVER_MODEL_ID", "failover-model")
    get_settings.cache_clear()

    primary = respx.post(PRIMARY).mock(side_effect=httpx.ConnectError("refused"))
    failover = respx.post(FAILOVER).mock(
        return_value=httpx.Response(200, stream=ChunkStream([sse_event("hi")]))
    )

    result = StreamResult()
    chunks = [c async for c in forward_stream({"model": "client-model", "stream": True}, result)]

    assert primary.call_count == 2 and failover.call_count == 1
    assert result.content == "hi"
    assert result.status_code == 200
    assert result.provider == "failover"
    assert result.rewritten_model == "failover-model"  # Must match the failover's target model
    
    # Assert actual requests
    assert json.loads(primary.calls.last.request.content)["model"] == "primary-model"
    assert json.loads(failover.calls.last.request.content)["model"] == "failover-model"
