"""Upstream forwarding: retries, backoff, transparent pass-through, graceful fail."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.config import get_settings
from app.upstream import UpstreamError, forward_chat_completion

UPSTREAM = "https://upstream.test/v1/chat/completions"

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
}


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Neutralize sleeps so retry tests run instantly."""
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr("app.upstream.asyncio.sleep", _no_sleep)


@respx.mock
async def test_happy_path_forwards_and_returns_json():
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=_COMPLETION))
    status, payload, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    assert status == 200
    assert payload == _COMPLETION
    # The upstream key is attached, request body forwarded verbatim.
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer test-upstream-key"


@respx.mock
async def test_4xx_is_forwarded_not_retried():
    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad"}})
    )
    status, payload, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    assert status == 400
    assert route.call_count == 1  # 4xx is the client's problem — no retry
    assert payload["error"]["message"] == "bad"


@respx.mock
async def test_5xx_is_retried_then_succeeds():
    route = respx.post(UPSTREAM).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_COMPLETION),
        ]
    )
    status, _, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    assert status == 200
    assert route.call_count == 2


@respx.mock
async def test_transport_error_retried_then_raises_upstream_error():
    # max_retries default = 2 => 3 total tries, all failing.
    get_settings.cache_clear()
    route = respx.post(UPSTREAM).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(UpstreamError):
        await forward_chat_completion({"model": "m", "messages": []})
    assert route.call_count == get_settings().upstream_max_retries + 1


@respx.mock
async def test_exhausted_5xx_returns_last_response():
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(500, json={"error": "x"}))
    status, _, _, _ = await forward_chat_completion({"model": "m", "messages": []})
    # After exhausting retries we return the final 5xx transparently, not raise.
    assert status == 500
    assert route.call_count == get_settings().upstream_max_retries + 1
