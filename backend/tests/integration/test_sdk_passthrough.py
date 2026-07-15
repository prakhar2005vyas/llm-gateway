"""Acceptance test for Phase 0.

The official `openai` SDK, unmodified, talks to the gateway and gets a valid
completion — it cannot tell it isn't talking to OpenAI directly. Only the
`base_url` changes; the in-memory ASGI transport is a test harness detail so we
never open a socket or hit a real provider. The upstream provider is mocked with
respx (respx intercepts the gateway's own httpx call, not the in-process
ASGI transport).
"""
from __future__ import annotations

import httpx
import pytest
import respx
from openai import AsyncOpenAI

from app.main import app

UPSTREAM = "https://upstream.test/v1/chat/completions"

_UPSTREAM_COMPLETION = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
}


@pytest.fixture
async def gateway_client():
    """An unmodified AsyncOpenAI client whose base_url points at the gateway."""
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://gateway.test")
    client = AsyncOpenAI(
        base_url="http://gateway.test/v1",
        api_key="client-does-not-need-a-real-key-in-phase-0",
        http_client=http_client,
    )
    yield client
    await http_client.aclose()


@respx.mock
async def test_sdk_gets_valid_completion(gateway_client):
    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, json=_UPSTREAM_COMPLETION)
    )

    completion = await gateway_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello."}],
    )

    # Response is forwarded unchanged and parses back into SDK objects.
    assert completion.choices[0].message.content == "Hello there!"
    assert completion.model == "gpt-4o-mini"
    assert completion.usage.total_tokens == 12

    # The gateway actually forwarded to the upstream, verbatim body.
    assert route.called
    forwarded = route.calls.last.request
    assert b'"model":"gpt-4o-mini"' in forwarded.content.replace(b" ", b"")


@respx.mock
async def test_upstream_down_yields_clean_502_not_a_crash(gateway_client, monkeypatch):
    # Keep the test fast: no retry backoff sleeps for this failure path.
    from app.config import get_settings

    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "0")
    get_settings.cache_clear()

    respx.post(UPSTREAM).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(Exception) as excinfo:
        await gateway_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            timeout=30,
        )
    # openai SDK raises APIStatusError for the 502 the gateway returns.
    assert "502" in str(excinfo.value) or "unavailable" in str(excinfo.value).lower()
