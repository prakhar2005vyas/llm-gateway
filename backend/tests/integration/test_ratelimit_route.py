"""429 behavior end-to-end: over-limit rejected cleanly, within-limit unharmed."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.config import get_settings
from app.db import dispose_db, init_db
from app.main import app
from app.ratelimit import limiter

UPSTREAM = "https://upstream.test/v1/chat/completions"

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"},
         "finish_reason": "stop"}
    ],
}


@pytest.fixture(autouse=True)
async def _env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    limiter.reset_for_tests()
    await dispose_db()
    await init_db()
    yield
    await dispose_db()
    get_settings.cache_clear()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        yield c


def _req(key: str):
    return {
        "url": "/v1/chat/completions",
        "json": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        "headers": {"Authorization": f"Bearer {key}"},
    }


@respx.mock
async def test_over_limit_gets_clean_429_without_touching_upstream(client):
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=COMPLETION))

    for _ in range(3):
        r = await client.post(**_req("sk-alice"))
        assert r.status_code == 200
    assert route.call_count == 3

    r = await client.post(**_req("sk-alice"))
    assert r.status_code == 429
    # OpenAI-shaped error + a usable Retry-After — the SDK backs off on this.
    assert r.json()["error"]["type"] == "rate_limit_error"
    assert int(r.headers["retry-after"]) >= 1
    # The rejected request never reached (or paid for) the upstream.
    assert route.call_count == 3


@respx.mock
async def test_within_limit_traffic_unaffected_by_noisy_neighbor(client):
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=COMPLETION))

    # Alice burns through her budget and beyond.
    for _ in range(6):
        await client.post(**_req("sk-alice"))
    # Bob's requests still sail through at full speed — no degradation.
    for _ in range(3):
        r = await client.post(**_req("sk-bob"))
        assert r.status_code == 200
    assert route.call_count == 3 + 3  # alice's 3 admitted + bob's 3
