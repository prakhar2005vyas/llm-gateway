"""Phase 5 end-to-end: PII never leaves the gateway, in either direction.

The full lifecycle under test, both shapes:
  client sends RAW PII → gateway masks → upstream mock receives PLACEHOLDERS
  ONLY → mock echoes placeholders → gateway unmasks → client receives RAW
  values restored → the Trace row holds MASKED text exclusively.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx
from sqlalchemy import select

from app.db import dispose_db, init_db, session
from app.main import app
from app.models import Trace

UPSTREAM = "https://upstream.test/v1/chat/completions"

RAW_EMAIL = "alice@corp-secret.com"
RAW_CARD = "4111 1111 1111 1111"
RAW_PHONE = "+91 98765 43210"

PROMPT = (
    f"I'm {RAW_EMAIL}. Charge card {RAW_CARD} and text a receipt "
    f"to {RAW_PHONE}."
)


def _assert_no_raw_pii(blob: str, where: str) -> None:
    for raw in (RAW_EMAIL, RAW_CARD, RAW_PHONE):
        assert raw not in blob, f"RAW PII LEAKED into {where}: {raw!r}"


@pytest.fixture(autouse=True)
async def _db():
    await dispose_db()
    await init_db()
    yield
    await dispose_db()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        yield c


@respx.mock
async def test_non_streaming_full_pii_lifecycle(client):
    # The mock upstream echoes placeholders back — exactly what a model that
    # followed instructions would do.
    upstream_reply = {
        "id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant",
                        "content": "Done: receipt sent to <PHONE_1> for card "
                                   "<CARD_1>, confirmation to <EMAIL_1>."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
    }
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=upstream_reply))

    r = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini",
              "messages": [{"role": "user", "content": PROMPT}]},
    )
    assert r.status_code == 200

    # 1. THE OUTBOUND PROOF: the upstream saw placeholders, never raw PII.
    outbound = route.calls.last.request.content.decode("utf-8")
    _assert_no_raw_pii(outbound, "the upstream request")
    assert "<EMAIL_1>" in outbound and "<CARD_1>" in outbound and "<PHONE_1>" in outbound

    # 2. THE INBOUND PROOF: the client got raw values restored.
    content = r.json()["choices"][0]["message"]["content"]
    assert content == (
        f"Done: receipt sent to {RAW_PHONE} for card {RAW_CARD}, "
        f"confirmation to {RAW_EMAIL}."
    )

    # 3. THE TRACE PROOF: the stored row is masked-only, both directions.
    async with session() as s:
        trace = (await s.execute(select(Trace))).scalars().one()
    request_blob = json.dumps(trace.request_body)
    response_blob = json.dumps(trace.response_body)
    _assert_no_raw_pii(request_blob, "Trace.request_body")
    _assert_no_raw_pii(response_blob, "Trace.response_body")
    assert "<EMAIL_1>" in request_blob      # masked, not merely absent
    assert "<PHONE_1>" in response_blob
    assert trace.outcome == "ok" and trace.total_tokens == 50


@respx.mock
async def test_streaming_full_pii_lifecycle_with_split_placeholder(client):
    """The split-chunk case from the task: one upstream chunk ends mid-
    placeholder ('...<EMA'), the next completes it ('IL_1>...'). The client
    must receive whole raw values and never a fractured placeholder."""

    def chunk(payload: dict) -> bytes:
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    ev1 = chunk({"id": "s1", "object": "chat.completion.chunk", "created": 1,
                 "model": "gpt-4o-mini",
                 "choices": [{"index": 0,
                              "delta": {"content": "Emailing <EMAIL_1> about card <CARD_1> now"},
                              "finish_reason": None}]})
    ev2 = chunk({"id": "s1", "object": "chat.completion.chunk", "created": 1,
                 "model": "gpt-4o-mini",
                 "choices": [{"index": 0, "delta": {},
                              "finish_reason": "stop"}]})
    done = b"data: [DONE]\n\n"

    # Cut ev1 exactly mid-placeholder: '...Emailing <EMA' | 'IL_1> about...'
    cut = ev1.find(b"<EMAIL_1>") + 4
    assert ev1[cut - 4:cut] == b"<EMA"
    upstream_chunks = [ev1[:cut], ev1[cut:], ev2, done]

    class ChunkStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for c in upstream_chunks:
                yield c

    route = respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, stream=ChunkStream())
    )

    received: list[bytes] = []
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "stream": True,
              "messages": [{"role": "user", "content": PROMPT}]},
    ) as resp:
        assert resp.status_code == 200
        async for raw_chunk in resp.aiter_raw():
            received.append(raw_chunk)

    # 1. OUTBOUND: the provider saw placeholders only.
    outbound = route.calls.last.request.content.decode("utf-8")
    _assert_no_raw_pii(outbound, "the upstream stream request")
    assert "<EMAIL_1>" in outbound

    # 2. INBOUND: raw values restored in the client's stream...
    total = b"".join(received).decode("utf-8")
    assert f"Emailing {RAW_EMAIL} about card {RAW_CARD} now" in total
    assert "<EMAIL_1>" not in total and "<CARD_1>" not in total
    # ...and NO delivered chunk ends in a fractured placeholder signature.
    for c in received:
        s = c.decode("utf-8", errors="ignore")
        assert not s.endswith(("<", "<E", "<EM", "<EMA", "<EMAI", "<EMAIL",
                               "<EMAIL_", "<EMAIL_1", "<CARD", "<CARD_")), (
            f"client received fractured placeholder tail: {s[-20:]!r}"
        )

    # 3. TRACE: the reassembled stream trace is masked-only.
    async with session() as s_:
        trace = (await s_.execute(select(Trace))).scalars().one()
    request_blob = json.dumps(trace.request_body)
    response_blob = json.dumps(trace.response_body)
    _assert_no_raw_pii(request_blob, "Trace.request_body (stream)")
    _assert_no_raw_pii(response_blob, "Trace.response_body (stream)")
    assert "<EMAIL_1>" in response_blob  # harvested in placeholder space
    assert trace.outcome == "ok"


@respx.mock
async def test_non_pii_requests_still_byte_transparent(client):
    """No PII → the Phase 2 verbatim-relay contract is untouched."""
    body_bytes = b'data: {"model":"m","choices":[{"index":0,"delta":{"content":"2<3 ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

    class OneShot(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body_bytes

    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, stream=OneShot()))

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "stream": True,
              "messages": [{"role": "user", "content": "compare 2 and 3"}]},
    ) as resp:
        received = b"".join([c async for c in resp.aiter_raw()])
    assert received == body_bytes  # byte-identical, '<' and all
