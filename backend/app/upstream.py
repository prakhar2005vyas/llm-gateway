"""Upstream forwarding — the actual proxy hop.

Two paths, one provider (OpenAI-compatible), request body forwarded verbatim:

* `forward_chat_completion` — non-streaming: POST, await full JSON, return it.
* `forward_stream` — streaming (Phase 2): httpx `client.stream()`, relaying
  raw SSE bytes to the caller as they arrive while a ByteStreamProcessor
  taps a copy to reassemble the full response text for the cold-path trace.

Reliability (per CLAUDE.md): every external call gets a timeout and a bounded
number of retries with exponential backoff. For streams, retries apply ONLY
until the first byte of a successful response has been relayed — after that a
retry would replay tokens into a half-consumed client stream, so mid-stream
failures degrade loudly (StreamResult.error) instead of retrying. On repeated
connection failure we raise `UpstreamError` so the route can return a clean
502 rather than crashing the process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .config import get_settings
from .sse import ByteStreamProcessor

logger = logging.getLogger(__name__)

# Transport-level faults worth retrying (DNS, connect, read timeout, ...).
_RETRYABLE_EXC = (httpx.TransportError,)


class UpstreamError(Exception):
    """Raised when the upstream provider is unreachable after all retries."""


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 0.5s, 1s, 2s, ... capped at 8s."""
    return min(0.5 * (2**attempt), 8.0)


async def forward_chat_completion(body: dict) -> tuple[int, dict]:
    """Forward a chat-completions request upstream and return (status, json).

    Retries transient transport errors and 5xx responses up to
    `UPSTREAM_MAX_RETRIES` times. A non-5xx HTTP response (including 4xx) is
    forwarded transparently — the gateway does not mask the provider's own
    client errors.
    """
    settings = get_settings()
    url = settings.chat_completions_url
    headers = {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
    }
    max_tries = settings.upstream_max_retries + 1
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
        for attempt in range(max_tries):
            try:
                resp = await client.post(url, json=body, headers=headers)
            except _RETRYABLE_EXC as exc:
                last_exc = exc
                logger.warning(
                    "upstream network error (attempt %d/%d): %s",
                    attempt + 1,
                    max_tries,
                    exc,
                )
            else:
                # Retry server errors; forward everything else as-is.
                if resp.status_code >= 500 and attempt < max_tries - 1:
                    logger.warning(
                        "upstream %d (attempt %d/%d), retrying",
                        resp.status_code,
                        attempt + 1,
                        max_tries,
                    )
                else:
                    return resp.status_code, _safe_json(resp)

            if attempt < max_tries - 1:
                await asyncio.sleep(_backoff_seconds(attempt))

    raise UpstreamError(
        f"upstream unreachable after {max_tries} attempts: {last_exc}"
    ) from last_exc


def _safe_json(resp: httpx.Response) -> dict:
    """Parse JSON, degrading to a structured error rather than throwing."""
    try:
        return resp.json()
    except ValueError:
        logger.error("upstream returned non-JSON body (status %d)", resp.status_code)
        return {
            "error": {
                "message": "upstream returned a non-JSON response",
                "type": "gateway_error",
                "upstream_status": resp.status_code,
            }
        }


# ---------------------------------------------------------------------------
# Streaming (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    """Everything the cold path needs, filled in as the stream is consumed.

    Only meaningful once the byte generator has been fully drained (the
    route hands this object to the trace-writing BackgroundTask, which runs
    after the response — and therefore the generator — has finished).
    """

    status_code: int | None = None
    # Reassembled assistant text accumulates as a list of delta pieces and
    # joins on read. Attribute `str +=` is quadratic (CPython's in-place
    # realloc trick doesn't apply to attribute stores — measured 90× slower
    # at 100k chunks) and that CPU burn would land on the event loop while
    # relaying. `content` is a property below.
    _content_parts: list[str] = field(default_factory=list, repr=False)
    # model/usage/finish_reason harvested from the event stream. `usage` is
    # only present when the client asked via stream_options.include_usage —
    # absent means token counts are unknown (honest NULL), Phase 2 does not
    # guess.
    model_id: str | None = None
    usage: dict | None = None
    finish_reason: str | None = None
    # Set when the stream ended abnormally (mid-stream disconnect, invalid
    # UTF-8, non-JSON data lines...). The trace records outcome=inconclusive.
    error: str | None = None
    events_seen: int = 0
    # Filled by the route's TTFT wrapper: ms from request start to the first
    # byte relayed to the client (the Phase 2 headline metric).
    ttft_ms: int | None = None

    @property
    def content(self) -> str:
        """The reassembled assistant text (joined on demand)."""
        return "".join(self._content_parts)

    def harvest(self, event_text: str) -> None:
        """Fold one complete SSE event into the accumulated state.

        Parse failures are recorded, never raised — parsing exists for the
        trace copy; it must not be able to kill a healthy relay.
        """
        self.events_seen += 1
        for line in event_text.split("\n"):
            line = line.strip("\r")
            if not line.startswith("data:"):
                continue  # comments / event: / id: lines — not payload
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except ValueError:
                self.error = self.error or f"non-JSON data line in SSE event: {data[:80]!r}"
                logger.warning("stream: unparseable data line (len=%d)", len(data))
                continue
            if not isinstance(payload, dict):
                continue
            self.model_id = payload.get("model") or self.model_id
            if isinstance(payload.get("usage"), dict):
                self.usage = payload["usage"]
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    if isinstance(piece, str):
                        self._content_parts.append(piece)
                if choice.get("finish_reason"):
                    self.finish_reason = choice["finish_reason"]


async def forward_stream(body: dict, result: StreamResult) -> AsyncIterator[bytes]:
    """Open a streaming chat-completions call; yield raw bytes as they arrive.

    The caller wraps this generator in a StreamingResponse. Contract:

    * Raw upstream bytes are yielded VERBATIM — the client sees exactly what
      the provider sent. The ByteStreamProcessor works on a tap, so a parse
      bug can corrupt the trace copy but never the client's stream.
    * Retries (exponential backoff, same knobs as non-streaming) cover the
      CONNECTION phase only: transport errors before the response starts,
      and 5xx statuses (whose bodies we drain without relaying). Once the
      first byte is out, failures set `result.error` and end the stream —
      no retry can safely replay a half-relayed response.
    * A non-2xx final status yields the upstream's error body as plain JSON
      bytes (result.status_code carries it; the route sets the real HTTP
      status from it before streaming starts).
    * Raises UpstreamError only when the connection phase exhausted retries.
    """
    settings = get_settings()
    url = settings.chat_completions_url
    headers = {
        "Authorization": f"Bearer {settings.upstream_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    max_tries = settings.upstream_max_retries + 1
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
        for attempt in range(max_tries):
            try:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code >= 500 and attempt < max_tries - 1:
                        await resp.aread()  # drain; do NOT relay a retryable 5xx
                        logger.warning(
                            "stream upstream %d (attempt %d/%d), retrying",
                            resp.status_code, attempt + 1, max_tries,
                        )
                    elif resp.status_code >= 400:
                        # Terminal error status: relay body as-is, no retry.
                        result.status_code = resp.status_code
                        yield await resp.aread()
                        return
                    else:
                        result.status_code = resp.status_code
                        async for chunk in _relay(resp, result):
                            yield chunk
                        return
            except _RETRYABLE_EXC as exc:
                if result.status_code is not None:
                    # Bytes already relayed — mid-stream failure, no retry.
                    result.error = f"stream interrupted: {type(exc).__name__}: {exc}"
                    logger.error("stream interrupted after %d event(s): %s",
                                 result.events_seen, exc)
                    return
                last_exc = exc
                logger.warning(
                    "stream connect error (attempt %d/%d): %s",
                    attempt + 1, max_tries, exc,
                )
            if attempt < max_tries - 1:
                await asyncio.sleep(_backoff_seconds(attempt))

    raise UpstreamError(
        f"upstream unreachable after {max_tries} attempts: {last_exc}"
    ) from last_exc


async def _relay(resp: httpx.Response, result: StreamResult) -> AsyncIterator[bytes]:
    """Yield raw bytes; tap each chunk through the framer into `result`."""
    proc = ByteStreamProcessor()
    async for chunk in resp.aiter_bytes():
        yield chunk  # client first — real-time relay is the product
        try:
            for event in proc.feed(chunk):
                result.harvest(event)
        except UnicodeDecodeError as exc:
            # Trace copy is damaged; the client still got the raw bytes.
            result.error = result.error or f"invalid UTF-8 in stream: {exc}"
            logger.error("stream: invalid UTF-8 while reassembling: %s", exc)
            proc = ByteStreamProcessor()  # resync framer for later events
    try:
        tail = proc.flush()
    except UnicodeDecodeError as exc:
        result.error = result.error or f"stream ended mid-character: {exc}"
        logger.error("stream: truncated multibyte character at EOF: %s", exc)
        return
    if tail:
        result.harvest(tail)
