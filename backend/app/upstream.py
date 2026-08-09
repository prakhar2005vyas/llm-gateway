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
from .metrics import FAILOVER_COUNT, normalize_model_name
from .sse import ByteStreamProcessor

logger = logging.getLogger(__name__)

# Transport-level faults worth retrying (DNS, connect, read timeout, ...).
_RETRYABLE_EXC = (httpx.TransportError,)



def _rewrite_model(body: dict) -> dict:
    """Return a copy of body with the model field normalised for the upstream.

    * UPSTREAM_MODEL_ID set   → force every request onto that model. One
      unconditional override catches all legacy model strings (Fireworks
      accounts/*/models/*, OpenAI ids, vision strings, …) so nothing
      unexpected reaches the provider. A real deployment pins its model this
      way: docker-compose.yml sets UPSTREAM_MODEL_ID for prod/demo.
    * UPSTREAM_MODEL_ID empty → passthrough: forward the client's model string
      verbatim. Never hardcode a model id here (CLAUDE.md: everything is
      config-driven); an unconfigured gateway stays a transparent pipe.
    """
    settings = get_settings()
    target = settings.upstream_model_id
    if not target:
        return body  # passthrough — no override configured

    incoming = body.get("model")
    if incoming != target:
        logger.info("model rewrite: '%s' -> '%s'", incoming, target)

    return {**body, "model": target}


class UpstreamError(Exception):
    """Raised when the upstream provider is unreachable after all retries."""


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 0.5s, 1s, 2s, ... capped at 8s."""
    return min(0.5 * (2**attempt), 8.0)


def _count_failover(body: dict) -> None:
    """Phase 7: bump the retry/failover counter. Called exactly where a
    retry is scheduled or a provider failover is logged. Never raises —
    metrics must never affect forwarding."""
    try:
        FAILOVER_COUNT.labels(model=normalize_model_name(body.get("model"))).inc()
    except Exception as exc:  # noqa: BLE001
        logger.error("failover metric update failed (non-fatal): %s: %s",
                     type(exc).__name__, exc)


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible target. Same schema = swap base URL + key."""

    name: str
    base_url: str
    api_key: str

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def providers() -> list[Provider]:
    """Failover chain: primary, then the optional secondary (Phase 4).

    Each provider gets its FULL retry budget before the chain advances.
    Cross-schema targets (Anthropic) need a translation layer — DOCUMENT-ONLY.
    """
    s = get_settings()
    chain = [Provider("primary", s.upstream_base_url, s.upstream_api_key)]
    if s.failover_base_url:
        chain.append(Provider("failover", s.failover_base_url, s.failover_api_key))
    return chain


async def forward_chat_completion(body: dict) -> tuple[int, dict]:
    """Forward through the provider chain and return (status, json).

    Per provider: retries transport errors and 5xx with backoff (Phase 0
    behavior, unchanged). When the PRIMARY fails completely — transport
    errors exhausted, or still 5xx on its final attempt — the request
    reroutes to the failover provider, which gets its own retry budget.

    A 4xx NEVER advances the chain: it's the client's error and retargeting
    it would just replay a bad request (and mask the provider's message).
    Only when every provider is down: the last 5xx body is returned
    transparently if one exists, else UpstreamError → the route's clean 502.
    """
    last_exc: Exception | None = None
    last_5xx: tuple[int, dict] | None = None
    body = _rewrite_model(body)
    chain = providers()

    for provider in chain:
        is_last = provider is chain[-1]
        try:
            status, payload = await _forward_via(provider, body)
        except UpstreamError as exc:
            last_exc = exc
            if not is_last:
                logger.error(
                    "provider %r unreachable after all retries — FAILING OVER: %s",
                    provider.name, exc,
                )
                _count_failover(body)
            continue
        if status >= 500 and not is_last:
            last_5xx = (status, payload)
            logger.error(
                "provider %r still %d after all retries — FAILING OVER",
                provider.name, status,
            )
            _count_failover(body)
            continue
        if provider.name != "primary":
            logger.warning("request served by %r provider", provider.name)
        return status, payload

    if last_5xx is not None:
        return last_5xx
    raise UpstreamError(
        f"all {len(chain)} provider(s) unreachable: {last_exc}"
    ) from last_exc


async def _forward_via(provider: Provider, body: dict) -> tuple[int, dict]:
    """One provider, full retry budget (transport errors + 5xx, backoff)."""
    settings = get_settings()
    max_tries = settings.upstream_max_retries + 1
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
        for attempt in range(max_tries):
            try:
                resp = await client.post(
                    provider.chat_url, json=body, headers=provider.headers
                )
            except _RETRYABLE_EXC as exc:
                last_exc = exc
                logger.warning(
                    "%s network error (attempt %d/%d): %s",
                    provider.name, attempt + 1, max_tries, exc,
                )
            else:
                # Retry server errors; forward everything else as-is.
                if resp.status_code >= 500 and attempt < max_tries - 1:
                    logger.warning(
                        "%s %d (attempt %d/%d), retrying",
                        provider.name, resp.status_code, attempt + 1, max_tries,
                    )
                else:
                    return resp.status_code, _safe_json(resp)

            if attempt < max_tries - 1:
                _count_failover(body)  # a retry is now committed to
                await asyncio.sleep(_backoff_seconds(attempt))

    raise UpstreamError(
        f"{provider.name} unreachable after {max_tries} attempts: {last_exc}"
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
    * Failover (Phase 4): applies to the CONNECTION phase only, under the
      exact same replay-safety rule as retries — once the first byte has been
      relayed, neither retry nor provider switch is possible. A dead primary
      (transport-exhausted or final-5xx) hands off to the failover provider
      with a fresh retry budget. 4xx never advances the chain.
    * Raises UpstreamError only when every provider exhausted its retries.
    """
    settings = get_settings()
    max_tries = settings.upstream_max_retries + 1
    last_exc: Exception | None = None
    body = _rewrite_model(body)
    chain = providers()

    for provider in chain:
        is_last = provider is chain[-1]
        headers = dict(provider.headers, Accept="text/event-stream")
        failed_over = False

        async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
            for attempt in range(max_tries):
                try:
                    async with client.stream(
                        "POST", provider.chat_url, json=body, headers=headers
                    ) as resp:
                        if resp.status_code >= 500 and attempt < max_tries - 1:
                            await resp.aread()  # drain; do NOT relay a retryable 5xx
                            logger.warning(
                                "stream %s %d (attempt %d/%d), retrying",
                                provider.name, resp.status_code, attempt + 1, max_tries,
                            )
                        elif resp.status_code >= 500 and not is_last:
                            # Provider exhausted on 5xx → next provider.
                            await resp.aread()
                            logger.error(
                                "stream: provider %r still %d after all retries "
                                "— FAILING OVER", provider.name, resp.status_code,
                            )
                            _count_failover(body)
                            failed_over = True
                            break
                        elif resp.status_code >= 400:
                            # Terminal error status: relay body as-is, no retry.
                            result.status_code = resp.status_code
                            yield await resp.aread()
                            return
                        else:
                            if provider.name != "primary":
                                logger.warning(
                                    "stream served by %r provider", provider.name
                                )
                            result.status_code = resp.status_code
                            async for chunk in _relay(resp, result):
                                yield chunk
                            return
                except _RETRYABLE_EXC as exc:
                    if result.status_code is not None:
                        # Bytes already relayed — mid-stream failure: no
                        # retry, no failover (replay would corrupt the client).
                        result.error = f"stream interrupted: {type(exc).__name__}: {exc}"
                        logger.error("stream interrupted after %d event(s): %s",
                                     result.events_seen, exc)
                        return
                    last_exc = exc
                    logger.warning(
                        "stream %s connect error (attempt %d/%d): %s",
                        provider.name, attempt + 1, max_tries, exc,
                    )
                if attempt < max_tries - 1:
                    _count_failover(body)  # a retry is now committed to
                    await asyncio.sleep(_backoff_seconds(attempt))

        if not is_last and not failed_over:
            logger.error(
                "stream: provider %r unreachable after all retries — FAILING OVER: %s",
                provider.name, last_exc,
            )
            _count_failover(body)

    raise UpstreamError(
        f"all {len(chain)} provider(s) unreachable for stream: {last_exc}"
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
