"""OpenAI-compatible chat-completions endpoint.

Phase 2: transparent pass-through for BOTH shapes. Non-streaming: forward,
return JSON. Streaming: relay raw SSE bytes as they arrive (StreamingResponse)
while a tap reassembles the full text for the trace.

Hot/cold seam (SPEC.md step 8) in both branches: the Trace write rides a
starlette BackgroundTask attached to the response. For streams, Starlette runs
it only after the generator is fully drained — so the StreamResult it reads is
complete, and the client never waits on Postgres.

TTFT: the route awaits the FIRST upstream chunk before constructing the
StreamingResponse. That timestamp is time-to-first-token, and it also means a
connect-phase failure still becomes a clean 502 JSON error (HTTP status must
be decided before the first byte is sent; it cannot change mid-stream).
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .. import cache, coalesce, ratelimit
from ..auth import require_gateway_key
from ..metrics import TTFT_LATENCY, normalize_model_name
from ..pii import StreamUnmasker, mask_request, unmask_payload
from ..tracing import record_stream_trace, record_trace
from ..upstream import (
    StreamResult,
    UpstreamError,
    forward_chat_completion,
    forward_stream,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions", dependencies=[Depends(require_gateway_key)])
async def chat_completions(request: Request):
    # SPEC lifecycle step 2: per-key rate limit, before any work is spent on
    # the request. The check is in-memory and awaits nothing — an over-limit
    # burst costs O(1) per rejection and cannot degrade within-limit traffic.
    decision = ratelimit.limiter.check(
        ratelimit.client_key(request.headers.get("authorization"))
    )
    if not decision.allowed:
        return _error(
            429,
            "rate limit exceeded: retry after "
            f"{decision.retry_after_seconds}s",
            "rate_limit_error",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )

    try:
        body = await request.json()
    except ValueError:
        return _error(400, "request body must be valid JSON", "invalid_request_error")

    if not isinstance(body, dict):
        return _error(400, "request body must be a JSON object", "invalid_request_error")

    # Golden-dataset runner support: forces the eval sampling gate open for
    # THIS request (rate/coin-flip bypassed; eligibility and shed cap still
    # apply). Single-tenant tool — see evals.maybe_enqueue_eval.
    force_eval = request.headers.get("x-gateway-force-eval") == "1"

    if body.get("stream"):
        return await _streamed(body)
    return await _non_streamed(body, force_eval=force_eval)


# ---------------------------------------------------------------------------
# Non-streaming (Phase 1 behavior, unchanged)
# ---------------------------------------------------------------------------

async def _non_streamed(body: dict, force_eval: bool = False) -> JSONResponse:
    started = time.perf_counter()

    # SPEC lifecycle steps 3-4: mask → embed → semantic cache lookup, all
    # before any upstream call. A hit short-circuits the request entirely:
    # the cached body goes straight back, zero upstream calls, $0. consult()
    # never raises — cache trouble degrades to a miss.
    decision = await cache.consult(body)
    if decision.hit is not None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return JSONResponse(
            status_code=200,
            content=decision.hit.response_body,
            headers={"x-gateway-cache": "hit"},
            background=BackgroundTask(
                record_trace,
                request_body=body,
                response_body=decision.hit.response_body,
                status_code=200,
                latency_ms=latency_ms,
                outcome="ok",
                cache_decision=decision,  # → cache_hit=True, cost=0, hit stats
            ),
        )

    # SPEC lifecycle step 3 (Phase 5): PII masking at the FORWARD BOUNDARY.
    # Cache/coalesce guards above ran on the RAW body (PII-bearing requests
    # are neither cached nor coalesced); from here on, only the MASKED body
    # exists: it goes upstream, into coalescing suppliers, and into traces.
    # The provider never sees raw PII; the map lives only in this request.
    masked_body, pii_map = mask_request(body)

    # SPEC lifecycle step 5: exact-match coalescing. Concurrent identical
    # (deterministic) requests collapse into ONE upstream call — the leader
    # forwards, followers await its future and share the result. get_or_run's
    # dict.setdefault is the atomic compare-and-set; a leader failure is
    # shared with every follower and lands in the same except-arm below.
    is_follower = False
    try:
        if coalesce.is_coalesceable(body):
            (status_code, payload), is_leader = await coalesce.coalescer.get_or_run(
                coalesce.request_key(body),
                lambda: forward_chat_completion(masked_body),
            )
            is_follower = not is_leader
        else:
            status_code, payload = await forward_chat_completion(masked_body)
    except UpstreamError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("forward failed: %s", exc)
        return _error(
            502,
            f"upstream provider unavailable: {exc}",
            "gateway_error",
            background=BackgroundTask(
                record_trace,
                request_body=masked_body,  # traces hold MASKED text only
                response_body=None,
                status_code=502,
                latency_ms=latency_ms,
                outcome="upstream_error",
                error_message=str(exc),
            ),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    # The client gets the UNMASKED copy; the trace task below binds `payload`
    # — the upstream's own placeholder-space dict, untouched — so the trace
    # stays masked without any extra bookkeeping.
    client_payload = unmask_payload(payload, pii_map)
    return JSONResponse(
        status_code=status_code,
        content=client_payload,
        headers={"x-gateway-cache": "miss"} if decision.cacheable else None,
        background=BackgroundTask(
            record_trace,
            request_body=masked_body,
            response_body=payload,
            status_code=status_code,
            latency_ms=latency_ms,
            outcome="ok" if status_code < 400 else "upstream_error",
            error_message=None,
            cache_decision=decision,  # miss + ok response → cold-path store
            coalesced=is_follower,  # follower → cost 0; leader carries spend
            force_eval=force_eval,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming (Phase 2)
# ---------------------------------------------------------------------------

async def _streamed(body: dict):
    started = time.perf_counter()
    result = StreamResult()

    # Phase 5: only the MASKED body leaves the gateway. StreamResult harvests
    # the provider's placeholder-space bytes, so the stream trace is masked
    # by construction; the client-facing relay below unmasks.
    masked_body, pii_map = mask_request(body)
    upstream_gen = forward_stream(masked_body, result)

    # Await the first chunk BEFORE building the response: this is both the
    # TTFT measurement point and the last moment the HTTP status can change.
    try:
        first_chunk = await anext(upstream_gen)
    except StopAsyncIteration:
        first_chunk = None  # upstream closed with an empty body
    except UpstreamError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("stream connect failed: %s", exc)
        result.error = str(exc)
        result.status_code = 502
        return _error(
            502,
            f"upstream provider unavailable: {exc}",
            "gateway_error",
            background=BackgroundTask(
                record_stream_trace,
                request_body=masked_body,
                result=result,
                latency_ms=latency_ms,
            ),
        )

    result.ttft_ms = int((time.perf_counter() - started) * 1000)
    status_code = result.status_code or 502
    # Phase 7: TTFT histogram — the streaming headline metric, observed at
    # the same instant it's measured. Model label from the CLIENT's request
    # (normalized: unknown ids collapse to one bounded label value).
    try:
        TTFT_LATENCY.labels(
            model=normalize_model_name(body.get("model"))
        ).observe(result.ttft_ms / 1000.0)
    except Exception as exc:  # noqa: BLE001 — metrics must never hurt the relay
        logger.error("TTFT metric update failed (non-fatal): %s: %s",
                     type(exc).__name__, exc)

    async def relay() -> AsyncIterator[bytes]:
        # No PII → verbatim byte relay (Phase 2 transparency, untouched).
        # With PII → every outbound chunk passes through the StreamUnmasker:
        # placeholders restored, partial-placeholder tails held back so the
        # client never sees a fractured placeholder. If the unmasker hits
        # genuinely invalid UTF-8, it degrades to verbatim passthrough for
        # the remainder (placeholders left visible beats a killed stream —
        # loud log either way).
        unmasker = StreamUnmasker(pii_map) if pii_map else None

        async def _out(chunk: bytes) -> bytes:
            nonlocal unmasker
            if unmasker is None:
                return chunk
            try:
                return unmasker.feed(chunk)
            except UnicodeDecodeError as exc:
                logger.error(
                    "stream unmasker hit invalid UTF-8 — degrading to "
                    "verbatim relay (placeholders may reach the client): %s",
                    exc,
                )
                unmasker = None
                return chunk

        if first_chunk is not None:
            if out := await _out(first_chunk):
                yield out
        async for chunk in upstream_gen:
            if out := await _out(chunk):
                yield out
        if unmasker is not None:
            try:
                if tail := unmasker.flush():
                    yield tail
            except UnicodeDecodeError as exc:
                logger.error("stream unmasker flush failed (tail dropped): %s", exc)

    # Non-2xx: upstream rejected the request before streaming (auth, quota,
    # bad model...). forward_stream yielded its error body as plain bytes —
    # relay it under the true status, as JSON, not as an event stream.
    media_type = "text/event-stream" if status_code < 400 else "application/json"

    async def trace_when_drained() -> None:
        # BackgroundTask args bind at construction time, but total latency
        # only exists once the stream has drained — so it's computed HERE,
        # when Starlette runs the task after the response completes.
        # allow_cache_store: masked bodies look PII-free, so the population
        # guard downstream can't see the collision hazard — the route can.
        await record_stream_trace(
            allow_cache_store=not pii_map,
            request_body=masked_body,
            result=result,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    return StreamingResponse(
        relay(),
        status_code=status_code,
        media_type=media_type,
        background=BackgroundTask(trace_when_drained),
    )


def _error(
    status_code: int,
    message: str,
    err_type: str,
    background: BackgroundTask | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return an OpenAI-shaped error object so SDK clients parse it cleanly."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type}},
        background=background,
        headers=headers,
    )
