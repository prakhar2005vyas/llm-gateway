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

    if body.get("stream"):
        return await _streamed(body)
    return await _non_streamed(body)


# ---------------------------------------------------------------------------
# Non-streaming (Phase 1 behavior, unchanged)
# ---------------------------------------------------------------------------

async def _non_streamed(body: dict) -> JSONResponse:
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
                lambda: forward_chat_completion(body),
            )
            is_follower = not is_leader
        else:
            status_code, payload = await forward_chat_completion(body)
    except UpstreamError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("forward failed: %s", exc)
        return _error(
            502,
            f"upstream provider unavailable: {exc}",
            "gateway_error",
            background=BackgroundTask(
                record_trace,
                request_body=body,
                response_body=None,
                status_code=502,
                latency_ms=latency_ms,
                outcome="upstream_error",
                error_message=str(exc),
            ),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers={"x-gateway-cache": "miss"} if decision.cacheable else None,
        background=BackgroundTask(
            record_trace,
            request_body=body,
            response_body=payload,
            status_code=status_code,
            latency_ms=latency_ms,
            outcome="ok" if status_code < 400 else "upstream_error",
            error_message=None,
            cache_decision=decision,  # miss + ok response → cold-path store
            coalesced=is_follower,  # follower → cost 0; leader carries spend
        ),
    )


# ---------------------------------------------------------------------------
# Streaming (Phase 2)
# ---------------------------------------------------------------------------

async def _streamed(body: dict):
    started = time.perf_counter()
    result = StreamResult()
    upstream_gen = forward_stream(body, result)

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
                request_body=body,
                result=result,
                latency_ms=latency_ms,
            ),
        )

    result.ttft_ms = int((time.perf_counter() - started) * 1000)
    status_code = result.status_code or 502

    async def relay() -> AsyncIterator[bytes]:
        if first_chunk is not None:
            yield first_chunk
        async for chunk in upstream_gen:
            yield chunk

    # Non-2xx: upstream rejected the request before streaming (auth, quota,
    # bad model...). forward_stream yielded its error body as plain bytes —
    # relay it under the true status, as JSON, not as an event stream.
    media_type = "text/event-stream" if status_code < 400 else "application/json"

    async def trace_when_drained() -> None:
        # BackgroundTask args bind at construction time, but total latency
        # only exists once the stream has drained — so it's computed HERE,
        # when Starlette runs the task after the response completes.
        await record_stream_trace(
            request_body=body,
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
