"""OpenAI-compatible chat-completions endpoint.

Phase 1: transparent non-streaming pass-through + the hot/cold seam.
The hot path is: parse → forward → return. The Trace write (cost calc +
Postgres insert) rides a starlette BackgroundTask attached to the response,
which Starlette runs only after the response body has been flushed to the
client — the client never waits on the database (SPEC.md lifecycle step 8).

Streaming lands in Phase 2; until then `stream: true` gets a clean, explicit
error instead of a half-working response.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from ..auth import require_gateway_key
from ..tracing import record_trace
from ..upstream import UpstreamError, forward_chat_completion

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions", dependencies=[Depends(require_gateway_key)])
async def chat_completions(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "request body must be valid JSON", "invalid_request_error")

    if not isinstance(body, dict):
        return _error(400, "request body must be a JSON object", "invalid_request_error")

    if body.get("stream"):
        # Honest degraded state rather than pretending it works (CLAUDE.md).
        return _error(
            400,
            "streaming is not supported yet (arrives in Phase 2); "
            "retry with stream=false",
            "invalid_request_error",
        )

    started = time.perf_counter()
    try:
        status_code, payload = await forward_chat_completion(body)
    except UpstreamError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.error("forward failed: %s", exc)
        # A failed forward is still an observability event — trace it too.
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

    # HOT PATH ENDS HERE. The trace write (cost lookup + Postgres insert)
    # runs after this response is flushed; a DB failure can no longer touch
    # the client (record_trace never raises — see tracing.py).
    return JSONResponse(
        status_code=status_code,
        content=payload,
        background=BackgroundTask(
            record_trace,
            request_body=body,
            response_body=payload,
            status_code=status_code,
            latency_ms=latency_ms,
            outcome="ok" if status_code < 400 else "upstream_error",
            error_message=None,
        ),
    )


def _error(
    status_code: int,
    message: str,
    err_type: str,
    background: BackgroundTask | None = None,
) -> JSONResponse:
    """Return an OpenAI-shaped error object so SDK clients parse it cleanly."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type}},
        background=background,
    )
