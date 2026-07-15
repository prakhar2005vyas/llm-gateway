"""Cold-path trace recording — runs AFTER the client connection is released.

This is the receiving end of the hot/cold seam (SPEC.md lifecycle step 9).
The route attaches `record_trace` as a starlette BackgroundTask on the
response; Starlette invokes it once the response body has been sent, so the
client never waits on Postgres.

Failure contract (non-negotiable):
* Nothing raised here may ever reach the client — the response is already
  gone. Every exception is caught, logged, and swallowed AT THIS BOUNDARY
  (this is the one place where catching everything is correct, and it is
  not silent: the error is logged loudly with a clear degraded-state marker).
* Logs never include request/response bodies — they can carry PII. We log
  exception class + message + trace id only.
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import select

from .costs import cost_for_model
from .db import session
from .models import ModelPrice, Trace

logger = logging.getLogger(__name__)


def _extract_usage(response_body: dict | None) -> tuple[int | None, int | None, int | None]:
    """Pull (prompt, completion, total) token counts from an OpenAI-shaped
    response. Missing/malformed usage → Nones (cost becomes honest NULL)."""
    if not isinstance(response_body, dict):
        return None, None, None
    usage = response_body.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    def _int_or_none(v: object) -> int | None:
        return v if isinstance(v, int) and not isinstance(v, bool) else None

    return (
        _int_or_none(usage.get("prompt_tokens")),
        _int_or_none(usage.get("completion_tokens")),
        _int_or_none(usage.get("total_tokens")),
    )


async def record_trace(
    *,
    request_body: dict | None,
    response_body: dict | None,
    status_code: int | None,
    latency_ms: int | None,
    outcome: str,
    error_message: str | None = None,
) -> None:
    """Compute cost and persist one Trace row. Never raises."""
    trace_id = uuid.uuid4()
    try:
        # The client asked for this model id, and the price table is keyed by
        # what clients ask for ("gpt-4o-mini"), not the resolved snapshot id
        # the provider reports back ("gpt-4o-mini-2024-07-18"). Fall back to
        # the response's id only when the request had none.
        model_id: str | None = None
        if isinstance(request_body, dict):
            model_id = request_body.get("model") or None
        if model_id is None and isinstance(response_body, dict):
            model_id = response_body.get("model") or None

        prompt_tokens, completion_tokens, total_tokens = _extract_usage(response_body)

        async with session() as s:
            price_row = None
            if model_id is not None:
                price_row = await s.get(ModelPrice, model_id)
            prices = {price_row.model_id: price_row} if price_row else {}

            cost = cost_for_model(model_id, prompt_tokens, completion_tokens, prices)

            s.add(
                Trace(
                    id=trace_id,
                    request_body=request_body,
                    response_body=response_body,
                    model_id=model_id,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost,
                    outcome=outcome,
                    error_message=error_message,
                )
            )
        logger.debug("trace %s recorded (model=%s outcome=%s)", trace_id, model_id, outcome)
    except Exception as exc:  # noqa: BLE001 — deliberate: see module docstring
        # DEGRADED STATE: the request succeeded but its trace is lost.
        # Class+message only — bodies may contain PII, they stay out of logs.
        logger.error(
            "TRACE WRITE FAILED (trace %s dropped, client unaffected): %s: %s",
            trace_id,
            type(exc).__name__,
            exc,
        )
