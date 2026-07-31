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

import asyncio
import logging
import time
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

from . import cache, evals
from .config import get_settings
from .costs import cost_for_model
from .db import session
from .metrics import REQUEST_COUNT, REQUEST_LATENCY, normalize_model_name
from .models import ModelPrice, Trace


def _response_content(response_body: dict | None) -> str | None:
    """choices[0].message.content, defensively."""
    if not isinstance(response_body, dict):
        return None
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None

if TYPE_CHECKING:
    from .cache import CacheDecision
    from .upstream import StreamResult

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
    ttft_ms: int | None = None,
    cache_decision: "CacheDecision | None" = None,
    coalesced: bool = False,
    force_eval: bool = False,
) -> None:
    """Compute cost and persist one Trace row. Never raises.

    `cache_decision` carries the hot path's cache consultation:
    * hit  → trace records cache_hit=True with cost_usd = literal 0 (truly
      free — no upstream call happened; 0 here is honest, unlike unknowns),
      and the entry's hit stats are bumped.
    * miss → on a successful, storable response, the (already-masked,
      already-embedded) prompt and the response are inserted into the cache.
    """
    trace_id = uuid.uuid4()
    is_cache_hit = cache_decision is not None and cache_decision.hit is not None

    # The client asked for this model id, and the price table is keyed by
    # what clients ask for ("gpt-4o-mini"), not the resolved snapshot id
    # the provider reports back ("gpt-4o-mini-2024-07-18"). Fall back to
    # the response's id only when the request had none.
    model_id: str | None = None
    if isinstance(request_body, dict):
        model_id = request_body.get("model") or None
    if model_id is None and isinstance(response_body, dict):
        model_id = response_body.get("model") or None

    # Phase 7: Prometheus counters — process-local, so they update even when
    # the DB write below fails (metrics and traces degrade independently).
    # Own never-raise boundary, same contract as everything in this module.
    try:
        model_label = normalize_model_name(model_id)
        status_label = str(status_code) if status_code is not None else "none"
        REQUEST_COUNT.labels(
            model=model_label,
            status_code=status_label,
            cache_hit=str(is_cache_hit).lower(),
        ).inc()
        if latency_ms is not None:
            REQUEST_LATENCY.labels(
                model=model_label, status_code=status_label
            ).observe(latency_ms / 1000.0)
    except Exception as exc:  # noqa: BLE001 — metrics must never hurt tracing
        logger.error("metrics update failed (non-fatal): %s: %s",
                     type(exc).__name__, exc)

    try:
        prompt_tokens, completion_tokens, total_tokens = _extract_usage(response_body)

        async def _write() -> None:
            nonlocal prompt_tokens, completion_tokens, total_tokens
            async with session() as s:
                if is_cache_hit or coalesced:
                    # No upstream call happened FOR THIS REQUEST (cache hit,
                    # or coalesced follower sharing the leader's call): cost
                    # is a LITERAL zero (honest "free", distinct from NULL
                    # "unknown"), tokens stay NULL — the leader's trace
                    # carries the real spend exactly once.
                    cost = Decimal("0")
                    prompt_tokens = completion_tokens = total_tokens = None
                else:
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
                        cache_hit=is_cache_hit,
                        coalesced=coalesced,
                        outcome=outcome,
                        error_message=error_message,
                        ttft_ms=ttft_ms,
                    )
                )

        # Strict ceiling on the write (frozen-DB fix): this BackgroundTask
        # runs inside the ASGI cycle, so a write wedged on a frozen DB would
        # hold the keep-alive connection hostage — the NEXT request pipelined
        # on it would stall behind the freeze. At the timeout the write is
        # cancelled (session rolls back) and shed via the except-arm below.
        await asyncio.wait_for(
            _write(), timeout=get_settings().trace_write_timeout_seconds
        )
        logger.debug("trace %s recorded (model=%s outcome=%s)", trace_id, model_id, outcome)

        # Phase 6: sampled LLM-as-judge eval — LAST cold-path stage, strictly
        # after the trace commit (the FK needs the row; the client needed
        # nothing from us long ago). One choke point covers both shapes:
        # record_stream_trace flows through here too. Fire-and-forget with
        # shed-at-cap inside; never blocks, never raises.
        evals.maybe_enqueue_eval(
            trace_id=trace_id,
            request_body=request_body,
            response_content=_response_content(response_body),
            outcome=outcome,
            cache_hit=is_cache_hit,
            coalesced=coalesced,
            force=force_eval,
        )
    except Exception as exc:  # noqa: BLE001 — deliberate: see module docstring
        # DEGRADED STATE: the request succeeded but its trace is lost.
        # Class+message only — bodies may contain PII, they stay out of logs.
        logger.error(
            "TRACE WRITE FAILED (trace %s dropped, client unaffected): %s: %s",
            trace_id,
            type(exc).__name__,
            exc,
        )

    # Cache bookkeeping AFTER the trace attempt, isolated from it — each of
    # these owns its own never-raise boundary, plus the same write-timeout
    # ceiling as the trace itself (these too run inside the ASGI cycle and
    # would hold the connection hostage on a frozen DB).
    if cache_decision is None:
        return
    try:
        timeout = get_settings().trace_write_timeout_seconds
        if is_cache_hit:
            await asyncio.wait_for(
                cache.record_hit(cache_decision.hit.id), timeout=timeout
            )
        elif (
            outcome == "ok"
            and not coalesced  # the leader stores once; followers would only re-upsert
            and cache_decision.cacheable
            and cache_decision.masked_prompt is not None
            and cache_decision.embedding is not None
            and isinstance(request_body, dict)
            and cache.is_storable_response(response_body)
        ):
            await asyncio.wait_for(
                cache.store(
                    model_id=request_body["model"],
                    masked_prompt=cache_decision.masked_prompt,
                    embedding=cache_decision.embedding,
                    response_body=response_body,  # type: ignore[arg-type]
                ),
                timeout=timeout,
            )
    except Exception as exc:  # noqa: BLE001 — mostly TimeoutError: shed loudly
        logger.error(
            "CACHE BOOKKEEPING TIMED OUT/FAILED (skipped, client unaffected): %s: %s",
            type(exc).__name__, exc,
        )


async def record_stream_trace(
    *,
    request_body: dict | None,
    result: "StreamResult",
    latency_ms: int,
    allow_cache_store: bool = True,
) -> None:
    """Trace one streamed request from its populated StreamResult.

    Runs as a BackgroundTask — Starlette executes it only after the
    StreamingResponse has fully drained the generator, so `result` is
    complete by the time this reads it (no synchronization needed).

    The trace's response_body is SYNTHESIZED: streams have no single JSON
    response, so we store the reassembled content in OpenAI response shape,
    tagged "gateway.reassembled": true so nobody mistakes it for upstream's
    literal bytes. usage is included only if the provider sent it.
    """
    ok = result.error is None and (result.status_code or 0) < 400
    if result.error is not None:
        outcome = "inconclusive"  # stream started but ended abnormally
    elif ok:
        outcome = "ok"
    else:
        outcome = "upstream_error"

    response_body: dict = {
        "object": "chat.completion",
        "gateway.reassembled": True,
        "model": result.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": result.finish_reason,
            }
        ],
        "gateway.events_seen": result.events_seen,
    }
    if result.usage is not None:
        response_body["usage"] = result.usage

    await record_trace(
        request_body=request_body,
        response_body=response_body,
        status_code=result.status_code,
        latency_ms=latency_ms,
        outcome=outcome,
        error_message=result.error,
        ttft_ms=result.ttft_ms,
    )

    # Cache population from streams (Phase 3). Streams skip the hot-path
    # cache consult (a JSON hit can't answer an SSE request yet), so the
    # embedding is computed HERE, in the cold path — get_embedding runs on
    # the threadpool either way, and this task is already off the client's
    # critical path. The reassembled text can then serve future
    # NON-streaming duplicates of the same prompt.
    # allow_cache_store: the route passes False when the RAW request carried
    # PII — request_body here is the MASKED copy (looks PII-free), so the
    # is_cacheable_request guard alone can no longer see the hazard.
    if (
        outcome == "ok"
        and allow_cache_store
        and isinstance(request_body, dict)
        and cache.is_cacheable_request(request_body)
        and cache.is_storable_response(response_body)
    ):
        try:
            from .embeddings import get_embedding
            from .pii import mask_prompt

            masked = mask_prompt(cache.serialize_messages(request_body))  # type: ignore[arg-type]
            embedding = await get_embedding(masked)
            # The cached copy must parse as a completion in the openai SDK,
            # which requires id/created — the reassembled trace body has
            # neither (streams carry them per-chunk).
            cacheable_body = {
                "id": f"chatcmpl-gwcache-{uuid.uuid4().hex[:24]}",
                "created": int(time.time()),
                **response_body,
            }
            await cache.store(
                model_id=request_body["model"],
                masked_prompt=masked,
                embedding=embedding,
                response_body=cacheable_body,
            )
        except Exception as exc:  # noqa: BLE001 — cache must never break the task
            logger.error(
                "STREAM CACHE POPULATION FAILED (skipped, client unaffected): %s: %s",
                type(exc).__name__, exc,
            )
