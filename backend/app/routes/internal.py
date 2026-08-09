"""Internal read API for the trace browser (Phase 7 UI).

Serves the React frontend, not gateway clients: a thin, read-only window
onto the traces table. Mounted under /internal (see main.py) so it can never
collide with the OpenAI-compatible surface under /v1.

Design notes:
* Same optional auth seam as the proxy (require_gateway_key): if a gateway
  key is configured, the internal API demands it too — the trace list holds
  masked-but-real conversation data.
* cost_usd is serialized as a STRING. It is stored as Numeric(12,8) exactly;
  pushing it through a float would reintroduce the rounding error the
  Decimal pipeline exists to avoid. The UI formats it, never does math on it.
* DB down: the gateway keeps proxying while observability degrades (the
  standing rule) — this endpoint returns a clean 503 rather than a stack
  trace, same honesty as the trace writer's loud shed.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import select

from ..auth import require_gateway_key
from ..db import session
from ..models import Trace

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_gateway_key)])

_MAX_LIMIT = 100


def _serialize(t: Trace) -> dict:
    return {
        "id": str(t.id),
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "model": t.model_id,
        "status_code": t.status_code,
        "latency_ms": t.latency_ms,
        "ttft_ms": t.ttft_ms,
        "prompt_tokens": t.prompt_tokens,
        "completion_tokens": t.completion_tokens,
        "total_tokens": t.total_tokens,
        # Exact Numeric(12,8) as a string; null = honest "cost unknown".
        "cost_usd": str(t.cost_usd) if t.cost_usd is not None else None,
        "cache_hit": t.cache_hit,
        "coalesced": t.coalesced,
        "outcome": t.outcome,
        "error_message": t.error_message,
        "test_name": t.test_name,
    }


@router.get("/traces")
async def list_traces(
    limit: int = Query(default=_MAX_LIMIT, ge=1, le=_MAX_LIMIT),
):
    """The `limit` most recent traces, newest first, as a JSON list."""
    try:
        async with session() as s:
            rows = (
                (
                    await s.execute(
                        select(Trace).order_by(Trace.created_at.desc()).limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [_serialize(t) for t in rows]
    except Exception as exc:  # noqa: BLE001 — read API must degrade, not crash
        logger.error(
            "internal trace query failed (DB down?): %s: %s",
            type(exc).__name__, exc,
        )
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "trace store unavailable",
                               "type": "gateway_degraded"}},
        )
