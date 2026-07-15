"""Upstream forwarding — the actual proxy hop.

Phase 0 scope: a single OpenAI-compatible provider, non-streaming. The request
body is forwarded verbatim and the response is returned unchanged (transparent
proxy). Later phases layer caching/coalescing/failover on top of this seam.

Reliability (per CLAUDE.md): every external call gets a timeout and a bounded
number of retries with exponential backoff. Retries cover transient network
faults and upstream 5xx. On repeated failure we raise `UpstreamError` so the
route can return a clean 502 rather than crashing the process.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from .config import get_settings

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
