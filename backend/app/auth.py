"""Gateway API-key auth (optional in Phase 0).

This is the seam for lifecycle step 1 in SPEC.md. When `GATEWAY_API_KEY` is
empty the gateway runs in transparent pass-through mode and this check is a
no-op — which is what lets the unmodified `openai` SDK talk to it in tests.
"""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_gateway_key(authorization: str | None = Header(default=None)) -> None:
    """Validate the client's bearer token against `GATEWAY_API_KEY`.

    Disabled (no-op) when `GATEWAY_API_KEY` is unset. Comparison is
    constant-time (hmac.compare_digest) — a plain `!=` short-circuits on the
    first differing byte and leaks prefix length as a timing side channel
    (A-2 audit fix).
    """
    expected = get_settings().gateway_api_key
    if not expected:
        return

    token = _extract_bearer(authorization) or ""
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing gateway API key",
        )


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()
