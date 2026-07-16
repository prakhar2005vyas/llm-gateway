"""Per-key rate limiting — in-memory sliding window (Phase 4, v1).

Sliding window over request timestamps (a deque per key): accurate at window
boundaries, unlike fixed-window counters that admit 2× bursts straddling the
reset instant. O(evicted) per check, no locks — the check contains no await,
so on a single event loop it is atomic by construction (same argument as the
coalescer's CAS).

Key = the client's bearer token (whatever Authorization they sent; the SDK
always sends one). No token → one shared "anonymous" bucket.

Single-process by design: each replica enforces its own budget. Cross-replica
counters need Redis — excluded scope, documented in SCALING.md.

Memory bound: one deque per active key, each capped at `limit` entries by the
algorithm itself; empty deques are pruned on check. Abusive keys cost
O(limit) floats each — bounded, no unbounded growth.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

from .config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after_seconds: int = 0  # ceil'd, ready for the Retry-After header


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> RateDecision:
        """Admit or reject one request for `key`. Never raises."""
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return RateDecision(allowed=True)

        limit = settings.rate_limit_requests
        window = settings.rate_limit_window_seconds
        now = time.monotonic()

        dq = self._hits.get(key)
        if dq is None:
            dq = deque()
            self._hits[key] = dq

        # Evict timestamps that have slid out of the window.
        cutoff = now - window
        while dq and dq[0] <= cutoff:
            dq.popleft()

        if len(dq) >= limit:
            # Oldest in-window hit determines when capacity next frees up.
            retry_after = max(1, math.ceil(dq[0] + window - now))
            logger.warning(
                "rate limit exceeded for key %s… (%d req / %.0fs window)",
                key[:8], limit, window,
            )
            return RateDecision(allowed=False, retry_after_seconds=retry_after)

        dq.append(now)
        return RateDecision(allowed=True)

    def reset_for_tests(self) -> None:
        self._hits.clear()


limiter = SlidingWindowLimiter()


def client_key(authorization: str | None) -> str:
    """Rate-limit key from the Authorization header (bearer token or shared
    anonymous bucket). The raw token is the key — never logged in full."""
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    return "anonymous"
