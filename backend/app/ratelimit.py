"""Per-key rate limiting — in-memory sliding window (Phase 4, v1).

Sliding window over request timestamps (a deque per key): accurate at window
boundaries, unlike fixed-window counters that admit 2× bursts straddling the
reset instant. Amortized O(1) per check (each timestamp is evicted at most
once), no locks — the check contains no await, so on a single event loop it
is atomic by construction (same argument as the coalescer's CAS).

Key = the client's bearer token (whatever Authorization they sent; the SDK
always sends one). No token → one shared "anonymous" bucket.

Single-process by design: each replica enforces its own budget. Cross-replica
counters need Redis — excluded scope, documented in SCALING.md.

Memory bound (B-1 audit fix): two mechanisms, because per-check eviction
alone cannot reclaim ABANDONED keys (a key that stops sending traffic is
never checked again, so its deque would linger forever):
* each check evicts that key's expired timestamps and pops the key if its
  deque empties;
* a lazy global sweep (every _SWEEP_EVERY checks, or whenever the key map
  exceeds _SWEEP_PRESSURE) walks all keys, evicts expired timestamps, and
  pops empty keys. Amortized cost stays O(1) per request; resident state is
  bounded by keys active within the current window × limit timestamps.

Log hygiene (A-2 audit fix): raw tokens are secrets — log lines carry a
sha256 hash prefix for correlation, never any substring of the token itself.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

from .config import get_settings

logger = logging.getLogger(__name__)

_SWEEP_EVERY = 1024      # checks between lazy global sweeps
_SWEEP_PRESSURE = 4096   # key-map size that forces an immediate sweep


def _key_id(key: str) -> str:
    """Loggable correlation id for a key: hash prefix, never token bytes."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after_seconds: int = 0  # ceil'd, ready for the Retry-After header


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._checks_since_sweep = 0

    def check(self, key: str) -> RateDecision:
        """Admit or reject one request for `key`. Never raises."""
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return RateDecision(allowed=True)

        limit = settings.rate_limit_requests
        window = settings.rate_limit_window_seconds
        now = time.monotonic()

        self._checks_since_sweep += 1
        if (
            self._checks_since_sweep >= _SWEEP_EVERY
            or len(self._hits) > _SWEEP_PRESSURE
        ):
            self._sweep(now - window)

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
                "rate limit exceeded for key sha256:%s (%d req / %.0fs window)",
                _key_id(key), limit, window,
            )
            return RateDecision(allowed=False, retry_after_seconds=retry_after)

        dq.append(now)
        return RateDecision(allowed=True)

    def _sweep(self, cutoff: float) -> None:
        """Evict expired timestamps everywhere; pop keys left empty."""
        before = len(self._hits)
        for key in list(self._hits):
            dq = self._hits[key]
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if not dq:
                del self._hits[key]
        self._checks_since_sweep = 0
        if before != len(self._hits):
            logger.debug(
                "rate-limit sweep: %d -> %d tracked key(s)", before, len(self._hits)
            )

    @property
    def tracked_keys(self) -> int:
        """Current key-map size (tests/metrics)."""
        return len(self._hits)

    def reset_for_tests(self) -> None:
        self._hits.clear()
        self._checks_since_sweep = 0


limiter = SlidingWindowLimiter()


def client_key(authorization: str | None) -> str:
    """Rate-limit key from the Authorization header (bearer token or shared
    anonymous bucket). The raw token is the key — never logged (see A-2)."""
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    return "anonymous"
