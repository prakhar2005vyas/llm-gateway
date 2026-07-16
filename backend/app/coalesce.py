"""Exact-match request coalescing — single-flight upstream calls (Phase 4).

50 identical uncached requests arrive in the same instant: without this, the
gateway pays 50×. With it, the FIRST request (the leader) goes upstream; the
other 49 (followers) await the leader's future and share its result. SPEC
locked scope #3.

The atomic compare-and-set (terminology per CLAUDE.md — this is NOT "ACID",
there is no transaction): `dict.setdefault(key, new_future)` is one atomic
operation on the event loop. There is no `await` between "is someone already
in flight?" and "then I am the leader" — no interleaving is possible, so no
lock is needed. That single line is the whole concurrency argument.

Scope decisions (deliberate):
* EXACT match only — key is SHA-256 of (model + masked prompt). Semantic
  coalescing would compute cosine similarity against a live in-flight queue
  on the hot path: too slow, explicitly excluded by SPEC.
* In-memory, single-process. Cross-replica coalescing needs distributed
  locks with TTL/lease/holder-death handling — DOCUMENT-ONLY in SCALING.md.
* Deterministic requests only (explicit temperature <= COALESCE_MAX_TEMPERATURE).
  Collapsing N creative calls into one identical answer is a correctness bug
  (SPEC names this exact trap). Omitted temperature defaults to 1.0 upstream
  → not coalesceable.
* Non-streaming only in this phase: followers share one buffered result;
  fan-out of a single live SSE stream to N clients is a different machine.
* Masked prompt in the key — same PII discipline as the cache: raw prompt
  text never feeds derived artifacts, even in-memory keys (SPEC §6).

Error sharing: if the leader's upstream call fails, every follower gets the
SAME exception — they were all waiting on that one call, and each request's
own error path (clean 502, trace row) handles it from there. Result sharing:
followers receive the leader's payload BY REFERENCE; treat it as immutable
(the gateway only serializes it, never mutates).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .config import get_settings
from .pii import contains_pii, mask_prompt

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_coalesceable(body: dict) -> bool:
    """Deterministic, plain-text, non-streaming requests only."""
    settings = get_settings()
    if not settings.coalesce_enabled:
        return False
    if body.get("stream"):
        return False
    temperature = body.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        return False  # omitted temperature = 1.0 upstream = creative = bypass
    if temperature > settings.coalesce_max_temperature:
        return False
    if not isinstance(body.get("model"), str):
        return False
    serialized = _serialize(body)
    if serialized is None:
        return False
    # PII collision guard (Phase 5): masked keys erase the difference between
    # two users' distinct PII values — sharing one in-flight result across
    # them would leak user A's answer to user B. PII-bearing requests are
    # never coalesced.
    return not contains_pii(serialized)


def request_key(body: dict) -> str:
    """SHA-256 over (model, MASKED prompt). Callers guarantee is_coalesceable."""
    masked = mask_prompt(_serialize(body))  # type: ignore[arg-type]
    h = hashlib.sha256()
    h.update(body["model"].encode())
    h.update(b"\x00")
    h.update(masked.encode())
    return h.hexdigest()


def _serialize(body: dict) -> str | None:
    """Canonical text of the conversation (same shape the cache uses)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            return None
        lines.append(f"{m.get('role', 'user')}: {m['content']}")
    return "\n".join(lines)


class Coalescer:
    """The pending map. One instance per process (module singleton below)."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future] = {}

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    async def get_or_run(
        self, key: str, supplier: Callable[[], Awaitable[T]]
    ) -> tuple[T, bool]:
        """Single-flight execution of `supplier` per key.

        Returns (result, is_leader). Exactly one concurrent caller per key
        runs the supplier; the rest await its future. The pending entry is
        removed the moment the result is known — a LATER request with the
        same key starts a fresh flight (coalescing is for concurrent
        duplicates; serving finished results to later requests is the
        semantic cache's job, with its own staleness rules).
        """
        loop = asyncio.get_running_loop()
        new_future: asyncio.Future = loop.create_future()
        existing = self._pending.setdefault(key, new_future)  # <- the CAS

        if existing is not new_future:
            # Follower: someone else is already in flight for this key.
            logger.debug("coalesced follower waiting (key=%s…)", key[:12])
            return await existing, False

        # Leader: we own the flight.
        try:
            result = await supplier()
        except BaseException as exc:
            # Share the failure with every follower, then re-raise for our
            # own error path. Touching .exception() marks it retrieved so a
            # zero-follower flight can't emit "exception was never retrieved".
            if not new_future.cancelled():
                new_future.set_exception(exc)
                new_future.exception()
            raise
        else:
            if not new_future.cancelled():
                new_future.set_result(result)
            return result, True
        finally:
            # Remove while holding the result — after this line the key is
            # free and a new flight may start; existing followers already
            # hold a reference to OUR future, unaffected by the pop.
            self._pending.pop(key, None)


coalescer = Coalescer()
