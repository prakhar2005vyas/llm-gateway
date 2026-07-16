"""Sliding-window limiter unit tests."""
from __future__ import annotations

import pytest

from app.config import get_settings
from app.ratelimit import SlidingWindowLimiter, client_key


@pytest.fixture
def small_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_admits_up_to_limit_then_rejects(small_limit):
    lim = SlidingWindowLimiter()
    assert all(lim.check("k").allowed for _ in range(3))
    d = lim.check("k")
    assert not d.allowed
    assert d.retry_after_seconds >= 1  # a real, usable Retry-After


def test_keys_are_isolated(small_limit):
    lim = SlidingWindowLimiter()
    for _ in range(3):
        assert lim.check("alice").allowed
    assert not lim.check("alice").allowed
    # Alice being over-limit costs Bob nothing.
    assert lim.check("bob").allowed


def test_window_slides_capacity_back(small_limit, monkeypatch):
    lim = SlidingWindowLimiter()
    fake_now = [1000.0]
    monkeypatch.setattr("app.ratelimit.time.monotonic", lambda: fake_now[0])

    for _ in range(3):
        assert lim.check("k").allowed
    assert not lim.check("k").allowed

    fake_now[0] += 61  # the whole window slides past
    assert lim.check("k").allowed  # capacity restored, no manual reset


def test_disabled_admits_everything(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    get_settings.cache_clear()
    lim = SlidingWindowLimiter()
    assert all(lim.check("k").allowed for _ in range(50))
    get_settings.cache_clear()


def test_client_key_extraction():
    assert client_key("Bearer sk-abc123") == "sk-abc123"
    assert client_key("bearer sk-abc123") == "sk-abc123"  # case-insensitive scheme
    assert client_key(None) == "anonymous"
    assert client_key("") == "anonymous"
    assert client_key("Basic dXNlcg==") == "anonymous"  # non-bearer → shared bucket


# ---------------------------------------------------------------------------
# B-1 audit fix: the key map must not grow forever
# ---------------------------------------------------------------------------

def test_key_pruned_when_deque_empties_on_check(small_limit, monkeypatch):
    lim = SlidingWindowLimiter()
    fake_now = [1000.0]
    monkeypatch.setattr("app.ratelimit.time.monotonic", lambda: fake_now[0])

    lim.check("alice")
    assert lim.tracked_keys == 1
    fake_now[0] += 61  # alice's only hit expires
    lim.check("bob")  # unrelated check triggers nothing global yet...
    lim.check("alice")  # ...but alice's own check evicts, re-adds her hit
    assert lim.tracked_keys == 2  # alice + bob, no ghosts


def test_abandoned_keys_reclaimed_by_sweep(small_limit, monkeypatch):
    """THE B-1 scenario: attacker rotates tokens once each and vanishes.
    The sweep must reclaim every abandoned key."""
    import app.ratelimit as rl

    lim = SlidingWindowLimiter()
    fake_now = [1000.0]
    monkeypatch.setattr("app.ratelimit.time.monotonic", lambda: fake_now[0])
    monkeypatch.setattr(rl, "_SWEEP_EVERY", 10)  # sweep often for the test

    for i in range(500):
        lim.check(f"rotating-token-{i}")
    assert lim.tracked_keys == 500  # all live inside the window — correct

    fake_now[0] += 61  # the whole window slides past; all 500 are garbage
    for _ in range(10):  # enough checks to cross the sweep threshold
        lim.check("survivor")
    assert lim.tracked_keys == 1, (
        f"{lim.tracked_keys} keys still tracked — abandoned keys leak"
    )


def test_sweep_pressure_triggers_immediately(small_limit, monkeypatch):
    import app.ratelimit as rl

    lim = SlidingWindowLimiter()
    fake_now = [1000.0]
    monkeypatch.setattr("app.ratelimit.time.monotonic", lambda: fake_now[0])
    monkeypatch.setattr(rl, "_SWEEP_PRESSURE", 50)

    for i in range(60):
        lim.check(f"k{i}")
    fake_now[0] += 61
    lim.check("fresh")  # map > pressure threshold → sweep fires NOW
    assert lim.tracked_keys == 1


# ---------------------------------------------------------------------------
# A-2 audit fix: no token bytes in logs
# ---------------------------------------------------------------------------

def test_over_limit_log_carries_hash_never_token_bytes(small_limit, caplog):
    import logging

    lim = SlidingWindowLimiter()
    secret = "sk-supersecrettoken12345"
    for _ in range(3):
        lim.check(secret)
    with caplog.at_level(logging.WARNING, logger="app.ratelimit"):
        assert not lim.check(secret).allowed

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "sha256:" in joined  # correlation id present...
    # ...but no substring of the token itself (check prefixes of every length)
    for n in range(4, len(secret) + 1):
        assert secret[:n] not in joined
