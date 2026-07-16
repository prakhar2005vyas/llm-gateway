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
