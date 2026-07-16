"""Shared fixtures. Guarantees tests never hit a real provider."""
from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch):
    """Isolate settings between tests and pin a fake upstream by default.

    The cache is cleared before and after each test so per-test env overrides
    take effect and never leak.
    """
    monkeypatch.setenv("UPSTREAM_BASE_URL", "https://upstream.test/v1")
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-upstream-key")
    monkeypatch.setenv("GATEWAY_API_KEY", "")
    # Hermetic DB default: no test may ever require (or accidentally reach)
    # a real Postgres. Tests that need rows override/reset via app.db helpers.
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    get_settings.cache_clear()
    # The rate limiter is a process-wide singleton; without a reset, requests
    # accumulate across tests inside one monotonic window and unrelated tests
    # start drawing 429s.
    from app.ratelimit import limiter

    limiter.reset_for_tests()
    yield
    get_settings.cache_clear()
