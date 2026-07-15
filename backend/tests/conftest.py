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
    yield
    get_settings.cache_clear()
