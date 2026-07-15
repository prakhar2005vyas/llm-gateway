"""Config is env-driven and URLs are composed correctly."""
from __future__ import annotations

from app.config import Settings, get_settings


def test_chat_completions_url_strips_trailing_slash():
    s = Settings(upstream_base_url="https://api.openai.com/v1/")
    assert s.chat_completions_url == "https://api.openai.com/v1/chat/completions"


def test_defaults_are_sane():
    s = Settings()
    assert s.upstream_max_retries >= 0
    assert s.upstream_timeout_seconds > 0
    assert s.gateway_api_key == ""  # pass-through by default


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("UPSTREAM_MAX_RETRIES", "5")
    monkeypatch.setenv("UPSTREAM_BASE_URL", "https://fireworks.test/v1")
    get_settings.cache_clear()
    s = get_settings()
    assert s.upstream_max_retries == 5
    assert s.chat_completions_url == "https://fireworks.test/v1/chat/completions"
