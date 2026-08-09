"""Unit tests for request_context — ContextVars and RequestContextFilter.

Covers:
* ContextVar defaults outside any request context.
* set_request_context / reset_request_context lifecycle.
* RequestContextFilter populates LogRecord attributes.
* Isolation: one async task's context does not bleed into another.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from app.request_context import (
    RequestContextFilter,
    get_request_id,
    get_test_name,
    reset_request_context,
    set_request_context,
)


# ---------------------------------------------------------------------------
# ContextVar defaults
# ---------------------------------------------------------------------------

def test_default_request_id_outside_context():
    assert get_request_id() == "-"


def test_default_test_name_outside_context():
    assert get_test_name() == "default"


# ---------------------------------------------------------------------------
# set / reset lifecycle
# ---------------------------------------------------------------------------

def test_set_and_reset_request_context():
    tokens = set_request_context(request_id="abc12345", test_name="my-test")
    assert get_request_id() == "abc12345"
    assert get_test_name() == "my-test"
    reset_request_context(*tokens)
    # Defaults restored after reset.
    assert get_request_id() == "-"
    assert get_test_name() == "default"


def test_nested_context_restore():
    outer = set_request_context(request_id="outer-id", test_name="outer")
    inner = set_request_context(request_id="inner-id", test_name="inner")
    assert get_request_id() == "inner-id"
    reset_request_context(*inner)
    assert get_request_id() == "outer-id"
    reset_request_context(*outer)
    assert get_request_id() == "-"


# ---------------------------------------------------------------------------
# RequestContextFilter
# ---------------------------------------------------------------------------

def test_filter_adds_attributes_to_log_record():
    tokens = set_request_context(request_id="req-xyz", test_name="suite-A")
    try:
        filt = RequestContextFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        result = filt.filter(record)
        assert result is True  # filter never drops records
        assert record.request_id == "req-xyz"
        assert record.test_name == "suite-A"
    finally:
        reset_request_context(*tokens)


def test_filter_uses_defaults_when_no_context():
    filt = RequestContextFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="startup", args=(), exc_info=None,
    )
    filt.filter(record)
    assert record.request_id == "-"
    assert record.test_name == "default"


def test_filter_format_string_renders(caplog):
    """End-to-end: log line produced with the gateway's format includes tags."""
    tokens = set_request_context(request_id="f00dbeef", test_name="e2e-suite")
    try:
        with caplog.at_level(logging.DEBUG, logger="app"):
            # The filter must be on the logger caplog uses.
            filt = RequestContextFilter()
            caplog.handler.addFilter(filt)
            logger = logging.getLogger("app.test_ctx")
            logger.info("probe message")
        # At least one record must carry our context attributes.
        matching = [r for r in caplog.records if "probe message" in r.message]
        assert matching, "no matching log record found"
        assert matching[0].request_id == "f00dbeef"
        assert matching[0].test_name == "e2e-suite"
    finally:
        reset_request_context(*tokens)
        caplog.handler.removeFilter(filt)


# ---------------------------------------------------------------------------
# Async isolation — ContextVar must not bleed across tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_isolation_across_tasks():
    """Two concurrent async tasks see their own context values."""

    async def task_a():
        tokens = set_request_context(request_id="task-a", test_name="test-a")
        await asyncio.sleep(0)  # yield to let task_b run
        assert get_request_id() == "task-a"
        reset_request_context(*tokens)

    async def task_b():
        tokens = set_request_context(request_id="task-b", test_name="test-b")
        await asyncio.sleep(0)
        assert get_request_id() == "task-b"
        reset_request_context(*tokens)

    # asyncio.create_task copies the current context; mutations inside each
    # task are isolated from the other.
    await asyncio.gather(
        asyncio.create_task(task_a()),
        asyncio.create_task(task_b()),
    )
    # Neither task's context leaked into this coroutine.
    assert get_request_id() == "-"
