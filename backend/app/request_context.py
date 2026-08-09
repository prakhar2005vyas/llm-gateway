"""Per-request context propagation via ContextVars.

Two ContextVars carry the tracing identifiers through the async call stack:

* ``request_id``  — an 8-hex-char ID sourced from the ``X-Request-ID``
  request header, or generated fresh if the header is absent.
* ``test_name``   — a free-form label sourced from the ``X-Test-Name``
  header, used to associate log lines with the test that triggered them.

``RequestContextFilter`` reads these vars and injects them as attributes on
every ``logging.LogRecord`` so the format string in ``main.py`` can expose
them without touching any individual logger.  The filter must be installed on
each root-logger **Handler** (not on the Logger itself) — Python only runs a
logger's filter list for records *originated* on that logger; propagated
records from child loggers skip the parent logger's filters and go straight to
the parent's handlers.  Installing the filter on each handler ensures it runs
for every record regardless of where it originated.

Default values are chosen to produce readable log lines even when no request
context is active (startup, teardown, pure background work):
  [Test: default]  [Req: -]
"""
from __future__ import annotations

import logging
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# ContextVars — one per tracing dimension
# ---------------------------------------------------------------------------

#: Short unique ID for a single HTTP request.
_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

#: Human-readable label for the test (or caller) that sent this request.
_test_name_var: ContextVar[str] = ContextVar("test_name", default="default")


# ---------------------------------------------------------------------------
# Public accessors — used by the middleware and tracing cold path
# ---------------------------------------------------------------------------

def get_request_id() -> str:
    """Current request ID, or ``"-"`` outside a request context."""
    return _request_id_var.get()


def get_test_name() -> str:
    """Current test name, or ``"default"`` outside a request context."""
    return _test_name_var.get()


def set_request_context(*, request_id: str, test_name: str) -> tuple:
    """Set both context vars atomically.  Returns (id_token, name_token) for
    later ``reset()`` calls in the middleware's ``finally`` block."""
    id_token = _request_id_var.set(request_id)
    name_token = _test_name_var.set(test_name)
    return id_token, name_token


def reset_request_context(id_token, name_token) -> None:  # type: ignore[no-untyped-def]
    """Restore both vars to their previous state (called in middleware finally)."""
    _request_id_var.reset(id_token)
    _test_name_var.reset(name_token)


# ---------------------------------------------------------------------------
# Logging filter — injects context attributes on every LogRecord
# ---------------------------------------------------------------------------

class RequestContextFilter(logging.Filter):
    """Attach ``request_id`` and ``test_name`` to every LogRecord.

    Install once on the root logger::

        logging.getLogger().addFilter(RequestContextFilter())

    After that every logger in every module automatically emits the context
    tags without any per-logger configuration.  When no request context is
    active the ContextVar defaults apply (``"-"`` / ``"default"``), so the
    format string never raises ``KeyError``.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.request_id = _request_id_var.get()
        record.test_name = _test_name_var.get()
        return True  # never drop records — this filter only annotates
