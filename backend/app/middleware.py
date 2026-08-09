"""Request-tracing ASGI middleware.

Reads ``X-Request-ID`` and ``X-Test-Name`` from incoming request headers,
sets the corresponding ContextVars for the duration of the request, and
echoes both values back in the response headers.

Why raw ASGI instead of ``BaseHTTPMiddleware``
----------------------------------------------
``BaseHTTPMiddleware`` (and the ``@app.middleware("http")`` decorator that
wraps it) buffers the entire response body before forwarding it to the
client.  This gateway forwards SSE streams via ``StreamingResponse`` — a
buffered wrapper would break streaming entirely.  A raw ASGI middleware
intercepts only the ``http.response.start`` message (headers) and lets the
body chunks pass through untouched, so streaming behaviour is preserved.

ContextVar lifecycle
--------------------
``set_request_context`` is called before ``await self.app(scope, receive,
send_with_headers)`` and ``reset_request_context`` runs in the ``finally``
block, which executes only after the wrapped app (including its
BackgroundTasks) has fully completed.  Starlette runs BackgroundTasks
synchronously inside the app's ``__call__`` (they are ``await``-ed inline,
not spawned as separate asyncio Tasks), so the ContextVars are still live
when ``record_trace`` executes — no explicit value-capture in the route
handlers is required.
"""
from __future__ import annotations

import uuid

from .request_context import reset_request_context, set_request_context

_HEADER_REQUEST_ID = b"x-request-id"
_HEADER_TEST_NAME = b"x-test-name"


class RequestTracingMiddleware:
    """Raw ASGI middleware that stamps every request with tracing context."""

    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            # Pass WebSocket / lifespan scopes through untouched.
            await self.app(scope, receive, send)
            return

        # ------------------------------------------------------------------
        # Extract / generate tracing identifiers from request headers.
        # Headers arrive as a list of (name_bytes, value_bytes) pairs.
        # ------------------------------------------------------------------
        raw_headers: dict[bytes, bytes] = {
            k.lower(): v for k, v in scope.get("headers", [])
        }

        raw_id = raw_headers.get(_HEADER_REQUEST_ID, b"").decode("latin-1").strip()
        request_id = raw_id if raw_id else uuid.uuid4().hex[:8]

        raw_name = raw_headers.get(_HEADER_TEST_NAME, b"").decode("latin-1").strip()
        test_name = raw_name if raw_name else "default"

        # ------------------------------------------------------------------
        # Set ContextVars so every logger and record_trace() sees them.
        # ------------------------------------------------------------------
        id_token, name_token = set_request_context(
            request_id=request_id, test_name=test_name
        )

        # ------------------------------------------------------------------
        # Wrap the ASGI ``send`` callable to inject response headers on the
        # ``http.response.start`` message only.  Body chunks are forwarded
        # verbatim — this is what keeps SSE streaming intact.
        # ------------------------------------------------------------------
        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers: list[tuple[bytes, bytes]] = list(
                    message.get("headers", [])
                )
                headers.append((_HEADER_REQUEST_ID, request_id.encode()))
                headers.append((_HEADER_TEST_NAME, test_name.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            reset_request_context(id_token, name_token)
