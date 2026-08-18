"""LLM Gateway — FastAPI application entrypoint.

Phase 1: transparent OpenAI-compatible proxy + the hot/cold seam (traces
written via BackgroundTask after the response is flushed) + `/health`.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import get_settings
from .db import dispose_db, init_db
from .embeddings import warmup
from .middleware import RequestTracingMiddleware
from .request_context import RequestContextFilter
from .routes import chat, internal
from .seed import seed_model_prices
from .upstream import Provider, _get_routes, providers
import httpx
import asyncio

settings = get_settings()

_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[Test: %(test_name)s] [Req: %(request_id)s] %(message)s"
)

# Python's logging propagation model: a filter added to a Logger is only
# consulted for records *originated* on that logger — propagated records from
# child loggers (app.cache, app.embeddings, …) skip the parent logger's filter
# list and go straight to the parent's *handlers*.  The filter must therefore
# live on each Handler, not on the Logger itself.
#
# Execution order: uvicorn installs its own StreamHandler on the root logger
# BEFORE it imports the ASGI app (main.py), so basicConfig() would be a no-op
# here.  We instead update every existing handler in-place and fall back to
# basicConfig only when no handlers are present yet (unit-test bare-Python
# import, or first-time process start before any logging has been configured).
_root_logger = logging.getLogger()
_root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
if not _root_logger.handlers:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=_LOG_FORMAT,
    )
_context_filter = RequestContextFilter()
_formatter = logging.Formatter(_LOG_FORMAT)
for _handler in _root_logger.handlers:
    _handler.setFormatter(_formatter)
    _handler.addFilter(_context_filter)  # filter on the handler, not the logger

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure schema on boot. If the DB is down, the gateway still starts and
    # still proxies — every trace write will log its own loud failure until
    # the DB returns. Observability degrades; the product does not go down
    # with it (same principle as the Phase 7 frozen-Postgres chaos test).
    try:
        await init_db()
        await seed_model_prices()
    except Exception as exc:  # noqa: BLE001 — startup must not crash on DB-down
        logger.error(
            "DATABASE UNAVAILABLE AT STARTUP — proxying continues, traces will "
            "be dropped until it recovers: %s: %s",
            type(exc).__name__,
            exc,
        )
    try:
        await warmup()
    except Exception as exc:  # noqa: BLE001 — a missing model degrades to lazy load
        logger.error(
            "EMBEDDING MODEL WARMUP FAILED — first cacheable request will pay "
            "the load cost: %s: %s",
            type(exc).__name__,
            exc,
        )
    yield
    await dispose_db()


app = FastAPI(title="LLM Gateway", version="0.2.0", lifespan=lifespan)
# Raw ASGI middleware: stamps every request with X-Request-ID / X-Test-Name
# context vars and echoes both back in response headers.  Must be added after
# the app is constructed so it wraps the complete router/middleware stack.
app.add_middleware(RequestTracingMiddleware)
app.include_router(chat.router)
# Phase 7 UI: read-only trace browser API, deliberately outside /v1.
app.include_router(internal.router, prefix="/internal", tags=["Internal"])


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    # Phase 7: Prometheus scrape target. A plain route (not an ASGI mount) so
    # the bare path serves directly — no 307 to /metrics/. No auth, no rate
    # limit, no proxy lifecycle; it renders process-local counters only.
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> dict:
    all_providers = providers()
    routes = _get_routes()
    for route in routes.values():
        all_providers.append(
            Provider(
                name=route.get("provider_label", "routed"),
                base_url=route["base_url"],
                api_key=route.get("api_key", ""),
            )
        )
    if settings.failover_base_url:
        all_providers.append(
            Provider(
                name="failover",
                base_url=settings.failover_base_url,
                api_key=settings.failover_api_key,
            )
        )
    
    async def check_provider(provider: Provider) -> tuple[str, str]:
        models_url = f"{provider.base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(models_url, headers=provider.headers)
                return provider.name, "ok" if resp.status_code == 200 else f"error {resp.status_code}"
        except Exception as exc:
            return provider.name, f"error {type(exc).__name__}"
    
    results = await asyncio.gather(*(check_provider(p) for p in all_providers))
    
    status_dict = {}
    for name, stat in results:
        status_dict[name] = stat
    
    return {"status": "ok", "version": app.version, "providers": status_dict}