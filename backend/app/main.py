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
from .routes import chat
from .seed import seed_model_prices

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
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
    yield
    await dispose_db()


app = FastAPI(title="LLM Gateway", version="0.2.0", lifespan=lifespan)
app.include_router(chat.router)


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    # Phase 7: Prometheus scrape target. A plain route (not an ASGI mount) so
    # the bare path serves directly — no 307 to /metrics/. No auth, no rate
    # limit, no proxy lifecycle; it renders process-local counters only.
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}
