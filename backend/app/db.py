"""Async database engine, session factory, and transactional unit-of-work.

Design notes:

* One engine per process, created lazily and disposed on shutdown. The engine
  owns a **bounded** connection pool (pool_size + max_overflow, with a checkout
  timeout) — bounded on purpose: the cold path writes through this pool, and
  under DB slowness we want checkout attempts to fail fast so backpressure
  logic (Phase 1's BackgroundTask seam, later the bounded queue) can shed work
  instead of piling up waiters and OOMing.

* ACID applies here in its literal sense: `session()` yields an
  AsyncSession wrapped in a transaction that either commits atomically or
  rolls back on any exception. Postgres gives the I/C/D; the context manager
  guarantees the A is used correctly. (Terminology guard from CLAUDE.md: the
  Phase 4 coalescing lock is "atomic compare-and-set", NOT "ACID" — that term
  is reserved for real DB transactions like these.)

* `expire_on_commit=False`: the cold path logs a Trace and moves on; it never
  re-reads attributes after commit, so the extra refresh round-trip per commit
  would be pure overhead.

* Tests inject `sqlite+aiosqlite://` URLs. SQLite has no server-side pool to
  bound, so pool kwargs are applied only for non-SQLite URLs.

* Schema creation: `create_all()` on startup is fine for v1 (single service,
  additive schema). Alembic migrations become worth their weight when the
  schema starts changing under data; documented tradeoff, not an accident.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import get_settings
from .models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Process-wide engine, created lazily so config/tests can override first."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        # hide_parameters: on a failed statement, SQLAlchemy embeds the bound
        # parameters in the exception text — for a failed Trace INSERT that is
        # the full request/response body (PII) leaking into error logs. This
        # flag replaces them with "[parameters hidden]" engine-wide.
        kwargs: dict = {"echo": settings.db_echo, "hide_parameters": True}
        if not url.startswith("sqlite"):
            kwargs.update(
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_timeout=settings.db_pool_timeout_seconds,
                pool_recycle=settings.db_pool_recycle_seconds,
                # Validate connections on checkout — serverless PG (Neon)
                # kills idle conns; better one cheap ping than a mid-write
                # "connection closed" surprise.
                pool_pre_ping=True,
            )
        _engine = create_async_engine(url, **kwargs)
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info("database engine created (dialect=%s)", _engine.dialect.name)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    """Transactional unit-of-work: commit on success, rollback on any error.

    Usage:
        async with session() as s:
            s.add(trace)
        # committed here, or rolled back if the block raised
    """
    factory = get_session_factory()
    async with factory() as s:
        try:
            yield s
            await s.commit()
        except BaseException:
            await s.rollback()
            raise


async def init_db() -> None:
    """Create tables that don't exist yet. Idempotent; called on startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database schema ensured")


async def dispose_db() -> None:
    """Close the pool cleanly (shutdown hook). Safe to call when never opened."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database engine disposed")
