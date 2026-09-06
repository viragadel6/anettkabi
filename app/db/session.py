"""Async engine and session management with statement timeouts."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

__all__ = ["create_engine", "create_session_factory", "session_scope", "set_statement_timeout"]

_engines: dict[str, AsyncEngine] = {}
_session_factories: dict[str, async_sessionmaker[AsyncSession]] = {}


def create_engine(database_url: str) -> AsyncEngine:
    """Create (or return cached) async engine for a database URL.

    Parameters:
        database_url: SQLAlchemy asyncpg URL.

    Returns:
        A configured AsyncEngine.
    """
    if database_url in _engines:
        return _engines[database_url]
    settings = get_settings()
    engine = create_async_engine(
        database_url,
        pool_size=settings.database.db_pool_size,
        max_overflow=settings.database.db_max_overflow,
        pool_pre_ping=True,
        echo=False,
        connect_args={"timeout": 15, "server_settings": {"application_name": "vsfx-service"}},
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_timeout(dbapi_connection: Any, _record: Any) -> None:
        timeout_ms = settings.database.db_statement_timeout_ms
        with dbapi_connection.cursor() as cursor:
            cursor.execute(f"SET statement_timeout = {int(timeout_ms)}")
        dbapi_connection.commit()

    _engines[database_url] = engine
    return engine


def create_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create (or return cached) session factory bound to an engine.

    Parameters:
        database_url: SQLAlchemy asyncpg URL.

    Returns:
        An async_sessionmaker producing AsyncSessions.
    """
    if database_url not in _session_factories:
        _session_factories[database_url] = async_sessionmaker(
            bind=create_engine(database_url),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factories[database_url]


@contextlib.asynccontextmanager
async def session_scope(database_url: str | None = None) -> AsyncIterator[AsyncSession]:
    """Provide a transactional session scope with commit/rollback.

    Parameters:
        database_url: Optional explicit URL; defaults to configured.

    Yields:
        An AsyncSession that commits on success and rolls back on error.

    Raises:
        ServiceError: storage_failed is NOT raised here; DB errors propagate
            to the caller for taxonomy mapping upstream.
    """
    url = database_url or get_settings().database.database_url
    factory = create_session_factory(url)
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def set_statement_timeout(session: AsyncSession, timeout_ms: int) -> None:
    """Override the statement timeout for one session.

    Parameters:
        session: Target session.
        timeout_ms: New timeout in milliseconds.
    """
    await session.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
