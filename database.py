from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import Settings, get_settings
from observability.tracing import database_transaction_span

__all__ = [
    "check_database_health",
    "dispose_db",
    "get_engine",
    "get_session_factory",
    "init_db",
    "session_scope",
]

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db(settings: Settings | None = None) -> AsyncEngine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine
    resolved = settings or get_settings()
    _engine = create_async_engine(
        resolved.database.DATABASE_URL,
        pool_size=resolved.database.POOL_SIZE,
        max_overflow=resolved.database.MAX_OVERFLOW,
        pool_timeout=resolved.database.POOL_TIMEOUT,
        pool_recycle=resolved.database.POOL_RECYCLE,
        pool_pre_ping=True,
        echo=False,
        connect_args={
            "timeout": 15,
            "command_timeout": 60,
            "server_settings": {"application_name": "task-engine"},
        },
    )
    _session_factory = async_sessionmaker[AsyncSession](
        bind=_engine,
        expire_on_commit=False,
        autoflush=False,
    )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        return init_db()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        init_db()
    assert _session_factory is not None
    return _session_factory


@contextlib.asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    operation: str = "transaction",
) -> AsyncIterator[AsyncSession]:
    factory = session_factory or get_session_factory()
    session = factory()
    with database_transaction_span(operation):
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_health(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> bool:
    factory = session_factory or get_session_factory()
    try:
        async with factory() as session:
            await session.execute(select(1))
        return True
    except Exception:
        return False


async def dispose_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def ensure_schema_ready() -> bool:
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name IN ('tasks', 'outbox_events')"
            )
        )
        return bool(result.scalar_one() == 2)
