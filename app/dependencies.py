"""FastAPI dependency providers: sessions, storage, moderation, quotas."""

from __future__ import annotations

import threading
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import create_session_factory
from app.errors import ErrorCode, ServiceError
from app.services.moderation import ModerationService, build_moderation_service
from app.services.storage import StorageService, get_storage_service

__all__ = [
    "ApiKeyId",
    "ModerationDep",
    "SessionDep",
    "SettingsDep",
    "StorageDep",
    "require_scope",
]

_moderation_lock = threading.Lock()
_moderation_cache: dict[str, ModerationService] = {}


def get_settings_dep() -> Settings:
    """Return the process settings (dependency form).

    Returns:
        Cached Settings.
    """
    return get_settings()


async def get_session() -> AsyncIterator[AsyncSession]:
    """Provide a per-request session with commit/rollback semantics.

    Yields:
        An AsyncSession.

    Raises:
        ServiceError: storage-related DB errors propagate for mapping.
    """
    settings = get_settings()
    factory = create_session_factory(settings.database.database_url)
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_storage_dep() -> StorageService:
    """Return the shared storage service.

    Returns:
        StorageService singleton.
    """
    return get_storage_service()


def get_moderation_dep() -> ModerationService:
    """Return the shared moderation service.

    Returns:
        ModerationService singleton (built once per process).
    """
    if "moderation" not in _moderation_cache:
        with _moderation_lock:
            if "moderation" not in _moderation_cache:
                _moderation_cache["moderation"] = build_moderation_service()
    return _moderation_cache["moderation"]


def get_api_key_id(request: Request) -> uuid.UUID:
    """Read the authenticated API key id set by AuthMiddleware.

    Parameters:
        request: Current request.

    Returns:
        The caller's key UUID.

    Raises:
        ServiceError: unauthorized when auth did not run (dev mode disabled
            returns a nil-key marker instead when auth is off).
    """
    value = getattr(request.state, "api_key_id", None)
    if value is not None:
        return uuid.UUID(str(value))
    settings = get_settings()
    if not settings.auth.auth_required:
        return uuid.UUID(int=0)
    raise ServiceError(ErrorCode.UNAUTHORIZED, "authentication context missing")


def require_scope(scope: str):
    """Build a dependency enforcing a scope on the caller's key.

    Parameters:
        scope: Required scope string.

    Returns:
        A dependency callable raising forbidden_scope when absent.
    """

    def _checker(request: Request) -> uuid.UUID:
        """Verify the scope on the request's key.

        Parameters:
            request: Current request.

        Returns:
            The API key id.

        Raises:
            ServiceError: forbidden_scope when the key lacks the scope.
        """
        key_id = get_api_key_id(request)
        scopes = getattr(request.state, "api_key_scopes", [])
        if scopes and scope not in scopes and "admin" not in scopes:
            raise ServiceError(
                ErrorCode.FORBIDDEN_SCOPE,
                f"this endpoint requires the {scope!r} scope",
            )
        return key_id

    return _checker


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[StorageService, Depends(get_storage_dep)]
ModerationDep = Annotated[ModerationService, Depends(get_moderation_dep)]
ApiKeyId = Annotated[uuid.UUID, Depends(get_api_key_id)]
