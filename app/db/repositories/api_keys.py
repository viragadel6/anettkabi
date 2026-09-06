"""ApiKey repository: creation, prefix lookup, secret verification, disable."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ApiKey

__all__ = ["ApiKeyRepository"]


class ApiKeyRepository:
    """CRUD + verification helpers for api_keys rows."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session.

        Parameters:
            session: Active AsyncSession.
        """
        self._session = session

    async def create(
        self,
        *,
        name: str,
        key_prefix: str,
        key_hash: str,
        scopes: list[str],
        rate_limit_rpm: int,
        concurrency_limit: int,
        monthly_seconds_quota: float,
    ) -> ApiKey:
        """Insert a new API key row.

        Parameters:
            name: Human label.
            key_prefix: 8-char lookup prefix.
            key_hash: Argon2 hash of the secret part.
            scopes: Granted scopes.
            rate_limit_rpm: Per-key requests/minute.
            concurrency_limit: Per-key concurrent jobs.
            monthly_seconds_quota: Per-key monthly billed seconds.

        Returns:
            The persisted ApiKey.
        """
        row = ApiKey(
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            scopes=scopes,
            rate_limit_rpm=rate_limit_rpm,
            concurrency_limit=concurrency_limit,
            monthly_seconds_quota=monthly_seconds_quota,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_prefix(self, key_prefix: str) -> ApiKey | None:
        """Fetch an API key by its prefix.

        Parameters:
            key_prefix: The 8-char prefix.

        Returns:
            The ApiKey or None.
        """
        result = await self._session.execute(select(ApiKey).where(ApiKey.key_prefix == key_prefix))
        return result.scalar_one_or_none()

    async def get(self, api_key_id: uuid.UUID) -> ApiKey | None:
        """Fetch an API key by id.

        Parameters:
            api_key_id: Key UUID.

        Returns:
            The ApiKey or None.
        """
        return await self._session.get(ApiKey, api_key_id)

    async def touch_last_used(self, api_key_id: uuid.UUID, when: dt.datetime | None = None) -> None:
        """Update last_used_at without an explicit commit.

        Parameters:
            api_key_id: Key UUID.
            when: Timestamp; defaults to now.
        """
        await self._session.execute(
            update(ApiKey)
            .where(ApiKey.id == api_key_id)
            .values(last_used_at=when or dt.datetime.now(tz=dt.UTC))
        )

    async def disable(self, api_key_id: uuid.UUID) -> None:
        """Disable a key (soft) by stamping disabled_at.

        Parameters:
            api_key_id: Key UUID.
        """
        await self._session.execute(
            update(ApiKey).where(ApiKey.id == api_key_id).values(disabled_at=func.now())
        )

    async def list_all(self, limit: int = 100) -> list[ApiKey]:
        """List keys newest-first.

        Parameters:
            limit: Maximum rows.

        Returns:
            List of ApiKey rows.
        """
        result = await self._session.execute(
            select(ApiKey).order_by(ApiKey.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
