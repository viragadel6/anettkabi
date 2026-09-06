"""Asset repository: registration, lookup by key, retention sweeps."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Asset, AssetKind

__all__ = ["AssetRepository"]


class AssetRepository:
    """CRUD helpers for assets rows."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session.

        Parameters:
            session: Active AsyncSession.
        """
        self._session = session

    async def register(
        self,
        *,
        kind: AssetKind,
        storage_key: str,
        bucket: str,
        content_type: str,
        size_bytes: int,
        sha256: str,
        duration_s: float | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        has_audio: bool | None = None,
        video_codec: str | None = None,
        audio_codec: str | None = None,
        expires_at: dt.datetime | None = None,
    ) -> Asset:
        """Insert or refresh an asset row keyed by storage_key.

        Parameters:
            kind: Asset kind.
            storage_key: Object key.
            bucket: Bucket name.
            content_type: MIME type.
            size_bytes: Object size.
            sha256: Content digest.
            duration_s, width, height, fps, has_audio, video_codec, audio_codec:
                Optional probe metadata.
            expires_at: Retention deadline.

        Returns:
            The persisted Asset row.
        """
        existing = await self.get_by_key(storage_key)
        if existing is not None:
            return existing
        row = Asset(
            kind=kind,
            storage_key=storage_key,
            bucket=bucket,
            content_type=content_type,
            size_bytes=size_bytes,
            sha256=sha256,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
            has_audio=has_audio,
            video_codec=video_codec,
            audio_codec=audio_codec,
            expires_at=expires_at,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, asset_id: uuid.UUID) -> Asset | None:
        """Fetch an asset by id.

        Parameters:
            asset_id: Asset UUID.

        Returns:
            The Asset or None.
        """
        return await self._session.get(Asset, asset_id)

    async def get_by_key(self, storage_key: str) -> Asset | None:
        """Fetch an asset by its object key.

        Parameters:
            storage_key: Object key.

        Returns:
            The Asset or None.
        """
        result = await self._session.execute(select(Asset).where(Asset.storage_key == storage_key))
        return result.scalar_one_or_none()

    async def mark_expired_deleted(self, asset_ids: list[uuid.UUID]) -> int:
        """Mark swept assets as deleted in the database.

        Parameters:
            asset_ids: Swept asset UUIDs.

        Returns:
            Number of rows updated.
        """
        if not asset_ids:
            return 0
        result = await self._session.execute(
            update(Asset)
            .where(Asset.id.in_(asset_ids))
            .values(deleted_at=func.now(), expires_at=None)
        )
        return int(result.rowcount or 0)

    async def expired_before(self, when: dt.datetime, limit: int = 200) -> list[Asset]:
        """List output assets whose retention deadline has passed.

        Parameters:
            when: Reference timestamp (now).
            limit: Maximum rows.

        Returns:
            List of expired Asset rows not yet swept.
        """
        result = await self._session.execute(
            select(Asset)
            .where(
                Asset.expires_at.is_not(None),
                Asset.expires_at < when,
                Asset.deleted_at.is_(None),
                Asset.kind == AssetKind.OUTPUT_VIDEO,
            )
            .limit(limit)
        )
        return list(result.scalars().all())
