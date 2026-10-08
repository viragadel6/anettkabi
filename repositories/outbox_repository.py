from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models import OutboxEvent

__all__ = ["OutboxRepository"]


class OutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def fetch_unpublished_events(self, batch_size: int) -> list[OutboxEvent]:
        result = await self._session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.published.is_(False))
            .order_by(OutboxEvent.id.asc())
            .limit(batch_size)
        )
        return list(result.scalars().all())

    async def mark_published(self, event_ids: list[int]) -> int:
        if not event_ids:
            return 0
        statement = (
            update(OutboxEvent)
            .where(OutboxEvent.id.in_(event_ids))
            .values(published=True)
        )
        result = await self._session.execute(statement)
        return int(result.rowcount)

    async def fetch_events_for_aggregate(
        self, aggregate_id: uuid.UUID, limit: int = 1000
    ) -> list[OutboxEvent]:
        result = await self._session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.aggregate_id == aggregate_id)
            .order_by(OutboxEvent.id.asc())
            .limit(limit)
        )
        return list(result.scalars().all())
