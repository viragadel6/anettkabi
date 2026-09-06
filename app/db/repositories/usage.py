"""Usage repository: billed-seconds recording and monthly aggregation."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UsageEvent

__all__ = ["UsageRepository"]


class UsageRepository:
    """Billed usage queries for quota enforcement and reporting."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session.

        Parameters:
            session: Active AsyncSession.
        """
        self._session = session

    async def record(
        self,
        *,
        api_key_id: uuid.UUID,
        prediction_id: str | None,
        billed_seconds: float,
        gpu_ms: int,
    ) -> UsageEvent:
        """Insert a usage event.

        Parameters:
            api_key_id: Caller key.
            prediction_id: Related prediction or None.
            billed_seconds: Seconds billed for this event.
            gpu_ms: GPU time in milliseconds.

        Returns:
            The persisted UsageEvent row.
        """
        row = UsageEvent(
            api_key_id=api_key_id,
            prediction_id=prediction_id,
            billed_seconds=billed_seconds,
            gpu_ms=gpu_ms,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def month_seconds(self, api_key_id: uuid.UUID, at: dt.datetime | None = None) -> float:
        """Sum billed seconds for the calendar month containing `at`.

        Parameters:
            api_key_id: Caller key.
            at: Reference timestamp (default now, UTC).

        Returns:
            Total billed seconds this month.
        """
        reference = at or dt.datetime.now(tz=dt.UTC)
        month_start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month: dt.datetime
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year + 1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month + 1)
        result = await self._session.execute(
            select(func.coalesce(func.sum(UsageEvent.billed_seconds), 0.0)).where(
                UsageEvent.api_key_id == api_key_id,
                UsageEvent.created_at >= month_start,
                UsageEvent.created_at < next_month,
            )
        )
        return float(result.scalar_one())

    async def month_gpu_ms(self, api_key_id: uuid.UUID, at: dt.datetime | None = None) -> int:
        """Sum GPU milliseconds for the calendar month containing `at`.

        Parameters:
            api_key_id: Caller key.
            at: Reference timestamp (default now, UTC).

        Returns:
            Total GPU milliseconds this month.
        """
        reference = at or dt.datetime.now(tz=dt.UTC)
        month_start = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month: dt.datetime
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year + 1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month + 1)
        result = await self._session.execute(
            select(func.coalesce(func.sum(UsageEvent.gpu_ms), 0)).where(
                UsageEvent.api_key_id == api_key_id,
                UsageEvent.created_at >= month_start,
                UsageEvent.created_at < next_month,
            )
        )
        return int(result.scalar_one())
