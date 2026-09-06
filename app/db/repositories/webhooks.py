"""Webhook delivery repository: persistence of every attempt."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import WebhookDelivery

__all__ = ["WebhookRepository"]


class WebhookRepository:
    """Persistence for webhook delivery attempts."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session.

        Parameters:
            session: Active AsyncSession.
        """
        self._session = session

    async def record(
        self,
        *,
        prediction_id: str,
        url: str,
        attempt: int,
        signature: str,
        status_code: int | None = None,
        response_snippet: str = "",
        error: str = "",
        scheduled_at: dt.datetime | None = None,
        delivered_at: dt.datetime | None = None,
    ) -> WebhookDelivery:
        """Insert one delivery attempt row.

        Parameters:
            prediction_id: Related prediction.
            url: Target URL.
            attempt: 1-based attempt number.
            signature: HMAC signature header value sent.
            status_code: Response status when an HTTP exchange happened.
            response_snippet: First bytes of the response body.
            error: Error description on failure.
            scheduled_at: When the attempt was queued.
            delivered_at: When a response was received.

        Returns:
            The persisted WebhookDelivery row.
        """
        row = WebhookDelivery(
            prediction_id=prediction_id,
            url=url,
            attempt=attempt,
            status_code=status_code,
            response_snippet=response_snippet,
            signature=signature,
            error=error,
            scheduled_at=scheduled_at,
            delivered_at=delivered_at,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def for_prediction(self, prediction_id: str) -> list[WebhookDelivery]:
        """List all delivery attempts for a prediction.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            Attempt rows ordered by attempt number.
        """
        result = await self._session.execute(
            select(WebhookDelivery)
            .where(WebhookDelivery.prediction_id == prediction_id)
            .order_by(WebhookDelivery.attempt)
        )
        return list(result.scalars().all())

    async def get(self, delivery_id: uuid.UUID) -> WebhookDelivery | None:
        """Fetch a delivery by id.

        Parameters:
            delivery_id: Delivery UUID.

        Returns:
            The WebhookDelivery or None.
        """
        return await self._session.get(WebhookDelivery, delivery_id)
