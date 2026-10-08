from __future__ import annotations

import asyncio
import json
import logging

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import session_scope
from observability.tracing import outbox_dispatch_span
from repositories.outbox_repository import OutboxRepository

__all__ = ["OutboxRelay"]

logger = logging.getLogger("worker.outbox_relay")

STREAM_MAX_LENGTH = 100000


class OutboxRelay:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis_client: aioredis.Redis,
        stream_name: str,
        poll_interval_seconds: float = 0.25,
        batch_size: int = 100,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis_client
        self._stream_name = stream_name
        self._poll_interval_seconds = poll_interval_seconds
        self._batch_size = batch_size
        self._stop_event = asyncio.Event()
        self._running = False
        self.healthy = False
        self.last_published_event_id = 0
        self.published_event_count = 0

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        self._running = True
        self.healthy = True
        logger.info("OutboxRelay starting, publishing to stream %s", self._stream_name)
        try:
            while not self._stop_event.is_set():
                try:
                    processed = await self._relay_batch()
                    self.healthy = True
                    if processed == 0:
                        await self._sleep_or_stop(self._poll_interval_seconds)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Outbox relay batch failed; retrying after backoff")
                    self.healthy = False
                    await self._sleep_or_stop(self._poll_interval_seconds * 4.0)
        finally:
            self._running = False
            logger.info("OutboxRelay stopped")

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _relay_batch(self) -> int:
        async with session_scope(self._session_factory, "outbox.fetch_unpublished") as session:
            repository = OutboxRepository(session)
            events = await repository.fetch_unpublished_events(self._batch_size)
            if not events:
                return 0
            with outbox_dispatch_span(len(events)):
                for event in events:
                    await self._redis.xadd(
                        self._stream_name,
                        {
                            "event_id": str(event.id),
                            "aggregate_type": event.aggregate_type,
                            "aggregate_id": str(event.aggregate_id),
                            "event_type": event.event_type,
                            "payload": json.dumps(event.payload, default=str),
                            "created_at": event.created_at.isoformat(),
                        },
                        maxlen=STREAM_MAX_LENGTH,
                        approximate=True,
                    )
                published = await repository.mark_published([event.id for event in events])
            self.last_published_event_id = events[-1].id
            self.published_event_count += published
            logger.debug("Published %d outbox events up to id %d", published, events[-1].id)
            return published
