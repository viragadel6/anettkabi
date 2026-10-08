from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

import redis.asyncio as aioredis

__all__ = [
    "TASK_SUBMITTED_CHANNEL",
    "WakeSubscription",
    "publish_task_submitted",
]

logger = logging.getLogger("coordination.wakeup")

TASK_SUBMITTED_CHANNEL = "task_engine:task_submitted"


async def publish_task_submitted(
    redis_client: aioredis.Redis,
    task_type: str,
    channel: str = TASK_SUBMITTED_CHANNEL,
) -> bool:
    try:
        await redis_client.publish(channel, task_type)
        return True
    except Exception:
        logger.warning(
            "Failed to publish task submission wakeup for %s; "
            "workers will pick the task up via polling",
            task_type,
        )
        return False


class WakeSubscription:
    def __init__(
        self,
        redis_client: aioredis.Redis,
        on_wake: Callable[[], None],
        channel: str = TASK_SUBMITTED_CHANNEL,
    ) -> None:
        self._redis = redis_client
        self._on_wake = on_wake
        self._channel = channel
        self._task: asyncio.Task[None] | None = None
        self._pubsub: aioredis.client.PubSub | None = None

    @property
    def active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        self._task = asyncio.create_task(
            self._listen(), name="task-submission-wake-subscription"
        )

    async def _listen(self) -> None:
        backoff = 0.5
        while True:
            try:
                self._pubsub = self._redis.pubsub()
                await self._pubsub.subscribe(self._channel)
                backoff = 0.5
                async for message in self._pubsub.listen():
                    if message.get("type") == "message":
                        self._on_wake()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Wake subscription interrupted; retrying in %.1fs "
                    "(polling remains active as fallback)",
                    backoff,
                )
                await self._cleanup_pubsub()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 10.0)

    async def _cleanup_pubsub(self) -> None:
        if self._pubsub is not None:
            with contextlib.suppress(Exception):
                await self._pubsub.unsubscribe(self._channel)
            with contextlib.suppress(Exception):
                await self._pubsub.aclose()
            self._pubsub = None

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await self._cleanup_pubsub()
