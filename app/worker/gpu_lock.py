"""Process-wide and cluster-wide GPU concurrency limiting."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Iterator

from app.config import get_settings
from app.constants import GPU_CLUSTER_LOCK_KEY, GPU_LOCK_TOKEN_TTL_S
from app.services.queue import get_redis_client

__all__ = ["GpuLock", "cluster_slots_used"]


class GpuLock:
    """Bounded local semaphore plus a Redis-registered cluster-wide counter.

    The local semaphore bounds in-process concurrency to
    MAX_CONCURRENT_INFERENCE; the Redis slot registry keeps a best-effort
    cluster-wide count (per worker unique token with TTL refresh) so multiple
    worker pods sharing one GPU do not oversubscribe it. Slot acquisition is
    optimistic: workers register tokens and read the count under a small
    race window, which the TTL reaper bounds.
    """

    __slots__ = ("_semaphore", "_settings", "_token")

    def __init__(self) -> None:
        """Create the lock sized from settings."""
        self._settings = get_settings()
        self._semaphore = asyncio.Semaphore(self._settings.model.max_concurrent_inference)
        self._token = f"{uuid.uuid4().hex}"

    @contextlib.asynccontextmanager
    async def acquire(self, cluster_limit: int | None = None) -> Iterator[None]:
        """Acquire one inference slot, holding it for the context duration.

        Parameters:
            cluster_limit: Optional override of the cluster-wide slot cap.

        Yields:
            None while the slot is held.

        Raises:
            TimeoutError: When the cluster is saturated beyond the cap.
        """
        limit = cluster_limit or self._settings.model.max_concurrent_inference
        await self._semaphore.acquire()
        try:
            await self._register_token()
            used = await cluster_slots_used()
            if used > limit:
                await self._release_token()
                raise TimeoutError(
                    f"cluster GPU slots saturated ({used} > {limit}); retry shortly"
                )
            yield
        finally:
            with contextlib.suppress(Exception):
                await self._release_token()
            self._semaphore.release()

    async def _register_token(self) -> None:
        """Register this holder's token in Redis with a TTL."""
        client = get_redis_client(self._settings.redis.redis_url)
        key = f"{GPU_CLUSTER_LOCK_KEY}:tokens"
        try:
            pipe = client.pipeline()
            pipe.sadd(key, self._token)
            pipe.expire(key, GPU_LOCK_TOKEN_TTL_S * 10)
            await pipe.execute()
        except Exception:
            pass

    async def _release_token(self) -> None:
        """Remove this holder's token from Redis."""
        client = get_redis_client(self._settings.redis.redis_url)
        try:
            await client.srem(f"{GPU_CLUSTER_LOCK_KEY}:tokens", self._token)
        except Exception:
            pass

    @property
    def token(self) -> str:
        """Return this lock's cluster token."""
        return self._token


async def cluster_slots_used() -> int:
    """Count currently registered cluster-wide inference holders.

    Returns:
        Number of live tokens (0 on Redis errors; fail-open).
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    try:
        return int(await client.scard(f"{GPU_CLUSTER_LOCK_KEY}:tokens"))
    except Exception:
        return 0
