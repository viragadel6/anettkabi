from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType

import redis.asyncio as aioredis

from config import Settings, get_settings

__all__ = [
    "ACQUIRE_SCRIPT",
    "LEASE_KEY_PREFIX",
    "LeaseAcquisitionError",
    "RedisLeaseManager",
    "RELEASE_SCRIPT",
    "RENEW_SCRIPT",
    "close_redis_client",
    "create_redis_client",
    "distributed_lease",
]

logger = logging.getLogger("coordination.distributed_lock")

LEASE_KEY_PREFIX = "task_engine:lease:"

ACQUIRE_SCRIPT = """
if redis.call('exists', KEYS[1]) == 0 then
    redis.call('set', KEYS[1], ARGV[1], 'PX', ARGV[2])
    return 1
elseif redis.call('get', KEYS[1]) == ARGV[1] then
    redis.call('pexpire', KEYS[1], ARGV[2])
    return 1
else
    return 0
end
"""

RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""

RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

_redis_client: aioredis.Redis | None = None


def create_redis_client(settings: Settings | None = None) -> aioredis.Redis:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    resolved = settings or get_settings()
    _redis_client = aioredis.Redis.from_url(
        resolved.redis.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=resolved.redis.REDIS_MAX_CONNECTIONS,
        socket_timeout=resolved.redis.REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=resolved.redis.REDIS_SOCKET_CONNECT_TIMEOUT,
        health_check_interval=30,
        retry_on_timeout=True,
    )
    return _redis_client


def get_redis_client() -> aioredis.Redis:
    if _redis_client is None:
        return create_redis_client()
    return _redis_client


async def close_redis_client() -> None:
    global _redis_client
    if _redis_client is not None:
        with contextlib.suppress(Exception):
            await _redis_client.aclose()
        _redis_client = None


class LeaseAcquisitionError(RuntimeError):
    def __init__(self, resource_id: str, worker_id: str) -> None:
        self.resource_id = resource_id
        self.worker_id = worker_id
        super().__init__(
            f"Worker {worker_id} failed to acquire lease for resource {resource_id}"
        )


class RedisLeaseManager:
    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client
        self._acquire_script = redis_client.register_script(ACQUIRE_SCRIPT)
        self._renew_script = redis_client.register_script(RENEW_SCRIPT)
        self._release_script = redis_client.register_script(RELEASE_SCRIPT)

    @staticmethod
    def _key(resource_id: str) -> str:
        return f"{LEASE_KEY_PREFIX}{resource_id}"

    async def acquire(
        self, resource_id: str, token: str, ttl_ms: int
    ) -> bool:
        result = await self._acquire_script(
            keys=[self._key(resource_id)], args=[token, ttl_ms]
        )
        return int(result) == 1

    async def renew(self, resource_id: str, token: str, ttl_ms: int) -> bool:
        result = await self._renew_script(
            keys=[self._key(resource_id)], args=[token, ttl_ms]
        )
        return int(result) == 1

    async def release(self, resource_id: str, token: str) -> bool:
        result = await self._release_script(keys=[self._key(resource_id)], args=[token])
        return int(result) == 1

    async def try_acquire_set_nx(
        self, resource_id: str, token: str, ttl_ms: int
    ) -> bool:
        acquired = await self._redis.set(
            self._key(resource_id), token, nx=True, px=ttl_ms
        )
        return bool(acquired)


class DistributedLease:
    def __init__(
        self,
        redis_client: aioredis.Redis,
        resource_id: str,
        worker_id: str,
        ttl_ms: int,
    ) -> None:
        self._manager = RedisLeaseManager(redis_client)
        self.resource_id = resource_id
        self.worker_id = worker_id
        self.ttl_ms = int(ttl_ms)
        self.token = f"{worker_id}:{uuid.uuid4().hex}"
        self._refresher_task: asyncio.Task[None] | None = None
        self._released = False

    async def __aenter__(self) -> DistributedLease:
        acquired = await self._manager.acquire(self.resource_id, self.token, self.ttl_ms)
        if not acquired:
            raise LeaseAcquisitionError(self.resource_id, self.worker_id)
        self._refresher_task = asyncio.create_task(
            self._refresh_loop(), name=f"lease-refresh-{self.resource_id}"
        )
        return self

    async def _refresh_loop(self) -> None:
        refresh_interval = max(self.ttl_ms / 3.0 / 1000.0, 0.05)
        while True:
            await asyncio.sleep(refresh_interval)
            try:
                renewed = await self._manager.renew(
                    self.resource_id, self.token, self.ttl_ms
                )
                if not renewed:
                    logger.warning(
                        "Lease for resource %s lost; token no longer owns the key",
                        self.resource_id,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Lease refresh failed for resource %s; will retry next cycle",
                    self.resource_id,
                )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._refresher_task is not None:
            self._refresher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._refresher_task
            self._refresher_task = None
        if not self._released:
            self._released = True
            try:
                await self._manager.release(self.resource_id, self.token)
            except Exception:
                logger.exception(
                    "Failed to release lease for resource %s; it will expire via TTL",
                    self.resource_id,
                )


@asynccontextmanager
async def distributed_lease(
    redis_client: aioredis.Redis,
    resource_id: str,
    worker_id: str,
    ttl_ms: int,
) -> AsyncIterator[DistributedLease]:
    lease = DistributedLease(redis_client, resource_id, worker_id, ttl_ms)
    async with lease:
        yield lease
