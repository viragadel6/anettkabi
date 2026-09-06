"""Redis Streams job queue with consumer groups, autoclaim, and dead-lettering."""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass

import redis.asyncio as aioredis

from app.config import get_settings
from app.errors import ErrorCode, ServiceError

__all__ = [
    "JobMessage",
    "ack_job",
    "claim_stale_jobs",
    "close_redis_clients",
    "enqueue_job",
    "ensure_stream",
    "get_redis_client",
    "move_to_dead_letter",
    "read_jobs",
    "stream_depth",
]

_clients: dict[str, aioredis.Redis] = {}


def get_redis_client(redis_url: str) -> aioredis.Redis:
    """Return a shared async Redis client for a URL.

    Parameters:
        redis_url: Connection URL.

    Returns:
        A cached redis.asyncio.Redis instance.
    """
    if redis_url not in _clients:
        _clients[redis_url] = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=15,
            health_check_interval=30,
        )
    return _clients[redis_url]


async def close_redis_clients() -> None:
    """Close all cached Redis clients."""
    for client in _clients.values():
        with contextlib.suppress(Exception):
            await client.aclose()
    _clients.clear()


@dataclass(slots=True)
class JobMessage:
    """A queue message payload.

    Attributes:
        prediction_id: Prediction to process.
        api_key_id: Owning key id (string UUID).
        attempt: Attempt counter (1-based).
        enqueued_at: Epoch seconds of enqueue.
        priority: Lower runs first.
        stream_id: Redis Stream message id once read.
    """

    prediction_id: str
    api_key_id: str
    attempt: int
    enqueued_at: float
    priority: int
    stream_id: str = ""


async def ensure_stream(stream: str, group: str) -> None:
    """Create the stream and consumer group when missing.

    Parameters:
        stream: Stream key.
        group: Consumer group name.

    Raises:
        ServiceError: internal_error when Redis is unreachable.
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    try:
        await client.xgroup_create(stream, group, id="0-0", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise ServiceError(ErrorCode.INTERNAL_ERROR, f"queue setup failed: {exc}") from exc


async def enqueue_job(
    prediction_id: str,
    api_key_id: str,
    *,
    attempt: int = 1,
    priority: int = 5,
) -> str:
    """Append a job message to the stream (trimmed to QUEUE_MAX_LEN).

    Parameters:
        prediction_id: Prediction id.
        api_key_id: Owning key id.
        attempt: Attempt counter.
        priority: Priority bucket (lower first).

    Returns:
        The stream message id.

    Raises:
        ServiceError: internal_error when Redis rejects the write.
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    payload = json.dumps(
        {
            "prediction_id": prediction_id,
            "api_key_id": api_key_id,
            "attempt": attempt,
            "enqueued_at": time.time(),
            "priority": priority,
        },
        separators=(",", ":"),
    )
    try:
        return await client.xadd(
            settings.redis.queue_stream_name,
            {"payload": payload},
            maxlen=settings.redis.queue_max_len,
            approximate=True,
            nomkstream=False,
        )
    except aioredis.RedisError as exc:
        raise ServiceError(ErrorCode.INTERNAL_ERROR, f"failed to enqueue job: {exc}") from exc


async def read_jobs(consumer_name: str, count: int = 1, block_ms: int = 5000) -> list[JobMessage]:
    """Read new messages for a consumer of the configured group.

    Parameters:
        consumer_name: This worker's consumer name.
        count: Max messages to fetch.
        block_ms: XREADGROUP block time.

    Returns:
        Parsed JobMessage list (may be empty on timeout).

    Raises:
        ServiceError: internal_error when the group is missing (recreated once).
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    stream = settings.redis.queue_stream_name
    group = settings.redis.queue_group
    try:
        responses = await client.xreadgroup(
            group,
            consumer_name,
            {stream: ">"},
            count=count,
            block=block_ms,
        )
    except aioredis.ResponseError as exc:
        if "NOGROUP" in str(exc):
            await ensure_stream(stream, group)
            return []
        raise ServiceError(ErrorCode.INTERNAL_ERROR, f"queue read failed: {exc}") from exc
    except aioredis.RedisError as exc:
        raise ServiceError(ErrorCode.INTERNAL_ERROR, f"queue read failed: {exc}") from exc
    messages: list[JobMessage] = []
    for _stream_name, entries in responses:
        for entry_id, fields in entries:
            message = _parse_entry(entry_id, fields)
            if message is not None:
                messages.append(message)
    return messages


async def claim_stale_jobs(consumer_name: str, idle_ms: int | None = None, count: int = 1) -> list[JobMessage]:
    """Steal messages idle beyond the claim threshold (XAUTOCLAIM).

    Parameters:
        consumer_name: Claiming consumer.
        idle_ms: Idle threshold; defaults to settings.
        count: Max messages to claim.

    Returns:
        Claimed JobMessage list.
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    threshold = idle_ms if idle_ms is not None else settings.redis.queue_claim_idle_ms
    try:
        _next, entries, _deleted = await client.xautoclaim(
            settings.redis.queue_stream_name,
            settings.redis.queue_group,
            consumer_name,
            min_idle_time=threshold,
            count=count,
        )
    except (aioredis.RedisError, aioredis.ResponseError):
        return []
    messages: list[JobMessage] = []
    for entry_id, fields in entries:
        message = _parse_entry(entry_id, fields)
        if message is not None:
            messages.append(message)
    return messages


def _parse_entry(entry_id: str, fields: dict[str, str]) -> JobMessage | None:
    """Decode one stream entry into a JobMessage.

    Parameters:
        entry_id: Redis stream id.
        fields: Entry fields mapping.

    Returns:
        The decoded message, or None for malformed entries.
    """
    raw = fields.get("payload")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return JobMessage(
            prediction_id=str(data["prediction_id"]),
            api_key_id=str(data["api_key_id"]),
            attempt=int(data.get("attempt", 1)),
            enqueued_at=float(data.get("enqueued_at", time.time())),
            priority=int(data.get("priority", 5)),
            stream_id=entry_id,
        )
    except (KeyError, ValueError, TypeError):
        return None


async def ack_job(stream_id: str) -> None:
    """Acknowledge a processed message.

    Parameters:
        stream_id: The stream entry id.
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    with contextlib.suppress(aioredis.RedisError):
        await client.xack(settings.redis.queue_stream_name, settings.redis.queue_group, stream_id)


async def move_to_dead_letter(message: JobMessage, reason: str) -> None:
    """Copy an exhausted job to the dead-letter stream and ack the original.

    Parameters:
        message: The terminal job message.
        reason: Terminal error description.
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    with contextlib.suppress(aioredis.RedisError):
        await client.xadd(
            f"{settings.redis.queue_stream_name}:dlq",
            {
                "payload": json.dumps(
                    {
                        "prediction_id": message.prediction_id,
                        "api_key_id": message.api_key_id,
                        "attempt": message.attempt,
                        "reason": reason,
                        "ts": time.time(),
                    }
                )
            },
            maxlen=settings.redis.queue_max_len,
            approximate=True,
        )
        await ack_job(message.stream_id)


async def stream_depth() -> int:
    """Return pending (undelivered) message count in the stream.

    Returns:
        Number of entries currently in the stream, 0 on error.
    """
    settings = get_settings()
    client = get_redis_client(settings.redis.redis_url)
    try:
        return int(await client.xlen(settings.redis.queue_stream_name))
    except aioredis.RedisError:
        return 0


def new_consumer_name() -> str:
    """Generate a unique consumer name for this process.

    Returns:
        Host-unique consumer identifier.
    """
    return f"worker-{uuid.uuid4().hex[:12]}"
