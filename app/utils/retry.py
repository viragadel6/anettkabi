"""Retry helpers with exponential backoff and full jitter."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

__all__ = ["backoff_delay_s", "retry_async"]

T = TypeVar("T")


def backoff_delay_s(attempt: int, base_s: float, max_s: float) -> float:
    """Compute a full-jitter exponential backoff delay.

    Parameters:
        attempt: Zero-based attempt index.
        base_s: Base delay in seconds.
        max_s: Maximum delay in seconds.

    Returns:
        A random delay in [0, min(max_s, base_s * 2**attempt)).
    """
    ceiling = min(max_s, base_s * (2**attempt))
    return random.uniform(0.0, ceiling) if ceiling > 0 else 0.0


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_s: float,
    max_s: float,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run an async operation with bounded retries and jittered backoff.

    Parameters:
        operation: Coroutine factory to execute.
        attempts: Maximum total attempts (>= 1).
        base_s: Backoff base delay.
        max_s: Backoff ceiling.
        retry_on: Exception types eligible for retry.
        sleep: Awaitable sleep used between attempts (injectable for tests).

    Returns:
        The first successful result.

    Raises:
        Exception: The last raised exception when attempts are exhausted or the
            exception type is not retryable.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except retry_on as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            await sleep(backoff_delay_s(attempt, base_s, max_s))
    assert last_error is not None
    raise last_error
