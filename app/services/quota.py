"""Quota and concurrency enforcement for prediction creation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.db.repositories.predictions import PredictionRepository
from app.db.repositories.usage import UsageRepository
from app.db.session import session_scope
from app.errors import ErrorCode, ServiceError
from app.utils.time import utc_now

__all__ = ["QuotaSnapshot", "enforce_quotas", "usage_snapshot"]


@dataclass(slots=True)
class QuotaSnapshot:
    """Current period usage versus configured allowances.

    Attributes:
        month_seconds: Billed seconds used this month.
        quota_seconds: Allowed seconds per month.
        active_jobs: Non-terminal predictions right now.
        concurrency_limit: Allowed concurrent jobs.
    """

    month_seconds: float
    quota_seconds: float
    active_jobs: int
    concurrency_limit: int


async def usage_snapshot(
    api_key_id: uuid.UUID,
    *,
    quota_seconds: float,
    concurrency_limit: int,
) -> QuotaSnapshot:
    """Read the current usage snapshot for a key.

    Parameters:
        api_key_id: Caller key.
        quota_seconds: Monthly allowance.
        concurrency_limit: Concurrent-job allowance.

    Returns:
        QuotaSnapshot with fresh counts.
    """
    async with session_scope() as session:
        month_seconds = await UsageRepository(session).month_seconds(api_key_id, utc_now())
        active = await PredictionRepository(session).count_active_for_key(api_key_id)
    return QuotaSnapshot(
        month_seconds=month_seconds,
        quota_seconds=quota_seconds,
        active_jobs=active,
        concurrency_limit=concurrency_limit,
    )


async def enforce_quotas(
    api_key_id: uuid.UUID,
    *,
    quota_seconds: float,
    concurrency_limit: int,
) -> QuotaSnapshot:
    """Raise when the key exceeds quota or concurrency, else return the snapshot.

    Parameters:
        api_key_id: Caller key.
        quota_seconds: Monthly allowance.
        concurrency_limit: Concurrent-job allowance.

    Returns:
        QuotaSnapshot when allowed.

    Raises:
        ServiceError: quota_exceeded or concurrency_limited.
    """
    snapshot = await usage_snapshot(
        api_key_id,
        quota_seconds=quota_seconds,
        concurrency_limit=concurrency_limit,
    )
    if snapshot.month_seconds >= snapshot.quota_seconds:
        raise ServiceError(
            ErrorCode.QUOTA_EXCEEDED,
            f"monthly quota exhausted ({snapshot.month_seconds:.1f}s of "
            f"{snapshot.quota_seconds:.0f}s billed)",
        )
    if snapshot.active_jobs >= snapshot.concurrency_limit:
        raise ServiceError(
            ErrorCode.CONCURRENCY_LIMITED,
            f"concurrency limit reached ({snapshot.active_jobs} active of "
            f"{snapshot.concurrency_limit} allowed)",
            retry_after_s=30.0,
        )
    return snapshot
