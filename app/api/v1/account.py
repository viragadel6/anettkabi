"""Account usage endpoint: current-period usage versus quota."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.db.repositories.usage import UsageRepository
from app.dependencies import ApiKeyId, SessionDep, SettingsDep
from app.utils.time import isoformat_ms

__all__ = ["router"]

router = APIRouter(prefix="/account", tags=["account"])


@router.get("/usage")
async def account_usage(
    request: Request,
    api_key_id: ApiKeyId,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Return the caller's usage for the current billing month.

    Returns:
        Envelope with used seconds, quota, and utilization.
    """
    quota = getattr(request.state, "api_key_monthly_quota", None)
    quota_seconds = (
        float(quota) if quota is not None else settings.auth.monthly_seconds_quota
    )
    repo = UsageRepository(session)
    month_seconds = await repo.month_seconds(api_key_id)
    gpu_ms = await repo.month_gpu_ms(api_key_id)
    data = {
        "period_start": isoformat_ms(_month_start()),
        "used_seconds": round(month_seconds, 3),
        "quota_seconds": quota_seconds,
        "utilization": round(month_seconds / quota_seconds, 4) if quota_seconds else 0.0,
        "gpu_ms": gpu_ms,
    }
    return {"code": 200, "message": "success", "data": data}


def _month_start():
    """Return the first moment of the current UTC month.

    Returns:
        Timezone-aware datetime.
    """
    import datetime as dt

    now = dt.datetime.now(tz=dt.UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
