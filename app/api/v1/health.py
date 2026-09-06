"""Health, readiness, version, and metrics endpoints."""

from __future__ import annotations

import shutil
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from app import SERVICE_NAME, __version__
from app.config import get_settings
from app.services.queue import get_redis_client
from app.services.storage import get_storage_service
from app.telemetry.metrics import render_metrics
from app.utils.time import utc_now_isoformat

__all__ = ["router"]

router = APIRouter(tags=["health"])


@router.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness probe.

    Returns:
        Minimal ok payload.
    """
    return {"status": "ok", "service": SERVICE_NAME, "time": utc_now_isoformat()}


@router.get("/readyz", include_in_schema=False)
async def readyz() -> dict[str, Any]:
    """Readiness probe with per-dependency checks.

    Returns:
        Dict with overall status and a checks map covering database, redis,
        storage, ffmpeg, and weights availability.
    """
    checks: dict[str, bool] = {}
    checks["database"] = await _check_database()
    checks["redis"] = await _check_redis()
    checks["storage"] = await _check_storage()
    checks["ffmpeg"] = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    checks["weights"] = _check_weights()
    overall = "ok" if all(checks.values()) else "degraded"
    return {"status": overall, "checks": checks, "version": __version__}


@router.get("/version", include_in_schema=False)
async def version() -> dict[str, str]:
    """Service version metadata.

    Returns:
        Version and model identifiers.
    """
    settings = get_settings()
    return {
        "version": __version__,
        "service": SERVICE_NAME,
        "model": settings.model.model_id,
        "model_variant": settings.model.model_variant,
    }


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> Response:
    """Prometheus exposition endpoint.

    Returns:
        Metrics text response.
    """
    settings = get_settings()
    if not settings.telemetry.metrics_enabled:
        return Response(status_code=404)
    body = render_metrics()
    return Response(content=body, media_type="text/plain; version=0.0.4")


async def _check_database() -> bool:
    """Probe the configured database with a trivial query.

    Returns:
        True when reachable.
    """
    from sqlalchemy import text

    from app.db.session import create_engine

    try:
        engine = create_engine(get_settings().database.database_url)
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def _check_redis() -> bool:
    """Probe Redis with a PING.

    Returns:
        True when reachable.
    """
    try:
        client = get_redis_client(get_settings().redis.redis_url)
        return bool(await client.ping())
    except Exception:
        return False


def _check_storage() -> bool:
    """Probe the bucket with a HEAD.

    Returns:
        True when reachable.
    """
    try:
        get_storage_service().ensure_bucket()
    except Exception:
        return False
    return True


def _check_weights() -> bool:
    """Check manifest presence for the configured variant.

    Returns:
        True when every required artifact is present and checksum-valid.
    """
    from app.ml.registry import get_variant
    from app.ml.weights import load_manifest

    settings = get_settings()
    try:
        spec = get_variant(settings.model.model_variant)
        load_manifest(settings.model.weights_manifest_path)
        from pathlib import Path

        return all(
            (Path(settings.model.weights_dir) / filename).is_file()
            for filename in spec.file_list
        )
    except Exception:
        return False
