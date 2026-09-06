"""Integration fixtures: live postgres/redis/minio from the compose test profile."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INTEGRATION_ENV = {
    "VSFX_SERVER__PUBLIC_BASE_URL": "http://localhost:8000",
    "VSFX_DATABASE__DATABASE_URL": os.environ.get(
        "VSFX_DATABASE__DATABASE_URL",
        "postgresql+asyncpg://vsfx:vsfx@localhost:5432/vsfx",
    ),
    "VSFX_REDIS__REDIS_URL": os.environ.get("VSFX_REDIS__REDIS_URL", "redis://localhost:6379/0"),
    "VSFX_STORAGE__S3_ENDPOINT_URL": os.environ.get(
        "VSFX_STORAGE__S3_ENDPOINT_URL", "http://localhost:9000"
    ),
    "VSFX_STORAGE__S3_BUCKET": os.environ.get("VSFX_STORAGE__S3_BUCKET", "vsfx"),
    "VSFX_STORAGE__S3_ACCESS_KEY_ID": os.environ.get(
        "VSFX_STORAGE__S3_ACCESS_KEY_ID", "minioadmin"
    ),
    "VSFX_STORAGE__S3_SECRET_ACCESS_KEY": os.environ.get(
        "VSFX_STORAGE__S3_SECRET_ACCESS_KEY", "minioadmin"
    ),
    "VSFX_STORAGE__S3_FORCE_PATH_STYLE": "true",
    "VSFX_STORAGE__S3_KEY_PREFIX": "vsfx-it",
    "VSFX_SAFETY__SAFETY_ENABLED": "false",
    "VSFX_AUTH__AUTH_REQUIRED": "true",
    "VSFX_WEBHOOK__WEBHOOK_SIGNING_SECRET": "integration-secret",
    "VSFX_TELEMETRY__LOG_LEVEL": "WARNING",
    "VSFX_JOBS__WORK_DIR": str(REPO_ROOT / ".pytest-work"),
}


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _services_available() -> bool:
    return (
        _port_open("localhost", 5432)
        and _port_open("localhost", 6379)
        and _port_open("localhost", 9000)
    )


pytestmark = pytest.mark.integration

if not _services_available():
    pytest.skip(
        "integration services (postgres/redis/minio) not reachable; "
        "run: docker compose -f docker/docker-compose.yml --profile test up -d",
        allow_module_level=True,
    )


@pytest.fixture
def integration_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Apply the integration environment and ffmpeg shims when needed.

    Parameters:
        monkeypatch: Pytest monkeypatch.
        tmp_path: Scratch directory.

    Yields:
        The applied environment mapping.
    """
    import shutil

    for name, value in INTEGRATION_ENV.items():
        monkeypatch.setenv(name, value)
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from app.config import get_settings

    get_settings.cache_clear()
    yield INTEGRATION_ENV
    get_settings.cache_clear()
