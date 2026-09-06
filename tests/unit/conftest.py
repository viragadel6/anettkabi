"""Shared pytest fixtures: path bootstrap, settings env, ffmpeg shims."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = REPO_ROOT / "sdk" / "python"
for candidate in (str(REPO_ROOT), str(SDK_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

os.environ.setdefault("PYTEST_CURRENT_TEST", "1")

TEST_ENV = {
    "VSFX_SERVER__HOST": "127.0.0.1",
    "VSFX_SERVER__PORT": "8901",
    "VSFX_SERVER__PUBLIC_BASE_URL": "http://127.0.0.1:8901",
    "VSFX_SERVER__CORS_ORIGINS": '["http://localhost:5173"]',
    "VSFX_DATABASE__DATABASE_URL": "postgresql+asyncpg://vsfx:vsfx@127.0.0.1:5432/vsfx_test",
    "VSFX_REDIS__REDIS_URL": "redis://127.0.0.1:6379/15",
    "VSFX_STORAGE__S3_ENDPOINT_URL": "http://127.0.0.1:9000",
    "VSFX_STORAGE__S3_BUCKET": "vsfx-test",
    "VSFX_STORAGE__S3_ACCESS_KEY_ID": "minioadmin",
    "VSFX_STORAGE__S3_SECRET_ACCESS_KEY": "minioadmin",
    "VSFX_STORAGE__S3_FORCE_PATH_STYLE": "true",
    "VSFX_STORAGE__S3_KEY_PREFIX": "vsfx",
    "VSFX_MEDIA__MAX_VIDEO_DURATION_S": "300",
    "VSFX_MODEL__MODEL_VARIANT": "small_16k",
    "VSFX_MODEL__AUDIO_SAMPLE_RATE": "16000",
    "VSFX_DEFAULTS__DEFAULT_STEPS": "32",
    "VSFX_DEFAULTS__DEFAULT_CFG": "3.0",
    "VSFX_OUTPUT__OUTPUT_AUDIO_CODEC": "aac",
    "VSFX_JOBS__JOB_MAX_ATTEMPTS": "3",
    "VSFX_JOBS__JOB_TOTAL_TIMEOUT_S": "1800",
    "VSFX_JOBS__WORK_DIR": str(REPO_ROOT / ".pytest-work"),
    "VSFX_AUTH__AUTH_REQUIRED": "true",
    "VSFX_SAFETY__SAFETY_ENABLED": "true",
    "VSFX_SAFETY__SAFETY_BLOCKLIST_PATH": str(REPO_ROOT / "config" / "safety_blocklist.txt"),
    "VSFX_WEBHOOK__WEBHOOK_SIGNING_SECRET": "test-webhook-secret",
    "VSFX_TELEMETRY__METRICS_ENABLED": "true",
    "VSFX_TELEMETRY__LOG_LEVEL": "INFO",
}


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Provide a fully-populated settings environment with ffmpeg shims.

    Parameters:
        monkeypatch: Pytest monkeypatch.
        tmp_path: Temporary directory for the shim binaries.

    Yields:
        The applied environment mapping.
    """
    for name, value in TEST_ENV.items():
        monkeypatch.setenv(name, value)
    for tool in ("ffmpeg", "ffprobe"):
        shim = tmp_path / tool
        shim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        shim.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    from app.config import get_settings

    get_settings.cache_clear()
    yield TEST_ENV
    get_settings.cache_clear()
