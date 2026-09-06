"""Prediction payload rendering shared by API endpoints, webhooks, and SDK docs."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.constants import API_PREFIX
from app.db.models import Prediction
from app.utils.time import isoformat_ms

__all__ = ["prediction_envelope", "render_prediction_payload"]

_INPUT_KEYS = (
    "video",
    "prompt",
    "negative_prompt",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "duration",
    "start_time",
    "audio_mode",
    "sfx_gain_db",
    "original_audio_gain_db",
    "target_loudness_lufs",
    "output_format",
    "enable_safety_checker",
)


def render_prediction_payload(row: Prediction, settings: Settings) -> dict[str, Any]:
    """Render one prediction row into the standard `data` payload.

    Parameters:
        row: The prediction row.
        settings: Settings (public base URL for `urls.get`).

    Returns:
        JSON-serializable payload dict.
    """
    stored_input = dict(row.input or {})
    rendered_input: dict[str, Any] = {}
    for key in _INPUT_KEYS:
        rendered_input[key] = stored_input.get(key)
    rendered_input.setdefault("video", "")
    rendered_input.setdefault("prompt", "")
    rendered_input.setdefault("negative_prompt", "")
    rendered_input.setdefault("seed", 0)
    rendered_input.setdefault("num_inference_steps", settings.defaults.default_steps)
    rendered_input.setdefault("guidance_scale", settings.defaults.default_cfg)
    rendered_input.setdefault("duration", None)
    rendered_input.setdefault("audio_mode", settings.defaults.default_audio_mode)
    rendered_input.setdefault("sfx_gain_db", settings.defaults.default_sfx_gain_db)
    rendered_input.setdefault(
        "original_audio_gain_db", settings.defaults.default_original_gain_db
    )
    rendered_input.setdefault("target_loudness_lufs", settings.defaults.default_target_lufs)
    rendered_input.setdefault("output_format", "mp4")
    rendered_input.setdefault("enable_safety_checker", settings.safety.safety_enabled)
    base = settings.server.public_base_url.rstrip("/")
    timings = dict(row.timings or {})
    return {
        "id": row.id,
        "model": row.model,
        "status": row.status.value,
        "input": rendered_input,
        "outputs": list(row.outputs or []),
        "urls": {"get": f"{base}{API_PREFIX}/predictions/{row.id}/result"},
        "has_nsfw_contents": list(row.has_nsfw_contents or []),
        "error": row.error or "",
        "error_code": row.error_code or "",
        "progress": int(row.progress or 0),
        "stage": row.stage or "",
        "timings": timings,
        "execution_time": int(row.execution_time_ms or 0),
        "created_at": isoformat_ms(row.created_at),
        "started_at": isoformat_ms(row.started_at),
        "completed_at": isoformat_ms(row.completed_at),
    }


def prediction_envelope(row: Prediction, settings: Settings) -> dict[str, Any]:
    """Wrap a rendered payload in the standard envelope.

    Parameters:
        row: The prediction row.
        settings: Settings.

    Returns:
        The full envelope dict.
    """
    return {
        "code": 200,
        "message": "success",
        "data": render_prediction_payload(row, settings),
    }
