"""Model schema endpoint exposing inputs/outputs JSON Schema and limits."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter

from app.config import get_settings
from app.constants import MAX_PROMPT_CHARS, MAX_SEED
from app.ml.registry import get_variant

__all__ = ["router"]

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/video-to-video-sfx")
async def model_schema() -> dict[str, Any]:
    """Describe the model's input/output schema, defaults, and limits.

    Returns:
        Envelope carrying the JSON Schema plus variant metadata.
    """
    settings = get_settings()
    spec = get_variant(settings.model.model_variant)
    manifest_revision = _manifest_revision()
    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["video"],
        "properties": {
            "video": {
                "anyOf": [
                    {"type": "string", "format": "uri"},
                    {"type": "string", "pattern": "^data:"},
                ],
                "description": "Source video URL, data: URI, or upload URL/file part",
            },
            "prompt": {"type": "string", "maxLength": MAX_PROMPT_CHARS, "default": ""},
            "negative_prompt": {"type": "string", "maxLength": MAX_PROMPT_CHARS, "default": ""},
            "seed": {
                "type": "integer",
                "minimum": -1,
                "maximum": MAX_SEED,
                "default": -1,
                "description": "-1 generates and echoes a random seed",
            },
            "num_inference_steps": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            "guidance_scale": {"type": "number", "minimum": 0.0, "maximum": 15.0, "default": 4.5},
            "duration": {
                "anyOf": [{"type": "number", "minimum": 0.5}, {"type": "null"}],
                "default": None,
                "description": "Generation window; null = full video",
            },
            "start_time": {"type": "number", "minimum": 0, "default": 0.0},
            "audio_mode": {"type": "string", "enum": ["replace", "mix", "duck"], "default": "replace"},
            "sfx_gain_db": {"type": "number", "minimum": -30, "maximum": 12, "default": 0.0},
            "original_audio_gain_db": {"type": "number", "minimum": -60, "maximum": 12, "default": -6.0},
            "duck_threshold_db": {"type": "number", "minimum": -60, "maximum": 0, "default": -24.0},
            "duck_ratio": {"type": "number", "minimum": 1, "maximum": 20, "default": 4.0},
            "duck_attack_ms": {"type": "number", "minimum": 1, "maximum": 500, "default": 15.0},
            "duck_release_ms": {"type": "number", "minimum": 10, "maximum": 2000, "default": 250.0},
            "target_loudness_lufs": {
                "anyOf": [{"type": "number", "minimum": -40, "maximum": 0}, {"type": "null"}],
                "default": settings.defaults.default_target_lufs,
            },
            "true_peak_db": {"type": "number", "minimum": -12, "maximum": 0, "default": -1.0},
            "output_format": {"type": "string", "enum": ["mp4"], "default": "mp4"},
            "video_handling": {"type": "string", "enum": ["copy", "reencode"], "default": "copy"},
            "return_audio_only": {"type": "boolean", "default": False},
            "enable_safety_checker": {"type": "boolean", "default": True},
            "webhook_url": {"anyOf": [{"type": "string", "format": "uri"}, {"type": "null"}]},
            "metadata": {"type": "object", "default": {}},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "minLength": 26, "maxLength": 26},
            "status": {
                "type": "string",
                "enum": ["created", "queued", "processing", "completed", "failed", "canceled"],
            },
            "outputs": {"type": "array", "items": {"type": "string", "format": "uri"}},
            "has_nsfw_contents": {"type": "array", "items": {"type": "boolean"}},
            "error": {"type": "string"},
            "error_code": {"type": "string"},
            "progress": {"type": "integer", "minimum": 0, "maximum": 100},
        },
    }
    data = {
        "id": settings.model.model_id,
        "variant": spec.name,
        "weight_revision": manifest_revision,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "limits": {
            "max_video_duration_s": settings.media.max_video_duration_s,
            "min_video_duration_s": settings.media.min_video_duration_s,
            "max_upload_bytes": settings.server.max_upload_bytes,
            "max_video_pixels": settings.media.max_video_pixels,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "allowed_video_codecs": settings.media.allowed_video_codecs,
        },
        "audio": {
            "sample_rate": spec.sample_rate,
            "n_mels": spec.n_mels,
            "latent_fps": spec.latent_fps,
            "output_sample_rate": settings.output.output_audio_sr,
            "output_channels": settings.output.output_audio_channels,
        },
        "licence": spec.licence,
    }
    return {"code": 200, "message": "success", "data": data}


def _manifest_revision() -> str:
    """Read the weight revision hash from the manifest.

    Returns:
        The revision string or `unavailable` when the manifest is absent.
    """
    try:
        from app.ml.weights import load_manifest

        manifest = load_manifest(get_settings().model.weights_manifest_path)
    except Exception:
        return "unavailable"
    return manifest.revision


def schema_digest(payload: dict[str, Any]) -> str:
    """Hash a schema payload for drift checks (CI helper).

    Parameters:
        payload: Schema dict.

    Returns:
        SHA-256 hex prefix.
    """
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:16]
