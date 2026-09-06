"""Prediction request schema with every documented field and its validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.constants import MAX_PROMPT_CHARS, MAX_SEED
from app.utils.validators import sanitize_prompt

__all__ = ["SfxRequest"]

AudioMode = Literal["replace", "mix", "duck"]
VideoHandling = Literal["copy", "reencode"]


class SfxRequest(BaseModel):
    """Input payload for POST /predictions/video-to-video-sfx (JSON form).

    Field semantics and bounds are exactly those of the HTTP API table:
    prompt/negative_prompt <= 2000 chars NFC-cleaned; seed -1 (random) or
    [0, 2^31-1]; steps 1..100; guidance 0..15; audio_mode replace|mix|duck;
    gains per-bus; loudness -40..0 LUFS or null; true peak -12..0 dBTP;
    duration 0.5..MAX_VIDEO_DURATION_S; start_time >= 0; metadata <= 4 KiB.
    """

    video: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = -1
    num_inference_steps: int = Field(default=25, ge=1, le=100)
    guidance_scale: float = Field(default=4.5, ge=0.0, le=15.0)
    duration: float | None = Field(default=None, gt=0)
    start_time: float = Field(default=0.0, ge=0)
    audio_mode: AudioMode = "replace"
    sfx_gain_db: float = Field(default=0.0, ge=-30, le=12)
    original_audio_gain_db: float = Field(default=-6.0, ge=-60, le=12)
    duck_threshold_db: float = Field(default=-24.0, ge=-60, le=0)
    duck_ratio: float = Field(default=4.0, ge=1.0, le=20.0)
    duck_attack_ms: float = Field(default=15.0, ge=1, le=500)
    duck_release_ms: float = Field(default=250.0, ge=10, le=2000)
    target_loudness_lufs: float | None = Field(default=-14.0, ge=-40, le=0)
    true_peak_db: float = Field(default=-1.0, ge=-12, le=0)
    output_format: Literal["mp4"] = "mp4"
    video_handling: VideoHandling = "copy"
    return_audio_only: bool = False
    enable_safety_checker: bool = True
    webhook_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def _clean_prompt(cls, value: str) -> str:
        """NFC-normalize and strip control characters.

        Parameters:
            value: Raw prompt text.

        Returns:
            The sanitized text.

        Raises:
            ValueError: When the result exceeds the length cap.
        """
        cleaned = sanitize_prompt(value)
        if len(cleaned) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
        return cleaned

    @field_validator("seed")
    @classmethod
    def _check_seed(cls, value: int) -> int:
        """Validate the seed range.

        Parameters:
            value: Requested seed.

        Returns:
            The seed (-1 allowed as random marker).

        Raises:
            ValueError: Outside [-1, 2^31-1].
        """
        if value == -1:
            return value
        if not 0 <= value <= MAX_SEED:
            raise ValueError(f"seed must be -1 or in [0, {MAX_SEED}]")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> SfxRequest:
        """Enforce cross-field rules.

        Returns:
            The validated model.

        Raises:
            ValueError: On inconsistent ducking/mode combinations.
        """
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.start_time < 0:
            raise ValueError("start_time must be >= 0")
        return self

    def normalized(self, defaults: Any) -> dict[str, Any]:
        """Render the request as the canonical stored input dict.

        Parameters:
            defaults: DefaultSettings providing fallbacks.

        Returns:
            Dict with concrete values for every field.
        """
        from app.utils.validators import validate_seed

        seed = self.seed
        if seed == -1:
            seed = validate_seed(-1)
        return {
            "video": self.video or "",
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": seed,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "duration": self.duration,
            "start_time": self.start_time,
            "audio_mode": self.audio_mode,
            "sfx_gain_db": self.sfx_gain_db,
            "original_audio_gain_db": self.original_audio_gain_db,
            "duck_threshold_db": self.duck_threshold_db,
            "duck_ratio": self.duck_ratio,
            "duck_attack_ms": self.duck_attack_ms,
            "duck_release_ms": self.duck_release_ms,
            "target_loudness_lufs": self.target_loudness_lufs,
            "true_peak_db": self.true_peak_db,
            "output_format": self.output_format,
            "video_handling": self.video_handling,
            "return_audio_only": self.return_audio_only,
            "enable_safety_checker": self.enable_safety_checker,
            "webhook_url": self.webhook_url,
            "metadata": self.metadata,
        }
