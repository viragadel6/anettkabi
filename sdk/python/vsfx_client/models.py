"""Request/response models for the vsfx client SDK."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "AccountUsage",
    "ModelInfo",
    "Prediction",
    "SfxParams",
    "UploadAck",
    "UploadPresign",
    "WebhookTestResult",
]

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "canceled"})
ACTIVE_STATUSES = frozenset({"starting", "queued", "processing"})


@dataclass(slots=True)
class SfxParams:
    """Tunable generation parameters.

    Attributes:
        prompt: Positive sound-design prompt.
        negative_prompt: Negative prompt.
        seed: Deterministic seed (0..2^32-2).
        num_inference_steps: Flow integration steps.
        guidance_scale: Classifier-free guidance scale (0 disables).
        duration: Output audio length cap in seconds.
        start_time: In-video generation start offset.
        audio_mode: replace | mix | duck.
        sfx_gain_db: SFX channel gain.
        original_audio_gain_db: Original audio gain.
        duck_threshold_db: Ducking threshold.
        duck_ratio: Ducking ratio.
        duck_attack_ms: Ducking attack.
        duck_release_ms: Ducking release.
        target_loudness_lufs: Output loudness target.
        true_peak_db: Output true-peak ceiling.
        video_handling: copy | reencode.
        return_audio_only: Emit WAV-only output.
        enable_safety_checker: Prompt moderation toggle.
        webhook_url: Completion webhook (https).
        metadata: Opaque passthrough map.
    """

    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    duration: float | None = None
    start_time: float | None = None
    audio_mode: str | None = None
    sfx_gain_db: float | None = None
    original_audio_gain_db: float | None = None
    duck_threshold_db: float | None = None
    duck_ratio: float | None = None
    duck_attack_ms: float | None = None
    duck_release_ms: float | None = None
    target_loudness_lufs: float | None = None
    true_peak_db: float | None = None
    video_handling: str | None = None
    return_audio_only: bool | None = None
    enable_safety_checker: bool | None = None
    webhook_url: str | None = None
    metadata: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        """Render non-None fields as a JSON body.

        Returns:
            Dict suitable as the create-prediction JSON body.
        """
        payload = {
            key: value
            for key, value in asdict(self).items()
            if value is not None and key != "metadata"
        }
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class Prediction:
    """A prediction resource.

    Attributes:
        id: ULID-style identifier.
        status: starting | queued | processing | succeeded | failed | canceled.
        prompt: Echo of the prompt (hashed server-side in logs only).
        created_at: ISO-8601 creation timestamp.
        started_at / completed_at: Lifecycle timestamps when reached.
        output: Output file URLs on success.
        error: Error payload on failure.
        metrics: Server-side timing metrics.
        urls: Polling URL.
        raw: Full raw payload (envelope data).
    """

    id: str
    status: str
    prompt: str = ""
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    output: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    urls: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Whether the prediction reached a final state.

        Returns:
            True for succeeded/failed/canceled.
        """
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        """Whether the prediction is still progressing.

        Returns:
            True for starting/queued/processing.
        """
        return self.status in ACTIVE_STATUSES

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Prediction:
        """Build a Prediction from the API envelope data.

        Parameters:
            payload: Envelope `data` object.

        Returns:
            The parsed Prediction.
        """
        return cls(
            id=str(payload.get("id", "")),
            status=str(payload.get("status", "")),
            prompt=str(payload.get("prompt", "")),
            created_at=str(payload.get("created_at", "")),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            output=payload.get("output"),
            error=payload.get("error"),
            metrics=dict(payload.get("metrics") or {}),
            urls=dict(payload.get("urls") or {}),
            raw=payload,
        )


@dataclass(slots=True)
class UploadAck:
    """Result of a direct multipart upload.

    Attributes:
        url: Retrievable URL to pass as `video`.
        path: Object storage path.
        expires_at: ISO-8601 expiry.
    """

    url: str
    path: str
    expires_at: str = ""


@dataclass(slots=True)
class UploadPresign:
    """Result of a presign request.

    Attributes:
        upload_url: Presigned PUT target.
        url: Retrievable URL to pass as `video`.
        path: Object storage path.
        expires_at: ISO-8601 expiry.
    """

    upload_url: str
    url: str
    path: str
    expires_at: str = ""


@dataclass(slots=True)
class ModelInfo:
    """Model card payload."""

    raw: dict[str, Any]

    @property
    def id(self) -> str:
        """Model identifier.

        Returns:
            The id string.
        """
        return str(self.raw.get("id", "video-to-video-sfx"))


@dataclass(slots=True)
class AccountUsage:
    """Usage snapshot for the caller's key."""

    raw: dict[str, Any]

    @property
    def seconds_generated(self) -> float:
        """Audio seconds generated in the current window.

        Returns:
            Seconds total.
        """
        return float(self.raw.get("seconds_generated", 0.0))

    @property
    def predictions(self) -> int:
        """Prediction count in the current window.

        Returns:
            Count total.
        """
        return int(self.raw.get("predictions", 0))


@dataclass(slots=True)
class WebhookTestResult:
    """Outcome of a signed webhook probe."""

    accepted: bool
    status_code: int
    raw: dict[str, Any]
