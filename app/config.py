"""Typed, validated, fail-fast application configuration (Pydantic v2 settings)."""

from __future__ import annotations

import ipaddress
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigError

__all__ = [
    "AudioMode",
    "Settings",
    "get_settings",
    "reset_settings",
]

AudioMode = Literal["replace", "mix", "duck"]
DeviceSetting = Literal["auto", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"]
DtypeSetting = Literal["bf16", "fp16", "fp32"]
VariantName = Literal["small_16k", "medium_44k", "large_44k"]


class _Section(BaseSettings):
    """Base class for nested configuration sections."""

    model_config = SettingsConfigDict(extra="forbid", case_sensitive=False)


class ServerSettings(_Section):
    """HTTP server binding and request limits.

    Attributes:
        host: Bind address for Uvicorn.
        port: Bind port.
        workers: Number of Uvicorn workers.
        public_base_url: External base URL used when building `urls.get`.
        root_path: ASGI root path when mounted behind a proxy.
        cors_origins: Allowed CORS origins (list); empty disables CORS.
        request_timeout_s: Per-request upper bound enforced by middleware.
        max_upload_bytes: Largest accepted multipart body.
        trusted_proxy_ips: IPs whose X-Forwarded-For header is trusted.
    """

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=64)
    public_base_url: str = "http://localhost:8000"
    root_path: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    request_timeout_s: float = Field(default=300.0, gt=0)
    max_upload_bytes: int = Field(default=536870912, ge=1024)
    trusted_proxy_ips: list[str] = Field(default_factory=list)

    @field_validator("public_base_url")
    @classmethod
    def _validate_public_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("public_base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("trusted_proxy_ips")
    @classmethod
    def _validate_proxy_ips(cls, value: list[str]) -> list[str]:
        for entry in value:
            ipaddress.ip_address(entry)
        return value


class DatabaseSettings(_Section):
    """PostgreSQL connection pool settings.

    Attributes:
        database_url: SQLAlchemy async URL (postgresql+asyncpg://...).
        db_pool_size: Core pool connections per process.
        db_max_overflow: Extra connections allowed under load.
        db_statement_timeout_ms: Postgres statement_timeout applied per session.
    """

    database_url: str
    db_pool_size: int = Field(default=10, ge=1, le=200)
    db_max_overflow: int = Field(default=20, ge=0, le=200)
    db_statement_timeout_ms: int = Field(default=15000, ge=100)

    @field_validator("database_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "database_url must use the postgresql+asyncpg:// driver, "
                f"got: {value[:40]}..."
            )
        return value


class RedisSettings(_Section):
    """Redis connection and queue topology.

    Attributes:
        redis_url: Redis connection URL.
        queue_stream_name: Redis Stream used as the job queue.
        queue_group: Consumer group name.
        queue_max_len: Approximate maxlen trimming for the stream.
        queue_claim_idle_ms: Idle time after which messages are autoclaimed.
    """

    redis_url: str = "redis://localhost:6379/0"
    queue_stream_name: str = "vsfx:jobs"
    queue_group: str = "vsfx-workers"
    queue_max_len: int = Field(default=100000, ge=100)
    queue_claim_idle_ms: int = Field(default=120000, ge=5000)


class StorageSettings(_Section):
    """S3-compatible object storage settings.

    Attributes:
        s3_endpoint_url: Custom endpoint (MinIO) or empty for AWS.
        s3_region: Bucket region.
        s3_bucket: Bucket name.
        s3_access_key_id / s3_secret_access_key: Credentials.
        s3_force_path_style: Use path-style addressing (MinIO).
        s3_presign_ttl_s: Presigned GET lifetime.
        cdn_base_url: When set, result URLs are CDN-based instead of presigned.
        s3_key_prefix: Prefix for all object keys.
        result_retention_days: Days until output assets are swept.
    """

    s3_endpoint_url: str = ""
    s3_region: str = "us-east-1"
    s3_bucket: str
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_force_path_style: bool = True
    s3_presign_ttl_s: int = Field(default=86400, ge=60)
    cdn_base_url: str = ""
    s3_key_prefix: str = "vsfx"
    result_retention_days: int = Field(default=30, ge=1)

    @field_validator("s3_key_prefix")
    @classmethod
    def _validate_prefix(cls, value: str) -> str:
        stripped = value.strip("/")
        if stripped and not stripped.replace("/", "").replace("-", "").replace("_", "").isalnum():
            raise ValueError("s3_key_prefix may contain only [A-Za-z0-9/_-]")
        return stripped

    @field_validator("cdn_base_url")
    @classmethod
    def _validate_cdn(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("cdn_base_url must start with http:// or https://")
        return value.rstrip("/")


class MediaSettings(_Section):
    """Input media limits and download safety.

    Attributes:
        max_video_duration_s: Longest accepted source video.
        min_video_duration_s: Shortest accepted source video.
        max_video_pixels: Width*height ceiling.
        allowed_video_mime: Accepted Content-Type values for uploads.
        allowed_video_ext: Accepted file extensions.
        allowed_video_codecs: Accepted video stream codecs after probing.
        download_max_bytes: Hard byte ceiling for URL downloads.
        download_timeout_s: Total download timeout.
        download_allowed_schemes: URL schemes permitted for `video`.
        download_block_private_cidrs: Reject private/loopback/link-local targets (SSRF).
    """

    max_video_duration_s: float = Field(default=300.0, gt=0)
    min_video_duration_s: float = Field(default=0.5, ge=0)
    max_video_pixels: int = Field(default=8294400, ge=1)
    allowed_video_mime: list[str] = Field(
        default_factory=lambda: [
            "video/mp4",
            "video/quicktime",
            "video/webm",
            "video/x-matroska",
            "video/x-msvideo",
            "video/mpeg",
            "video/3gpp",
        ]
    )
    allowed_video_ext: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".3gp"]
    )
    allowed_video_codecs: list[str] = Field(
        default_factory=lambda: [
            "h264",
            "hevc",
            "av1",
            "vp9",
            "vp8",
            "mpeg4",
            "mpeg2video",
            "mpeg1video",
            "h263",
        ]
    )
    download_max_bytes: int = Field(default=536870912, ge=1024)
    download_timeout_s: float = Field(default=120.0, gt=0)
    download_allowed_schemes: list[str] = Field(default_factory=lambda: ["http", "https"])
    download_block_private_cidrs: bool = True


class ModelSettings(_Section):
    """Generative model, device, and inference-window settings.

    Attributes:
        model_id: Public model identifier.
        model_variant: Registry variant selecting architecture and weights.
        weights_dir: Directory holding weight artifacts.
        weights_manifest_path: JSON manifest with per-file URL + SHA-256.
        device: Device selection string.
        dtype: Compute dtype for inference.
        torch_compile: Wrap the generator with torch.compile.
        attention_backend: Attention kernel selection.
        model_window_s: Native generation window length.
        window_overlap_s: Overlap between consecutive windows.
        visual_fps: CLIP semantic frame rate.
        sync_fps: Sync encoder frame rate.
        audio_sample_rate: Model audio sample rate (must match variant).
        latent_hop: Mel-frame downsample factor of the VAE.
        max_concurrent_inference: Inference calls per process.
        inference_timeout_s: Upper bound for one full generation.
        warmup_on_start: Run a warmup generation at worker start.
    """

    model_id: str = "video-to-video-sfx"
    model_variant: VariantName = "small_16k"
    weights_dir: Path = Path("./weights")
    weights_manifest_path: Path = Path("./weights/manifest.json")
    device: DeviceSetting = "auto"
    dtype: DtypeSetting = "fp32"
    torch_compile: bool = False
    attention_backend: Literal["sdpa", "flash", "math"] = "sdpa"
    model_window_s: float = Field(default=10.0, gt=0.5)
    window_overlap_s: float = Field(default=1.0, ge=0)
    visual_fps: float = Field(default=8.0, gt=0.5)
    sync_fps: float = Field(default=25.0, gt=1)
    audio_sample_rate: int = Field(default=16000, gt=0)
    latent_hop: int = Field(default=4, ge=1)
    max_concurrent_inference: int = Field(default=1, ge=1, le=64)
    inference_timeout_s: float = Field(default=900.0, gt=1)
    warmup_on_start: bool = True

    @model_validator(mode="after")
    def _validate_window(self) -> ModelSettings:
        if self.window_overlap_s >= self.model_window_s / 2:
            raise ValueError("window_overlap_s must be < model_window_s / 2")
        if self.model_window_s * self.visual_fps < 8:
            raise ValueError("model_window_s too short for the visual stream")
        return self


class DefaultSettings(_Section):
    """Generation defaults applied when the request omits fields.

    Attributes:
        default_steps: Default sampler step count.
        default_cfg: Default classifier-free guidance scale.
        default_audio_mode: Default combination mode with original audio.
        default_target_lufs: Default loudness target; null disables normalization.
        default_true_peak_db: Default true-peak ceiling.
        default_sfx_gain_db: Default SFX bus gain.
        default_original_gain_db: Default original-audio bus gain.
    """

    default_steps: int = Field(default=25, ge=1, le=100)
    default_cfg: float = Field(default=4.5, ge=0.0, le=15.0)
    default_audio_mode: AudioMode = "replace"
    default_target_lufs: float | None = Field(default=-14.0, ge=-40, le=0)
    default_true_peak_db: float = Field(default=-1.0, ge=-12, le=0)
    default_sfx_gain_db: float = Field(default=0.0, ge=-30, le=12)
    default_original_gain_db: float = Field(default=-6.0, ge=-60, le=12)


class OutputSettings(_Section):
    """Output MP4/AAC encoding settings.

    Attributes:
        output_audio_codec: Audio codec written into MP4.
        output_audio_bitrate: AAC bitrate string.
        output_audio_sr: Output audio sample rate.
        output_audio_channels: Output audio channel count.
        video_reencode_policy: Whether the video stream is copied when possible.
        h264_crf: CRF used when re-encoding.
        h264_preset: x264 preset used when re-encoding.
        faststart: Move moov atom to the front.
    """

    output_audio_codec: Literal["aac"] = "aac"
    output_audio_bitrate: str = "192k"
    output_audio_sr: int = Field(default=48000, gt=0)
    output_audio_channels: int = Field(default=2, ge=1, le=2)
    video_reencode_policy: Literal["copy_if_possible", "always"] = "copy_if_possible"
    h264_crf: int = Field(default=18, ge=0, le=51)
    h264_preset: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"
    ] = "medium"
    faststart: bool = True

    @field_validator("output_audio_bitrate")
    @classmethod
    def _validate_bitrate(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized.endswith(("k", "m")) or not normalized[:-1].isdigit():
            raise ValueError("output_audio_bitrate must look like '192k' or '1m'")
        return normalized


class JobSettings(_Section):
    """Worker job orchestration settings.

    Attributes:
        job_max_attempts: Delivery attempts before dead-lettering.
        job_backoff_base_s: Exponential retry backoff base.
        job_backoff_max_s: Retry backoff ceiling.
        job_total_timeout_s: Wall-clock budget for one job attempt.
        job_heartbeat_interval_s: Heartbeat write period.
        job_stale_after_s: Heartbeat age marking a job stale.
        work_dir: Scratch directory for downloads and intermediates.
        keep_intermediate: Preserve intermediates for debugging.
    """

    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_backoff_base_s: float = Field(default=5.0, ge=0.1)
    job_backoff_max_s: float = Field(default=120.0, ge=1)
    job_total_timeout_s: float = Field(default=1800.0, ge=30)
    job_heartbeat_interval_s: float = Field(default=10.0, ge=1)
    job_stale_after_s: float = Field(default=180.0, ge=30)
    work_dir: Path = Path("/tmp/vsfx-work")
    keep_intermediate: bool = False


class AuthSettings(_Section):
    """Authentication, rate-limit, and quota settings.

    Attributes:
        auth_required: When false, requests run without an API key (dev only).
        api_key_header: Header carrying the Bearer credential.
        rate_limit_rpm: Requests per minute per key.
        rate_limit_burst: Token bucket burst.
        concurrent_jobs_per_key: Concurrent non-terminal predictions per key.
        monthly_seconds_quota: Billed-seconds allowance per calendar month.
    """

    auth_required: bool = True
    api_key_header: str = "Authorization"
    rate_limit_rpm: int = Field(default=60, ge=1)
    rate_limit_burst: int = Field(default=30, ge=1)
    concurrent_jobs_per_key: int = Field(default=3, ge=1)
    monthly_seconds_quota: float = Field(default=36000.0, gt=0)


class SafetySettings(_Section):
    """Prompt moderation settings.

    Attributes:
        safety_enabled: Toggle moderation entirely.
        safety_blocklist_path: Path to the blocklist term file.
        safety_mode: `block` rejects flagged prompts; `flag` proceeds and marks output.
    """

    safety_enabled: bool = True
    safety_blocklist_path: Path = Path("./config/safety_blocklist.txt")
    safety_mode: Literal["flag", "block"] = "block"


class WebhookSettings(_Section):
    """Outbound webhook delivery settings.

    Attributes:
        webhook_enabled: Allow prediction webhooks.
        webhook_timeout_s: Per-delivery HTTP timeout.
        webhook_max_attempts: Delivery attempts.
        webhook_signing_secret: HMAC-SHA256 secret; empty disables signing.
    """

    webhook_enabled: bool = True
    webhook_timeout_s: float = Field(default=15.0, gt=0)
    webhook_max_attempts: int = Field(default=5, ge=1, le=20)
    webhook_signing_secret: str = ""


class TelemetrySettings(_Section):
    """Logging and metrics settings.

    Attributes:
        log_level: Root level.
        log_format: `json` or `console`.
        metrics_enabled: Expose /metrics.
        metrics_path: Prometheus endpoint path.
        sentry_dsn: Optional; empty means tracing is a structured no-op.
    """

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    sentry_dsn: str = ""

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper


class Settings(BaseSettings):
    """Root application settings assembled from environment variables.

    Attributes:
        server, database, redis, storage, media, model, defaults, output,
        jobs, auth, safety, webhook, telemetry: configuration sections.
    """

    model_config = SettingsConfigDict(
        env_prefix="VSFX_",
        env_nested_delimiter="__",
        extra="forbid",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    server: ServerSettings
    database: DatabaseSettings
    redis: RedisSettings
    storage: StorageSettings
    media: MediaSettings
    model: ModelSettings
    defaults: DefaultSettings
    output: OutputSettings
    jobs: JobSettings
    auth: AuthSettings
    safety: SafetySettings
    webhook: WebhookSettings
    telemetry: TelemetrySettings

    @model_validator(mode="after")
    def _validate_runtime(self) -> Settings:
        errors: list[str] = []
        for tool in ("ffmpeg", "ffprobe"):
            if shutil.which(tool) is None:
                errors.append(f"required binary '{tool}' not found on PATH")
        try:
            variant_sr = {"small_16k": 16000, "medium_44k": 44100, "large_44k": 44100}[
                self.model.model_variant
            ]
            if self.model.audio_sample_rate != variant_sr:
                errors.append(
                    f"model.audio_sample_rate ({self.model.audio_sample_rate}) does not match "
                    f"variant {self.model.model_variant} ({variant_sr})"
                )
        except KeyError:
            errors.append(f"unknown model variant {self.model.model_variant}")
        if errors:
            raise ValueError(
                "startup validation failed: "
                + "; ".join(errors)
                + ". Install FFmpeg or fix the VSFX_MODEL__ settings."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings instance.

    Returns:
        The cached Settings, constructed on first call.

    Raises:
        ConfigError: If any required setting is missing or invalid.
    """
    try:
        return Settings()
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc


def reset_settings() -> None:
    """Clear the cached Settings so the next `get_settings()` re-reads the env."""
    get_settings.cache_clear()
