"""Input validation and normalization planning against configured limits."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from app.config import get_settings
from app.constants import MP4_COPY_SAFE_CODECS
from app.errors import ErrorCode, ServiceError
from app.media.probe import MediaInfo

__all__ = ["NormalizationPlan", "validate_video"]


@dataclass(slots=True)
class NormalizationPlan:
    """Decisions derived from validating a source video.

    Attributes:
        info: The source MediaInfo.
        effective_duration_s: Duration after trimming to the request window.
        copy_video: Whether the video stream can be stream-copied into MP4.
        reencode_reason: Human reason when copy is impossible.
        apply_rotation: Whether rotation metadata must be honored on re-encode.
        even_dimensions: Whether width/height are already even.
    """

    info: MediaInfo
    effective_duration_s: float
    copy_video: bool
    reencode_reason: str
    apply_rotation: bool
    even_dimensions: bool


def validate_video(
    info: MediaInfo,
    *,
    duration: float | None = None,
    start_time: float = 0.0,
    video_handling: str = "copy",
) -> NormalizationPlan:
    """Validate a probed source video and derive the normalization plan.

    Parameters:
        info: Probe result.
        duration: Requested generation duration (None = full clip).
        start_time: Requested start offset.
        video_handling: Client preference `copy` or `reencode`.

    Returns:
        NormalizationPlan describing downstream handling.

    Raises:
        ServiceError: no_video_stream, video_too_short, video_too_long,
            corrupt_media, unsupported_media_type, or parameter_out_of_range.
    """
    settings = get_settings()
    if not info.has_video:
        raise ServiceError(ErrorCode.NO_VIDEO_STREAM, "input file contains no video stream")
    if info.width <= 0 or info.height <= 0 or info.duration_s <= 0:
        raise ServiceError(ErrorCode.CORRUPT_MEDIA, "video stream metadata is incomplete")
    if info.video_codec not in settings.media.allowed_video_codecs:
        raise ServiceError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"video codec {info.video_codec!r} is not supported; allowed: "
            f"{settings.media.allowed_video_codecs}",
        )
    pixels = info.width * info.height
    if pixels > settings.media.max_video_pixels:
        raise ServiceError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"video resolution {info.width}x{info.height} exceeds the "
            f"{settings.media.max_video_pixels}-pixel limit",
        )
    if info.duration_s < settings.media.min_video_duration_s:
        raise ServiceError(
            ErrorCode.VIDEO_TOO_SHORT,
            f"video is {info.duration_s:.3f}s; minimum is {settings.media.min_video_duration_s}s",
        )
    if info.duration_s > settings.media.max_video_duration_s + 0.001:
        raise ServiceError(
            ErrorCode.VIDEO_TOO_LONG,
            f"video is {info.duration_s:.3f}s; maximum is {settings.media.max_video_duration_s}s",
        )
    if start_time < 0:
        raise ServiceError(
            ErrorCode.PARAMETER_OUT_OF_RANGE, "start_time must be >= 0"
        )
    if duration is None:
        duration = info.duration_s - start_time
    if duration <= 0:
        raise ServiceError(
            ErrorCode.PARAMETER_OUT_OF_RANGE,
            "duration must be positive and start_time must be < video duration",
        )
    if start_time >= info.duration_s:
        raise ServiceError(
            ErrorCode.PARAMETER_OUT_OF_RANGE,
            f"start_time {start_time}s must be < video duration {info.duration_s:.3f}s",
        )
    effective = min(duration, info.duration_s - start_time)
    if effective < settings.media.min_video_duration_s:
        raise ServiceError(
            ErrorCode.VIDEO_TOO_SHORT,
            f"requested window is {effective:.3f}s; minimum is "
            f"{settings.media.min_video_duration_s}s",
        )
    policy = settings.output.video_reencode_policy
    copy = info.video_codec in MP4_COPY_SAFE_CODECS and policy == "copy_if_possible"
    if video_handling == "reencode":
        copy = False
    reason = ""
    if not copy:
        if info.video_codec not in MP4_COPY_SAFE_CODECS:
            reason = f"codec {info.video_codec} is not MP4-copy-safe"
        elif policy == "always":
            reason = "policy configured to always re-encode"
        else:
            reason = "client requested re-encode"
    even = info.width % 2 == 0 and info.height % 2 == 0
    if not even:
        copy = False
        reason = reason or "odd dimensions require re-encode for yuv420p"
    return NormalizationPlan(
        info=info,
        effective_duration_s=effective,
        copy_video=copy,
        reencode_reason=reason,
        apply_rotation=info.rotation_degrees != 0,
        even_dimensions=even,
    )


def output_frame_rate(info: MediaInfo) -> Fraction:
    """Return the frame rate to use for output timing.

    Parameters:
        info: Source probe.

    Returns:
        avg frame rate when present, else r frame rate, else 25.
    """
    return info.avg_frame_rate or info.r_frame_rate or Fraction(25)
