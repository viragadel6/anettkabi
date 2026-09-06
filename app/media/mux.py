"""MP4 muxing with explicit FFmpeg argument arrays and post-mux verification."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.constants import DURATION_TOLERANCE_MS
from app.errors import ErrorCode, ServiceError
from app.media.ffmpeg import FfmpegError, run_ffmpeg
from app.media.probe import probe_media
from app.utils.hashing import prompt_hash

__all__ = ["MuxResult", "build_mux_args", "mux_video_audio", "verify_output"]


@dataclass(slots=True)
class MuxResult:
    """Outcome of muxing plus verification.

    Attributes:
        path: Final MP4 path.
        duration_s: Output audio/video duration.
        faststart: moov-before-mdat verification result.
        audio_rms: Measured output audio RMS.
        delta_ms: |audio_duration - video_duration| in ms.
        reencoded: Whether the video stream was re-encoded.
    """

    path: Path
    duration_s: float
    faststart: bool
    audio_rms: float
    delta_ms: float
    reencoded: bool


def build_mux_args(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    copy_video: bool,
    audio_bitrate: str,
    audio_sample_rate: int,
    audio_channels: int,
    h264_crf: int,
    h264_preset: str,
    faststart: bool,
    duration_s: float,
    metadata: dict[str, str],
    rotation: int,
) -> list[str]:
    """Build the explicit ffmpeg argument array.

    Parameters:
        video_path: Source video file.
        audio_path: Generated WAV file.
        output_path: Destination MP4.
        copy_video: Stream-copy the video track when True.
        audio_bitrate: AAC bitrate string.
        audio_sample_rate: Output audio rate.
        audio_channels: Output channel count.
        h264_crf: CRF for re-encode.
        h264_preset: Preset for re-encode.
        faststart: Add +faststart.
        duration_s: Exact expected duration (drives -t).
        metadata: MP4 metadata tags.
        rotation: Source rotation to preserve (0/90/180/270).

    Returns:
        The full argument list (without the binary).
    """
    args = [
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if copy_video:
        args += ["-c:v", "copy"]
    else:
        args += [
            "-c:v",
            "libx264",
            "-crf",
            str(h264_crf),
            "-preset",
            h264_preset,
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ]
        if rotation:
            args += ["-metadata:s:v:0", f"rotate={rotation}"]
    args += [
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-ar",
        str(audio_sample_rate),
        "-ac",
        str(audio_channels),
        "-t",
        f"{duration_s:.6f}",
        "-map_metadata",
        "0",
    ]
    for key, value in metadata.items():
        args += ["-metadata", f"{key}={value}"]
    if faststart:
        args += ["-movflags", "+faststart"]
    args.append(str(output_path))
    return args


async def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    copy_video: bool,
    duration_s: float,
    metadata: dict[str, str],
    rotation: int = 0,
) -> MuxResult:
    """Mux video + generated audio into a verified faststart MP4.

    On verification failure the mux is retried once with re-encoding, then the
    call fails with `mux_failed`.

    Parameters:
        video_path: Source video.
        audio_path: Generated WAV (48 kHz stereo recommended).
        output_path: Destination MP4 path.
        copy_video: Attempt stream copy first.
        duration_s: Expected output duration.
        metadata: Tag map.
        rotation: Source rotation degrees.

    Returns:
        MuxResult describing the verified output.

    Raises:
        ServiceError: mux_failed when both attempts fail verification.
    """
    settings = get_settings()
    attempt_specs: list[tuple[bool, str]] = [
        (copy_video, "copy" if copy_video else "reencode"),
        (False, "reencode-fallback"),
    ]
    last_error = ""
    for index, (use_copy, label) in enumerate(attempt_specs):
        args = build_mux_args(
            video_path,
            audio_path,
            output_path,
            copy_video=use_copy,
            audio_bitrate=settings.output.output_audio_bitrate,
            audio_sample_rate=settings.output.output_audio_sr,
            audio_channels=settings.output.output_audio_channels,
            h264_crf=settings.output.h264_crf,
            h264_preset=settings.output.h264_preset,
            faststart=settings.output.faststart,
            duration_s=duration_s,
            metadata=metadata,
            rotation=rotation,
        )
        try:
            await run_ffmpeg(args, timeout_s=max(120.0, duration_s * 8), operation=f"mux-{label}")
        except FfmpegError as exc:
            last_error = exc.stderr_tail
            if index == len(attempt_specs) - 1:
                raise ServiceError(
                    ErrorCode.MUX_FAILED,
                    f"mux failed on all attempts: {exc.message}",
                ) from exc
            continue
        try:
            result = await verify_output(output_path, duration_s=duration_s)
        except ServiceError as exc:
            last_error = exc.message
            if index == len(attempt_specs) - 1:
                raise
            continue
        result.reencoded = not use_copy
        return result
    raise ServiceError(ErrorCode.MUX_FAILED, f"mux failed: {last_error}")


async def verify_output(path: Path, *, duration_s: float | None = None) -> MuxResult:
    """Probe and verify structural guarantees of the output MP4.

    Parameters:
        path: Output file.
        duration_s: Expected duration when known.

    Returns:
        MuxResult with verification numbers.

    Raises:
        ServiceError: mux_failed for structural violations.
    """
    info = await probe_media(path)
    if not info.has_video or not info.has_audio:
        raise ServiceError(
            ErrorCode.MUX_FAILED,
            f"output must contain one video and one audio stream (video={info.has_video}, audio={info.has_audio})",
        )
    if not info.faststart:
        raise ServiceError(ErrorCode.MUX_FAILED, "moov atom is not at the front (faststart)")
    audio_duration = info.duration_s
    delta_ms = abs(audio_duration - (duration_s or audio_duration)) * 1000.0
    if duration_s is not None and delta_ms > DURATION_TOLERANCE_MS:
        raise ServiceError(
            ErrorCode.MUX_FAILED,
            f"output duration {audio_duration:.3f}s deviates from expected "
            f"{duration_s:.3f}s by {delta_ms:.1f}ms (tolerance {DURATION_TOLERANCE_MS}ms)",
        )
    audio_rms_value = await measure_audio_rms(path)
    return MuxResult(
        path=path,
        duration_s=audio_duration,
        faststart=info.faststart,
        audio_rms=audio_rms_value,
        delta_ms=delta_ms,
        reencoded=False,
    )


async def measure_audio_rms(path: Path) -> float:
    """Extract the output audio and compute its RMS level.

    Parameters:
        path: MP4 file.

    Returns:
        RMS in linear scale (0.0 on extraction failure).

    Raises:
        ServiceError: mux_failed when audio extraction fails.
    """
    import numpy as np
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)
    try:
        await run_ffmpeg(
            [
                "-y",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-f",
                "wav",
                str(wav_path),
            ],
            timeout_s=120.0,
            operation="rms-extract",
        )
        data, _rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        if data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    except (FfmpegError, RuntimeError, OSError) as exc:
        if isinstance(exc, FfmpegError):
            raise ServiceError(ErrorCode.MUX_FAILED, f"audio verification failed: {exc.message}") from exc
        return 0.0
    finally:
        wav_path.unlink(missing_ok=True)


def output_metadata_tags(
    *, model_variant: str, prompt: str, seed: int, prediction_id: str
) -> dict[str, str]:
    """Build MP4 metadata tags for provenance.

    Parameters:
        model_variant: Registry variant name.
        prompt: Generation prompt (only its hash is embedded).
        seed: Seed used.
        prediction_id: Prediction id.

    Returns:
        Metadata dict for ffmpeg -metadata args.
    """
    from app.utils.time import utc_now_isoformat

    return {
        "encoder": "video-to-video-sfx",
        "model_variant": model_variant,
        "prompt_sha256": prompt_hash(prompt),
        "seed": str(seed),
        "prediction_id": prediction_id,
        "creation_time": utc_now_isoformat(),
    }
