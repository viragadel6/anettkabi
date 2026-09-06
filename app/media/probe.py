"""Typed ffprobe parsing into a MediaInfo dataclass with exact rational rates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from app.media.ffmpeg import ffprobe_frames, run_ffprobe

__all__ = ["MediaInfo", "parse_fraction", "parse_probe_payload", "probe_media"]


def parse_fraction(value: str | None) -> Fraction:
    """Parse an ffprobe rational string like `30000/1001`.

    Parameters:
        value: Rational string, plain number, or None.

    Returns:
        The Fraction (0 when unparseable).
    """
    if not value or value in ("0/0", "N/A"):
        return Fraction(0)
    try:
        if "/" in value:
            num, _, den = value.partition("/")
            return Fraction(int(num), int(den or 1))
        return Fraction(float(value)).limit_denominator(1000000)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


@dataclass(slots=True)
class MediaInfo:
    """Fully typed media probe result.

    Attributes:
        container: Format name (e.g. mov,mp4,mpegts).
        duration_s: Best duration estimate in seconds.
        duration_exact: Whether duration came from format/stream (vs frame counting).
        nb_frames: Reported frame count (0 when absent).
        avg_frame_rate: Average frame rate as exact Fraction.
        r_frame_rate: Base/nominal frame rate as exact Fraction.
        width: Video width.
        height: Video height.
        sar: Sample aspect ratio.
        dar: Display aspect ratio.
        rotation_degrees: Rotation from side data (normalized to 0/90/180/270).
        pixel_format: Pixel format name.
        bit_depth: Bits per component (derived from pix_fmt when known).
        video_codec: Video codec name.
        audio_codec: Audio codec name (empty when absent).
        audio_sample_rate: Audio sample rate (0 when absent).
        audio_channels: Audio channel count.
        audio_layout: Channel layout string.
        start_time_s: Container start_time offset.
        video_start_time_s: Video stream start_time.
        audio_start_time_s: Audio stream start_time.
        fragmented: Whether the MP4 is fragmented (moof present).
        faststart: Whether moov precedes mdat (best-effort).
        has_video: Video stream presence.
        has_audio: Audio stream presence.
        bit_rate: Container bitrate when reported.
        raw: The raw ffprobe JSON payload.
    """

    container: str = ""
    duration_s: float = 0.0
    duration_exact: bool = False
    nb_frames: int = 0
    avg_frame_rate: Fraction = Fraction(0)
    r_frame_rate: Fraction = Fraction(0)
    width: int = 0
    height: int = 0
    sar: Fraction = Fraction(1)
    dar: Fraction = Fraction(1)
    rotation_degrees: int = 0
    pixel_format: str = ""
    bit_depth: int = 8
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int = 0
    audio_channels: int = 0
    audio_layout: str = ""
    start_time_s: float = 0.0
    video_start_time_s: float = 0.0
    audio_start_time_s: float = 0.0
    fragmented: bool = False
    faststart: bool = False
    has_video: bool = False
    has_audio: bool = False
    bit_rate: int = 0
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        """Return the effective frame rate as float."""
        rate = self.avg_frame_rate or self.r_frame_rate
        return float(rate) if rate else 0.0

    @property
    def vfr(self) -> bool:
        """Return True when avg and r frame rates disagree materially."""
        if not self.avg_frame_rate or not self.r_frame_rate:
            return False
        return abs(float(self.avg_frame_rate) - float(self.r_frame_rate)) > 0.5

    def display_dimensions(self) -> tuple[int, int]:
        """Return post-rotation display dimensions.

        Returns:
            (width, height) after applying rotation metadata.
        """
        if self.rotation_degrees in (90, 270):
            return self.height, self.width
        return self.width, self.height

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable summary for API responses.

        Returns:
            Dict of key probe fields.
        """
        return {
            "container": self.container,
            "duration_s": round(self.duration_s, 3),
            "fps": round(self.fps, 3),
            "vfr": self.vfr,
            "width": self.width,
            "height": self.height,
            "rotation_degrees": self.rotation_degrees,
            "pixel_format": self.pixel_format,
            "video_codec": self.video_codec,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
            "audio_layout": self.audio_layout,
            "display_width": self.display_dimensions()[0],
            "display_height": self.display_dimensions()[1],
        }


_BIT_DEPTH_BY_FORMAT = {
    "yuv420p10le": 10,
    "yuv422p10le": 10,
    "yuv444p10le": 10,
    "p010le": 10,
    "gray10le": 10,
    "yuv420p12le": 12,
    "yuv444p12le": 12,
    "gbrp10le": 10,
    "gbrp12le": 12,
}


def _rotation_from_stream(stream: dict[str, object]) -> int:
    """Extract normalized rotation degrees from side data or tags.

    Parameters:
        stream: One ffprobe stream dict.

    Returns:
        Rotation in {0, 90, 180, 270}.
    """
    candidates: list[float] = []
    side_data = stream.get("side_data_list") or []
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict) and "rotation" in entry:
                value = entry["rotation"]
                if isinstance(value, (int, float)):
                    candidates.append(float(value))
    tags = stream.get("tags")
    if isinstance(tags, dict):
        for key in ("rotate", "rotation"):
            value = tags.get(key)
            if value is not None:
                try:
                    candidates.append(float(value))
                except (TypeError, ValueError):
                    continue
    for value in candidates:
        normalized = round(-value) % 360
        if normalized:
            return normalized
    return 0


def _to_float(value: object) -> float:
    """Coerce an ffprobe scalar to float (0.0 when unparseable).

    Parameters:
        value: Raw value.

    Returns:
        Parsed float or 0.0.
    """
    if value is None or value == "N/A":
        return 0.0
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: object) -> int:
    """Coerce an ffprobe scalar to int (0 when unparseable).

    Parameters:
        value: Raw value.

    Returns:
        Parsed int or 0.
    """
    if value is None or value == "N/A":
        return 0
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def parse_probe_payload(payload: dict[str, object]) -> MediaInfo:
    """Convert a parsed ffprobe JSON payload into MediaInfo.

    Parameters:
        payload: Parsed JSON from `ffprobe -show_streams -show_format`.

    Returns:
        The populated MediaInfo.
    """
    info = MediaInfo()
    info.raw = payload
    fmt = payload.get("format") or {}
    if isinstance(fmt, dict):
        info.container = str(fmt.get("format_name") or "")
        info.duration_s = _to_float(fmt.get("duration"))
        info.duration_exact = info.duration_s > 0
        info.start_time_s = _to_float(fmt.get("start_time"))
        info.bit_rate = _to_int(fmt.get("bit_rate"))
    streams = payload.get("streams") or []
    video: dict[str, object] | None = None
    audio: dict[str, object] | None = None
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        codec_type = str(stream.get("codec_type") or "")
        if codec_type == "video" and video is None:
            video = stream
        elif codec_type == "audio" and audio is None:
            audio = stream
    if video is not None:
        info.has_video = True
        info.video_codec = str(video.get("codec_name") or "")
        info.width = _to_int(video.get("width"))
        info.height = _to_int(video.get("height"))
        info.avg_frame_rate = parse_fraction(str(video.get("avg_frame_rate") or ""))
        info.r_frame_rate = parse_fraction(str(video.get("r_frame_rate") or ""))
        info.nb_frames = _to_int(video.get("nb_frames"))
        info.sar = parse_fraction(str(video.get("sample_aspect_ratio") or "")) or Fraction(1)
        info.dar = parse_fraction(str(video.get("display_aspect_ratio") or "")) or Fraction(1)
        info.pixel_format = str(video.get("pix_fmt") or "")
        info.bit_depth = _BIT_DEPTH_BY_FORMAT.get(info.pixel_format, 8)
        info.rotation_degrees = _rotation_from_stream(video)
        info.video_start_time_s = _to_float(video.get("start_time"))
        if not info.duration_exact:
            stream_duration = _to_float(video.get("duration"))
            if stream_duration > 0:
                info.duration_s = stream_duration
                info.duration_exact = True
            elif info.r_frame_rate and info.nb_frames:
                info.duration_s = float(Fraction(info.nb_frames, 1) / info.r_frame_rate)
                info.duration_exact = False
    if audio is not None:
        info.has_audio = True
        info.audio_codec = str(audio.get("codec_name") or "")
        info.audio_sample_rate = _to_int(audio.get("sample_rate"))
        info.audio_channels = _to_int(audio.get("channels"))
        info.audio_layout = str(audio.get("channel_layout") or "")
        info.audio_start_time_s = _to_float(audio.get("start_time"))
    return info


async def probe_media(path: Path, *, count_frames_fallback: bool = True) -> MediaInfo:
    """Probe a media file into MediaInfo.

    Parameters:
        path: Media file path.
        count_frames_fallback: When duration is missing, decode-count frames.

    Returns:
        The populated MediaInfo.

    Raises:
        FfmpegError: When ffprobe fails (maps to corrupt_media).
    """
    stdout = await run_ffprobe(
        [
            "-show_streams",
            "-show_format",
            "-show_chapters",
            "-show_entries",
            "format_tags=major_brand:stream_tags=rotate,rotation",
            str(path),
        ]
    )
    payload = json.loads(stdout or "{}")
    info = parse_probe_payload(payload)
    if not info.duration_s and count_frames_fallback and info.has_video:
        frames = await ffprobe_frames(path)
        if frames and info.r_frame_rate:
            info.nb_frames = frames
            info.duration_s = float(Fraction(frames, 1) / info.r_frame_rate)
            info.duration_exact = False
    info.fragmented = _detect_fragmented(path)
    info.faststart = info.faststart or _moov_before_mdat(path)
    return info


def _detect_fragmented(path: Path) -> bool:
    """Check for fragmented MP4 structure (moof atoms).

    Parameters:
        path: File to scan.

    Returns:
        True when a moof box is found in the first 8 MiB.
    """
    try:
        with path.open("rb") as handle:
            window = handle.read(8 * 1024 * 1024)
    except OSError:
        return False
    return b"moof" in window


def _moov_before_mdat(path: Path) -> bool:
    """Check whether moov precedes mdat (faststart heuristic).

    Parameters:
        path: File to scan.

    Returns:
        True when moov appears before mdat or no mdat exists.
    """
    try:
        with path.open("rb") as handle:
            window = handle.read(32 * 1024 * 1024)
    except OSError:
        return False
    moov = window.find(b"moov")
    mdat = window.find(b"mdat")
    if mdat == -1:
        return True
    return moov != -1 and moov < mdat
