"""Deterministic PyAV frame sampling: semantic + sync streams with exact timestamps."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import av
import numpy as np
import torch
from torch.nn.functional import interpolate

from app.constants import (
    CLIP_SHORTER_SIDE,
    CROP_SIZE,
    SYNC_SEGMENT_FRAMES,
    SYNC_SHORTER_SIDE,
)

__all__ = ["FrameStreams", "decode_frames", "extract_frame_streams", "resize_and_crop"]


@dataclass(slots=True)
class FrameStreams:
    """Sampled frame tensors for both conditioning streams.

    Attributes:
        visual: uint8 tensor [T_v, 3, CROP, CROP] (CLIP preprocessing pending).
        visual_timestamps: Exact media-time seconds per visual frame.
        sync: uint8 tensor [T_s, 3, CROP, CROP].
        sync_timestamps: Exact media-time seconds per sync frame.
        edge_clamps: Number of sampled frames edge-clamped past the last decoded frame.
        start_s: Requested window start.
        duration_s: Requested window duration.
        source_fps: Source average fps reported by the container.
    """

    visual: torch.Tensor
    visual_timestamps: torch.Tensor
    sync: torch.Tensor
    sync_timestamps: torch.Tensor
    edge_clamps: int
    start_s: float
    duration_s: float
    source_fps: float
    segment_groups: list[list[int]] = field(default_factory=list)

    def sync_segment_count(self) -> int:
        """Return the number of full 16-frame/stride-8 segments available."""
        count = len(self.sync_timestamps)
        if count < SYNC_SEGMENT_FRAMES:
            return 0
        return 1 + (count - SYNC_SEGMENT_FRAMES) // 8


def resize_and_crop(frame_rgb: np.ndarray, shorter_side: int, crop: int) -> np.ndarray:
    """Resize the shorter side and center-crop to `crop` (bicubic, uint8-safe).

    Parameters:
        frame_rgb: HxWx3 uint8 array.
        shorter_side: Target shorter-side length.
        crop: Final square crop size.

    Returns:
        crop x crop x 3 uint8 array.
    """
    height, width = frame_rgb.shape[:2]
    if height < width:
        new_h = shorter_side
        new_w = max(1, round(width * shorter_side / height))
    else:
        new_w = shorter_side
        new_h = max(1, round(height * shorter_side / width))
    tensor = torch.from_numpy(frame_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    resized = interpolate(tensor, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
    resized = resized.clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).numpy()
    top = max(0, (new_h - crop) // 2)
    left = max(0, (new_w - crop) // 2)
    return resized[top : top + crop, left : left + crop, :]


def _apply_rotation(frame_rgb: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate an RGB frame by multiples of 90 degrees.

    Parameters:
        frame_rgb: HxWx3 array.
        rotation: 0/90/180/270 clockwise.

    Returns:
        The rotated array.
    """
    if rotation == 90:
        return np.ascontiguousarray(np.rot90(frame_rgb, k=-1, axes=(0, 1)))
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(frame_rgb, k=2, axes=(0, 1)))
    if rotation == 270:
        return np.ascontiguousarray(np.rot90(frame_rgb, k=1, axes=(0, 1)))
    return frame_rgb


def decode_frames(path, start_s: float, duration_s: float, rotation: int = 0) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (media_time_seconds, RGB frame) once over the requested window.

    Parameters:
        path: Video file path.
        start_s: Window start in seconds.
        duration_s: Window duration.
        rotation: Container rotation to apply.

    Yields:
        Tuples of (pts_seconds, HxWx3 uint8 array).

    Raises:
        av.error.FFmpegError: When the container or codec cannot be decoded.
    """
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base.numerator) / float(stream.time_base.denominator)
        stream_start = float(stream.start_time or 0) * time_base
        end_s = start_s + duration_s
        container.seek(
            int(max(0.0, start_s - 1.0) / time_base) + int(stream.start_time or 0),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        for frame in container.decode(stream):
            pts = float(frame.pts if frame.pts is not None else 0) * time_base - stream_start
            if pts + 1e-6 < start_s:
                continue
            if pts > end_s + 0.5:
                break
            rgb = frame.to_ndarray(format="rgb24")
            rgb = _apply_rotation(rgb, rotation)
            yield pts, rgb
    finally:
        container.close()


def _target_times(start_s: float, duration_s: float, fps: float) -> list[float]:
    """Compute evenly spaced sample timestamps on a half-open window.

    Parameters:
        start_s: Window start.
        duration_s: Window duration.
        fps: Sampling rate.

    Returns:
        Timestamp list of length ceil(duration * fps), guaranteed >= 1.
    """
    count = max(1, math.ceil(round(duration_s * fps, 6)))
    return [start_s + (index / fps) for index in range(count)]


def extract_frame_streams(
    path,
    *,
    start_s: float,
    duration_s: float,
    visual_fps: float,
    sync_fps: float,
    rotation: int = 0,
    source_fps: float = 0.0,
) -> FrameStreams:
    """Decode the video once and sample both conditioning streams.

    Frames are chosen by nearest-preceding-timestamp selection; frames past the
    end of the decoded range are edge-clamped (repetition of the last real frame)
    and counted in `edge_clamps`.

    Parameters:
        path: Video file path.
        start_s: Window start seconds.
        duration_s: Window duration seconds.
        visual_fps: Semantic stream rate.
        sync_fps: Sync stream rate.
        rotation: Container rotation degrees.
        source_fps: Source average fps (metadata only).

    Returns:
        FrameStreams with uint8 tensors and exact timestamps.

    Raises:
        av.error.FFmpegError: On decode failure.
        ValueError: When no frames could be decoded.
    """
    visual_targets = _target_times(start_s, duration_s, visual_fps)
    sync_targets = _target_times(start_s, duration_s, sync_fps)
    visual_buf: list[np.ndarray] = []
    sync_buf: list[np.ndarray] = []
    edge_clamps = 0

    class _Selector:
        """Nearest-preceding-frame sampler over one target list."""

        def __init__(self, targets: list[float], side: int) -> None:
            """Prepare the sampler.

            Parameters:
                targets: Sorted target timestamps.
                side: Resize shorter-side target.
            """
            self.targets = targets
            self.side = side
            self.cursor = 0
            self.current: np.ndarray | None = None

        def observe(self, pts: float, frame: np.ndarray) -> None:
            """Record a decoded frame and consume eligible targets.

            Parameters:
                pts: Frame timestamp.
                frame: Prepared (cropped) array.
            """
            while self.cursor < len(self.targets) and self.targets[self.cursor] < pts - 1e-6:
                if self.current is not None:
                    self.emit(self.current)
                self.cursor += 1
            self.current = frame

        def emit(self, frame: np.ndarray) -> None:
            """Append one prepared frame to this stream's buffer.

            Parameters:
                frame: crop x crop x 3 uint8 array.
            """
            if self.side == CLIP_SHORTER_SIDE:
                visual_buf.append(frame)
            else:
                sync_buf.append(frame)

        def finish(self, last_frame: np.ndarray | None) -> int:
            """Flush remaining targets by edge-clamping to the last frame.

            Parameters:
                last_frame: Last decoded prepared frame (any stream).

            Returns:
                Number of clamped samples emitted.
            """
            clamped = 0
            while self.cursor < len(self.targets):
                source = self.current if self.current is not None else last_frame
                if source is None:
                    raise ValueError("no decoded frames available for sampling")
                self.emit(source)
                clamped += 1
                self.cursor += 1
            return clamped

    visual_selector = _Selector(visual_targets, CLIP_SHORTER_SIDE)
    sync_selector = _Selector(sync_targets, SYNC_SHORTER_SIDE)
    last_raw: np.ndarray | None = None
    decoded_any = False
    for pts, rgb in decode_frames(path, start_s, duration_s, rotation):
        decoded_any = True
        visual_prepared = resize_and_crop(rgb, CLIP_SHORTER_SIDE, CROP_SIZE)
        sync_prepared = resize_and_crop(rgb, SYNC_SHORTER_SIDE, CROP_SIZE)
        last_raw = visual_prepared
        visual_selector.observe(pts, visual_prepared)
        sync_selector.observe(pts, sync_prepared)
    if not decoded_any:
        raise ValueError("no frames decoded; video stream is empty or corrupt")
    edge_clamps += visual_selector.finish(last_raw)
    edge_clamps += sync_selector.finish(last_raw)

    visual_tensor = torch.from_numpy(np.stack(visual_buf).transpose(0, 3, 1, 2).copy())
    sync_tensor = torch.from_numpy(np.stack(sync_buf).transpose(0, 3, 1, 2).copy())
    groups: list[list[int]] = []
    count = sync_tensor.shape[0]
    offset = 0
    while offset + SYNC_SEGMENT_FRAMES <= count:
        groups.append(list(range(offset, offset + SYNC_SEGMENT_FRAMES)))
        offset += 8
    return FrameStreams(
        visual=visual_tensor,
        visual_timestamps=torch.tensor(visual_targets, dtype=torch.float64),
        sync=sync_tensor,
        sync_timestamps=torch.tensor(sync_targets, dtype=torch.float64),
        edge_clamps=edge_clamps,
        start_s=start_s,
        duration_s=duration_s,
        source_fps=source_fps,
        segment_groups=groups,
    )
