"""Sliding-window generation scheduling and crossfaded overlap-add reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["WindowPlan", "equal_power_crossfade_add", "plan_windows", "slice_by_timestamps"]


@dataclass(slots=True)
class WindowPlan:
    """One generation window over the target timeline.

    Attributes:
        index: Window ordinal.
        start_s: Window start in media time.
        length_s: Window length.
        overlap_s: Overlap with the previous window.
        crop_start_s / crop_end_s: Region owned exclusively by this window
            (half the overlap trimmed on each shared boundary).
    """

    index: int
    start_s: float
    length_s: float
    overlap_s: float
    crop_start_s: float
    crop_end_s: float


def plan_windows(duration_s: float, window_s: float, overlap_s: float) -> list[WindowPlan]:
    """Compute the window schedule covering [0, duration].

    Parameters:
        duration_s: Total generation duration.
        window_s: Native model window.
        overlap_s: Crossfade overlap (< window/2).

    Returns:
        WindowPlan list (>= 1 window).

    Raises:
        ValueError: When overlap >= window/2 or arguments are non-positive.
    """
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    if overlap_s >= window_s / 2:
        raise ValueError("overlap must be < window/2")
    stride = window_s - overlap_s
    plans: list[WindowPlan] = []
    start = 0.0
    index = 0
    while True:
        length = min(window_s, duration_s - start)
        if length <= 0:
            break
        crop_start = start + (overlap_s / 2 if index > 0 else 0.0)
        crop_end = start + length - (overlap_s / 2 if start + length < duration_s - 1e-9 else 0.0)
        plans.append(
            WindowPlan(
                index=index,
                start_s=start,
                length_s=length,
                overlap_s=overlap_s,
                crop_start_s=crop_start,
                crop_end_s=crop_end,
            )
        )
        if start + length >= duration_s - 1e-9:
            break
        start += stride
        index += 1
    return plans


def slice_by_timestamps(
    timestamps: torch.Tensor, start_s: float, end_s: float
) -> tuple[int, int]:
    """Locate the [first, last) indices of timestamps within a window.

    Timestamps at exactly start are included; the first timestamp >= end is
    excluded, and a tolerance of one frame is applied so boundary frames are
    not dropped.

    Parameters:
        timestamps: Monotonic [T] tensor of media seconds.
        start_s: Window start.
        end_s: Window end.

    Returns:
        (first_index, last_index_exclusive); (0, 0) when empty.
    """
    if timestamps.numel() == 0:
        return 0, 0
    frame = float(timestamps[0]) if timestamps.numel() == 1 else float(timestamps[1] - timestamps[0])
    tolerance = max(abs(frame) * 0.5, 1e-6)
    values = timestamps.tolist()
    first = 0
    for index, value in enumerate(values):
        if value >= start_s - tolerance:
            first = index
            break
    last = len(values)
    for index in range(first, len(values)):
        if values[index] > end_s + tolerance:
            last = index
            break
    return first, max(first, last)


def equal_power_crossfade_add(
    total: torch.Tensor,
    piece: torch.Tensor,
    sample_offset: int,
    fade_samples: int,
) -> torch.Tensor:
    """Blend `piece` into `total` with raised-cosine (equal-power) crossfades.

    Overlapping regions sum with cos/sin power windows so constant-amplitude
    signals remain constant; non-overlapping regions are added directly.

    Parameters:
        total: [channels, samples] accumulator (modified copy returned).
        piece: [channels, piece_samples] window waveform.
        sample_offset: Where the piece begins in total.
        fade_samples: Crossfade length in samples.

    Returns:
        The updated accumulator.

    Raises:
        ValueError: On inconsistent shapes or offsets.
    """
    if total.dim() != 2 or piece.dim() != 2:
        raise ValueError("expected 2D [channels, samples] tensors")
    if total.shape[0] != piece.shape[0]:
        raise ValueError("channel mismatch")
    if sample_offset < 0 or sample_offset + piece.shape[-1] > total.shape[-1]:
        raise ValueError("piece exceeds accumulator bounds")
    result = total.clone()
    end = sample_offset + piece.shape[-1]
    fade = min(fade_samples, piece.shape[-1] // 2)
    if fade <= 0:
        result[..., sample_offset:end] += piece
        return result
    angle = torch.arange(fade, device=piece.device, dtype=torch.float32) / fade * (3.141592653589793 / 2)
    fade_in = torch.sin(angle).reshape(1, -1)
    fade_out = torch.cos(angle).reshape(1, -1)
    head = sample_offset
    head_end = sample_offset + fade
    tail_start = end - fade
    result[..., head:head_end] = result[..., head:head_end] * fade_out.to(result.dtype) + piece[..., :fade] * fade_in.to(piece.dtype)
    result[..., head_end:tail_start] += piece[..., fade : piece.shape[-1] - fade]
    result[..., tail_start:end] = result[..., tail_start:end] * fade_out.flip(-1).to(result.dtype) + piece[..., -fade:] * fade_in.flip(-1).to(piece.dtype)
    return result


def target_samples(duration_s: float, sample_rate: int) -> int:
    """Exact output sample count.

    Parameters:
        duration_s: Duration seconds.
        sample_rate: Rate Hz.

    Returns:
        round(duration * rate).
    """
    return round(duration_s * sample_rate)
