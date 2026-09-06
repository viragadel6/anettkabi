"""Waveform post-processing: resampling, upmix, DC removal, fades, exact length."""

from __future__ import annotations

import math

import numpy as np
import soxr
import torch

from app.constants import FADE_SECONDS

__all__ = [
    "apply_fades",
    "fit_exact_length",
    "normalize_peak",
    "remove_dc",
    "resample",
    "sanitize",
    "to_stereo",
]


def sanitize(waveform: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf samples with zeros and clamp to [-1, 1].

    Parameters:
        waveform: Float tensor [channels, samples] or [samples].

    Returns:
        The sanitized tensor (same shape, float32).
    """
    data = waveform.float()
    data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    return data.clamp(-1.0, 1.0)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample with soxr (high quality, polyphase) keeping channel layout.

    Parameters:
        waveform: [channels, samples] float tensor.
        orig_sr: Source sample rate.
        target_sr: Destination rate.

    Returns:
        The resampled [channels, samples'] tensor.
    """
    if orig_sr == target_sr:
        return waveform
    data = waveform.detach().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] == 1:
        out = soxr.resample(data[0], orig_sr, target_sr, quality="HQ")
        return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)
    channels = [soxr.resample(channel, orig_sr, target_sr, quality="HQ") for channel in data]
    return torch.from_numpy(np.stack(channels).astype(np.float32))


def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Convert mono to stereo; pass through stereo; downmix >2 to stereo.

    Parameters:
        waveform: [channels, samples] tensor.

    Returns:
        [2, samples] tensor.
    """
    if waveform.shape[0] == 2:
        return waveform
    if waveform.shape[0] == 1:
        return waveform.repeat(2, 1)
    left = waveform[0::2].mean(dim=0)
    right = waveform[1::2].mean(dim=0) if waveform.shape[0] > 2 else left
    return torch.stack([left, right])


def remove_dc(waveform: torch.Tensor) -> torch.Tensor:
    """Remove per-channel DC offset.

    Parameters:
        waveform: [channels, samples].

    Returns:
        The mean-subtracted tensor.
    """
    if waveform.numel() == 0:
        return waveform
    return waveform - waveform.mean(dim=-1, keepdim=True)


def apply_fades(waveform: torch.Tensor, sample_rate: int, fade_s: float = FADE_SECONDS) -> torch.Tensor:
    """Apply linear fade-in/out of `fade_s` seconds.

    Parameters:
        waveform: [channels, samples].
        sample_rate: Rate for the fade length conversion.
        fade_s: Fade duration (0 disables).

    Returns:
        The faded tensor.
    """
    samples = waveform.shape[-1]
    fade_len = min(samples // 2, round(fade_s * sample_rate))
    if fade_len <= 0:
        return waveform
    ramp = torch.linspace(0.0, 1.0, fade_len, dtype=waveform.dtype)
    result = waveform.clone()
    result[..., :fade_len] = result[..., :fade_len] * ramp
    result[..., samples - fade_len :] = result[..., samples - fade_len :] * ramp.flip(0)
    return result


def fit_exact_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Trim or zero-pad the last axis to exactly `target_samples`.

    Parameters:
        waveform: [channels, samples].
        target_samples: Required sample count.

    Returns:
        Tensor whose last dimension equals target_samples.

    Raises:
        ValueError: When target_samples is negative.
    """
    if target_samples < 0:
        raise ValueError("target_samples must be >= 0")
    current = waveform.shape[-1]
    if current == target_samples:
        return waveform
    if current > target_samples:
        return waveform[..., :target_samples]
    padding = torch.zeros(
        (*waveform.shape[:-1], target_samples - current), dtype=waveform.dtype
    )
    return torch.cat([waveform, padding], dim=-1)


def normalize_peak(waveform: torch.Tensor, peak_db: float) -> torch.Tensor:
    """Scale the waveform so its absolute peak reaches `peak_db` (when nonzero).

    Parameters:
        waveform: [channels, samples].
        peak_db: Target peak in dBFS.

    Returns:
        The scaled tensor.
    """
    peak = waveform.abs().max().item()
    if peak < 1e-8:
        return waveform
    target = 10 ** (peak_db / 20.0)
    return waveform * (target / peak)


def db_to_gain(db: float) -> float:
    """Convert decibels to a linear gain factor.

    Parameters:
        db: Decibel value.

    Returns:
        The linear gain.
    """
    return 10.0 ** (db / 20.0)


def samples_for(duration_s: float, sample_rate: int) -> int:
    """Compute the exact sample count for a duration.

    Parameters:
        duration_s: Duration in seconds.
        sample_rate: Rate in Hz.

    Returns:
        round(duration_s * sample_rate) as int.
    """
    return round(duration_s * sample_rate)


def rms(waveform: torch.Tensor) -> float:
    """Compute RMS level of a waveform.

    Parameters:
        waveform: Any-shape tensor.

    Returns:
        Root-mean-square value.
    """
    if waveform.numel() == 0:
        return 0.0
    return float(math.sqrt(float((waveform.float() ** 2).mean().item())))
