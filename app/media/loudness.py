"""ITU-R BS.1770 loudness measurement, gain staging, and true-peak limiting."""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
import torch

from app.constants import LOUDNESS_BLOCK_SECONDS

__all__ = ["TruePeakLimiter", "gain_to_lufs", "measure_lufs", "measure_true_peak_db"]


def _as_numpy(waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
    """Convert [channels, samples] torch to pyloudnorm orientation.

    Parameters:
        waveform: Tensor waveform.
        sample_rate: Rate in Hz (unused, kept for interface clarity).

    Returns:
        Float64 [samples, channels] array.
    """
    del sample_rate
    data = waveform.detach().cpu().to(torch.float64).numpy()
    if data.ndim == 1:
        data = data[None, :]
    return data.T


def measure_lufs(waveform: torch.Tensor, sample_rate: int) -> float:
    """Measure integrated loudness in LUFS.

    Parameters:
        waveform: [channels, samples] tensor.
        sample_rate: Rate in Hz.

    Returns:
        Integrated LUFS (-inf for silence).
    """
    meter = pyln.Meter(int(sample_rate), block_size=LOUDNESS_BLOCK_SECONDS)
    data = _as_numpy(waveform, sample_rate)
    if data.shape[0] < int(sample_rate * LOUDNESS_BLOCK_SECONDS):
        return float("-inf")
    return float(meter.integrated_loudness(data))


def gain_to_lufs(
    waveform: torch.Tensor, sample_rate: int, target_lufs: float
) -> tuple[torch.Tensor, float, float]:
    """Apply a linear gain to reach a target integrated loudness.

    Parameters:
        waveform: [channels, samples] tensor.
        sample_rate: Rate in Hz.
        target_lufs: Desired LUFS.

    Returns:
        (scaled tensor, measured input LUFS, applied gain dB).
    """
    measured = measure_lufs(waveform, sample_rate)
    if measured == float("-inf") or not math.isfinite(measured):
        return waveform, measured, 0.0
    gain_db = target_lufs - measured
    gain = 10.0 ** (gain_db / 20.0)
    return waveform * gain, measured, gain_db


def measure_true_peak_db(waveform: torch.Tensor, oversample: int = 4) -> float:
    """Estimate true peak (dBTP) via polyphase 4x oversampling.

    Parameters:
        waveform: [channels, samples] tensor.
        oversample: Oversampling factor.

    Returns:
        True peak in dBTP.
    """
    data = waveform.detach().cpu().to(torch.float64).numpy()
    if data.ndim == 1:
        data = data[None, :]
    peak = 0.0
    for channel in data:
        up = np.repeat(channel, oversample)
        kernel = np.array([0.25, 0.5, 0.25])
        up = np.convolve(up, kernel, mode="same")
        peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * math.log10(peak))


class TruePeakLimiter:
    """Look-ahead soft limiter guaranteeing output below a true-peak ceiling.

    Uses a soft-knee gain envelope driven by a sliding-window peak of the
    (internally 4x-oversampled) signal with attack/release smoothing.
    """

    __slots__ = ("attack_ms", "ceiling", "oversample", "release_ms")

    def __init__(
        self,
        ceiling_db: float,
        *,
        attack_ms: float = 5.0,
        release_ms: float = 150.0,
        oversample: int = 4,
    ) -> None:
        """Configure the limiter.

        Parameters:
            ceiling_db: Ceiling in dBTP (e.g. -1.0).
            attack_ms: Look-ahead attack window.
            release_ms: Gain-release time constant.
            oversample: True-peak oversampling factor.
        """
        self.ceiling = 10.0 ** (ceiling_db / 20.0)
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.oversample = oversample

    def process(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Limit the waveform.

        Parameters:
            waveform: [channels, samples] tensor.
            sample_rate: Rate in Hz.

        Returns:
            The limited tensor, guaranteed not to exceed the ceiling.
        """
        data = waveform.detach().cpu().to(torch.float64).numpy()
        if data.ndim == 1:
            data = data[None, :]
        ceiling = self.ceiling * 0.995
        attack_n = max(1, int(sample_rate * self.attack_ms / 1000.0))
        release_n = max(1, int(sample_rate * self.release_ms / 1000.0))
        out = np.empty_like(data)
        for index, channel in enumerate(data):
            env = np.abs(channel)
            for factor in range(2, self.oversample + 1):
                stretched = np.repeat(channel, factor)
                kernel = np.array([0.25, 0.5, 0.25])[: min(3, stretched.size)]
                if kernel.size:
                    stretched = np.convolve(stretched, kernel, mode="same")
                env = np.maximum(env, np.abs(stretched)[::factor][: env.size])
            look = _sliding_window_max(env, attack_n)
            needed = np.where(look > ceiling, ceiling / np.maximum(look, 1e-12), 1.0)
            gain = _smooth_attack_release(needed, attack_n, release_n)
            limited = channel * gain
            over = np.abs(limited) > ceiling
            if np.any(over):
                limited = np.clip(limited, -ceiling, ceiling)
            out[index] = limited
        return torch.from_numpy(out.astype(np.float32))


def _sliding_window_max(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a causal sliding-window maximum.

    Parameters:
        values: Input envelope.
        window: Window length.

    Returns:
        Windowed maximum array (same length).
    """
    if window <= 1 or values.size == 0:
        return values
    padded = np.concatenate([np.zeros(window - 1), values])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return windows.max(axis=1)


def _smooth_attack_release(target: np.ndarray, attack_n: int, release_n: int) -> np.ndarray:
    """One-pole smooth a gain envelope with separate attack/release speeds.

    Parameters:
        target: Desired gain per sample.
        attack_n: Attack coefficient window.
        release_n: Release coefficient window.

    Returns:
        The smoothed gain envelope.
    """
    smoothed = np.empty_like(target)
    current = 1.0
    attack_coeff = math.exp(-1.0 / max(attack_n, 1))
    release_coeff = math.exp(-1.0 / max(release_n, 1))
    for index, wanted in enumerate(target):
        if wanted < current:
            current = attack_coeff * current + (1 - attack_coeff) * wanted
        else:
            current = release_coeff * current + (1 - release_coeff) * wanted
        smoothed[index] = current
    return smoothed
