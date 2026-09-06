"""Exact STFT/log-mel frontend matching the training configuration."""

from __future__ import annotations

import math

import torch
from torch import nn

from app.ml.registry import VariantSpec

__all__ = ["MelSpectrogram", "mel_frames_for", "waveform_from_mel_length"]


def mel_frames_for(duration_s: float, hop_length: int, sample_rate: int) -> int:
    """Number of mel frames produced for a duration.

    Parameters:
        duration_s: Duration seconds.
        hop_length: STFT hop.
        sample_rate: Rate in Hz.

    Returns:
        Frame count (center-padded STFT convention: 1 + samples // hop).
    """
    samples = round(duration_s * sample_rate)
    return 1 + samples // hop_length


def waveform_from_mel_length(frames: int, hop_length: int) -> int:
    """Waveform sample count a vocoder produces for mel frames.

    Parameters:
        frames: Mel frame count.
        hop_length: Upsampling product of the vocoder.

    Returns:
        samples = frames * hop_length.
    """
    return frames * hop_length


def _hann_window(win_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a periodic Hann window.

    Parameters:
        win_length: Window length.
        device: Target device.
        dtype: Target dtype.

    Returns:
        The window tensor.
    """
    return torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)


class MelSpectrogram(nn.Module):
    """Log-mel spectrogram transform (STFT -> mel filterbank -> log).

    Uses torch.stft with reflect padding and a precomputed mel basis from a
    slaney-style filterbank, exactly reproducing the training transform when
    constructed with the same VariantSpec.
    """

    def __init__(self, spec: VariantSpec) -> None:
        """Initialize from a variant spec.

        Parameters:
            spec: Variant with STFT geometry.
        """
        super().__init__()
        self.spec = spec
        self.n_fft = spec.n_fft
        self.hop_length = spec.hop_length
        self.win_length = spec.win_length
        self.n_mels = spec.n_mels
        self.fmin = spec.fmin
        self.fmax = spec.fmax or spec.sample_rate / 2
        self.clip_value = spec.mel_clip_value
        self.register_buffer("window", _hann_window(spec.win_length, torch.device("cpu"), torch.float32), persistent=False)
        self.register_buffer("mel_basis", self._build_mel_basis(), persistent=False)

    def _build_mel_basis(self) -> torch.Tensor:
        """Build the mel filterbank matrix.

        Returns:
            [n_mels, n_fft // 2 + 1] float32 matrix.
        """
        bins = self.n_fft // 2 + 1
        fft_freqs = torch.linspace(0, self.spec.sample_rate / 2, bins)
        mel_points = torch.linspace(
            self._hz_to_mel(self.fmin), self._hz_to_mel(self.fmax), self.n_mels + 2
        )
        hz_points = self._mel_to_hz(mel_points)
        weights = torch.zeros(self.n_mels, bins)
        for band in range(self.n_mels):
            left = hz_points[band]
            center = hz_points[band + 1]
            right = hz_points[band + 2]
            up = (fft_freqs - left) / (center - left).clamp(min=1e-9)
            down = (right - fft_freqs) / (right - center).clamp(min=1e-9)
            weights[band] = torch.clamp(torch.minimum(up, down), min=0.0)
        enorm = 2.0 / (hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels])
        weights *= enorm.unsqueeze(1)
        return weights

    @staticmethod
    def _hz_to_mel(hz: float | torch.Tensor) -> torch.Tensor:
        """Convert Hz to mels (HTK / Slaney-hybrid used by librosa).

        Parameters:
            hz: Frequency value(s).

        Returns:
            Mel value tensor.
        """
        hz_t = torch.as_tensor(hz, dtype=torch.float32)
        f_min = 0.0
        f_sp = 200.0 / 3
        mels = (hz_t - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if hz_t.numel() > 1:
            log_t = hz_t >= min_log_hz
            mels = torch.where(
                log_t,
                min_log_mel + torch.log(hz_t / min_log_hz) / logstep,
                mels,
            )
        elif float(hz_t) >= min_log_hz:
            mels = min_log_mel + torch.log(hz_t / min_log_hz) / logstep
        return mels

    @staticmethod
    def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
        """Convert mels to Hz.

        Parameters:
            mels: Mel values.

        Returns:
            Hz values.
        """
        f_sp = 200.0 / 3
        freqs = mels * f_sp
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep = math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        return torch.where(log_t, min_log_hz * torch.exp(logstep * (mels - min_log_mel)), freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute the log-mel spectrogram.

        Parameters:
            waveform: [B, samples] or [1, samples] float tensor.

        Returns:
            [B, n_mels, frames] log-mel tensor.
        """
        data = waveform.to(torch.float32)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        mono = data.mean(dim=0) if data.shape[0] > 1 else data[0]
        window = self.window.to(mono.device)
        stft = torch.stft(
            mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = stft.abs().pow(2)
        mel = torch.matmul(self.mel_basis.to(power.device), power)
        logmel = torch.log(torch.clamp(mel, min=self.clip_value))
        return logmel.transpose(0, 1).unsqueeze(0)

    def num_frames(self, samples: int) -> int:
        """Frames produced for a waveform length.

        Parameters:
            samples: Waveform sample count.

        Returns:
            Mel frame count (center-padded convention).
        """
        return 1 + samples // self.hop_length
