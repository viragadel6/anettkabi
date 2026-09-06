"""Audio bus mixing: replace, gain-mix, and sidechain ducking modes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.media.audio_ops import db_to_gain, remove_dc, resample, sanitize, to_stereo
from app.media.loudness import TruePeakLimiter

__all__ = ["MixParams", "envelope_follower", "mix_audio"]


@dataclass(slots=True)
class MixParams:
    """Parameters controlling bus mixing.

    Attributes:
        audio_mode: replace | mix | duck.
        sfx_gain_db: Gain on the generated bus.
        original_gain_db: Gain on the original-audio bus.
        duck_threshold_db: Sidechain threshold in dBFS.
        duck_ratio: Compression ratio.
        duck_attack_ms / duck_release_ms: Envelope times.
        limiter_ceiling_db: Final bus true-peak ceiling.
    """

    audio_mode: str
    sfx_gain_db: float = 0.0
    original_gain_db: float = -6.0
    duck_threshold_db: float = -24.0
    duck_ratio: float = 4.0
    duck_attack_ms: float = 15.0
    duck_release_ms: float = 250.0
    limiter_ceiling_db: float = -1.0


def envelope_follower(
    signal: torch.Tensor, sample_rate: int, attack_ms: float, release_ms: float
) -> torch.Tensor:
    """Compute a smoothed absolute envelope (attack faster than release).

    Parameters:
        signal: [channels, samples] tensor.
        sample_rate: Rate in Hz.
        attack_ms: Attack time.
        release_ms: Release time.

    Returns:
        Envelope tensor with the shape [1, samples].
    """
    import math

    data = signal.abs().amax(dim=0, keepdim=True).double()
    attack_n = max(1.0, sample_rate * attack_ms / 1000.0)
    release_n = max(1.0, sample_rate * release_ms / 1000.0)
    attack_coeff = math.exp(-1.0 / attack_n)
    release_coeff = math.exp(-1.0 / release_n)
    envelope = torch.empty_like(data)
    current = 0.0
    values = data[0].tolist()
    out = []
    for value in values:
        if value > current:
            current = attack_coeff * current + (1 - attack_coeff) * value
        else:
            current = release_coeff * current + (1 - release_coeff) * value
        out.append(current)
    envelope[0] = torch.tensor(out, dtype=torch.float64)
    return envelope.float()


def _duck_gain(
    envelope: torch.Tensor, threshold: float, ratio: float
) -> torch.Tensor:
    """Map an envelope to a ducking gain curve above threshold.

    Parameters:
        envelope: Envelope values [1, samples].
        threshold: Linear threshold.
        ratio: Compression ratio.

    Returns:
        Gain multiplier tensor [1, samples].
    """
    above = envelope > threshold
    excess_db = 20.0 * torch.log10(envelope / threshold + 1e-12)
    reduced_db = excess_db * (1.0 / ratio - 1.0)
    gain = torch.pow(10.0, reduced_db / 20.0)
    return torch.where(above, gain.clamp(max=1.0), torch.ones_like(gain))


def mix_audio(
    sfx: torch.Tensor,
    original: torch.Tensor | None,
    sfx_sample_rate: int,
    original_sample_rate: int | None,
    output_sample_rate: int,
    params: MixParams,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Combine the generated SFX with the original audio per `audio_mode`.

    Parameters:
        sfx: Generated stereo [2, samples] tensor.
        original: Original audio tensor (any channel count) or None.
        sfx_sample_rate: Rate of `sfx`.
        original_sample_rate: Rate of `original` when present.
        output_sample_rate: Destination rate.
        params: MixParams with gains and ducking settings.

    Returns:
        (mixed [2, samples] tensor at output_sample_rate, metadata dict).
    """
    metadata: dict[str, object] = {"audio_mode": params.audio_mode}
    sfx = sanitize(sfx)
    sfx = resample(sfx, sfx_sample_rate, output_sample_rate)
    sfx = to_stereo(sfx)
    sfx = sfx * db_to_gain(params.sfx_gain_db)
    if params.audio_mode == "replace" or original is None:
        metadata["original_audio_present"] = original is not None
        limiter = TruePeakLimiter(params.limiter_ceiling_db)
        result = limiter.process(sfx, output_sample_rate)
        metadata["mixed_buses"] = ["sfx"]
        return result, metadata
    original = sanitize(original)
    assert original_sample_rate is not None
    original = resample(original, original_sample_rate, output_sample_rate)
    original = to_stereo(original)
    original = original * db_to_gain(params.original_gain_db)
    original = remove_dc(original)
    length = min(sfx.shape[-1], original.shape[-1])
    if sfx.shape[-1] != original.shape[-1]:
        metadata["length_mismatch_trimmed"] = True
        sfx = sfx[..., :length]
        original = original[..., :length]
    if params.audio_mode == "mix":
        summed = sfx + original
        metadata["mixed_buses"] = ["sfx", "original"]
    elif params.audio_mode == "duck":
        envelope = envelope_follower(
            sfx, output_sample_rate, params.duck_attack_ms, params.duck_release_ms
        )
        threshold = db_to_gain(params.duck_threshold_db)
        gain = _duck_gain(envelope, threshold, params.duck_ratio)
        ducked = original * gain
        metadata["mixed_buses"] = ["sfx", "original(ducked)"]
        metadata["duck_max_reduction_db"] = float(
            20.0 * torch.log10(gain.min().clamp(min=1e-6)).item()
        )
        summed = sfx + ducked
    else:
        raise ValueError(f"unknown audio_mode {params.audio_mode!r}")
    limiter = TruePeakLimiter(params.limiter_ceiling_db)
    result = limiter.process(summed, output_sample_rate)
    return result, metadata
