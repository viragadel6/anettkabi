"""Rotary positional embeddings with a shared fractional time base."""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["RotaryPositionEmbedding", "apply_rope", "rope_time_positions"]


def rope_time_positions(seconds: torch.Tensor, fps: float) -> torch.Tensor:
    """Convert timestamps to token positions on a shared seconds-based axis.

    Parameters:
        seconds: Timestamp tensor [T].
        fps: Native frames-per-second of the stream.

    Returns:
        Position tensor [T] = seconds * fps (fractional).
    """
    return seconds.to(torch.float32) * float(fps)


class RotaryPositionEmbedding(nn.Module):
    """Rotary embedding over arbitrary (possibly fractional) positions.

    Frequencies are computed in float32 and applied in the module dtype so
    streams sampled at different rates share one physical time base: pass
    `seconds * native_fps` as positions and identical real moments rotate
    identically across streams.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_positions: int = 65536) -> None:
        """Initialize the rotary embedding.

        Parameters:
            dim: Head dimension to rotate (must be even).
            base: Frequency base.
            max_positions: Cached table length.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("rope dim must be even")
        self.dim = dim
        self.base = base
        self.max_positions = max_positions

    def _cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables for a position tensor.

        Parameters:
            positions: [T] or [B, T] fractional positions.

        Returns:
            (cos, sin) of shape [T, dim] or [B, T, dim].
        """
        device = positions.device
        dtype = torch.float32
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float64) / self.dim)
        )
        angles = positions.to(torch.float64).unsqueeze(-1) * inv_freq.to(torch.float64)
        cos = angles.cos().to(dtype)
        sin = angles.sin().to(dtype)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        return cos, sin

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables for the given positions.

        Parameters:
            positions: Fractional positions [T] or [B, T].

        Returns:
            Broadcastable cos/sin tensors.
        """
        return self._cos_sin(positions)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to the second-to-last axis.

    Pairs adjacent feature channels (GPT-NeoX style) on tensors of
    shape [B, H, T, D].

    Parameters:
        x: Input tensor [B, H, T, D].
        cos: Cos table broadcastable to [B, 1, T, D].
        sin: Sin table broadcastable to [B, 1, T, D].

    Returns:
        The rotated tensor, same shape.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos1 = cos[..., 0::2].to(x.dtype)
    sin1 = sin[..., 0::2].to(x.dtype)
    rotated1 = x1 * cos1 - x2 * sin1
    rotated2 = x1 * sin1 + x2 * cos1
    out = torch.empty_like(x)
    out[..., 0::2] = rotated1
    out[..., 1::2] = rotated2
    return out


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Classic sinusoidal position table (used for text tokens).

    Parameters:
        length: Table length.
        dim: Feature dimension (even).
        device: Target device.

    Returns:
        [length, dim] float32 table.
    """
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    table = torch.zeros(length, dim, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * div_term)
    table[:, 1::2] = torch.cos(position * div_term)
    return table
