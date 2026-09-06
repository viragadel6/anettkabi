"""Sync encoder: segment-wise spatio-temporal transformer for onset alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from app.ml.registry import SyncEncoderSpec

__all__ = ["SyncEncoder", "TemporalTransformer"]

SYNC_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
SYNC_NORMALIZE_STD = (0.229, 0.224, 0.225)


class TemporalTransformer(nn.Module):
    """Transformer encoder over per-frame aggregated patch tokens."""

    def __init__(self, dim: int, depth: int, heads: int) -> None:
        """Initialize the stack.

        Parameters:
            dim: Width.
            depth: Blocks.
            heads: Heads.
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder.

        Parameters:
            x: [B, T, D].

        Returns:
            [B, T, D].
        """
        return self.norm(self.blocks(x))


class SyncEncoder(nn.Module):
    """High-rate synchronization feature extractor.

    Input: 16-frame segments of 224x224 RGB frames (stride-8 grouping).
    A Conv3d patch stem produces per-frame tokens; a temporal transformer
    mixes them; the last `out_per_segment` frame tokens are emitted as the
    segment's features, giving ~25 tokens/s with stride-8 segments on a
    25 fps stream.
    """

    def __init__(self, spec: SyncEncoderSpec) -> None:
        """Build the encoder.

        Parameters:
            spec: Sync encoder hyperparameters.
        """
        super().__init__()
        self.spec = spec
        temporal, spatial_h, spatial_w = spec.patch
        self.patch_embed = nn.Conv3d(
            3,
            spec.dim,
            kernel_size=(temporal, spatial_h, spatial_w),
            stride=(1, spatial_h, spatial_w),
        )
        tokens_per_frame = (224 // spatial_w) * (224 // spatial_h)
        self.frame_norm = nn.LayerNorm(spec.dim)
        self.frame_proj = nn.Linear(spec.dim * tokens_per_frame, spec.dim)
        self.frame_pos = nn.Parameter(torch.randn(spec.segment_frames, spec.dim) * 0.02)
        self.transformer = TemporalTransformer(spec.dim, spec.depth, spec.heads)
        self.out_norm = nn.LayerNorm(spec.dim)
        self.tokens_per_frame = tokens_per_frame

    def forward(self, segment: torch.Tensor) -> torch.Tensor:
        """Encode one segment.

        Parameters:
            segment: [B, 3, segment_frames, 224, 224] float tensor
                (normalized per SYNC statistics).

        Returns:
            [B, out_per_segment, dim] features.
        """
        frames = self.spec.segment_frames
        if segment.shape[2] < frames:
            pad = frames - segment.shape[2]
            segment = F.pad(segment, (0, 0, 0, 0, 0, pad))
        elif segment.shape[2] > frames:
            segment = segment[:, :, :frames]
        embedded = self.patch_embed(segment)
        b, d, t, h, w = embedded.shape
        per_frame = embedded.permute(0, 2, 1, 3, 4).reshape(b, t, d * h * w)
        tokens = self.frame_proj(self.frame_norm(per_frame))
        tokens = tokens + self.frame_pos.to(tokens.dtype)[: t].unsqueeze(0)
        encoded = self.transformer(tokens)
        tail = encoded[:, -self.spec.out_per_segment :, :]
        return self.out_norm(tail)

    @torch.no_grad()
    def encode_segments(
        self,
        frames_uint8: torch.Tensor,
        groups: list[list[int]],
        device: torch.device,
        *,
        batch_segments: int = 8,
    ) -> tuple[torch.Tensor, list[float]]:
        """Encode grouped frame indices into high-rate features.

        Parameters:
            frames_uint8: [T, 3, 224, 224] uint8 sync stream.
            groups: Frame-index lists (16 entries each, stride 8).
            device: Compute device.
            batch_segments: Segments per forward pass.

        Returns:
            (features [N, out_per_segment, dim], segment_start_seconds_coeff)
            where coefficient `i` corresponds to group i's start frame.

        Raises:
            ValueError: When groups are empty.
        """
        if not groups:
            raise ValueError("no sync segments provided")
        self.eval()
        mean = torch.tensor(SYNC_NORMALIZE_MEAN, device=device).reshape(3, 1, 1, 1)
        std = torch.tensor(SYNC_NORMALIZE_STD, device=device).reshape(3, 1, 1, 1)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(groups), batch_segments):
            chunk = groups[start : start + batch_segments]
            batch = torch.stack(
                [
                    torch.stack([frames_uint8[idx] for idx in group])
                    for group in chunk
                ]
            ).to(device).to(torch.float32) / 255.0
            batch = (batch - mean) / std
            outputs.append(self.forward(batch).float().cpu())
        return torch.cat(outputs, dim=0), [float(group[0]) for group in groups]

    @staticmethod
    def seconds_for_frames(frame_indices: list[float], fps: float) -> list[float]:
        """Convert frame indices to media seconds.

        Parameters:
            frame_indices: Frame positions in the sync stream.
            fps: Sync stream rate.

        Returns:
            Timestamps in seconds.
        """
        return [index / fps for index in frame_indices]
