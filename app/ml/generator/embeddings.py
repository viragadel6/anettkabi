"""Timestep/modality embeddings and adaLN-Zero modulation heads."""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = [
    "AdaLNZeroModulation",
    "ModalityEmbedding",
    "PooledConditionEmbedding",
    "TimestepEmbedding",
]


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep frequency map followed by an MLP."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256, max_period: float = 10000.0) -> None:
        """Initialize the embedding.

        Parameters:
            hidden_dim: Output width.
            frequency_dim: Sinusoid count.
            max_period: Base period.
        """
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _frequencies(self, t: torch.Tensor) -> torch.Tensor:
        """Compute half-cos/half-sin features for t in [0, 1].

        Parameters:
            t: [B] or [B, 1] float tensor.

        Returns:
            [B, frequency_dim] features.
        """
        half = self.frequency_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half, device=t.device, dtype=torch.float32
        ) / half
        angles = t.to(torch.float32).reshape(-1, 1) * torch.exp(exponent).unsqueeze(0) * 1000.0
        return torch.cat([angles.cos(), angles.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed flow-matching timesteps.

        Parameters:
            t: [B] or [B, 1] float tensor in [0, 1].

        Returns:
            [B, hidden_dim] embedding.
        """
        return self.mlp(self._frequencies(t))


class ModalityEmbedding(nn.Module):
    """Learned additive embeddings distinguishing token modalities."""

    def __init__(self, dim: int, modalities: int) -> None:
        """Initialize the table.

        Parameters:
            dim: Token width.
            modalities: Number of distinct modalities.
        """
        super().__init__()
        self.embedding = nn.Embedding(modalities, dim)

    def forward(self, modality_index: int, batch: int, length: int, device: torch.device) -> torch.Tensor:
        """Produce a broadcastable [B, T, D] modality embedding.

        Parameters:
            modality_index: Modality id.
            batch: Batch size.
            length: Token count.
            device: Target device.

        Returns:
            The embedding tensor.
        """
        base = self.embedding.weight[modality_index].to(device).reshape(1, 1, -1)
        return base.expand(batch, length, -1)


class AdaLNZeroModulation(nn.Module):
    """adaLN-Zero: conditioning vector to per-block shift/scale/gate triples."""

    def __init__(self, condition_dim: int, target_dim: int, num_outputs: int = 6) -> None:
        """Initialize the modulation head.

        Parameters:
            condition_dim: Width of the pooled conditioning vector.
            target_dim: Width of the modulated tensor.
            num_outputs: Number of produced parameters (6 for joint blocks).
        """
        super().__init__()
        self.num_outputs = num_outputs
        self.linear = nn.Linear(condition_dim, num_outputs * target_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, condition: torch.Tensor, dtype: torch.dtype | None = None) -> list[torch.Tensor]:
        """Produce modulation parameters.

        Parameters:
            condition: [B, condition_dim].
            dtype: Optional cast for the outputs.

        Returns:
            List of `num_outputs` tensors [B, target_dim] (zeros-initialized).
        """
        out = self.linear(condition.to(self.linear.weight.dtype)).chunk(self.num_outputs, dim=-1)
        if dtype is not None:
            out = [piece.to(dtype) for piece in out]
        return list(out)


class PooledConditionEmbedding(nn.Module):
    """Fuses timestep, pooled text, and pooled visual vectors into one global vector."""

    def __init__(
        self,
        timestep_dim: int,
        text_dim: int,
        visual_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize the fusion MLP.

        Parameters:
            timestep_dim: Timestep embedding width.
            text_dim: Pooled CLIP text width.
            visual_dim: Mean-pooled visual feature width.
            output_dim: Fused width fed to adaLN heads.
        """
        super().__init__()
        self.input_dim = timestep_dim + text_dim + visual_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        timestep_embedding: torch.Tensor,
        pooled_text: torch.Tensor,
        pooled_visual: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse conditioning vectors.

        Parameters:
            timestep_embedding: [B, timestep_dim].
            pooled_text: [B, text_dim].
            pooled_visual: [B, visual_dim].

        Returns:
            [B, output_dim] fused condition.
        """
        parts = [
            timestep_embedding.reshape(timestep_embedding.shape[0], -1),
            pooled_text.reshape(pooled_text.shape[0], -1),
            pooled_visual.reshape(pooled_visual.shape[0], -1),
        ]
        return self.net(torch.cat(parts, dim=-1))
