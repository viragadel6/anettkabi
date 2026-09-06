"""The MMDiT generator network over audio latents with multimodal conditioning."""

from __future__ import annotations

import torch
from torch import nn

from app.ml.generator.embeddings import (
    ModalityEmbedding,
    PooledConditionEmbedding,
    TimestepEmbedding,
)
from app.ml.generator.layers import ATTENTION_BACKENDS, JointMMDITBlock, SingleStreamBlock
from app.ml.generator.rope import RotaryPositionEmbedding, sinusoidal_positions
from app.ml.registry import MMDITSpec

__all__ = ["MODALITY_IDS", "MMDiTGenerator"]

MODALITY_IDS = {"audio": 0, "text": 1, "visual": 2, "sync": 3, "memory": 4}
NUM_MODALITIES = 5


class MMDiTGenerator(nn.Module):
    """Conditional flow-matching transformer predicting latent velocities.

    Streams:
      * audio latent tokens (the denoised stream, length T_a)
      * context tokens = CLIP text tokens + visual tokens + sync tokens
        (+ optional memory tokens from a previous window's latents)

    N joint two-stream blocks attend over the concatenation; M single-stream
    blocks refine the audio tokens. Global conditioning (timestep + pooled text
    + pooled visual) drives adaLN-Zero modulation.
    """

    def __init__(
        self,
        spec: MMDITSpec,
        latent_channels: int,
        latent_fps: float = 25.0,
        backend: str = "sdpa",
    ) -> None:
        """Build the network.

        Parameters:
            spec: MMDIT hyperparameters.
            latent_channels: VAE latent channel count.
            latent_fps: Latent frames per second for the audio time base.
            backend: Attention backend string.
        """
        super().__init__()
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(f"unsupported attention backend {backend!r}")
        self.spec = spec
        self.latent_fps = latent_fps
        dim = spec.d_model
        self.dim = dim
        self.backend = backend
        self.latent_in = nn.Linear(latent_channels, dim)
        self.latent_out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.latent_out = nn.Linear(dim, latent_channels)
        nn.init.zeros_(self.latent_out.weight)
        nn.init.zeros_(self.latent_out.bias)
        self.text_in = nn.Linear(spec.text_dim, dim)
        self.visual_in = nn.Linear(spec.visual_dim, dim)
        self.sync_in = nn.Linear(spec.sync_dim, dim)
        self.memory_in = nn.Linear(latent_channels, dim)
        self.modality = ModalityEmbedding(dim, NUM_MODALITIES)
        self.timestep = TimestepEmbedding(dim)
        self.pooled_text_in = nn.Linear(spec.text_dim, spec.text_dim)
        self.pooled_visual_in = nn.Linear(spec.visual_dim, spec.visual_dim)
        self.global_cond = PooledConditionEmbedding(dim, spec.text_dim, spec.visual_dim, dim)
        self.joint_blocks = nn.ModuleList(
            [JointMMDITBlock(dim, spec.num_heads, backend) for _ in range(spec.joint_blocks)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(dim, spec.num_heads, backend) for _ in range(spec.single_blocks)]
        )
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.rope = RotaryPositionEmbedding(spec.d_model // spec.num_heads)
        self.text_positions_cache: torch.Tensor | None = None

    def _text_positions(self, length: int, device: torch.device) -> torch.Tensor:
        """Return (cached) sinusoidal positions for text tokens.

        Parameters:
            length: Token count.
            device: Target device.

        Returns:
            [length, dim] position table.
        """
        if (
            self.text_positions_cache is None
            or self.text_positions_cache.shape[0] < length
            or self.text_positions_cache.device != device
        ):
            self.text_positions_cache = sinusoidal_positions(
                max(length, self.spec.max_text_tokens), self.dim, device
            )
        return self.text_positions_cache[:length]

    def _rope_tables(
        self, seconds: torch.Tensor, fps: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ROPE cos/sin for a timestamp vector.

        Parameters:
            seconds: [T] timestamps.
            fps: Native rate of the stream.
            device: Target device.

        Returns:
            (cos, sin) shaped [1, 1, T, head_dim].
        """
        from app.ml.generator.rope import rope_time_positions

        positions = rope_time_positions(seconds, fps).to(device)
        cos, sin = self.rope(positions)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor,
        pooled_text: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_seconds: torch.Tensor,
        sync_tokens: torch.Tensor,
        sync_seconds: torch.Tensor,
        visual_fps: float,
        sync_fps: float,
        pooled_visual: torch.Tensor,
        memory_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow velocity field.

        Parameters:
            x_t: [B, T_a, C] noisy latents.
            t: [B] flow time in [0, 1].
            text_tokens: [B, 77, text_dim] CLIP token features.
            pooled_text: [B, text_dim] pooled text embedding.
            visual_tokens: [B, T_v, visual_dim] CLIP visual features.
            visual_seconds: [T_v] timestamps of visual tokens.
            sync_tokens: [B, T_s, sync_dim] sync features.
            sync_seconds: [T_s] timestamps of sync tokens.
            pooled_visual: [B, visual_dim] mean-pooled visual embedding.
            memory_tokens: Optional [B, T_m, C] previous-window latent tokens.

        Returns:
            [B, T_a, C] predicted velocity.
        """
        batch = x_t.shape[0]
        device = x_t.device
        audio = self.latent_in(x_t)
        text = self.text_in(text_tokens)
        visual = self.visual_in(visual_tokens)
        sync = self.sync_in(sync_tokens)
        pieces = [text, visual, sync]
        audio = audio + self.modality(MODALITY_IDS["audio"], batch, audio.shape[1], device)
        text = text + self.modality(MODALITY_IDS["text"], batch, text.shape[1], device)
        visual = visual + self.modality(MODALITY_IDS["visual"], batch, visual.shape[1], device)
        sync = sync + self.modality(MODALITY_IDS["sync"], batch, sync.shape[1], device)
        if memory_tokens is not None and memory_tokens.shape[1] > 0:
            memory = self.memory_in(memory_tokens)
            memory = memory + self.modality(MODALITY_IDS["memory"], batch, memory.shape[1], device)
            pieces.append(memory)
        context = torch.cat(pieces, dim=1)
        timestep_embedding = self.timestep(t)
        global_condition = self.global_cond(
            timestep_embedding,
            self.pooled_text_in(pooled_text),
            self.pooled_visual_in(pooled_visual),
        )
        audio_seconds = (
            torch.arange(x_t.shape[1], device=device, dtype=torch.float32) / self.latent_fps
        )
        cos_a, sin_a = self._rope_tables(audio_seconds, 1.0, device)
        context_seconds = torch.cat(
            [
                torch.arange(text.shape[1], device=device, dtype=torch.float32),
                visual_seconds.to(device),
                sync_seconds.to(device),
            ]
        )
        if memory_tokens is not None and memory_tokens.shape[1] > 0:
            context_seconds = torch.cat(
                [context_seconds, torch.zeros(memory_tokens.shape[1], device=device)]
            )
        cos_c, sin_c = self._rope_tables(context_seconds, 1.0, device)
        for block in self.joint_blocks:
            audio, context = block(
                audio,
                context,
                global_condition,
                global_condition,
                cos_x=cos_a,
                sin_x=sin_a,
                cos_c=cos_c,
                sin_c=sin_c,
            )
        for block in self.single_blocks:
            audio = block(audio, global_condition, cos=cos_a, sin=sin_a)
        shift, scale = self.final_adaLN(global_condition).chunk(2, dim=-1)
        audio = self.latent_out_norm(audio) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        velocity = self.latent_out(audio)
        return velocity.to(x_t.dtype)
