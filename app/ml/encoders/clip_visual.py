"""In-repo CLIP ViT-B/16 image tower with OpenAI-CLIP-compatible weights."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

__all__ = ["CLIP_NORMALIZE_MEAN", "CLIP_NORMALIZE_STD", "CLIPVisualEncoder", "ViTLayer"]

CLIP_NORMALIZE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORMALIZE_STD = (0.26862954, 0.26130258, 0.27577711)


class ViTLayer(nn.Module):
    """Pre-norm ViT block matching open_clip's ResidualAttentionBlock."""

    def __init__(self, d_model: int, n_head: int) -> None:
        """Initialize the block.

        Parameters:
            d_model: Width.
            n_head: Heads.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=False)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention + MLP.

        Parameters:
            x: [T+1, B, D] patch + class tokens.

        Returns:
            Updated tokens.
        """
        normed = self.ln_1(x)
        attn, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn
        return x + self.mlp(self.ln_2(x))


class _VisualTransformer(nn.Module):
    """Patch embedding + transformer stack (open_clip visual naming)."""

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int) -> None:
        """Build the stack.

        Parameters:
            input_resolution: Input square size.
            patch_size: Patch size.
            width: Width.
            layers: Depth.
            heads: Heads.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(
            3, width, kernel_size=patch_size, stride=patch_size, bias=False
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.resblocks = nn.Sequential(*[ViTLayer(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify and run the stack.

        Parameters:
            x: [B, 3, H, W] normalized images.

        Returns:
            [B, T, D] pre-pool tokens.
        """
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(2, 0, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[1], 1, x.shape[-1], device=x.device, dtype=x.dtype
            ), x],
            dim=0,
        )
        x = x + self.positional_embedding.to(x.dtype)[:, None, :]
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        return x.permute(1, 0, 2)


class CLIPVisualEncoder(nn.Module):
    """CLIP ViT-B/16 image tower producing per-frame embeddings.

    Naming (visual conv1/class_embedding/.../ln_post/proj) is nested under the
    `visual.` prefix expected in exported checkpoints.
    """

    def __init__(
        self,
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512,
    ) -> None:
        """Build the tower.

        Parameters:
            input_resolution: Input image size.
            patch_size: Patch size.
            width: Transformer width.
            layers: Depth.
            heads: Heads.
            output_dim: Projection width.
        """
        super().__init__()
        self.visual = _VisualTransformer(input_resolution, patch_size, width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, output_dim))
        nn.init.normal_(self.proj, std=width**-0.5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Parameters:
            images: [B, 3, H, W] CLIP-normalized images.

        Returns:
            [B, output_dim] class-token projections (one per frame).
        """
        tokens = self.visual(images)
        class_token = tokens[:, 0, :]
        class_token = self.ln_post(class_token)
        return class_token @ self.proj.to(class_token.dtype)

    @torch.no_grad()
    def encode_frames(self, frames_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Normalize uint8 frames on device and encode them.

        Parameters:
            frames_uint8: [T, 3, H, W] uint8 tensor.
            device: Compute device.

        Returns:
            [T, output_dim] float32 embeddings.
        """
        self.eval()
        frames = frames_uint8.to(device).to(torch.float32) / 255.0
        mean = torch.tensor(CLIP_NORMALIZE_MEAN, device=device).reshape(1, 3, 1, 1)
        std = torch.tensor(CLIP_NORMALIZE_STD, device=device).reshape(1, 3, 1, 1)
        frames = (frames - mean) / std
        outputs: list[torch.Tensor] = []
        batch = 32
        for start in range(0, frames.shape[0], batch):
            outputs.append(self.forward(frames[start : start + batch]).float().cpu())
        return torch.cat(outputs, dim=0)

    @staticmethod
    def patch_token_count(input_resolution: int, patch_size: int) -> int:
        """Return the number of patch tokens for an input size.

        Parameters:
            input_resolution: Image size.
            patch_size: Patch size.

        Returns:
            (resolution // patch)^2.
        """
        return (input_resolution // patch_size) ** 2
