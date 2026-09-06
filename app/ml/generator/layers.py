"""Transformer building blocks: RMSNorm, QK-norm attention, SwiGLU, joint blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as fn
from torch import nn

from app.ml.generator.embeddings import AdaLNZeroModulation
from app.ml.generator.rope import apply_rope

__all__ = [
    "ATTENTION_BACKENDS",
    "JointMMDITBlock",
    "QKNormalizedAttention",
    "RMSNorm",
    "SingleStreamBlock",
    "SwiGLU",
]


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    """Scaled dot-product attention over [B, H, T, D] tensors.

    Parameters:
        q, k, v: Query/key/value tensors.
        backend: One of `sdpa`, `flash`, `math`.

    Returns:
        The attention output [B, H, T, D].
    """
    if backend == "sdpa" or backend == "flash":
        dropout_p = 0.0
        return fn.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn = scores.softmax(dim=-1)
    return torch.matmul(attn, v)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm without mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the norm.

        Parameters:
            dim: Feature width.
            eps: Numerical epsilon.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last axis.

        Parameters:
            x: Any-shape tensor.

        Returns:
            The normalized tensor.
        """
        norm = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight.to(x.dtype)


class QKNormalizedAttention(nn.Module):
    """Multi-head self-attention with RMS-normalized Q/K and packed projections."""

    def __init__(self, dim: int, num_heads: int, backend: str = "sdpa") -> None:
        """Initialize attention.

        Parameters:
            dim: Token width.
            num_heads: Head count (dim must divide evenly).
            backend: Attention kernel selection.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.backend = backend if backend in ATTENTION_BACKENDS else "sdpa"

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attention with optional rotary embeddings.

        Parameters:
            x: [B, T, D] tokens.
            cos/sin: Optional ROPE tables.

        Returns:
            [B, T, D] output tokens.
        """
        batch, length, dim = x.shape
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = _attention(q, k, v, self.backend)
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        """Initialize the MLP.

        Parameters:
            dim: Input width.
            expansion: Hidden width multiplier.
        """
        super().__init__()
        hidden = dim * expansion
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated MLP.

        Parameters:
            x: [B, T, D].

        Returns:
            [B, T, D].
        """
        return self.down_proj(fn.silu(self.gate_proj(x)) * self.up_proj(x))


class SingleStreamBlock(nn.Module):
    """adaLN-Zero transformer block with optional RoPE self-attention."""

    def __init__(self, dim: int, num_heads: int, backend: str = "sdpa", mlp_expansion: int = 4) -> None:
        """Initialize the block.

        Parameters:
            dim: Token width.
            num_heads: Heads.
            backend: Attention backend.
            mlp_expansion: SwiGLU expansion.
        """
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = QKNormalizedAttention(dim, num_heads, backend)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_expansion)
        self.adaLN = AdaLNZeroModulation(dim, dim, num_outputs=6)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one block.

        Parameters:
            x: [B, T, D] audio tokens.
            condition: [B, D_cond] global condition.
            cos/sin: Optional ROPE tables.

        Returns:
            [B, T, D] updated tokens.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            condition, dtype=x.dtype
        )
        normed = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn = self.attn(normed, cos, sin)
        x = x + gate_msa.unsqueeze(1) * attn
        normed = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp = self.mlp(normed)
        return x + gate_mlp.unsqueeze(1) * mlp


class JointMMDITBlock(nn.Module):
    """Two-stream MMDiT block: audio and context attend jointly, modulated separately."""

    def __init__(self, dim: int, num_heads: int, backend: str = "sdpa") -> None:
        """Initialize the joint block.

        Parameters:
            dim: Shared token width of both streams.
            num_heads: Heads.
            backend: Attention backend.
        """
        super().__init__()
        self.norm1_x = RMSNorm(dim)
        self.norm1_c = RMSNorm(dim)
        self.attn_x = QKNormalizedAttention(dim, num_heads, backend)
        self.attn_c = QKNormalizedAttention(dim, num_heads, backend)
        self.norm2_x = RMSNorm(dim)
        self.norm2_c = RMSNorm(dim)
        self.mlp_x = SwiGLU(dim)
        self.mlp_c = SwiGLU(dim)
        self.adaLN_x = AdaLNZeroModulation(dim, dim, 6)
        self.adaLN_c = AdaLNZeroModulation(dim, dim, 6)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cond_x: torch.Tensor,
        cond_c: torch.Tensor,
        cos_x: torch.Tensor | None = None,
        sin_x: torch.Tensor | None = None,
        cos_c: torch.Tensor | None = None,
        sin_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the joint block.

        Attention operates over the concatenation of audio (x) and context (c)
        tokens so every token sees every other token (joint attention), while
        value/output projections and modulation remain per-stream.

        Parameters:
            x: [B, Tx, D] audio tokens.
            c: [B, Tc, D] context tokens (text + visual/sync).
            cond_x / cond_c: [B, D] per-stream global conditions.
            cos_x/sin_x/cos_c/sin_c: Optional per-stream ROPE tables.

        Returns:
            (updated x, updated c).
        """
        shift_x, scale_x, gate_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_x(
            cond_x, dtype=x.dtype
        )
        shift_c, scale_c, gate_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_c(
            cond_c, dtype=c.dtype
        )
        normed_x = self.norm1_x(x) * (1 + scale_x.unsqueeze(1)) + shift_x.unsqueeze(1)
        normed_c = self.norm1_c(c) * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)
        q_x = self.attn_x.in_proj(normed_x).chunk(3, dim=-1)
        q_c = self.attn_c.in_proj(normed_c).chunk(3, dim=-1)
        heads = self.attn_x.num_heads
        head_dim = self.attn_x.head_dim
        batch = x.shape[0]

        def _heads_qkv(parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor], length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = parts
            q = q.view(batch, length, heads, head_dim).transpose(1, 2)
            k = k.view(batch, length, heads, head_dim).transpose(1, 2)
            v = v.view(batch, length, heads, head_dim).transpose(1, 2)
            return self.attn_x.q_norm(q), self.attn_x.k_norm(k), v

        q_xh, k_xh, v_xh = _heads_qkv(q_x, x.shape[1])
        q_ch, k_ch, v_ch = _heads_qkv(q_c, c.shape[1])
        if cos_x is not None and sin_x is not None:
            q_xh = apply_rope(q_xh, cos_x, sin_x)
            k_xh = apply_rope(k_xh, cos_x, sin_x)
        if cos_c is not None and sin_c is not None:
            q_ch = apply_rope(q_ch, cos_c, sin_c)
            k_ch = apply_rope(k_ch, cos_c, sin_c)
        q_all = torch.cat([q_xh, q_ch], dim=2)
        k_all = torch.cat([k_xh, k_ch], dim=2)
        v_all = torch.cat([v_xh, v_ch], dim=2)
        out_all = _attention(q_all, k_all, v_all, self.attn_x.backend)
        out_x = out_all[:, :, : x.shape[1]]
        out_c = out_all[:, :, x.shape[1] :]
        out_x = out_x.transpose(1, 2).reshape(batch, x.shape[1], -1)
        out_c = out_c.transpose(1, 2).reshape(batch, c.shape[1], -1)
        x = x + gate_x.unsqueeze(1) * self.attn_x.out_proj(out_x)
        c = c + gate_c.unsqueeze(1) * self.attn_c.out_proj(out_c)
        normed_x = self.norm2_x(x) * (1 + scale_mlp_x.unsqueeze(1)) + shift_mlp_x.unsqueeze(1)
        normed_c = self.norm2_c(c) * (1 + scale_mlp_c.unsqueeze(1)) + shift_mlp_c.unsqueeze(1)
        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(normed_x)
        c = c + gate_mlp_c.unsqueeze(1) * self.mlp_c(normed_c)
        return x, c


ATTENTION_BACKENDS = frozenset({"sdpa", "flash", "math"})
