"""In-repo CLIP text tower (ViT-B/16) with OpenAI-CLIP-compatible weights."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

__all__ = ["CLIPTextEncoder", "TransformerTextLayer"]


class TransformerTextLayer(nn.Module):
    """Pre-norm transformer layer matching open_clip's ResidualAttentionBlock."""

    def __init__(self, d_model: int, n_head: int) -> None:
        """Initialize the layer.

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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Run one layer with the CLIP causal mask.

        Parameters:
            x: [T, B, D] tokens.
            attn_mask: Additive causal mask.

        Returns:
            [T, B, D].
        """
        normed = self.ln_1(x)
        attn, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        x = x + attn
        return x + self.mlp(self.ln_2(x))


class CLIPTextEncoder(nn.Module):
    """CLIP ViT-B/16 text tower.

    Module naming (token_embedding, positional_embedding, ln_final,
    transformer.resblocks.N, text_projection) mirrors open_clip so exported
    checkpoints load with strict=True and no remapping.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        context_length: int = 77,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        output_dim: int = 512,
    ) -> None:
        """Build the tower.

        Parameters:
            vocab_size: Tokenizer vocabulary size.
            embed_dim: Token embedding width.
            context_length: Token context.
            width: Transformer width.
            layers: Depth.
            heads: Attention heads.
            output_dim: Projection output width.
        """
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, embed_dim))
        nn.init.normal_(self.positional_embedding, std=0.02)
        class _Transformer(nn.Module):
            """Container preserving open_clip's transformer.resblocks.N naming."""

            def __init__(self, depth: int) -> None:
                """Build the resblock stack.

                Parameters:
                    depth: Number of layers.
                """
                super().__init__()
                self.resblocks = nn.Sequential(
                    *[TransformerTextLayer(width, heads) for _ in range(depth)]
                )

            def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
                """Run all resblocks.

                Parameters:
                    x: [T, B, D].
                    attn_mask: Causal mask.

                Returns:
                    [T, B, D].
                """
                for resblock in self.resblocks:
                    x = resblock(x, attn_mask)
                return x

        self.transformer = _Transformer(layers)
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        nn.init.normal_(self.text_projection, std=width**-0.5)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to per-token features and the pooled embedding.

        Parameters:
            tokens: [B, context_length] long tensor.

        Returns:
            (token_features [B, T, output_dim], pooled [B, output_dim]).

        Raises:
            ValueError: When the token length mismatches the context.
        """
        if tokens.shape[1] != self.context_length:
            raise ValueError(
                f"expected {self.context_length} tokens, got {tokens.shape[1]}"
            )
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        mask = self.attn_mask.to(x.dtype)
        x = self.transformer(x, mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot_positions = tokens.argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), eot_positions]
        if self.text_projection is not None:
            pooled = pooled @ self.text_projection.to(pooled.dtype)
        return x, pooled

    @torch.no_grad()
    def encode_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference wrapper applying no-grad + eval semantics.

        Parameters:
            tokens: [B, T] ids.

        Returns:
            (token features, pooled embedding).
        """
        self.eval()
        tokens_features, pooled = self.forward(tokens)
        return tokens_features.float(), pooled.float()
