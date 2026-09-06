"""Mel-spectrogram VAE: encoder/decoder with fixed latent scaling constants."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from app.ml.registry import VAESpec

__all__ = ["AudioVAE", "ResidualBlock1D"]


class ResidualBlock1D(nn.Module):
    """Dilated residual block over the mel-time axis."""

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 3) -> None:
        """Initialize the block.

        Parameters:
            channels: Width.
            dilation: Conv dilation.
            kernel_size: Conv kernel.
        """
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=kernel_size // 2),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual mapping.

        Parameters:
            x: [B, C, T].

        Returns:
            [B, C, T].
        """
        return x + self.act(self.block(x))


class AudioVAE(nn.Module):
    """Convolutional VAE mapping mel spectrograms to compact latent sequences.

    The encoder downsamples the time axis by `latent_hop` via strided convs;
    the decoder mirrors it with transposed convs. A fixed scale/shift pair
    (loaded from the checkpoint buffers) normalizes latents to unit scale for
    the flow model.
    """

    def __init__(self, spec: VAESpec, n_mels: int) -> None:
        """Build the VAE.

        Parameters:
            spec: VAE hyperparameters.
            n_mels: Input mel bands.
        """
        super().__init__()
        self.spec = spec
        self.latent_hop = spec.latent_hop
        self.latent_channels = spec.latent_channels
        self.input_proj = nn.Conv1d(n_mels, spec.base_channels, kernel_size=7, padding=3)
        stages = len(spec.multipliers)
        if spec.latent_hop != 2**stages and spec.latent_hop % (2**stages) != 0:
            raise ValueError(
                f"latent_hop {spec.latent_hop} not reachable with {stages} stride-2 stages"
            )
        self.extra_stride = spec.latent_hop // 2**stages
        encoder_layers: list[nn.Module] = []
        in_ch = spec.base_channels
        for multiplier in spec.multipliers:
            out_ch = spec.base_channels * multiplier
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                )
            )
            for _ in range(spec.num_res_blocks):
                encoder_layers.append(ResidualBlock1D(out_ch))
            in_ch = out_ch
        if self.extra_stride > 1:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, in_ch, kernel_size=self.extra_stride * 2, stride=self.extra_stride, padding=self.extra_stride // 2),
                    nn.GroupNorm(8, in_ch),
                    nn.SiLU(),
                )
            )
        self.encoder_body = nn.Sequential(*encoder_layers)
        self.to_mean = nn.Conv1d(in_ch, spec.latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(in_ch, spec.latent_channels, kernel_size=1)
        self.from_latent = nn.Conv1d(spec.latent_channels, in_ch, kernel_size=1)
        decoder_layers: list[nn.Module] = []
        if self.extra_stride > 1:
            decoder_layers.append(
                nn.ConvTranspose1d(
                    in_ch, in_ch, kernel_size=self.extra_stride * 2, stride=self.extra_stride, padding=self.extra_stride // 2
                )
            )
        for stage in range(len(spec.multipliers) - 1, -1, -1):
            out_ch = spec.base_channels * spec.multipliers[stage]
            next_ch = spec.base_channels * spec.multipliers[stage - 1] if stage > 0 else spec.base_channels
            decoder_layers.append(
                nn.ConvTranspose1d(out_ch, next_ch, kernel_size=4, stride=2, padding=1)
            )
            decoder_layers.append(nn.GroupNorm(8, next_ch))
            decoder_layers.append(nn.SiLU())
            for _ in range(spec.num_res_blocks):
                decoder_layers.append(ResidualBlock1D(next_ch))
        self.decoder_body = nn.Sequential(*decoder_layers)
        self.out_proj = nn.Conv1d(spec.base_channels, n_mels, kernel_size=7, padding=3)
        self.register_buffer("latent_mean", torch.zeros(1), persistent=True)
        self.register_buffer("latent_scale", torch.ones(1), persistent=True)

    def encode(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode log-mel to (mean, logvar) latents.

        Parameters:
            mel: [B, n_mels, T] log-mel.

        Returns:
            (mean, logvar) each [B, C, T'].
        """
        hidden = self.encoder_body(F.silu(self.input_proj(mel)))
        return self.to_mean(hidden), self.to_logvar(hidden)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample with the reparameterization trick.

        Parameters:
            mean: Posterior mean.
            logvar: Posterior log variance.

        Returns:
            Sampled latents.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)
            return mean + std * noise
        return mean

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        """Scale raw latents to unit scale for the flow model.

        Parameters:
            latent: Raw latents.

        Returns:
            Normalized latents.
        """
        scale = self.latent_scale.to(latent.dtype).clamp(min=1e-6)
        return (latent - self.latent_mean.to(latent.dtype)) / scale

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        """Invert `normalize`.

        Parameters:
            latent: Normalized latents.

        Returns:
            Raw-scale latents.
        """
        scale = self.latent_scale.to(latent.dtype)
        return latent * scale + self.latent_mean.to(latent.dtype)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents back to log-mel.

        Parameters:
            latent: [B, C, T'] raw-scale latents.

        Returns:
            [B, n_mels, T] log-mel reconstruction.
        """
        hidden = self.decoder_body(self.from_latent(latent))
        return self.out_proj(hidden)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode/reparameterize/decode pass (training path).

        Parameters:
            mel: [B, n_mels, T].

        Returns:
            (reconstruction, mean, logvar).
        """
        mean, logvar = self.encode(mel)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    @staticmethod
    def vae_loss(
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        *,
        kl_weight: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Composite reconstruction + KL objective.

        Parameters:
            reconstruction: Decoded log-mel.
            target: Original log-mel.
            mean / logvar: Posterior parameters.
            kl_weight: KL contribution weight.

        Returns:
            (loss, parts dict of detached scalars).
        """
        recon = F.mse_loss(reconstruction, target)
        kl = -0.5 * torch.mean(
            torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        )
        total = recon + kl_weight * kl
        return total, {
            "recon": float(recon.detach()),
            "kl": float(kl.detach()),
            "total": float(total.detach()),
        }

    def latent_length(self, mel_frames: int) -> int:
        """Latent frames for a mel length.

        Parameters:
            mel_frames: Mel frame count.

        Returns:
            Downsampled length (>= 1).
        """
        return max(1, mel_frames // self.latent_hop)

    def mel_length(self, latent_frames: int) -> int:
        """Mel frames reconstructed from latent frames.

        Parameters:
            latent_frames: Latent count.

        Returns:
            Upsampled mel length.
        """
        return latent_frames * self.latent_hop
