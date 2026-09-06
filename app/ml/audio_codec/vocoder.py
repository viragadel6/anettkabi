"""BigVGAN-style neural vocoder: MRF residual blocks, weight norm, chunked decode."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm as weight_norm_param

from app.ml.registry import VocoderSpec

__all__ = ["AntiAliasedActivation", "BigVGANVocoder", "MRFBlock"]


def _norm(conv: nn.Conv1d) -> nn.Conv1d:
    """Apply parametrized weight norm to a conv.

    Parameters:
        conv: Convolution layer.

    Returns:
        The wrapped convolution.
    """
    return weight_norm_param(conv)


class AntiAliasedActivation(nn.Module):
    """Low-pass filtered LeakyReLU (BigVGAN anti-aliased nonlinearity)."""

    def __init__(self, channels: int) -> None:
        """Initialize the smoothing kernel.

        Parameters:
            channels: Channel count for the depthwise filter.
        """
        super().__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.lowpass = nn.Conv1d(
            channels, channels, kernel_size=12, padding=5, groups=channels, bias=False
        )
        nn.init.constant_(self.lowpass.weight, 1.0 / 12.0)
        self.lowpass.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation then smoothing.

        Parameters:
            x: [B, C, T].

        Returns:
            Filtered activations.
        """
        return self.lowpass(self.activation(x))


class MRFBlock(nn.Module):
    """Multi-receptive-field residual block with parallel dilated branches."""

    def __init__(self, channels: int, kernel_sizes: tuple[int, ...], num_blocks: int) -> None:
        """Initialize parallel branches.

        Parameters:
            channels: Width.
            kernel_sizes: Kernel sizes (one per branch).
            num_blocks: Number of stacked convs per branch.
        """
        super().__init__()
        self.branches = nn.ModuleList()
        for kernel in kernel_sizes:
            layers: list[nn.Module] = []
            for block in range(num_blocks):
                dilation = 3**block
                padding = dilation * (kernel - 1) // 2
                conv = _norm(
                    nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=padding)
                )
                layers.append(conv)
                layers.append(AntiAliasedActivation(channels))
            self.branches.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum branch outputs.

        Parameters:
            x: [B, C, T].

        Returns:
            [B, C, T].
        """
        out = x
        for branch in self.branches:
            out = out + branch(x)
        return out


class BigVGANVocoder(nn.Module):
    """Mel-to-waveform upsampling generator.

    The product of `upsample_rates` equals the mel hop length; output length
    is exactly `mel_frames * hop`. Weight-norm parametrizations are removed by
    `prepare_inference()` after loading, mirroring BigVGAN's runtime export.
    """

    def __init__(self, spec: VocoderSpec, n_mels: int) -> None:
        """Build the vocoder.

        Parameters:
            spec: Vocoder hyperparameters.
            n_mels: Mel bands.
        """
        super().__init__()
        self.spec = spec
        self.hop = 1
        for rate in spec.upsample_rates:
            self.hop *= rate
        self.conv_pre = _norm(nn.Conv1d(n_mels, spec.base_channels, kernel_size=7, padding=3))
        self.upsamples = nn.ModuleList()
        channels = spec.base_channels
        for rate in spec.upsample_rates:
            kernel = spec.upsample_kernel * rate
            out_channels = max(64, channels // 2)
            self.upsamples.append(
                _norm(
                    nn.ConvTranspose1d(
                        channels, out_channels, kernel_size=kernel, stride=rate, padding=kernel // 2 - rate // 2
                    )
                )
            )
            channels = out_channels
        self.mrf = nn.ModuleList(
            [MRFBlock(channels, spec.res_kernel_sizes, spec.num_mrf_blocks) for _ in spec.upsample_rates]
        )
        self.activation = AntiAliasedActivation(channels)
        self.conv_post = _norm(nn.Conv1d(channels, 1, kernel_size=7, padding=3))
        self._inference_ready = False

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocode full or chunked mel to waveform.

        Parameters:
            mel: [B, n_mels, T].

        Returns:
            [B, 1, T * hop] waveform in [-1, 1].
        """
        hidden = self.conv_pre(mel)
        for upsample, mrf in zip(self.upsamples, self.mrf, strict=True):
            hidden = upsample(hidden)
            hidden = mrf(hidden) + hidden
        hidden = self.activation(hidden)
        return torch.tanh(self.conv_post(hidden))

    @torch.no_grad()
    def vocode_chunked(
        self,
        mel: torch.Tensor,
        *,
        chunk_frames: int = 64,
        overlap_frames: int = 8,
    ) -> torch.Tensor:
        """Decode long spectrograms in overlap-add chunks.

        Adjacent chunks cross-fade over `overlap_frames` mel frames using a
        raised-cosine window so stitching is phase-safe at the frame level.

        Parameters:
            mel: [1, n_mels, T] (single example).
            overlap_frames: Overlap in mel frames between chunks.
            chunk_frames: Working chunk size.

        Returns:
            [1, 1, T * hop] waveform.
        """
        if mel.shape[0] != 1:
            raise ValueError("chunked decode expects batch size 1")
        total_frames = mel.shape[-1]
        hop = self.hop
        if total_frames <= chunk_frames + overlap_frames:
            return self.forward(mel)
        out = torch.zeros(1, 1, total_frames * hop, device=mel.device, dtype=torch.float32)
        weight = torch.zeros(1, 1, total_frames * hop, device=mel.device, dtype=torch.float32)
        stride = chunk_frames - overlap_frames
        fade = torch.hann_window(overlap_frames * hop, device=mel.device, periodic=True) * 2.0
        start = 0
        while start < total_frames:
            end = min(total_frames, start + chunk_frames)
            piece = self.forward(mel[..., start:end])
            sample_start = start * hop
            sample_end = end * hop
            length = sample_end - sample_start
            w = torch.ones(length, device=mel.device, dtype=torch.float32)
            if start > 0:
                w[: overlap_frames * hop] = fade[: min(overlap_frames * hop, length)]
            if end < total_frames:
                tail = min(overlap_frames * hop, length)
                w[length - tail :] = fade[-tail:]
            out[..., sample_start:sample_end] += piece.reshape(1, -1)[..., :length] * w
            weight[..., sample_start:sample_end] += w
            if end == total_frames:
                break
            start += stride
        out = out / weight.clamp(min=1e-6)
        return out.clamp(-1.0, 1.0)

    def prepare_inference(self) -> None:
        """Remove weight-norm parametrizations for faster inference.

        Must be called after state loading; makes buffers contiguous.
        """
        if self._inference_ready:
            return
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.ConvTranspose1d):
                try:
                    parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                except ValueError:
                    continue
        self._inference_ready = True

    def output_length(self, mel_frames: int) -> int:
        """Waveform samples produced for mel frames.

        Parameters:
            mel_frames: Frame count.

        Returns:
            mel_frames * hop.
        """
        return mel_frames * self.hop
