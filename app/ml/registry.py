"""Declarative variant registry: architecture hyperparameters and weight artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["REGISTRY", "WEIGHT_ROLES", "VariantSpec", "get_variant"]

WEIGHT_ROLES = (
    "clip_visual",
    "clip_text",
    "clip_tokenizer_vocab",
    "clip_tokenizer_merges",
    "sync_encoder",
    "vae",
    "vocoder",
    "generator",
    "null_embeds",
)


@dataclass(frozen=True, slots=True)
class MMDITSpec:
    """Generator transformer hyperparameters.

    Attributes:
        d_model: Token width.
        num_heads: Attention heads.
        joint_blocks: Number of two-stream joint MMDiT blocks.
        single_blocks: Number of single-stream audio blocks.
        text_dim: CLIP text embedding width.
        visual_dim: CLIP visual embedding width.
        sync_dim: Sync encoder output width.
        max_text_tokens: Tokenizer context length.
    """

    d_model: int
    num_heads: int
    joint_blocks: int
    single_blocks: int
    text_dim: int = 512
    visual_dim: int = 512
    sync_dim: int = 384
    max_text_tokens: int = 77


@dataclass(frozen=True, slots=True)
class VAESpec:
    """Audio mel-VAE hyperparameters.

    Attributes:
        base_channels: Conv width after the input projection.
        multipliers: Channel multipliers per downsample stage.
        latent_channels: Bottleneck channels.
        latent_hop: Temporal downsample factor over mel frames.
        num_res_blocks: Residual blocks per stage.
    """

    base_channels: int
    multipliers: tuple[int, ...]
    latent_channels: int
    latent_hop: int
    num_res_blocks: int


@dataclass(frozen=True, slots=True)
class VocoderSpec:
    """Neural vocoder hyperparameters.

    Attributes:
        upsample_rates: Per-stage upsampling factors; product == mel hop.
        upsample_kernel: Base kernel multiplier (kernel = 2 * rate).
        res_kernel_sizes: Dilated kernel sizes inside residual blocks.
        base_channels: Model width.
        num_mrf_blocks: Multi-receptive-field block count.
    """

    upsample_rates: tuple[int, ...]
    upsample_kernel: int = 2
    res_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    base_channels: int = 512
    num_mrf_blocks: int = 3


@dataclass(frozen=True, slots=True)
class SyncEncoderSpec:
    """Sync encoder hyperparameters.

    Attributes:
        dim: Feature width.
        depth: Transformer blocks over the 16-frame time axis.
        heads: Attention heads.
        patch: Conv3d kernel (temporal, spatial, spatial).
        segment_frames: Frames per segment (16).
        segment_stride: Segment stride in frames (8).
        out_per_segment: Tokens emitted per segment (8 -> 25 fps at 25 fps input).
    """

    dim: int
    depth: int
    heads: int
    patch: tuple[int, int, int] = (2, 16, 16)
    segment_frames: int = 16
    segment_stride: int = 8
    out_per_segment: int = 8


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Complete specification of one model variant.

    Attributes:
        name: Registry key.
        sample_rate: Audio sample rate in Hz.
        n_fft / hop_length / win_length: STFT geometry.
        n_mels: Mel bands.
        fmin / fmax: Mel edges.
        latent_fps: Latent frames per second after VAE downsampling.
        mel_clip_value: Log-mel clip floor (matches training).
        mmdit: Generator spec.
        vae: VAE spec.
        vocoder: Vocoder spec.
        sync: Sync encoder spec.
        window_s: Native generation window.
        overlap_s: Window crossfade.
        licence: Attribution note for derived weights.
        files: role -> filename inside WEIGHTS_DIR.
    """

    name: str
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    fmin: float
    fmax: float
    latent_fps: float
    mel_clip_value: float
    mmdit: MMDITSpec
    vae: VAESpec
    vocoder: VocoderSpec
    sync: SyncEncoderSpec
    window_s: float
    overlap_s: float
    licence: str
    files: dict[str, str] = field(default_factory=dict)

    @property
    def mel_fps(self) -> float:
        """Return mel frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def file_list(self) -> list[str]:
        """Return the weight filenames required by this variant."""
        return list(self.files.values())


def _variant_files(variant: str) -> dict[str, str]:
    """Build the per-variant weight filename map.

    Parameters:
        variant: Variant name.

    Returns:
        Mapping of role to filename.
    """
    return {
        "clip_visual": f"{variant}/clip_visual.safetensors",
        "clip_text": f"{variant}/clip_text.safetensors",
        "clip_tokenizer_vocab": "shared/clip_bpe_vocab.json",
        "clip_tokenizer_merges": "shared/clip_bpe_merges.txt",
        "sync_encoder": f"{variant}/sync_encoder.safetensors",
        "vae": f"{variant}/vae.safetensors",
        "vocoder": f"{variant}/vocoder.safetensors",
        "generator": f"{variant}/generator.safetensors",
        "null_embeds": f"{variant}/null_embeds.safetensors",
    }


_CLIP_LICENCE = (
    "CLIP ViT-B/16 towers derive from OpenAI CLIP weights (MIT); exported via "
    "open_clip_torch with key remapping. Remaining components are trained with "
    "this repository's training pipeline on FSD50K (CC-BY-4.0) and VGGSound "
    "(audio from YouTube; see dataset terms) and are MIT-licensed as trained."
)

_REGISTRY: dict[str, VariantSpec] = {
    "small_16k": VariantSpec(
        name="small_16k",
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        win_length=640,
        n_mels=64,
        fmin=0.0,
        fmax=8000.0,
        latent_fps=25.0,
        mel_clip_value=1e-5,
        mmdit=MMDITSpec(d_model=512, num_heads=8, joint_blocks=8, single_blocks=4),
        vae=VAESpec(
            base_channels=128,
            multipliers=(1, 2, 4),
            latent_channels=32,
            latent_hop=4,
            num_res_blocks=2,
        ),
        vocoder=VocoderSpec(
            upsample_rates=(5, 4, 4, 2),
            res_kernel_sizes=(3, 7, 11),
            base_channels=384,
            num_mrf_blocks=3,
        ),
        sync=SyncEncoderSpec(dim=384, depth=4, heads=6),
        window_s=10.0,
        overlap_s=1.0,
        licence=_CLIP_LICENCE,
        files=_variant_files("small_16k"),
    ),
    "medium_44k": VariantSpec(
        name="medium_44k",
        sample_rate=44100,
        n_fft=2048,
        hop_length=441,
        win_length=1764,
        n_mels=96,
        fmin=0.0,
        fmax=22050.0,
        latent_fps=25.0,
        mel_clip_value=1e-5,
        mmdit=MMDITSpec(d_model=768, num_heads=12, joint_blocks=16, single_blocks=6),
        vae=VAESpec(
            base_channels=192,
            multipliers=(1, 2, 4),
            latent_channels=64,
            latent_hop=4,
            num_res_blocks=2,
        ),
        vocoder=VocoderSpec(
            upsample_rates=(7, 7, 3, 3),
            res_kernel_sizes=(3, 7, 11, 15),
            base_channels=512,
            num_mrf_blocks=3,
        ),
        sync=SyncEncoderSpec(dim=384, depth=4, heads=6),
        window_s=10.0,
        overlap_s=1.0,
        licence=_CLIP_LICENCE,
        files=_variant_files("medium_44k"),
    ),
    "large_44k": VariantSpec(
        name="large_44k",
        sample_rate=44100,
        n_fft=2048,
        hop_length=441,
        win_length=1764,
        n_mels=96,
        fmin=0.0,
        fmax=22050.0,
        latent_fps=25.0,
        mel_clip_value=1e-5,
        mmdit=MMDITSpec(d_model=1024, num_heads=16, joint_blocks=24, single_blocks=8),
        vae=VAESpec(
            base_channels=256,
            multipliers=(1, 2, 4, 4),
            latent_channels=64,
            latent_hop=4,
            num_res_blocks=3,
        ),
        vocoder=VocoderSpec(
            upsample_rates=(7, 7, 3, 3),
            res_kernel_sizes=(3, 7, 11, 15),
            base_channels=768,
            num_mrf_blocks=4,
        ),
        sync=SyncEncoderSpec(dim=512, depth=6, heads=8),
        window_s=10.0,
        overlap_s=1.0,
        licence=_CLIP_LICENCE,
        files=_variant_files("large_44k"),
    ),
}

REGISTRY = _REGISTRY


def get_variant(name: str) -> VariantSpec:
    """Look up a variant spec by name.

    Parameters:
        name: Variant key.

    Returns:
        The VariantSpec.

    Raises:
        KeyError: When the variant is unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model variant {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
