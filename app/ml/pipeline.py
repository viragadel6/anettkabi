"""VideoToSfxPipeline: the single orchestrating class for end-to-end generation."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.config import get_settings
from app.errors import ErrorCode, ServiceError
from app.media.frames import FrameStreams, extract_frame_streams
from app.ml.audio_codec.vae import AudioVAE
from app.ml.audio_codec.vocoder import BigVGANVocoder
from app.ml.device import (
    DeviceContext,
    assert_cuda_oom_or_reraise,
    autocast_ctx,
    resolve_device,
)
from app.ml.encoders.clip_text import CLIPTextEncoder
from app.ml.encoders.clip_visual import CLIPVisualEncoder
from app.ml.encoders.sync_encoder import SyncEncoder
from app.ml.encoders.tokenizer import CliPTokenizer
from app.ml.generator.flow_matching import FlowMatchingModel
from app.ml.generator.mmdit import MMDiTGenerator
from app.ml.generator.sampler import integrate_flow
from app.ml.registry import SyncEncoderSpec, VariantSpec, get_variant
from app.ml.seeding import derive_generator, derive_subseed, seed_everything
from app.ml.weights import clear_module_cache, ensure_weight_file, load_module_weights
from app.ml.windowing import equal_power_crossfade_add, plan_windows, slice_by_timestamps

__all__ = ["GeneratedAudio", "VideoToSfxPipeline"]

ProgressCallback = Callable[[float, str], None]
CancelCallback = Callable[[], bool]


@dataclass(slots=True)
class GeneratedAudio:
    """Result of one generation.

    Attributes:
        waveform: [2, samples] float32 waveform at the variant sample rate.
        sample_rate: Waveform rate.
        seed: Concrete seed used.
        timings: Per-stage millisecond timings.
        windows: Window scheduling metadata.
        edge_clamps: Edge-clamped sampled frames (from feature extraction).
    """

    waveform: torch.Tensor
    sample_rate: int
    seed: int
    timings: dict[str, float]
    windows: list[dict[str, float]]
    edge_clamps: int = 0


class VideoToSfxPipeline:
    """Loads all artifacts and generates time-aligned SFX via windowed flow matching."""

    def __init__(self, variant_name: str | None = None) -> None:
        """Resolve settings and variant without touching the GPU yet.

        Parameters:
            variant_name: Override of the configured variant.

        Raises:
            ServiceError: Propagated from registry lookups for unknown variants.
        """
        self._settings = get_settings()
        self._variant_name = variant_name or self._settings.model.model_variant
        self.spec: VariantSpec = get_variant(self._variant_name)
        self.device_context: DeviceContext = resolve_device()
        self._build_lock = threading.Lock()
        self._built = False
        self.tokenizer: CliPTokenizer | None = None
        self.text_encoder: CLIPTextEncoder | None = None
        self.visual_encoder: CLIPVisualEncoder | None = None
        self.sync_encoder: SyncEncoder | None = None
        self.vae: AudioVAE | None = None
        self.vocoder: BigVGANVocoder | None = None
        self.generator: MMDiTGenerator | None = None
        self._forward_module: MMDiTGenerator | None = None
        self.flow: FlowMatchingModel | None = None
        self.null_text_tokens: torch.Tensor | None = None
        self.null_pooled_text: torch.Tensor | None = None
        self.null_pooled_visual: torch.Tensor | None = None

    def build(self) -> None:
        """Instantiate modules and strictly load all verified weights.

        Raises:
            ServiceError: weights_unavailable when artifacts are missing,
                checksum-invalid, or structurally incompatible.
        """
        with self._build_lock:
            if self._built:
                return
            spec = self.spec
            vocab_path = ensure_weight_file(spec, "clip_tokenizer_vocab")
            merges_path = ensure_weight_file(spec, "clip_tokenizer_merges")
            tokenizer = CliPTokenizer(vocab_path, merges_path)
            text_encoder = CLIPTextEncoder(vocab_size=tokenizer.vocab_size)
            load_module_weights(text_encoder, spec, "clip_text")
            visual_encoder = CLIPVisualEncoder()
            load_module_weights(visual_encoder, spec, "clip_visual")
            sync_encoder = SyncEncoder(spec.sync)
            load_module_weights(sync_encoder, spec, "sync_encoder")
            vae = AudioVAE(spec.vae, spec.n_mels)
            load_module_weights(vae, spec, "vae")
            vocoder = BigVGANVocoder(spec.vocoder, spec.n_mels)
            load_module_weights(vocoder, spec, "vocoder")
            vocoder.prepare_inference()
            generator = MMDiTGenerator(
                spec.mmdit,
                latent_channels=spec.vae.latent_channels,
                latent_fps=spec.latent_fps,
                backend=self._settings.model.attention_backend,
            )
            load_module_weights(generator, spec, "generator")
            nulls = _load_null_embeds(spec)
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
            self.visual_encoder = visual_encoder
            self.sync_encoder = sync_encoder
            self.vae = vae
            self.vocoder = vocoder
            self.generator = generator
            self.null_text_tokens = nulls["text_tokens"]
            self.null_pooled_text = nulls["pooled_text"]
            self.null_pooled_visual = nulls["pooled_visual"]
            self.flow = FlowMatchingModel(velocity_fn=self._velocity_fn)
            device = self.device_context.device
            for module in (
                text_encoder,
                visual_encoder,
                sync_encoder,
                vae,
                vocoder,
                generator,
            ):
                module.to(device)
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)
            if self.device_context.is_cuda and self._settings.model.torch_compile:
                self._forward_module = torch.compile(generator)  # type: ignore[assignment]
            else:
                self._forward_module = generator
            from app.telemetry.metrics import WEIGHTS_LOADED

            WEIGHTS_LOADED.set(7)
            self._built = True

    def _velocity_fn(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        pooled_text: torch.Tensor | None = None,
        visual_tokens: torch.Tensor | None = None,
        visual_seconds: torch.Tensor | None = None,
        sync_tokens: torch.Tensor | None = None,
        sync_seconds: torch.Tensor | None = None,
        pooled_visual: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the MMDiT velocity field (bound as flow.velocity_fn).

        Omitted conditioning falls back to the true null embeds (empty-prompt
        text encoding and averaged null visual input from the artifact), never
        to zero-filled stand-ins.

        Parameters:
            x_t: [1, T, C] latents.
            t: [1] flow time.
            text_tokens / pooled_text: Text conditioning.
            visual_tokens / visual_seconds / pooled_visual: Visual conditioning.
            sync_tokens / sync_seconds: Sync conditioning.
            memory_tokens: Previous-window latent tokens.

        Returns:
            [1, T, C] velocity.

        Raises:
            ServiceError: gpu_out_of_memory reclassified from CUDA OOM.
        """
        assert self._forward_module is not None
        device = self.device_context.device
        mmdit_spec = self.spec.mmdit
        tokens = (
            text_tokens.to(device)
            if text_tokens is not None
            else self.null_text_tokens.to(device)
        )
        pooled = (
            pooled_text.to(device)
            if pooled_text is not None
            else self.null_pooled_text.to(device)
        )
        visual = (
            visual_tokens.to(device)
            if visual_tokens is not None
            else torch.zeros(1, 1, mmdit_spec.visual_dim, device=device)
        )
        visual_seconds = (
            visual_seconds.to(device) if visual_seconds is not None else torch.zeros(1, device=device)
        )
        sync = (
            sync_tokens.to(device)
            if sync_tokens is not None
            else torch.zeros(1, 1, mmdit_spec.sync_dim, device=device)
        )
        sync_seconds = (
            sync_seconds.to(device) if sync_seconds is not None else torch.zeros(1, device=device)
        )
        pooled_vis = (
            pooled_visual.to(device)
            if pooled_visual is not None
            else self.null_pooled_visual.to(device)
        )
        try:
            with autocast_ctx(self.device_context):
                return self._forward_module(
                    x_t.to(device),
                    t.to(device),
                    text_tokens=tokens,
                    pooled_text=pooled,
                    visual_tokens=visual,
                    visual_seconds=visual_seconds,
                    sync_tokens=sync,
                    sync_seconds=sync_seconds,
                    pooled_visual=pooled_vis,
                    memory_tokens=memory_tokens.to(device) if memory_tokens is not None else None,
                )
        except torch.cuda.OutOfMemoryError as error:
            assert_cuda_oom_or_reraise(error)
            raise

    def warmup(self) -> dict[str, float]:
        """Run a minimal end-to-end generation to warm kernels and allocations.

        Returns:
            Timings dict of the warmup run.

        Raises:
            ServiceError: weights_unavailable when artifacts cannot load.
        """
        self.build()
        started = time.perf_counter()
        self.generate_from_features(
            visual_features=torch.zeros(2, 512),
            visual_seconds=torch.zeros(2, dtype=torch.float64),
            sync_tokens=torch.zeros(2, self.spec.sync.dim),
            sync_seconds=torch.zeros(2, dtype=torch.float64),
            duration_s=0.32,
            prompt="",
            negative_prompt="",
            seed=0,
            steps=2,
            guidance=0.0,
        )
        return {"warmup_ms": (time.perf_counter() - started) * 1000.0}

    def generate(
        self,
        video_path: Path,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        duration: float | None,
        start_time: float,
        progress_cb: ProgressCallback | None = None,
        cancel_cb: CancelCallback | None = None,
    ) -> GeneratedAudio:
        """Generate SFX for a video file.

        Parameters:
            video_path: Local video file.
            prompt: Sound-design prompt ("" = video-only conditioning).
            negative_prompt: Negative prompt ("" = none).
            seed: Concrete seed (>= 0).
            steps: Sampler steps.
            guidance: CFG scale (0 = unconditional).
            duration: Generation window length (None = full video).
            start_time: Media offset.
            progress_cb: Optional (fraction, stage) reporter.
            cancel_cb: Optional cooperative cancellation.

        Returns:
            GeneratedAudio with the waveform and metadata.

        Raises:
            ServiceError: weights_unavailable, inference_failed, or
                gpu_out_of_memory.
        """
        self.build()
        seed_everything(seed)
        timings: dict[str, float] = {}
        from app.media.probe import probe_media
        from app.media.validate import validate_video

        t0 = time.perf_counter()
        info = _run_coroutine(probe_media(video_path, count_frames_fallback=False))
        plan = validate_video(info, duration=duration, start_time=start_time)
        duration_s = plan.effective_duration_s
        timings["probe_ms"] = (time.perf_counter() - t0) * 1000.0
        if progress_cb:
            progress_cb(0.05, "extract_features")
        t0 = time.perf_counter()
        streams: FrameStreams = extract_frame_streams(
            video_path,
            start_s=start_time,
            duration_s=duration_s,
            visual_fps=self._settings.model.visual_fps,
            sync_fps=self._settings.model.sync_fps,
            rotation=info.rotation_degrees,
            source_fps=info.fps,
        )
        if progress_cb:
            progress_cb(0.35, "extract_features")
        device = self.device_context.device
        assert self.visual_encoder is not None
        assert self.sync_encoder is not None
        visual_features = self.visual_encoder.encode_frames(streams.visual, device)
        groups = streams.segment_groups or _default_groups(
            streams.sync.shape[0], self.spec.sync
        )
        sync_features, sync_starts = self.sync_encoder.encode_segments(
            streams.sync, groups, device
        )
        sync_tokens, sync_seconds = _flatten_sync(
            sync_features, sync_starts, self._settings.model.sync_fps, self.spec.sync
        )
        timings["feature_extraction_ms"] = (time.perf_counter() - t0) * 1000.0
        if progress_cb:
            progress_cb(0.5, "extract_features")
        waveform, windows = self.generate_from_features(
            visual_features=visual_features,
            visual_seconds=streams.visual_timestamps,
            sync_tokens=sync_tokens,
            sync_seconds=sync_seconds,
            duration_s=duration_s,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
        return GeneratedAudio(
            waveform=waveform,
            sample_rate=self.spec.sample_rate,
            seed=seed,
            timings=timings,
            windows=windows,
            edge_clamps=streams.edge_clamps,
        )

    def generate_from_features(
        self,
        *,
        visual_features: torch.Tensor,
        visual_seconds: torch.Tensor,
        sync_tokens: torch.Tensor,
        sync_seconds: torch.Tensor,
        duration_s: float,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        progress_cb: ProgressCallback | None = None,
        cancel_cb: CancelCallback | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, float]]]:
        """Windowed latent generation, decode, and crossfaded overlap-add.

        Parameters:
            visual_features: [T_v, D] CLIP features over the whole clip.
            visual_seconds: [T_v] media-relative seconds.
            sync_tokens: [T_s, D_s] sync features over the whole clip.
            sync_seconds: [T_s] seconds.
            duration_s: Generation length.
            prompt / negative_prompt: Text conditioning.
            seed: Concrete seed.
            steps: Sampler steps.
            guidance: CFG scale.
            progress_cb / cancel_cb: Interaction hooks.

        Returns:
            ([2, samples] waveform, window metadata list).

        Raises:
            ServiceError: inference_failed, inference_timeout, or gpu_out_of_memory.
        """
        self.build()
        assert self.tokenizer is not None
        assert self.text_encoder is not None
        assert self.vae is not None
        assert self.vocoder is not None
        spec = self.spec
        if self.flow is not None:
            self.flow.guidance_scale = guidance
        total_samples = round(duration_s * spec.sample_rate)
        window_s = min(self._settings.model.model_window_s, spec.window_s)
        overlap_s = min(self._settings.model.window_overlap_s, window_s / 4)
        plans = plan_windows(duration_s, window_s, overlap_s)
        accumulator = torch.zeros(2, total_samples, dtype=torch.float32)
        device = self.device_context.device
        negative_text = negative_prompt.strip()
        with torch.no_grad():
            tokens = self.tokenizer.encode_batch([prompt or "", negative_text])
            text_tokens, pooled_text = self.text_encoder.encode_tokens(tokens.to(device))
            if negative_text:
                uncond_tokens = text_tokens[1:2].to(device)
                uncond_pooled = pooled_text[1:2].to(device)
            else:
                assert self.null_text_tokens is not None
                assert self.null_pooled_text is not None
                uncond_tokens = self.null_text_tokens.to(device)
                uncond_pooled = self.null_pooled_text.to(device)
            previous_latents: torch.Tensor | None = None
            window_meta: list[dict[str, float]] = []
            for plan in plans:
                if cancel_cb and cancel_cb():
                    raise ServiceError(
                        ErrorCode.INFERENCE_FAILED,
                        "generation canceled",
                        details={"canceled": True},
                    )
                window_seed = derive_subseed(seed, "window", plan.index)
                generator = derive_generator(window_seed)
                vf, vl = slice_by_timestamps(
                    visual_seconds, plan.start_s, plan.start_s + plan.length_s
                )
                visual_slice = visual_features[vf:vl]
                visual_slice_seconds = visual_seconds[vf:vl].float()
                if visual_slice.shape[0] == 0:
                    visual_slice = visual_features[:1]
                    visual_slice_seconds = torch.zeros(1)
                sf, sl = slice_by_timestamps(
                    sync_seconds, plan.start_s, plan.start_s + plan.length_s
                )
                sync_slice = sync_tokens[sf:sl]
                sync_slice_seconds = sync_seconds[sf:sl].float()
                if sync_slice.shape[0] == 0:
                    sync_slice = torch.zeros(1, spec.sync.dim)
                    sync_slice_seconds = torch.zeros(1)
                latent_frames = max(1, round(plan.length_s * spec.latent_fps))
                memory = None
                if previous_latents is not None and plan.overlap_s > 0:
                    memory_frames = max(1, round(plan.overlap_s * spec.latent_fps))
                    memory = previous_latents[:, -memory_frames:, :]
                noise = torch.randn(
                    1, latent_frames, spec.vae.latent_channels, generator=generator
                )
                fraction_base = plan.index / max(1, len(plans))
                fraction_span = 1.0 / max(1, len(plans))
                conditioning = {
                    "text_tokens": text_tokens[0:1],
                    "pooled_text": pooled_text[0:1],
                    "visual_tokens": visual_slice.unsqueeze(0),
                    "visual_seconds": visual_slice_seconds,
                    "sync_tokens": sync_slice.unsqueeze(0),
                    "sync_seconds": sync_slice_seconds,
                    "pooled_visual": visual_slice.mean(dim=0, keepdim=True),
                    "memory_tokens": memory,
                }
                uncond_conditioning = {
                    "text_tokens": uncond_tokens,
                    "pooled_text": uncond_pooled,
                }
                assert self.flow is not None
                flow = self.flow
                window_conditioning = dict(conditioning)
                window_uncond = dict(uncond_conditioning)

                def _velocity(
                    x: torch.Tensor,
                    t: torch.Tensor,
                    _flow: FlowMatchingModel = flow,
                    _cond: dict[str, torch.Tensor | None] = window_conditioning,
                    _uncond: dict[str, torch.Tensor | None] = window_uncond,
                ) -> torch.Tensor:
                    return _flow.velocity(x, t, conditioning=_cond, uncond_conditioning=_uncond)

                started = time.perf_counter()
                try:
                    latents = integrate_flow(
                        noise.to(device),
                        _velocity,
                        num_steps=steps,
                        progress_cb=(
                            _progress_reporter(progress_cb, fraction_base, fraction_span)
                            if progress_cb
                            else None
                        ),
                        cancel_cb=cancel_cb,
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    assert_cuda_oom_or_reraise(error)
                except RuntimeError as error:
                    if "canceled" in str(error):
                        raise ServiceError(
                            ErrorCode.INFERENCE_FAILED,
                            "generation canceled",
                            details={"canceled": True},
                        ) from error
                    raise ServiceError(
                        ErrorCode.INFERENCE_FAILED,
                        f"integration failed in window {plan.index}: {error}",
                    ) from error
                inference_ms = (time.perf_counter() - started) * 1000.0
                previous_latents = latents.float().cpu()
                decoded = self.vae.decode(self.vae.denormalize(latents))
                piece_mono = self.vocoder.vocode_chunked(decoded).reshape(1, -1)
                piece = _mono_to_stereo(piece_mono)
                offset = round(plan.start_s * spec.sample_rate)
                fade = round(plan.overlap_s * spec.sample_rate)
                accumulator = equal_power_crossfade_add(
                    accumulator, piece[:, : total_samples - offset], offset, fade
                )
                window_meta.append(
                    {
                        "index": float(plan.index),
                        "start_s": plan.start_s,
                        "length_s": plan.length_s,
                        "inference_ms": inference_ms,
                    }
                )
        accumulator = accumulator - accumulator.mean(dim=-1, keepdim=True)
        accumulator = accumulator.clamp(-1.0, 1.0)
        return accumulator, window_meta


def _progress_reporter(
    progress_cb: ProgressCallback, base: float, span: float
) -> Callable[[float], None]:
    """Build a step-callback mapped into this window's progress band.

    Parameters:
        progress_cb: The outer reporter.
        base: Window's base fraction.
        span: Window's fraction span.

    Returns:
        A fraction-only callback.
    """

    def _report(fraction: float) -> None:
        progress_cb(base + span * fraction, "generate")

    return _report


def _mono_to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Convert [1, samples] mono to [2, samples] stereo.

    Parameters:
        waveform: Mono tensor.

    Returns:
        Stereo tensor.

    Raises:
        ValueError: On non-mono input.
    """
    if waveform.dim() != 2 or waveform.shape[0] != 1:
        raise ValueError("expected mono [1, samples] waveform")
    return waveform.repeat(2, 1)


def _default_groups(frame_count: int, spec: SyncEncoderSpec) -> list[list[int]]:
    """Build stride-8 16-frame groups for a sync frame count.

    Parameters:
        frame_count: Available frames.
        spec: Sync spec.

    Returns:
        Group index lists.
    """
    groups: list[list[int]] = []
    offset = 0
    while offset + spec.segment_frames <= frame_count:
        groups.append(list(range(offset, offset + spec.segment_frames)))
        offset += spec.segment_stride
    return groups


def _flatten_sync(
    features: torch.Tensor, segment_starts: list[float], sync_fps: float, spec: SyncEncoderSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten per-segment features into a per-token time-aligned sequence.

    Token i of a segment starting at frame f is timestamped
    (f + segment_frames - out_per_segment + i) / fps so tokens align with the
    tail frames the encoder emits.

    Parameters:
        features: [N, out_per_segment, D].
        segment_starts: Start frame index per segment.
        sync_fps: Sync stream rate.
        spec: Sync spec.

    Returns:
        (tokens [M, D], seconds [M]).
    """
    tokens: list[torch.Tensor] = []
    seconds: list[float] = []
    tail_start = spec.segment_frames - spec.out_per_segment
    for index, start_frame in enumerate(segment_starts):
        block = features[index]
        for token_index in range(block.shape[0]):
            tokens.append(block[token_index])
            frame = start_frame + tail_start + token_index
            seconds.append(frame / sync_fps)
    return torch.stack(tokens), torch.tensor(seconds, dtype=torch.float64)


def _load_null_embeds(spec: VariantSpec) -> dict[str, torch.Tensor]:
    """Load the true unconditional conditioning embeds artifact.

    The artifact is produced by `training.export_weights` by encoding the
    empty prompt with the frozen text tower and averaging empty-input visual
    features with the frozen visual tower.

    Parameters:
        spec: Variant spec.

    Returns:
        Dict with text_tokens [1, 77, D], pooled_text [1, D], pooled_visual [1, Dv].

    Raises:
        ServiceError: weights_unavailable when keys are missing or unloadable.
    """
    from safetensors.torch import load_file

    path = ensure_weight_file(spec, "null_embeds")
    try:
        payload = load_file(str(path))
    except Exception as exc:
        raise ServiceError(ErrorCode.WEIGHTS_UNAVAILABLE, f"cannot load null embeds: {exc}") from exc
    required = {"text_tokens", "pooled_text", "pooled_visual"}
    missing = required - set(payload)
    if missing:
        raise ServiceError(
            ErrorCode.WEIGHTS_UNAVAILABLE,
            f"null embeds artifact missing keys: {sorted(missing)}",
        )
    return {
        "text_tokens": payload["text_tokens"].float(),
        "pooled_text": payload["pooled_text"].float(),
        "pooled_visual": payload["pooled_visual"].float(),
    }


def _run_coroutine(awaitable: Any) -> Any:
    """Execute an awaitable in a throwaway event loop (sync worker context).

    Parameters:
        awaitable: Coroutine to run.

    Returns:
        The coroutine's result.
    """
    return asyncio.run(awaitable)


def reset_pipeline_cache() -> None:
    """Clear cached weight modules (tests)."""
    clear_module_cache()
