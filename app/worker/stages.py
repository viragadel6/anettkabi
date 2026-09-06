"""Pipeline stages: each timed, reported into timings/stage/progress."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.constants import STAGE_PROGRESS_BASES
from app.db.models import AssetKind
from app.db.repositories.assets import AssetRepository
from app.db.session import session_scope
from app.errors import ErrorCode, ServiceError
from app.media.audio_ops import (
    apply_fades,
    db_to_gain,
    fit_exact_length,
    remove_dc,
    resample,
    samples_for,
    sanitize,
    to_stereo,
)
from app.media.loudness import gain_to_lufs, measure_lufs, measure_true_peak_db
from app.media.mixer import MixParams, mix_audio
from app.media.mux import mux_video_audio, output_metadata_tags
from app.media.probe import MediaInfo, probe_media
from app.media.validate import NormalizationPlan, validate_video
from app.ml.pipeline import GeneratedAudio, VideoToSfxPipeline
from app.services.downloader import Downloader
from app.services.moderation import ModerationService
from app.services.storage import StorageService, output_key_for
from app.telemetry.metrics import (
    AUDIO_VIDEO_DURATION_DELTA_MS,
    INFERENCE_SECONDS,
    STAGE_DURATION_SECONDS,
)
from app.utils.time import utc_now

__all__ = ["JobArtifacts", "StageTimings", "Stages"]

_logger = structlog.get_logger("vsfx.stages")

ProgressReporter = Callable[[str, int], None]


@dataclass
class StageTimings:
    """Accumulated per-stage milliseconds keyed by `<stage>_ms`."""

    values: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, ms: float) -> None:
        """Record one stage duration.

        Parameters:
            stage: Stage name from STAGE_PROGRESS_BASES.
            ms: Duration in milliseconds.
        """
        self.values[f"{stage}_ms"] = round(self.values.get(f"{stage}_ms", 0.0) + ms, 2)
        STAGE_DURATION_SECONDS.labels(stage=stage).observe(ms / 1000.0)


@dataclass
class JobArtifacts:
    """Files and metadata threaded between stages of one job.

    Attributes:
        video_path: Downloaded source video.
        audio_wav: Mixed/generated WAV.
        output_mp4: Final MP4.
        audio_only_wav: Optional return_audio_only asset.
        info: Probe result.
        plan: Validation/normalization plan.
        generated: Generation result.
        mix_metadata: Mixer metadata.
        nsfw_flags: Moderation flags per output element.
        upload_sha: Source sha256.
        source_content_type: Effective source MIME.
    """

    video_path: Path | None = None
    audio_wav: Path | None = None
    output_mp4: Path | None = None
    audio_only_wav: Path | None = None
    info: MediaInfo | None = None
    plan: NormalizationPlan | None = None
    generated: GeneratedAudio | None = None
    mix_metadata: dict[str, object] = field(default_factory=dict)
    nsfw_flags: list[bool] = field(default_factory=lambda: [False])
    upload_sha: str = ""
    source_content_type: str = "video/mp4"


class Stages:
    """Executable stage functions for the job runner."""

    def __init__(
        self,
        work_dir: Path,
        storage: StorageService,
        pipeline: VideoToSfxPipeline,
        moderation: ModerationService,
    ) -> None:
        """Bind stage collaborators.

        Parameters:
            work_dir: Per-job scratch directory.
            storage: Storage service for uploads.
            pipeline: Loaded generation pipeline.
            moderation: Prompt moderation service.
        """
        self._work_dir = work_dir
        self._storage = storage
        self._pipeline = pipeline
        self._moderation = moderation

    async def download(
        self,
        video_ref: str,
        timings: StageTimings,
        report: ProgressReporter,
        *,
        storage_key: str | None = None,
    ) -> Path:
        """Fetch the source video (URL, data URI, or object store).

        Parameters:
            video_ref: URL / data URI / uploaded object URL.
            timings: Timing accumulator.
            report: Progress reporter.
            storage_key: Direct object key when known.

        Returns:
            The downloaded file path.

        Raises:
            ServiceError: download_failed / invalid_video_url / storage_failed.
        """
        started = time.perf_counter()
        report("download", STAGE_PROGRESS_BASES["download"][0])
        settings = get_settings()
        if storage_key:
            destination = self._work_dir / f"source{Path(storage_key).suffix or '.mp4'}"
            self._storage.download_to_file(storage_key, destination)
            result_path = destination
        else:
            downloader = Downloader(self._work_dir)
            result = await downloader.download(video_ref)
            result_path = result.path
        report("download", STAGE_PROGRESS_BASES["download"][1])
        timings.add("download", (time.perf_counter() - started) * 1000.0)
        del settings
        return result_path

    async def probe(
        self, path: Path, timings: StageTimings, report: ProgressReporter
    ) -> MediaInfo:
        """Probe the source file.

        Parameters:
            path: Video file.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            MediaInfo.

        Raises:
            ServiceError: corrupt_media via FfmpegError mapping.
        """
        started = time.perf_counter()
        report("probe", STAGE_PROGRESS_BASES["probe"][0])
        info = await probe_media(path)
        report("probe", STAGE_PROGRESS_BASES["probe"][1])
        timings.add("probe", (time.perf_counter() - started) * 1000.0)
        return info

    def validate(
        self,
        info: MediaInfo,
        request: dict[str, Any],
        timings: StageTimings,
        report: ProgressReporter,
    ) -> NormalizationPlan:
        """Validate limits and derive the normalization plan.

        Parameters:
            info: Probe result.
            request: Normalized prediction input.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            The NormalizationPlan.

        Raises:
            ServiceError: Any validation error code from `validate_video`.
        """
        started = time.perf_counter()
        report("validate", STAGE_PROGRESS_BASES["validate"][0])
        plan = validate_video(
            info,
            duration=request.get("duration"),
            start_time=float(request.get("start_time") or 0.0),
            video_handling=str(request.get("video_handling") or "copy"),
        )
        report("validate", STAGE_PROGRESS_BASES["validate"][1])
        timings.add("validate", (time.perf_counter() - started) * 1000.0)
        return plan

    def moderate(
        self,
        request: dict[str, Any],
        timings: StageTimings,
        report: ProgressReporter,
    ) -> list[bool]:
        """Run prompt moderation.

        Parameters:
            request: Normalized input.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            One flag per planned output element.

        Raises:
            ServiceError: prompt_blocked in `block` mode.
        """
        started = time.perf_counter()
        report("moderate", STAGE_PROGRESS_BASES["moderate"][0])
        prompt = str(request.get("prompt") or "")
        negative = str(request.get("negative_prompt") or "")
        flagged = self._moderation.moderate(prompt) or self._moderation.moderate(negative)
        elements = 2 if request.get("return_audio_only") else 1
        report("moderate", STAGE_PROGRESS_BASES["moderate"][1])
        timings.add("moderate", (time.perf_counter() - started) * 1000.0)
        return [flagged] * elements

    async def generate(
        self,
        video_path: Path,
        request: dict[str, Any],
        plan: NormalizationPlan,
        timings: StageTimings,
        report: ProgressReporter,
        cancel_check: Callable[[], bool],
    ) -> GeneratedAudio:
        """Run the generative pipeline (blocking; call from an executor).

        Parameters:
            video_path: Source video.
            request: Normalized input.
            plan: Validation plan.
            timings: Timing accumulator.
            report: Progress reporter.
            cancel_check: Cancellation probe.

        Returns:
            GeneratedAudio.

        Raises:
            ServiceError: weights_unavailable / inference_failed / gpu_out_of_memory.
        """
        started = time.perf_counter()

        def _progress(fraction: float, stage: str) -> None:
            base, top = STAGE_PROGRESS_BASES["generate"]
            report(stage if stage == "generate" else "extract_features", int(base + (top - base) * fraction))

        def _run() -> GeneratedAudio:
            return self._pipeline.generate(
                video_path,
                prompt=str(request.get("prompt") or ""),
                negative_prompt=str(request.get("negative_prompt") or ""),
                seed=int(request.get("seed") or 0),
                steps=int(request.get("num_inference_steps") or 25),
                guidance=float(request.get("guidance_scale") or 4.5),
                duration=request.get("duration"),
                start_time=float(request.get("start_time") or 0.0),
                progress_cb=_progress,
                cancel_cb=cancel_check,
            )

        loop = asyncio.get_running_loop()
        timeout = get_settings().model.inference_timeout_s
        try:
            generated = await asyncio.wait_for(
                loop.run_in_executor(None, _run), timeout=timeout
            )
        except TimeoutError as exc:
            raise ServiceError(
                ErrorCode.INFERENCE_TIMEOUT,
                f"generation exceeded {timeout}s budget",
            ) from exc
        INFERENCE_SECONDS.observe((time.perf_counter() - started) / 1.0)
        timings.add("generate", (time.perf_counter() - started) * 1000.0)
        timings.values.update(generated.timings)
        return generated

    def decode_audio(
        self,
        generated: GeneratedAudio,
        request: dict[str, Any],
        plan: NormalizationPlan,
        timings: StageTimings,
        report: ProgressReporter,
    ) -> Path:
        """Persist the generated waveform as a WAV at the variant rate.

        Parameters:
            generated: Generation result.
            request: Normalized input.
            plan: Validation plan.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            The WAV path.
        """
        import soundfile as sf

        started = time.perf_counter()
        report("decode_audio", STAGE_PROGRESS_BASES["decode_audio"][0])
        target = samples_for(plan.effective_duration_s, generated.sample_rate)
        waveform = fit_exact_length(generated.waveform, target)
        path = self._work_dir / "generated.wav"
        sf.write(
            str(path),
            waveform.numpy().T,
            generated.sample_rate,
            subtype="PCM_16",
        )
        report("decode_audio", STAGE_PROGRESS_BASES["decode_audio"][1])
        timings.add("decode_audio", (time.perf_counter() - started) * 1000.0)
        return path

    def post_process(
        self,
        wav_path: Path,
        request: dict[str, Any],
        plan: NormalizationPlan,
        timings: StageTimings,
        report: ProgressReporter,
    ) -> Path:
        """Loudness normalization, limiting, fades; write the output-rate WAV.

        Parameters:
            wav_path: Generated WAV.
            request: Normalized input.
            plan: Validation plan.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            Path of the processed WAV (48 kHz stereo).

        Raises:
            ServiceError: parameter_out_of_range for invalid targets.
        """
        import numpy as np
        import soundfile as sf

        started = time.perf_counter()
        report("post_process", STAGE_PROGRESS_BASES["post_process"][0])
        settings = get_settings()
        data, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = torch_from_numpy(data)
        waveform = resample(waveform, rate, settings.output.output_audio_sr)
        waveform = to_stereo(waveform)
        waveform = sanitize(waveform)
        waveform = remove_dc(waveform)
        target_lufs = request.get("target_loudness_lufs")
        true_peak = float(request.get("true_peak_db") or settings.defaults.default_true_peak_db)
        input_lufs = measure_lufs(waveform, settings.output.output_audio_sr)
        if target_lufs is not None:
            waveform, measured, gain_db = gain_to_lufs(
                waveform, settings.output.output_audio_sr, float(target_lufs)
            )
            _logger.info(
                "loudness_normalized",
                input_lufs=measured,
                target_lufs=target_lufs,
                gain_db=round(gain_db, 2),
            )
        from app.media.loudness import TruePeakLimiter

        limiter = TruePeakLimiter(true_peak)
        waveform = limiter.process(waveform, settings.output.output_audio_sr)
        waveform = apply_fades(waveform, settings.output.output_audio_sr)
        target = samples_for(plan.effective_duration_s, settings.output.output_audio_sr)
        waveform = fit_exact_length(waveform, target)
        out_path = self._work_dir / "sfx_processed.wav"
        sf.write(
            str(out_path),
            np.clip(waveform.numpy().T, -1.0, 1.0),
            settings.output.output_audio_sr,
            subtype="PCM_24",
        )
        output_peak = measure_true_peak_db(waveform)
        timings.values["loudness_input_lufs"] = input_lufs
        timings.values["loudness_output_peak_dbtp"] = output_peak
        report("post_process", STAGE_PROGRESS_BASES["post_process"][1])
        timings.add("post_process", (time.perf_counter() - started) * 1000.0)
        return out_path

    async def mix(
        self,
        sfx_wav: Path,
        video_path: Path,
        request: dict[str, Any],
        plan: NormalizationPlan,
        artifacts: JobArtifacts,
        timings: StageTimings,
        report: ProgressReporter,
    ) -> Path:
        """Combine SFX with original audio per audio_mode.

        Parameters:
            sfx_wav: Processed SFX WAV.
            video_path: Source video (original audio source).
            request: Normalized input.
            plan: Validation plan.
            artifacts: Job artifact bag (mix metadata written here).
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            Path of the final mixed WAV.

        Raises:
            ServiceError: mux_failed when original audio extraction fails.
        """
        import numpy as np
        import soundfile as sf

        started = time.perf_counter()
        report("mix", STAGE_PROGRESS_BASES["mix"][0])
        settings = get_settings()
        sfx_data, sfx_rate = sf.read(str(sfx_wav), dtype="float32", always_2d=True)
        sfx = torch_from_numpy(sfx_data.T)
        original = None
        original_rate = None
        if plan.info.has_audio and str(request.get("audio_mode")) != "replace":
            extract_path = self._work_dir / "original.wav"
            from app.media.ffmpeg import run_ffmpeg

            await run_ffmpeg(
                [
                    "-y",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "2",
                    "-ar",
                    str(settings.output.output_audio_sr),
                    "-f",
                    "wav",
                    str(extract_path),
                ],
                timeout_s=300.0,
                operation="extract-original-audio",
            )
            orig_data, orig_rate = sf.read(str(extract_path), dtype="float32", always_2d=True)
            original = torch_from_numpy(orig_data.T)
            original_rate = orig_rate
        params = MixParams(
            audio_mode=str(request.get("audio_mode") or "replace"),
            sfx_gain_db=float(request.get("sfx_gain_db") or 0.0),
            original_gain_db=float(request.get("original_audio_gain_db") or -6.0),
            duck_threshold_db=float(request.get("duck_threshold_db") or -24.0),
            duck_ratio=float(request.get("duck_ratio") or 4.0),
            duck_attack_ms=float(request.get("duck_attack_ms") or 15.0),
            duck_release_ms=float(request.get("duck_release_ms") or 250.0),
            limiter_ceiling_db=float(
                request.get("true_peak_db") or settings.defaults.default_true_peak_db
            ),
        )
        mixed, metadata = mix_audio(
            sfx,
            original,
            sfx_rate,
            original_rate,
            settings.output.output_audio_sr,
            params,
        )
        artifacts.mix_metadata = metadata
        out_path = self._work_dir / "mixed.wav"
        sf.write(
            str(out_path),
            np.clip(mixed.numpy().T, -1.0, 1.0),
            settings.output.output_audio_sr,
            subtype="PCM_24",
        )
        report("mix", STAGE_PROGRESS_BASES["mix"][1])
        timings.add("mix", (time.perf_counter() - started) * 1000.0)
        return out_path

    async def mux(
        self,
        video_path: Path,
        audio_path: Path,
        request: dict[str, Any],
        plan: NormalizationPlan,
        prediction_id: str,
        timings: StageTimings,
        report: ProgressReporter,
    ) -> Path:
        """Mux and verify the final MP4.

        Parameters:
            video_path: Source video.
            audio_path: Final WAV.
            request: Normalized input.
            plan: Validation plan.
            prediction_id: Prediction id for metadata.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            Path of the verified MP4.

        Raises:
            ServiceError: mux_failed on failed verification.
        """
        started = time.perf_counter()
        report("mux", STAGE_PROGRESS_BASES["mux"][0])
        settings = get_settings()
        output_path = self._work_dir / "output.mp4"
        copy_video = bool(plan.copy_video) and str(request.get("video_handling")) != "reencode"
        metadata = output_metadata_tags(
            model_variant=settings.model.model_variant,
            prompt=str(request.get("prompt") or ""),
            seed=int(request.get("seed") or 0),
            prediction_id=prediction_id,
        )
        result = await mux_video_audio(
            video_path,
            audio_path,
            output_path,
            copy_video=copy_video,
            duration_s=plan.effective_duration_s,
            metadata=metadata,
            rotation=plan.info.rotation_degrees,
        )
        AUDIO_VIDEO_DURATION_DELTA_MS.set(result.delta_ms)
        if result.audio_rms < 1e-5:
            _logger.warning(
                "output_audio_near_silent",
                prediction_id=prediction_id,
                rms=result.audio_rms,
            )
        report("mux", STAGE_PROGRESS_BASES["mux"][1])
        timings.add("mux", (time.perf_counter() - started) * 1000.0)
        return output_path

    def write_audio_only(
        self, mixed_wav: Path, timings: StageTimings, report: ProgressReporter
    ) -> Path:
        """Copy the mixed WAV as the optional audio-only asset.

        Parameters:
            mixed_wav: Final mixed WAV.
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            Path of the audio-only asset.
        """
        started = time.perf_counter()
        target = self._work_dir / "audio.wav"
        target.write_bytes(mixed_wav.read_bytes())
        timings.add("post", (time.perf_counter() - started) * 1000.0)
        return target

    async def upload(
        self,
        output_path: Path,
        audio_only_path: Path | None,
        prediction_id: str,
        request: dict[str, Any],
        artifacts: JobArtifacts,
        timings: StageTimings,
        report: ProgressReporter,
    ) -> list[str]:
        """Upload outputs to object storage and register asset rows.

        Parameters:
            output_path: Final MP4.
            audio_only_path: Optional WAV asset.
            prediction_id: Prediction id.
            request: Normalized input.
            artifacts: Artifact bag (upload sha reused).
            timings: Timing accumulator.
            report: Progress reporter.

        Returns:
            List of output URLs (MP4 first).

        Raises:
            ServiceError: storage_failed on upload errors.
        """
        started = time.perf_counter()
        report("upload", STAGE_PROGRESS_BASES["upload"][0])
        settings = get_settings()
        prefix = settings.storage.s3_key_prefix
        retention_days = settings.storage.result_retention_days
        urls: list[str] = []
        expires = utc_now().timestamp() + retention_days * 86400
        mp4_key = output_key_for(prefix, prediction_id, "output.mp4")
        self._storage.put_file(
            output_path,
            mp4_key,
            content_type="video/mp4",
            metadata={
                "prediction-id": prediction_id,
                "model": settings.model.model_id,
                "seed": str(request.get("seed") or 0),
            },
        )
        urls.append(self._storage.public_url(mp4_key))
        if audio_only_path is not None:
            wav_key = output_key_for(prefix, prediction_id, "audio.wav")
            self._storage.put_file(
                audio_only_path,
                wav_key,
                content_type="audio/wav",
                metadata={"prediction-id": prediction_id},
            )
            urls.append(self._storage.public_url(wav_key))
        import datetime as dt

        async with session_scope() as session:
            repo = AssetRepository(session)
            await repo.register(
                kind=AssetKind.OUTPUT_VIDEO,
                storage_key=mp4_key,
                bucket=self._storage.bucket,
                content_type="video/mp4",
                size_bytes=output_path.stat().st_size,
                sha256=artifacts.upload_sha,
                duration_s=artifacts.plan.effective_duration_s if artifacts.plan else None,
                width=artifacts.info.width if artifacts.info else None,
                height=artifacts.info.height if artifacts.info else None,
                fps=artifacts.info.fps if artifacts.info else None,
                has_audio=True,
                video_codec=artifacts.info.video_codec if artifacts.info else None,
                audio_codec="aac",
                expires_at=dt.datetime.fromtimestamp(expires, tz=dt.UTC),
            )
            if audio_only_path is not None:
                await repo.register(
                    kind=AssetKind.DEBUG_AUDIO,
                    storage_key=output_key_for(prefix, prediction_id, "audio.wav"),
                    bucket=self._storage.bucket,
                    content_type="audio/wav",
                    size_bytes=audio_only_path.stat().st_size,
                    sha256="",
                    expires_at=dt.datetime.fromtimestamp(expires, tz=dt.UTC),
                )
        report("upload", STAGE_PROGRESS_BASES["upload"][1])
        timings.add("upload", (time.perf_counter() - started) * 1000.0)
        return urls


def torch_from_numpy(data: Any) -> Any:
    """Convert a float32 numpy array to a torch tensor (channels-first).

    Parameters:
        data: [samples, channels] array.

    Returns:
        [channels, samples] tensor.
    """
    import numpy as np
    import torch

    return torch.from_numpy(np.ascontiguousarray(data.T))


def mix_params_from_request(request: dict[str, Any]) -> MixParams:
    """Build MixParams from a normalized request (unit tests).

    Parameters:
        request: Normalized input.

    Returns:
        The MixParams.
    """
    settings = get_settings()
    return MixParams(
        audio_mode=str(request.get("audio_mode") or "replace"),
        sfx_gain_db=float(request.get("sfx_gain_db") or 0.0),
        original_gain_db=float(request.get("original_audio_gain_db") or -6.0),
        duck_threshold_db=float(request.get("duck_threshold_db") or -24.0),
        duck_ratio=float(request.get("duck_ratio") or 4.0),
        duck_attack_ms=float(request.get("duck_attack_ms") or 15.0),
        duck_release_ms=float(request.get("duck_release_ms") or 250.0),
        limiter_ceiling_db=float(
            request.get("true_peak_db") or settings.defaults.default_true_peak_db
        ),
    )


def db_to_gain_checked(db: float) -> float:
    """Convert dB to linear gain with bounds checking.

    Parameters:
        db: Decibel value.

    Returns:
        Linear gain.

    Raises:
        ServiceError: parameter_out_of_range outside [-120, 60].
    """
    if not -120.0 <= db <= 60.0:
        raise ServiceError(ErrorCode.PARAMETER_OUT_OF_RANGE, f"gain {db} dB out of range")
    return db_to_gain(db)
