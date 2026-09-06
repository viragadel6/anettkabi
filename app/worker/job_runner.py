"""Job runner: executes the full stage pipeline for one claimed job."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.constants import (
    NEVER_RETRY_ERROR_CODES,
    STAGE_PROGRESS_BASES,
    TRANSIENT_ERROR_CODES_FOR_RETRY,
)
from app.db.models import PredictionStatus
from app.db.repositories.predictions import PredictionRepository
from app.db.repositories.usage import UsageRepository
from app.db.session import session_scope
from app.errors import ErrorCode, ServiceError
from app.services.moderation import ModerationService
from app.services.storage import StorageService, get_storage_service
from app.services.webhook_sender import WebhookSender
from app.utils.files import assert_disk_headroom, ensure_dir, remove_quietly
from app.utils.time import utc_now
from app.worker.heartbeat import HeartbeatReporter
from app.worker.stages import JobArtifacts, Stages, StageTimings

__all__ = ["JobRunner", "RunnerOutcome"]

_logger = structlog.get_logger("vsfx.worker.runner")


class RunnerOutcome:
    """Terminal result of one job attempt."""

    __slots__ = ("billed_seconds", "error", "error_code", "gpu_ms", "retryable", "status")

    def __init__(
        self,
        status: PredictionStatus,
        error_code: str = "",
        error: str = "",
        retryable: bool = False,
        billed_seconds: float = 0.0,
        gpu_ms: int = 0,
    ) -> None:
        """Store the outcome.

        Parameters:
            status: Terminal prediction status for this attempt.
            error_code: Taxonomy code on failure.
            error: Safe message on failure.
            retryable: Whether the queue may retry this attempt.
            billed_seconds: Billable seconds consumed.
            gpu_ms: GPU milliseconds consumed.
        """
        self.status = status
        self.error_code = error_code
        self.error = error
        self.retryable = retryable
        self.billed_seconds = billed_seconds
        self.gpu_ms = gpu_ms


class JobRunner:
    """Owns the execution of a single prediction job attempt."""

    def __init__(
        self,
        storage: StorageService | None = None,
        moderation: ModerationService | None = None,
        webhook_sender: WebhookSender | None = None,
        pipeline_holder: Any = None,
    ) -> None:
        """Wire collaborators (lazy defaults for storage/pipeline).

        Parameters:
            storage: Storage service override.
            moderation: Moderation service override.
            webhook_sender: Webhook sender override.
            pipeline_holder: Callable returning a built VideoToSfxPipeline
                (lazy so worker startup does not block on GPU import).
        """
        self._settings = get_settings()
        self._storage = storage
        self._moderation = moderation
        self._webhook_sender = webhook_sender
        self._pipeline_holder = pipeline_holder

    def _require_storage(self) -> StorageService:
        """Return the storage service, constructing it lazily."""
        if self._storage is None:
            self._storage = get_storage_service()
        return self._storage

    def _require_moderation(self) -> ModerationService:
        """Return the moderation service, constructing it lazily."""
        if self._moderation is None:
            from app.services.moderation import build_moderation_service

            self._moderation = build_moderation_service()
        return self._moderation

    def _require_pipeline(self) -> Any:
        """Return the built pipeline via the holder."""
        if self._pipeline_holder is None:
            raise ServiceError(
                ErrorCode.WEIGHTS_UNAVAILABLE,
                "worker started without a pipeline holder; refusing to fabricate audio",
            )
        return self._pipeline_holder()

    async def run(
        self,
        prediction_id: str,
        api_key_id: str,
        worker_id: str,
        attempt: int,
    ) -> RunnerOutcome:
        """Execute the full pipeline for one claimed prediction.

        Parameters:
            prediction_id: Prediction id.
            api_key_id: Owning API key id.
            worker_id: This consumer's name.
            attempt: Attempt number.

        Returns:
            RunnerOutcome describing the terminal state of this attempt.
        """
        settings = self._settings
        started = time.perf_counter()
        timings = StageTimings()
        job_dir = ensure_dir(Path(settings.jobs.work_dir) / prediction_id)
        artifacts = JobArtifacts()
        heartbeat = HeartbeatReporter(prediction_id, settings.jobs.job_heartbeat_interval_s)
        request = await self._load_request(prediction_id)
        progress_state = {"stage": "download", "progress": 2}

        def report(stage: str, progress: int) -> None:
            progress_state["stage"] = stage
            progress_state["progress"] = int(progress)
            heartbeat.update(stage, int(progress))

        pipeline = None
        try:
            assert_disk_headroom(job_dir, 2 * 1024 * 1024 * 1024)
            pipeline = self._require_pipeline()
            stages = Stages(job_dir, self._require_storage(), pipeline, self._require_moderation())
            heartbeat.start()
            storage_key = self._storage_key_for(request)
            artifacts.video_path = await stages.download(
                str(request.get("video") or ""), timings, report, storage_key=storage_key
            )
            artifacts.info = await stages.probe(artifacts.video_path, timings, report)
            artifacts.plan = stages.validate(artifacts.info, request, timings, report)
            artifacts.nsfw_flags = await asyncio_to_thread(
                stages.moderate, request, timings, report
            )
            cancel_requested = await self._cancel_requested(prediction_id)
            if cancel_requested:
                return RunnerOutcome(PredictionStatus.CANCELED)
            generated = await stages.generate(
                artifacts.video_path,
                request,
                artifacts.plan,
                timings,
                report,
                lambda: self._cancel_requested_sync(prediction_id),
            )
            artifacts.generated = generated
            request["seed"] = generated.seed
            wav_path = await asyncio_to_thread(
                stages.decode_audio, generated, request, artifacts.plan, timings, report
            )
            processed = await asyncio_to_thread(
                stages.post_process, wav_path, request, artifacts.plan, timings, report
            )
            mixed = await stages.mix(
                processed, artifacts.video_path, request, artifacts.plan, artifacts, timings, report
            )
            artifacts.audio_wav = mixed
            if request.get("return_audio_only"):
                artifacts.audio_only_wav = await asyncio_to_thread(
                    stages.write_audio_only, mixed, timings, report
                )
            artifacts.output_mp4 = await stages.mux(
                artifacts.video_path,
                mixed,
                request,
                artifacts.plan,
                prediction_id,
                timings,
                report,
            )
            report("verify", STAGE_PROGRESS_BASES["verify"][1])
            urls = await stages.upload(
                artifacts.output_mp4,
                artifacts.audio_only_wav,
                prediction_id,
                request,
                artifacts,
                timings,
                report,
            )
            report("finalize", 100)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            timings.values["total_ms"] = float(elapsed_ms)
            timings.values["queue_ms"] = await self._queue_wait_ms(prediction_id)
            completed = await self._finalize_completed(
                prediction_id, urls, artifacts, timings, elapsed_ms
            )
            billed_seconds = round(elapsed_ms / 1000.0, 3)
            await self._record_usage(api_key_id, prediction_id, billed_seconds)
            if completed:
                _logger.info(
                    "job_completed",
                    prediction_id=prediction_id,
                    elapsed_ms=elapsed_ms,
                    outputs=len(urls),
                )
            await self._maybe_send_webhook(prediction_id)
            return RunnerOutcome(
                PredictionStatus.COMPLETED, billed_seconds=billed_seconds, gpu_ms=elapsed_ms
            )
        except ServiceError as error:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            retryable = error.code in TRANSIENT_ERROR_CODES_FOR_RETRY and error.code not in NEVER_RETRY_ERROR_CODES
            billed = round(elapsed_ms / 1000.0, 3)
            await self._record_usage(api_key_id, prediction_id, billed)
            await self._finalize_failed(prediction_id, error, attempt)
            _logger.exception(
                "job_failed",
                prediction_id=prediction_id,
                error_code=error.code,
                message=error.message,
                retryable=retryable,
                attempt=attempt,
            )
            await self._maybe_send_webhook(prediction_id)
            return RunnerOutcome(
                PredictionStatus.FAILED,
                error_code=error.code,
                error=error.message,
                retryable=retryable,
                billed_seconds=billed,
                gpu_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            error = ServiceError(
                ErrorCode.INTERNAL_ERROR,
                "unexpected worker failure; request id logged server-side",
                details={"type": type(exc).__name__},
            )
            _logger.exception("job_crashed", prediction_id=prediction_id, error=str(exc))
            await self._finalize_failed(prediction_id, error, attempt)
            await self._maybe_send_webhook(prediction_id)
            return RunnerOutcome(
                PredictionStatus.FAILED,
                error_code=error.code,
                error=error.message,
                retryable=False,
                billed_seconds=round(elapsed_ms / 1000.0, 3),
                gpu_ms=elapsed_ms,
            )
        finally:
            await heartbeat.stop()
            if not settings.jobs.keep_intermediate:
                remove_quietly(job_dir)

    def _storage_key_for(self, request: dict[str, Any]) -> str | None:
        """Derive the direct object key for same-service upload URLs.

        Parameters:
            request: Normalized input.

        Returns:
            Object key when resolvable, else None.
        """
        video = str(request.get("video") or "")
        public_base = self._settings.server.public_base_url
        marker = "/uploads/"
        if public_base and video.startswith(public_base) and marker in video:
            tail = video.split(marker, 1)[1].lstrip("/")
            prefix = self._settings.storage.s3_key_prefix.strip("/")
            if prefix:
                return f"{prefix}/uploads/{tail}"
            return f"uploads/{tail}"
        return None

    async def _load_request(self, prediction_id: str) -> dict[str, Any]:
        """Load and validate the stored input payload.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            The normalized input dict.

        Raises:
            ServiceError: prediction_not_found when missing.
        """
        async with session_scope() as session:
            row = await PredictionRepository(session).get(prediction_id)
            if row is None:
                raise ServiceError(
                    ErrorCode.PREDICTION_NOT_FOUND,
                    f"prediction {prediction_id} disappeared mid-flight",
                )
            return dict(row.input or {})

    async def _cancel_requested(self, prediction_id: str) -> bool:
        """Check the cancel flag in the database.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            True when cancellation was requested.
        """
        async with session_scope() as session:
            row = await PredictionRepository(session).get(prediction_id)
            return bool(row and row.canceled_requested)

    def _cancel_requested_sync(self, prediction_id: str) -> bool:
        """Synchronous cancel probe for the sampler loop.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            True when cancellation was requested (polled with short TTL).
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._cancel_requested(prediction_id), loop
        )
        try:
            return future.result(timeout=5.0)
        except (TimeoutError, RuntimeError):
            return False

    async def _queue_wait_ms(self, prediction_id: str) -> float:
        """Compute queue wait from queued_at to started_at.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            Milliseconds queued.
        """
        async with session_scope() as session:
            row = await PredictionRepository(session).get(prediction_id)
            if row is None or row.queued_at is None or row.started_at is None:
                return 0.0
            delta = (row.started_at - row.queued_at).total_seconds()
            return round(max(0.0, delta) * 1000.0, 2)

    async def _finalize_completed(
        self,
        prediction_id: str,
        urls: list[str],
        artifacts: JobArtifacts,
        timings: StageTimings,
        elapsed_ms: int,
    ) -> bool:
        """Persist the completed transition.

        Parameters:
            prediction_id: Prediction id.
            urls: Output URLs.
            artifacts: Artifact bag.
            timings: Timing accumulator.
            elapsed_ms: Wall time.

        Returns:
            True when the transition succeeded.
        """
        from app.telemetry.metrics import PREDICTIONS_TOTAL

        async with session_scope() as session:
            repo = PredictionRepository(session)
            done = await repo.complete(
                prediction_id,
                outputs=urls,
                has_nsfw_contents=artifacts.nsfw_flags[: len(urls)],
                output_asset_id=None,
                timings=timings.values,
                execution_time_ms=elapsed_ms,
            )
            if done:
                PREDICTIONS_TOTAL.labels(status="completed").inc()
            return done

    async def _finalize_failed(
        self, prediction_id: str, error: ServiceError, attempt: int
    ) -> None:
        """Persist the failed transition.

        Parameters:
            prediction_id: Prediction id.
            error: The failure.
            attempt: Attempt number.
        """
        from app.telemetry.metrics import PREDICTIONS_TOTAL

        async with session_scope() as session:
            repo = PredictionRepository(session)
            row = await repo.get(prediction_id)
            done = await repo.fail(prediction_id, error=error.message, error_code=error.code)
            if done:
                PREDICTIONS_TOTAL.labels(status="failed").inc()
            elif row is not None and row.canceled_requested:
                await repo.mark_canceled(prediction_id)

    async def _record_usage(
        self, api_key_id: str, prediction_id: str, billed_seconds: float
    ) -> None:
        """Write a usage event.

        Parameters:
            api_key_id: Caller key id.
            prediction_id: Prediction id.
            billed_seconds: Seconds to bill.
        """
        try:
            async with session_scope() as session:
                await UsageRepository(session).record(
                    api_key_id=uuid.UUID(api_key_id),
                    prediction_id=prediction_id,
                    billed_seconds=billed_seconds,
                    gpu_ms=int(billed_seconds * 1000),
                )
        except Exception as exc:
            _logger.warning("usage_record_failed", prediction_id=prediction_id, error=str(exc))

    async def _maybe_send_webhook(self, prediction_id: str) -> None:
        """Deliver the terminal-state webhook when configured.

        Parameters:
            prediction_id: Prediction id.
        """
        try:
            async with session_scope() as session:
                row = await PredictionRepository(session).get(prediction_id)
                if row is None or not row.webhook_url:
                    return
            sender = self._webhook_sender or WebhookSender()
            await sender.deliver_for_prediction(prediction_id)
        except Exception as exc:
            _logger.warning("webhook_dispatch_failed", prediction_id=prediction_id, error=str(exc))


async def asyncio_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking stage function in the default executor.

    Parameters:
        func: Callable to run.
        *args, **kwargs: Arguments for the callable.

    Returns:
        The callable result.
    """
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def queue_position_estimate(depth: int) -> int:
    """Estimate queue position for progress reporting.

    Parameters:
        depth: Current stream depth.

    Returns:
        Estimated position (0 when empty).
    """
    return max(0, depth - 1)


def utc_now_ref() -> Any:
    """Return current UTC time (test seam).

    Returns:
        Timezone-aware datetime.
    """
    return utc_now()
