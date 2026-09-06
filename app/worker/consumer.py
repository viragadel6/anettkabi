"""Redis Streams consumer loop: claim, run, ack, retry, dead-letter, shutdown."""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from typing import Any

import structlog

from app.config import get_settings
from app.services.queue import (
    JobMessage,
    ack_job,
    claim_stale_jobs,
    enqueue_job,
    ensure_stream,
    move_to_dead_letter,
    new_consumer_name,
    read_jobs,
    stream_depth,
)
from app.utils.retry import backoff_delay_s
from app.worker.gpu_lock import GpuLock
from app.worker.heartbeat import reap_stale_jobs, refresh_queue_depth
from app.worker.job_runner import JobRunner

__all__ = ["WorkerConsumer"]

_logger = structlog.get_logger("vsfx.worker.consumer")


class WorkerConsumer:
    """Long-running consumer bound to the configured stream and group."""

    def __init__(
        self,
        runner: JobRunner,
        *,
        consumer_name: str | None = None,
        gpu_lock: GpuLock | None = None,
    ) -> None:
        """Wire the consumer.

        Parameters:
            runner: Job execution engine.
            consumer_name: Unique consumer id (generated when omitted).
            gpu_lock: Cluster GPU limiter override.
        """
        self._settings = get_settings()
        self._runner = runner
        self._consumer = consumer_name or new_consumer_name()
        self._gpu_lock = gpu_lock or GpuLock()
        self._stopping = asyncio.Event()
        self._in_flight: set[asyncio.Task[Any]] = set()

    @property
    def consumer_name(self) -> str:
        """Return this consumer's name."""
        return self._consumer

    def request_stop(self) -> None:
        """Signal graceful shutdown (SIGTERM/SIGINT or tests)."""
        self._stopping.set()

    async def run_forever(self) -> int:
        """Consume jobs until stopped; returns an exit code.

        Returns:
            0 on graceful shutdown.

        Raises:
            Exception: Fatal setup errors propagate after logging.
        """
        await ensure_stream(self._settings.redis.queue_stream_name, self._settings.redis.queue_group)
        reaper_task = asyncio.create_task(self._reaper_loop())
        sweeper_task = asyncio.create_task(self._retention_loop())
        _logger.info("worker_started", consumer=self._consumer)
        exit_code = 0
        try:
            while not self._stopping.is_set():
                await refresh_queue_depth()
                messages = await read_jobs(self._consumer, count=1, block_ms=2000)
                if not messages:
                    messages = await claim_stale_jobs(self._consumer, count=1)
                if not messages:
                    await self._cancel_requested_sweep()
                    continue
                for message in messages:
                    task = asyncio.create_task(self._process(message))
                    self._in_flight.add(task)
                    task.add_done_callback(self._in_flight.discard)
        except Exception as exc:
            _logger.exception("worker_fatal", error=str(exc))
            exit_code = 1
        finally:
            await self._drain()
            reaper_task.cancel()
            sweeper_task.cancel()
            for task in (reaper_task, sweeper_task):
                try:
                    await task
                except (asyncio.CancelledError, TimeoutError):
                    pass
            _logger.info("worker_stopped", consumer=self._consumer)
        return exit_code

    async def _process(self, message: JobMessage) -> None:
        """Claim, run, and settle one job message.

        Parameters:
            message: The claimed queue message.
        """
        settings = self._settings
        started = time.perf_counter()
        claimed = await self._mark_processing(message)
        if not claimed:
            await ack_job(message.stream_id)
            return
        try:
            async with asyncio.timeout(settings.jobs.job_total_timeout_s):
                outcome = await self._runner.run(
                    message.prediction_id,
                    message.api_key_id,
                    self._consumer,
                    message.attempt,
                )
        except TimeoutError:
            from app.errors import ErrorCode, ServiceError

            error = ServiceError(
                ErrorCode.INFERENCE_TIMEOUT,
                f"job exceeded the {settings.jobs.job_total_timeout_s}s total budget",
            )
            outcome = await self._settle_timeout(message, error)
        if outcome.status.value in ("completed", "canceled"):
            await ack_job(message.stream_id)
        else:
            if outcome.retryable and message.attempt < settings.jobs.job_max_attempts:
                await self._schedule_retry(message)
            else:
                await move_to_dead_letter(
                    message,
                    outcome.error_code or "internal_error",
                )
        _logger.info(
            "job_settled",
            prediction_id=message.prediction_id,
            status=outcome.status.value,
            attempt=message.attempt,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _settle_timeout(self, message: JobMessage, error: Any) -> Any:
        """Mark a timed-out job failed and produce a failed outcome.

        Parameters:
            message: The job message.
            error: The ServiceError describing the timeout.

        Returns:
            A failed RunnerOutcome.
        """
        from app.db.models import PredictionStatus
        from app.db.repositories.predictions import PredictionRepository
        from app.db.session import session_scope
        from app.worker.job_runner import RunnerOutcome

        async with session_scope() as session:
            await PredictionRepository(session).fail(
                message.prediction_id, error=error.message, error_code=error.code
            )
        return RunnerOutcome(
            PredictionStatus.FAILED,
            error_code=error.code,
            error=error.message,
            retryable=True,
        )

    async def _mark_processing(self, message: JobMessage) -> bool:
        """Transition the prediction queued → processing.

        Parameters:
            message: Claimed message.

        Returns:
            True when this worker owns the job.
        """
        from app.db.repositories.predictions import PredictionRepository
        from app.db.session import session_scope

        async with session_scope() as session:
            repo = PredictionRepository(session)
            row = await repo.get(message.prediction_id)
            if row is None:
                return False
            if row.canceled_requested:
                await repo.mark_canceled(message.prediction_id)
                return False
            if row.status.value == "processing" and row.worker_id == self._consumer:
                return True
            return await repo.mark_processing(
                message.prediction_id, self._consumer, message.attempt
            )

    async def _schedule_retry(self, message: JobMessage) -> None:
        """Requeue the message with backoff and ack the original entry.

        Parameters:
            message: The failed message.
        """
        from app.db.repositories.predictions import PredictionRepository
        from app.db.session import session_scope

        settings = self._settings
        next_attempt = message.attempt + 1
        delay = backoff_delay_s(
            message.attempt - 1,
            settings.jobs.job_backoff_base_s,
            settings.jobs.job_backoff_max_s,
        )
        async with session_scope() as session:
            await PredictionRepository(session).requeue(
                message.prediction_id, attempt=next_attempt
            )
        if delay > 0:
            await asyncio.sleep(delay)
        await enqueue_job(
            message.prediction_id,
            message.api_key_id,
            attempt=next_attempt,
            priority=message.priority + 1,
        )
        await ack_job(message.stream_id)

    async def _cancel_requested_sweep(self) -> None:
        """Cancel queued rows whose clients requested cancellation.

        Parameters: None.
        """
        from app.db.models import PredictionStatus
        from app.db.repositories.predictions import PredictionRepository
        from app.db.session import session_scope

        async with session_scope() as session:
            repo = PredictionRepository(session)
            pending = await repo.cancel_requested_pending(limit=50)
            for row in pending:
                if row.status in (
                    PredictionStatus.CREATED,
                    PredictionStatus.QUEUED,
                ):
                    await repo.mark_canceled(row.id)

    async def _reaper_loop(self) -> None:
        """Periodically reap stale jobs and refresh queue depth."""
        interval = max(5.0, self._settings.jobs.job_heartbeat_interval_s)
        while True:
            try:
                reaped = await reap_stale_jobs()
                if reaped:
                    _logger.warning("reaped_stale_jobs", count=len(reaped), ids=reaped)
            except Exception as exc:
                _logger.debug("reaper_failed", error=str(exc))
            await asyncio.sleep(interval)

    async def _retention_loop(self) -> None:
        """Periodically sweep expired output objects."""
        from app.services.storage import sweep_retention

        while True:
            try:
                await sweep_retention()
            except Exception as exc:
                _logger.debug("retention_sweep_failed", error=str(exc))
            await asyncio.sleep(3600.0)

    async def _drain(self) -> None:
        """Wait for in-flight tasks (bounded by job timeout), then stop."""
        if not self._in_flight:
            return
        _logger.info("worker_draining", in_flight=len(self._in_flight))
        done, pending = await asyncio.wait(
            set(self._in_flight), timeout=self._settings.jobs.job_total_timeout_s
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                _logger.error("in_flight_task_error", error=str(exc))


async def run_worker(runner: JobRunner | None = None) -> int:
    """Entry point: build the consumer, install signal handlers, run.

    Parameters:
        runner: Optional runner override.

    Returns:
        Process exit code.
    """
    consumer = WorkerConsumer(runner or JobRunner())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, consumer.request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: consumer.request_stop())
    code = await consumer.run_forever()
    sys.exit(code)


def queue_depth_snapshot() -> int:
    """Synchronous queue depth helper (tests).

    Returns:
        Current stream depth or 0.
    """
    import asyncio

    try:
        return asyncio.run(stream_depth())
    except Exception:
        return 0
