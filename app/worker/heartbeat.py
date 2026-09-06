"""Worker heartbeat writer and stale-job reaper."""

from __future__ import annotations

import asyncio
import datetime as dt

import structlog

from app.config import get_settings
from app.db.repositories.predictions import PredictionRepository
from app.db.session import session_scope
from app.telemetry.metrics import QUEUE_DEPTH
from app.utils.time import utc_now

__all__ = ["HeartbeatReporter", "reap_stale_jobs"]

_logger = structlog.get_logger("vsfx.worker.heartbeat")


class HeartbeatReporter:
    """Periodically stamps heartbeat_at for a running prediction."""

    __slots__ = ("_interval_s", "_prediction_id", "_progress", "_stage", "_stop", "_task")

    def __init__(self, prediction_id: str, interval_s: float) -> None:
        """Configure the reporter.

        Parameters:
            prediction_id: Prediction to keep alive.
            interval_s: Write period in seconds.
        """
        self._prediction_id = prediction_id
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._stage: str | None = None
        self._progress: int | None = None

    def update(self, stage: str | None = None, progress: int | None = None) -> None:
        """Record the latest stage/progress for the next heartbeat write.

        Parameters:
            stage: Current stage label.
            progress: Current progress percent.
        """
        if stage is not None:
            self._stage = stage
        if progress is not None:
            self._progress = progress

    async def _loop(self) -> None:
        """Write heartbeats until stopped."""
        while not self._stop.is_set():
            try:
                async with session_scope() as session:
                    await PredictionRepository(session).heartbeat(
                        self._prediction_id, stage=self._stage, progress=self._progress
                    )
            except Exception as exc:
                _logger.debug("heartbeat_write_failed", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                continue

    def start(self) -> None:
        """Start the heartbeat loop (idempotent)."""
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the loop and write one final heartbeat."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, TimeoutError):
                pass
            self._task = None
        try:
            async with session_scope() as session:
                await PredictionRepository(session).heartbeat(
                    self._prediction_id, stage=self._stage, progress=self._progress
                )
        except Exception:
            pass


async def reap_stale_jobs() -> list[str]:
    """Mark stale processing rows failed (they are requeued by the queue loop).

    Parameters: None.

    Returns:
        List of reaped prediction ids.

    Raises:
        Exception: Database errors propagate to the caller's scheduler.
    """
    settings = get_settings()
    cutoff = utc_now() - dt.timedelta(seconds=settings.jobs.job_stale_after_s)
    reaped: list[str] = []
    async with session_scope() as session:
        repo = PredictionRepository(session)
        stale = await repo.stale_processing(cutoff)
        for prediction in stale:
            failed = await repo.fail(
                prediction.id,
                error="job stalled: heartbeat expired; the queue will retry it",
                error_code="inference_timeout",
            )
            if failed:
                reaped.append(prediction.id)
    await refresh_queue_depth()
    return reaped


async def refresh_queue_depth() -> None:
    """Publish the current queue depth gauge."""
    from app.services.queue import stream_depth

    QUEUE_DEPTH.set(await stream_depth())
