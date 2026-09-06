"""Worker process entry point: pipeline warmup, moderation, consumer loop."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.logging_config import configure_logging
from app.services.moderation import build_moderation_service
from app.worker.consumer import run_worker
from app.worker.job_runner import JobRunner

__all__ = ["PipelineHolder", "build_runner", "main"]

_logger = structlog.get_logger("vsfx.worker.main")


class PipelineHolder:
    """Process-level lazy holder for the built inference pipeline."""

    __slots__ = ("_instance",)

    def __init__(self) -> None:
        """Start with no instance loaded."""
        self._instance: Any = None

    def get(self) -> Any:
        """Return the built pipeline, constructing it on first use.

        Returns:
            The built VideoToSfxPipeline.

        Raises:
            ServiceError: weights_unavailable propagated from the build.
        """
        if self._instance is None:
            from app.ml.pipeline import VideoToSfxPipeline

            pipeline = VideoToSfxPipeline()
            pipeline.build()
            self._instance = pipeline
        return self._instance


def build_runner(holder: PipelineHolder | None = None) -> JobRunner:
    """Assemble a JobRunner with a lazily-bound pipeline.

    Parameters:
        holder: Optional holder override (tests).

    Returns:
        Configured JobRunner.
    """
    moderation = build_moderation_service()
    return JobRunner(moderation=moderation, pipeline_holder=(holder or PipelineHolder()).get)


async def main(argv: list[str] | None = None) -> int:
    """Worker main coroutine.

    Parameters:
        argv: Unused argv (kept for interface parity).

    Returns:
        Process exit code.

    Raises:
        Exception: weights_unavailable propagated when warmup fails at start.
    """
    del argv
    settings = get_settings()
    Path(settings.jobs.work_dir).mkdir(parents=True, exist_ok=True)
    heartbeat_file = Path(
        os.environ.get("VSFX_WORKER_HEARTBEAT_FILE", "/tmp/vsfx-worker-heartbeat")
    )
    holder = PipelineHolder()
    if settings.model.warmup_on_start:
        _logger.info("worker_warming_up")
        holder.get()
        _logger.info("worker_weights_ready", variant=settings.model.model_variant)
    runner = build_runner(holder)

    async def _touch() -> None:
        while True:
            heartbeat_file.write_text(str(asyncio.get_running_loop().time()))
            await asyncio.sleep(5.0)

    touch_task = asyncio.create_task(_touch())
    try:
        code = await run_worker(runner)
    finally:
        touch_task.cancel()
        try:
            await touch_task
        except (asyncio.CancelledError, TimeoutError):
            pass
    return code


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(main()))
