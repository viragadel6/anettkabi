from __future__ import annotations

import asyncio
import logging
import signal

from config import Settings, get_settings
from coordination.distributed_lock import close_redis_client, create_redis_client
from database import dispose_db, get_session_factory, init_db
from observability.tracing import configure_tracing, shutdown_tracing
from worker.engine import TaskWorker
from worker.outbox_relay import OutboxRelay

__all__ = [
    "WorkerLifecycle",
    "install_signal_handlers",
    "main",
    "run_worker",
]

logger = logging.getLogger("worker.lifecycle")


def install_signal_handlers(loop: asyncio.AbstractEventLoop, shutdown_event: asyncio.Event) -> None:
    def _trigger_shutdown(sig: signal.Signals) -> None:
        logger.info("Received signal %s; initiating graceful shutdown", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _trigger_shutdown, sig)


class WorkerLifecycle:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._shutdown_event = asyncio.Event()
        self.worker: TaskWorker | None = None
        self.relay: OutboxRelay | None = None
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        configure_tracing(
            self._settings.OTEL_SERVICE_NAME_WORKER, self._settings.ENVIRONMENT
        )
        init_db(self._settings)
        session_factory = get_session_factory()
        redis_client = create_redis_client(self._settings)
        self.worker = TaskWorker(
            session_factory,
            redis_client,
            self._settings.worker,
            submission_channel=self._settings.TASK_SUBMISSION_CHANNEL,
        )
        self.relay = OutboxRelay(
            session_factory,
            redis_client,
            self._settings.OUTBOX_STREAM_NAME,
        )
        loop = asyncio.get_running_loop()
        install_signal_handlers(loop, self._shutdown_event)
        self._tasks = [
            asyncio.create_task(self.worker.run(), name="task-worker"),
            asyncio.create_task(self.relay.run(), name="outbox-relay"),
        ]
        logger.info("Worker lifecycle started all background services")

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        assert self.worker is not None
        assert self.relay is not None
        logger.info(
            "Draining worker for up to %.1f seconds",
            self._settings.worker.DRAIN_TIMEOUT_SECONDS,
        )
        self.relay.stop()
        self.worker.request_stop()
        worker_task = self._tasks[0]
        try:
            await asyncio.wait_for(
                worker_task,
                timeout=self._settings.worker.DRAIN_TIMEOUT_SECONDS + 10.0,
            )
        except asyncio.TimeoutError:
            logger.error("Worker did not finish draining; forcing shutdown")
            worker_task.cancel()
        for background_task in self._tasks[1:]:
            background_task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await dispose_db()
        await close_redis_client()
        shutdown_tracing()
        logger.info("Worker lifecycle shutdown complete")

    async def run(self) -> None:
        await self.start()
        try:
            await self.wait_for_shutdown()
        finally:
            await self.stop()


async def run_worker(settings: Settings | None = None) -> None:
    lifecycle = WorkerLifecycle(settings)
    await lifecycle.run()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Interrupted; exiting")


if __name__ == "__main__":
    main()
