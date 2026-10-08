from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import inspect
import logging
import socket
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import WorkerSettings
from coordination.distributed_lock import LeaseAcquisitionError, distributed_lease
from coordination.wakeup import TASK_SUBMITTED_CHANNEL, WakeSubscription
from database import session_scope
from domain.exceptions import HandlerNotRegisteredError, TaskOwnershipError
from domain.state_machine import TaskState
from models import OutboxEvent, Task
from observability.metrics import (
    active_worker_leases,
    queue_dispatch_lag_seconds,
    record_execution_failure,
    record_execution_success,
    record_task_dead_lettered,
)
from observability.tracing import record_span_error, task_execution_span
from repositories.task_repository import TaskRepository

__all__ = [
    "TASK_HANDLER_REGISTRY",
    "TaskHandler",
    "TaskWorker",
    "task_handler",
]

logger = logging.getLogger("worker.engine")

TaskHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None] | dict[str, Any] | None]

TASK_HANDLER_REGISTRY: dict[str, TaskHandler] = {}


def task_handler(task_type: str) -> Callable[[TaskHandler], TaskHandler]:
    def decorator(handler: TaskHandler) -> TaskHandler:
        if not task_type:
            raise ValueError("task_type must be a non-empty string")
        if task_type in TASK_HANDLER_REGISTRY:
            raise ValueError(f"A handler for task type {task_type!r} is already registered")
        TASK_HANDLER_REGISTRY[task_type] = handler
        return handler

    return decorator


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


class TaskWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        redis_client: aioredis.Redis,
        worker_settings: WorkerSettings,
        worker_id: str | None = None,
        submission_channel: str = TASK_SUBMITTED_CHANNEL,
    ) -> None:
        self._session_factory = session_factory
        self._redis = redis_client
        self._settings = worker_settings
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"
        self._semaphore = asyncio.Semaphore(worker_settings.CONCURRENCY)
        self._stop_event = asyncio.Event()
        self._wake_event = asyncio.Event()
        self._submission_channel = submission_channel
        self._wake_subscription = WakeSubscription(
            redis_client, self._wake_event.set, submission_channel
        )
        self._in_flight: set[asyncio.Task[None]] = set()
        self._running = False
        self._max_poll_interval_seconds = 2.0
        self._reclaim_every_cycles = 15

    @property
    def running(self) -> bool:
        return self._running

    @property
    def healthy(self) -> bool:
        return self._running and not self._stop_event.is_set()

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        self._running = True
        logger.info("TaskWorker %s starting with concurrency %d", self.worker_id, self._settings.CONCURRENCY)
        await self._warm_pool()
        await self._wake_subscription.start()
        base_interval = self._settings.POLL_INTERVAL_MILLISECONDS / 1000.0
        poll_interval = base_interval
        cycle = 0
        try:
            while not self._stop_event.is_set():
                if len(self._in_flight) >= self._settings.CONCURRENCY:
                    woke = await self._sleep_or_stop(base_interval)
                    if woke:
                        poll_interval = base_interval
                    continue
                batch = await self._poll_batch()
                cycle += 1
                if batch:
                    poll_interval = base_interval
                    for task in batch:
                        runner = asyncio.create_task(
                            self._guarded_execute(task),
                            name=f"task-runner-{task.id}",
                        )
                        self._in_flight.add(runner)
                        runner.add_done_callback(self._in_flight.discard)
                else:
                    poll_interval = min(poll_interval * 2.0, self._max_poll_interval_seconds)
                if cycle % self._reclaim_every_cycles == 0:
                    await self._reclaim_expired_locks()
                if batch and len(batch) >= self._settings.BATCH_SIZE:
                    continue
                woke = await self._sleep_or_stop(poll_interval)
                if woke:
                    poll_interval = base_interval
        finally:
            await self._wake_subscription.stop()
            await self._drain(self._settings.DRAIN_TIMEOUT_SECONDS)
            await self._release_worker_locks()
            self._running = False
            logger.info("TaskWorker %s stopped", self.worker_id)

    async def shutdown(self, timeout_seconds: float | None = None) -> None:
        self.request_stop()
        timeout = timeout_seconds or self._settings.DRAIN_TIMEOUT_SECONDS
        deadline = time.monotonic() + timeout
        while self._running and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def _warm_pool(self) -> None:
        lease = dt.timedelta(seconds=self._settings.LOCK_TTL_SECONDS)

        def scratch_task() -> Task:
            return Task(
                idempotency_key=f"warmup-{uuid.uuid4().hex}"[:64],
                task_type="warmup.prepare",
                state=TaskState.PENDING,
                payload={},
                retry_count=0,
                max_retries=5,
                scheduled_at=dt.datetime.now(tz=dt.UTC),
            )

        async def warm_one() -> None:
            session = self._session_factory()
            try:
                repository = TaskRepository(session)
                await repository.acquire_next_batch(
                    self._settings.BATCH_SIZE, self.worker_id, lease
                )
                session.add(scratch_task())
                await session.flush()
                acquired = await repository.acquire_next_batch(1, self.worker_id, lease)
                if acquired:
                    scratch_id = acquired[0].id
                    await repository.mark_running(scratch_id, self.worker_id, lease)
                    await repository.record_success(scratch_id, self.worker_id, {}, 0.0)
                    await repository.extend_task_lock(scratch_id, self.worker_id, lease)
                session.add(scratch_task())
                await session.flush()
                pending_rows = await repository.acquire_next_batch(1, self.worker_id, lease)
                if pending_rows:
                    failure_id = pending_rows[0].id
                    await repository.mark_running(failure_id, self.worker_id, lease)
                    await repository.record_failure(failure_id, self.worker_id, {}, 0.0)
                    await session.execute(
                        sa_update(Task)
                        .where(Task.id == failure_id)
                        .values(state=TaskState.DEAD_LETTER)
                    )
                    await repository.requeue_dead_lettered_task(failure_id)
                session.add(
                    OutboxEvent(
                        aggregate_type="Task",
                        aggregate_id=uuid.uuid4(),
                        event_type="Warmup",
                        payload={"worker_id": self.worker_id},
                    )
                )
                await session.flush()
                await repository.reclaim_expired_locks()
                await repository.release_worker_locks(self.worker_id)
            except Exception:
                logger.warning("Pool warmup failed; statements will prepare on demand")
            finally:
                await session.rollback()
                await session.close()

        await asyncio.gather(
            *[warm_one() for _ in range(self._settings.CONCURRENCY)],
            return_exceptions=True,
        )

    async def _sleep_or_stop(self, seconds: float) -> bool:
        stop_waiter = asyncio.ensure_future(self._stop_event.wait())
        wake_waiter = asyncio.ensure_future(self._wake_event.wait())
        done, pending = await asyncio.wait(
            {stop_waiter, wake_waiter},
            timeout=seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for waiter in pending:
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter
        woke = wake_waiter in done
        if woke:
            self._wake_event.clear()
        return woke

    async def _poll_batch(self) -> list[Task]:
        lease_duration = dt.timedelta(seconds=self._settings.LOCK_TTL_SECONDS)
        try:
            async with session_scope(self._session_factory, "task.acquire_batch") as session:
                repository = TaskRepository(session)
                return await repository.acquire_next_batch(
                    batch_size=self._settings.BATCH_SIZE,
                    worker_id=self.worker_id,
                    lease_duration=lease_duration,
                )
        except Exception:
            logger.exception("Failed to acquire task batch; will retry next poll")
            return []

    async def _guarded_execute(self, task: Task) -> None:
        async with self._semaphore:
            try:
                await self._execute_with_lease(task)
            except Exception:
                logger.exception("Unhandled error while executing task %s", task.id)

    async def _execute_with_lease(self, task: Task) -> None:
        ttl_ms = int(self._settings.LOCK_TTL_SECONDS * 1000)
        try:
            async with distributed_lease(self._redis, str(task.id), self.worker_id, ttl_ms):
                active_worker_leases.inc()
                try:
                    await self._run_task(task)
                finally:
                    active_worker_leases.dec()
        except LeaseAcquisitionError:
            logger.warning(
                "Task %s is already leased by another worker; skipping", task.id
            )

    async def _run_task(self, task: Task) -> None:
        lease_duration = dt.timedelta(seconds=self._settings.LOCK_TTL_SECONDS)
        try:
            async with session_scope(self._session_factory, "task.mark_running") as session:
                repository = TaskRepository(session)
                await repository.mark_running(task.id, self.worker_id, lease_duration)
        except TaskOwnershipError:
            logger.warning("Task %s lease lost before execution started", task.id)
            return

        dispatch_lag = (_utc_now() - task.scheduled_at).total_seconds()
        queue_dispatch_lag_seconds.observe(max(dispatch_lag, 0.0))

        heartbeat_task = asyncio.create_task(
            self._db_heartbeat(task.id, lease_duration),
            name=f"task-heartbeat-{task.id}",
        )
        started_at = time.perf_counter()
        try:
            with task_execution_span(
                str(task.id), task.task_type, task.retry_count, self.worker_id
            ) as span:
                handler = TASK_HANDLER_REGISTRY.get(task.task_type)
                try:
                    if handler is None:
                        raise HandlerNotRegisteredError(task.task_type)
                    result = await self._invoke_handler(handler, task.payload)
                    duration = time.perf_counter() - started_at
                    result_payload = result if isinstance(result, dict) else {"result": result}
                    async with session_scope(
                        self._session_factory, "task.record_success"
                    ) as session:
                        repository = TaskRepository(session)
                        await repository.record_success(
                            task.id, self.worker_id, result_payload, duration
                        )
                    record_execution_success(task.task_type, duration)
                    logger.info(
                        "Task %s (%s) completed in %.4fs",
                        task.id,
                        task.task_type,
                        duration,
                    )
                except Exception as exc:
                    duration = time.perf_counter() - started_at
                    error_detail = {
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=20),
                        "worker_id": self.worker_id,
                        "failed_at": _utc_now().isoformat(),
                    }
                    dead_lettered = False
                    try:
                        async with session_scope(
                            self._session_factory, "task.record_failure"
                        ) as session:
                            repository = TaskRepository(session)
                            _, dead_lettered = await repository.record_failure(
                                task.id, self.worker_id, error_detail, duration
                            )
                    except Exception:
                        logger.exception(
                            "Failed to persist failure state for task %s; "
                            "lease expiry will requeue it",
                            task.id,
                        )
                    if dead_lettered:
                        record_task_dead_lettered(task.task_type, duration)
                        logger.error(
                            "Task %s (%s) moved to DEAD_LETTER after %d retries",
                            task.id,
                            task.task_type,
                            task.retry_count + 1,
                        )
                    else:
                        record_execution_failure(task.task_type, duration)
                        logger.error(
                            "Task %s (%s) failed: %s", task.id, task.task_type, exc
                        )
                    record_span_error(span, exc)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat_task

    async def _invoke_handler(
        self, handler: TaskHandler, payload: dict[str, Any]
    ) -> Any:
        if inspect.iscoroutinefunction(handler):
            return await handler(payload)
        return await asyncio.to_thread(handler, payload)

    async def _db_heartbeat(
        self, task_id: uuid.UUID, lease_duration: dt.timedelta
    ) -> None:
        interval = self._settings.HEARTBEAT_INTERVAL_SECONDS
        while True:
            await asyncio.sleep(interval)
            try:
                async with session_scope(
                    self._session_factory, "task.extend_lock"
                ) as session:
                    repository = TaskRepository(session)
                    extended = await repository.extend_task_lock(
                        task_id, self.worker_id, lease_duration
                    )
                if not extended:
                    logger.warning(
                        "Lost database lease for task %s; heartbeat stopping", task_id
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Heartbeat failed for task %s; retrying next interval", task_id
                )

    async def _reclaim_expired_locks(self) -> None:
        try:
            async with session_scope(self._session_factory, "task.reclaim_locks") as session:
                repository = TaskRepository(session)
                reclaimed = await repository.reclaim_expired_locks()
            if reclaimed:
                logger.info("Reclaimed %d expired task locks", reclaimed)
        except Exception:
            logger.exception("Failed to reclaim expired task locks")

    async def _drain(self, timeout_seconds: float) -> None:
        if not self._in_flight:
            return
        logger.info(
            "Draining %d in-flight tasks with %.1fs timeout",
            len(self._in_flight),
            timeout_seconds,
        )
        pending = set(self._in_flight)
        done, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
        for finished in done:
            exception = finished.exception()
            if exception is not None:
                logger.error("Task runner raised during drain: %s", exception)
        if still_pending:
            logger.warning(
                "Cancelling %d tasks that exceeded the drain timeout", len(still_pending)
            )
            for runner in still_pending:
                runner.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)

    async def _release_worker_locks(self) -> None:
        try:
            async with session_scope(
                self._session_factory, "task.release_worker_locks"
            ) as session:
                repository = TaskRepository(session)
                released = await repository.release_worker_locks(self.worker_id)
            if released:
                logger.info(
                    "Released %d database locks held by worker %s",
                    released,
                    self.worker_id,
                )
        except Exception:
            logger.exception("Failed to release worker locks on shutdown")
