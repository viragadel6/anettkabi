from __future__ import annotations

import datetime as dt
import random
import uuid
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from domain.exceptions import (
    InvalidStateTransitionError,
    TaskNotFoundError,
    TaskOwnershipError,
)
from domain.state_machine import TaskState, TaskStateMachine
from models import OutboxEvent, Task

__all__ = [
    "BACKOFF_JITTER_HIGH",
    "BACKOFF_JITTER_LOW",
    "BACKOFF_MAX_SECONDS",
    "TaskRepository",
]

BACKOFF_MAX_SECONDS = 3600.0
BACKOFF_JITTER_LOW = 0.8
BACKOFF_JITTER_HIGH = 1.2


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def compute_backoff_seconds(backoff_base_seconds: float, retry_count: int) -> float:
    raw = backoff_base_seconds * (2 ** retry_count)
    capped = min(BACKOFF_MAX_SECONDS, raw)
    return capped * random.uniform(BACKOFF_JITTER_LOW, BACKOFF_JITTER_HIGH)


class TaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_task_with_idempotency(
        self,
        task_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
        delay_seconds: float = 0.0,
        max_retries: int = 5,
        scheduled_at: dt.datetime | None = None,
    ) -> tuple[Task, bool]:
        if scheduled_at is None:
            scheduled_at = _utc_now() + dt.timedelta(seconds=delay_seconds)
        statement = (
            pg_insert(Task)
            .values(
                idempotency_key=idempotency_key,
                task_type=task_type,
                state=TaskState.PENDING,
                payload=payload,
                retry_count=0,
                max_retries=max_retries,
                scheduled_at=scheduled_at,
            )
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
            .returning(Task)
        )
        result = await self._session.execute(statement)
        created_row = result.scalar_one_or_none()
        if created_row is not None:
            await self._session.flush()
            await self._session.refresh(created_row)
            self._session.add(
                OutboxEvent(
                    aggregate_type="Task",
                    aggregate_id=created_row.id,
                    event_type="TaskCreated",
                    payload={
                        "task_id": str(created_row.id),
                        "task_type": created_row.task_type,
                        "idempotency_key": created_row.idempotency_key,
                        "state": TaskState.PENDING.value,
                        "scheduled_at": created_row.scheduled_at.isoformat(),
                    },
                )
            )
            return created_row, True
        existing = await self.get_by_idempotency_key(idempotency_key)
        if existing is None:
            raise TaskNotFoundError(idempotency_key)
        return existing, False

    async def get_task(self, task_id: uuid.UUID) -> Task | None:
        result = await self._session.execute(
            select(Task).where(Task.id == task_id)
        )
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, idempotency_key: str) -> Task | None:
        result = await self._session.execute(
            select(Task).where(Task.idempotency_key == idempotency_key)
        )
        return result.scalar_one_or_none()

    async def acquire_next_batch(
        self,
        batch_size: int,
        worker_id: str,
        lease_duration: dt.timedelta,
    ) -> list[Task]:
        statement = (
            select(Task)
            .where(
                Task.state == TaskState.PENDING,
                Task.scheduled_at <= func.now(),
                or_(
                    Task.lock_expires_at.is_(None),
                    Task.lock_expires_at < func.now(),
                ),
            )
            .order_by(Task.scheduled_at.asc(), Task.id.asc())
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(statement)
        tasks = list(result.scalars().all())
        for task in tasks:
            TaskStateMachine.validate_transition(task.state, TaskState.ACQUIRED)
            task.state = TaskState.ACQUIRED
            task.locked_by = worker_id
            task.lock_expires_at = func.now() + lease_duration
            self._session.add(task)
            self._session.add(
                OutboxEvent(
                    aggregate_type="Task",
                    aggregate_id=task.id,
                    event_type="TaskAcquired",
                    payload={
                        "task_id": str(task.id),
                        "task_type": task.task_type,
                        "worker_id": worker_id,
                        "state": TaskState.ACQUIRED.value,
                        "lease_seconds": lease_duration.total_seconds(),
                    },
                )
            )
        await self._session.flush()
        for task in tasks:
            await self._session.refresh(task)
        return tasks

    async def mark_running(
        self,
        task_id: uuid.UUID,
        worker_id: str,
        lease_duration: dt.timedelta,
    ) -> Task:
        task = await self._get_owned_task(task_id, worker_id)
        TaskStateMachine.validate_transition(task.state, TaskState.RUNNING)
        task.state = TaskState.RUNNING
        task.lock_expires_at = _utc_now() + lease_duration
        await self._session.flush()
        await self._session.refresh(task)
        return task

    async def extend_task_lock(
        self, task_id: uuid.UUID, worker_id: str, lease_duration: dt.timedelta
    ) -> bool:
        statement = (
            update(Task)
            .where(
                Task.id == task_id,
                Task.locked_by == worker_id,
                Task.state.in_([TaskState.ACQUIRED, TaskState.RUNNING]),
            )
            .values(lock_expires_at=func.now() + lease_duration)
        )
        result = await self._session.execute(statement)
        return result.rowcount > 0

    async def record_success(
        self,
        task_id: uuid.UUID,
        worker_id: str,
        result_payload: dict[str, Any] | None,
        duration_seconds: float,
    ) -> Task:
        task = await self._get_owned_task(task_id, worker_id)
        TaskStateMachine.validate_transition(task.state, TaskState.COMPLETED)
        task.state = TaskState.COMPLETED
        task.result = result_payload if result_payload is not None else {}
        task.error_detail = None
        task.locked_by = None
        task.lock_expires_at = None
        self._session.add(
            OutboxEvent(
                aggregate_type="Task",
                aggregate_id=task.id,
                event_type="TaskCompleted",
                payload={
                    "task_id": str(task.id),
                    "task_type": task.task_type,
                    "worker_id": worker_id,
                    "state": TaskState.COMPLETED.value,
                    "retry_count": task.retry_count,
                    "duration_seconds": duration_seconds,
                },
            )
        )
        await self._session.flush()
        await self._session.refresh(task)
        return task

    async def record_failure(
        self,
        task_id: uuid.UUID,
        worker_id: str,
        error_detail: dict[str, Any],
        duration_seconds: float,
    ) -> tuple[Task, bool]:
        task = await self._get_owned_task(task_id, worker_id)
        TaskStateMachine.validate_transition(task.state, TaskState.FAILED)
        next_retry_count = task.retry_count + 1
        exhausted = next_retry_count >= task.max_retries
        task.retry_count = next_retry_count
        task.error_detail = error_detail
        task.locked_by = None
        task.lock_expires_at = None
        if exhausted:
            TaskStateMachine.validate_transition(task.state, TaskState.DEAD_LETTER)
            task.state = TaskState.DEAD_LETTER
            event_type = "TaskDeadLettered"
        else:
            TaskStateMachine.validate_transition(TaskState.FAILED, TaskState.PENDING)
            backoff_seconds = compute_backoff_seconds(
                task.backoff_base_seconds, task.retry_count
            )
            task.state = TaskState.PENDING
            task.scheduled_at = _utc_now() + dt.timedelta(seconds=backoff_seconds)
            event_type = "TaskFailed"
        self._session.add(
            OutboxEvent(
                aggregate_type="Task",
                aggregate_id=task.id,
                event_type=event_type,
                payload={
                    "task_id": str(task.id),
                    "task_type": task.task_type,
                    "worker_id": worker_id,
                    "state": task.state.value
                    if isinstance(task.state, TaskState)
                    else str(task.state),
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                    "duration_seconds": duration_seconds,
                    "error_detail": error_detail,
                },
            )
        )
        await self._session.flush()
        await self._session.refresh(task)
        return task, exhausted

    async def requeue_dead_lettered_task(self, task_id: uuid.UUID) -> Task:
        result = await self._session.execute(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        task = result.scalar_one_or_none()
        if task is None:
            raise TaskNotFoundError(task_id)
        TaskStateMachine.validate_transition(task.state, TaskState.PENDING)
        task.state = TaskState.PENDING
        task.retry_count = 0
        task.error_detail = None
        task.locked_by = None
        task.lock_expires_at = None
        task.scheduled_at = _utc_now()
        self._session.add(
            OutboxEvent(
                aggregate_type="Task",
                aggregate_id=task.id,
                event_type="TaskRequeued",
                payload={
                    "task_id": str(task.id),
                    "task_type": task.task_type,
                    "state": TaskState.PENDING.value,
                    "reason": "manual_retry",
                },
            )
        )
        await self._session.flush()
        await self._session.refresh(task)
        return task

    async def reclaim_expired_locks(self) -> int:
        statement = (
            update(Task)
            .where(
                Task.state.in_([TaskState.ACQUIRED, TaskState.RUNNING]),
                Task.lock_expires_at.is_not(None),
                Task.lock_expires_at < func.now(),
            )
            .values(
                state=TaskState.PENDING,
                locked_by=None,
                lock_expires_at=None,
                scheduled_at=func.now(),
            )
        )
        result = await self._session.execute(statement)
        return int(result.rowcount)

    async def release_worker_locks(self, worker_id: str) -> int:
        statement = (
            update(Task)
            .where(
                Task.locked_by == worker_id,
                Task.state.in_([TaskState.ACQUIRED, TaskState.RUNNING]),
            )
            .values(
                state=TaskState.PENDING,
                locked_by=None,
                lock_expires_at=None,
                scheduled_at=func.now(),
            )
        )
        result = await self._session.execute(statement)
        return int(result.rowcount)

    async def count_by_state(self) -> dict[str, int]:
        result = await self._session.execute(
            select(Task.state, func.count(Task.id)).group_by(Task.state)
        )
        return {str(state): int(count) for state, count in result.all()}

    async def _get_owned_task(self, task_id: uuid.UUID, worker_id: str) -> Task:
        result = await self._session.execute(
            select(Task).where(Task.id == task_id).with_for_update()
        )
        task = result.scalar_one_or_none()
        if task is None:
            raise TaskNotFoundError(task_id)
        if task.locked_by != worker_id:
            raise TaskOwnershipError(task_id, worker_id, task.locked_by)
        return task
