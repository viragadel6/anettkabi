from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from coordination.wakeup import publish_task_submitted
from database import check_database_health, session_scope
from domain.exceptions import InvalidStateTransitionError, TaskNotFoundError
from observability.metrics import record_task_submission
from repositories.outbox_repository import OutboxRepository
from repositories.task_repository import TaskRepository
from schemas import (
    TaskCreateRequest,
    TaskDetailResponse,
    TaskEventPayload,
    TaskResponse,
)

__all__ = ["router"]

router = APIRouter()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def _fallback_idempotency_key(request_body: TaskCreateRequest) -> str:
    canonical = json.dumps(
        {
            "task_type": request_body.task_type,
            "payload": request_body.payload,
            "delay_seconds": request_body.delay_seconds,
            "max_retries": request_body.max_retries,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@router.post(
    "/v1/tasks",
    response_model=TaskResponse,
    status_code=201,
    responses={200: {"model": TaskResponse, "description": "Existing task returned"}},
)
async def create_task(request: Request, body: TaskCreateRequest) -> JSONResponse:
    session_factory = request.app.state.session_factory
    idempotency_key = (
        getattr(request.state, "idempotency_key", None)
        or _fallback_idempotency_key(body)
    )
    async with session_scope(session_factory, "api.create_task") as session:
        repository = TaskRepository(session)
        task, created = await repository.create_task_with_idempotency(
            task_type=body.task_type,
            payload=body.payload,
            idempotency_key=idempotency_key,
            delay_seconds=body.delay_seconds,
            max_retries=body.max_retries,
        )
        response_model = TaskResponse.from_model(task)
    record_task_submission(body.task_type)
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        await publish_task_submitted(
            redis_client,
            body.task_type,
            request.app.state.settings.TASK_SUBMISSION_CHANNEL,
        )
    status_code = 201 if created else 200
    return JSONResponse(
        content=response_model.model_dump(mode="json"),
        status_code=status_code,
        headers={"X-Task-Created": "true" if created else "false"},
    )


@router.get("/v1/tasks/{task_id}", response_model=TaskDetailResponse)
async def get_task(request: Request, task_id: uuid.UUID) -> TaskDetailResponse:
    session_factory = request.app.state.session_factory
    async with session_scope(session_factory, "api.get_task") as session:
        repository = TaskRepository(session)
        task = await repository.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        outbox_repository = OutboxRepository(session)
        events = await outbox_repository.fetch_events_for_aggregate(task_id)
    return TaskDetailResponse(
        task=TaskResponse.from_model(task),
        execution_history=[TaskEventPayload.model_validate(event) for event in events],
    )


@router.post("/v1/tasks/{task_id}/retry", response_model=TaskResponse)
async def retry_task(request: Request, task_id: uuid.UUID) -> TaskResponse:
    session_factory = request.app.state.session_factory
    async with session_scope(session_factory, "api.retry_task") as session:
        repository = TaskRepository(session)
        try:
            requeued = await repository.requeue_dead_lettered_task(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"Task {task_id} not found"
            ) from exc
        except InvalidStateTransitionError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Task {task_id} in state {exc.current_state!s} cannot be "
                    "retried; only DEAD_LETTER tasks can be requeued"
                ),
            ) from exc
        response_model = TaskResponse.from_model(requeued)
    redis_client = getattr(request.app.state, "redis_client", None)
    if redis_client is not None:
        await publish_task_submitted(
            redis_client,
            response_model.task_type,
            request.app.state.settings.TASK_SUBMISSION_CHANNEL,
        )
    return response_model


@router.get("/health/live")
async def health_live(request: Request) -> dict[str, object]:
    relay = getattr(request.app.state, "relay", None)
    return {
        "status": "live",
        "checked_at": _utc_now().isoformat(),
        "outbox_relay_state": relay.state if relay is not None else "absent",
    }


@router.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    checks: dict[str, bool] = {}
    session_factory = request.app.state.session_factory
    settings = request.app.state.settings
    checks["database"] = await check_database_health(session_factory)
    try:
        redis_client = request.app.state.redis_client
        await redis_client.ping()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False
    relay = getattr(request.app.state, "relay", None)
    if relay is None or not relay.running:
        checks["relay_loop"] = False
    else:
        checks["relay_loop"] = relay.healthy
    worker_loop_ok = False
    if checks["database"]:
        try:
            threshold = settings.worker.PENDING_STALENESS_THRESHOLD_SECONDS
            async with session_scope(session_factory, "api.health_staleness") as session:
                repository = TaskRepository(session)
                stale_since = await repository.oldest_stale_pending_at(threshold)
            worker_loop_ok = stale_since is None
        except Exception:
            worker_loop_ok = False
    checks["worker_loop"] = worker_loop_ok
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "checked_at": _utc_now().isoformat(),
            "checks": checks,
        },
    )
