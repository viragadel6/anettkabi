from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.state_machine import TaskState
from models import Task

__all__ = [
    "TaskCreateRequest",
    "TaskDetailResponse",
    "TaskEventPayload",
    "TaskResponse",
]


class TaskCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_type: str = Field(min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)
    delay_seconds: float = Field(default=0.0, ge=0.0, le=86400.0)
    max_retries: int = Field(default=5, ge=0, le=100)

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: str) -> str:
        if not all(char.isalnum() or char in {".", "_", "-"} for char in value):
            raise ValueError(
                "task_type may only contain letters, digits, dots, underscores, and hyphens"
            )
        return value


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    idempotency_key: str
    task_type: str
    state: TaskState
    payload: dict[str, Any]
    result: dict[str, Any] | None = None
    error_detail: dict[str, Any] | None = None
    retry_count: int
    max_retries: int
    scheduled_at: dt.datetime
    created_at: dt.datetime
    updated_at: dt.datetime

    @classmethod
    def from_model(cls, task: Task) -> TaskResponse:
        return cls.model_validate(task)


class TaskEventPayload(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    aggregate_type: str
    aggregate_id: uuid.UUID
    event_type: str
    payload: dict[str, Any]
    published: bool
    created_at: dt.datetime


class TaskDetailResponse(BaseModel):
    task: TaskResponse
    execution_history: list[TaskEventPayload]
