from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from domain.state_machine import TaskState

__all__ = [
    "Base",
    "OutboxEvent",
    "Task",
    "TaskState",
    "utc_now",
]

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[TaskState] = mapped_column(
        SAEnum(
            TaskState,
            name="task_state",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
            create_constraint=False,
        ),
        nullable=False,
        default=TaskState.PENDING,
        server_default=text("'PENDING'"),
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    max_retries: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5, server_default=text("5")
    )
    backoff_base_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, default=2.0, server_default=text("2.0")
    )
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lock_expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scheduled_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CLOCK_TIMESTAMP()"),
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CLOCK_TIMESTAMP()"),
        onupdate=text("CLOCK_TIMESTAMP()"),
    )

    __table_args__ = (
        CheckConstraint("max_retries >= 0", name="max_retries_non_negative"),
        CheckConstraint("retry_count >= 0", name="retry_count_non_negative"),
        CheckConstraint("backoff_base_seconds > 0", name="backoff_base_positive"),
        Index(
            "ix_tasks_state_scheduled_at_lock_expires_at",
            "state",
            "scheduled_at",
            "lock_expires_at",
        ),
        Index("ix_tasks_idempotency_key", "idempotency_key", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id}, task_type={self.task_type!r}, "
            f"state={self.state!r}, retry_count={self.retry_count})"
        )


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    published: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("FALSE")
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CLOCK_TIMESTAMP()"),
    )

    __table_args__ = (
        Index(
            "ix_outbox_events_unpublished",
            "id",
            postgresql_where=text("published = FALSE"),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"OutboxEvent(id={self.id}, aggregate_type={self.aggregate_type!r}, "
            f"event_type={self.event_type!r}, published={self.published})"
        )
