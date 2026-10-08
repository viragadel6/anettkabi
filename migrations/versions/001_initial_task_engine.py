from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TASK_STATE_VALUES = ("PENDING", "ACQUIRED", "RUNNING", "COMPLETED", "FAILED", "DEAD_LETTER")


def upgrade() -> None:
    task_state = postgresql.ENUM(*TASK_STATE_VALUES, name="task_state", create_type=False)
    task_state.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "tasks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=128), nullable=False),
        sa.Column(
            "state",
            postgresql.ENUM(name="task_state", create_type=False),
            server_default="PENDING",
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_detail", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_retries", sa.Integer(), server_default="5", nullable=False),
        sa.Column(
            "backoff_base_seconds", sa.Float(), server_default="2.0", nullable=False
        ),
        sa.Column("locked_by", sa.String(length=128), nullable=True),
        sa.Column("lock_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CLOCK_TIMESTAMP()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CLOCK_TIMESTAMP()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tasks"),
        sa.CheckConstraint("max_retries >= 0", name="ck_tasks_max_retries_non_negative"),
        sa.CheckConstraint("retry_count >= 0", name="ck_tasks_retry_count_non_negative"),
        sa.CheckConstraint(
            "backoff_base_seconds > 0", name="ck_tasks_backoff_base_positive"
        ),
    )

    op.create_index(
        "ix_tasks_state_scheduled_at_lock_expires_at",
        "tasks",
        ["state", "scheduled_at", "lock_expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_tasks_idempotency_key",
        "tasks",
        ["idempotency_key"],
        unique=True,
    )

    op.create_table(
        "outbox_events",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            sa.Identity(always=True, start=1, increment=1),
            nullable=False,
        ),
        sa.Column("aggregate_type", sa.String(length=64), nullable=False),
        sa.Column("aggregate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "published", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CLOCK_TIMESTAMP()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_outbox_events"),
    )

    op.create_index(
        "ix_outbox_events_unpublished",
        "outbox_events",
        ["id"],
        unique=False,
        postgresql_where=sa.text("published = FALSE"),
    )


def downgrade() -> None:
    op.drop_index("ix_outbox_events_unpublished", table_name="outbox_events")
    op.drop_table("outbox_events")
    op.drop_index("ix_tasks_idempotency_key", table_name="tasks")
    op.drop_index("ix_tasks_state_scheduled_at_lock_expires_at", table_name="tasks")
    op.drop_table("tasks")
    postgresql.ENUM(name="task_state").drop(op.get_bind(), checkfirst=True)
