"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-01-05 10:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("key_prefix", sa.String(length=16), nullable=False),
        sa.Column("key_hash", sa.String(length=512), nullable=False),
        sa.Column("scopes", postgresql.JSONB(), server_default='["predictions:write"]', nullable=False),
        sa.Column("rate_limit_rpm", sa.Integer(), nullable=False),
        sa.Column("concurrency_limit", sa.Integer(), nullable=False),
        sa.Column("monthly_seconds_quota", sa.Numeric(12, 2), nullable=False),
        sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_api_keys")),
        sa.UniqueConstraint("key_prefix", name=op.f("uq_api_keys_key_prefix")),
    )
    op.create_index(op.f("ix_api_keys_key_prefix"), "api_keys", ["key_prefix"])
    prediction_status = postgresql.ENUM(
        "created",
        "queued",
        "processing",
        "completed",
        "failed",
        "canceled",
        name="vsfx_prediction_status",
    )
    prediction_status.create(op.get_bind())
    asset_kind = postgresql.ENUM(
        "source_video", "output_video", "debug_audio", name="vsfx_asset_kind"
    )
    asset_kind.create(op.get_bind())
    op.create_table(
        "assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("kind", asset_kind, nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("bucket", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("duration_s", sa.Numeric(10, 3), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Numeric(10, 3), nullable=True),
        sa.Column("has_audio", sa.Boolean(), nullable=True),
        sa.Column("video_codec", sa.String(length=64), nullable=True),
        sa.Column("audio_codec", sa.String(length=64), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_assets")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_assets_storage_key")),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=26), nullable=False),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("status", prediction_status, nullable=False),
        sa.Column("input", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("outputs", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("has_nsfw_contents", postgresql.JSONB(), server_default="[]", nullable=False),
        sa.Column("error", sa.Text(), nullable=False, server_default=""),
        sa.Column("error_code", sa.Text(), nullable=False, server_default=""),
        sa.Column("progress", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("stage", sa.Text(), nullable=False, server_default=""),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("source_video_asset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("assets.id", use_alter=True), nullable=True),
        sa.Column("output_asset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("assets.id", use_alter=True), nullable=True),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("execution_time_ms", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("timings", postgresql.JSONB(), server_default="{}", nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("canceled_requested", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("input_hash", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], name=op.f("fk_predictions_api_key_id_api_keys")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name=op.f("ck_predictions_progress_range")),
    )
    op.create_index(op.f("ix_predictions_api_key_id"), "predictions", ["api_key_id"])
    op.create_index(
        "ix_predictions_api_key_created",
        "predictions",
        ["api_key_id", sa.text("created_at DESC")],
    )
    op.create_index("ix_predictions_status_heartbeat", "predictions", ["status", "heartbeat_at"])
    op.create_index("ix_predictions_idempotency", "predictions", ["idempotency_key"])
    op.create_unique_constraint(
        "uq_predictions_key_idempotency", "predictions", ["api_key_id", "idempotency_key"]
    )
    op.create_table(
        "usage_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_id", sa.String(length=26), nullable=True),
        sa.Column("billed_seconds", sa.Numeric(10, 3), nullable=False),
        sa.Column("gpu_ms", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["api_key_id"], ["api_keys.id"], name=op.f("fk_usage_events_api_key_id_api_keys")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_usage_events")),
    )
    op.create_index(
        "ix_usage_events_monthly",
        "usage_events",
        ["api_key_id", sa.text("date_trunc('month', created_at)")],
        postgresql_using="btree",
    )


def downgrade() -> None:
    op.drop_index("ix_usage_events_monthly", table_name="usage_events")
    op.drop_table("usage_events")
    op.drop_constraint("uq_predictions_key_idempotency", "predictions", type_="unique")
    op.drop_index("ix_predictions_idempotency", table_name="predictions")
    op.drop_index("ix_predictions_status_heartbeat", table_name="predictions")
    op.drop_index("ix_predictions_api_key_created", table_name="predictions")
    op.drop_index(op.f("ix_predictions_api_key_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("assets")
    op.execute("DROP TYPE IF EXISTS vsfx_asset_kind")
    op.execute("DROP TYPE IF EXISTS vsfx_prediction_status")
    op.drop_index(op.f("ix_api_keys_key_prefix"), table_name="api_keys")
    op.drop_table("api_keys")
