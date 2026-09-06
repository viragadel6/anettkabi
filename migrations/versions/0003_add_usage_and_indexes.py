"""add usage and indexes

Revision ID: 0003
Revises: 0002
Create Date: 2026-01-05 10:02:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_predictions_canceled_requested",
        "predictions",
        ["canceled_requested"],
        postgresql_where=sa.text("canceled_requested AND deleted_at IS NULL"),
    )
    op.create_index(
        "ix_predictions_created_at",
        "predictions",
        ["created_at"],
    )
    op.create_index(
        "ix_assets_expires_at",
        "assets",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL AND deleted_at IS NULL"),
    )
    op.execute(
        "ALTER TABLE usage_events ALTER COLUMN billed_seconds SET DEFAULT 0"
    )
    op.execute("ALTER TABLE usage_events ALTER COLUMN gpu_ms SET DEFAULT 0")


def downgrade() -> None:
    op.drop_index("ix_assets_expires_at", table_name="assets")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_canceled_requested", table_name="predictions")
