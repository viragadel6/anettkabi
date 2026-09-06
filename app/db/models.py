"""ORM models: api_keys, predictions, assets, usage_events, webhook_deliveries."""

from __future__ import annotations

import datetime as dt
import enum
from typing import Any

from sqlalchemy import (
    SMALLINT,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utc_now_column

__all__ = [
    "ApiKey",
    "Asset",
    "AssetKind",
    "Prediction",
    "PredictionStatus",
    "UsageEvent",
    "WebhookDelivery",
]


class PredictionStatus(enum.StrEnum):
    """Lifecycle states of a prediction."""

    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AssetKind(enum.StrEnum):
    """Kinds of stored assets."""

    SOURCE_VIDEO = "source_video"
    OUTPUT_VIDEO = "output_video"
    DEBUG_AUDIO = "debug_audio"


TERMINAL_STATUSES = (
    PredictionStatus.COMPLETED,
    PredictionStatus.FAILED,
    PredictionStatus.CANCELED,
)


class ApiKey(Base):
    """An API key: prefix is stored for lookup; the secret only as an argon2 hash."""

    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    key_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default='["predictions:write"]')
    rate_limit_rpm: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    concurrency_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    monthly_seconds_quota: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False, default=36000)
    disabled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now_column, server_default=func.now())
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="api_key")

    @property
    def is_active(self) -> bool:
        """Return True when the key is enabled."""
        return self.disabled_at is None


class Prediction(Base):
    """A prediction job record with full lifecycle bookkeeping."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_api_key_created", "api_key_id", "created_at"),
        Index("ix_predictions_status_heartbeat", "status", "heartbeat_at"),
        Index("ix_predictions_idempotency", "idempotency_key"),
        UniqueConstraint("api_key_id", "idempotency_key", name="uq_predictions_key_idempotency"),
        CheckConstraint("progress >= 0 AND progress <= 100", name="progress_range"),
    )

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    api_key_id: Mapped[UUID] = mapped_column(ForeignKey("api_keys.id"), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[PredictionStatus] = mapped_column(
        Enum(PredictionStatus, name="vsfx_prediction_status", native_enum=True, length=16),
        nullable=False,
        default=PredictionStatus.CREATED,
    )
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    outputs: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default="[]")
    has_nsfw_contents: Mapped[list[bool]] = mapped_column(JSONB, nullable=False, server_default="[]")
    error: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    error_code: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    progress: Mapped[int] = mapped_column(SMALLINT, nullable=False, default=0, server_default="0")
    stage: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_video_asset_id: Mapped[UUID | None] = mapped_column(ForeignKey("assets.id", use_alter=True), nullable=True)
    output_asset_id: Mapped[UUID | None] = mapped_column(ForeignKey("assets.id", use_alter=True), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    queued_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now_column, server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now_column, onupdate=utc_now_column, server_default=func.now())
    execution_time_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default="0")
    timings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    canceled_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    api_key: Mapped[ApiKey] = relationship(back_populates="predictions", foreign_keys=[api_key_id])
    source_video_asset: Mapped[Asset | None] = relationship(foreign_keys=[source_video_asset_id], lazy="joined")
    output_asset: Mapped[Asset | None] = relationship(foreign_keys=[output_asset_id], lazy="joined")
    webhook_deliveries: Mapped[list[WebhookDelivery]] = relationship(back_populates="prediction")

    @property
    def is_terminal(self) -> bool:
        """Return True when the prediction reached a terminal state."""
        return self.status in TERMINAL_STATUSES


class Asset(Base):
    """A stored object (source video, output video, auxiliary audio)."""

    __tablename__ = "assets"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    kind: Mapped[AssetKind] = mapped_column(
        Enum(AssetKind, name="vsfx_asset_kind", native_enum=True, length=32), nullable=False
    )
    storage_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False, default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    duration_s: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Numeric(10, 3), nullable=True)
    has_audio: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    video_codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    audio_codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now_column, server_default=func.now())
    deleted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UsageEvent(Base):
    """A billed usage record for one prediction execution."""

    __tablename__ = "usage_events"
    __table_args__ = (
        Index("ix_usage_events_monthly", "api_key_id", func.date_trunc("month", "created_at")),
    )

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    api_key_id: Mapped[UUID] = mapped_column(ForeignKey("api_keys.id"), nullable=False)
    prediction_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    billed_seconds: Mapped[float] = mapped_column(Numeric(10, 3), nullable=False, default=0)
    gpu_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now_column, server_default=func.now())


class WebhookDelivery(Base):
    """A persisted webhook delivery attempt."""

    __tablename__ = "webhook_deliveries"
    __table_args__ = (
        Index("ix_webhook_deliveries_prediction", "prediction_id", "attempt"),
    )

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    prediction_id: Mapped[str] = mapped_column(String(26), ForeignKey("predictions.id"), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_snippet: Mapped[str] = mapped_column(Text, nullable=False, default="")
    signature: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    scheduled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    delivered_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    prediction: Mapped[Prediction] = relationship(back_populates="webhook_deliveries")
