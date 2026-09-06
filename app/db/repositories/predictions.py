"""Prediction repository with atomic, guarded state transitions."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Prediction, PredictionStatus

__all__ = ["TRANSITIONS", "PredictionRepository"]

_SENTINEL = object()
TRANSITIONS: dict[tuple[PredictionStatus, PredictionStatus], bool] = {
    (PredictionStatus.CREATED, PredictionStatus.QUEUED): True,
    (PredictionStatus.CREATED, PredictionStatus.CANCELED): True,
    (PredictionStatus.QUEUED, PredictionStatus.PROCESSING): True,
    (PredictionStatus.QUEUED, PredictionStatus.CANCELED): True,
    (PredictionStatus.QUEUED, PredictionStatus.QUEUED): True,
    (PredictionStatus.PROCESSING, PredictionStatus.COMPLETED): True,
    (PredictionStatus.PROCESSING, PredictionStatus.FAILED): True,
    (PredictionStatus.PROCESSING, PredictionStatus.CANCELED): True,
    (PredictionStatus.PROCESSING, PredictionStatus.PROCESSING): True,
}


class PredictionRepository:
    """Persistence and guarded transitions for predictions rows."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session.

        Parameters:
            session: Active AsyncSession.
        """
        self._session = session

    async def create(
        self,
        *,
        prediction_id: str,
        api_key_id: uuid.UUID,
        model: str,
        input_payload: dict[str, Any],
        input_hash: str,
        webhook_url: str | None,
        idempotency_key: str | None,
    ) -> Prediction:
        """Insert a new prediction in state `created`.

        Parameters:
            prediction_id: 26-char ULID primary key.
            api_key_id: Owning key.
            model: Model identifier.
            input_payload: Normalized input (secrets stripped).
            input_hash: SHA-256 of the normalized input for idempotency.
            webhook_url: Optional webhook target.
            idempotency_key: Optional client idempotency key.

        Returns:
            The persisted Prediction row.
        """
        row = Prediction(
            id=prediction_id,
            api_key_id=api_key_id,
            model=model,
            status=PredictionStatus.CREATED,
            input=input_payload,
            input_hash=input_hash,
            webhook_url=webhook_url,
            idempotency_key=idempotency_key,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, prediction_id: str) -> Prediction | None:
        """Fetch a prediction by id, excluding soft-deleted rows.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            The Prediction or None.
        """
        result = await self._session.execute(
            select(Prediction).where(Prediction.id == prediction_id, Prediction.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def mark_queued(self, prediction_id: str) -> bool:
        """Atomically move created → queued.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            True when the transition happened.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id, Prediction.status == PredictionStatus.CREATED)
            .values(status=PredictionStatus.QUEUED, queued_at=func.now(), progress=1, stage="queued")
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def mark_processing(self, prediction_id: str, worker_id: str, attempts: int) -> bool:
        """Atomically move queued → processing and stamp worker identity.

        Parameters:
            prediction_id: Prediction id.
            worker_id: Consuming worker name.
            attempts: Attempt counter after claim.

        Returns:
            True when the transition happened.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id, Prediction.status == PredictionStatus.QUEUED)
            .values(
                status=PredictionStatus.PROCESSING,
                started_at=func.now(),
                heartbeat_at=func.now(),
                worker_id=worker_id,
                attempts=attempts,
                stage="download",
                progress=2,
                error="",
                error_code="",
            )
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def requeue(self, prediction_id: str, *, attempt: int) -> bool:
        """Move processing → queued for a retry attempt.

        Parameters:
            prediction_id: Prediction id.
            attempt: Next attempt number.

        Returns:
            True when the transition happened.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id, Prediction.status == PredictionStatus.PROCESSING)
            .values(
                status=PredictionStatus.QUEUED,
                worker_id=None,
                heartbeat_at=None,
                attempts=attempt,
                stage="queued",
            )
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def complete(
        self,
        prediction_id: str,
        *,
        outputs: list[str],
        has_nsfw_contents: list[bool],
        output_asset_id: uuid.UUID | None,
        timings: dict[str, Any],
        execution_time_ms: int,
    ) -> bool:
        """Atomically move processing → completed with outputs.

        Parameters:
            prediction_id: Prediction id.
            outputs: Result URLs.
            has_nsfw_contents: One boolean per output.
            output_asset_id: Optional output asset FK.
            timings: Stage timing map.
            execution_time_ms: Total wall time.

        Returns:
            True when the transition happened.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id, Prediction.status == PredictionStatus.PROCESSING)
            .values(
                status=PredictionStatus.COMPLETED,
                outputs=outputs,
                has_nsfw_contents=has_nsfw_contents,
                output_asset_id=output_asset_id,
                timings=timings,
                execution_time_ms=execution_time_ms,
                completed_at=func.now(),
                progress=100,
                stage="completed",
                error="",
                error_code="",
                worker_id=None,
            )
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def fail(
        self,
        prediction_id: str,
        *,
        error: str,
        error_code: str,
        allowed_statuses: tuple[PredictionStatus, ...] = (
            PredictionStatus.PROCESSING,
            PredictionStatus.QUEUED,
            PredictionStatus.CREATED,
        ),
    ) -> bool:
        """Atomically move a non-terminal state → failed.

        Parameters:
            prediction_id: Prediction id.
            error: Human-readable safe message.
            error_code: Taxonomy code.
            allowed_statuses: States from which failure is legal.

        Returns:
            True when the transition happened.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(
                Prediction.id == prediction_id,
                Prediction.status.in_(allowed_statuses),
            )
            .values(
                status=PredictionStatus.FAILED,
                error=error,
                error_code=error_code,
                completed_at=func.now(),
                outputs=[],
                worker_id=None,
            )
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def cancel_requested(self, prediction_id: str) -> Prediction | None:
        """Set the cancel-request flag; returns the row when settable.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            The Prediction when the request was recorded, None otherwise.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(
                Prediction.id == prediction_id,
                Prediction.deleted_at.is_(None),
                Prediction.canceled_requested.is_(False),
                Prediction.status.notin_(
                    [
                        PredictionStatus.COMPLETED,
                        PredictionStatus.FAILED,
                        PredictionStatus.CANCELED,
                    ]
                ),
            )
            .values(canceled_requested=True)
            .returning(Prediction)
        )
        return result.scalar_one_or_none()

    async def mark_canceled(self, prediction_id: str) -> bool:
        """Atomically move a live state → canceled.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            True when the transition happened.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(
                Prediction.id == prediction_id,
                Prediction.status.in_(
                    [
                        PredictionStatus.CREATED,
                        PredictionStatus.QUEUED,
                        PredictionStatus.PROCESSING,
                    ]
                ),
            )
            .values(
                status=PredictionStatus.CANCELED,
                completed_at=func.now(),
                outputs=[],
                worker_id=None,
                stage="canceled",
            )
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def heartbeat(self, prediction_id: str, *, stage: str | None = None, progress: int | None = None) -> bool:
        """Refresh heartbeat_at (and optionally stage/progress) while processing.

        Parameters:
            prediction_id: Prediction id.
            stage: New stage label when provided.
            progress: New progress value when provided.

        Returns:
            True when a processing row was updated.
        """
        values: dict[str, Any] = {"heartbeat_at": func.now()}
        if stage is not None:
            values["stage"] = stage
        if progress is not None:
            values["progress"] = max(0, min(100, progress))
        result = await self._session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id, Prediction.status == PredictionStatus.PROCESSING)
            .values(**values)
            .returning(Prediction.id)
        )
        return result.scalar_one_or_none() is not None

    async def update_timings(self, prediction_id: str, timings: dict[str, Any]) -> None:
        """Merge timing entries into the timings JSONB column.

        Parameters:
            prediction_id: Prediction id.
            timings: Timing keys to merge.
        """
        row = await self.get(prediction_id)
        if row is None:
            return
        merged = dict(row.timings or {})
        merged.update(timings)
        await self._session.execute(
            update(Prediction).where(Prediction.id == prediction_id).values(timings=merged)
        )

    async def set_source_asset(self, prediction_id: str, asset_id: uuid.UUID) -> None:
        """Attach the normalized source asset to the prediction.

        Parameters:
            prediction_id: Prediction id.
            asset_id: Asset UUID.
        """
        await self._session.execute(
            update(Prediction)
            .where(Prediction.id == prediction_id)
            .values(source_video_asset_id=asset_id)
        )

    async def soft_delete(self, prediction_id: str) -> Prediction | None:
        """Soft-delete a prediction; only legal from terminal states.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            The row when deleted, None when not found or non-terminal.
        """
        result = await self._session.execute(
            update(Prediction)
            .where(
                Prediction.id == prediction_id,
                Prediction.deleted_at.is_(None),
                Prediction.status.in_(
                    [
                        PredictionStatus.COMPLETED,
                        PredictionStatus.FAILED,
                        PredictionStatus.CANCELED,
                    ]
                ),
            )
            .values(deleted_at=func.now())
            .returning(Prediction)
        )
        return result.scalar_one_or_none()

    async def find_idempotent(
        self, api_key_id: uuid.UUID, idempotency_key: str, window_s: int
    ) -> Prediction | None:
        """Find a recent prediction with the same key for replay.

        Parameters:
            api_key_id: Caller key.
            idempotency_key: Client-supplied key.
            window_s: Replay window in seconds.

        Returns:
            The most recent matching Prediction or None.
        """
        cutoff = dt.datetime.now(tz=dt.UTC) - dt.timedelta(seconds=window_s)
        result = await self._session.execute(
            select(Prediction)
            .where(
                Prediction.api_key_id == api_key_id,
                Prediction.idempotency_key == idempotency_key,
                Prediction.deleted_at.is_(None),
                Prediction.created_at >= cutoff,
            )
            .order_by(Prediction.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_for_key(
        self,
        api_key_id: uuid.UUID,
        *,
        status: PredictionStatus | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Prediction], str | None]:
        """Cursor-paginated listing scoped to an API key.

        Parameters:
            api_key_id: Caller key.
            status: Optional status filter.
            limit: Page size.
            cursor: Opaque ULID cursor (exclusive lower bound, descending).

        Returns:
            (rows, next_cursor) where next_cursor is None on the last page.
        """
        stmt = select(Prediction).where(
            Prediction.api_key_id == api_key_id,
            Prediction.deleted_at.is_(None),
        )
        if status is not None:
            stmt = stmt.where(Prediction.status == status)
        if cursor:
            stmt = stmt.where(Prediction.id < cursor)
        stmt = stmt.order_by(Prediction.id.desc()).limit(limit + 1)
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            next_cursor = rows[-1].id
        return rows, next_cursor

    async def count_active_for_key(self, api_key_id: uuid.UUID) -> int:
        """Count non-terminal predictions for a key.

        Parameters:
            api_key_id: Caller key.

        Returns:
            The number of active (created/queued/processing) predictions.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(Prediction)
            .where(
                Prediction.api_key_id == api_key_id,
                Prediction.deleted_at.is_(None),
                Prediction.status.in_(
                    [
                        PredictionStatus.CREATED,
                        PredictionStatus.QUEUED,
                        PredictionStatus.PROCESSING,
                    ]
                ),
            )
        )
        return int(result.scalar_one())

    async def stale_processing(self, older_than: dt.datetime, limit: int = 50) -> list[Prediction]:
        """Return processing rows whose heartbeat is too old.

        Parameters:
            older_than: Heartbeat cutoff timestamp.
            limit: Maximum rows.

        Returns:
            List of stale Prediction rows.
        """
        result = await self._session.execute(
            select(Prediction)
            .where(
                Prediction.status == PredictionStatus.PROCESSING,
                Prediction.heartbeat_at.is_not(None),
                Prediction.heartbeat_at < older_than,
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def cancel_requested_pending(self, limit: int = 100) -> list[Prediction]:
        """Return live rows with cancel_requested set.

        Parameters:
            limit: Maximum rows.

        Returns:
            List of Prediction rows awaiting cancellation.
        """
        result = await self._session.execute(
            select(Prediction)
            .where(
                Prediction.canceled_requested.is_(True),
                Prediction.status.in_(
                    [
                        PredictionStatus.CREATED,
                        PredictionStatus.QUEUED,
                        PredictionStatus.PROCESSING,
                    ]
                ),
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    def input_to_json(payload: dict[str, Any]) -> str:
        """Serialize normalized input deterministically for hashing.

        Parameters:
            payload: Normalized input dict.

        Returns:
            Canonical JSON string with sorted keys.
        """
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
