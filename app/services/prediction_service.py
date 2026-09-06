"""Prediction creation service: validation, persistence, storage, enqueue."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from app.config import get_settings
from app.constants import UPLOAD_ASSET_TTL_S
from app.db.models import AssetKind, Prediction
from app.db.repositories.assets import AssetRepository
from app.db.repositories.predictions import PredictionRepository
from app.db.session import session_scope
from app.errors import ErrorCode, ServiceError
from app.services.idempotency import check_idempotency, hash_normalized_input
from app.services.moderation import ModerationService
from app.services.queue import enqueue_job
from app.services.storage import StorageService, source_key_for
from app.utils.files import ensure_dir
from app.utils.hashing import sha256_file
from app.utils.mime import extension_for_mime
from app.utils.time import utc_now
from app.utils.validators import VideoUrl, validate_video_url

__all__ = ["PredictionService", "PredictionSubmission"]

_logger = structlog.get_logger("vsfx.predictions")


@dataclass(slots=True)
class PredictionSubmission:
    """A validated creation request bound to one API key.

    Attributes:
        api_key_id: Owning key id.
        normalized_input: Canonical stored input dict.
        idempotency_key: Optional client key.
        uploaded_file: Optional in-memory upload path (multipart path).
    """

    api_key_id: uuid.UUID
    normalized_input: dict[str, Any]
    idempotency_key: str | None
    uploaded_file: Path | None = None


class PredictionService:
    """Orchestrates prediction creation end to end (API-side)."""

    __slots__ = ("_moderation", "_storage")

    def __init__(self, storage: StorageService, moderation: ModerationService) -> None:
        """Bind collaborators.

        Parameters:
            storage: Storage service for source normalization.
            moderation: Prompt moderation.
        """
        self._storage = storage
        self._moderation = moderation

    async def create(self, submission: PredictionSubmission) -> Prediction:
        """Validate, moderate, persist, and enqueue a prediction.

        Parameters:
            submission: The validated submission.

        Returns:
            The queued Prediction row.

        Raises:
            ServiceError: prompt_blocked, idempotency_key_conflict, or
                storage/queue failures.
        """
        settings = get_settings()
        normalized = dict(submission.normalized_input)
        if submission.uploaded_file is not None:
            normalized["video"] = await self._persist_upload(submission.uploaded_file)
        video_ref = str(normalized.get("video") or "")
        if not video_ref:
            raise ServiceError(ErrorCode.MISSING_VIDEO, "video is required")
        video_url = validate_video_url(
            video_ref,
            settings.media.download_allowed_schemes,
            block_private_cidrs=settings.media.download_block_private_cidrs,
            public_base_url=settings.server.public_base_url,
        )
        normalized["video"] = self._video_reference(video_url)
        if normalized.get("enable_safety_checker", True):
            self._moderation.moderate(str(normalized.get("prompt") or ""))
            self._moderation.moderate(str(normalized.get("negative_prompt") or ""))
        webhook_url = normalized.get("webhook_url")
        if webhook_url:
            from app.utils.validators import validate_webhook_url

            validate_webhook_url(
                str(webhook_url), block_private_cidrs=settings.media.download_block_private_cidrs
            )
        else:
            normalized["webhook_url"] = None
        input_hash = hash_normalized_input(normalized)
        if submission.idempotency_key:
            decision = await check_idempotency(
                submission.api_key_id, submission.idempotency_key, input_hash
            )
            if decision.replay is not None:
                return decision.replay
        async with session_scope() as session:
            repo = PredictionRepository(session)
            from app.utils.ids import new_prediction_id

            prediction_id = new_prediction_id()
            row = await repo.create(
                prediction_id=prediction_id,
                api_key_id=submission.api_key_id,
                model=settings.model.model_id,
                input_payload=normalized,
                input_hash=input_hash,
                webhook_url=str(normalized.get("webhook_url") or "") or None,
                idempotency_key=submission.idempotency_key,
            )
            await repo.mark_queued(prediction_id)
        try:
            await enqueue_job(prediction_id, str(submission.api_key_id))
        except ServiceError:
            async with session_scope() as session:
                await PredictionRepository(session).fail(
                    prediction_id,
                    error="queue enqueue failed",
                    error_code=ErrorCode.INTERNAL_ERROR[0],
                )
            raise
        _logger.info(
            "prediction_created",
            prediction_id=prediction_id,
            prompt_hash=None,
            video_kind=video_url.kind,
        )
        return row

    async def _persist_upload(self, path: Path) -> str:
        """Upload a multipart source file and return its retrievable URL.

        Parameters:
            path: Temp file staged by the endpoint.

        Returns:
            The public/presigned URL of the stored object.

        Raises:
            ServiceError: video_too_large or storage_failed.
        """
        settings = get_settings()
        if path.stat().st_size > settings.media.download_max_bytes:
            raise ServiceError(
                ErrorCode.VIDEO_TOO_LARGE,
                f"uploaded file exceeds {settings.media.download_max_bytes} bytes",
            )
        digest = sha256_file(path)
        suffix = path.suffix.lower() or ".mp4"
        key = source_key_for(settings.storage.s3_key_prefix, digest, suffix)
        if not self._storage.exists(key):
            self._storage.put_file(
                path, key, content_type=f"video/{suffix.lstrip('.')}"
                if suffix in (".mp4", ".webm")
                else "video/mp4"
            )
        expires = utc_now().timestamp() + UPLOAD_ASSET_TTL_S
        import datetime as dt

        async with session_scope() as session:
            await AssetRepository(session).register(
                kind=AssetKind.SOURCE_VIDEO,
                storage_key=key,
                bucket=self._storage.bucket,
                content_type=extension_for_mime(f"video/{suffix.lstrip('.')}"),
                size_bytes=path.stat().st_size,
                sha256=digest,
                expires_at=dt.datetime.fromtimestamp(expires, tz=dt.UTC),
            )
        return self._storage.public_url(key)

    @staticmethod
    def _video_reference(video_url: VideoUrl) -> str:
        """Reduce a parsed video URL to its stored reference form.

        Parameters:
            video_url: Parsed input.

        Returns:
            The reference string stored in input.video.

        Raises:
            ServiceError: invalid_video_url for data URIs too large to inline.
        """
        if video_url.kind == "data":
            if len(video_url.data_uri) > 64 * 1024 * 1024:
                raise ServiceError(
                    ErrorCode.VIDEO_TOO_LARGE,
                    "data: URIs are limited to 64 MiB; use the uploads endpoint",
                )
            return video_url.data_uri
        return video_url.url


def stage_multipart_upload(work_dir: Path, filename: str, payload: bytes) -> Path:
    """Write multipart bytes to a scratch file for staging.

    Parameters:
        work_dir: Scratch directory.
        filename: Original filename (extension preserved).
        payload: File bytes.

    Returns:
        The staged file path.

    Raises:
        ServiceError: storage_failed on I/O errors.
    """
    from app.utils.files import safe_suffix

    ensure_dir(work_dir)
    target = work_dir / f"upload-{uuid.uuid4().hex}{safe_suffix(filename, '.mp4')}"
    try:
        target.write_bytes(payload)
    except OSError as exc:
        raise ServiceError(ErrorCode.STORAGE_FAILED, f"cannot stage upload: {exc}") from exc
    return target
