"""Prediction endpoints: create, get, list, cancel, delete."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, File, Form, Header, Query, Request, UploadFile

from app.api.schemas.prediction import prediction_envelope
from app.api.schemas.sfx_request import SfxRequest
from app.config import Settings, get_settings
from app.constants import MAX_METADATA_BYTES, PREDICTION_ID_LEN
from app.db.models import PredictionStatus
from app.db.repositories.predictions import PredictionRepository
from app.dependencies import ApiKeyId, ModerationDep, SessionDep, SettingsDep, StorageDep
from app.errors import ErrorCode, ServiceError
from app.services.prediction_service import (
    PredictionService,
    PredictionSubmission,
    stage_multipart_upload,
)
from app.services.quota import enforce_quotas
from app.utils.files import remove_quietly
from app.utils.hashing import prompt_hash
from app.utils.time import utc_now
from app.utils.validators import parse_metadata, validate_seed

__all__ = ["router"]

_logger = structlog.get_logger("vsfx.api.predictions")
router = APIRouter(prefix="/predictions", tags=["predictions"])


def _concurrency_and_quota_settings(request: Request) -> tuple[float, int]:
    """Read per-key quota/concurrency from the authenticated context.

    Parameters:
        request: Current request.

    Returns:
        (monthly_seconds_quota, concurrency_limit).
    """
    quota = getattr(request.state, "api_key_monthly_quota", None)
    concurrency = getattr(request.state, "api_key_concurrency_limit", None)
    settings = get_settings()
    return (
        float(quota) if quota is not None else settings.auth.monthly_seconds_quota,
        int(concurrency) if concurrency is not None else settings.auth.concurrent_jobs_per_key,
    )


async def _create_from_parts(
    request: Request,
    api_key_id: uuid.UUID,
    settings: Settings,
    storage: StorageDep,
    moderation: ModerationDep,
    *,
    json_body: SfxRequest | None = None,
    form_fields: dict[str, Any] | None = None,
    upload_file: UploadFile | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Shared creation path for JSON and multipart submissions.

    Parameters:
        request: Current request (per-key limits).
        api_key_id: Caller key.
        settings: Settings.
        storage: Storage service.
        moderation: Moderation service.
        json_body: Parsed JSON body when applicable.
        form_fields: Normalized form fields when applicable.
        upload_file: Multipart file part when applicable.
        idempotency_key: Optional Idempotency-Key header.

    Returns:
        The response envelope dict.

    Raises:
        ServiceError: Any taxonomy error from validation/quotas/creation.
    """
    quota, concurrency = _concurrency_and_quota_settings(request)
    await enforce_quotas(api_key_id, quota_seconds=quota, concurrency_limit=concurrency)
    if json_body is not None:
        normalized = json_body.normalized(settings.defaults)
        staged = None
    else:
        assert form_fields is not None
        base = SfxRequest(**form_fields)
        normalized = base.normalized(settings.defaults)
        staged = None
        if upload_file is not None:
            payload = await upload_file.read()
            staged = stage_multipart_upload(
                settings.jobs.work_dir / "uploads",
                upload_file.filename or "upload.mp4",
                payload,
            )
        if staged is None:
            raise ServiceError(ErrorCode.MISSING_VIDEO, "multipart requests must include a 'video' file part")
    try:
        service = PredictionService(storage, moderation)
        row = await service.create(
            PredictionSubmission(
                api_key_id=api_key_id,
                normalized_input=normalized,
                idempotency_key=idempotency_key,
                uploaded_file=staged,
            )
        )
    finally:
        if staged is not None:
            remove_quietly(staged)
    _logger.info(
        "prediction_submitted",
        prediction_id=row.id,
        api_key_id=str(api_key_id)[:8],
        prompt_hash=prompt_hash(str(normalized.get("prompt") or "")),
        prompt_len=len(str(normalized.get("prompt") or "")),
    )
    return prediction_envelope(row, settings)


@router.post("/video-to-video-sfx", status_code=200)
async def create_prediction(
    request: Request,
    api_key_id: ApiKeyId,
    settings: SettingsDep,
    storage: StorageDep,
    moderation: ModerationDep,
    body: SfxRequest | None = None,
    video: Annotated[UploadFile | None, File(description="Source video (multipart mode)")] = None,
    prompt: Annotated[str | None, Form()] = None,
    negative_prompt: Annotated[str | None, Form()] = None,
    seed: Annotated[int | None, Form()] = None,
    num_inference_steps: Annotated[int | None, Form()] = None,
    guidance_scale: Annotated[float | None, Form()] = None,
    duration: Annotated[float | None, Form()] = None,
    start_time: Annotated[float | None, Form()] = None,
    audio_mode: Annotated[str | None, Form()] = None,
    sfx_gain_db: Annotated[float | None, Form()] = None,
    original_audio_gain_db: Annotated[float | None, Form()] = None,
    duck_threshold_db: Annotated[float | None, Form()] = None,
    duck_ratio: Annotated[float | None, Form()] = None,
    duck_attack_ms: Annotated[float | None, Form()] = None,
    duck_release_ms: Annotated[float | None, Form()] = None,
    target_loudness_lufs: Annotated[float | None, Form()] = None,
    true_peak_db: Annotated[float | None, Form()] = None,
    video_handling: Annotated[str | None, Form()] = None,
    return_audio_only: Annotated[bool | None, Form()] = None,
    enable_safety_checker: Annotated[bool | None, Form()] = None,
    webhook_url: Annotated[str | None, Form()] = None,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Create a video-to-video SFX prediction (JSON or multipart).

    Returns:
        The standard envelope with data.status queued and urls.get populated.

    Raises:
        ServiceError: full taxonomy (validation, auth, quotas, moderation).
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type or any(
        value is not None for value in (video, prompt, seed, num_inference_steps)
    ):
        fields = _form_fields(locals())
        metadata = parse_metadata({}, MAX_METADATA_BYTES)
        fields["metadata"] = metadata
        return await _create_from_parts(
            request,
            api_key_id,
            settings,
            storage,
            moderation,
            form_fields=fields,
            upload_file=video,
            idempotency_key=idempotency_key,
        )
    if body is None:
        raise ServiceError(
            ErrorCode.INVALID_REQUEST,
            "provide application/json or multipart/form-data with a 'video' field",
        )
    return await _create_from_parts(
        request,
        api_key_id,
        settings,
        storage,
        moderation,
        json_body=body,
        idempotency_key=idempotency_key,
    )


def _form_fields(scope: dict[str, Any]) -> dict[str, Any]:
    """Extract form-provided values, dropping Nones and framework keys.

    Parameters:
        scope: The endpoint's locals.

    Returns:
        Dict of provided form fields.

    Raises:
        ServiceError: parameter_out_of_range for invalid enum values.
    """
    reserved = {
        "request",
        "api_key_id",
        "settings",
        "storage",
        "moderation",
        "body",
        "video",
        "idempotency_key",
        "metadata",
    }
    allowed_enums = {
        "audio_mode": ("replace", "mix", "duck"),
        "video_handling": ("copy", "reencode"),
    }
    fields: dict[str, Any] = {}
    for key, value in scope.items():
        if key in reserved or key.startswith("_") or value is None:
            continue
        if key in allowed_enums:
            if value not in allowed_enums[key]:
                raise ServiceError(
                    ErrorCode.PARAMETER_OUT_OF_RANGE,
                    f"{key} must be one of {allowed_enums[key]}",
                )
            fields[key] = value
        elif key not in fields:
            fields[key] = value
    if "seed" in fields:
        fields["seed"] = validate_seed(int(fields["seed"]))
    return fields


@router.get("/{prediction_id}/result")
async def get_prediction_result(
    prediction_id: str,
    api_key_id: ApiKeyId,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Fetch the current prediction envelope.

    Parameters:
        prediction_id: Prediction id.
        api_key_id: Caller key.
        session: DB session.
        settings: Settings.

    Returns:
        The standard envelope.

    Raises:
        ServiceError: prediction_not_found.
    """
    return await _fetch(prediction_id, api_key_id, session, settings)


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    api_key_id: ApiKeyId,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Alias of /{id}/result.

    Raises:
        ServiceError: prediction_not_found.
    """
    return await _fetch(prediction_id, api_key_id, session, settings)


async def _fetch(
    prediction_id: str,
    api_key_id: uuid.UUID,
    session: Any,
    settings: Settings,
) -> dict[str, Any]:
    """Load and render one prediction scoped to the caller.

    Parameters:
        prediction_id: Prediction id.
        api_key_id: Caller key.
        session: DB session.
        settings: Settings.

    Returns:
        The envelope dict.

    Raises:
        ServiceError: prediction_not_found.
    """
    _validate_id(prediction_id)
    row = await PredictionRepository(session).get(prediction_id)
    if row is None or row.api_key_id != api_key_id:
        raise ServiceError(
            ErrorCode.PREDICTION_NOT_FOUND,
            f"prediction {prediction_id} does not exist for this key",
        )
    return prediction_envelope(row, settings)


@router.get("")
async def list_predictions(
    api_key_id: ApiKeyId,
    session: SessionDep,
    settings: SettingsDep,
    status: PredictionStatus | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
) -> dict[str, Any]:
    """List the caller's predictions with cursor pagination.

    Returns:
        Paginated envelope with rows and next_cursor.

    Raises:
        ServiceError: invalid_request for malformed cursors.
    """
    if cursor:
        _validate_id(cursor)
    rows, next_cursor = await PredictionRepository(session).list_for_key(
        api_key_id, status=status, limit=limit, cursor=cursor
    )
    from app.api.schemas.prediction import render_prediction_payload

    return {
        "code": 200,
        "message": "success",
        "data": [render_prediction_payload(row, settings) for row in rows],
        "next_cursor": next_cursor,
    }


@router.post("/{prediction_id}/cancel")
async def cancel_prediction(
    prediction_id: str,
    api_key_id: ApiKeyId,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Request cancellation of a live prediction.

    Returns:
        Envelope with the updated (still-live) row.

    Raises:
        ServiceError: prediction_not_found or prediction_not_cancelable.
    """
    _validate_id(prediction_id)
    repo = PredictionRepository(session)
    row = await repo.get(prediction_id)
    if row is None or row.api_key_id != api_key_id:
        raise ServiceError(ErrorCode.PREDICTION_NOT_FOUND, f"prediction {prediction_id} not found")
    if row.is_terminal:
        raise ServiceError(
            ErrorCode.PREDICTION_NOT_CANCELABLE,
            f"prediction is already {row.status.value}",
        )
    updated = await repo.cancel_requested(prediction_id)
    if updated is None:
        raise ServiceError(
            ErrorCode.PREDICTION_NOT_CANCELABLE,
            "cancellation could not be recorded",
        )
    if updated.status == PredictionStatus.CREATED:
        await repo.mark_canceled(prediction_id)
        updated = await repo.get(prediction_id)
        assert updated is not None
    return prediction_envelope(updated, settings)


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    api_key_id: ApiKeyId,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Soft-delete a terminal prediction and remove output objects.

    Returns:
        Minimal success envelope.

    Raises:
        ServiceError: prediction_not_found or prediction_not_cancelable for
            non-terminal rows.
    """
    _validate_id(prediction_id)
    repo = PredictionRepository(session)
    row = await repo.get(prediction_id)
    if row is None or row.api_key_id != api_key_id:
        raise ServiceError(ErrorCode.PREDICTION_NOT_FOUND, f"prediction {prediction_id} not found")
    if not row.is_terminal:
        raise ServiceError(
            ErrorCode.PREDICTION_NOT_CANCELABLE,
            "only terminal predictions can be deleted; cancel it first",
        )
    deleted = await repo.soft_delete(prediction_id)
    if deleted is None:
        raise ServiceError(ErrorCode.PREDICTION_NOT_FOUND, "prediction already deleted")
    from app.services.storage import get_storage_service, output_key_for

    storage = get_storage_service()
    prefix = settings.storage.s3_key_prefix
    for filename in ("output.mp4", "audio.wav"):
        key = output_key_for(prefix, prediction_id, filename)
        try:
            storage.delete(key)
        except ServiceError:
            pass
    return {"code": 200, "message": "deleted", "data": {"id": prediction_id, "deleted": True}}


def _validate_id(prediction_id: str) -> None:
    """Validate prediction id shape.

    Parameters:
        prediction_id: Candidate id.

    Raises:
        ServiceError: invalid_request for malformed ids.
    """
    if len(prediction_id) != PREDICTION_ID_LEN or not prediction_id.isascii() or not prediction_id.islower():
        raise ServiceError(
            ErrorCode.INVALID_REQUEST,
            f"prediction ids are {PREDICTION_ID_LEN}-character lowercase strings",
        )
    if not all(char in "0123456789abcdefghjkmnpqrstvwxyz" for char in prediction_id):
        raise ServiceError(ErrorCode.INVALID_REQUEST, "prediction id contains invalid characters")


def now_reference() -> Any:
    """Return current UTC time (test seam).

    Returns:
        Timezone-aware datetime.
    """
    return utc_now()
