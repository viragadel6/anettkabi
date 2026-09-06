"""Upload endpoints: direct multipart upload and presigned PUT."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile

from app.api.schemas.upload import PresignRequest, PresignResult, UploadResult
from app.constants import UPLOAD_ASSET_TTL_S
from app.db.models import AssetKind
from app.db.repositories.assets import AssetRepository
from app.dependencies import SessionDep, SettingsDep, StorageDep
from app.errors import ErrorCode, ServiceError
from app.media.probe import probe_media
from app.services.storage import source_key_for
from app.utils.files import remove_quietly, safe_suffix
from app.utils.hashing import sha256_file
from app.utils.mime import content_type_matches, guess_video_mime
from app.utils.time import isoformat_ms, utc_now

__all__ = ["router"]

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("")
async def upload_video(
    settings: SettingsDep,
    storage: StorageDep,
    session: SessionDep,
    file: Annotated[UploadFile, File(description="Video file")],
) -> dict[str, Any]:
    """Store an uploaded video and return its retrievable URL.

    Returns:
        Envelope with the UploadResult payload including a probe summary.

    Raises:
        ServiceError: unsupported_media_type or video_too_large.
    """
    head = await file.read(64)
    await file.seek(0)
    declared = file.content_type or ""
    filename = file.filename or "upload.mp4"
    sniffed = guess_video_mime(head, declared, filename)
    suffix = safe_suffix(filename, ".mp4")
    if suffix not in settings.media.allowed_video_ext:
        raise ServiceError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"extension {suffix!r} is not allowed; allowed: {settings.media.allowed_video_ext}",
        )
    if not content_type_matches(sniffed, [*settings.media.allowed_video_mime, "application/octet-stream"]):
        raise ServiceError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"content type {sniffed!r} does not look like a supported video",
        )
    work = settings.jobs.work_dir / "uploads"
    work.mkdir(parents=True, exist_ok=True)
    staged = work / f"upload-{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with staged.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.server.max_upload_bytes:
                    raise ServiceError(
                        ErrorCode.VIDEO_TOO_LARGE,
                        f"upload exceeds {settings.server.max_upload_bytes} bytes",
                    )
                handle.write(chunk)
        if size < 1024:
            raise ServiceError(ErrorCode.CORRUPT_MEDIA, "file is too small to be a video")
        digest = sha256_file(staged)
        key = source_key_for(settings.storage.s3_key_prefix, digest, suffix)
        if not storage.exists(key):
            storage.put_file(staged, key, content_type=sniffed)
        probe_summary: dict[str, Any] | None = None
        duration_s = None
        try:
            info = await probe_media(staged)
            probe_summary = info.summary()
            duration_s = info.duration_s
        except Exception:
            probe_summary = None
        expires = utc_now().timestamp() + UPLOAD_ASSET_TTL_S
        asset = await AssetRepository(session).register(
            kind=AssetKind.SOURCE_VIDEO,
            storage_key=key,
            bucket=storage.bucket,
            content_type=sniffed,
            size_bytes=size,
            sha256=digest,
            duration_s=duration_s,
            expires_at=dt.datetime.fromtimestamp(expires, tz=dt.UTC),
        )
        payload = UploadResult(
            url=storage.public_url(key),
            asset_id=str(asset.id),
            expires_at=isoformat_ms(asset.expires_at),
            content_type=sniffed,
            size_bytes=size,
            probe=probe_summary,
        )
        return {"code": 200, "message": "success", "data": payload.model_dump()}
    finally:
        remove_quietly(staged)


@router.post("/presign")
async def presign_upload(
    settings: SettingsDep,
    storage: StorageDep,
    session: SessionDep,
    request: PresignRequest,
) -> dict[str, Any]:
    """Issue a presigned PUT URL for direct uploads.

    Returns:
        Envelope with the PresignResult payload.

    Raises:
        ServiceError: video_too_large for oversized announcements.
    """
    if request.size_bytes > settings.server.max_upload_bytes:
        raise ServiceError(
            ErrorCode.VIDEO_TOO_LARGE,
            f"announced size exceeds {settings.server.max_upload_bytes} bytes",
        )
    suffix = safe_suffix(request.filename, ".mp4")
    if suffix not in settings.media.allowed_video_ext:
        raise ServiceError(
            ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            f"extension {suffix!r} is not allowed",
        )
    content_type = request.content_type or "video/mp4"
    token = uuid.uuid4().hex
    key = source_key_for(
        settings.storage.s3_key_prefix,
        f"direct-{token}",
        suffix,
        dt.datetime.now(tz=dt.UTC),
    )
    upload_url = storage.presign_put(key, content_type=content_type)
    payload = PresignResult(
        upload_url=upload_url,
        storage_key=key,
        url=storage.public_url(key),
        expires_at=isoformat_ms(
            dt.datetime.now(tz=dt.UTC) + dt.timedelta(seconds=settings.storage.s3_presign_ttl_s)
        ),
        required_headers={"Content-Type": content_type},
    )
    await AssetRepository(session).register(
        kind=AssetKind.SOURCE_VIDEO,
        storage_key=key,
        bucket=storage.bucket,
        content_type=content_type,
        size_bytes=request.size_bytes,
        sha256="",
    )
    return {"code": 200, "message": "success", "data": payload.model_dump()}
