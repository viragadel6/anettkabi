"""S3-compatible object storage: uploads, presigning, key layout, retention sweep."""

from __future__ import annotations

import datetime as dt
import mimetypes
import threading
import uuid
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings
from app.constants import (
    CACHE_CONTROL_IMMUTABLE,
    DISPOSITION_INLINE,
    MULTIPART_CHUNK_BYTES,
    MULTIPART_THRESHOLD_BYTES,
)
from app.errors import ErrorCode, ServiceError

__all__ = [
    "StorageService",
    "get_storage_service",
    "output_key_for",
    "source_key_for",
    "sweep_retention",
]

_lock = threading.Lock()
_singleton: StorageService | None = None


def source_key_for(prefix: str, sha256: str, ext: str, at: dt.datetime | None = None) -> str:
    """Build the deterministic object key for a source video.

    Parameters:
        prefix: Configured key prefix (may be empty).
        sha256: Content digest.
        ext: File extension including dot.
        at: Timestamp used for the yyyy/mm/dd path segments.

    Returns:
        The object key string.
    """
    moment = at or dt.datetime.now(tz=dt.UTC)
    parts = list(filter(None, (prefix, "uploads", f"{moment:%Y}", f"{moment:%m}", f"{moment:%d}")))
    return "/".join(parts) + f"/{sha256}{ext}"


def output_key_for(prefix: str, prediction_id: str, filename: str = "output.mp4") -> str:
    """Build the object key for a prediction output.

    Parameters:
        prefix: Configured key prefix.
        prediction_id: Prediction id.
        filename: Output filename (`output.mp4` or `audio.wav`).

    Returns:
        The object key string.
    """
    parts = list(filter(None, (prefix, "outputs", prediction_id)))
    return "/".join(parts) + f"/{filename}"


class StorageService:
    """High-level S3 client wrapper used by the API and worker."""

    def __init__(self, settings: Any) -> None:
        """Construct the service from resolved settings.

        Parameters:
            settings: The root Settings object.
        """
        self._settings = settings
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.storage.s3_endpoint_url or None,
            region_name=settings.storage.s3_region,
            aws_access_key_id=settings.storage.s3_access_key_id or None,
            aws_secret_access_key=settings.storage.s3_secret_access_key or None,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path" if settings.storage.s3_force_path_style else "auto"},
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=120,
            ),
        )
        self._bucket = settings.storage.s3_bucket
        self._transfer = TransferConfig(
            multipart_threshold=MULTIPART_THRESHOLD_BYTES,
            multipart_chunksize=MULTIPART_CHUNK_BYTES,
            use_threads=True,
            max_concurrency=4,
        )

    @property
    def bucket(self) -> str:
        """Return the configured bucket name."""
        return self._bucket

    def ensure_bucket(self) -> None:
        """Create the bucket when it does not exist (dev/MinIO convenience).

        Raises:
            ServiceError: storage_failed when creation fails.
        """
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError:
            try:
                self._client.create_bucket(Bucket=self._bucket)
            except (ClientError, BotoCoreError) as exc:
                raise ServiceError(
                    ErrorCode.STORAGE_FAILED,
                    f"cannot create bucket {self._bucket}: {exc}",
                ) from exc
        except BotoCoreError as exc:
            raise ServiceError(
                ErrorCode.STORAGE_FAILED,
                f"bucket {self._bucket} unreachable: {exc}",
            ) from exc

    def put_file(
        self,
        path: Path,
        key: str,
        *,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> int:
        """Upload a local file with multipart when large, with retries.

        Parameters:
            path: Local file.
            key: Target object key.
            content_type: MIME type.
            metadata: x-amz-meta-* values.

        Returns:
            Uploaded size in bytes.

        Raises:
            ServiceError: storage_failed after retry exhaustion.
        """
        size = path.stat().st_size
        extra: dict[str, Any] = {
            "ContentType": content_type,
            "CacheControl": CACHE_CONTROL_IMMUTABLE,
            "ContentDisposition": f'{DISPOSITION_INLINE}; filename="{path.name}"',
        }
        if metadata:
            extra["Metadata"] = {k: v[:256] for k, v in metadata.items()}

        def _do() -> None:
            self._client.upload_file(
                str(path),
                self._bucket,
                key,
                Config=self._transfer,
                ExtraArgs=extra,
            )

        try:
            retry_sync(_do, attempts=4, base_s=1.0, max_s=30.0)
        except (ClientError, BotoCoreError, OSError) as exc:
            raise ServiceError(
                ErrorCode.STORAGE_FAILED,
                f"upload of {key} failed: {exc}",
            ) from exc
        return size

    def put_bytes(self, payload: bytes, key: str, *, content_type: str) -> None:
        """Upload an in-memory payload.

        Parameters:
            payload: Bytes to store.
            key: Target object key.
            content_type: MIME type.

        Raises:
            ServiceError: storage_failed on client errors.
        """
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=payload,
                ContentType=content_type,
                CacheControl=CACHE_CONTROL_IMMUTABLE,
                ContentDisposition=DISPOSITION_INLINE,
            )
        except (ClientError, BotoCoreError) as exc:
            raise ServiceError(ErrorCode.STORAGE_FAILED, f"put of {key} failed: {exc}") from exc

    def presign_put(self, key: str, *, content_type: str, ttl_s: int | None = None) -> str:
        """Create a presigned PUT URL.

        Parameters:
            key: Object key.
            content_type: Content-Type the client must send.
            ttl_s: Expiry; defaults to settings.

        Returns:
            The presigned URL string.

        Raises:
            ServiceError: storage_failed on client errors.
        """
        settings = get_settings()
        try:
            return self._client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self._bucket,
                    "Key": key,
                    "ContentType": content_type,
                },
                ExpiresIn=ttl_s or settings.storage.s3_presign_ttl_s,
            )
        except (ClientError, BotoCoreError) as exc:
            raise ServiceError(ErrorCode.STORAGE_FAILED, f"presign failed: {exc}") from exc

    def presign_get(self, key: str, ttl_s: int | None = None) -> str:
        """Create a presigned GET URL.

        Parameters:
            key: Object key.
            ttl_s: Expiry; defaults to settings.

        Returns:
            The presigned URL string.

        Raises:
            ServiceError: storage_failed on client errors.
        """
        settings = get_settings()
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=ttl_s or settings.storage.s3_presign_ttl_s,
            )
        except (ClientError, BotoCoreError) as exc:
            raise ServiceError(ErrorCode.STORAGE_FAILED, f"presign failed: {exc}") from exc

    def public_url(self, key: str, *, ttl_s: int | None = None) -> str:
        """Resolve the delivery URL for a key (CDN or presigned GET).

        Parameters:
            key: Object key.
            ttl_s: Presign TTL when presigning.

        Returns:
            The external URL string.
        """
        settings = get_settings()
        if settings.storage.cdn_base_url:
            return f"{settings.storage.cdn_base_url}/{key.lstrip('/')}"
        return self.presign_get(key, ttl_s)

    def head_size(self, key: str) -> int | None:
        """Return object size via HEAD, None when missing."""
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except (ClientError, BotoCoreError):
            return None
        return int(response.get("ContentLength") or 0)

    def download_to_file(self, key: str, destination: Path) -> None:
        """Download an object to a local path.

        Parameters:
            key: Object key.
            destination: Local file path.

        Raises:
            ServiceError: storage_failed on client errors.
        """
        try:
            self._client.download_file(self._bucket, key, str(destination), Config=self._transfer)
        except (ClientError, BotoCoreError, OSError) as exc:
            raise ServiceError(ErrorCode.STORAGE_FAILED, f"download of {key} failed: {exc}") from exc

    def delete(self, key: str) -> bool:
        """Delete one object.

        Parameters:
            key: Object key.

        Returns:
            True when deleted, False when absent.

        Raises:
            ServiceError: storage_failed on client errors.
        """
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise ServiceError(ErrorCode.STORAGE_FAILED, f"delete of {key} failed: {exc}") from exc
        return True

    def exists(self, key: str) -> bool:
        """Check object existence.

        Parameters:
            key: Object key.

        Returns:
            True when the object exists.
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except (ClientError, BotoCoreError):
            return False
        return True

    def guess_content_type(self, filename: str) -> str:
        """Guess a MIME type for a filename.

        Parameters:
            filename: Name to inspect.

        Returns:
            A MIME string (video/mp4 default).
        """
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def retry_sync(operation: Any, *, attempts: int, base_s: float, max_s: float) -> Any:
    """Run a blocking operation with retries and full-jitter backoff.

    Parameters:
        operation: Callable to run.
        attempts: Total attempts.
        base_s: Backoff base.
        max_s: Backoff ceiling.

    Returns:
        The operation result.

    Raises:
        Exception: The last error after exhaustion.
    """
    import random
    import time as _time

    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last = exc
            if attempt == attempts - 1:
                raise
            ceiling = min(max_s, base_s * (2**attempt))
            _time.sleep(random.uniform(0.0, ceiling))
    assert last is not None
    raise last


class _StorageSingleton:
    """Process-level storage holder avoiding module-level `global` mutation."""

    _instance: StorageService | None = None

    @classmethod
    def get(cls) -> StorageService:
        """Return the shared instance, constructing it on first use.

        Returns:
            The shared StorageService instance.
        """
        with _lock:
            if cls._instance is None:
                cls._instance = StorageService(get_settings())
            return cls._instance


def get_storage_service() -> StorageService:
    """Return the process-wide StorageService singleton.

    Returns:
        The shared StorageService instance.
    """
    return _StorageSingleton.get()


async def sweep_retention(batch_limit: int = 200) -> list[str]:
    """Delete expired output objects and mark their asset rows.

    Parameters:
        batch_limit: Maximum objects per sweep.

    Returns:
        Swept object keys.
    """
    import asyncio

    from app.db.models import AssetKind
    from app.db.repositories.assets import AssetRepository
    from app.db.session import session_scope
    from app.utils.time import utc_now

    storage = get_storage_service()
    async with session_scope() as session:
        repo = AssetRepository(session)
        expired = await repo.expired_before(utc_now(), limit=batch_limit)
        keys: list[str] = []
        asset_ids: list[uuid.UUID] = []
        for asset in expired:
            if asset.kind != AssetKind.OUTPUT_VIDEO:
                continue
            try:
                await asyncio.to_thread(storage.delete, asset.storage_key)
                keys.append(asset.storage_key)
                asset_ids.append(asset.id)
            except ServiceError:
                continue
        await repo.mark_expired_deleted(asset_ids)
    return keys
