"""Upload endpoint schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["PresignRequest", "PresignResult", "UploadResult"]


class UploadResult(BaseModel):
    """Response payload of POST /uploads.

    Attributes:
        url: Retrivable URL to pass as `video`.
        asset_id: Created asset id.
        expires_at: ISO-8601 expiry.
        content_type: Detected MIME.
        size_bytes: Stored size.
        probe: ffprobe summary when probing succeeded.
    """

    url: str
    asset_id: str
    expires_at: str | None = None
    content_type: str
    size_bytes: int
    probe: dict[str, Any] | None = None


class PresignRequest(BaseModel):
    """Request payload of POST /uploads/presign.

    Attributes:
        filename: Suggested filename (extension drives content type).
        content_type: MIME the client will send.
        size_bytes: Announced size for validation.
    """

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = "video/mp4"
    size_bytes: int = Field(gt=0)


class PresignResult(BaseModel):
    """Response payload of POST /uploads/presign.

    Attributes:
        upload_url: Presigned PUT target.
        storage_key: Object key the service expects.
        url: Retrivable URL to pass as `video`.
        expires_at: Presign expiry.
        required_headers: Headers the client must send.
    """

    upload_url: str
    storage_key: str
    url: str
    expires_at: str
    required_headers: dict[str, str] = Field(default_factory=dict)
