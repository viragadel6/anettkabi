"""Error schema models mirroring the taxonomy."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ErrorBody", "ErrorFieldDetail"]


class ErrorFieldDetail(BaseModel):
    """One validation failure field.

    Attributes:
        field: Dotted field path.
        issue: Human description.
    """

    field: str
    issue: str


class ErrorBody(BaseModel):
    """Client-visible error payload.

    Attributes:
        code: HTTP status.
        message: Safe reason.
        error_code: Machine code.
        request_id: Correlation id.
        details: Optional structured details.
    """

    code: int
    message: str
    error_code: str
    request_id: str = ""
    details: dict[str, object] | None = None
    retry_after: float | None = Field(default=None)
