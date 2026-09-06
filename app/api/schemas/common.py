"""Shared response envelope schemas and cursors."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

__all__ = ["Envelope", "HealthReport", "PaginatedEnvelope"]

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    """Standard success/error envelope.

    Attributes:
        code: HTTP-like status code.
        message: Short reason.
        data: Payload.
    """

    code: int = 200
    message: str = "success"
    data: T


class PaginatedEnvelope(BaseModel):
    """Envelope page with a cursor.

    Attributes:
        code: HTTP-like status.
        message: Short reason.
        data: Row payloads.
        next_cursor: Cursor for the next page (None on last).
    """

    code: int = 200
    message: str = "success"
    data: list[dict[str, Any]] = Field(default_factory=list)
    next_cursor: str | None = None


class HealthReport(BaseModel):
    """Dependency health breakdown.

    Attributes:
        status: Overall status (ok/degraded/down).
        checks: Per-dependency results.
        version: Service version.
    """

    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    version: str = ""
