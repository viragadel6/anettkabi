"""Time helpers: timezone-aware UTC timestamps and ISO-8601 rendering."""

from __future__ import annotations

import datetime as _dt

__all__ = [
    "isoformat_ms",
    "monotonic_ms",
    "parse_isoformat",
    "utc_now",
    "utc_now_isoformat",
]


def utc_now() -> _dt.datetime:
    """Return the current timezone-aware UTC datetime.

    Returns:
        Current UTC time with tzinfo=UTC.
    """
    return _dt.datetime.now(tz=_dt.UTC)


def isoformat_ms(value: _dt.datetime | None) -> str | None:
    """Render a datetime as ISO-8601 UTC with millisecond precision.

    Parameters:
        value: The datetime to render (None passes through).

    Returns:
        String like `2024-06-01T12:00:00.000Z`, or None.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=_dt.UTC)
    value = value.astimezone(_dt.UTC)
    return (
        value.strftime("%Y-%m-%dT%H:%M:%S") + "." + f"{value.microsecond // 1000:03d}Z"
    )


def utc_now_isoformat() -> str:
    """Return the current UTC time as an ISO-8601 millisecond string.

    Returns:
        The formatted timestamp string.
    """
    return isoformat_ms(utc_now()) or ""


def parse_isoformat(value: str) -> _dt.datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    Parameters:
        value: Timestamp string (with Z or offset).

    Returns:
        The parsed datetime in UTC.

    Raises:
        ValueError: If the string is not a valid timestamp.
    """
    parsed = _dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed.astimezone(_dt.UTC)


def monotonic_ms() -> float:
    """Return a monotonic clock reading in milliseconds.

    Returns:
        Milliseconds from a monotonic clock suitable for durations.
    """
    return _dt.datetime.now(tz=_dt.UTC).timestamp() * 1000.0
