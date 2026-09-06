"""Structlog configuration: JSON or console rendering, redaction-safe binders."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import get_settings

__all__ = ["configure_logging", "get_logger"]

_RESERVED = {
    "event",
    "level",
    "timestamp",
    "logger",
    "request_id",
    "prediction_id",
    "stage",
}


def _timestamper(_: Any, __: Any, event_dict: structlog.typing.EventDict) -> structlog.typing.EventDict:
    from app.utils.time import utc_now_isoformat

    event_dict.setdefault("timestamp", utc_now_isoformat())
    return event_dict


def _add_service(_: Any, __: Any, event_dict: structlog.typing.EventDict) -> structlog.typing.EventDict:
    event_dict.setdefault("service", "video-sfx-service")
    return event_dict


def configure_logging() -> None:
    """Initialize structlog and stdlib logging according to telemetry settings.

    Raises:
        ConfigError: Propagated from settings loading when configuration is invalid.
    """
    settings = get_settings()
    level = getattr(logging, settings.telemetry.log_level)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _timestamper,
        _add_service,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any
    if settings.telemetry.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)
    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Parameters:
        name: Logger name, typically `__name__`.
        **initial_values: Key/value pairs bound to every emitted event.

    Returns:
        A configured bound logger.
    """
    return structlog.get_logger(name).bind(**initial_values)
