"""OpenTelemetry-compatible correlation: trace/span id propagation, no-op export."""

from __future__ import annotations

import contextvars
import secrets
from typing import Any

import structlog

__all__ = ["TraceContext", "bind_trace", "current_trace", "new_span_id", "new_trace_id"]

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("vsfx_trace_id", default=None)
_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("vsfx_span_id", default=None)
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("vsfx_request_id", default=None)


def new_trace_id() -> str:
    """Generate a 32-hex-char W3C-compatible trace id.

    Returns:
        The trace id string.
    """
    return secrets.token_hex(16)


def new_span_id() -> str:
    """Generate a 16-hex-char W3C-compatible span id.

    Returns:
        The span id string.
    """
    return secrets.token_hex(8)


class TraceContext:
    """Context manager binding trace/span/request ids to structured logs.

    When OpenTelemetry is installed and a span is active, its ids are reused;
    otherwise fresh ids are generated so every log line stays correlatable
    without an exporter being configured.
    """

    __slots__ = ("_tokens", "request_id", "span_id", "trace_id")

    def __init__(self, request_id: str | None = None, trace_id: str | None = None) -> None:
        active = _active_otel_ids()
        self.trace_id = trace_id or active[0] or new_trace_id()
        self.span_id = active[1] or new_span_id()
        self.request_id = request_id or new_span_id()
        self._tokens: list[contextvars.Token[Any]] = []

    def __enter__(self) -> TraceContext:
        self._tokens = [
            _trace_id_var.set(self.trace_id),
            _span_id_var.set(self.span_id),
            _request_id_var.set(self.request_id),
        ]
        structlog.contextvars.bind_contextvars(
            trace_id=self.trace_id,
            span_id=self.span_id,
            request_id=self.request_id,
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        for token in reversed(self._tokens):
            token.var.reset(token)
        structlog.contextvars.unbind_contextvars("trace_id", "span_id", "request_id")

    def headers(self) -> dict[str, str]:
        """Return W3C traceparent propagation headers.

        Returns:
        A dict with `traceparent` and `x-request-id`.
        """
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "x-request-id": self.request_id,
        }


def _active_otel_ids() -> tuple[str | None, str | None]:
    """Return the active OpenTelemetry trace/span ids when available.

    Returns:
        (trace_id, span_id) hex strings, or (None, None).
    """
    try:
        from opentelemetry.trace import get_current_span

        context = get_current_span().get_span_context()
        if context and context.is_valid:
            return f"{context.trace_id:032x}", f"{context.span_id:016x}"
    except Exception:
        return None, None
    return None, None


def current_trace() -> dict[str, str | None]:
    """Return the currently bound trace identifiers.

    Returns:
        Dict with trace_id, span_id, request_id (None when unbound).
    """
    return {
        "trace_id": _trace_id_var.get(),
        "span_id": _span_id_var.get(),
        "request_id": _request_id_var.get(),
    }


def bind_trace(prediction_id: str, api_key_id: str | None = None) -> None:
    """Bind prediction identity onto the structlog context.

    Parameters:
        prediction_id: Current prediction id.
        api_key_id: Optional caller key id (already opaque).
    """
    values: dict[str, str] = {"prediction_id": prediction_id}
    if api_key_id:
        values["api_key_id"] = api_key_id
    structlog.contextvars.bind_contextvars(**values)
