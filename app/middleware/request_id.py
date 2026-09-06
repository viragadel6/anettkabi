"""Request-id middleware: assigns or propagates X-Request-Id and trace ids."""

from __future__ import annotations

from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.telemetry.tracing import TraceContext

__all__ = ["RequestIdMiddleware"]

_HEADER = "x-request-id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Binds a request id + trace context for every request."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Attach request identity, run the handler, echo the header.

        Parameters:
            request: Incoming request.
            call_next: Next middleware/route callable.

        Returns:
            The response with X-Request-Id set.
        """
        incoming = request.headers.get(_HEADER)
        request_id = incoming or TraceContext().request_id
        with TraceContext(request_id=request_id) as ctx:
            structlog.contextvars.bind_contextvars(
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "",
            )
            request.state.request_id = request_id
            request.state.trace_id = ctx.trace_id
            response = await call_next(request)
            response.headers[_HEADER] = request_id
            response.headers["traceparent"] = ctx.headers()["traceparent"]
            structlog.contextvars.unbind_contextvars("method", "path", "client_ip")
            return response


def get_request_id(request: Request) -> str:
    """Read the request id assigned by the middleware.

    Parameters:
        request: Starlette request.

    Returns:
        The request id string (falls back to a fresh id when absent).
    """
    value: Any = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    return TraceContext().request_id
