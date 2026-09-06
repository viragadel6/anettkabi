"""Access-log middleware emitting one structured line per request plus metrics."""

from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.middleware.request_id import get_request_id
from app.telemetry.metrics import observe_request

__all__ = ["AccessLogMiddleware"]

_logger = structlog.get_logger("vsfx.access")


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status, duration and records Prometheus metrics."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Measure the handler and emit the access event.

        Parameters:
            request: Incoming request.
            call_next: Next middleware/route callable.

        Returns:
            The upstream response.
        """
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration_s = time.perf_counter() - started
            route = getattr(request.scope.get("route"), "path", None) or request.url.path
            observe_request(route, status, duration_s)
            _logger.info(
                "http_request",
                request_id=get_request_id(request),
                method=request.method,
                path=request.url.path,
                status=status,
                duration_ms=round(duration_s * 1000, 2),
            )
