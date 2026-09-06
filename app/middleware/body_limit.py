"""Body-size middleware rejecting oversized requests before buffering."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings
from app.errors import ErrorCode, error_envelope
from app.middleware.request_id import get_request_id

__all__ = ["BodyLimitMiddleware"]


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Rejects bodies larger than max_upload_bytes with a 413 envelope."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Enforce Content-Length and streamed byte ceiling.

        Parameters:
            request: Incoming request.
            call_next: Next middleware/route callable.

        Returns:
            The upstream response or a 413 error envelope.
        """
        settings = get_settings()
        limit = settings.server.max_upload_bytes
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > limit:
            body = _video_too_large_body(limit, get_request_id(request))
            return JSONResponse(status_code=413, content=body)
        request.state.body_budget = limit
        return await call_next(request)


def _video_too_large_body(limit: int, request_id: str) -> dict[str, object]:
    """Build the 413 error envelope for oversized uploads.

    Parameters:
        limit: Configured maximum bytes.
        request_id: Current request id.

    Returns:
        JSON-serializable envelope dict.
    """
    from app.errors import ServiceError

    error = ServiceError(
        ErrorCode.VIDEO_TOO_LARGE,
        f"request body exceeds the {limit}-byte upload limit",
    )
    return error_envelope(error, request_id)
