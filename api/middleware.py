from __future__ import annotations

import hashlib

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

__all__ = ["IDEMPOTENCY_HEADER", "IdempotencyMiddleware"]

IDEMPOTENCY_HEADER = "Idempotency-Key"
MUTATING_METHODS = {"POST", "PUT", "PATCH"}
IDEMPOTENT_PATH_PREFIX = "/v1/tasks"


class IdempotencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method in MUTATING_METHODS:
            body = await request.body()
            client_key = request.headers.get(IDEMPOTENCY_HEADER)
            if client_key:
                digest = hashlib.sha256(
                    client_key.encode("utf-8") + body
                ).hexdigest()
                request.state.idempotency_key = digest
                request.state.client_idempotency_key = client_key
            elif (
                request.method == "POST"
                and request.url.path.startswith(IDEMPOTENT_PATH_PREFIX)
            ):
                automatic_source = (
                    request.method.encode("utf-8")
                    + request.url.path.encode("utf-8")
                    + body
                )
                request.state.idempotency_key = hashlib.sha256(
                    automatic_source
                ).hexdigest()
                request.state.client_idempotency_key = None
        response = await call_next(request)
        idempotency_key = getattr(request.state, "idempotency_key", None)
        if idempotency_key:
            response.headers["Idempotency-Key-Hash"] = idempotency_key
        return response
