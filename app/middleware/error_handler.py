"""Global exception handlers mapping every failure to the standard envelope."""

from __future__ import annotations

import traceback
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import ErrorCode, ServiceError, error_envelope
from app.middleware.request_id import get_request_id
from app.utils.time import utc_now_isoformat

__all__ = ["register_exception_handlers"]

_logger = structlog.get_logger("vsfx.errors")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach uniform JSON error handlers to the application.

    Parameters:
        app: The FastAPI application.
    """

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        request_id = get_request_id(request)
        _logger.error(
            "service_error",
            request_id=request_id,
            error_code=exc.code,
            status=exc.http_status,
            message=exc.message,
            details=exc.details,
            stack=traceback.format_exc(),
        )
        response = JSONResponse(
            status_code=exc.http_status,
            content=error_envelope(exc, request_id),
        )
        if exc.retry_after_s is not None:
            response.headers["Retry-After"] = str(int(max(exc.retry_after_s, 1.0)))
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = get_request_id(request)
        errors = exc.errors()
        fields = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]) or "body",
                "issue": error.get("msg", "invalid value"),
            }
            for error in errors
        ]
        service_error = ServiceError(
            ErrorCode.INVALID_REQUEST,
            "request validation failed",
            details={"fields": fields},
        )
        _logger.warning(
            "validation_error",
            request_id=request_id,
            fields=fields,
        )
        return JSONResponse(
            status_code=service_error.http_status,
            content=error_envelope(service_error, request_id),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = get_request_id(request)
        if exc.status_code == 404:
            code: tuple[str, int] = ErrorCode.PREDICTION_NOT_FOUND
            message = "resource not found"
        elif exc.status_code == 405:
            code = ErrorCode.INVALID_REQUEST
            message = "method not allowed"
        elif exc.status_code == 413:
            code = ErrorCode.VIDEO_TOO_LARGE
            message = "request body too large"
        else:
            code = ErrorCode.INTERNAL_ERROR
            message = str(exc.detail) if exc.detail else "request failed"
        service_error = ServiceError(code, message)
        _logger.warning(
            "http_error",
            request_id=request_id,
            status=exc.status_code,
            message=message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(service_error, request_id),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = get_request_id(request)
        _logger.error(
            "unhandled_exception",
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc),
            stack=traceback.format_exception(type(exc), exc, exc.__traceback__),
        )
        service_error = ServiceError(
            ErrorCode.INTERNAL_ERROR,
            "internal server error; quote request id when reporting",
        )
        return JSONResponse(
            status_code=500,
            content=error_envelope(service_error, request_id),
        )


def error_body_for(error_code: str, message: str, request_id: str) -> dict[str, Any]:
    """Construct an envelope dict programmatically (used by CLI/tests).

    Parameters:
        error_code: Taxonomy code.
        message: Safe message.
        request_id: Correlation id.

    Returns:
        The JSON envelope dict.
    """
    service_error = ServiceError(error_code, message)
    body = error_envelope(service_error, request_id)
    body["generated_at"] = utc_now_isoformat()
    return body
