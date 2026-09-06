"""Typed errors raised by the vsfx client SDK."""

from __future__ import annotations

__all__ = [
    "APIError",
    "AuthenticationError",
    "ClientError",
    "ConcurrencyLimitedError",
    "ConnectionError",
    "IdempotencyConflictError",
    "QuotaExceededError",
    "RateLimitedError",
    "ServerError",
    "ServiceUnavailableError",
    "TransportError",
    "ValidationError",
]


class APIError(Exception):
    """Base error for all non-2xx API interactions.

    Attributes:
        error_code: Machine error code from the taxonomy.
        http_status: HTTP status code.
        message: Safe human message.
        request_id: Correlation id of the failing request.
        details: Optional structured details.
        retry_after: Seconds to wait when the error is retryable (else None).
    """

    __slots__ = ("details", "error_code", "http_status", "message", "request_id", "retry_after")

    def __init__(
        self,
        error_code: str,
        http_status: int,
        message: str,
        request_id: str = "",
        details: dict | None = None,
        retry_after: float | None = None,
    ) -> None:
        """Store error fields.

        Parameters:
            error_code: Machine code.
            http_status: HTTP status.
            message: Human message.
            request_id: Correlation id.
            details: Structured details.
            retry_after: Retry hint in seconds.
        """
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status
        self.message = message
        self.request_id = request_id
        self.details = details or {}
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        """Whether a sane client should retry this error with backoff.

        Returns:
            True for rate-limit/concurrency/5xx families.
        """
        if isinstance(self, (RateLimitedError, ConcurrencyLimitedError, ServerError)):
            return True
        return self.http_status >= 500 and not isinstance(self, QuotaExceededError)


class ClientError(APIError):
    """4xx family error."""


class ValidationError(ClientError):
    """Request rejected by schema/semantic validation (4xx)."""


class AuthenticationError(ClientError):
    """Missing or invalid API key (401)."""


class RateLimitedError(ClientError):
    """Per-key rate budget exhausted (429, rate_limited)."""


class ConcurrencyLimitedError(ClientError):
    """Too many in-flight predictions (429, concurrency_limited)."""


class QuotaExceededError(ClientError):
    """Monthly usage quota exhausted (402)."""


class IdempotencyConflictError(ClientError):
    """Idempotency key reused with a different payload (409)."""


class ServerError(APIError):
    """5xx family error."""


class ServiceUnavailableError(ServerError):
    """Weights unavailable or GPU saturation (503)."""


class TransportError(APIError):
    """Transport-level failure (DNS, refused, timeout)."""

    def __init__(self, message: str) -> None:
        """Build a transport error.

        Parameters:
            message: Failure description.
        """
        super().__init__("transport_error", 0, message)


_BY_ERROR_CODE = {
    "unauthorized": AuthenticationError,
    "forbidden_scope": AuthenticationError,
    "rate_limited": RateLimitedError,
    "concurrency_limited": ConcurrencyLimitedError,
    "quota_exceeded": QuotaExceededError,
    "idempotency_key_conflict": IdempotencyConflictError,
    "weights_unavailable": ServiceUnavailableError,
    "gpu_out_of_memory": ServiceUnavailableError,
    "invalid_request": ValidationError,
    "missing_video": ValidationError,
    "invalid_video_url": ValidationError,
    "unsupported_media_type": ValidationError,
    "video_too_large": ValidationError,
    "video_too_long": ValidationError,
    "video_too_short": ValidationError,
    "no_video_stream": ValidationError,
    "corrupt_media": ValidationError,
    "download_failed": ServerError,
    "download_forbidden_host": ValidationError,
    "prompt_too_long": ValidationError,
    "prompt_blocked": ValidationError,
    "seed_out_of_range": ValidationError,
    "parameter_out_of_range": ValidationError,
    "prediction_not_found": ClientError,
    "prediction_not_cancelable": ClientError,
    "inference_failed": ServerError,
    "inference_timeout": ServerError,
    "mux_failed": ServerError,
    "storage_failed": ServerError,
    "internal_error": ServerError,
}


def error_from_payload(
    error_code: str,
    http_status: int,
    message: str,
    request_id: str = "",
    details: dict | None = None,
    retry_after: float | None = None,
) -> APIError:
    """Instantiate the most specific error class for a payload.

    Parameters:
        error_code: Machine code.
        http_status: HTTP status.
        message: Human message.
        request_id: Correlation id.
        details: Structured details.
        retry_after: Retry hint.

    Returns:
        An APIError subclass instance.
    """
    cls = _BY_ERROR_CODE.get(error_code)
    if cls is None:
        cls = ServerError if http_status >= 500 else ClientError
    return cls(error_code, http_status, message, request_id, details, retry_after)
