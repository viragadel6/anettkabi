"""Error taxonomy: machine codes, HTTP mapping, and the ServiceError carrier."""

from __future__ import annotations

from typing import Any

from app.constants import API_PREFIX

__all__ = ["ConfigError", "ErrorCode", "ServiceError", "error_envelope"]


class ErrorCode:
    """Namespace of machine-readable error codes mapped to HTTP statuses."""

    INVALID_REQUEST = ("invalid_request", 400)
    MISSING_VIDEO = ("missing_video", 400)
    INVALID_VIDEO_URL = ("invalid_video_url", 400)
    UNSUPPORTED_MEDIA_TYPE = ("unsupported_media_type", 415)
    VIDEO_TOO_LARGE = ("video_too_large", 413)
    VIDEO_TOO_LONG = ("video_too_long", 422)
    VIDEO_TOO_SHORT = ("video_too_short", 422)
    NO_VIDEO_STREAM = ("no_video_stream", 422)
    CORRUPT_MEDIA = ("corrupt_media", 422)
    DOWNLOAD_FAILED = ("download_failed", 502)
    DOWNLOAD_FORBIDDEN_HOST = ("download_forbidden_host", 400)
    PROMPT_TOO_LONG = ("prompt_too_long", 400)
    PROMPT_BLOCKED = ("prompt_blocked", 422)
    SEED_OUT_OF_RANGE = ("seed_out_of_range", 400)
    PARAMETER_OUT_OF_RANGE = ("parameter_out_of_range", 400)
    UNAUTHORIZED = ("unauthorized", 401)
    FORBIDDEN_SCOPE = ("forbidden_scope", 403)
    RATE_LIMITED = ("rate_limited", 429)
    CONCURRENCY_LIMITED = ("concurrency_limited", 429)
    QUOTA_EXCEEDED = ("quota_exceeded", 402)
    IDEMPOTENCY_KEY_CONFLICT = ("idempotency_key_conflict", 409)
    PREDICTION_NOT_FOUND = ("prediction_not_found", 404)
    PREDICTION_NOT_CANCELABLE = ("prediction_not_cancelable", 409)
    WEIGHTS_UNAVAILABLE = ("weights_unavailable", 503)
    INFERENCE_FAILED = ("inference_failed", 500)
    INFERENCE_TIMEOUT = ("inference_timeout", 504)
    GPU_OUT_OF_MEMORY = ("gpu_out_of_memory", 503)
    MUX_FAILED = ("mux_failed", 500)
    STORAGE_FAILED = ("storage_failed", 502)
    INTERNAL_ERROR = ("internal_error", 500)


_CODE_MAP: dict[str, tuple[str, int]] = {}
for _attr, _pair in vars(ErrorCode).items():
    if _attr.isupper() and isinstance(_pair, tuple) and len(_pair) == 2:
        _CODE_MAP[_pair[0]] = _pair
del _attr, _pair


def http_status_for(code: str) -> int:
    """Return the HTTP status associated with an error code.

    Parameters:
        code: Machine-readable error code.

    Returns:
        The mapped HTTP status, or 500 for unknown codes.
    """
    entry = _CODE_MAP.get(code)
    return entry[1] if entry else 500


class ServiceError(Exception):
    """An error carrying a taxonomy code, HTTP status, and safe message.

    Attributes:
        code: Machine-readable error code.
        http_status: HTTP status for the response.
        message: Safe, human-readable reason (never leaks internals).
        details: Optional structured payload merged into the error envelope.
        retry_after_s: Optional Retry-After hint for 429 responses.
    """

    def __init__(
        self,
        code_and_status: tuple[str, int] | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        if isinstance(code_and_status, str):
            entry = _CODE_MAP.get(code_and_status)
            if entry is None:
                raise ValueError(f"unknown error code: {code_and_status}")
            code, status = entry
        else:
            code, status = code_and_status
        super().__init__(message)
        self.code = code
        self.http_status = status
        self.message = message
        self.details = details or {}
        self.retry_after_s = retry_after_s

    def __repr__(self) -> str:
        return f"ServiceError(code={self.code!r}, status={self.http_status}, message={self.message!r})"


class ConfigError(ServiceError):
    """Raised at startup when configuration is missing or invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.INTERNAL_ERROR, f"configuration error: {message}")


def error_envelope(
    error: ServiceError,
    request_id: str,
    *,
    prediction_id: str | None = None,
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the standard JSON error envelope for a ServiceError.

    Parameters:
        error: The error to render.
        request_id: Correlation id of the failing request.
        prediction_id: Optional prediction id the error relates to.
        prediction: Optional partial prediction payload to embed.

    Returns:
        A JSON-serializable envelope dict.
    """
    payload = prediction or {
        "id": prediction_id or "",
        "model": "",
        "status": "failed",
        "input": {},
        "outputs": [],
        "urls": {"get": f"{API_PREFIX}/predictions/{prediction_id}" if prediction_id else ""},
        "has_nsfw_contents": [False],
        "error": error.message,
        "error_code": error.code,
        "progress": 0,
        "stage": "",
        "timings": {},
        "execution_time": 0,
        "created_at": None,
        "started_at": None,
        "completed_at": None,
    }
    if prediction is not None:
        payload = dict(payload)
        payload["status"] = "failed"
        payload["error"] = error.message
        payload["error_code"] = error.code
        payload["outputs"] = []
    body: dict[str, Any] = {
        "code": error.http_status,
        "message": error.message,
        "data": payload,
        "error_code": error.code,
        "request_id": request_id,
    }
    if error.details:
        body["details"] = error.details
    if error.retry_after_s is not None:
        body["retry_after"] = error.retry_after_s
    return body
