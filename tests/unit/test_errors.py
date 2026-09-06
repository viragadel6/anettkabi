"""Error taxonomy: statuses, envelopes, and ServiceError behavior."""

from __future__ import annotations

import pytest

from app.errors import ErrorCode, ServiceError, error_envelope, http_status_for


def test_every_taxonomy_code_has_status() -> None:
    names = [name for name in dir(ErrorCode) if name.isupper()]
    assert len(names) >= 25
    for name in names:
        code, status = getattr(ErrorCode, name)
        assert isinstance(code, str)
        assert code
        assert 400 <= status <= 504


def test_http_status_for_known_and_unknown() -> None:
    assert http_status_for("rate_limited") == 429
    assert http_status_for("weights_unavailable") == 503
    assert http_status_for("definitely_not_a_code") == 500


def test_service_error_from_string_lookup() -> None:
    error = ServiceError("quota_exceeded", "monthly quota exhausted")
    assert error.code == "quota_exceeded"
    assert error.http_status == 402
    assert error.retry_after_s is None


def test_service_error_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="unknown error code"):
        ServiceError("made_up_code", "nope")


def test_service_error_from_tuple() -> None:
    error = ServiceError(ErrorCode.INFERENCE_TIMEOUT, "too slow", retry_after_s=5.0)
    assert error.http_status == 504
    assert error.retry_after_s == 5.0


def test_error_envelope_shape() -> None:
    error = ServiceError(ErrorCode.RATE_LIMITED, "slow down", retry_after_s=1.5)
    envelope = error_envelope(error, "req-123")
    assert envelope["code"] == 429
    assert envelope["error_code"] == "rate_limited"
    assert envelope["request_id"] == "req-123"
    assert envelope["retry_after"] == 1.5


def test_error_envelope_embeds_prediction_context() -> None:
    error = ServiceError(ErrorCode.INFERENCE_FAILED, "boom", details={"stage": "generate"})
    envelope = error_envelope(error, "req-9", prediction_id="01PRED")
    assert envelope["details"]["stage"] == "generate"
    assert envelope["data"]["id"] == "01PRED"
    assert envelope["data"]["status"] == "failed"
    assert envelope["data"]["error_code"] == "inference_failed"


def test_repr_is_safe() -> None:
    error = ServiceError(ErrorCode.UNAUTHORIZED, "bad key")
    assert "bad key" in repr(error)
    assert "unauthorized" in repr(error)
