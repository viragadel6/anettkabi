"""Prompt/seed/URL/metadata validator behavior."""

from __future__ import annotations

import pytest

from app.constants import MAX_PROMPT_CHARS, MAX_SEED
from app.errors import ErrorCode, ServiceError
from app.utils.validators import (
    parse_metadata,
    sanitize_prompt,
    validate_seed,
    validate_webhook_url,
)


def test_sanitize_prompt_strips_control_chars_and_normalizes() -> None:
    raw = "  whoosh\u200b\u0000 with émoji  "
    assert sanitize_prompt(raw) == "whoosh with émoji"


def test_sanitize_prompt_enforces_cap() -> None:
    with pytest.raises(ServiceError) as excinfo:
        sanitize_prompt("a" * (MAX_PROMPT_CHARS + 1))
    assert excinfo.value.code == ErrorCode.PROMPT_TOO_LONG[0]


@pytest.mark.parametrize("seed", [0, 1, 42, MAX_SEED])
def test_validate_seed_accepts_range(seed: int) -> None:
    assert validate_seed(seed) == seed


@pytest.mark.parametrize("seed", [-2, MAX_SEED + 1, 2**40, -100])
def test_validate_seed_rejects_out_of_range(seed: int) -> None:
    with pytest.raises(ServiceError) as excinfo:
        validate_seed(seed)
    assert excinfo.value.code == ErrorCode.SEED_OUT_OF_RANGE[0]


def test_validate_seed_random_marker() -> None:
    assert 0 <= validate_seed(-1) <= MAX_SEED


def test_parse_metadata_accepts_object() -> None:
    assert parse_metadata({"b": 2, "a": "x"}, 4096) == {"b": 2, "a": "x"}


def test_parse_metadata_rejects_non_object() -> None:
    with pytest.raises(ServiceError) as excinfo:
        parse_metadata(["not", "an", "object"], 4096)
    assert excinfo.value.code == ErrorCode.INVALID_REQUEST[0]


def test_parse_metadata_enforces_byte_cap() -> None:
    oversized = {"blob": "x" * 5000}
    with pytest.raises(ServiceError):
        parse_metadata(oversized, 1024)


def test_validate_webhook_url_requires_https() -> None:
    with pytest.raises(ServiceError) as excinfo:
        validate_webhook_url("http://localhost:9443/cb", block_private_cidrs=True)
    assert excinfo.value.code == ErrorCode.INVALID_REQUEST[0]


def test_validate_webhook_url_rejects_private_hosts() -> None:
    with pytest.raises(ServiceError) as excinfo:
        validate_webhook_url("https://127.0.0.1:9443/cb", block_private_cidrs=True)
    assert excinfo.value.code == ErrorCode.DOWNLOAD_FORBIDDEN_HOST[0]


def test_validate_webhook_url_rejects_garbage() -> None:
    with pytest.raises(ServiceError):
        validate_webhook_url("not a url at all", block_private_cidrs=True)


def test_validate_webhook_url_accepts_resolvable_https() -> None:
    url = validate_webhook_url("https://localhost:9443/cb", block_private_cidrs=False)
    assert url == "https://localhost:9443/cb"
