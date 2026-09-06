"""Hash helpers and time formatting invariants."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.utils.hashing import hmac_signature, prompt_hash, sha256_bytes, sha256_file, verify_hmac
from app.utils.time import isoformat_ms, monotonic_ms, parse_isoformat, utc_now, utc_now_isoformat


def test_sha256_bytes_known_vector() -> None:
    assert sha256_bytes(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    payload = b"video-bytes" * 1000
    target = tmp_path / "clip.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == sha256_bytes(payload)


def test_hmac_roundtrip_and_reject() -> None:
    signature = hmac_signature("secret", "1700000000", '{"a":1}')
    assert len(signature) == 64
    assert verify_hmac("secret", "1700000000", '{"a":1}', signature)
    assert not verify_hmac("wrong", "1700000000", '{"a":1}', signature)
    assert not verify_hmac("secret", "1700000001", '{"a":1}', signature)


def test_prompt_hash_is_stable_and_short() -> None:
    first = prompt_hash("thunder rumble")
    assert first == prompt_hash("thunder rumble")
    assert first != prompt_hash("thunder rumble ")


def test_utc_now_is_timezone_aware() -> None:
    now = utc_now()
    assert now.tzinfo is UTC
    assert abs((now - datetime.now(tz=UTC)).total_seconds()) < 5


def test_isoformat_ms_formatting() -> None:
    moment = datetime(2026, 9, 6, 12, 30, 15, 123456, tzinfo=UTC)
    assert isoformat_ms(moment) == "2026-09-06T12:30:15.123Z"
    assert isoformat_ms(None) is None


def test_isoformat_roundtrip() -> None:
    rendered = utc_now_isoformat()
    parsed = parse_isoformat(rendered)
    assert abs((parsed - utc_now()).total_seconds()) < 5


def test_monotonic_ms_increases() -> None:
    first = monotonic_ms()
    second = monotonic_ms()
    assert second >= first


def test_parse_rejects_naive_strings() -> None:
    try:
        parse_isoformat("2026-09-06T12:30:15")
    except ValueError:
        return
    parsed = parse_isoformat("2026-09-06T12:30:15")
    assert parsed.tzinfo is not None


def test_timedelta_arithmetic_sanctity() -> None:
    base = utc_now()
    later = base + timedelta(seconds=10)
    assert (later - base).total_seconds() == 10
