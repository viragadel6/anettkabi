"""ULID generation and parsing invariants."""

from __future__ import annotations

import time

import pytest

from app.utils.ids import new_api_secret, new_prediction_id, new_ulid, ulid_timestamp_ms

ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"


def test_new_ulid_shape_and_alphabet() -> None:
    ulid = new_ulid()
    assert len(ulid) == 26
    assert all(char in ALPHABET for char in ulid)


def test_ulids_sort_monotonically_within_millisecond_window() -> None:
    first = new_ulid()
    second = new_ulid()
    assert first != second
    assert ulid_timestamp_ms(first) <= ulid_timestamp_ms(second)


def test_ulid_timestamp_matches_clock() -> None:
    before = int(time.time() * 1000)
    ulid = new_ulid()
    after = int(time.time() * 1000)
    decoded = ulid_timestamp_ms(ulid)
    assert before <= decoded <= after


def test_prediction_ids_share_ulid_shape() -> None:
    identifier = new_prediction_id()
    assert len(identifier) == 26
    assert identifier == identifier.lower()


def test_api_secret_shape() -> None:
    secret = new_api_secret(40)
    assert len(secret) == 40
    assert secret != new_api_secret(40)


@pytest.mark.parametrize("length", [16, 32, 64])
def test_api_secret_lengths(length: int) -> None:
    assert len(new_api_secret(length)) == length
