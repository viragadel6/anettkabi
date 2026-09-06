"""ULID-like identifier generation (26-char Crockford base32, sortable)."""

from __future__ import annotations

import os
import secrets
import threading
import time

from app.constants import PREDICTION_ID_LEN

__all__ = ["new_api_secret", "new_prediction_id", "new_ulid", "ulid_timestamp_ms"]

_ENCODING = "0123456789abcdefghjkmnpqrstvwxyz"
_ENCODING_LEN = len(_ENCODING)
_TIME_BITS = 48
_RANDOM_BITS = 80
_lock = threading.Lock()


class _MonotonicState:
    """Thread-safe ULID randomness state ensuring monotonic ids within a millisecond."""

    __slots__ = ("random", "ts_ms")

    def __init__(self) -> None:
        self.ts_ms = 0
        self.random = 0

    def next_random(self, ts_ms: int) -> int:
        """Return the next 80-bit random value for the given timestamp.

        Parameters:
            ts_ms: Current millisecond timestamp.

        Returns:
            A strictly increasing 80-bit integer within equal timestamps.
        """
        with _lock:
            if ts_ms == self.ts_ms:
                self.random += 1
            else:
                self.random = int.from_bytes(secrets.token_bytes(10), "big")
                self.ts_ms = ts_ms
            return self.random & ((1 << _RANDOM_BITS) - 1)


_state = _MonotonicState()


def _encode(value: int, bit_length: int, char_count: int) -> str:
    chars: list[str] = []
    for shift in range((char_count - 1) * 5, -1, -5):
        chars.append(_ENCODING[(value >> shift) & 0x1F])
    if bit_length:
        assert len(chars) == char_count
    return "".join(chars)


def _monotonic_random(ts_ms: int) -> int:
    """Produce an 80-bit random value strictly increasing within the same millisecond.

    Parameters:
        ts_ms: Current timestamp in milliseconds.

    Returns:
        A per-process unique 80-bit integer.
    """
    return _state.next_random(ts_ms)


def new_ulid() -> str:
    """Generate a new 26-character lowercase ULID.

    Returns:
        A sortable ULID string encoding millisecond time plus randomness.
    """
    ts_ms = int(time.time() * 1000) & ((1 << _TIME_BITS) - 1)
    random_part = _monotonic_random(ts_ms)
    combined = (ts_ms << _RANDOM_BITS) | random_part
    return _encode(combined, _TIME_BITS + _RANDOM_BITS, PREDICTION_ID_LEN)


def new_prediction_id() -> str:
    """Alias of `new_ulid` used for prediction primary keys.

    Returns:
        A new 26-character prediction id.
    """
    return new_ulid()


def ulid_timestamp_ms(ulid: str) -> int:
    """Decode the millisecond timestamp embedded in a ULID.

    Parameters:
        ulid: A 26-character ULID string.

    Returns:
        Milliseconds since the Unix epoch encoded in the id.

    Raises:
        ValueError: If the string is not a valid 26-char Crockford base32 ULID.
    """
    if len(ulid) != PREDICTION_ID_LEN:
        raise ValueError(f"ULID must be {PREDICTION_ID_LEN} characters")
    value = 0
    for char in ulid:
        idx = _ENCODING.find(char)
        if idx < 0:
            normalized = char.lower().translate(str.maketrans("ilo", "110"))
            idx = _ENCODING.find(normalized)
        if idx < 0:
            raise ValueError(f"invalid ULID character: {char!r}")
        value = (value << 5) | idx
    return value >> _RANDOM_BITS


def new_api_secret(length: int) -> str:
    """Generate a URL-safe random secret of a exact character length.

    Parameters:
        length: Desired character count.

    Returns:
        A random alphanumeric string of exactly `length` characters.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def new_uuid_hex() -> str:
    """Return a fresh random hex token (32 hex chars).

    Returns:
        A random 128-bit hex string.
    """
    return os.urandom(16).hex()
