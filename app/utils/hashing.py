"""Hashing helpers: streaming SHA-256, HMAC signatures, prompt hashing."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

__all__ = [
    "hmac_signature",
    "prompt_hash",
    "sha256_bytes",
    "sha256_file",
    "verify_hmac",
]

_CHUNK = 1024 * 1024

def sha256_file(path: Path) -> str:
    """Compute the hex SHA-256 digest of a file in bounded memory.

    Parameters:
        path: File to hash.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_bytes(payload: bytes) -> str:
    """Compute the hex SHA-256 digest of a byte string.

    Parameters:
        payload: Bytes to hash.

    Returns:
        The lowercase hex digest.
    """
    return hashlib.sha256(payload).hexdigest()

def hmac_signature(secret: str, timestamp: str, body: str) -> str:
    """Compute the webhook signature HMAC-SHA256 over `"{timestamp}.{body}"`.

    Parameters:
        secret: Shared signing secret.
        timestamp: Unix-epoch seconds string sent in X-Webhook-Timestamp.
        body: Raw request body string.

    Returns:
        Hex digest string.
    """
    message = f"{timestamp}.{body}".encode()
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()

def verify_hmac(secret: str, timestamp: str, body: str, signature: str) -> bool:
    """Constant-time verify a webhook signature.

    Parameters:
        secret: Shared signing secret.
        timestamp: Timestamp header value.
        body: Raw body received.
        signature: Received `sha256=<hex>` or bare hex signature.

    Returns:
        True when the signature matches.
    """
    expected = hmac_signature(secret, timestamp, body)
    candidate = signature.removeprefix("sha256=").strip().lower()
    return hmac.compare_digest(expected, candidate)

def prompt_hash(prompt: str) -> str:
    """Hash a prompt for logging without storing its contents.

    Parameters:
        prompt: User-supplied prompt text.

    Returns:
        A 16-hex-char truncated SHA-256 digest.
    """
    return sha256_bytes(prompt.encode("utf-8"))[:16]
