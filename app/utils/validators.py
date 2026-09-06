"""Request-content validators: prompts, seeds, URLs, and SSRF-safe host checks."""

from __future__ import annotations

import ipaddress
import socket
import unicodedata
import uuid
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from app.constants import MAX_PROMPT_CHARS, MAX_SEED
from app.errors import ErrorCode, ServiceError

__all__ = [
    "VideoUrl",
    "parse_metadata",
    "sanitize_prompt",
    "validate_seed",
    "validate_video_url",
    "validate_webhook_url",
]


class VideoUrl:
    """Parsed representation of an accepted `video` input.

    Attributes:
        kind: One of `http`, `data`, `upload`.
        url: The original URL string (http/upload) or empty for data URIs.
        data_uri: The full data: URI when kind == data.
    """

    __slots__ = ("data_uri", "kind", "url")

    def __init__(self, kind: str, url: str, data_uri: str = "") -> None:
        self.kind = kind
        self.url = url
        self.data_uri = data_uri


def sanitize_prompt(value: str) -> str:
    """Normalize a prompt: NFC, control-character stripping, length cap.

    Parameters:
        value: Raw prompt text.

    Returns:
        The sanitized prompt string.

    Raises:
        ServiceError: prompt_too_long when the normalized text exceeds the cap.
    """
    normalized = unicodedata.normalize("NFC", value or "")
    cleaned = "".join(ch for ch in normalized if unicodedata.category(ch) not in {"Cc", "Cf"})
    if len(cleaned) > MAX_PROMPT_CHARS:
        raise ServiceError(
            ErrorCode.PROMPT_TOO_LONG,
            f"prompt exceeds {MAX_PROMPT_CHARS} characters after normalization",
        )
    return cleaned.strip()


def validate_seed(seed: int) -> int:
    """Validate a requested seed value.

    Parameters:
        seed: Client-supplied seed; -1 requests a server-generated seed.

    Returns:
        The seed to use (generated when -1).

    Raises:
        ServiceError: seed_out_of_range for values outside [-1, 2^31-1].
    """
    if seed == -1:
        import secrets

        return secrets.randbelow(MAX_SEED + 1)
    if not 0 <= seed <= MAX_SEED:
        raise ServiceError(
            ErrorCode.SEED_OUT_OF_RANGE,
            f"seed must be -1 (random) or an integer in [0, {MAX_SEED}]",
        )
    return seed


def _assert_public_host(hostname: str, port: int, block_private: bool) -> None:
    """Resolve and validate a hostname against private networks.

    Parameters:
        hostname: Host from the parsed URL.
        port: Port from the parsed URL.
        block_private: Whether private ranges must be rejected.

    Raises:
        ServiceError: download_forbidden_host when resolution fails or is private.
    """
    try:
        infos = socket.getaddrinfo(hostname, port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ServiceError(
            ErrorCode.DOWNLOAD_FORBIDDEN_HOST,
            f"cannot resolve host {hostname!r}",
        ) from exc
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not block_private:
            return
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ServiceError(
                ErrorCode.DOWNLOAD_FORBIDDEN_HOST,
                f"host {hostname!r} resolves to a blocked address ({ip})",
            )


def validate_video_url(
    raw: str,
    allowed_schemes: list[str],
    *,
    block_private_cidrs: bool,
    public_base_url: str,
) -> VideoUrl:
    """Validate and classify a `video` URL input.

    Parameters:
        raw: The supplied URL string.
        allowed_schemes: Permitted schemes (from settings).
        block_private_cidrs: Enforce SSRF protection.
        public_base_url: This deployment's base URL; same-origin upload URLs are allowed.

    Returns:
        A VideoUrl describing the input.

    Raises:
        ServiceError: invalid_video_url or download_forbidden_host.
    """
    value = (raw or "").strip()
    if not value:
        raise ServiceError(ErrorCode.MISSING_VIDEO, "video field is empty")
    if value.startswith("data:"):
        header = value[5:].split(",", 1)
        if len(header) != 2 or not header[1]:
            raise ServiceError(ErrorCode.INVALID_VIDEO_URL, "malformed data: URI")
        return VideoUrl(kind="data", url="", data_uri=value)
    parsed = urlparse(value)
    if parsed.scheme not in allowed_schemes:
        raise ServiceError(
            ErrorCode.INVALID_VIDEO_URL,
            f"URL scheme {parsed.scheme!r} is not allowed; use one of {allowed_schemes}",
        )
    if not parsed.hostname:
        raise ServiceError(ErrorCode.INVALID_VIDEO_URL, "URL has no hostname")
    if any(c in value for c in " \t\n\r"):
        raise ServiceError(ErrorCode.INVALID_VIDEO_URL, "URL contains whitespace")
    if parsed.username or parsed.password:
        raise ServiceError(ErrorCode.INVALID_VIDEO_URL, "URLs with credentials are not accepted")
    public_host = urlparse(public_base_url).hostname or ""
    if public_host and parsed.hostname == public_host:
        return VideoUrl(kind="upload", url=value)
    _assert_public_host(parsed.hostname, parsed.port or 443, block_private_cidrs)
    return VideoUrl(kind="http", url=value)


def validate_webhook_url(raw: str, *, block_private_cidrs: bool) -> str:
    """Validate an outbound webhook URL (https only, SSRF-guarded).

    Parameters:
        raw: The supplied URL.
        block_private_cidrs: Enforce private-range rejection.

    Returns:
        The normalized URL string.

    Raises:
        ServiceError: invalid_request or download_forbidden_host.
    """
    parsed = urlparse(raw)
    if parsed.scheme != "https":
        raise ServiceError(ErrorCode.INVALID_REQUEST, "webhook_url must use https")
    if not parsed.hostname:
        raise ServiceError(ErrorCode.INVALID_REQUEST, "webhook_url has no hostname")
    _assert_public_host(parsed.hostname, parsed.port or 443, block_private_cidrs)
    return raw


def parse_metadata(metadata: Any, max_bytes: int) -> dict[str, Any]:
    """Validate and bound the `metadata` request field.

    Parameters:
        metadata: Arbitrary JSON object supplied by the client.
        max_bytes: Maximum serialized size in bytes.

    Returns:
        The metadata dict.

    Raises:
        ServiceError: invalid_request when not an object or too large.
    """
    import json

    if metadata in (None, ""):
        return {}
    if not isinstance(metadata, dict):
        raise ServiceError(ErrorCode.INVALID_REQUEST, "metadata must be a JSON object")
    serialized = json.dumps(metadata, ensure_ascii=False, default=str)
    if len(serialized.encode("utf-8")) > max_bytes:
        raise ServiceError(ErrorCode.INVALID_REQUEST, f"metadata exceeds {max_bytes} bytes serialized")
    return metadata


def is_uuid(value: str) -> bool:
    """Check whether a string is a valid UUID.

    Parameters:
        value: Candidate string.

    Returns:
        True for valid UUIDs.
    """
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def safe_object_name(name: str) -> str:
    """Reject path traversal in object names.

    Parameters:
        name: Proposed object name component.

    Returns:
        The basename of the supplied path.

    Raises:
        ServiceError: invalid_request when the name contains separators.
    """
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ServiceError(ErrorCode.INVALID_REQUEST, "invalid object name")
    return PurePosixPath(name).name
