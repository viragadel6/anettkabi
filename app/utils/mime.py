"""MIME-type and content sniffing helpers for uploaded/downloaded media."""

from __future__ import annotations

import binascii

__all__ = ["content_type_matches", "extension_for_mime", "guess_video_mime", "looks_like_mp4"]

_MP4_BRANDS = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"}


def looks_like_mp4(head: bytes) -> bool:
    """Check a file head for ISO-BMFF structure.

    Parameters:
        head: First bytes of the file (>= 12 bytes recommended).

    Returns:
        True when an MP4 box structure is detected.
    """
    if len(head) < 12:
        return False
    size = int.from_bytes(head[0:4], "big")
    if size < 8 or size > 10**9:
        return False
    return head[4:8] in _MP4_BRANDS or head[4:8] == b"ftyp"


def guess_video_mime(head: bytes, declared: str | None, filename: str) -> str:
    """Determine the effective video MIME type for a payload.

    Parameters:
        head: First bytes of content (up to 64).
        declared: Content-Type reported by the client/server, if any.
        filename: Filename used as a last-resort hint.

    Returns:
        The best-effort MIME type string.
    """
    lowered = filename.lower()
    if looks_like_mp4(head):
        return "video/mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm" if ".webm" in lowered else "video/x-matroska"
    if len(head) >= 12 and head[4:8] == b"ftypqt  ":
        return "video/quicktime"
    if declared:
        return declared.split(";")[0].strip().lower()
    table = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mpg": "video/mpeg",
        ".mpeg": "video/mpeg",
        ".3gp": "video/3gpp",
    }
    for ext, mime in table.items():
        if lowered.endswith(ext):
            return mime
    return "application/octet-stream"


def content_type_matches(candidate: str, allowed: list[str]) -> bool:
    """Check a MIME type against an allowlist ignoring parameters.

    Parameters:
        candidate: MIME type to test.
        allowed: Permitted MIME types.

    Returns:
        True when allowed, after lowercasing and parameter stripping.
    """
    normalized = candidate.split(";", maxsplit=1)[0].strip().lower()
    return normalized in {entry.strip().lower() for entry in allowed}


def extension_for_mime(mime: str, fallback: str = ".mp4") -> str:
    """Map a MIME type to a canonical file extension.

    Parameters:
        mime: MIME type string.
        fallback: Extension returned for unknown types.

    Returns:
        The mapped extension including the leading dot.
    """
    table = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-matroska": ".mkv",
        "video/x-msvideo": ".avi",
        "video/mpeg": ".mpeg",
        "video/3gpp": ".3gp",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
    }
    return table.get(mime.split(";", maxsplit=1)[0].strip().lower(), fallback)


def decode_data_uri_suffix(uri: str) -> str:
    """Return the subtype suffix of a data: URI for extension mapping.

    Parameters:
        uri: Full data: URI.

    Returns:
        A dotted extension guess, `.bin` when unknown.

    Raises:
        ValueError: When the URI is not a data: URI.
    """
    if not uri.startswith("data:"):
        raise ValueError("not a data URI")
    header = uri.split(",", 1)[0]
    mime = header[5:].split(";", 1)[0] or "application/octet-stream"
    return extension_for_mime(mime, ".bin")


def hex_ok(value: str) -> bool:
    """Validate that a string is lowercase hex (used for digests).

    Parameters:
        value: Candidate digest.

    Returns:
        True when every character is a hex digit.
    """
    try:
        binascii.unhexlify(value)
    except (binascii.Error, ValueError):
        return False
    return value == value.lower()
