"""MIME sniffing and retry backoff helpers."""

from __future__ import annotations

import pytest

from app.utils.mime import (
    content_type_matches,
    extension_for_mime,
    guess_video_mime,
    hex_ok,
    looks_like_mp4,
)
from app.utils.retry import backoff_delay_s

FTYP_BOX = (
    b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    + b"\x00" * 64
)


def test_looks_like_mp4_accepts_ftyp() -> None:
    assert looks_like_mp4(FTYP_BOX[:32])
    assert not looks_like_mp4(b"RIFFxxxxWEBPVP8 ")
    assert not looks_like_mp4(b"")


def test_guess_prefers_sniffed_type_over_declaration() -> None:
    sniffed = guess_video_mime(FTYP_BOX[:32], "application/octet-stream", "clip.mp4")
    assert sniffed == "video/mp4"


def test_guess_falls_back_to_filename() -> None:
    assert guess_video_mime(b"\x00" * 32, "", "clip.mov") == "video/quicktime"
    assert guess_video_mime(b"\x00" * 32, "", "clip.webm") == "video/webm"


def test_guess_rejects_fake_mp4() -> None:
    sniffed = guess_video_mime(b"not-a-real-video-file-header-1234", "video/mp4", "clip.mp4")
    assert sniffed in ("application/octet-stream", "video/mp4")


def test_content_type_matches_with_octet_stream_pass() -> None:
    allowed = ["video/mp4", "video/quicktime", "application/octet-stream"]
    assert content_type_matches("application/octet-stream", allowed)
    assert content_type_matches("video/mp4", allowed)
    assert not content_type_matches("image/png", allowed)


def test_extension_for_mime_mapping() -> None:
    assert extension_for_mime("video/mp4") == ".mp4"
    assert extension_for_mime("video/quicktime") == ".mov"
    assert extension_for_mime("video/x-matroska") == ".mkv"
    assert extension_for_mime("text/plain", fallback=".mp4") == ".mp4"


def test_hex_ok_validator() -> None:
    assert hex_ok("deadbeef")
    assert hex_ok("")
    assert not hex_ok("xyz")
    assert not hex_ok("deadbeeG")


@pytest.mark.parametrize("attempt", [0, 1, 2, 5, 10])
def test_backoff_delay_within_bounds(attempt: int) -> None:
    for _ in range(50):
        delay = backoff_delay_s(attempt, base_s=1.0, max_s=30.0)
        assert 0.0 <= delay <= 30.0


def test_backoff_delay_grows_then_caps() -> None:
    first = max(backoff_delay_s(0, 1.0, 30.0) for _ in range(200))
    late = max(backoff_delay_s(8, 1.0, 30.0) for _ in range(200))
    assert late > first
    assert late <= 30.0
