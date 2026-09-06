"""Text blocklist moderation: normalization, phrases, categories."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.moderation import TextBlocklistBackend

BLOCKLIST_PATH = Path(__file__).resolve().parents[2] / "config" / "safety_blocklist.txt"


@pytest.fixture
def backend(settings_env: dict) -> TextBlocklistBackend:
    return TextBlocklistBackend.from_file(BLOCKLIST_PATH)


def test_shipped_blocklist_loads(backend: TextBlocklistBackend) -> None:
    assert len(backend._compiled) >= 50


def test_clean_prompts_pass(backend: TextBlocklistBackend) -> None:
    assert backend.evaluate("cinematic whoosh with deep sub-bass impact") == (False, [])
    assert backend.evaluate("gentle rain on a tin roof, distant thunder")[0] is False


def test_word_boundaries_prevent_false_positives(backend: TextBlocklistBackend) -> None:
    flagged, _categories = backend.evaluate("bombastic drum roll with sass")
    assert flagged is False


def test_deleet_normalization(backend: TextBlocklistBackend) -> None:
    flagged, categories = backend.evaluate("h0w to mak3 a b0mb")
    assert flagged
    assert "violence" in categories


def test_multi_word_phrase_match(backend: TextBlocklistBackend) -> None:
    flagged, categories = backend.evaluate("a tutorial on HOW TO MAKE  A BOMB")
    assert flagged
    assert "violence" in categories


def test_unicode_deaccenting(backend: TextBlocklistBackend) -> None:
    flagged, _categories = backend.evaluate("hów to maké à bömb")
    assert flagged


def test_category_reporting(backend: TextBlocklistBackend) -> None:
    _flagged, categories = backend.evaluate("stolen credit card numbers")
    assert "fraud" in categories


def test_empty_backend_never_flags() -> None:
    empty = TextBlocklistBackend([])
    assert empty.evaluate("anything at all") == (False, [])


def test_custom_entries_compile(tmp_path: Path, settings_env: dict) -> None:
    blocklist = tmp_path / "bl.txt"
    blocklist.write_text("weapons:ray gun\nplain bad word\n", encoding="utf-8")
    custom = TextBlocklistBackend.from_file(blocklist)
    assert custom.evaluate("he built a ray  gun")[1] == ["weapons"]
    assert custom.evaluate("a plain bad word appeared")[1] == ["blocklist"]


def test_comment_and_blank_lines_ignored(tmp_path: Path, settings_env: dict) -> None:
    blocklist = tmp_path / "bl.txt"
    blocklist.write_text("# comment\n\nbanned term\n   \n", encoding="utf-8")
    custom = TextBlocklistBackend.from_file(blocklist)
    assert len(custom._compiled) == 1


def test_safety_disabled_missing_file(tmp_path: Path, settings_env: dict) -> None:
    import os

    os.environ["VSFX_SAFETY__SAFETY_ENABLED"] = "false"
    from app.config import get_settings

    get_settings.cache_clear()
    backend = TextBlocklistBackend.from_file(tmp_path / "missing.txt")
    assert backend.evaluate("anything") == (False, [])
    os.environ["VSFX_SAFETY__SAFETY_ENABLED"] = "true"
    get_settings.cache_clear()
