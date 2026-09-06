"""CLI parser and dispatch behavior (offline paths only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.vsfx import build_parser, main


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.delenv("VSFX_API_KEY", raising=False)
    monkeypatch.delenv("VSFX_BASE_URL", raising=False)
    return home


def test_parser_builds_full_tree() -> None:
    parser = build_parser()
    assert parser.prog == "vsfx"
    help_text = parser.format_help()
    for command in ("predict", "upload", "account", "models", "webhooks", "config", "status"):
        assert command in help_text


def test_create_requires_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "create", "--prompt", "x"])


def test_create_prompt_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "create", "--video", "https://x/y.mp4"])


def test_audio_mode_choices_enforced() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["predict", "create", "--video", "https://x/y.mp4", "--prompt", "p", "--audio-mode", "blend"]
        )


def test_config_show_runs_offline(config_home: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["config", "show", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "default"
    assert payload["api_key"] == "-"


def test_config_set_then_show(config_home: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["config", "set", "--api-key", "vsfx_cli_test", "--base-url", "https://api.example.com"]) == 0
    capsys.readouterr()
    assert main(["config", "show", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_url"] == "https://api.example.com"
    assert payload["api_key"].startswith("vsfx_cli_t")


def test_config_set_requires_values(config_home: Path) -> None:
    assert main(["config", "set"]) == 2


def test_missing_api_key_maps_to_exit_2(config_home: Path) -> None:
    assert main(["predict", "list"]) == 2


def test_json_flag_accepted_after_subcommand(config_home: Path) -> None:
    args = build_parser().parse_args(["config", "show", "--json"])
    assert args.json is True


def test_json_flag_accepted_before_subcommand() -> None:
    args = build_parser().parse_args(["--json", "config", "show"])
    assert args.json is True


def test_global_key_override(config_home: Path) -> None:
    args = build_parser().parse_args(["--api-key", "vsfx_inline", "predict", "list"])
    assert args.api_key == "vsfx_inline"
