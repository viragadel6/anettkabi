"""CLI profile configuration persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.config import CliProfile, ConfigError, load_profile, require_api_key, save_profile


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.delenv("VSFX_API_KEY", raising=False)
    monkeypatch.delenv("VSFX_BASE_URL", raising=False)
    return home


def test_load_default_without_file(config_home: Path) -> None:
    profile = load_profile()
    assert profile.name == "default"
    assert profile.api_key == ""
    assert profile.base_url.startswith("http")


def test_save_then_load_roundtrip(config_home: Path) -> None:
    save_profile(CliProfile(name="prod", base_url="https://api.example.com", api_key="vsfx_k"))
    profile = load_profile("prod")
    assert profile.base_url == "https://api.example.com"
    assert profile.api_key == "vsfx_k"


def test_profiles_do_not_clobber_each_other(config_home: Path) -> None:
    save_profile(CliProfile(name="prod", base_url="https://api.example.com", api_key="k1"))
    save_profile(CliProfile(name="staging", base_url="https://staging.example.com", api_key="k2"))
    assert load_profile("prod").api_key == "k1"
    assert load_profile("staging").api_key == "k2"


def test_env_overrides_file(config_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_profile(CliProfile(name="default", base_url="https://api.example.com", api_key="file-key"))
    monkeypatch.setenv("VSFX_API_KEY", "env-key")
    assert load_profile().api_key == "env-key"
    monkeypatch.delenv("VSFX_API_KEY")
    assert load_profile().api_key == "file-key"


def test_require_api_key_guidance(config_home: Path) -> None:
    with pytest.raises(ConfigError, match="VSFX_API_KEY"):
        require_api_key(load_profile())


def test_invalid_toml_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "xdg2"
    (home / "vsfx").mkdir(parents=True)
    (home / "vsfx" / "config.toml").write_text("not [valid toml", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    with pytest.raises(ConfigError, match="cannot parse"):
        load_profile()
