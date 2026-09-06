"""TOML profile configuration for the vsfx CLI (~/.config/vsfx/config.toml)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CliProfile", "ConfigError", "config_path", "load_profile", "save_profile"]

DEFAULT_BASE_URL = "http://localhost:8000"


class ConfigError(Exception):
    """Raised for missing or invalid CLI configuration."""


@dataclass(slots=True)
class CliProfile:
    """One named CLI profile.

    Attributes:
        name: Profile name.
        base_url: Service origin.
        api_key: Bearer key.
    """

    name: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""


def config_path() -> Path:
    """Return the config file location.

    Returns:
        Path under $XDG_CONFIG_HOME or ~/.config.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg) if xdg else Path.home() / ".config"
    return root / "vsfx" / "config.toml"


def load_profile(name: str = "default") -> CliProfile:
    """Load a profile merged with environment overrides.

    Precedence: env VSFX_API_KEY / VSFX_BASE_URL, then profile, then defaults.

    Parameters:
        name: Profile name.

    Returns:
        The resolved CliProfile.

    Raises:
        ConfigError: When the config file is invalid.
    """
    profile = CliProfile(name=name)
    path = config_path()
    if path.is_file():
        try:
            payload: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ConfigError(f"cannot parse {path}: {exc}") from exc
        profiles = payload.get("profiles")
        section = profiles.get(name) if isinstance(profiles, dict) else None
        if isinstance(section, dict):
            profile.base_url = str(section.get("base_url", profile.base_url))
            profile.api_key = str(section.get("api_key", profile.api_key))
    profile.api_key = os.environ.get("VSFX_API_KEY", profile.api_key)
    profile.base_url = os.environ.get("VSFX_BASE_URL", profile.base_url)
    return profile


def save_profile(profile: CliProfile) -> Path:
    """Persist a profile (rewrite preserving other profiles).

    Parameters:
        profile: Profile to store.

    Returns:
        The config path.

    Raises:
        ConfigError: On write failures.
    """
    path = config_path()
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise ConfigError(f"cannot parse {path}: {exc}") from exc
    profiles = payload.setdefault("profiles", {})
    profiles[profile.name] = {"base_url": profile.base_url, "api_key": profile.api_key}
    lines = []
    for profile_name, section in sorted(profiles.items()):
        lines.append(f"[profiles.{profile_name}]")
        lines.append(f'base_url = "{section["base_url"]}"')
        lines.append(f'api_key = "{section["api_key"]}"')
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def require_api_key(profile: CliProfile) -> str:
    """Return the API key or fail with guidance.

    Parameters:
        profile: Resolved profile.

    Returns:
        The key.

    Raises:
        ConfigError: When no key is configured.
    """
    if not profile.api_key:
        raise ConfigError(
            "no API key configured; set VSFX_API_KEY or run "
            "`vsfx config set --api-key VSFX_... [--base-url URL]`"
        )
    return profile.api_key
