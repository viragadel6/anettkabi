"""Client construction for the vsfx CLI (SDK bootstrap + profile binding)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from cli.config import CliProfile

__all__ = ["build_client"]


def _ensure_sdk_importable() -> None:
    """Add the in-repo SDK to sys.path when the package is not installed.

    Raises:
        ImportError: When the SDK cannot be located at all.
    """
    try:
        importlib.import_module("vsfx_client")
    except ImportError:
        candidate = Path(__file__).resolve().parents[1] / "sdk" / "python"
        if not (candidate / "vsfx_client").is_dir():
            raise
        sys.path.insert(0, str(candidate))


def build_client(profile: CliProfile, api_key: str):
    """Create a sync SDK client for the resolved profile.

    Parameters:
        profile: CLI profile.
        api_key: Required API key.

    Returns:
        Configured vsfx_client.Client.
    """
    _ensure_sdk_importable()
    from vsfx_client import Client

    return Client(api_key, base_url=profile.base_url)
