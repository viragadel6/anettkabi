"""Config subcommand: set/show CLI profiles."""

from __future__ import annotations

import sys
from typing import Any

from cli.config import ConfigError, config_path, load_profile, save_profile
from cli.output import emit_json, emit_table, print_line


def run_config_show(args: Any) -> int:
    """Handle `vsfx config show`.

    Parameters:
        args: Parsed args.

    Returns:
        Exit code.
    """
    profile = load_profile(args.profile)
    masked = profile.api_key[:10] + "..." if profile.api_key else "-"
    if args.json:
        emit_json(
            {
                "profile": profile.name,
                "base_url": profile.base_url,
                "api_key": masked,
                "config_path": str(config_path()),
            }
        )
    else:
        emit_table(
            ["field", "value"],
            [
                ("profile", profile.name),
                ("base_url", profile.base_url),
                ("api_key", masked),
                ("config_path", str(config_path())),
            ],
        )
    return 0


def run_config_set(args: Any) -> int:
    """Handle `vsfx config set`.

    Parameters:
        args: Parsed args with --api-key/--base-url.

    Returns:
        Exit code 2 on conflicts.
    """
    if not args.api_key and not args.base_url:
        print_line("nothing to set; provide --api-key and/or --base-url")
        return 2
    if args.api_key and args.api_key.startswith("-"):
        raise ConfigError("api key looks like a flag")
    profile = load_profile(args.profile)
    if args.base_url:
        profile.base_url = args.base_url
    if args.api_key:
        profile.api_key = args.api_key
    path = save_profile(profile)
    print_line(f"saved profile {profile.name} to {path}")
    return 0


def run_config_set_stdin(args: Any) -> int:
    """Handle `vsfx config set-stdin` (key piped privately).

    Parameters:
        args: Parsed args.

    Returns:
        Exit code.
    """
    if sys.stdin.isatty():
        print_line("pipe the key: printf '%s' 'VSFX_...' | vsfx config set-stdin")
        return 2
    key = sys.stdin.read().strip()
    if not key:
        print_line("empty key")
        return 2
    profile = load_profile(args.profile)
    profile.api_key = key
    save_profile(profile)
    print_line(f"saved profile {profile.name}")
    return 0
