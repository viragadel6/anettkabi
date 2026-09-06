"""Health/models/webhook command re-exports for the dispatcher."""

from __future__ import annotations

from typing import Any

from vsfx_client import Client

from cli.output import emit_json, print_line

__all__ = ["run_health", "run_ready", "run_version"]


def run_health(client: Client, args: Any) -> int:
    """Handle `vsfx status health`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    payload = client.health()
    if args.json:
        emit_json(payload)
    else:
        print_line(str(payload))
    return 0


def run_ready(client: Client, args: Any) -> int:
    """Handle `vsfx status ready`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code 1 when not ready.
    """
    payload = client.ready()
    if args.json:
        emit_json(payload)
        ready = str(payload.get("status", "")) == "ready"
    else:
        ready = str(payload.get("status", "")) == "ready"
        print_line(f"ready: {payload}")
    return 0 if ready else 1


def run_version(client: Client, args: Any) -> int:
    """Handle `vsfx status version`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    payload = client.version()
    if args.json:
        emit_json(payload)
    else:
        print_line(str(payload))
    return 0
