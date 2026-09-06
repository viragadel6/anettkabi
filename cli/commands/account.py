"""Account, models, and webhook subcommands."""

from __future__ import annotations

from typing import Any

from vsfx_client import Client

from cli.output import emit_json, emit_table, print_line


def run_usage(client: Client, args: Any) -> int:
    """Handle `vsfx account usage`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    usage = client.usage()
    if args.json:
        emit_json(usage.raw)
    else:
        emit_table(["field", "value"], list(usage.raw.items()))
    return 0


def run_models(client: Client, args: Any) -> int:
    """Handle `vsfx models show`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    info = client.model_info()
    if args.json:
        emit_json(info.raw)
    else:
        emit_table(["field", "value"], list(info.raw.items())[:24])
    return 0


def run_webhook_test(client: Client, args: Any) -> int:
    """Handle `vsfx webhooks test --url URL`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    result = client.test_webhook(args.url)
    if args.json:
        emit_json(result.raw)
    else:
        print_line(f"accepted={result.accepted} status={result.status_code or '-'}")
    return 0 if result.accepted else 1
