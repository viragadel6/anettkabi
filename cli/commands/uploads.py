"""Upload subcommands: direct upload and presigned upload."""

from __future__ import annotations

from typing import Any

from vsfx_client import Client

from cli.output import emit_json, emit_table, print_line


def run_upload(client: Client, args: Any) -> int:
    """Handle `vsfx upload FILE`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    ack = client.upload(args.path)
    if args.json:
        emit_json({"url": ack.url, "path": ack.path, "expires_at": ack.expires_at})
    else:
        emit_table(
            ["field", "value"],
            [("url", ack.url), ("expires_at", ack.expires_at or "-")],
        )
    print_line("pass it with: vsfx predict create --video URL --prompt ...")
    return 0


def run_presign(client: Client, args: Any) -> int:
    """Handle `vsfx upload presign --filename NAME`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    presign = client.presign(filename=args.filename, content_type=args.content_type)
    if args.json:
        emit_json(
            {
                "upload_url": presign.upload_url,
                "url": presign.url,
                "expires_at": presign.expires_at,
            }
        )
    else:
        emit_table(
            ["field", "value"],
            [("upload_url", presign.upload_url), ("url", presign.url)],
        )
    return 0


def run_put(client: Client, args: Any) -> int:
    """Handle `vsfx upload put FILE` (presign + PUT + report URL).

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    ack = client.presigned_upload(args.path)
    if args.json:
        emit_json({"url": ack.url, "path": ack.path, "expires_at": ack.expires_at})
    else:
        emit_table(
            ["field", "value"],
            [("url", ack.url), ("expires_at", ack.expires_at or "-")],
        )
    return 0
