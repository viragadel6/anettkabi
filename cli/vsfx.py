"""vsfx: command-line client for the Video-to-Video SFX service."""

from __future__ import annotations

import argparse
from typing import Any

from cli.client_factory import build_client
from cli.commands import account, config_cmd, predict, uploads, webhooks
from cli.config import ConfigError, load_profile, require_api_key
from cli.output import emit_error


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argparse tree.

    Returns:
        Root parser.
    """
    parser = argparse.ArgumentParser(
        prog="vsfx",
        description="Video-to-Video SFX command-line client",
    )
    parser.add_argument("--profile", default="default", help="config profile (default: default)")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--quiet", action="store_true", help="suppress progress lines")
    parser.add_argument("--api-key", default=None, help="override the configured API key")
    parser.add_argument("--base-url", default=None, help="override the configured base URL")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS)

    predict_cmd = sub.add_parser("predict", help="prediction lifecycle", parents=[common])
    predict_sub = predict_cmd.add_subparsers(dest="predict_command", required=True)

    create = predict_sub.add_parser("create", help="create a prediction", parents=[common])
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="source video https URL")
    source.add_argument("--file", help="local video file (multipart upload)")
    create.add_argument("--prompt", required=True)
    create.add_argument("--negative-prompt", dest="negative_prompt")
    create.add_argument("--seed", type=int)
    create.add_argument("--steps", type=int, dest="steps")
    create.add_argument("--guidance", type=float)
    create.add_argument("--duration", type=float)
    create.add_argument("--start-time", type=float, dest="start_time")
    create.add_argument(
        "--audio-mode", dest="audio_mode", choices=("replace", "mix", "duck")
    )
    create.add_argument("--sfx-gain-db", type=float, dest="sfx_gain_db")
    create.add_argument("--original-gain-db", type=float, dest="original_gain_db")
    create.add_argument("--duck-threshold-db", type=float, dest="duck_threshold_db")
    create.add_argument("--duck-ratio", type=float, dest="duck_ratio")
    create.add_argument("--duck-attack-ms", type=float, dest="duck_attack_ms")
    create.add_argument("--duck-release-ms", type=float, dest="duck_release_ms")
    create.add_argument("--loudness", type=float, help="target loudness in LUFS")
    create.add_argument("--true-peak", type=float, dest="true_peak")
    create.add_argument("--video-handling", dest="video_handling", choices=("copy", "reencode"))
    create.add_argument("--audio-only", dest="audio_only", action="store_true")
    create.add_argument("--no-safety", dest="no_safety", action="store_true")
    create.add_argument("--webhook-url", dest="webhook_url")
    create.add_argument("--idempotency-key", dest="idempotency_key")
    create.add_argument("--wait", action="store_true", help="block until terminal")
    create.add_argument("--poll-interval", type=float, default=2.0, dest="poll_interval")
    create.add_argument("--timeout", type=float, default=None)
    create.add_argument("--out", help="download output here once succeeded")

    get_parser = predict_sub.add_parser("get", help="fetch one prediction", parents=[common])
    get_parser.add_argument("prediction_id")
    wait_parser = predict_sub.add_parser("wait", help="wait for terminal state", parents=[common])
    wait_parser.add_argument("prediction_id")
    wait_parser.add_argument("--poll-interval", type=float, default=2.0, dest="poll_interval")
    wait_parser.add_argument("--timeout", type=float, default=None)
    wait_parser.add_argument("--out")
    list_parser = predict_sub.add_parser("list", help="list predictions", parents=[common])
    list_parser.add_argument("--status")
    list_parser.add_argument("--limit", type=int)
    list_parser.add_argument("--after")
    cancel_parser = predict_sub.add_parser("cancel", help="request cancellation", parents=[common])
    cancel_parser.add_argument("prediction_id")
    delete_parser = predict_sub.add_parser("delete", help="soft-delete", parents=[common])
    delete_parser.add_argument("prediction_id")
    download_parser = predict_sub.add_parser("download", help="download output", parents=[common])
    download_parser.add_argument("prediction_id")
    download_parser.add_argument("--out")
    follow_parser = predict_sub.add_parser("follow", help="live status line", parents=[common])
    follow_parser.add_argument("prediction_id")
    follow_parser.add_argument("--poll-interval", type=float, default=2.0, dest="poll_interval")

    upload_cmd = sub.add_parser("upload", help="video uploads", parents=[common])
    upload_sub = upload_cmd.add_subparsers(dest="upload_command", required=True)
    direct = upload_sub.add_parser("direct", help="POST /uploads multipart", parents=[common])
    direct.add_argument("path")
    presign_parser = upload_sub.add_parser("presign", help="presigned PUT for a filename", parents=[common])
    presign_parser.add_argument("--filename", required=True)
    presign_parser.add_argument("--content-type", default="video/mp4", dest="content_type")
    put_parser = upload_sub.add_parser("put", help="presign + PUT a local file", parents=[common])
    put_parser.add_argument("path")

    account_cmd = sub.add_parser("account", help="account information", parents=[common])
    account_sub = account_cmd.add_subparsers(dest="account_command", required=True)
    account_sub.add_parser("usage", help="usage snapshot", parents=[common])

    models_cmd = sub.add_parser("models", help="model information", parents=[common])
    models_sub = models_cmd.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("show", help="model card", parents=[common])

    webhooks_cmd = sub.add_parser("webhooks", help="webhook utilities", parents=[common])
    webhooks_sub = webhooks_cmd.add_subparsers(dest="webhooks_command", required=True)
    webhook_test = webhooks_sub.add_parser("test", help="send a signed probe", parents=[common])
    webhook_test.add_argument("--url", required=True)

    config_cmd_parser = sub.add_parser("config", help="CLI configuration", parents=[common])
    config_sub = config_cmd_parser.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", help="show active profile", parents=[common])
    config_set = config_sub.add_parser("set", help="set profile values", parents=[common])
    config_set.add_argument("--api-key", dest="api_key")
    config_set.add_argument("--base-url", dest="base_url")
    config_sub.add_parser("set-stdin", help="read API key from stdin", parents=[common])

    status_cmd = sub.add_parser("status", help="service probes", parents=[common])
    status_sub = status_cmd.add_subparsers(dest="status_command", required=True)
    status_sub.add_parser("health", help="liveness", parents=[common])
    status_sub.add_parser("ready", help="readiness", parents=[common])
    status_sub.add_parser("version", help="service version", parents=[common])

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Parameters:
        argv: Argument vector (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "json"):
        args.json = False
    if not hasattr(args, "quiet"):
        args.quiet = False
    try:
        if args.command == "config":
            return _run_config(args)
        profile = load_profile(args.profile)
        if args.base_url:
            profile.base_url = args.base_url
        api_key = args.api_key or require_api_key(profile)
        client = build_client(profile, api_key)
        try:
            return _dispatch(client, args)
        finally:
            client.close()
    except ConfigError as exc:
        emit_error(str(exc))
        return 2
    except BrokenPipeError:
        return 0


def _run_config(args: Any) -> int:
    """Dispatch config subcommands (no API key needed).

    Parameters:
        args: Parsed args.

    Returns:
        Exit code.
    """
    if args.config_command == "show":
        return config_cmd.run_config_show(args)
    if args.config_command == "set":
        return config_cmd.run_config_set(args)
    return config_cmd.run_config_set_stdin(args)


def _dispatch(client: Any, args: Any) -> int:
    """Route parsed args to command handlers.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code 2 for unknown routing.
    """
    table = {
        ("predict", "create"): lambda: predict.run_create(client, args),
        ("predict", "get"): lambda: predict.run_get(client, args),
        ("predict", "list"): lambda: predict.run_list(client, args),
        ("predict", "cancel"): lambda: predict.run_cancel(client, args),
        ("predict", "delete"): lambda: predict.run_delete(client, args),
        ("predict", "wait"): lambda: predict.run_wait(client, args),
        ("predict", "download"): lambda: predict.run_download(client, args),
        ("predict", "follow"): lambda: predict.run_follow(client, args),
        ("upload", "direct"): lambda: uploads.run_upload(client, args),
        ("upload", "presign"): lambda: uploads.run_presign(client, args),
        ("upload", "put"): lambda: uploads.run_put(client, args),
        ("account", "usage"): lambda: account.run_usage(client, args),
        ("models", "show"): lambda: account.run_models(client, args),
        ("webhooks", "test"): lambda: webhooks.run_webhook_test(client, args),
        ("status", "health"): lambda: webhooks.run_health(client, args),
        ("status", "ready"): lambda: webhooks.run_ready(client, args),
        ("status", "version"): lambda: webhooks.run_version(client, args),
    }
    handler = table.get((args.command, getattr(args, f"{args.command}_command", None)))
    if handler is None:
        emit_error(f"unrouted command {args.command}")
        return 2
    return int(handler())


if __name__ == "__main__":
    raise SystemExit(main())
