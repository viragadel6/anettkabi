"""Predict subcommands: create/get/list/cancel/delete/wait/download."""

from __future__ import annotations

import sys
from typing import Any

from vsfx_client import Client, SfxParams

from cli.output import emit_json, emit_table, print_line


def _params_from_args(args: Any) -> SfxParams:
    """Build SfxParams from parsed CLI args.

    Parameters:
        args: Argparse namespace.

    Returns:
        Parameters object.
    """
    return SfxParams(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        duration=args.duration,
        start_time=args.start_time,
        audio_mode=args.audio_mode,
        sfx_gain_db=args.sfx_gain_db,
        original_audio_gain_db=args.original_gain_db,
        duck_threshold_db=args.duck_threshold_db,
        duck_ratio=args.duck_ratio,
        duck_attack_ms=args.duck_attack_ms,
        duck_release_ms=args.duck_release_ms,
        target_loudness_lufs=args.loudness,
        true_peak_db=args.true_peak,
        video_handling=args.video_handling,
        return_audio_only=args.audio_only,
        enable_safety_checker=None if args.no_safety is None else (not args.no_safety),
        webhook_url=args.webhook_url,
    )


def run_create(client: Client, args: Any) -> int:
    """Handle `vsfx predict create`.

    Parameters:
        client: SDK client.
        args: Parsed args with .video, .idempotency_key, .wait, .out.

    Returns:
        Exit code.
    """
    params = _params_from_args(args)
    if args.file is not None:
        prediction = client.create_multipart(args.file, params, idempotency_key=args.idempotency_key)
    else:
        prediction = client.create(args.video, params, idempotency_key=args.idempotency_key)
    if args.json:
        emit_json(prediction.raw)
        return 0
    print_line(f"created {prediction.id} ({prediction.status})")
    if args.wait:
        prediction = client.wait(
            prediction.id,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            on_poll=lambda current: print_line(f"  {current.status}"),
        )
    if prediction.status == "succeeded" and args.out:
        destination = client.download(prediction, args.out)
        print_line(f"downloaded {destination}")
    if prediction.is_terminal:
        emit_table(
            ["id", "status"],
            [(prediction.id, prediction.status)],
        )
        return 0 if prediction.status == "succeeded" else 1
    print_line(f"poll with: vsfx predict get {prediction.id}")
    return 0


def run_get(client: Client, args: Any) -> int:
    """Handle `vsfx predict get ID`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    prediction = client.get(args.prediction_id)
    if args.json:
        emit_json(prediction.raw)
    else:
        emit_table(
            ["field", "value"],
            [
                ("id", prediction.id),
                ("status", prediction.status),
                ("created_at", prediction.created_at),
                ("completed_at", prediction.completed_at or "-"),
                ("error", str((prediction.error or {}).get("error_code", "-"))),
            ],
        )
    return 0


def run_list(client: Client, args: Any) -> int:
    """Handle `vsfx predict list`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    predictions = client.list(status=args.status, limit=args.limit, after=args.after)
    if args.json:
        emit_json([prediction.raw for prediction in predictions])
    else:
        emit_table(
            ["id", "status", "created_at"],
            [(prediction.id, prediction.status, prediction.created_at) for prediction in predictions],
        )
    return 0


def run_cancel(client: Client, args: Any) -> int:
    """Handle `vsfx predict cancel ID`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    prediction = client.cancel(args.prediction_id)
    if args.json:
        emit_json(prediction.raw)
    else:
        print_line(f"{prediction.id} -> {prediction.status}")
    return 0


def run_delete(client: Client, args: Any) -> int:
    """Handle `vsfx predict delete ID`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    client.delete(args.prediction_id)
    print_line(f"deleted {args.prediction_id}")
    return 0


def run_wait(client: Client, args: Any) -> int:
    """Handle `vsfx predict wait ID`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code 0 on success, 1 otherwise.
    """
    prediction = client.wait(
        args.prediction_id,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        on_poll=lambda current: print_line(f"  {current.status}", quiet=args.quiet),
    )
    if args.json:
        emit_json(prediction.raw)
    if prediction.status == "succeeded" and args.out:
        client.download(prediction, args.out)
    if prediction.status != "succeeded":
        print_line(f"terminal status: {prediction.status}")
        return 1
    return 0


def run_download(client: Client, args: Any) -> int:
    """Handle `vsfx predict download ID`.

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    prediction = client.get(args.prediction_id)
    destination = client.download(prediction, args.out or f"{prediction.id}.mp4")
    print_line(f"saved {destination}")
    return 0


def run_follow(client: Client, args: Any) -> int:
    """Handle `vsfx predict follow ID` (stream status until terminal).

    Parameters:
        client: SDK client.
        args: Parsed args.

    Returns:
        Exit code.
    """
    import time

    while True:
        prediction = client.get(args.prediction_id)
        sys.stdout.write(f"\r{prediction.id}: {prediction.status:<12}")
        sys.stdout.flush()
        if prediction.is_terminal:
            sys.stdout.write("\n")
            return 0 if prediction.status == "succeeded" else 1
        time.sleep(args.poll_interval)
