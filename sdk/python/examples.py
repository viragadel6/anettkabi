"""End-to-end usage examples for the vsfx Python SDK.

Run against a live deployment:

    VSFX_API_KEY=... VSFX_BASE_URL=... python examples.py demo /tmp/clip.mp4
"""

from __future__ import annotations

import sys
import time

from vsfx_client import (
    APIError,
    AsyncClient,
    Client,
    QuotaExceededError,
    RateLimitedError,
    SfxParams,
    TransportError,
)

DEMO_PROMPT = "cinematic whoosh leading to a deep sub-bass impact, tight tail"


def sync_demo(base_url: str, api_key: str, video: str) -> None:
    """Synchronous flow: create, poll, download.

    Parameters:
        base_url: Service origin.
        api_key: API key.
        video: Video URL or local path.
    """
    with Client(api_key, base_url=base_url) as client:
        info = client.model_info()
        print("model:", info.raw.get("id", "video-to-video-sfx"))
        prediction = client.create(video, SfxParams(prompt=DEMO_PROMPT, audio_mode="duck", seed=42))
        print("created:", prediction.id, prediction.status)
        while prediction.is_active:
            time.sleep(2.0)
            prediction = client.get(prediction.id)
            print("status:", prediction.status)
        if prediction.status == "succeeded":
            destination = client.download(prediction, f"{prediction.id}.mp4")
            print("downloaded:", destination)
        else:
            print("finished:", prediction.status, prediction.error)


async def async_demo(base_url: str, api_key: str, video: str) -> None:
    """Async flow with concurrent waits and typed error handling.

    Parameters:
        base_url: Service origin.
        api_key: API key.
        video: Video URL.
    """
    import asyncio

    from vsfx_client import gather_predictions

    async with AsyncClient(api_key, base_url=base_url) as client:
        first = await client.create(video, SfxParams(prompt=DEMO_PROMPT, seed=1))
        second = await client.create(video, SfxParams(prompt=DEMO_PROMPT, seed=2))
        try:
            results = await asyncio.wait_for(
                gather_predictions(client, [first.id, second.id]), timeout=600.0
            )
        except RateLimitedError as exc:
            print("rate limited; retry after", exc.retry_after)
            return
        except QuotaExceededError:
            print("quota exhausted this month")
            return
        except TransportError as exc:
            print("transport failure:", exc.message)
            return
        for prediction in results:
            print(prediction.id, prediction.status, (prediction.metrics or {}).get("total_s"))


def error_demo(base_url: str, api_key: str) -> None:
    """Retryable-error handling pattern.

    Parameters:
        base_url: Service origin.
        api_key: API key.
    """
    with Client(api_key, base_url=base_url) as client:
        attempt = 0
        while True:
            try:
                client.health()
            except APIError as exc:
                if not exc.retryable or attempt >= 5:
                    raise
                time.sleep(min(10.0, 2.0**attempt))
                attempt += 1
            else:
                return


def main() -> int:
    """CLI dispatcher for the examples.

    Returns:
        0 on success, 2 on usage errors.
    """
    import asyncio
    import os

    args = sys.argv[1:]
    if len(args) < 2 or args[0] not in {"sync", "async", "errors"}:
        print("usage: python examples.py {sync|async|errors} VIDEO_OR_NOTHING")
        return 2
    base_url = os.environ.get("VSFX_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("VSFX_API_KEY", "")
    if not api_key:
        print("set VSFX_API_KEY")
        return 2
    mode = args[0]
    video = args[1] if len(args) > 1 else ""
    if mode == "sync":
        sync_demo(base_url, api_key, video)
    elif mode == "async":
        asyncio.run(async_demo(base_url, api_key, video))
    else:
        error_demo(base_url, api_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
