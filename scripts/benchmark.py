"""Benchmark end-to-end latency and throughput against a running deployment."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


async def one_run(
    client: httpx.AsyncClient,
    base_url: str,
    video_url: str,
    prompt: str,
    timeout_s: float,
) -> dict[str, float]:
    """Submit and track one prediction.

    Parameters:
        client: HTTP client.
        base_url: API base URL.
        video_url: Source video URL.
        prompt: Generation prompt.
        timeout_s: Wall budget.

    Returns:
        Dict with submit_ms, total_ms, status.
    """
    started = time.perf_counter()
    response = await client.post(
        f"{base_url}/api/v1/predictions/video-to-video-sfx",
        json={"video": video_url, "prompt": prompt, "num_inference_steps": 10},
    )
    response.raise_for_status()
    submit_ms = (time.perf_counter() - started) * 1000
    payload = response.json()["data"]
    while True:
        await asyncio.sleep(1.0)
        poll = await client.get(f"{base_url}/api/v1/predictions/{payload['id']}/result")
        poll.raise_for_status()
        data = poll.json()["data"]
        if data["status"] in ("completed", "failed", "canceled"):
            total_ms = (time.perf_counter() - started) * 1000
            return {
                "submit_ms": round(submit_ms, 1),
                "total_ms": round(total_ms, 1),
                "status": float(data["status"] == "completed"),
            }
        if time.perf_counter() - started > timeout_s:
            return {"submit_ms": round(submit_ms, 1), "total_ms": -1.0, "status": 0.0}


async def run(base_url: str, api_key: str, video_url: str, prompt: str, runs: int, concurrency: int, timeout_s: float) -> None:
    """Execute the benchmark and print aggregate statistics.

    Parameters:
        base_url: API base.
        api_key: Bearer key.
        video_url: Source video.
        prompt: Prompt.
        runs: Total runs.
        concurrency: Parallel workers.
        timeout_s: Per-run budget.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(index: int) -> dict[str, float]:
            async with semaphore:
                print(f"run {index + 1}/{runs} starting")
                return await one_run(client, base_url, video_url, prompt, timeout_s)

        results = await asyncio.gather(*(_guarded(i) for i in range(runs)))
    totals = [r["total_ms"] for r in results if r["total_ms"] > 0]
    print(f"completed: {sum(r['status'] for r in results):.0f}/{runs}")
    if totals:
        print(f"total_ms p50={statistics.median(totals):.0f} max={max(totals):.0f} min={min(totals):.0f}")


def main() -> int:
    """CLI entry.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--prompt", default="crisp footsteps on gravel, wind")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.api_key, args.video_url, args.prompt, args.runs, args.concurrency, args.timeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
