"""Seed development data: a dev API key and a sample queued prediction."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.constants import API_KEY_PREFIX, API_KEY_RANDOM_LEN, API_KEY_SECRET_LEN  # noqa: E402
from app.db.models import AssetKind  # noqa: E402
from app.db.repositories.api_keys import ApiKeyRepository  # noqa: E402
from app.db.repositories.assets import AssetRepository  # noqa: E402
from app.db.repositories.predictions import PredictionRepository  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.middleware.auth import hash_api_key  # noqa: E402
from app.services.queue import enqueue_job  # noqa: E402
from app.utils.ids import new_api_secret, new_prediction_id  # noqa: E402


async def seed(sample_video_url: str) -> str:
    """Insert a dev key and one queued sample prediction.

    Parameters:
        sample_video_url: URL used as the sample input.

    Returns:
        The generated API key plaintext.

    Raises:
        Exception: Database/queue errors propagate.
    """
    settings = get_settings()
    prefix = new_api_secret(API_KEY_RANDOM_LEN)
    secret = new_api_secret(API_KEY_SECRET_LEN)
    plaintext = f"{API_KEY_PREFIX}_{prefix}_{secret}"
    async with session_scope() as session:
        key = await ApiKeyRepository(session).create(
            name="dev-seed",
            key_prefix=prefix,
            key_hash=hash_api_key(secret),
            scopes=["predictions:write", "predictions:read", "admin"],
            rate_limit_rpm=600,
            concurrency_limit=10,
            monthly_seconds_quota=1_000_000,
        )
        asset = await AssetRepository(session).register(
            kind=AssetKind.SOURCE_VIDEO,
            storage_key="seed/placeholder-not-uploaded.mp4",
            bucket=settings.storage.s3_bucket,
            content_type="video/mp4",
            size_bytes=0,
            sha256="",
        )
        prediction_id = new_prediction_id()
        await PredictionRepository(session).create(
            prediction_id=prediction_id,
            api_key_id=key.id,
            model=settings.model.model_id,
            input_payload={
                "video": sample_video_url,
                "prompt": "soft rain on a tin roof, distant thunder",
                "negative_prompt": "",
                "seed": 1234,
                "num_inference_steps": 10,
                "guidance_scale": 4.5,
                "duration": None,
                "start_time": 0.0,
                "audio_mode": "replace",
                "sfx_gain_db": 0.0,
                "original_audio_gain_db": -6.0,
                "target_loudness_lufs": -14.0,
                "output_format": "mp4",
                "enable_safety_checker": True,
            },
            input_hash="seed",
            webhook_url=None,
            idempotency_key=None,
        )
        await PredictionRepository(session).set_source_asset(prediction_id, asset.id)
        await PredictionRepository(session).mark_queued(prediction_id)
    await enqueue_job(prediction_id, str(key.id))
    print(f"seeded prediction {prediction_id}")
    return plaintext


def main() -> int:
    """CLI entry.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-video",
        default="https://example.invalid/videos/sample.mp4",
    )
    args = parser.parse_args()
    key = asyncio.run(seed(args.sample_video))
    print(f"dev API key: {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
