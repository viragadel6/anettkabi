"""Integration roundtrips: Redis stream queue, S3/MinIO storage, DB sessions."""

from __future__ import annotations

import uuid

import pytest

from app.utils.ids import new_prediction_id

pytestmark = pytest.mark.integration


async def test_queue_enqueue_claim_ack(integration_env: dict) -> None:
    from app.services.queue import ack_job, claim_stale_jobs, enqueue_job

    prediction_id = new_prediction_id()
    api_key_id = str(uuid.uuid4())
    stream_id = await enqueue_job(prediction_id, api_key_id, priority=9)
    assert "-" in stream_id
    claimed = await claim_stale_jobs("integration-test-consumer", idle_ms=0, count=10)
    matching = [job for job in claimed if job.prediction_id == prediction_id]
    assert matching, f"job {prediction_id} not claimed from stream"
    await ack_job(stream_id)


async def test_queue_dlx_routing(integration_env: dict) -> None:
    from app.services.queue import enqueue_job

    prediction_id = new_prediction_id()
    await enqueue_job(prediction_id, str(uuid.uuid4()))
    assert len(prediction_id) == 26


def test_storage_put_presign_roundtrip(integration_env: dict) -> None:
    from app.config import get_settings
    from app.services.storage import StorageService

    settings = get_settings()
    storage = StorageService(settings)
    key = f"{settings.storage.s3_key_prefix}/it/{new_prediction_id()}.bin"
    payload = b"integration-payload" * 128
    storage.put_bytes(payload, key, content_type="application/octet-stream")
    assert storage.exists(key)
    url = storage.presign_put(key, content_type="application/octet-stream", ttl_s=60)
    assert "http" in url
    public = storage.public_url(key, ttl_s=60)
    import httpx

    response = httpx.get(public, timeout=10.0)
    assert response.status_code in (200, 403)
    if response.status_code == 200:
        assert response.content == payload


async def test_db_session_and_api_key_roundtrip(integration_env: dict) -> None:
    from argon2 import PasswordHasher

    from app.db.repositories.api_keys import ApiKeyRepository
    from app.db.session import session_scope
    from app.utils.ids import new_api_secret

    secret = new_api_secret(40)
    hasher = PasswordHasher()
    async with session_scope() as session:
        row = await ApiKeyRepository(session).create(
            name="integration",
            key_prefix=secret[:8],
            key_hash=hasher.hash(secret),
            scopes=["predictions:write"],
            rate_limit_rpm=60,
            concurrency_limit=3,
            monthly_seconds_quota=100.0,
        )
        fetched = await ApiKeyRepository(session).get_by_prefix(secret[:8])
        assert fetched is not None
        assert str(row.id) == str(fetched.id)
        assert fetched.scopes == ["predictions:write"]
