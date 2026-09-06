"""Signed webhook delivery with retries and persisted attempts."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC

import httpx
import structlog

from app.config import get_settings
from app.constants import HTTP_USER_AGENT
from app.db.repositories.predictions import PredictionRepository
from app.db.repositories.webhooks import WebhookRepository
from app.db.session import session_scope
from app.telemetry.metrics import WEBHOOK_DELIVERIES_TOTAL
from app.utils.hashing import hmac_signature
from app.utils.retry import backoff_delay_s
from app.utils.validators import validate_webhook_url

__all__ = ["WebhookSender", "sign_payload"]

_logger = structlog.get_logger("vsfx.webhooks")


def sign_payload(secret: str, timestamp: str, body: str) -> str:
    """Produce the X-Webhook-Signature header value.

    Parameters:
        secret: Signing secret.
        timestamp: X-Webhook-Timestamp value.
        body: Raw JSON body string.

    Returns:
        `sha256=<hex>` signature string.
    """
    return f"sha256={hmac_signature(secret, timestamp, body)}"


class WebhookSender:
    """Delivers prediction envelopes to client URLs with HMAC signing."""

    __slots__ = ("_client", "_settings")

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Bind settings and an optional injectable client.

        Parameters:
            client: Pre-configured AsyncClient (tests).
        """
        self._settings = get_settings()
        self._client = client

    async def deliver_for_prediction(self, prediction_id: str) -> bool:
        """Load the prediction and deliver its envelope to its webhook.

        Parameters:
            prediction_id: Prediction id.

        Returns:
            True when a delivery was attempted and ultimately accepted.
        """
        async with session_scope() as session:
            row = await PredictionRepository(session).get(prediction_id)
            if row is None or not row.webhook_url:
                return False
            from app.api.schemas.prediction import render_prediction_payload

            payload = render_prediction_payload(row, get_settings())
        import json

        body = json.dumps(
            {
                "code": 200 if row.status.value == "completed" else 422,
                "message": "prediction " + row.status.value,
                "data": payload,
            },
            separators=(",", ":"),
        )
        return await self.deliver(row.id, row.webhook_url, body)

    async def deliver(self, prediction_id: str, url: str, body: str) -> bool:
        """POST the signed payload with retries and persistence.

        Parameters:
            prediction_id: Related prediction.
            url: Target URL.
            body: Raw JSON body.

        Returns:
            True on ultimate 2xx acceptance.

        Raises:
            ServiceError: invalid_request for non-https or SSRF-blocked URLs.
        """
        settings = self._settings
        if not settings.webhook.webhook_enabled:
            return False
        url = validate_webhook_url(
            url, block_private_cidrs=settings.media.download_block_private_cidrs
        )
        secret = settings.webhook.webhook_signing_secret
        max_attempts = settings.webhook.webhook_max_attempts
        client = self._client or httpx.AsyncClient(
            timeout=settings.webhook.webhook_timeout_s,
            headers={"user-agent": HTTP_USER_AGENT},
        )
        owns_client = self._client is None
        delivery_id = uuid.uuid4()
        accepted = False
        try:
            for attempt in range(1, max_attempts + 1):
                timestamp = str(int(time.time()))
                headers = {
                    "content-type": "application/json",
                    "x-webhook-id": str(delivery_id),
                    "x-webhook-timestamp": timestamp,
                }
                if secret:
                    headers["x-webhook-signature"] = sign_payload(secret, timestamp, body)
                status_code: int | None = None
                snippet = ""
                error = ""
                try:
                    response = await client.post(url, content=body, headers=headers)
                    status_code = response.status_code
                    snippet = response.text[:512]
                    if 200 <= status_code < 300:
                        accepted = True
                        WEBHOOK_DELIVERIES_TOTAL.labels(result="success").inc()
                    else:
                        error = f"HTTP {status_code}"
                        WEBHOOK_DELIVERIES_TOTAL.labels(result=f"status_{status_code}").inc()
                except (httpx.HTTPError, OSError) as exc:
                    error = str(exc)[:512]
                    WEBHOOK_DELIVERIES_TOTAL.labels(result="error").inc()
                await self._record(
                    prediction_id,
                    url,
                    attempt,
                    headers.get("x-webhook-signature", ""),
                    status_code,
                    snippet,
                    error,
                    accepted,
                )
                if accepted:
                    break
                if attempt < max_attempts:
                    await asyncio.sleep(
                        backoff_delay_s(attempt - 1, 1.0, 30.0)
                    )
        finally:
            if owns_client:
                await client.aclose()
        if not accepted:
            _logger.warning(
                "webhook_delivery_exhausted",
                prediction_id=prediction_id,
                url=url,
                attempts=max_attempts,
            )
        return accepted

    async def _record(
        self,
        prediction_id: str,
        url: str,
        attempt: int,
        signature: str,
        status_code: int | None,
        snippet: str,
        error: str,
        accepted: bool,
    ) -> None:
        """Persist one delivery attempt.

        Parameters:
            prediction_id: Related prediction.
            url: Target URL.
            attempt: Attempt number.
            signature: Signature header sent.
            status_code: Response status or None.
            snippet: Response body snippet.
            error: Error text.
            accepted: Whether this attempt was accepted.
        """
        from datetime import datetime

        try:
            async with session_scope() as session:
                await WebhookRepository(session).record(
                    prediction_id=prediction_id,
                    url=url,
                    attempt=attempt,
                    signature=signature,
                    status_code=status_code,
                    response_snippet=snippet,
                    error=error,
                    delivered_at=datetime.now(tz=UTC) if accepted else None,
                )
        except Exception as exc:
            _logger.debug("webhook_record_failed", error=str(exc))
