"""Webhook test endpoint: signed probe delivery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.db.repositories.webhooks import WebhookRepository
from app.dependencies import ApiKeyId, SessionDep
from app.errors import ErrorCode, ServiceError
from app.services.webhook_sender import WebhookSender
from app.utils.validators import validate_webhook_url

__all__ = ["WebhookTestRequest", "router"]

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookTestRequest(BaseModel):
    """Request for POST /webhooks/test.

    Attributes:
        url: https target for the probe.
    """

    url: str = Field(min_length=8, max_length=2048)


@router.post("/test")
async def test_webhook(
    api_key_id: ApiKeyId,
    session: SessionDep,
    request: WebhookTestRequest,
) -> dict[str, Any]:
    """Send a signed test payload and return the delivery record.

    Returns:
        Envelope with the persisted delivery attempt.

    Raises:
        ServiceError: invalid_request for unsafe URLs.
    """
    from app.config import get_settings

    settings = get_settings()
    url = validate_webhook_url(
        request.url, block_private_cidrs=settings.media.download_block_private_cidrs
    )
    sender = WebhookSender()
    payload = (
        '{"code":200,"message":"webhook test","data":{"event":"test","api_key_scope":"test"}}'
    )
    accepted = await sender.deliver("00000000000000000000000000", url, payload)
    deliveries = await WebhookRepository(session).for_prediction("00000000000000000000000000")
    latest = deliveries[-1] if deliveries else None
    if latest is None:
        raise ServiceError(
            ErrorCode.INTERNAL_ERROR,
            "delivery attempt was not recorded; check webhook configuration",
        )
    data = {
        "delivered": accepted,
        "attempt": latest.attempt,
        "status_code": latest.status_code,
        "response_snippet": latest.response_snippet[:256],
        "signature_header_sent": latest.signature,
        "error": latest.error,
    }
    return {"code": 200, "message": "success" if accepted else "delivery_failed", "data": data}
