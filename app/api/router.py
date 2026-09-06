"""Top-level API router assembling the v1 subrouters."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import account, health, models, predictions, uploads, webhooks

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(predictions.router)
api_router.include_router(uploads.router)
api_router.include_router(models.router)
api_router.include_router(webhooks.router)
api_router.include_router(account.router)
api_router.include_router(health.router)
