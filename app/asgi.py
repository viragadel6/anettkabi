"""ASGI entry module for `uvicorn app.asgi:app`."""

from __future__ import annotations

from app.main import create_app

__all__ = ["app"]

app = create_app()
