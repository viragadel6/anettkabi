"""FastAPI application factory: middleware, routers, static UI, lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import SERVICE_NAME, __version__
from app.api.router import api_router
from app.config import get_settings
from app.constants import API_PREFIX
from app.logging_config import configure_logging
from app.middleware.access_log import AccessLogMiddleware
from app.middleware.auth import AuthMiddleware
from app.middleware.body_limit import BodyLimitMiddleware
from app.middleware.error_handler import register_exception_handlers
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIdMiddleware
from app.services.queue import close_redis_clients, ensure_stream

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """Build the configured FastAPI application.

    Returns:
        The application instance.

    Raises:
        ConfigError: Propagated from settings validation on first access.
    """
    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await ensure_stream(settings.redis.queue_stream_name, settings.redis.queue_group)
        yield
        await close_redis_clients()

    app = FastAPI(
        title="Video-to-Video SFX API",
        description=(
            "Generative sound-design service: video + optional prompt in, "
            "time-aligned sound effects muxed into an MP4 out."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        root_path=settings.server.root_path or "",
        lifespan=lifespan,
    )
    app.include_router(api_router, prefix=API_PREFIX)
    register_exception_handlers(app)
    if settings.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-Request-Id",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
                "Retry-After",
            ],
        )
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthMiddleware)
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIdMiddleware)
    _mount_web_ui(app)
    return app


def _mount_web_ui(app: FastAPI) -> None:
    """Serve the built web UI when its assets exist.

    Parameters:
        app: The application.
    """
    static_dir = Path(__file__).resolve().parent.parent / "web" / "dist"
    if (static_dir / "index.html").is_file() and (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))
    else:

        @app.get("/", include_in_schema=False)
        async def index_unbuilt() -> JSONResponse:
            return JSONResponse(
                {
                    "service": SERVICE_NAME,
                    "version": __version__,
                    "docs": "/docs",
                    "web_ui": "not built; run `make web` or `make docker-build`",
                }
            )
