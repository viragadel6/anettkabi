from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.middleware import IdempotencyMiddleware
from api.routes import router
from config import Settings, get_settings
from coordination.distributed_lock import close_redis_client, create_redis_client
from database import dispose_db, get_session_factory, init_db
from domain.exceptions import (
    DomainError,
    InvalidStateTransitionError,
    TaskNotFoundError,
    TaskOwnershipError,
)
from observability.tracing import configure_tracing, shutdown_tracing
from worker.outbox_relay import OutboxRelay

__all__ = ["create_app"]

logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_tracing(settings.OTEL_SERVICE_NAME_API, settings.ENVIRONMENT)
    init_db(settings)
    session_factory = get_session_factory()
    redis_client = create_redis_client(settings)

    relay = OutboxRelay(
        session_factory,
        redis_client,
        settings.OUTBOX_STREAM_NAME,
    )
    relay_task = asyncio.create_task(relay.run(), name="outbox-relay")

    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.redis_client = redis_client
    app.state.relay = relay
    app.state.relay_task = relay_task

    logger.info("API startup complete; outbox relay running")
    try:
        yield
    finally:
        relay.stop()
        try:
            await asyncio.wait_for(relay_task, timeout=10.0)
        except asyncio.TimeoutError:
            relay_task.cancel()
            try:
                await relay_task
            except (asyncio.CancelledError, Exception):
                pass
        await close_redis_client()
        await dispose_db()
        shutdown_tracing()
        logger.info("API shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is not None:
        get_settings.cache_clear()
    app = FastAPI(
        title="Task Orchestration Engine API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(IdempotencyMiddleware)
    app.include_router(router, tags=["tasks"])

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    FastAPIInstrumentor.instrument_app(app)

    @app.exception_handler(InvalidStateTransitionError)
    async def invalid_state_transition_handler(
        request: Request, exc: InvalidStateTransitionError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(TaskNotFoundError)
    async def task_not_found_handler(
        request: Request, exc: TaskNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(TaskOwnershipError)
    async def task_ownership_handler(
        request: Request, exc: TaskOwnershipError
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
