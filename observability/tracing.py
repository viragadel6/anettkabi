from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Span, StatusCode, Tracer

__all__ = [
    "configure_tracing",
    "database_transaction_span",
    "get_tracer",
    "outbox_dispatch_span",
    "record_span_error",
    "task_execution_span",
]

TRACER_NAME = "task_engine"


def configure_tracing(service_name: str, environment: str = "production") -> TracerProvider:
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "deployment.environment": environment,
        }
    )
    provider = TracerProvider(resource=resource)
    if os.environ.get("OTEL_TRACES_EXPORTER", "console").lower() != "none":
        provider.add_span_processor(
            BatchSpanProcessor(
                ConsoleSpanExporter(out=sys.stderr),
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=2000,
            )
        )
    trace.set_tracer_provider(provider)
    return provider


def get_tracer() -> Tracer:
    return trace.get_tracer(TRACER_NAME)


@contextlib.contextmanager
def task_execution_span(
    task_id: str,
    task_type: str,
    retry_count: int,
    worker_id: str,
) -> Iterator[Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span("task.execute") as span:
        span.set_attribute("task.id", task_id)
        span.set_attribute("task.type", task_type)
        span.set_attribute("task.retry_count", retry_count)
        span.set_attribute("worker.id", worker_id)
        yield span


@contextlib.contextmanager
def database_transaction_span(operation: str) -> Iterator[Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span("db.transaction") as span:
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.operation", operation)
        yield span


@contextlib.contextmanager
def outbox_dispatch_span(event_count: int) -> Iterator[Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span("outbox.dispatch") as span:
        span.set_attribute("messaging.system", "redis-streams")
        span.set_attribute("messaging.batch.message_count", event_count)
        yield span


def record_span_error(span: Span, exception: BaseException) -> None:
    span.set_status(StatusCode.ERROR, str(exception))
    span.record_exception(exception)


def shutdown_tracing() -> None:
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush(timeout_millis=5000)
        provider.shutdown()


def ensure_export_result(result: SpanExportResult) -> bool:
    return result == SpanExportResult.SUCCESS
