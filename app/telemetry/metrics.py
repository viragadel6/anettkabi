"""Prometheus metrics registry and helpers (counters, gauges, histograms)."""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.core import REGISTRY

from app.config import get_settings

__all__ = [
    "AUDIO_VIDEO_DURATION_DELTA_MS",
    "GPU_MEMORY_BYTES",
    "INFERENCE_SECONDS",
    "METRICS_REGISTRY",
    "PREDICTIONS_TOTAL",
    "QUEUE_DEPTH",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION_SECONDS",
    "STAGE_DURATION_SECONDS",
    "WEBHOOK_DELIVERIES_TOTAL",
    "WEIGHTS_LOADED",
    "render_metrics",
]

METRICS_REGISTRY: CollectorRegistry = REGISTRY

REQUESTS_TOTAL = Counter(
    "vsfx_requests_total",
    "HTTP requests processed",
    ["route", "status"],
    registry=METRICS_REGISTRY,
)
REQUEST_DURATION_SECONDS = Histogram(
    "vsfx_request_duration_seconds",
    "HTTP request duration",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    registry=METRICS_REGISTRY,
)
PREDICTIONS_TOTAL = Counter(
    "vsfx_predictions_total",
    "Predictions by terminal/intermediate status",
    ["status"],
    registry=METRICS_REGISTRY,
)
STAGE_DURATION_SECONDS = Histogram(
    "vsfx_stage_duration_seconds",
    "Worker stage durations",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 15, 30, 60, 120, 300, 900),
    registry=METRICS_REGISTRY,
)
QUEUE_DEPTH = Gauge(
    "vsfx_queue_depth",
    "Pending messages in the job stream",
    registry=METRICS_REGISTRY,
)
INFERENCE_SECONDS = Histogram(
    "vsfx_inference_seconds",
    "End-to-end inference (generation) wall time",
    registry=METRICS_REGISTRY,
    buckets=(0.5, 1, 2, 5, 10, 20, 45, 90, 180, 420, 900),
)
GPU_MEMORY_BYTES = Gauge(
    "vsfx_gpu_memory_bytes",
    "GPU memory allocated in bytes",
    ["device"],
    registry=METRICS_REGISTRY,
)
AUDIO_VIDEO_DURATION_DELTA_MS = Gauge(
    "vsfx_audio_video_duration_delta_ms",
    "Absolute audio/video duration mismatch of final MP4s",
    registry=METRICS_REGISTRY,
)
WEBHOOK_DELIVERIES_TOTAL = Counter(
    "vsfx_webhook_deliveries_total",
    "Webhook delivery attempts",
    ["result"],
    registry=METRICS_REGISTRY,
)
WEIGHTS_LOADED = Gauge(
    "vsfx_weights_loaded",
    "Number of weight artifacts successfully loaded (1 per artifact)",
    registry=METRICS_REGISTRY,
)


def render_metrics() -> bytes:
    """Render the Prometheus exposition format.

    Returns:
        Bytes of the text exposition payload.
    """
    settings = get_settings()
    if settings.telemetry.metrics_enabled:
        return generate_latest(METRICS_REGISTRY)
    return b""


def observe_request(route: str, status: int, duration_s: float) -> None:
    """Record one HTTP request.

    Parameters:
        route: Route template or raw path.
        status: Response status code.
        duration_s: Handler duration in seconds.
    """
    REQUESTS_TOTAL.labels(route=route, status=str(status)).inc()
    REQUEST_DURATION_SECONDS.labels(route=route).observe(duration_s)


def set_gpu_memory(device_index: int, allocated_bytes: int) -> None:
    """Publish current GPU memory usage.

    Parameters:
        device_index: CUDA device ordinal.
        allocated_bytes: torch.cuda allocated bytes.
    """
    GPU_MEMORY_BYTES.labels(device=f"cuda:{device_index}").set(allocated_bytes)


def collect_multiprocess(directory: str | None = None) -> bytes:
    """Aggregate metrics from worker processes when running multiprocess mode.

    Parameters:
        directory: prometheus_multiproc_dir; defaults to the env variable.

    Returns:
        Aggregated exposition bytes.
    """
    import os

    target = directory or os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not target:
        return b""
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry)
