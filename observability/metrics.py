from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "active_worker_leases",
    "queue_dispatch_lag_seconds",
    "record_execution_failure",
    "record_execution_success",
    "record_task_dead_lettered",
    "record_task_submission",
    "task_execution_duration_seconds",
    "task_executions_total",
    "task_submissions_total",
]

DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

DISPATCH_LAG_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.015,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    5.0,
    30.0,
)

task_submissions_total = Counter(
    "task_submissions_total",
    "Total number of task submissions accepted by the API, partitioned by task type.",
    ["task_type"],
)

task_executions_total = Counter(
    "task_executions_total",
    "Total number of task executions by outcome, partitioned by task type and status.",
    ["task_type", "status"],
)

task_execution_duration_seconds = Histogram(
    "task_execution_duration_seconds",
    "Wall-clock duration of task handler execution in seconds.",
    ["task_type"],
    buckets=DURATION_BUCKETS,
)

queue_dispatch_lag_seconds = Histogram(
    "queue_dispatch_lag_seconds",
    "Difference between pickup time and scheduled_at for dispatched tasks.",
    buckets=DISPATCH_LAG_BUCKETS,
)

active_worker_leases = Gauge(
    "active_worker_leases",
    "Number of task leases currently held by this worker process.",
)


def record_task_submission(task_type: str) -> None:
    task_submissions_total.labels(task_type=task_type).inc()


def record_execution_success(task_type: str, duration_seconds: float) -> None:
    task_executions_total.labels(task_type=task_type, status="success").inc()
    task_execution_duration_seconds.labels(task_type=task_type).observe(duration_seconds)


def record_execution_failure(task_type: str, duration_seconds: float) -> None:
    task_executions_total.labels(task_type=task_type, status="failure").inc()
    task_execution_duration_seconds.labels(task_type=task_type).observe(duration_seconds)


def record_task_dead_lettered(task_type: str, duration_seconds: float) -> None:
    task_executions_total.labels(task_type=task_type, status="dead_letter").inc()
    task_execution_duration_seconds.labels(task_type=task_type).observe(duration_seconds)
