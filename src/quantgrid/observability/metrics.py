"""Prometheus metrics with bounded label sets (never task_id / batch_id labels).

``create_metrics(registry)`` builds a fresh set of collectors; the module-level
singletons below are created once on the default registry, so importing the module
twice (or reloading it) never raises ``Duplicated timeseries``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

_HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_TASK_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
_QUEUE_BUCKETS = (0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 900)
_TICK_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


def create_metrics(registry: CollectorRegistry) -> SimpleNamespace:
    """Create every QuantGrid collector on ``registry`` and return them by name."""

    def counter(name: str, doc: str, labels: tuple[str, ...] = ()) -> Counter:
        return Counter(f"quantgrid_{name}", doc, labels, registry=registry)

    def gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
        return Gauge(f"quantgrid_{name}", doc, labels, registry=registry)

    def hist(
        name: str, doc: str, buckets: tuple[float, ...], labels: tuple[str, ...] = ()
    ) -> Histogram:
        return Histogram(f"quantgrid_{name}", doc, labels, buckets=buckets, registry=registry)

    return SimpleNamespace(
        http_requests_total=counter(
            "http_requests_total", "HTTP requests.", ("method", "route", "status")
        ),
        http_request_duration_seconds=hist(
            "http_request_duration_seconds",
            "HTTP request latency.",
            _HTTP_BUCKETS,
            ("method", "route"),
        ),
        batches_submitted_total=counter(
            "batches_submitted_total", "Batch submissions by outcome.", ("outcome",)
        ),
        planner_requests_total=counter("planner_requests_total", "Planner requests.", ("status",)),
        tasks_dispatched_total=counter("tasks_dispatched_total", "Tasks dispatched to the queue."),
        dispatch_redelivered_total=counter(
            "dispatch_redelivered_total", "Tasks re-dispatched after a missing claim."
        ),
        leases_reclaimed_total=counter(
            "leases_reclaimed_total", "Expired leases reclaimed.", ("outcome",)
        ),
        tasks_by_state=gauge("tasks_by_state", "Tasks per state.", ("state",)),
        live_workers=gauge("live_workers", "Workers with a fresh heartbeat."),
        coordinator_tick_seconds=hist(
            "coordinator_tick_seconds", "Coordinator tick duration.", _TICK_BUCKETS
        ),
        task_attempts_total=counter(
            "task_attempts_total", "Finished task attempts.", ("status", "error_code")
        ),
        task_duration_seconds=hist(
            "task_duration_seconds", "Task wall time.", _TASK_BUCKETS, ("outcome",)
        ),
        queue_wait_seconds=hist(
            "queue_wait_seconds", "Time from enqueue to claim.", _QUEUE_BUCKETS
        ),
        artifact_bytes_written_total=counter(
            "artifact_bytes_written_total", "Artifact bytes written."
        ),
        artifact_commits_total=counter(
            "artifact_commits_total", "Artifact commits (new|duplicate).", ("outcome",)
        ),
        heartbeat_failures_total=counter("heartbeat_failures_total", "Failed heartbeats."),
        lease_lost_total=counter("lease_lost_total", "Leases lost while running."),
        late_completions_total=counter(
            "late_completions_total", "Completions after lease loss.", ("checksum",)
        ),
        cache_hits_total=counter("cache_hits_total", "Result cache hits."),
        task_claims_total=counter("task_claims_total", "Claim attempts.", ("outcome",)),
    )


# Reuse collectors if this module is imported/reloaded again in the same process.
_ns: SimpleNamespace = getattr(REGISTRY, "_quantgrid_metrics", None) or create_metrics(REGISTRY)
REGISTRY._quantgrid_metrics = _ns  # type: ignore[attr-defined]
http_requests_total: Counter = _ns.http_requests_total
http_request_duration_seconds: Histogram = _ns.http_request_duration_seconds
batches_submitted_total: Counter = _ns.batches_submitted_total
planner_requests_total: Counter = _ns.planner_requests_total
tasks_dispatched_total: Counter = _ns.tasks_dispatched_total
dispatch_redelivered_total: Counter = _ns.dispatch_redelivered_total
leases_reclaimed_total: Counter = _ns.leases_reclaimed_total
tasks_by_state: Gauge = _ns.tasks_by_state
live_workers: Gauge = _ns.live_workers
coordinator_tick_seconds: Histogram = _ns.coordinator_tick_seconds
task_attempts_total: Counter = _ns.task_attempts_total
task_duration_seconds: Histogram = _ns.task_duration_seconds
queue_wait_seconds: Histogram = _ns.queue_wait_seconds
artifact_bytes_written_total: Counter = _ns.artifact_bytes_written_total
artifact_commits_total: Counter = _ns.artifact_commits_total
heartbeat_failures_total: Counter = _ns.heartbeat_failures_total
lease_lost_total: Counter = _ns.lease_lost_total
late_completions_total: Counter = _ns.late_completions_total
cache_hits_total: Counter = _ns.cache_hits_total
task_claims_total: Counter = _ns.task_claims_total


def start_metrics_server(port: int) -> None:
    """Serve the default registry on ``port``; a no-op for port 0 (disabled)."""
    if port:
        start_http_server(port)


def render_latest(registry: Any = None) -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for a ``/metrics`` response."""
    return generate_latest(registry or REGISTRY), CONTENT_TYPE_LATEST
