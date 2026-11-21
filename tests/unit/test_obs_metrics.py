"""Observability metrics registry (separate from the domain-metrics tests)."""

from __future__ import annotations

import importlib

import pytest
from prometheus_client import CollectorRegistry

from quantgrid.observability import metrics

NAMES = [
    "http_requests_total", "http_request_duration_seconds", "batches_submitted_total",
    "planner_requests_total", "tasks_dispatched_total", "dispatch_redelivered_total",
    "leases_reclaimed_total", "tasks_by_state", "live_workers", "coordinator_tick_seconds",
    "task_attempts_total", "task_duration_seconds", "queue_wait_seconds",
    "artifact_bytes_written_total", "artifact_commits_total", "heartbeat_failures_total",
    "lease_lost_total", "late_completions_total", "cache_hits_total", "task_claims_total",
]  # fmt: skip


def test_all_module_level_metrics_exist() -> None:
    for n in NAMES:
        assert hasattr(metrics, n), n


def test_factory_uses_isolated_registry_and_prefix() -> None:
    reg = CollectorRegistry()
    ns = metrics.create_metrics(reg)
    ns.http_requests_total.labels(method="GET", route="/x", status="200").inc()
    ns.live_workers.set(3)
    ns.task_duration_seconds.labels(outcome="succeeded").observe(0.5)
    body, ctype = metrics.render_latest(reg)
    text = body.decode()
    assert 'quantgrid_http_requests_total{method="GET",route="/x",status="200"} 1.0' in text
    assert "quantgrid_live_workers 3.0" in text
    assert "quantgrid_task_duration_seconds_bucket" in text
    assert ctype.startswith("text/plain")
    # a second registry is independent
    assert (
        "quantgrid_live_workers 3.0" not in metrics.render_latest(CollectorRegistry())[0].decode()
    )


def test_label_sets_are_bounded() -> None:
    ns = metrics.create_metrics(CollectorRegistry())
    for n in NAMES:
        labels = set(getattr(ns, n)._labelnames)
        assert not labels & {"task_id", "batch_id", "worker_id"}, n
    assert set(ns.task_attempts_total._labelnames) == {"status", "error_code"}
    with pytest.raises(ValueError, match="label"):
        ns.artifact_commits_total.labels(wrong="x")


def test_task_duration_buckets_cover_10ms_to_10min() -> None:
    b = metrics.task_duration_seconds._upper_bounds
    assert b[0] <= 0.01 and max(x for x in b if x != float("inf")) >= 600


def test_default_registry_render_and_reimport_safe() -> None:
    metrics.cache_hits_total.inc()
    importlib.reload(metrics)  # must not raise Duplicated timeseries
    body, _ = metrics.render_latest()
    assert b"quantgrid_cache_hits_total" in body


def test_start_metrics_server_zero_is_noop() -> None:
    metrics.start_metrics_server(0)
