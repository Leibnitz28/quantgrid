"""Celery task + worker lifecycle hooks. The task body only delegates to the runtime."""

from __future__ import annotations

from typing import Any

from celery.signals import worker_ready, worker_shutdown, worker_shutting_down

from quantgrid.workers.celery_app import TASK_NAME, celery_app
from quantgrid.workers.runtime import get_runtime


@celery_app.task(name=TASK_NAME, acks_late=True, ignore_result=True)
def run_task(task_id: str, trace: dict[str, str] | None = None) -> str:
    """Wake-up handler. Returns normally for expected outcomes (including "not claimable")
    so the message is acknowledged."""
    trace_id = None
    try:
        from quantgrid.observability.tracing import current_trace_id, span, use_context
    except ImportError:  # pragma: no cover
        return get_runtime().execute(task_id).outcome
    with use_context(trace or {}), span("worker.run_task", task_id=task_id):
        trace_id = current_trace_id()
        return get_runtime().execute(task_id, trace_id=trace_id).outcome


@worker_ready.connect
def _on_worker_ready(**_: Any) -> None:
    get_runtime().start()


@worker_shutting_down.connect
def _on_worker_shutting_down(**_: Any) -> None:
    get_runtime().begin_shutdown()


@worker_shutdown.connect
def _on_worker_shutdown(**_: Any) -> None:
    get_runtime().stop()
