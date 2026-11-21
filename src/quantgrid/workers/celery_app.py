"""Celery application used purely as a message transport (see ADR 0001).

Messages carry only a ``task_id`` (plus trace headers). PostgreSQL decides whether the
receiving worker may run the task, so Celery's own retry/result machinery is unused:

* ``task_acks_late`` + ``task_reject_on_worker_lost``: a message is acknowledged only after
  the handler returns; a crashed worker's message is not silently lost.
* ``worker_prefetch_multiplier = 1``: a worker holds at most one unacknowledged message.
* No result backend: task outcomes live in PostgreSQL.
* A long Redis ``visibility_timeout`` (1 h) avoids duplicate redelivery of long tasks;
  recovery of crashed work never waits for it (the lease reconciler handles that).
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue  # type: ignore[attr-defined]

from quantgrid.config import Settings, get_settings

TASK_NAME = "quantgrid.run_task"


def make_celery(settings: Settings | None = None) -> Celery:
    s = settings or get_settings()
    app = Celery("quantgrid", broker=s.effective_broker_url, include=["quantgrid.workers.tasks"])
    app.conf.update(
        task_default_queue=s.queue_name,
        task_queues=[Queue(s.queue_name, routing_key=s.queue_name)],  # type: ignore[no-untyped-call]
        task_routes={TASK_NAME: {"queue": s.queue_name, "routing_key": s.queue_name}},
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_acks_on_failure_or_timeout=True,
        worker_prefetch_multiplier=1,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        result_backend=None,
        task_serializer="json",
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        broker_transport_options={"visibility_timeout": 3600},
        worker_hijack_root_logger=False,
        worker_redirect_stdouts=False,
        worker_send_task_events=False,
        enable_utc=True,
        timezone="UTC",
    )
    return app


celery_app = make_celery()
