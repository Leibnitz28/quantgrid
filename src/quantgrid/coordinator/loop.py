"""Coordinator: lease reconciliation, bounded dispatch and batch-state refresh.

Exactly one coordinator is active at a time (``pg_try_advisory_lock`` held on a dedicated
autocommit connection; if the process dies the session ends and the lock is released, so a
standby takes over). The leader wakes on PostgreSQL ``NOTIFY quantgrid_events`` (batch
submitted, task finished, lease reclaimed) or every ``QG_COORDINATOR_TICK_S`` and runs::

    reconcile_expired_leases  -> RUNNING tasks with expired leases are re-queued/failed
    mark_dead_workers         -> stale heartbeat rows marked DEAD (capacity bookkeeping)
    dispatch_ready            -> top up the bounded window of wake-up messages
    refresh_batch_states      -> derive batch state from task counts

Every step is idempotent and safe to repeat; a crash between steps loses nothing.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import psycopg
from sqlalchemy import Engine, make_url, text

from quantgrid.config import Settings
from quantgrid.storage import repository as repo
from quantgrid.storage.db import NOTIFY_CHANNEL
from quantgrid.storage.repository import LEADER_LOCK_KEY, DispatchItem

log = logging.getLogger("quantgrid.coordinator")

Sender = Callable[[DispatchItem], None]


@dataclass
class TickReport:
    reclaimed: int = 0
    dead_workers: list[str] = field(default_factory=list)
    live_workers: int = 0
    window: int = 0
    dispatched: int = 0
    redelivered: int = 0
    send_failures: int = 0
    batch_changes: int = 0
    seconds: float = 0.0


def celery_sender(queue: str) -> Sender:
    from quantgrid.workers.celery_app import TASK_NAME, celery_app

    def send(item: DispatchItem) -> None:
        trace: dict[str, str] = {}
        try:
            from quantgrid.observability.tracing import inject_context

            trace = inject_context()
        except ImportError:  # pragma: no cover
            pass
        celery_app.send_task(
            TASK_NAME,
            args=[str(item.task_id)],
            kwargs={"trace": trace},
            queue=queue,
            routing_key=queue,
            retry=True,
            retry_policy={"max_retries": 3, "interval_start": 0.1, "interval_step": 0.5},
        )

    return send


def libpq_dsn(sqlalchemy_url: str) -> str:
    return (
        make_url(sqlalchemy_url).set(drivername="postgresql").render_as_string(hide_password=False)
    )


class Coordinator:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        sender: Sender,
        *,
        database_url: str | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.sender = sender
        self.database_url = database_url or settings.database_url
        self.stop_event = threading.Event()
        self.ticks = 0
        self.is_leader = False

    def tick(self) -> TickReport:
        t0 = time.monotonic()
        s = self.settings
        report = TickReport()
        reclaimed = repo.reconcile_expired_leases(self.engine, limit=s.reconcile_batch_limit)
        report.reclaimed = len(reclaimed)
        for r in reclaimed:
            _inc("leases_reclaimed_total", outcome=r.outcome.value)
            log.warning(
                "lease expired; task reclaimed",
                extra={
                    "task_id": str(r.task_id),
                    "batch_id": str(r.batch_id),
                    "attempt": r.attempt,
                    "worker_id": r.worker_id,
                    "outcome": r.outcome.value,
                    "expired_for_ms": r.expired_for_ms,
                },
            )
        report.dead_workers = repo.mark_dead_workers(
            self.engine, dead_after_s=s.worker_dead_after_s
        )
        for wid in report.dead_workers:
            log.warning("worker presumed dead", extra={"worker_id": wid})
        live, slots = repo.live_worker_capacity(self.engine, dead_after_s=s.worker_dead_after_s)
        report.live_workers = live
        report.window = max(s.dispatch_min_window, math.ceil(slots * s.dispatch_window_factor))
        items = repo.dispatch_ready(
            self.engine, window=report.window, lost_after_s=s.dispatch_lost_after_s
        )
        failed: list[DispatchItem] = []
        for item in items:
            try:
                self.sender(item)
            except Exception:
                failed.append(item)
                log.exception("failed to send wake-up", extra={"task_id": str(item.task_id)})
        if failed:
            self._clear_enqueued([f.task_id for f in failed])
        report.dispatched = len(items) - len(failed)
        report.send_failures = len(failed)
        report.redelivered = sum(1 for i in items if i.redelivery)
        _inc_by("tasks_dispatched_total", report.dispatched)
        _inc_by("dispatch_redelivered_total", report.redelivered)
        report.batch_changes = len(repo.refresh_batch_states(self.engine))
        report.seconds = time.monotonic() - t0
        self._update_gauges(live)
        _observe("coordinator_tick_seconds", report.seconds)
        self.ticks += 1
        return report

    def _clear_enqueued(self, task_ids: Sequence[object]) -> None:
        """Broker unavailable: un-mark so the next tick retries immediately."""
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE tasks SET enqueued_at = NULL WHERE id = ANY(CAST(:ids AS uuid[])) "
                    "AND state IN ('PENDING', 'RETRY_PENDING')"
                ),
                {"ids": list(task_ids)},
            )

    def _update_gauges(self, live: int) -> None:
        try:
            from quantgrid.observability import metrics
        except ImportError:  # pragma: no cover
            return
        gauge = getattr(metrics, "tasks_by_state", None)
        live_gauge = getattr(metrics, "live_workers", None)
        if live_gauge is not None:
            live_gauge.set(live)
        if gauge is None:
            return
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT t.state, count(*) AS n FROM tasks t "
                    "JOIN batches b ON b.id = t.batch_id "
                    "WHERE b.state IN ('ACCEPTED', 'RUNNING', 'CANCELING') GROUP BY t.state"
                )
            ).all()
        counts = {r.state: int(r.n) for r in rows}
        for state in ("PENDING", "RUNNING", "RETRY_PENDING", "SUCCEEDED", "FAILED", "CANCELED"):
            gauge.labels(state=state).set(counts.get(state, 0))

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        """Run until :meth:`stop`; re-elects after database errors."""
        while not self.stop_event.is_set():
            try:
                self._run_as_candidate()
            except (psycopg.Error, OSError) as exc:
                self.is_leader = False
                log.error(
                    "coordinator lost database connection; re-electing",
                    extra={"error": f"{type(exc).__name__}: {exc}"},
                )
                self.stop_event.wait(1.0)
            except Exception:
                self.is_leader = False
                log.exception("coordinator tick failed; retrying")
                self.stop_event.wait(1.0)

    def _run_as_candidate(self) -> None:
        dsn = libpq_dsn(self.database_url)
        with psycopg.connect(dsn, autocommit=True, application_name="qg-coordinator") as conn:
            while not self.stop_event.is_set():
                got = conn.execute("SELECT pg_try_advisory_lock(%s)", (LEADER_LOCK_KEY,)).fetchone()
                if got and got[0]:
                    break
                if not self.is_leader:
                    log.info("standby: another coordinator holds the leader lock")
                self.stop_event.wait(2.0)
            if self.stop_event.is_set():
                return
            self.is_leader = True
            log.info("elected coordinator leader")
            conn.execute(f"LISTEN {NOTIFY_CHANNEL}")
            try:
                while not self.stop_event.is_set():
                    report = self.tick()
                    if report.reclaimed or report.dispatched or report.batch_changes:
                        log.debug("tick", extra=report.__dict__)
                    self._wait_for_events(conn)
            finally:
                if not conn.closed:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (LEADER_LOCK_KEY,))
                self.is_leader = False

    def _wait_for_events(self, conn: psycopg.Connection[tuple[object, ...]]) -> None:
        """Block until a NOTIFY arrives or the tick interval elapses, then drain."""
        deadline = time.monotonic() + self.settings.coordinator_tick_s
        for _ in conn.notifies(timeout=self.settings.coordinator_tick_s, stop_after=1):
            break
        # Coalesce bursts into a single tick.
        remaining = max(0.0, min(0.01, deadline - time.monotonic()))
        for _ in conn.notifies(timeout=remaining):
            pass


def _metric(name: str) -> object | None:
    try:
        from quantgrid.observability import metrics
    except ImportError:  # pragma: no cover
        return None
    return getattr(metrics, name, None)


def _inc(name: str, **labels: str) -> None:
    metric = _metric(name)
    if metric is not None:
        (metric.labels(**labels) if labels else metric).inc()  # type: ignore[attr-defined,union-attr]


def _inc_by(name: str, amount: float) -> None:
    metric = _metric(name)
    if metric is not None and amount:
        metric.inc(amount)  # type: ignore[attr-defined]


def _observe(name: str, value: float) -> None:
    metric = _metric(name)
    if metric is not None:
        metric.observe(value)  # type: ignore[attr-defined]
