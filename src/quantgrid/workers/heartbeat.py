"""Worker heartbeat thread and in-flight task registry.

One daemon thread per worker process: every ``QG_HEARTBEAT_INTERVAL_S`` it upserts the
``worker_heartbeats`` row and extends the leases of the tasks this process is running
(fenced by attempt number). It relays two signals to the computing thread through each
task's cancel token:

* lease ``LOST``  -> ``TaskAborted("LEASE_LOST")`` - stop and never publish;
* ``CANCEL_REQUESTED`` -> ``TaskAborted("CANCELED")`` - stop and acknowledge the cancel.

Hard-timeout watchdog: a task still running ``max_runtime_s + lease_ttl_s`` after it
started (i.e. its cooperative deadline check is not being reached) is no longer extended,
so its lease expires and the coordinator re-queues or fails it even if this process hangs.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import psutil
from sqlalchemy import Engine

from quantgrid.config import Settings
from quantgrid.domain.types import DeadlineCancelToken
from quantgrid.storage import repository as repo
from quantgrid.storage.repository import LeaseStatus, WorkerInfo

log = logging.getLogger("quantgrid.worker.heartbeat")


@dataclass
class InflightTask:
    task_id: uuid.UUID
    attempt: int
    max_runtime_s: float
    token: DeadlineCancelToken
    started_mono: float = field(default_factory=time.monotonic)
    peak_rss_mb: float = 0.0

    def hard_deadline_passed(self, grace_s: float) -> bool:
        return time.monotonic() - self.started_mono > self.max_runtime_s + grace_s


class InflightRegistry:
    """Thread-safe map of the tasks this process is currently executing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[uuid.UUID, InflightTask] = {}

    def add(self, task: InflightTask) -> None:
        with self._lock:
            self._tasks[task.task_id] = task

    def remove(self, task_id: uuid.UUID) -> InflightTask | None:
        with self._lock:
            return self._tasks.pop(task_id, None)

    def snapshot(self) -> list[InflightTask]:
        with self._lock:
            return list(self._tasks.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)


def resource_profile(proc: psutil.Process) -> dict[str, Any]:
    mem = proc.memory_info()
    try:
        load = os.getloadavg()
    except OSError:  # pragma: no cover - not available on all platforms
        load = (0.0, 0.0, 0.0)
    return {
        "hostname": socket.gethostname(),
        "pid": proc.pid,
        "cpu_count": os.cpu_count(),
        "rss_mb": round(mem.rss / 1e6, 1),
        "cpu_times_user_s": round(proc.cpu_times().user, 3),
        "load_avg": [round(x, 2) for x in load],
        "python": platform.python_version(),
        "pool": "solo",
    }


class HeartbeatThread(threading.Thread):
    def __init__(
        self,
        engine: Engine,
        worker: WorkerInfo,
        registry: InflightRegistry,
        settings: Settings,
    ) -> None:
        super().__init__(name=f"heartbeat-{worker.worker_id}", daemon=True)
        self.engine = engine
        self.worker = worker
        self.registry = registry
        self.settings = settings
        self.state = "ACTIVE"
        self._stop_event = threading.Event()
        self._proc = psutil.Process()
        self.beats = 0
        self.failures = 0

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    def run(self) -> None:
        interval = self.settings.heartbeat_interval_s
        while True:
            self.beat_once()
            if self._stop_event.wait(interval):
                return

    def beat_once(self) -> dict[uuid.UUID, LeaseStatus]:
        tasks = self.registry.snapshot()
        profile = resource_profile(self._proc)
        rss = float(profile["rss_mb"])
        extendable: list[tuple[uuid.UUID, int]] = []
        for task in tasks:
            task.peak_rss_mb = max(task.peak_rss_mb, rss)
            if task.hard_deadline_passed(self.settings.lease_ttl_s):
                log.error(
                    "hard timeout watchdog: not extending lease",
                    extra={"task_id": str(task.task_id), "attempt": task.attempt},
                )
                task.token.abort("LEASE_LOST")
            else:
                extendable.append((task.task_id, task.attempt))
        try:
            statuses = repo.heartbeat(
                self.engine,
                self.worker,
                inflight=extendable,
                lease_ttl_s=self.settings.lease_ttl_s,
                resource_profile=profile,
                state=self.state,
            )
        except Exception:
            self.failures += 1
            _inc("heartbeat_failures_total")
            log.exception("heartbeat failed", extra={"failures": self.failures})
            return {}
        self.beats += 1
        by_id = {t.task_id: t for t in tasks}
        for task_id, status in statuses.items():
            tracked = by_id.get(task_id)
            if tracked is None:
                continue
            if status == LeaseStatus.LOST:
                _inc("lease_lost_total")
                log.warning("lease lost; aborting task", extra={"task_id": str(task_id)})
                tracked.token.abort("LEASE_LOST")
            elif status == LeaseStatus.CANCEL_REQUESTED:
                log.info("cancel requested; aborting task", extra={"task_id": str(task_id)})
                tracked.token.abort("CANCELED")
        return statuses


def _inc(metric: str, **labels: str) -> None:
    """Increment a metric from quantgrid.observability.metrics if it exists."""
    try:
        from quantgrid.observability import metrics
    except ImportError:  # pragma: no cover
        return
    counter = getattr(metrics, metric, None)
    if counter is None:
        return
    (counter.labels(**labels) if labels else counter).inc()
