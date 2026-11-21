"""Per-process worker runtime: identity, DB engine, heartbeat thread and executor."""

from __future__ import annotations

import logging
import os
import socket
import threading
from datetime import UTC, datetime

from sqlalchemy import Engine

from quantgrid.config import Settings, get_settings
from quantgrid.manifest import code_digest, env_sha
from quantgrid.storage import repository as repo
from quantgrid.storage.db import make_engine
from quantgrid.storage.repository import WorkerInfo
from quantgrid.workers.executor import ExecutionResult, TaskExecutor
from quantgrid.workers.heartbeat import HeartbeatThread, InflightRegistry

log = logging.getLogger("quantgrid.worker")


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


class WorkerRuntime:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        engine: Engine | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.worker_id = worker_id or self.settings.worker_id or default_worker_id()
        self.engine = engine or make_engine(
            pool_size=2, application_name=f"qg-worker-{self.worker_id}"
        )
        self.worker = WorkerInfo(
            worker_id=self.worker_id,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            started_at=datetime.now(UTC),
            concurrency=1,
            code_sha=code_digest(),
            env_sha=env_sha(),
        )
        self.registry = InflightRegistry()
        self.executor = TaskExecutor(
            engine=self.engine, settings=self.settings, worker=self.worker, registry=self.registry
        )
        self.heartbeat = HeartbeatThread(self.engine, self.worker, self.registry, self.settings)
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.heartbeat.beat_once()  # register immediately so the coordinator sees capacity
        self.heartbeat.start()
        log.info(
            "worker started",
            extra={
                "worker_id": self.worker_id,
                "code_sha": self.worker.code_sha,
                "env_sha": self.worker.env_sha,
            },
        )

    def begin_shutdown(self) -> None:
        """Warm shutdown: stop taking new work (Celery stops consuming), finish the current
        task, and advertise STOPPING so the coordinator stops counting this capacity."""
        self.heartbeat.state = "STOPPING"
        log.info(
            "worker shutting down",
            extra={"worker_id": self.worker_id, "inflight": len(self.registry)},
        )

    def abort_inflight(self) -> None:
        """Cold shutdown: ask running tasks to stop and release their leases early."""
        for task in self.registry.snapshot():
            task.token.abort("SHUTDOWN")

    def stop(self) -> None:
        self.heartbeat.stop()
        try:
            repo.mark_worker_state(self.engine, self.worker_id, "STOPPED")
        except Exception:  # pragma: no cover - best effort during shutdown
            log.exception("could not mark worker stopped")
        self.engine.dispose()
        log.info("worker stopped", extra={"worker_id": self.worker_id})

    def execute(self, task_id: str, *, trace_id: str | None = None) -> ExecutionResult:
        if not self._started:
            self.start()
        return self.executor.execute(task_id, trace_id=trace_id)


_runtime: WorkerRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> WorkerRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = WorkerRuntime()
        return _runtime
