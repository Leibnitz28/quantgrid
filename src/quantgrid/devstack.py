"""Launch and control real local coordinator / worker / API processes.

Used by the end-to-end tests and ``scripts/``. Each child has its own session so tests
can send real ``SIGKILL`` / ``SIGSTOP`` / ``SIGTERM`` to a worker.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import IO

from sqlalchemy import Engine, text

#: Environment applied to every child: one numeric thread per process, unbuffered output.
CHILD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

TERMINAL_BATCH_STATES = ("COMPLETED", "COMPLETED_WITH_FAILURES", "CANCELED")


@dataclass
class ManagedProcess:
    name: str
    popen: subprocess.Popen[bytes]
    log_path: Path
    _log_handle: IO[bytes] | None = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None

    def send(self, sig: int) -> None:
        if self.alive():
            os.kill(self.pid, sig)

    def kill(self) -> None:
        """SIGKILL: no cleanup or lease release, like a crashed host."""
        self.send(signal.SIGKILL)
        self.popen.wait(timeout=10)
        self._close_log()

    def stop(self, timeout: float = 30.0) -> int | None:
        """Graceful SIGTERM, escalating to SIGKILL after ``timeout`` seconds."""
        if self.alive():
            self.send(signal.SIGCONT)
            self.send(signal.SIGTERM)
            try:
                self.popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.send(signal.SIGKILL)
                self.popen.wait(timeout=10)
        self._close_log()
        return self.popen.returncode

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def tail(self, lines: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""


class LocalStack:
    """A set of child processes sharing one environment (DB URL, queue, data dirs)."""

    def __init__(
        self,
        *,
        log_dir: Path,
        env: Mapping[str, str] | None = None,
        python: str = sys.executable,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.env = {**os.environ, **CHILD_ENV_DEFAULTS, **(env or {})}
        self.processes: dict[str, ManagedProcess] = {}

    def _spawn(
        self, name: str, argv: list[str], extra_env: Mapping[str, str] | None = None
    ) -> ManagedProcess:
        if name in self.processes and self.processes[name].alive():
            raise RuntimeError(f"process {name!r} already running")
        log_path = self.log_dir / f"{name}.log"
        handle = log_path.open("ab")
        popen = subprocess.Popen(  # noqa: S603 - argv built from our own module paths
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**self.env, **(extra_env or {})},
            start_new_session=True,
        )
        proc = ManagedProcess(name, popen, log_path, handle)
        self.processes[name] = proc
        return proc

    def start_coordinator(self, name: str = "coordinator") -> ManagedProcess:
        return self._spawn(name, [self.python, "-m", "quantgrid.coordinator.main"])

    def start_worker(self, worker_id: str | None = None) -> ManagedProcess:
        wid = worker_id or f"w-{uuid.uuid4().hex[:6]}"
        return self._spawn(
            wid,
            [self.python, "-m", "quantgrid.workers.main", "--worker-id", wid],
            {"QG_WORKER_ID": wid},
        )

    def start_api(self, port: int, name: str = "api") -> ManagedProcess:
        return self._spawn(
            name,
            [
                self.python,
                "-m",
                "uvicorn",
                "quantgrid.api.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
        )

    def workers(self) -> dict[str, ManagedProcess]:
        return {n: p for n, p in self.processes.items() if n not in ("coordinator", "api")}

    def wait_for_workers(
        self, engine: Engine, worker_ids: Iterable[str], *, timeout: float = 60.0
    ) -> None:
        ids = list(worker_ids)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with engine.connect() as conn:
                n = conn.execute(
                    text(
                        "SELECT count(*) FROM worker_heartbeats WHERE worker_id = ANY(:ids) "
                        "AND state = 'ACTIVE' AND last_seen > now() - interval '30 seconds'"
                    ),
                    {"ids": ids},
                ).scalar_one()
            if n == len(ids):
                return
            for wid in ids:
                proc = self.processes.get(wid)
                if proc is not None and not proc.alive():
                    raise RuntimeError(f"worker {wid} exited early:\n{proc.tail()}")
            time.sleep(0.2)
        raise TimeoutError(f"workers {ids} did not register within {timeout}s")

    def wait_for_batch(
        self,
        engine: Engine,
        batch_id: uuid.UUID | str,
        *,
        timeout: float = 300.0,
        poll_s: float = 0.2,
        states: tuple[str, ...] = TERMINAL_BATCH_STATES,
    ) -> str:
        deadline = time.monotonic() + timeout
        state = "?"
        while time.monotonic() < deadline:
            with engine.connect() as conn:
                state = conn.execute(
                    text("SELECT state FROM batches WHERE id = :id"), {"id": str(batch_id)}
                ).scalar_one()
            if state in states:
                return str(state)
            coord = self.processes.get("coordinator")
            if coord is not None and not coord.alive():
                raise RuntimeError(f"coordinator exited:\n{coord.tail()}")
            time.sleep(poll_s)
        raise TimeoutError(f"batch {batch_id} still {state} after {timeout}s")

    def stop_all(self, timeout: float = 30.0) -> None:
        for name in sorted(self.processes, key=lambda n: n == "coordinator"):
            self.processes[name].stop(timeout)

    def __enter__(self) -> LocalStack:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop_all()
