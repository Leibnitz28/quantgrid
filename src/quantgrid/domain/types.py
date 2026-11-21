"""Runtime value types shared by the domain engine, workers and result codec."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from quantgrid.domain.errors import TaskAborted, TaskTimeout

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Bars:
    """Recorded input bars for a two-asset (pairs) dataset.

    ``ts`` is int64 nanoseconds since the Unix epoch, UTC, strictly increasing (gaps are
    allowed: the engine treats bars sequentially). Prices are strictly positive finite
    float64. ``sha256`` is the verified logical content hash (see ``domain.datasets``).
    """

    name: str
    ts: IntArray
    close_x: FloatArray
    close_y: FloatArray
    sha256: str

    def __len__(self) -> int:
        return int(self.ts.shape[0])


#: Scalar metric values stored in result manifests (``None`` = undefined, e.g. Sharpe of a
#: zero-variance series). Must be finite when not ``None``.
MetricValue = float | int | None


@dataclass(frozen=True)
class TaskOutput:
    """Deterministic output of one task.

    * ``metrics`` - small scalar summary (stored in Postgres ``result_artifacts.metrics``
      and embedded in the artifact).
    * ``series``  - equal-length 1-D arrays for the *test* segment, written to Parquet.
      Must contain ``ts`` (int64 ns UTC) plus float64/int64 columns.
    """

    metrics: Mapping[str, MetricValue]
    series: Mapping[str, npt.NDArray[np.generic]]
    diagnostics: Mapping[str, float | int | str] = field(default_factory=dict)


class CancelToken(Protocol):
    """Cooperative cancellation / deadline check. Long loops call :meth:`check` often
    (at least every ~50 ms of work). Raises :class:`TaskAborted` or :class:`TaskTimeout`."""

    def check(self) -> None: ...


class NullCancelToken:
    def check(self) -> None:
        return None


class DeadlineCancelToken:
    """Raises :class:`TaskTimeout` once ``max_runtime_s`` (monotonic) has elapsed, and
    :class:`TaskAborted` if :meth:`abort` was called (e.g. by a heartbeat thread)."""

    def __init__(self, max_runtime_s: float | None = None) -> None:
        self._deadline = None if max_runtime_s is None else time.monotonic() + max_runtime_s
        self._max_runtime_s = max_runtime_s
        self._abort_reason: str | None = None

    def abort(self, reason: str) -> None:
        if self._abort_reason is None:
            self._abort_reason = reason

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def check(self) -> None:
        reason = self._abort_reason
        if reason is not None:
            if reason not in ("CANCELED", "LEASE_LOST", "SHUTDOWN"):
                reason = "SHUTDOWN"
            raise TaskAborted(reason)  # type: ignore[arg-type]
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise TaskTimeout(f"task exceeded max_runtime_s={self._max_runtime_s}")
