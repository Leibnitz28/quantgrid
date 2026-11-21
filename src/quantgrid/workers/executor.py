"""Execute one task end-to-end under a lease (Celery-independent, unit-testable).

Flow for a wake-up message carrying ``task_id``::

    claim (conditional UPDATE, new attempt)      -- not claimable? drop the message
      -> re-validate the immutable spec (hash must match)
      -> cache lookup (same spec_hash + code_sha + env_sha, artifact checksum verified)
      -> load recorded bars (checksum-verified, per-process LRU)
      -> deterministic engine (cooperative cancel/deadline checks)
      -> encode deterministic Parquet -> stage (temp + fsync) -> commit (atomic rename, CAS)
      -> finalize_success (fenced: same owner + attempt, row-locked; publishes once)

Failure mapping (all fenced by the lease):
  TaskAborted(CANCELED)  -> acknowledge_cancel          (task CANCELED)
  TaskAborted(SHUTDOWN)  -> release_lease               (re-queued without waiting for TTL)
  TaskAborted(LEASE_LOST)-> nothing (coordinator already re-queued it)
  TaskExecutionError     -> record_failure(code, retryable) with exponential backoff
  any other exception    -> record_failure(TASK_ERROR, retryable=True)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import Engine

from quantgrid.config import Settings
from quantgrid.domain.errors import (
    ArtifactError,
    ErrorCode,
    SpecError,
    TaskAborted,
    TaskExecutionError,
)
from quantgrid.domain.specs import DatasetRef, TaskSpec
from quantgrid.domain.types import Bars, DeadlineCancelToken
from quantgrid.storage import repository as repo
from quantgrid.storage.repository import (
    ArtifactRecord,
    AttemptStats,
    ClaimOutcome,
    FinalizeOutcome,
    WorkerInfo,
)
from quantgrid.workers.heartbeat import InflightRegistry, InflightTask

log = logging.getLogger("quantgrid.worker")

BarsLoader = Callable[[DatasetRef, Path, bool], Bars]


def _default_bars_loader(ref: DatasetRef, data_dir: Path, auto_materialize: bool) -> Bars:
    from quantgrid.domain.datasets import load_bars

    return load_bars(ref, data_dir, auto_materialize=auto_materialize, verify=True)


class BarsLRU:
    """Per-process cache of verified bars keyed by (dataset name, sha256)."""

    def __init__(self, loader: BarsLoader, capacity: int = 4) -> None:
        self._loader = loader
        self._capacity = capacity
        self._items: OrderedDict[tuple[str, str], Bars] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, ref: DatasetRef, data_dir: Path, auto_materialize: bool) -> Bars:
        key = (ref.name, ref.sha256)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                self.hits += 1
                return self._items[key]
        bars = self._loader(ref, data_dir, auto_materialize)
        with self._lock:
            self.misses += 1
            self._items[key] = bars
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return bars


@dataclass(frozen=True)
class ExecutionResult:
    task_id: str
    outcome: str  # ClaimOutcome (when not claimed) or the final outcome
    attempt: int | None = None
    output_sha: str | None = None
    wall_ms: int | None = None


class TaskExecutor:
    def __init__(
        self,
        *,
        engine: Engine,
        settings: Settings,
        worker: WorkerInfo,
        registry: InflightRegistry,
        store: Any | None = None,
        bars_loader: BarsLoader | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.worker = worker
        self.registry = registry
        if store is None:
            from quantgrid.storage.artifacts import LocalArtifactStore

            store = LocalArtifactStore(Path(settings.artifact_root))
        self.store = store
        self.bars = BarsLRU(bars_loader or _default_bars_loader)

    def backoff_s(self, attempt: int) -> float:
        base = self.settings.retry_backoff_base_s
        return float(min(self.settings.retry_backoff_max_s, base * (2 ** max(0, attempt - 1))))

    def execute(self, task_id: str | uuid.UUID, *, trace_id: str | None = None) -> ExecutionResult:
        tid = uuid.UUID(str(task_id))
        outcome, claimed = repo.claim_task(
            self.engine,
            tid,
            self.worker.worker_id,
            lease_ttl_s=self.settings.lease_ttl_s,
            code_sha=self.worker.code_sha,
            env_sha=self.worker.env_sha,
            trace_id=trace_id,
        )
        if outcome != ClaimOutcome.CLAIMED or claimed is None:
            log.info("wake-up dropped", extra={"task_id": str(tid), "claim": outcome.value})
            _inc("task_claims_total", outcome=outcome.value)
            return ExecutionResult(str(tid), outcome.value)
        _inc("task_claims_total", outcome=ClaimOutcome.CLAIMED.value)
        _observe("queue_wait_seconds", claimed.queue_wait_ms / 1000.0)

        token = DeadlineCancelToken(max_runtime_s=float(claimed.max_runtime_s))
        inflight = InflightTask(
            claimed.task_id, claimed.attempt, float(claimed.max_runtime_s), token
        )
        self.registry.add(inflight)
        t0 = time.monotonic()
        c0 = time.thread_time()
        ctx = {"task_id": str(tid), "batch_id": str(claimed.batch_id), "attempt": claimed.attempt}

        def stats() -> AttemptStats:
            return AttemptStats(
                wall_ms=int((time.monotonic() - t0) * 1000),
                cpu_ms=int((time.thread_time() - c0) * 1000),
                peak_rss_mb=inflight.peak_rss_mb or None,
            )

        try:
            with _bind(**ctx):
                return self._run_claimed(claimed, token, stats)
        except TaskAborted as abort:
            return self._handle_abort(claimed, abort, stats())
        except TaskExecutionError as exc:
            return self._handle_failure(claimed, exc.code, str(exc), exc.retryable, stats())
        except Exception as exc:  # unexpected: retry, the next attempt may succeed
            log.exception("unexpected task error", extra=ctx)
            message = f"{type(exc).__name__}: {exc}"
            return self._handle_failure(claimed, ErrorCode.TASK_ERROR, message, True, stats())
        finally:
            self.registry.remove(claimed.task_id)

    def _run_claimed(
        self,
        claimed: repo.ClaimedTask,
        token: DeadlineCancelToken,
        stats: Callable[[], AttemptStats],
    ) -> ExecutionResult:
        try:
            spec = TaskSpec.model_validate(claimed.spec)
        except ValidationError as exc:
            raise SpecError(
                f"stored task spec failed validation: {exc.error_count()} errors"
            ) from exc
        if spec.spec_hash() != claimed.spec_hash:
            raise SpecError("stored task spec does not match its spec_hash (corrupted row?)")

        if claimed.cache_policy == "use":
            cached = self._try_cache(claimed, stats)
            if cached is not None:
                return cached

        from quantgrid.domain.engine import ENGINE_VERSION, run_task_spec
        from quantgrid.results.codec import encode_result

        bars = self.bars.get(
            spec.dataset, Path(self.settings.data_dir), self.settings.auto_materialize_datasets
        )
        token.check()
        output = run_task_spec(spec, bars, cancel=token, attempt=claimed.attempt)
        encoded = encode_result(claimed.spec_hash, output, engine_version=ENGINE_VERSION)
        token.check()

        staged = self.store.stage(encoded.data, hint=f"{claimed.task_id}.{claimed.attempt}")
        try:
            published = self.store.commit(staged)
        except Exception as exc:
            self.store.discard(staged)
            raise ArtifactError(f"artifact commit failed: {type(exc).__name__}: {exc}") from exc
        _inc("artifact_commits_total", outcome="new" if published.created else "duplicate")
        if published.created:
            _inc_by("artifact_bytes_written_total", float(published.size))

        attempt_stats = stats()
        result = repo.finalize_success(
            self.engine,
            task_id=claimed.task_id,
            attempt=claimed.attempt,
            worker_id=self.worker.worker_id,
            artifact=ArtifactRecord(
                output_uri=published.uri,
                output_sha=published.sha256,
                size_bytes=published.size,
                row_count=encoded.row_count,
                metrics=encoded.metrics,
                temp_uri=staged.temp_uri,
            ),
            stats=attempt_stats,
            code_sha=self.worker.code_sha or "",
            env_sha=self.worker.env_sha or "",
        )
        return self._report_finalize(claimed, result, published.sha256, attempt_stats)

    def _try_cache(
        self, claimed: repo.ClaimedTask, stats: Callable[[], AttemptStats]
    ) -> ExecutionResult | None:
        hit = repo.find_cached_result(
            self.engine,
            spec_hash=claimed.spec_hash,
            code_sha=self.worker.code_sha or "",
            env_sha=self.worker.env_sha or "",
        )
        if hit is None:
            return None
        if not self.store.verify(hit["output_uri"], hit["output_sha"]):
            log.warning(
                "cached artifact failed verification; recomputing",
                extra={"task_id": str(claimed.task_id), "uri": hit["output_uri"]},
            )
            return None
        attempt_stats = stats()
        result = repo.finalize_success(
            self.engine,
            task_id=claimed.task_id,
            attempt=claimed.attempt,
            worker_id=self.worker.worker_id,
            artifact=ArtifactRecord(
                output_uri=hit["output_uri"],
                output_sha=hit["output_sha"],
                size_bytes=int(hit["size_bytes"]),
                row_count=int(hit["row_count"]),
                metrics=dict(hit["metrics"]),
            ),
            stats=attempt_stats,
            code_sha=self.worker.code_sha or "",
            env_sha=self.worker.env_sha or "",
            cached_from_task_id=hit["task_id"],
        )
        if result.outcome == FinalizeOutcome.PUBLISHED:
            _inc("cache_hits_total")
        return self._report_finalize(claimed, result, hit["output_sha"], attempt_stats)

    def _report_finalize(
        self,
        claimed: repo.ClaimedTask,
        result: repo.FinalizeResult,
        sha: str,
        attempt_stats: AttemptStats,
    ) -> ExecutionResult:
        seconds = (attempt_stats.wall_ms or 0) / 1000.0
        if result.outcome == FinalizeOutcome.PUBLISHED:
            _inc("task_attempts_total", status="SUCCEEDED", error_code="")
            _observe("task_duration_seconds", seconds, outcome="succeeded")
        elif result.outcome == FinalizeOutcome.LEASE_LOST:
            checksum = (
                "unknown"
                if result.checksum_match is None
                else ("identical" if result.checksum_match else "mismatch")
            )
            _inc("late_completions_total", checksum=checksum)
            log.warning(
                "finished after losing the lease; result not published",
                extra={"task_id": str(claimed.task_id), "checksum": checksum},
            )
        else:
            _inc("task_attempts_total", status="CANCELED", error_code="")
        log.info(
            "task finished",
            extra={
                "task_id": str(claimed.task_id),
                "outcome": result.outcome.value,
                "output_sha": sha,
                "wall_ms": attempt_stats.wall_ms,
            },
        )
        return ExecutionResult(
            str(claimed.task_id), result.outcome.value, claimed.attempt, sha, attempt_stats.wall_ms
        )

    def _handle_abort(
        self, claimed: repo.ClaimedTask, abort: TaskAborted, attempt_stats: AttemptStats
    ) -> ExecutionResult:
        if abort.reason == "CANCELED":
            repo.acknowledge_cancel(
                self.engine,
                task_id=claimed.task_id,
                attempt=claimed.attempt,
                worker_id=self.worker.worker_id,
                stats=attempt_stats,
            )
            _inc("task_attempts_total", status="CANCELED", error_code="")
        elif abort.reason == "SHUTDOWN":
            repo.release_lease(
                self.engine,
                task_id=claimed.task_id,
                attempt=claimed.attempt,
                worker_id=self.worker.worker_id,
                reason="worker shutdown",
                stats=attempt_stats,
            )
        else:
            _inc("task_attempts_total", status="LEASE_LOST", error_code="")
        log.warning("task aborted", extra={"task_id": str(claimed.task_id), "reason": abort.reason})
        return ExecutionResult(
            str(claimed.task_id),
            f"ABORTED_{abort.reason}",
            claimed.attempt,
            None,
            attempt_stats.wall_ms,
        )

    def _handle_failure(
        self,
        claimed: repo.ClaimedTask,
        code: ErrorCode,
        message: str,
        retryable: bool,
        attempt_stats: AttemptStats,
    ) -> ExecutionResult:
        outcome = repo.record_failure(
            self.engine,
            task_id=claimed.task_id,
            attempt=claimed.attempt,
            worker_id=self.worker.worker_id,
            error_code=code,
            message=message,
            retryable=retryable,
            stats=attempt_stats,
            backoff_s=self.backoff_s(claimed.attempt),
        )
        _inc("task_attempts_total", status="FAILED", error_code=code.value)
        _observe("task_duration_seconds", (attempt_stats.wall_ms or 0) / 1000.0, outcome="failed")
        log.warning(
            "task attempt failed",
            extra={
                "task_id": str(claimed.task_id),
                "error_code": code.value,
                "outcome": outcome.value,
                "error": message,
            },
        )
        return ExecutionResult(
            str(claimed.task_id), outcome.value, claimed.attempt, None, attempt_stats.wall_ms
        )


def _metric(name: str) -> Any:
    try:
        from quantgrid.observability import metrics
    except ImportError:  # pragma: no cover
        return None
    return getattr(metrics, name, None)


def _inc(name: str, **labels: str) -> None:
    metric = _metric(name)
    if metric is not None:
        (metric.labels(**labels) if labels else metric).inc()


def _inc_by(name: str, amount: float) -> None:
    metric = _metric(name)
    if metric is not None:
        metric.inc(amount)


def _observe(name: str, value: float, **labels: str) -> None:
    metric = _metric(name)
    if metric is not None:
        (metric.labels(**labels) if labels else metric).observe(value)


class _bind:
    """Bind logging context if the observability package provides ``bind_context``."""

    def __init__(self, **kv: Any) -> None:
        self._kv = kv
        self._cm: Any = None

    def __enter__(self) -> None:
        try:
            from quantgrid.observability.logging import bind_context
        except ImportError:  # pragma: no cover
            return
        self._cm = bind_context(**self._kv)
        self._cm.__enter__()

    def __exit__(self, *exc: object) -> None:
        if self._cm is not None:
            self._cm.__exit__(*exc)
