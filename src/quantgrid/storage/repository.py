"""Transactional scheduler operations; the only code that changes scheduler state.

Rules (full argument in docs/correctness.md):
* PostgreSQL is the source of truth. Celery/Redis messages are wake-up hints only.
* A lease is fenced by ``(task_id, lease_owner, attempts)``; only the reconciler
  takes one away.
* Explicit row locks are ``FOR NO KEY UPDATE``: plain ``FOR UPDATE`` on the batch row
  deadlocks cancel against publish via foreign-key ``FOR KEY SHARE`` locks.
  Lock order is batch -> task.
* Lease arithmetic uses the database clock (``now()``), never worker clocks.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Connection, Engine, RowMapping, insert, select, text

from quantgrid.domain.errors import (
    RETRYABLE_TASK_ERRORS,
    Conflict,
    ErrorCode,
    NotFound,
    QuotaExceeded,
)
from quantgrid.domain.policy import QuotaPolicy
from quantgrid.domain.specs import DatasetInfo, ExpandedTask, ExperimentBatchSpec
from quantgrid.domain.states import (
    TERMINAL_BATCH_STATES,
    BatchState,
    TaskState,
    derive_batch_state,
)
from quantgrid.storage.db import notify, set_actor, set_reason, transaction
from quantgrid.storage.models import batches, result_artifacts, task_attempts, tasks

#: Advisory-lock key for coordinator leader election ("QGLD").
LEADER_LOCK_KEY = 0x5147_4C44
MAX_ERROR_MESSAGE = 2_000
MAX_COMPLETED_IDS_IN_REPORT = 1_000


def _uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _truncate(message: str | None, limit: int = MAX_ERROR_MESSAGE) -> str | None:
    if message is None:
        return None
    return message if len(message) <= limit else message[: limit - 3] + "..."


@dataclass(frozen=True)
class CreateBatchResult:
    batch_id: uuid.UUID
    created: bool  # False = idempotent replay of an existing logical batch
    task_count: int
    state: str


def create_batch(
    engine: Engine,
    *,
    user_id: str,
    idempotency_key: str,
    spec: ExperimentBatchSpec,
    tasks_expanded: Sequence[ExpandedTask],
    manifest: Mapping[str, Any],
    quota: QuotaPolicy,
    dataset: DatasetInfo,
    request_id: str | None = None,
    reproduces_batch_id: uuid.UUID | None = None,
) -> CreateBatchResult:
    """Insert a batch and its tasks atomically, idempotent per ``(user_id, idempotency_key)``.

    Same key and spec returns the existing batch (``created=False``); same key with a
    different spec raises :class:`Conflict`. Concurrent submits are serialised by the
    unique index (``ON CONFLICT DO NOTHING`` waits for the winner).
    """
    if not tasks_expanded:
        raise ValueError("a batch needs at least one task")
    spec_sha = spec.spec_sha()
    batch_id = uuid.uuid4()
    code = manifest.get("code", {})
    env = manifest.get("environment", {})
    with transaction(engine) as conn:
        inserted = conn.execute(
            text(
                """
                INSERT INTO batches (
                    id, user_id, idempotency_key, name, state, spec, spec_sha,
                    dataset_name, dataset_sha, code_sha, env_sha, manifest, quota,
                    task_count, max_attempts, priority, cache_policy,
                    reproduces_batch_id, request_id
                ) VALUES (
                    :id, :user_id, :key, :name, 'ACCEPTED', CAST(:spec AS jsonb), :spec_sha,
                    :dataset_name, :dataset_sha, :code_sha, :env_sha, CAST(:manifest AS jsonb),
                    CAST(:quota AS jsonb), :task_count, :max_attempts, :priority, :cache_policy,
                    :reproduces, :request_id
                )
                ON CONFLICT (user_id, idempotency_key) DO NOTHING
                RETURNING id
                """
            ),
            {
                "id": batch_id,
                "user_id": user_id,
                "key": idempotency_key,
                "name": spec.name,
                "spec": _json(spec.canonical()),
                "spec_sha": spec_sha,
                "dataset_name": dataset.name,
                "dataset_sha": dataset.sha256,
                "code_sha": code.get("code_sha", ""),
                "env_sha": env.get("env_sha", ""),
                "manifest": _json(dict(manifest)),
                "quota": _json(quota.model_dump(mode="json")),
                "task_count": len(tasks_expanded),
                "max_attempts": spec.max_attempts,
                "priority": spec.priority,
                "cache_policy": spec.cache_policy,
                "reproduces": reproduces_batch_id,
                "request_id": request_id,
            },
        ).first()
        if inserted is None:
            existing = conn.execute(
                select(
                    batches.c.id, batches.c.spec_sha, batches.c.task_count, batches.c.state
                ).where(batches.c.user_id == user_id, batches.c.idempotency_key == idempotency_key)
            ).one()
            if existing.spec_sha != spec_sha:
                raise Conflict(
                    "Idempotency-Key was already used for a different batch spec",
                    details={
                        "batch_id": str(existing.id),
                        "existing_spec_sha": existing.spec_sha,
                        "request_spec_sha": spec_sha,
                    },
                )
            return CreateBatchResult(existing.id, False, existing.task_count, existing.state)

        set_actor(conn, "api", f"batch submitted by {user_id}")
        conn.execute(
            insert(tasks),
            [
                {
                    "id": uuid.uuid4(),
                    "batch_id": batch_id,
                    "ordinal": t.ordinal,
                    "spec": t.spec.canonical(),
                    "spec_hash": t.spec_hash,
                    "state": TaskState.PENDING.value,
                    "max_attempts": spec.max_attempts,
                    "max_runtime_s": spec.budget.max_runtime_s,
                    "max_memory_mb": spec.budget.max_memory_mb,
                    "priority": spec.priority,
                    "cache_policy": spec.cache_policy,
                }
                for t in tasks_expanded
            ],
        )
        notify(conn, "batch")
    return CreateBatchResult(batch_id, True, len(tasks_expanded), BatchState.ACCEPTED.value)


def _json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def find_batch_by_key(engine: Engine, user_id: str, idempotency_key: str) -> dict[str, Any] | None:
    """The batch already bound to ``(user_id, idempotency_key)``, if any."""
    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT id, spec_sha, task_count, state FROM batches "
                    "WHERE user_id = :u AND idempotency_key = :k"
                ),
                {"u": user_id, "k": idempotency_key},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def active_batch_count(engine: Engine, user_id: str) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    "SELECT count(*) FROM batches WHERE user_id = :u "
                    "AND state IN ('ACCEPTED', 'RUNNING', 'CANCELING')"
                ),
                {"u": user_id},
            ).scalar_one()
        )


@dataclass(frozen=True)
class DispatchItem:
    task_id: uuid.UUID
    batch_id: uuid.UUID
    priority: int
    dispatch_count: int
    redelivery: bool  # previous wake-up was presumed lost


def outstanding_dispatches(conn: Connection, lost_after_s: float) -> int:
    return int(
        conn.execute(
            text(
                """
                SELECT count(*) FROM tasks
                WHERE state IN ('PENDING', 'RETRY_PENDING')
                  AND enqueued_at IS NOT NULL
                  AND enqueued_at > now() - make_interval(secs => :lost)
                """
            ),
            {"lost": lost_after_s},
        ).scalar_one()
    )


def dispatch_ready(
    engine: Engine,
    *,
    window: int,
    lost_after_s: float,
    limit: int = 1_000,
) -> list[DispatchItem]:
    """Mark up to ``window - outstanding`` eligible tasks as enqueued and return them.

    The caller sends the broker messages after commit. If it dies first, the tasks are
    re-dispatched after ``lost_after_s`` (``redelivery=True``); duplicates are harmless
    because claiming is conditional.
    """
    with transaction(engine) as conn:
        free = min(limit, window - outstanding_dispatches(conn, lost_after_s))
        if free <= 0:
            return []
        rows = conn.execute(
            text(
                """
                WITH candidates AS (
                    SELECT id, enqueued_at AS prev_enqueued_at
                    FROM tasks
                    WHERE state IN ('PENDING', 'RETRY_PENDING')
                      AND cancel_requested = false
                      AND (not_before IS NULL OR not_before <= now())
                      AND (enqueued_at IS NULL
                           OR enqueued_at <= now() - make_interval(secs => :lost))
                    ORDER BY priority DESC, ready_at, batch_id, ordinal
                    LIMIT :free
                    FOR NO KEY UPDATE SKIP LOCKED
                )
                UPDATE tasks AS t
                SET enqueued_at = now(), dispatch_count = t.dispatch_count + 1
                FROM candidates AS c
                WHERE t.id = c.id
                RETURNING t.id, t.batch_id, t.priority, t.ready_at, t.ordinal,
                          t.dispatch_count, (c.prev_enqueued_at IS NOT NULL) AS redelivery
                """
            ),
            {"lost": lost_after_s, "free": free},
        ).all()
    ordered = sorted(rows, key=lambda r: (-r.priority, r.ready_at, str(r.batch_id), r.ordinal))
    return [
        DispatchItem(r.id, r.batch_id, r.priority, r.dispatch_count, bool(r.redelivery))
        for r in ordered
    ]


class ClaimOutcome(StrEnum):
    CLAIMED = "CLAIMED"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_RUNNING = "ALREADY_RUNNING"
    TERMINAL = "TERMINAL"
    NOT_READY = "NOT_READY"  # retry backoff not elapsed
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    ATTEMPTS_EXHAUSTED = "ATTEMPTS_EXHAUSTED"


@dataclass(frozen=True)
class ClaimedTask:
    task_id: uuid.UUID
    batch_id: uuid.UUID
    attempt: int
    max_attempts: int
    spec: dict[str, Any]
    spec_hash: str
    max_runtime_s: int
    max_memory_mb: int
    cache_policy: str
    lease_expiry: datetime
    queue_wait_ms: int


def claim_task(
    engine: Engine,
    task_id: uuid.UUID | str,
    worker_id: str,
    *,
    lease_ttl_s: float,
    code_sha: str | None = None,
    env_sha: str | None = None,
    trace_id: str | None = None,
) -> tuple[ClaimOutcome, ClaimedTask | None]:
    """Take a lease on a claimable task and open a new attempt.

    Of several racing workers exactly one gets CLAIMED; the others get another outcome.
    """
    tid = _uuid(task_id)
    with transaction(engine) as conn:
        set_actor(conn, f"worker:{worker_id}", "claimed")
        row = conn.execute(
            text(
                """
                UPDATE tasks
                SET state = 'RUNNING',
                    lease_owner = :worker,
                    lease_expiry = now() + make_interval(secs => :ttl),
                    attempts = attempts + 1,
                    started_at = coalesce(started_at, now()),
                    enqueued_at = NULL
                WHERE id = :id
                  AND state IN ('PENDING', 'RETRY_PENDING')
                  AND cancel_requested = false
                  AND attempts < max_attempts
                  AND (not_before IS NULL OR not_before <= now())
                RETURNING batch_id, attempts, max_attempts, spec, spec_hash, max_runtime_s,
                          max_memory_mb, cache_policy, lease_expiry,
                          GREATEST(0, (EXTRACT(EPOCH FROM (now() - ready_at)) * 1000))::bigint
                              AS queue_wait_ms
                """
            ),
            {"id": tid, "worker": worker_id, "ttl": lease_ttl_s},
        ).first()
        if row is None:
            return _diagnose_unclaimable(conn, tid), None
        conn.execute(
            insert(task_attempts).values(
                task_id=tid,
                batch_id=row.batch_id,
                attempt=row.attempts,
                worker_id=worker_id,
                status="RUNNING",
                started_at=text("now()"),
                queue_wait_ms=row.queue_wait_ms,
                code_sha=code_sha,
                env_sha=env_sha,
                trace_id=trace_id,
            )
        )
    return ClaimOutcome.CLAIMED, ClaimedTask(
        task_id=tid,
        batch_id=row.batch_id,
        attempt=row.attempts,
        max_attempts=row.max_attempts,
        spec=dict(row.spec),
        spec_hash=row.spec_hash,
        max_runtime_s=row.max_runtime_s,
        max_memory_mb=row.max_memory_mb,
        cache_policy=row.cache_policy,
        lease_expiry=row.lease_expiry,
        queue_wait_ms=int(row.queue_wait_ms),
    )


def _diagnose_unclaimable(conn: Connection, task_id: uuid.UUID) -> ClaimOutcome:
    row = conn.execute(
        text(
            "SELECT state, cancel_requested, attempts, max_attempts, "
            "(not_before IS NOT NULL AND not_before > now()) AS backoff "
            "FROM tasks WHERE id = :id"
        ),
        {"id": task_id},
    ).first()
    if row is None:
        return ClaimOutcome.NOT_FOUND
    if row.state == TaskState.RUNNING:
        return ClaimOutcome.ALREADY_RUNNING
    if row.state in (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELED):
        return ClaimOutcome.TERMINAL
    if row.cancel_requested:
        return ClaimOutcome.CANCEL_REQUESTED
    if row.attempts >= row.max_attempts:
        return ClaimOutcome.ATTEMPTS_EXHAUSTED
    if row.backoff:
        return ClaimOutcome.NOT_READY
    return ClaimOutcome.ALREADY_RUNNING  # lost a race that resolved between statements


class LeaseStatus(StrEnum):
    OK = "OK"
    LOST = "LOST"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"


@dataclass(frozen=True)
class WorkerInfo:
    worker_id: str
    hostname: str
    pid: int
    started_at: datetime
    concurrency: int = 1
    code_sha: str | None = None
    env_sha: str | None = None


def heartbeat(
    engine: Engine,
    worker: WorkerInfo,
    *,
    inflight: Sequence[tuple[uuid.UUID, int]],
    lease_ttl_s: float,
    resource_profile: Mapping[str, Any],
    state: str = "ACTIVE",
) -> dict[uuid.UUID, LeaseStatus]:
    """Upsert the worker heartbeat and extend leases whose fence still matches.

    A lease past its expiry can still be extended until the reconciler reclaims it.
    Returns the status per in-flight task.
    """
    with transaction(engine) as conn:
        conn.execute(
            text(
                """
                INSERT INTO worker_heartbeats (worker_id, hostname, pid, state, started_at,
                    last_seen, concurrency, inflight_count, resource_profile, code_sha, env_sha)
                VALUES (:wid, :host, :pid, :state, :started, now(), :conc, :inflight,
                    CAST(:profile AS jsonb), :code_sha, :env_sha)
                ON CONFLICT (worker_id) DO UPDATE SET
                    hostname = EXCLUDED.hostname, pid = EXCLUDED.pid, state = EXCLUDED.state,
                    started_at = EXCLUDED.started_at, last_seen = now(),
                    concurrency = EXCLUDED.concurrency, inflight_count = EXCLUDED.inflight_count,
                    resource_profile = EXCLUDED.resource_profile,
                    code_sha = EXCLUDED.code_sha, env_sha = EXCLUDED.env_sha
                """
            ),
            {
                "wid": worker.worker_id,
                "host": worker.hostname,
                "pid": worker.pid,
                "state": state,
                "started": worker.started_at,
                "conc": worker.concurrency,
                "inflight": len(inflight),
                "profile": _json(dict(resource_profile)),
                "code_sha": worker.code_sha,
                "env_sha": worker.env_sha,
            },
        )
        if not inflight:
            return {}
        rows = conn.execute(
            text(
                """
                UPDATE tasks AS t
                SET lease_expiry = now() + make_interval(secs => :ttl)
                FROM unnest(CAST(:ids AS uuid[]), CAST(:attempts AS int[])) AS i(id, attempt)
                WHERE t.id = i.id AND t.attempts = i.attempt
                  AND t.state = 'RUNNING' AND t.lease_owner = :wid
                RETURNING t.id, t.cancel_requested
                """
            ),
            {
                "ttl": lease_ttl_s,
                "ids": [tid for tid, _ in inflight],
                "attempts": [a for _, a in inflight],
                "wid": worker.worker_id,
            },
        ).all()
    held = {r.id: bool(r.cancel_requested) for r in rows}
    return {
        tid: (
            LeaseStatus.LOST
            if tid not in held
            else LeaseStatus.CANCEL_REQUESTED
            if held[tid]
            else LeaseStatus.OK
        )
        for tid, _ in inflight
    }


def mark_worker_state(engine: Engine, worker_id: str, state: str) -> None:
    with transaction(engine) as conn:
        conn.execute(
            text("UPDATE worker_heartbeats SET state = :s, last_seen = now() WHERE worker_id = :w"),
            {"s": state, "w": worker_id},
        )


@dataclass(frozen=True)
class ArtifactRecord:
    output_uri: str
    output_sha: str
    size_bytes: int
    row_count: int
    metrics: Mapping[str, Any]
    temp_uri: str | None = None


@dataclass(frozen=True)
class AttemptStats:
    wall_ms: int | None = None
    cpu_ms: int | None = None
    peak_rss_mb: float | None = None


class FinalizeOutcome(StrEnum):
    PUBLISHED = "PUBLISHED"
    CANCELED = "CANCELED"  # cancel won the race; result discarded
    LEASE_LOST = "LEASE_LOST"  # zombie: lease was reclaimed; result NOT published


@dataclass(frozen=True)
class FinalizeResult:
    outcome: FinalizeOutcome
    published_sha: str | None = None
    checksum_match: bool | None = None  # vs the (eventually) published result, if known


def _lock_task(conn: Connection, task_id: uuid.UUID) -> RowMapping:
    row = (
        conn.execute(
            text(
                "SELECT id, batch_id, state, attempts, max_attempts, lease_owner, "
                "cancel_requested, "
                "spec_hash FROM tasks WHERE id = :id FOR NO KEY UPDATE"
            ),
            {"id": task_id},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise NotFound(f"task {task_id} not found")
    return row


def _owns(row: RowMapping, worker_id: str, attempt: int) -> bool:
    return bool(
        row["state"] == TaskState.RUNNING
        and row["lease_owner"] == worker_id
        and row["attempts"] == attempt
    )


def _record_late_completion(
    conn: Connection, task_id: uuid.UUID, attempt: int, output_sha: str, stats: AttemptStats
) -> FinalizeResult:
    """Late finish after losing the lease: never publish; flag a checksum mismatch."""
    conn.execute(
        text(
            """
            UPDATE task_attempts
            SET output_sha = :sha, late_completion_at = now(),
                wall_ms = coalesce(wall_ms, :wall), cpu_ms = coalesce(cpu_ms, :cpu)
            WHERE task_id = :id AND attempt = :attempt AND output_sha IS NULL
              AND status <> 'RUNNING'
            """
        ),
        {
            "sha": output_sha,
            "id": task_id,
            "attempt": attempt,
            "wall": stats.wall_ms,
            "cpu": stats.cpu_ms,
        },
    )
    published = conn.execute(
        select(result_artifacts.c.output_sha).where(result_artifacts.c.task_id == task_id)
    ).scalar_one_or_none()
    match = None if published is None else published == output_sha
    if match is False:
        conn.execute(
            text("UPDATE tasks SET checksum_mismatch = true WHERE id = :id"), {"id": task_id}
        )
    return FinalizeResult(FinalizeOutcome.LEASE_LOST, published, match)


def finalize_success(
    engine: Engine,
    *,
    task_id: uuid.UUID | str,
    attempt: int,
    worker_id: str,
    artifact: ArtifactRecord,
    stats: AttemptStats,
    code_sha: str,
    env_sha: str,
    cached_from_task_id: uuid.UUID | None = None,
) -> FinalizeResult:
    """Publish the task's single result; artifact bytes must already be committed.

    Holds the task row lock, so it serialises against cancel, the reconciler and other attempts.
    """
    tid = _uuid(task_id)
    with transaction(engine) as conn:
        set_actor(conn, f"worker:{worker_id}", "result published")
        row = _lock_task(conn, tid)
        if not _owns(row, worker_id, attempt):
            return _record_late_completion(conn, tid, attempt, artifact.output_sha, stats)
        if row["cancel_requested"]:
            set_reason(conn, "cancel acknowledged at publish (result discarded)")
            _close_attempt(
                conn,
                tid,
                attempt,
                "CANCELED",
                stats=stats,
                output_sha=artifact.output_sha,
                error_code=None,
                message=None,
            )
            _task_to(conn, tid, TaskState.CANCELED, terminal=True)
            notify(conn)
            return FinalizeResult(FinalizeOutcome.CANCELED)
        inserted = conn.execute(
            text(
                """
                INSERT INTO result_artifacts (id, task_id, batch_id, attempt, spec_hash,
                    output_uri, output_sha, temp_uri, size_bytes, row_count, metrics,
                    code_sha, env_sha, cached_from_task_id)
                VALUES (:rid, :tid, :bid, :attempt, :spec_hash, :uri, :sha, :temp, :size,
                    :rows, CAST(:metrics AS jsonb), :code_sha, :env_sha, :cached_from)
                ON CONFLICT (task_id) DO NOTHING
                RETURNING id
                """
            ),
            {
                "rid": uuid.uuid4(),
                "tid": tid,
                "bid": row["batch_id"],
                "attempt": attempt,
                "spec_hash": row["spec_hash"],
                "uri": artifact.output_uri,
                "sha": artifact.output_sha,
                "temp": artifact.temp_uri,
                "size": artifact.size_bytes,
                "rows": artifact.row_count,
                "metrics": _json(dict(artifact.metrics)),
                "code_sha": code_sha,
                "env_sha": env_sha,
                "cached_from": cached_from_task_id,
            },
        ).first()
        if inserted is None:  # pragma: no cover - impossible while we hold the RUNNING lease
            raise Conflict(f"task {tid} already has a published result")
        status = "CACHED" if cached_from_task_id is not None else "SUCCEEDED"
        _close_attempt(
            conn,
            tid,
            attempt,
            status,
            stats=stats,
            output_sha=artifact.output_sha,
            error_code=None,
            message=None,
        )
        mismatched_earlier = conn.execute(
            text(
                "SELECT count(*) FROM task_attempts WHERE task_id = :id AND output_sha IS NOT NULL "
                "AND output_sha <> :sha"
            ),
            {"id": tid, "sha": artifact.output_sha},
        ).scalar_one()
        conn.execute(
            text(
                """
                UPDATE tasks SET state = 'SUCCEEDED', lease_owner = NULL, lease_expiry = NULL,
                    finished_at = now(), last_error_code = NULL, last_error_message = NULL,
                    cache_hit = :cache_hit,
                    checksum_mismatch = checksum_mismatch OR :mismatch
                WHERE id = :id
                """
            ),
            {
                "id": tid,
                "cache_hit": cached_from_task_id is not None,
                "mismatch": bool(mismatched_earlier),
            },
        )
        notify(conn)
    return FinalizeResult(FinalizeOutcome.PUBLISHED, artifact.output_sha, True)


def _close_attempt(
    conn: Connection,
    task_id: uuid.UUID,
    attempt: int,
    status: str,
    *,
    stats: AttemptStats | None,
    output_sha: str | None,
    error_code: str | None,
    message: str | None,
) -> None:
    stats = stats or AttemptStats()
    conn.execute(
        text(
            """
            UPDATE task_attempts
            SET status = :status, ended_at = now(), error_code = :code,
                error_message = :message, output_sha = coalesce(:sha, output_sha),
                wall_ms = coalesce(:wall, wall_ms), cpu_ms = coalesce(:cpu, cpu_ms),
                peak_rss_mb = coalesce(:rss, peak_rss_mb)
            WHERE task_id = :id AND attempt = :attempt AND status = 'RUNNING'
            """
        ),
        {
            "status": status,
            "code": error_code,
            "message": _truncate(message),
            "sha": output_sha,
            "wall": stats.wall_ms,
            "cpu": stats.cpu_ms,
            "rss": stats.peak_rss_mb,
            "id": task_id,
            "attempt": attempt,
        },
    )


def _task_to(
    conn: Connection,
    task_id: uuid.UUID,
    state: TaskState,
    *,
    terminal: bool,
    error_code: str | None = None,
    message: str | None = None,
    not_before_s: float | None = None,
) -> None:
    conn.execute(
        text(
            """
            UPDATE tasks SET state = :state, lease_owner = NULL, lease_expiry = NULL,
                enqueued_at = NULL,
                finished_at = CASE WHEN :terminal THEN now() ELSE NULL END,
                ready_at = CASE WHEN :terminal THEN ready_at ELSE now() END,
                not_before = CASE WHEN CAST(:nb AS double precision) IS NULL THEN NULL
                                  ELSE now() + make_interval(secs => CAST(:nb AS double precision))
                             END,
                last_error_code = coalesce(:code, last_error_code),
                last_error_message = coalesce(:message, last_error_message)
            WHERE id = :id
            """
        ),
        {
            "state": state.value,
            "terminal": terminal,
            "nb": not_before_s,
            "code": error_code,
            "message": _truncate(message),
            "id": task_id,
        },
    )


class FailureOutcome(StrEnum):
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    LEASE_LOST = "LEASE_LOST"


def record_failure(
    engine: Engine,
    *,
    task_id: uuid.UUID | str,
    attempt: int,
    worker_id: str,
    error_code: ErrorCode,
    message: str,
    retryable: bool,
    stats: AttemptStats | None = None,
    backoff_s: float = 0.0,
) -> FailureOutcome:
    """Close a failed attempt; retry if retryable and attempts remain, else fail the task.

    A worker that no longer owns the lease changes nothing.
    """
    tid = _uuid(task_id)
    with transaction(engine) as conn:
        set_actor(conn, f"worker:{worker_id}", f"attempt {attempt} failed: {error_code}")
        row = _lock_task(conn, tid)
        if not _owns(row, worker_id, attempt):
            return FailureOutcome.LEASE_LOST
        _close_attempt(
            conn,
            tid,
            attempt,
            "FAILED",
            stats=stats,
            output_sha=None,
            error_code=error_code.value,
            message=message,
        )
        if row["cancel_requested"]:
            _task_to(
                conn,
                tid,
                TaskState.CANCELED,
                terminal=True,
                error_code=error_code.value,
                message=message,
            )
            outcome = FailureOutcome.CANCELED
        elif retryable and row["attempts"] < row["max_attempts"]:
            _task_to(
                conn,
                tid,
                TaskState.RETRY_PENDING,
                terminal=False,
                error_code=error_code.value,
                message=message,
                not_before_s=backoff_s,
            )
            outcome = FailureOutcome.RETRY_SCHEDULED
        else:
            _task_to(
                conn,
                tid,
                TaskState.FAILED,
                terminal=True,
                error_code=error_code.value,
                message=message,
            )
            outcome = FailureOutcome.FAILED
        notify(conn)
    return outcome


def acknowledge_cancel(
    engine: Engine,
    *,
    task_id: uuid.UUID | str,
    attempt: int,
    worker_id: str,
    stats: AttemptStats | None = None,
) -> bool:
    """Worker observed ``cancel_requested`` and stopped cooperatively."""
    tid = _uuid(task_id)
    with transaction(engine) as conn:
        set_actor(conn, f"worker:{worker_id}", "cancel acknowledged")
        row = _lock_task(conn, tid)
        if not _owns(row, worker_id, attempt) or not row["cancel_requested"]:
            return False
        _close_attempt(
            conn,
            tid,
            attempt,
            "CANCELED",
            stats=stats,
            output_sha=None,
            error_code=None,
            message="canceled cooperatively",
        )
        _task_to(conn, tid, TaskState.CANCELED, terminal=True)
        notify(conn)
    return True


def release_lease(
    engine: Engine,
    *,
    task_id: uuid.UUID | str,
    attempt: int,
    worker_id: str,
    reason: str,
    stats: AttemptStats | None = None,
) -> FailureOutcome | None:
    """Give a lease back early on shutdown instead of waiting for expiry."""
    tid = _uuid(task_id)
    with transaction(engine) as conn:
        set_actor(conn, f"worker:{worker_id}", f"lease released: {reason}")
        row = _lock_task(conn, tid)
        if not _owns(row, worker_id, attempt):
            return None
        _close_attempt(
            conn,
            tid,
            attempt,
            "LEASE_EXPIRED",
            stats=stats,
            output_sha=None,
            error_code=ErrorCode.WORKER_LOST.value,
            message=f"lease released: {reason}",
        )
        outcome = _after_lost_attempt(conn, tid, row)
        notify(conn)
    return outcome


def _after_lost_attempt(conn: Connection, task_id: uuid.UUID, row: RowMapping) -> FailureOutcome:
    if row["cancel_requested"]:
        _task_to(conn, task_id, TaskState.CANCELED, terminal=True)
        return FailureOutcome.CANCELED
    if row["attempts"] < row["max_attempts"]:
        _task_to(
            conn,
            task_id,
            TaskState.RETRY_PENDING,
            terminal=False,
            error_code=ErrorCode.WORKER_LOST.value,
            message="worker lost: lease expired or released",
        )
        return FailureOutcome.RETRY_SCHEDULED
    _task_to(
        conn,
        task_id,
        TaskState.FAILED,
        terminal=True,
        error_code=ErrorCode.WORKER_LOST.value,
        message="worker lost on final attempt: lease expired or released",
    )
    return FailureOutcome.FAILED


@dataclass(frozen=True)
class ReclaimedLease:
    task_id: uuid.UUID
    batch_id: uuid.UUID
    attempt: int
    worker_id: str
    outcome: FailureOutcome
    expired_for_ms: int


def reconcile_expired_leases(engine: Engine, *, limit: int = 500) -> list[ReclaimedLease]:
    """Reclaim RUNNING tasks whose lease expired.

    The attempt closes as LEASE_EXPIRED (WORKER_LOST); the task goes to RETRY_PENDING,
    FAILED (attempts exhausted) or CANCELED (cancel pending). ``SKIP LOCKED`` skips
    rows a worker is finalising; they are picked up on the next tick.
    """
    reclaimed: list[ReclaimedLease] = []
    with transaction(engine) as conn:
        set_actor(conn, "coordinator", "lease expired")
        rows = (
            conn.execute(
                text(
                    """
                SELECT id, batch_id, state, attempts, max_attempts, lease_owner,
                       cancel_requested, spec_hash,
                       (EXTRACT(EPOCH FROM (now() - lease_expiry)) * 1000)::bigint AS expired_ms
                FROM tasks
                WHERE state = 'RUNNING' AND lease_expiry < now()
                ORDER BY lease_expiry
                LIMIT :limit
                FOR NO KEY UPDATE SKIP LOCKED
                """
                ),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        for row in rows:
            _close_attempt(
                conn,
                row["id"],
                row["attempts"],
                "LEASE_EXPIRED",
                stats=None,
                output_sha=None,
                error_code=ErrorCode.WORKER_LOST.value,
                message=f"lease expired: worker {row['lease_owner']} stopped heartbeating",
            )
            outcome = _after_lost_attempt(conn, row["id"], row)
            reclaimed.append(
                ReclaimedLease(
                    row["id"],
                    row["batch_id"],
                    row["attempts"],
                    row["lease_owner"],
                    outcome,
                    int(row["expired_ms"]),
                )
            )
        if reclaimed:
            notify(conn)
    return reclaimed


def mark_dead_workers(engine: Engine, *, dead_after_s: float) -> list[str]:
    with transaction(engine) as conn:
        rows = conn.execute(
            text(
                """
                UPDATE worker_heartbeats SET state = 'DEAD'
                WHERE state IN ('ACTIVE', 'STOPPING')
                  AND last_seen < now() - make_interval(secs => :s)
                RETURNING worker_id
                """
            ),
            {"s": dead_after_s},
        ).all()
    return [r.worker_id for r in rows]


def live_worker_capacity(engine: Engine, *, dead_after_s: float) -> tuple[int, int]:
    """(live worker count, summed concurrency) of workers seen within ``dead_after_s``."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT count(*) AS n, coalesce(sum(concurrency), 0) AS slots
                FROM worker_heartbeats
                WHERE state = 'ACTIVE' AND last_seen >= now() - make_interval(secs => :s)
                """
            ),
            {"s": dead_after_s},
        ).one()
    return int(row.n), int(row.slots)


@dataclass(frozen=True)
class BatchStateChange:
    batch_id: uuid.UUID
    old: str
    new: str


def refresh_batch_states(
    engine: Engine, *, batch_ids: Sequence[uuid.UUID] | None = None
) -> list[BatchStateChange]:
    """Recompute non-terminal batch states from task counts.

    Done periodically rather than per completion so workers never contend on the batch row.
    """
    changes: list[BatchStateChange] = []
    with transaction(engine) as conn:
        query = (
            "SELECT id, state, cancel_requested_at, started_at FROM batches "
            "WHERE state IN ('ACCEPTED', 'RUNNING', 'CANCELING')"
        )
        params: dict[str, Any] = {}
        if batch_ids is not None:
            query += " AND id = ANY(CAST(:ids AS uuid[]))"
            params["ids"] = list(batch_ids)
        active = conn.execute(
            text(query + " ORDER BY submitted_at FOR NO KEY UPDATE SKIP LOCKED"), params
        ).all()
        if not active:
            return []
        ids = [r.id for r in active]
        counts: dict[uuid.UUID, dict[TaskState, int]] = {i: {} for i in ids}
        first_start: dict[uuid.UUID, datetime | None] = {}
        for r in conn.execute(
            text(
                "SELECT batch_id, state, count(*) AS n, min(started_at) AS first_start "
                "FROM tasks WHERE batch_id = ANY(CAST(:ids AS uuid[])) GROUP BY batch_id, state"
            ),
            {"ids": ids},
        ):
            counts[r.batch_id][TaskState(r.state)] = int(r.n)
            if r.first_start is not None:
                prev = first_start.get(r.batch_id)
                first_start[r.batch_id] = (
                    r.first_start if prev is None else min(prev, r.first_start)
                )
        for b in active:
            current = BatchState(b.state)
            new = derive_batch_state(
                current, counts[b.id], cancel_requested=b.cancel_requested_at is not None
            )
            needs_start = b.started_at is None and first_start.get(b.id) is not None
            if new == current and not needs_start:
                continue
            conn.execute(
                text(
                    """
                    UPDATE batches SET state = :state,
                        started_at = coalesce(started_at, :started),
                        finished_at = CASE WHEN :terminal THEN coalesce(finished_at, now())
                                           ELSE NULL END
                    WHERE id = :id
                    """
                ),
                {
                    "state": new.value,
                    "started": first_start.get(b.id),
                    "terminal": new in TERMINAL_BATCH_STATES,
                    "id": b.id,
                },
            )
            if new != current:
                changes.append(BatchStateChange(b.id, current.value, new.value))
    return changes


@dataclass
class CancelReport:
    batch_id: uuid.UUID
    state: str
    already_terminal: bool
    canceled_now: int = 0
    cancel_requested_running: int = 0
    completed_count: int = 0
    failed_count: int = 0
    completed_task_ids: list[uuid.UUID] = field(default_factory=list)
    completed_ids_truncated: bool = False


def _lock_batch(conn: Connection, batch_id: uuid.UUID, user_id: str | None) -> RowMapping:
    query = "SELECT * FROM batches WHERE id = :id"
    params: dict[str, Any] = {"id": batch_id}
    if user_id is not None:
        query += " AND user_id = :u"
        params["u"] = user_id
    row = conn.execute(text(query + " FOR NO KEY UPDATE"), params).mappings().first()
    if row is None:
        raise NotFound(f"batch {batch_id} not found")
    return row


def _task_counts(conn: Connection, batch_id: uuid.UUID) -> dict[TaskState, int]:
    return {
        TaskState(r.state): int(r.n)
        for r in conn.execute(
            text("SELECT state, count(*) AS n FROM tasks WHERE batch_id = :id GROUP BY state"),
            {"id": batch_id},
        )
    }


def cancel_batch(
    engine: Engine, batch_id: uuid.UUID | str, *, user_id: str | None, actor: str = "api"
) -> CancelReport:
    """Cooperative cancellation.

    PENDING/RETRY_PENDING tasks become CANCELED; RUNNING tasks get ``cancel_requested``
    and stop at the next heartbeat or publish. Completed results are kept. Row locks make
    cancel and publish exclusive: a task is reported completed or can never publish.
    """
    bid = _uuid(batch_id)
    with transaction(engine) as conn:
        batch = _lock_batch(conn, bid, user_id)
        report = CancelReport(batch_id=bid, state=batch["state"], already_terminal=False)
        if BatchState(batch["state"]) in TERMINAL_BATCH_STATES:
            report.already_terminal = True
        else:
            set_actor(conn, actor, "batch canceled")
            conn.execute(
                text(
                    "UPDATE batches SET cancel_requested_at = "
                    "coalesce(cancel_requested_at, now()) WHERE id = :id"
                ),
                {"id": bid},
            )
            report.canceled_now = len(
                conn.execute(
                    text(
                        """
                        UPDATE tasks SET state = 'CANCELED', cancel_requested = true,
                            finished_at = now(), enqueued_at = NULL
                        WHERE batch_id = :id AND state IN ('PENDING', 'RETRY_PENDING')
                        RETURNING id
                        """
                    ),
                    {"id": bid},
                ).all()
            )
            report.cancel_requested_running = len(
                conn.execute(
                    text(
                        "UPDATE tasks SET cancel_requested = true "
                        "WHERE batch_id = :id AND state = 'RUNNING' RETURNING id"
                    ),
                    {"id": bid},
                ).all()
            )
        counts = _task_counts(conn, bid)
        report.completed_count = counts.get(TaskState.SUCCEEDED, 0)
        report.failed_count = counts.get(TaskState.FAILED, 0)
        ids = (
            conn.execute(
                text(
                    "SELECT id FROM tasks WHERE batch_id = :id AND state = 'SUCCEEDED' "
                    "ORDER BY ordinal LIMIT :lim"
                ),
                {"id": bid, "lim": MAX_COMPLETED_IDS_IN_REPORT + 1},
            )
            .scalars()
            .all()
        )
        report.completed_ids_truncated = len(ids) > MAX_COMPLETED_IDS_IN_REPORT
        report.completed_task_ids = list(ids[:MAX_COMPLETED_IDS_IN_REPORT])
        if not report.already_terminal:
            new = derive_batch_state(BatchState(batch["state"]), counts, cancel_requested=True)
            if new.value != batch["state"]:
                conn.execute(
                    text(
                        "UPDATE batches SET state = :s, finished_at = CASE WHEN :t "
                        "THEN now() ELSE finished_at END WHERE id = :id"
                    ),
                    {"s": new.value, "t": new in TERMINAL_BATCH_STATES, "id": bid},
                )
            report.state = new.value
            notify(conn)
    return report


@dataclass
class RetryReport:
    batch_id: uuid.UUID
    state: str
    retried: int
    retried_task_ids: list[uuid.UUID]
    skipped_not_retryable: int
    skipped_attempt_quota: int


def retry_failed(
    engine: Engine,
    batch_id: uuid.UUID | str,
    *,
    user_id: str | None,
    extra_attempts: int = 1,
    include_non_retryable: bool = False,
    actor: str = "api",
) -> RetryReport:
    """Move eligible FAILED tasks to RETRY_PENDING with ``extra_attempts`` more attempts.

    Eligible: retryable last error (any, with ``include_non_retryable``) and within the
    batch's quota snapshot (attempts and worker-seconds).
    """
    if extra_attempts < 1:
        raise ValueError("extra_attempts must be >= 1")
    bid = _uuid(batch_id)
    with transaction(engine) as conn:
        batch = _lock_batch(conn, bid, user_id)
        if batch["cancel_requested_at"] is not None:
            raise Conflict("batch was canceled; failed tasks cannot be retried")
        quota = QuotaPolicy.model_validate(batch["quota"])
        failed = conn.execute(
            text(
                "SELECT id, attempts, last_error_code, max_runtime_s FROM tasks "
                "WHERE batch_id = :id AND state = 'FAILED' ORDER BY ordinal FOR NO KEY UPDATE"
            ),
            {"id": bid},
        ).all()
        retryable_codes = {c.value for c in RETRYABLE_TASK_ERRORS}
        eligible, skipped_code, skipped_quota = [], 0, 0
        for r in failed:
            if not include_non_retryable and r.last_error_code not in retryable_codes:
                skipped_code += 1
            elif r.attempts + extra_attempts > quota.max_attempts:
                skipped_quota += 1
            else:
                eligible.append(r)
        if eligible:
            consumed_ms = conn.execute(
                text(
                    "SELECT coalesce(sum(wall_ms), 0)::double precision "
                    "FROM task_attempts WHERE batch_id = :id"
                ),
                {"id": bid},
            ).scalar_one()
            projected_s = sum(r.max_runtime_s * extra_attempts for r in eligible)
            if consumed_ms / 1000.0 + projected_s > quota.max_worker_seconds_per_batch:
                raise QuotaExceeded(
                    "retrying would exceed the batch worker-time quota",
                    details={
                        "consumed_s": round(consumed_ms / 1000.0, 3),
                        "projected_s": projected_s,
                        "quota_s": quota.max_worker_seconds_per_batch,
                    },
                )
            set_actor(conn, actor, "retry-failed")
            conn.execute(
                text(
                    """
                    UPDATE tasks SET state = 'RETRY_PENDING', max_attempts = attempts + :extra,
                        ready_at = now(), not_before = NULL, enqueued_at = NULL, finished_at = NULL
                    WHERE id = ANY(CAST(:ids AS uuid[])) AND state = 'FAILED'
                    """
                ),
                {"extra": extra_attempts, "ids": [r.id for r in eligible]},
            )
            state = batch["state"]
            if state == BatchState.COMPLETED_WITH_FAILURES:
                conn.execute(
                    text("UPDATE batches SET state = 'RUNNING', finished_at = NULL WHERE id = :id"),
                    {"id": bid},
                )
                state = BatchState.RUNNING.value
            notify(conn)
        else:
            state = batch["state"]
    return RetryReport(
        bid, state, len(eligible), [r.id for r in eligible], skipped_code, skipped_quota
    )


def get_batch(
    engine: Engine, batch_id: uuid.UUID | str, *, user_id: str | None = None
) -> dict[str, Any]:
    with engine.connect() as conn:
        query = "SELECT * FROM batches WHERE id = :id"
        params: dict[str, Any] = {"id": _uuid(batch_id)}
        if user_id is not None:
            query += " AND user_id = :u"
            params["u"] = user_id
        row = conn.execute(text(query), params).mappings().first()
    if row is None:
        raise NotFound(f"batch {batch_id} not found")
    return dict(row)


def batch_progress(engine: Engine, batch_id: uuid.UUID | str) -> dict[str, Any]:
    """Counts per state, attempt/retry totals, cost (worker-seconds) and timing."""
    bid = _uuid(batch_id)
    with engine.connect() as conn:
        counts = {s.value: 0 for s in TaskState}
        for r in conn.execute(
            text("SELECT state, count(*) AS n FROM tasks WHERE batch_id = :id GROUP BY state"),
            {"id": bid},
        ):
            counts[r.state] = int(r.n)
        agg = (
            conn.execute(
                text(
                    """
                SELECT count(*) AS attempts,
                       count(*) FILTER (WHERE attempt > 1) AS retries,
                       count(*) FILTER (WHERE status = 'LEASE_EXPIRED') AS lease_expired,
                       count(*) FILTER (WHERE status = 'FAILED') AS failed_attempts,
                       count(*) FILTER (WHERE status = 'CACHED') AS cache_hits,
                       count(*) FILTER (WHERE late_completion_at IS NOT NULL) AS late_completions,
                       coalesce(sum(wall_ms), 0) AS wall_ms,
                       coalesce(sum(cpu_ms), 0) AS cpu_ms
                FROM task_attempts WHERE batch_id = :id
                """
                ),
                {"id": bid},
            )
            .mappings()
            .one()
        )
        results = conn.execute(
            text("SELECT count(*) FROM result_artifacts WHERE batch_id = :id"), {"id": bid}
        ).scalar_one()
        mismatches = conn.execute(
            text("SELECT count(*) FROM tasks WHERE batch_id = :id AND checksum_mismatch"),
            {"id": bid},
        ).scalar_one()
    total = sum(counts.values())
    done = counts["SUCCEEDED"] + counts["FAILED"] + counts["CANCELED"]
    return {
        "counts": counts,
        "total": total,
        "terminal": done,
        "progress": (done / total) if total else 0.0,
        "attempts": int(agg["attempts"]),
        "retries": int(agg["retries"]),
        "lease_expirations": int(agg["lease_expired"]),
        "failed_attempts": int(agg["failed_attempts"]),
        "cache_hits": int(agg["cache_hits"]),
        "late_completions": int(agg["late_completions"]),
        "published_results": int(results),
        "checksum_mismatches": int(mismatches),
        "cost": {
            "worker_seconds": round(int(agg["wall_ms"]) / 1000.0, 3),
            "cpu_seconds": round(int(agg["cpu_ms"]) / 1000.0, 3),
        },
    }


def get_task_detail(
    engine: Engine, task_id: uuid.UUID | str, *, user_id: str | None = None
) -> dict[str, Any]:
    tid = _uuid(task_id)
    with engine.connect() as conn:
        query = "SELECT t.* FROM tasks t JOIN batches b ON b.id = t.batch_id WHERE t.id = :id"
        params: dict[str, Any] = {"id": tid}
        if user_id is not None:
            query += " AND b.user_id = :u"
            params["u"] = user_id
        task = conn.execute(text(query), params).mappings().first()
        if task is None:
            raise NotFound(f"task {task_id} not found")
        attempts = (
            conn.execute(
                select(task_attempts)
                .where(task_attempts.c.task_id == tid)
                .order_by(task_attempts.c.attempt)
            )
            .mappings()
            .all()
        )
        artifact = (
            conn.execute(select(result_artifacts).where(result_artifacts.c.task_id == tid))
            .mappings()
            .first()
        )
        transitions = (
            conn.execute(
                text(
                    "SELECT from_state, to_state, attempt, actor, reason, at FROM task_transitions "
                    "WHERE task_id = :id ORDER BY id"
                ),
                {"id": tid},
            )
            .mappings()
            .all()
        )
    return {
        "task": dict(task),
        "attempts": [dict(a) for a in attempts],
        "artifact": dict(artifact) if artifact else None,
        "transitions": [dict(t) for t in transitions],
    }


def list_tasks(
    engine: Engine,
    batch_id: uuid.UUID | str,
    *,
    state: str | None = None,
    after_ordinal: int = -1,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, ordinal, state, attempts, max_attempts, spec_hash, lease_owner, "
        "last_error_code, cache_hit, checksum_mismatch, started_at, finished_at "
        "FROM tasks WHERE batch_id = :id AND ordinal > :after"
    )
    params: dict[str, Any] = {"id": _uuid(batch_id), "after": after_ordinal, "lim": limit}
    if state is not None:
        query += " AND state = :state"
        params["state"] = state
    with engine.connect() as conn:
        rows = conn.execute(text(query + " ORDER BY ordinal LIMIT :lim"), params).mappings().all()
    return [dict(r) for r in rows]


def list_results(
    engine: Engine,
    batch_id: uuid.UUID | str,
    *,
    after_ordinal: int = -1,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Published results joined with their immutable task specs, ordered by ordinal."""
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT t.id AS task_id, t.ordinal, t.spec_hash, t.spec, r.output_uri, r.output_sha,
                       r.size_bytes, r.row_count, r.metrics, r.attempt, r.finalized_at,
                       r.cached_from_task_id, r.code_sha, r.env_sha
                FROM result_artifacts r JOIN tasks t ON t.id = r.task_id
                WHERE r.batch_id = :id AND t.ordinal > :after
                ORDER BY t.ordinal
                LIMIT :lim
                """
                ),
                {"id": _uuid(batch_id), "after": after_ordinal, "lim": limit},
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def list_batches(
    engine: Engine, *, user_id: str, before: datetime | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    query = (
        "SELECT id, name, state, task_count, submitted_at, started_at, finished_at, spec_sha "
        "FROM batches WHERE user_id = :u"
    )
    params: dict[str, Any] = {"u": user_id, "lim": limit}
    if before is not None:
        query += " AND submitted_at < :before"
        params["before"] = before
    with engine.connect() as conn:
        rows = (
            conn.execute(text(query + " ORDER BY submitted_at DESC LIMIT :lim"), params)
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def find_cached_result(
    engine: Engine, *, spec_hash: str, code_sha: str, env_sha: str
) -> dict[str, Any] | None:
    """A published result for the same spec, code and environment, safe to reuse."""
    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    """
                SELECT task_id, output_uri, output_sha, size_bytes, row_count, metrics
                FROM result_artifacts
                WHERE spec_hash = :h AND code_sha = :c AND env_sha = :e
                ORDER BY finalized_at, task_id
                LIMIT 1
                """
                ),
                {"h": spec_hash, "c": code_sha, "e": env_sha},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def failed_tasks(
    engine: Engine, batch_id: uuid.UUID | str, *, limit: int = 50
) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT id, ordinal, attempts, last_error_code, last_error_message, spec_hash
                FROM tasks WHERE batch_id = :id AND state = 'FAILED'
                ORDER BY ordinal LIMIT :lim
                """
                ),
                {"id": _uuid(batch_id), "lim": limit},
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


def attempt_rows(engine: Engine, batch_id: uuid.UUID | str) -> list[dict[str, Any]]:
    """Raw attempt rows for benchmarking/verification."""
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(task_attempts)
                .where(task_attempts.c.batch_id == _uuid(batch_id))
                .order_by(task_attempts.c.id)
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]
