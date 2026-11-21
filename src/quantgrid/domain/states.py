"""Task, attempt and batch state machines.

TASK_TRANSITIONS is the source of truth. The PostgreSQL trigger from migration ``0001``
enforces the same table, so the two must stay in sync (checked by
``tests/integration/test_schema_triggers.py``).

Task lifecycle::

    PENDING ──claim──▶ RUNNING ──publish──▶ SUCCEEDED (terminal, immutable)
       │                 │ ├─error/lease expiry, attempts left─▶ RETRY_PENDING ─claim─▶ RUNNING
       │                 │ └─permanent error/no attempts left─▶ FAILED ─retry-failed─▶ RETRY_PENDING
       └─────cancel──────┴──────────────────cancel──────────────────▶ CANCELED (terminal, immutable)
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum


class TaskState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRY_PENDING = "RETRY_PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TASK_TRANSITIONS: Mapping[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.RUNNING, TaskState.CANCELED}),
    TaskState.RUNNING: frozenset(
        {TaskState.SUCCEEDED, TaskState.RETRY_PENDING, TaskState.FAILED, TaskState.CANCELED}
    ),
    TaskState.RETRY_PENDING: frozenset({TaskState.RUNNING, TaskState.CANCELED}),
    TaskState.FAILED: frozenset({TaskState.RETRY_PENDING}),  # only via explicit retry-failed
    TaskState.SUCCEEDED: frozenset(),
    TaskState.CANCELED: frozenset(),
}

CLAIMABLE_TASK_STATES: frozenset[TaskState] = frozenset(
    {TaskState.PENDING, TaskState.RETRY_PENDING}
)
ACTIVE_TASK_STATES: frozenset[TaskState] = frozenset(
    {TaskState.PENDING, TaskState.RUNNING, TaskState.RETRY_PENDING}
)
TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELED}
)
#: Terminal states that can never be left again (FAILED can, via retry-failed).
IMMUTABLE_TASK_STATES: frozenset[TaskState] = frozenset({TaskState.SUCCEEDED, TaskState.CANCELED})


class IllegalTransition(ValueError):
    pass


def can_transition(current: TaskState, target: TaskState) -> bool:
    return target in TASK_TRANSITIONS[current]


def assert_transition(current: TaskState, target: TaskState) -> None:
    if not can_transition(current, target):
        raise IllegalTransition(f"illegal task transition {current} -> {target}")


class AttemptStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"  # this attempt published the task's result
    CACHED = "CACHED"  # result reused from an identical spec (same code + env)
    FAILED = "FAILED"
    LEASE_EXPIRED = "LEASE_EXPIRED"  # closed by the coordinator; worker presumed lost
    CANCELED = "CANCELED"


ATTEMPT_TRANSITIONS: Mapping[AttemptStatus, frozenset[AttemptStatus]] = {
    AttemptStatus.RUNNING: frozenset(
        {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.CACHED,
            AttemptStatus.FAILED,
            AttemptStatus.LEASE_EXPIRED,
            AttemptStatus.CANCELED,
        }
    ),
    AttemptStatus.SUCCEEDED: frozenset(),
    AttemptStatus.CACHED: frozenset(),
    AttemptStatus.FAILED: frozenset(),
    AttemptStatus.LEASE_EXPIRED: frozenset(),
    AttemptStatus.CANCELED: frozenset(),
}


class BatchState(StrEnum):
    ACCEPTED = "ACCEPTED"  # persisted, no task claimed yet
    RUNNING = "RUNNING"
    CANCELING = "CANCELING"  # cancel requested, some tasks still running
    COMPLETED = "COMPLETED"  # every task SUCCEEDED
    COMPLETED_WITH_FAILURES = "COMPLETED_WITH_FAILURES"  # all terminal, >=1 FAILED
    CANCELED = "CANCELED"


BATCH_TRANSITIONS: Mapping[BatchState, frozenset[BatchState]] = {
    BatchState.ACCEPTED: frozenset(
        {
            BatchState.RUNNING,
            BatchState.CANCELING,
            BatchState.CANCELED,
            BatchState.COMPLETED,
            BatchState.COMPLETED_WITH_FAILURES,
        }
    ),
    BatchState.RUNNING: frozenset(
        {
            BatchState.CANCELING,
            BatchState.CANCELED,
            BatchState.COMPLETED,
            BatchState.COMPLETED_WITH_FAILURES,
        }
    ),
    BatchState.CANCELING: frozenset({BatchState.CANCELED}),
    BatchState.COMPLETED_WITH_FAILURES: frozenset({BatchState.RUNNING}),  # retry-failed
    BatchState.COMPLETED: frozenset(),
    BatchState.CANCELED: frozenset(),
}

TERMINAL_BATCH_STATES: frozenset[BatchState] = frozenset(
    {BatchState.COMPLETED, BatchState.COMPLETED_WITH_FAILURES, BatchState.CANCELED}
)


def derive_batch_state(
    current: BatchState,
    counts: Mapping[TaskState, int],
    *,
    cancel_requested: bool,
) -> BatchState:
    """Pure function: the batch state implied by its task counts.

    Returns ``current`` when nothing changed. Never returns an illegal successor of
    ``current`` (asserted by property tests).
    """
    pending = counts.get(TaskState.PENDING, 0)
    running = counts.get(TaskState.RUNNING, 0)
    retry_pending = counts.get(TaskState.RETRY_PENDING, 0)
    succeeded = counts.get(TaskState.SUCCEEDED, 0)
    failed = counts.get(TaskState.FAILED, 0)
    canceled = counts.get(TaskState.CANCELED, 0)
    total = pending + running + retry_pending + succeeded + failed + canceled
    active = pending + running + retry_pending

    if current in (BatchState.COMPLETED, BatchState.CANCELED):
        return current  # immutable terminal states

    if cancel_requested:
        target = BatchState.CANCELED if active == 0 else BatchState.CANCELING
    elif active == 0 and total > 0:
        target = BatchState.COMPLETED if succeeded == total else BatchState.COMPLETED_WITH_FAILURES
    elif (
        current == BatchState.ACCEPTED and running == 0 and retry_pending == 0 and pending == total
    ):
        target = BatchState.ACCEPTED
    else:
        target = BatchState.RUNNING

    if target == current or target in BATCH_TRANSITIONS[current]:
        return target
    # Defensive: e.g. CANCELING can only move to CANCELED. Stay put rather than jump illegally.
    return current
