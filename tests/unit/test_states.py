"""Unit tests for the task/attempt/batch state machines."""

from __future__ import annotations

import itertools

import pytest

from quantgrid.domain.states import (
    ACTIVE_TASK_STATES,
    ATTEMPT_TRANSITIONS,
    BATCH_TRANSITIONS,
    CLAIMABLE_TASK_STATES,
    IMMUTABLE_TASK_STATES,
    TASK_TRANSITIONS,
    TERMINAL_BATCH_STATES,
    TERMINAL_TASK_STATES,
    AttemptStatus,
    BatchState,
    IllegalTransition,
    TaskState,
    assert_transition,
    can_transition,
    derive_batch_state,
)

T = TaskState
B = BatchState


def test_tables_closed_over_enums() -> None:
    for table, enum in (
        (TASK_TRANSITIONS, TaskState),
        (ATTEMPT_TRANSITIONS, AttemptStatus),
        (BATCH_TRANSITIONS, BatchState),
    ):
        assert set(table) == set(enum)
        for targets in table.values():
            assert targets <= set(enum)


def test_no_self_loops() -> None:
    for table in (TASK_TRANSITIONS, ATTEMPT_TRANSITIONS, BATCH_TRANSITIONS):
        for state, targets in table.items():
            assert state not in targets


def test_terminal_task_states_have_no_exits_except_failed_retry() -> None:
    for state in TERMINAL_TASK_STATES:
        if state == T.FAILED:
            assert TASK_TRANSITIONS[state] == {T.RETRY_PENDING}
        else:
            assert TASK_TRANSITIONS[state] == frozenset()
    assert {T.SUCCEEDED, T.CANCELED} == IMMUTABLE_TASK_STATES
    assert IMMUTABLE_TASK_STATES | {T.FAILED} == TERMINAL_TASK_STATES


def test_active_and_terminal_partition_task_states() -> None:
    assert set(T) == ACTIVE_TASK_STATES | TERMINAL_TASK_STATES
    assert not ACTIVE_TASK_STATES & TERMINAL_TASK_STATES
    assert CLAIMABLE_TASK_STATES <= ACTIVE_TASK_STATES
    assert T.RUNNING not in CLAIMABLE_TASK_STATES


def test_every_state_reachable_from_pending() -> None:
    seen = {T.PENDING}
    frontier = [T.PENDING]
    while frontier:
        for nxt in TASK_TRANSITIONS[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(T)


@pytest.mark.parametrize(
    ("src", "dst", "ok"),
    [
        (T.PENDING, T.RUNNING, True),
        (T.PENDING, T.SUCCEEDED, False),
        (T.PENDING, T.FAILED, False),
        (T.RUNNING, T.PENDING, False),
        (T.RUNNING, T.RETRY_PENDING, True),
        (T.RETRY_PENDING, T.RUNNING, True),
        (T.RETRY_PENDING, T.SUCCEEDED, False),
        (T.FAILED, T.RETRY_PENDING, True),
        (T.FAILED, T.RUNNING, False),
        (T.SUCCEEDED, T.RUNNING, False),
        (T.SUCCEEDED, T.CANCELED, False),
        (T.CANCELED, T.PENDING, False),
    ],
)
def test_can_and_assert_transition(src: TaskState, dst: TaskState, ok: bool) -> None:
    assert can_transition(src, dst) is ok
    if ok:
        assert_transition(src, dst)
    else:
        with pytest.raises(IllegalTransition, match=src.value):
            assert_transition(src, dst)


def test_illegal_transition_is_value_error() -> None:
    assert issubclass(IllegalTransition, ValueError)


def test_attempt_transitions_only_from_running() -> None:
    assert ATTEMPT_TRANSITIONS[AttemptStatus.RUNNING] == set(AttemptStatus) - {
        AttemptStatus.RUNNING
    }
    for status in AttemptStatus:
        if status is not AttemptStatus.RUNNING:
            assert not ATTEMPT_TRANSITIONS[status]


def test_batch_terminal_states() -> None:
    assert {B.COMPLETED, B.COMPLETED_WITH_FAILURES, B.CANCELED} == TERMINAL_BATCH_STATES
    assert BATCH_TRANSITIONS[B.COMPLETED] == frozenset()
    assert BATCH_TRANSITIONS[B.CANCELED] == frozenset()
    assert BATCH_TRANSITIONS[B.COMPLETED_WITH_FAILURES] == {B.RUNNING}
    assert BATCH_TRANSITIONS[B.CANCELING] == {B.CANCELED}


# ---- derive_batch_state examples ------------------------------------------------------


def test_all_succeeded_completes() -> None:
    assert derive_batch_state(B.RUNNING, {T.SUCCEEDED: 5}, cancel_requested=False) is B.COMPLETED


def test_failures_complete_with_failures() -> None:
    counts = {T.SUCCEEDED: 3, T.FAILED: 1}
    assert (
        derive_batch_state(B.RUNNING, counts, cancel_requested=False) is B.COMPLETED_WITH_FAILURES
    )


def test_accepted_stays_until_a_task_runs() -> None:
    assert derive_batch_state(B.ACCEPTED, {T.PENDING: 4}, cancel_requested=False) is B.ACCEPTED
    assert (
        derive_batch_state(B.ACCEPTED, {T.PENDING: 3, T.RUNNING: 1}, cancel_requested=False)
        is B.RUNNING
    )
    assert (
        derive_batch_state(B.ACCEPTED, {T.PENDING: 3, T.SUCCEEDED: 1}, cancel_requested=False)
        is B.RUNNING
    )


def test_empty_counts_do_not_complete() -> None:
    assert derive_batch_state(B.ACCEPTED, {}, cancel_requested=False) is B.ACCEPTED
    assert derive_batch_state(B.RUNNING, {}, cancel_requested=False) is B.RUNNING


def test_cancel_flow() -> None:
    assert (
        derive_batch_state(B.RUNNING, {T.RUNNING: 1, T.PENDING: 2}, cancel_requested=True)
        is B.CANCELING
    )
    assert derive_batch_state(B.CANCELING, {T.RUNNING: 1}, cancel_requested=True) is B.CANCELING
    assert (
        derive_batch_state(B.CANCELING, {T.CANCELED: 2, T.SUCCEEDED: 1}, cancel_requested=True)
        is B.CANCELED
    )
    assert derive_batch_state(B.ACCEPTED, {T.CANCELED: 2}, cancel_requested=True) is B.CANCELED


def test_retry_failed_reopens_batch() -> None:
    counts = {T.SUCCEEDED: 2, T.RETRY_PENDING: 1}
    assert (
        derive_batch_state(B.COMPLETED_WITH_FAILURES, counts, cancel_requested=False) is B.RUNNING
    )


def test_absorbing_states() -> None:
    for state in (B.COMPLETED, B.CANCELED):
        for flag in (True, False):
            assert derive_batch_state(state, {T.PENDING: 3}, cancel_requested=flag) is state


def test_canceling_never_completes_directly() -> None:
    result = derive_batch_state(B.CANCELING, {T.SUCCEEDED: 3}, cancel_requested=False)
    assert result in {B.CANCELING, B.CANCELED}


def test_exhaustive_small_counts_legal() -> None:
    """Exhaustively check legality over all small count vectors."""
    for vec in itertools.product(range(3), repeat=len(T)):
        counts = dict(zip(T, vec, strict=True))
        for cur in B:
            for flag in (False, True):
                out = derive_batch_state(cur, counts, cancel_requested=flag)
                assert out is cur or out in BATCH_TRANSITIONS[cur]
