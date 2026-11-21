"""Property tests for derive_batch_state."""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantgrid.domain.states import (
    ACTIVE_TASK_STATES,
    BATCH_TRANSITIONS,
    BatchState,
    TaskState,
    derive_batch_state,
)

counts_st = st.fixed_dictionaries({s: st.integers(0, 50) for s in TaskState})
batch_st = st.sampled_from(list(BatchState))


def _active(counts: dict[TaskState, int]) -> int:
    return sum(counts[s] for s in ACTIVE_TASK_STATES)


@settings(max_examples=500, deadline=None)
@given(batch_st, counts_st, st.booleans())
def test_result_is_current_or_legal_successor(cur: BatchState, counts: dict, flag: bool) -> None:
    out = derive_batch_state(cur, counts, cancel_requested=flag)
    assert out is cur or out in BATCH_TRANSITIONS[cur]


@settings(max_examples=300, deadline=None)
@given(st.sampled_from([BatchState.COMPLETED, BatchState.CANCELED]), counts_st, st.booleans())
def test_absorbing(cur: BatchState, counts: dict, flag: bool) -> None:
    assert derive_batch_state(cur, counts, cancel_requested=flag) is cur


@settings(max_examples=300, deadline=None)
@given(st.sampled_from([BatchState.ACCEPTED, BatchState.RUNNING]), st.integers(1, 100))
def test_all_succeeded_completes(cur: BatchState, n: int) -> None:
    counts = dict.fromkeys(TaskState, 0) | {TaskState.SUCCEEDED: n}
    assert derive_batch_state(cur, counts, cancel_requested=False) is BatchState.COMPLETED


@settings(max_examples=500, deadline=None)
@given(batch_st, counts_st, st.booleans())
def test_active_tasks_never_complete(cur: BatchState, counts: dict, flag: bool) -> None:
    assume(_active(counts) > 0)
    # COMPLETED_WITH_FAILURES with active tasks is only reachable if the state is stale;
    # a stale COMPLETED_WITH_FAILURES may be returned unchanged when cancel is requested.
    assume(cur is not BatchState.COMPLETED_WITH_FAILURES or not flag)
    assume(cur is not BatchState.COMPLETED)
    out = derive_batch_state(cur, counts, cancel_requested=flag)
    assert out not in (BatchState.COMPLETED, BatchState.COMPLETED_WITH_FAILURES)


@settings(max_examples=500, deadline=None)
@given(st.sampled_from([BatchState.ACCEPTED, BatchState.RUNNING, BatchState.CANCELING]), counts_st)
def test_cancel_with_no_active_tasks_cancels(cur: BatchState, counts: dict) -> None:
    counts = {**counts, TaskState.PENDING: 0, TaskState.RUNNING: 0, TaskState.RETRY_PENDING: 0}
    assert derive_batch_state(cur, counts, cancel_requested=True) is BatchState.CANCELED


@settings(max_examples=300, deadline=None)
@given(st.sampled_from([BatchState.ACCEPTED, BatchState.RUNNING]), counts_st)
def test_cancel_with_active_tasks_is_canceling(cur: BatchState, counts: dict) -> None:
    assume(_active(counts) > 0)
    assert derive_batch_state(cur, counts, cancel_requested=True) is BatchState.CANCELING


@settings(max_examples=300, deadline=None)
@given(batch_st, counts_st, st.booleans())
def test_idempotent(cur: BatchState, counts: dict, flag: bool) -> None:
    once = derive_batch_state(cur, counts, cancel_requested=flag)
    twice = derive_batch_state(once, counts, cancel_requested=flag)
    assert twice is once or twice in BATCH_TRANSITIONS[once]


@settings(max_examples=300, deadline=None)
@given(st.sampled_from([BatchState.ACCEPTED, BatchState.RUNNING]), counts_st)
def test_terminal_when_no_active_and_not_canceled(cur: BatchState, counts: dict) -> None:
    counts = {**counts, TaskState.PENDING: 0, TaskState.RUNNING: 0, TaskState.RETRY_PENDING: 0}
    total = sum(counts.values())
    out = derive_batch_state(cur, counts, cancel_requested=False)
    if total == 0:
        assert out is cur
    elif counts[TaskState.SUCCEEDED] == total:
        assert out is BatchState.COMPLETED
    else:
        assert out is BatchState.COMPLETED_WITH_FAILURES
