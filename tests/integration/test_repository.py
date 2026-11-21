"""Scheduler repository against a real PostgreSQL: leases, fencing, idempotency, races."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, text

from quantgrid.domain.errors import Conflict, ErrorCode, NotFound, QuotaExceeded
from quantgrid.domain.policy import QuotaPolicy
from quantgrid.storage import repository as repo
from quantgrid.storage.repository import (
    AttemptStats,
    ClaimOutcome,
    FailureOutcome,
    FinalizeOutcome,
    LeaseStatus,
)

from .factories import artifact, make_spec, submit, task_ids

pytestmark = pytest.mark.integration

CODE, ENV = "c" * 64, "e" * 64


def worker(wid: str = "w1") -> repo.WorkerInfo:
    return repo.WorkerInfo(
        worker_id=wid,
        hostname="host",
        pid=1,
        started_at=datetime.now(UTC),
        code_sha=CODE,
        env_sha=ENV,
    )


def claim(engine: Engine, tid: uuid.UUID, wid: str = "w1", ttl: float = 30.0) -> repo.ClaimedTask:
    outcome, claimed = repo.claim_task(
        engine, tid, wid, lease_ttl_s=ttl, code_sha=CODE, env_sha=ENV
    )
    assert outcome == ClaimOutcome.CLAIMED, outcome
    assert claimed is not None
    return claimed


def publish(
    engine: Engine, c: repo.ClaimedTask, wid: str = "w1", tag: str = "r"
) -> repo.FinalizeResult:
    return repo.finalize_success(
        engine,
        task_id=c.task_id,
        attempt=c.attempt,
        worker_id=wid,
        artifact=artifact(tag),
        stats=AttemptStats(wall_ms=5, cpu_ms=4),
        code_sha=CODE,
        env_sha=ENV,
    )


def task_row(engine: Engine, tid: uuid.UUID) -> dict:
    return repo.get_task_detail(engine, tid)["task"]


# ----------------------------------------------------------------------------- creation


def test_create_batch_persists_tasks_in_order(clean_db: Engine) -> None:
    res = submit(clean_db, 5)
    assert res.created and res.task_count == 5
    tasks = repo.list_tasks(clean_db, res.batch_id)
    assert [t["ordinal"] for t in tasks] == [0, 1, 2, 3, 4]
    assert all(t["state"] == "PENDING" for t in tasks)
    assert len({t["spec_hash"] for t in tasks}) == 5
    batch = repo.get_batch(clean_db, res.batch_id)
    assert batch["state"] == "ACCEPTED" and batch["manifest"]["code"]["code_sha"]


def test_idempotent_resubmission_returns_same_batch(clean_db: Engine) -> None:
    first = submit(clean_db, 3, key="k-1")
    again = submit(clean_db, 3, key="k-1")
    assert again.batch_id == first.batch_id and not again.created
    with clean_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM batches")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM tasks")).scalar_one() == 3


def test_idempotency_key_reuse_with_different_spec_conflicts(clean_db: Engine) -> None:
    submit(clean_db, 3, key="k-2")
    with pytest.raises(Conflict):
        submit(clean_db, 4, key="k-2")
    # same key for a different user is a different logical batch
    other = submit(clean_db, 4, key="k-2", user_id="user-2")
    assert other.created


def test_concurrent_identical_submissions_create_one_batch(clean_db: Engine) -> None:
    spec = make_spec(6)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: submit(clean_db, spec=spec, key="race"), range(8)))
    assert len({r.batch_id for r in results}) == 1
    assert sum(r.created for r in results) == 1
    with clean_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM tasks")).scalar_one() == 6


# ----------------------------------------------------------------------------- dispatch


def test_dispatch_respects_window_and_redelivers_lost_messages(clean_db: Engine) -> None:
    res = submit(clean_db, 10)
    first = repo.dispatch_ready(clean_db, window=4, lost_after_s=60)
    assert len(first) == 4 and not any(i.redelivery for i in first)
    assert repo.dispatch_ready(clean_db, window=4, lost_after_s=60) == []
    # claiming one frees one slot
    claim(clean_db, first[0].task_id)
    nxt = repo.dispatch_ready(clean_db, window=4, lost_after_s=60)
    assert len(nxt) == 1 and nxt[0].task_id not in {i.task_id for i in first}
    # messages presumed lost after lost_after_s are re-dispatched (flagged)
    time.sleep(1.2)
    again = repo.dispatch_ready(clean_db, window=100, lost_after_s=1.0)
    redelivered = {i.task_id for i in again if i.redelivery}
    assert {i.task_id for i in first[1:]} | {nxt[0].task_id} <= redelivered
    assert all(i.batch_id == res.batch_id for i in again)


def test_concurrent_dispatchers_never_pick_the_same_task(clean_db: Engine) -> None:
    submit(clean_db, 40)
    with ThreadPoolExecutor(6) as pool:
        batches = list(
            pool.map(
                lambda _: repo.dispatch_ready(clean_db, window=100, lost_after_s=60, limit=10),
                range(6),
            )
        )
    picked = [i.task_id for b in batches for i in b]
    assert len(picked) == len(set(picked)) == 40


# ----------------------------------------------------------------------------- claim


def test_exactly_one_concurrent_claim_wins(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    barrier = threading.Barrier(10)

    def try_claim(i: int) -> ClaimOutcome:
        barrier.wait()
        return repo.claim_task(clean_db, tid, f"w{i}", lease_ttl_s=30)[0]

    with ThreadPoolExecutor(10) as pool:
        outcomes = list(pool.map(try_claim, range(10)))
    assert outcomes.count(ClaimOutcome.CLAIMED) == 1
    assert set(outcomes) <= {ClaimOutcome.CLAIMED, ClaimOutcome.ALREADY_RUNNING}
    detail = repo.get_task_detail(clean_db, tid)
    assert len(detail["attempts"]) == 1 and detail["task"]["attempts"] == 1


def test_claim_outcomes_for_unclaimable_tasks(clean_db: Engine) -> None:
    assert repo.claim_task(clean_db, uuid.uuid4(), "w1", lease_ttl_s=5)[0] == ClaimOutcome.NOT_FOUND
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    c = claim(clean_db, tid)
    assert repo.claim_task(clean_db, tid, "w2", lease_ttl_s=5)[0] == ClaimOutcome.ALREADY_RUNNING
    assert publish(clean_db, c).outcome == FinalizeOutcome.PUBLISHED
    assert repo.claim_task(clean_db, tid, "w2", lease_ttl_s=5)[0] == ClaimOutcome.TERMINAL


def test_claim_records_queue_wait_and_attempt_row(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    time.sleep(0.05)
    c = claim(clean_db, tid)
    assert c.attempt == 1 and c.queue_wait_ms >= 40
    att = repo.get_task_detail(clean_db, tid)["attempts"][0]
    assert att["status"] == "RUNNING" and att["worker_id"] == "w1" and att["code_sha"] == CODE


# ----------------------------------------------------------------------------- heartbeat


def test_heartbeat_extends_lease_and_reports_lost_and_cancel(clean_db: Engine) -> None:
    res = submit(clean_db, 2)
    t1, t2 = task_ids(clean_db, res.batch_id)
    c1 = claim(clean_db, t1, ttl=2)
    before = task_row(clean_db, t1)["lease_expiry"]
    status = repo.heartbeat(
        clean_db,
        worker(),
        inflight=[(t1, c1.attempt), (t2, 1)],
        lease_ttl_s=60,
        resource_profile={"rss_mb": 10},
    )
    assert status[t1] == LeaseStatus.OK
    assert status[t2] == LeaseStatus.LOST  # never claimed by w1
    assert task_row(clean_db, t1)["lease_expiry"] > before
    # wrong attempt number (fencing token) -> LOST
    assert (
        repo.heartbeat(
            clean_db, worker(), inflight=[(t1, 99)], lease_ttl_s=60, resource_profile={}
        )[t1]
        == LeaseStatus.LOST
    )
    repo.cancel_batch(clean_db, res.batch_id, user_id="user-1")
    assert (
        repo.heartbeat(clean_db, worker(), inflight=[(t1, 1)], lease_ttl_s=60, resource_profile={})[
            t1
        ]
        == LeaseStatus.CANCEL_REQUESTED
    )
    live, slots = repo.live_worker_capacity(clean_db, dead_after_s=30)
    assert (live, slots) == (1, 1)


# ----------------------------------------------------------------------------- publish


def test_publish_once_and_transitions_logged(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    c = claim(clean_db, tid)
    out = publish(clean_db, c)
    assert out.outcome == FinalizeOutcome.PUBLISHED
    # a duplicate publish of the same attempt is not "owned" any more -> never re-published
    dup = publish(clean_db, c, tag="other")
    assert dup.outcome == FinalizeOutcome.LEASE_LOST
    detail = repo.get_task_detail(clean_db, tid)
    assert (
        detail["task"]["state"] == "SUCCEEDED"
        and detail["artifact"]["output_sha"] == out.published_sha
    )
    assert [(t["from_state"], t["to_state"]) for t in detail["transitions"]] == [
        (None, "PENDING"),
        ("PENDING", "RUNNING"),
        ("RUNNING", "SUCCEEDED"),
    ]
    assert detail["transitions"][1]["actor"] == "worker:w1"
    assert detail["attempts"][0]["status"] == "SUCCEEDED"


def test_zombie_worker_cannot_publish_after_reclaim(clean_db: Engine) -> None:
    """Worker A stalls past its lease; B re-runs the task; A wakes up and tries to publish."""
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    a = claim(clean_db, tid, "wA", ttl=0.3)
    time.sleep(0.5)
    reclaimed = repo.reconcile_expired_leases(clean_db)
    assert [(r.task_id, r.outcome) for r in reclaimed] == [(tid, FailureOutcome.RETRY_SCHEDULED)]
    b = claim(clean_db, tid, "wB")
    assert b.attempt == 2
    zombie = publish(clean_db, a, "wA", tag="same")  # A finishes first, with identical output
    assert zombie.outcome == FinalizeOutcome.LEASE_LOST and zombie.published_sha is None
    winner = publish(clean_db, b, "wB", tag="same")
    assert winner.outcome == FinalizeOutcome.PUBLISHED
    detail = repo.get_task_detail(clean_db, tid)
    statuses = [(x["attempt"], x["status"], x["error_code"]) for x in detail["attempts"]]
    assert statuses == [(1, "LEASE_EXPIRED", "WORKER_LOST"), (2, "SUCCEEDED", None)]
    assert detail["attempts"][0]["output_sha"] == winner.published_sha  # late checksum recorded
    assert detail["attempts"][0]["late_completion_at"] is not None
    assert detail["task"]["checksum_mismatch"] is False
    with clean_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM result_artifacts")).scalar_one() == 1


def test_late_completion_with_different_checksum_is_flagged(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    a = claim(clean_db, tid, "wA", ttl=0.2)
    time.sleep(0.4)
    repo.reconcile_expired_leases(clean_db)
    b = claim(clean_db, tid, "wB")
    assert publish(clean_db, b, "wB", tag="v1").outcome == FinalizeOutcome.PUBLISHED
    late = publish(clean_db, a, "wA", tag="v2-nondeterministic")
    assert late.outcome == FinalizeOutcome.LEASE_LOST and late.checksum_match is False
    detail = repo.get_task_detail(clean_db, tid)
    assert detail["task"]["checksum_mismatch"] is True
    assert detail["artifact"]["attempt"] == 2  # the published result is attempt 2's


def test_stalled_but_not_reclaimed_owner_may_still_publish(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    c = claim(clean_db, tid, ttl=0.1)
    time.sleep(0.3)  # lease expired, but no reconciler ran -> nobody else could claim
    assert publish(clean_db, c).outcome == FinalizeOutcome.PUBLISHED


# ----------------------------------------------------------------------------- failures


def test_retryable_failure_backoff_then_exhaustion(clean_db: Engine) -> None:
    res = submit(clean_db, 1, max_attempts=2)
    tid = task_ids(clean_db, res.batch_id)[0]
    c1 = claim(clean_db, tid)
    out = repo.record_failure(
        clean_db,
        task_id=tid,
        attempt=1,
        worker_id="w1",
        error_code=ErrorCode.TRANSIENT_ERROR,
        message="boom",
        retryable=True,
        backoff_s=0.5,
    )
    assert out == FailureOutcome.RETRY_SCHEDULED and c1.attempt == 1
    assert repo.claim_task(clean_db, tid, "w1", lease_ttl_s=5)[0] == ClaimOutcome.NOT_READY
    assert repo.dispatch_ready(clean_db, window=10, lost_after_s=60) == []  # backoff respected
    time.sleep(0.6)
    assert [i.task_id for i in repo.dispatch_ready(clean_db, window=10, lost_after_s=60)] == [tid]
    claim(clean_db, tid)
    out2 = repo.record_failure(
        clean_db,
        task_id=tid,
        attempt=2,
        worker_id="w1",
        error_code=ErrorCode.TRANSIENT_ERROR,
        message="boom again",
        retryable=True,
    )
    assert out2 == FailureOutcome.FAILED
    row = task_row(clean_db, tid)
    assert row["state"] == "FAILED" and row["last_error_code"] == "TRANSIENT_ERROR"


def test_non_retryable_failure_fails_immediately(clean_db: Engine) -> None:
    res = submit(clean_db, 1, max_attempts=5)
    tid = task_ids(clean_db, res.batch_id)[0]
    claim(clean_db, tid)
    out = repo.record_failure(
        clean_db,
        task_id=tid,
        attempt=1,
        worker_id="w1",
        error_code=ErrorCode.DATA_INTEGRITY_ERROR,
        message="bad data",
        retryable=False,
    )
    assert out == FailureOutcome.FAILED


def test_failure_from_non_owner_is_ignored(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    claim(clean_db, tid, "w1")
    out = repo.record_failure(
        clean_db,
        task_id=tid,
        attempt=1,
        worker_id="intruder",
        error_code=ErrorCode.TASK_ERROR,
        message="x",
        retryable=True,
    )
    assert out == FailureOutcome.LEASE_LOST
    assert task_row(clean_db, tid)["state"] == "RUNNING"


def test_lease_expiry_on_last_attempt_fails_task(clean_db: Engine) -> None:
    res = submit(clean_db, 1, max_attempts=1)
    tid = task_ids(clean_db, res.batch_id)[0]
    claim(clean_db, tid, ttl=0.1)
    time.sleep(0.3)
    [r] = repo.reconcile_expired_leases(clean_db)
    assert r.outcome == FailureOutcome.FAILED
    assert task_row(clean_db, tid)["last_error_code"] == "WORKER_LOST"


def test_release_lease_requeues_immediately(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    claim(clean_db, tid)
    out = repo.release_lease(clean_db, task_id=tid, attempt=1, worker_id="w1", reason="shutdown")
    assert out == FailureOutcome.RETRY_SCHEDULED
    assert claim(clean_db, tid, "w2").attempt == 2


# ----------------------------------------------------------------------------- cancel


def test_cancel_reports_completed_and_stops_running(clean_db: Engine) -> None:
    res = submit(clean_db, 4)
    t = task_ids(clean_db, res.batch_id)
    done = claim(clean_db, t[0])
    publish(clean_db, done)
    running = claim(clean_db, t[1])
    report = repo.cancel_batch(clean_db, res.batch_id, user_id="user-1")
    assert report.canceled_now == 2 and report.cancel_requested_running == 1
    assert report.completed_count == 1 and report.completed_task_ids == [t[0]]
    assert report.state == "CANCELING"
    # the running worker publishes after cancel -> discarded, task CANCELED
    assert publish(clean_db, running).outcome == FinalizeOutcome.CANCELED
    changes = repo.refresh_batch_states(clean_db)
    assert [(c.old, c.new) for c in changes] == [("CANCELING", "CANCELED")]
    with clean_db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM result_artifacts")).scalar_one() == 1
    # canceled tasks are never dispatched or claimed
    assert repo.dispatch_ready(clean_db, window=10, lost_after_s=60) == []
    assert repo.claim_task(clean_db, t[2], "w9", lease_ttl_s=5)[0] == ClaimOutcome.TERMINAL
    # cancel is idempotent on a terminal batch
    assert repo.cancel_batch(clean_db, res.batch_id, user_id="user-1").already_terminal


def test_cancel_requires_owner(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    with pytest.raises(NotFound):
        repo.cancel_batch(clean_db, res.batch_id, user_id="someone-else")


def test_cancel_vs_publish_race_is_consistent(clean_db: Engine) -> None:
    """Either the task is reported completed by cancel, or it can never publish."""

    def race(batch_id: uuid.UUID, claimed: repo.ClaimedTask) -> tuple[object, object]:
        barrier = threading.Barrier(2)
        out: dict[str, object] = {}

        def do_cancel() -> None:
            barrier.wait()
            out["cancel"] = repo.cancel_batch(clean_db, batch_id, user_id="user-1")

        def do_publish() -> None:
            barrier.wait()
            out["publish"] = publish(clean_db, claimed)

        threads = [threading.Thread(target=do_cancel), threading.Thread(target=do_publish)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return out["cancel"], out["publish"]

    outcomes = set()
    for _ in range(15):
        res = submit(clean_db, 1)
        tid = task_ids(clean_db, res.batch_id)[0]
        report, result = race(res.batch_id, claim(clean_db, tid))
        assert isinstance(report, repo.CancelReport)
        assert isinstance(result, repo.FinalizeResult)
        outcomes.add(result.outcome)
        if result.outcome == FinalizeOutcome.PUBLISHED:
            assert report.completed_task_ids == [tid]
        else:
            assert result.outcome == FinalizeOutcome.CANCELED
            assert report.completed_count == 0
        expected = "SUCCEEDED" if result.outcome == FinalizeOutcome.PUBLISHED else "CANCELED"
        assert task_row(clean_db, tid)["state"] == expected


# ----------------------------------------------------------------------------- retry-failed


def test_retry_failed_requeues_eligible_tasks_within_quota(clean_db: Engine) -> None:
    res = submit(clean_db, 3, max_attempts=1, quota=QuotaPolicy(max_attempts=3))
    t = task_ids(clean_db, res.batch_id)
    for tid, code, retryable in [
        (t[0], ErrorCode.TRANSIENT_ERROR, True),
        (t[1], ErrorCode.SPEC_ERROR, False),
    ]:
        claim(clean_db, tid)
        repo.record_failure(
            clean_db,
            task_id=tid,
            attempt=1,
            worker_id="w1",
            error_code=code,
            message="x",
            retryable=retryable,
        )
    publish(clean_db, claim(clean_db, t[2]))
    assert repo.refresh_batch_states(clean_db)[0].new == "COMPLETED_WITH_FAILURES"
    report = repo.retry_failed(clean_db, res.batch_id, user_id="user-1", extra_attempts=1)
    assert report.retried == 1 and report.retried_task_ids == [t[0]]
    assert report.skipped_not_retryable == 1 and report.state == "RUNNING"
    row = task_row(clean_db, t[0])
    assert row["state"] == "RETRY_PENDING" and row["max_attempts"] == 2
    # attempt quota: max_attempts=3 in quota, so a second retry of 1 more works, a third doesn't
    c = claim(clean_db, t[0])
    repo.record_failure(
        clean_db,
        task_id=t[0],
        attempt=c.attempt,
        worker_id="w1",
        error_code=ErrorCode.TRANSIENT_ERROR,
        message="x",
        retryable=True,
    )
    assert repo.retry_failed(clean_db, res.batch_id, user_id="user-1").retried == 1
    c = claim(clean_db, t[0])
    repo.record_failure(
        clean_db,
        task_id=t[0],
        attempt=c.attempt,
        worker_id="w1",
        error_code=ErrorCode.TRANSIENT_ERROR,
        message="x",
        retryable=True,
    )
    third = repo.retry_failed(clean_db, res.batch_id, user_id="user-1")
    assert third.retried == 0 and third.skipped_attempt_quota == 1


def test_retry_failed_respects_worker_time_quota(clean_db: Engine) -> None:
    quota = QuotaPolicy(max_worker_seconds_per_batch=150, max_attempts=5)
    res = submit(clean_db, 1, max_attempts=1, quota=quota)  # max_runtime_s defaults to 120
    tid = task_ids(clean_db, res.batch_id)[0]
    claim(clean_db, tid)
    repo.record_failure(
        clean_db,
        task_id=tid,
        attempt=1,
        worker_id="w1",
        error_code=ErrorCode.TASK_TIMEOUT,
        message="slow",
        retryable=True,
        stats=AttemptStats(wall_ms=100_000),
    )
    with pytest.raises(QuotaExceeded):
        repo.retry_failed(clean_db, res.batch_id, user_id="user-1")


def test_retry_after_cancel_conflicts(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    repo.cancel_batch(clean_db, res.batch_id, user_id="user-1")
    with pytest.raises(Conflict):
        repo.retry_failed(clean_db, res.batch_id, user_id="user-1")


# ----------------------------------------------------------------------------- batch state


def test_batch_state_lifecycle(clean_db: Engine) -> None:
    res = submit(clean_db, 2)
    t = task_ids(clean_db, res.batch_id)
    assert repo.refresh_batch_states(clean_db) == []
    c = claim(clean_db, t[0])
    assert [x.new for x in repo.refresh_batch_states(clean_db)] == ["RUNNING"]
    publish(clean_db, c)
    publish(clean_db, claim(clean_db, t[1]))
    assert [x.new for x in repo.refresh_batch_states(clean_db)] == ["COMPLETED"]
    batch = repo.get_batch(clean_db, res.batch_id)
    assert batch["started_at"] is not None and batch["finished_at"] is not None
    progress = repo.batch_progress(clean_db, res.batch_id)
    assert progress["counts"]["SUCCEEDED"] == 2 and progress["published_results"] == 2
    assert progress["cost"]["worker_seconds"] == pytest.approx(0.01)


def test_cache_lookup_requires_same_code_and_env(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    c = claim(clean_db, tid)
    publish(clean_db, c)
    spec_hash = task_row(clean_db, tid)["spec_hash"]
    assert repo.find_cached_result(clean_db, spec_hash=spec_hash, code_sha=CODE, env_sha=ENV)
    assert (
        repo.find_cached_result(clean_db, spec_hash=spec_hash, code_sha="d" * 64, env_sha=ENV)
        is None
    )
    assert (
        repo.find_cached_result(clean_db, spec_hash=spec_hash, code_sha=CODE, env_sha="f" * 64)
        is None
    )


def test_dead_workers_are_marked(clean_db: Engine) -> None:
    repo.heartbeat(clean_db, worker("old"), inflight=[], lease_ttl_s=5, resource_profile={})
    time.sleep(0.3)
    assert repo.mark_dead_workers(clean_db, dead_after_s=0.1) == ["old"]
    assert repo.live_worker_capacity(clean_db, dead_after_s=0.1) == (0, 0)
