"""End-to-end: real coordinator + Celery worker processes + Redis + PostgreSQL.

Failure injection uses real signals: SIGKILL (crash) and SIGSTOP/SIGCONT (a stalled
"zombie" that resumes after its lease was reassigned). Every scenario ends with the
batch verifier, which must prove exactly one logical result per expected task.
"""

from __future__ import annotations

import os
import signal
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import redis
from sqlalchemy import Engine, text

from quantgrid.devstack import LocalStack
from quantgrid.domain.policy import PolicyService, QuotaPolicy
from quantgrid.domain.specs import DatasetRef, ExperimentBatchSpec, StrategyRef
from quantgrid.manifest import build_manifest
from quantgrid.storage import repository as repo

pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.slow]

REDIS_URL = os.environ.get("QG_TEST_REDIS_URL", "redis://127.0.0.1:6380/15")
QUOTA = QuotaPolicy(allow_fault_injection=True, max_attempts=10)


@pytest.fixture(scope="module")
def stack_env(db_url: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    try:
        redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2).ping()
    except redis.RedisError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Redis not reachable at {REDIS_URL}: {exc}")
    root = tmp_path_factory.mktemp("e2e")
    return {
        "QG_DATABASE_URL": db_url,
        "QG_REDIS_URL": REDIS_URL,
        "QG_QUEUE_NAME": f"qg.test.{uuid.uuid4().hex[:8]}",
        "QG_DATA_DIR": str(root / "data"),
        "QG_ARTIFACT_ROOT": str(root / "artifacts"),
        "QG_LEASE_TTL_S": "3",
        "QG_HEARTBEAT_INTERVAL_S": "0.5",
        "QG_WORKER_DEAD_AFTER_S": "4",
        "QG_COORDINATOR_TICK_S": "0.2",
        "QG_DISPATCH_LOST_AFTER_S": "20",
        "QG_RETRY_BACKOFF_BASE_S": "0.2",
        "QG_RETRY_BACKOFF_MAX_S": "1",
        "QG_ALLOW_FAULT_INJECTION": "true",
        "QG_LOG_LEVEL": "INFO",
        "QG_LOG_FORMAT": "json",
    }


@pytest.fixture
def stack(stack_env: dict[str, str], tmp_path: Path, clean_db: Engine) -> Iterator[LocalStack]:
    env = {**stack_env, "QG_ARTIFACT_ROOT": str(tmp_path / "artifacts")}  # per-test store
    with LocalStack(log_dir=tmp_path / "logs", env=env) as s:
        yield s


def _catalog() -> Any:
    from quantgrid.domain.datasets import get_catalog

    return get_catalog()


def make_spec(
    *,
    lookbacks: list[int],
    entry: list[float] | None = None,
    exit_: list[float] | None = None,
    seeds: list[int] | None = None,
    dataset: str = "pairs-ou-small-v1",
    **extra: Any,
) -> ExperimentBatchSpec:
    info = _catalog().get(dataset)
    assert info is not None
    return ExperimentBatchSpec(
        name="e2e",
        dataset=DatasetRef(name=dataset, sha256=info.sha256),
        strategy=StrategyRef(name="pairs_zscore", version="1"),
        grid={"lookback": lookbacks, "entry_z": entry or [2.0], "exit_z": exit_ or [0.5]},
        seeds=seeds or [0],
        **extra,
    )


def submit(engine: Engine, spec: ExperimentBatchSpec, *, key: str | None = None) -> uuid.UUID:
    catalog = _catalog()
    decision, admitted = PolicyService(catalog).admit(spec, QUOTA)
    assert admitted is not None, decision.violations
    res = repo.create_batch(
        engine,
        user_id="e2e",
        idempotency_key=key or uuid.uuid4().hex,
        spec=spec,
        tasks_expanded=admitted.tasks,
        manifest=build_manifest(admitted.dataset, spec.spec_sha()),
        quota=QUOTA,
        dataset=admitted.dataset,
    )
    return res.batch_id


def verify(engine: Engine, env: dict[str, str], batch_id: uuid.UUID, **kw: Any) -> Any:
    from quantgrid.results.verify import verify_batch
    from quantgrid.storage.artifacts import LocalArtifactStore

    report = verify_batch(engine, LocalArtifactStore(Path(env["QG_ARTIFACT_ROOT"])), batch_id, **kw)
    print(report.summary())
    assert report.ok, report.summary()
    return report


def start(stack: LocalStack, engine: Engine, n_workers: int) -> list[str]:
    stack.start_coordinator()
    ids = [f"w{i}-{uuid.uuid4().hex[:4]}" for i in range(n_workers)]
    for wid in ids:
        stack.start_worker(wid)
    stack.wait_for_workers(engine, ids, timeout=60)
    return ids


def running_on(engine: Engine, batch_id: uuid.UUID) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, lease_owner FROM tasks WHERE batch_id = :b AND state = 'RUNNING'"),
            {"b": batch_id},
        ).all()
    return [(str(r.id), r.lease_owner) for r in rows]


def wait_until(predicate: Any, timeout: float, what: str) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {what}")


# ======================================================================================


def test_happy_path_120_tasks_one_result_each(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    start(stack, clean_db, 3)
    spec = make_spec(
        lookbacks=[20, 30, 40, 50, 60], entry=[1.5, 2.0, 2.5], exit_=[0.0, 0.5], seeds=[1, 2, 3, 4]
    )
    assert spec.task_count() == 120
    bid = submit(clean_db, spec)
    assert stack.wait_for_batch(clean_db, bid, timeout=240) == "COMPLETED"
    report = verify(clean_db, stack.env, bid)
    assert report.published_results == 120 and report.counts["SUCCEEDED"] == 120


def test_sigkill_worker_mid_task_recovers_without_duplicates(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    ids = start(stack, clean_db, 3)
    spec = make_spec(lookbacks=list(range(20, 44)), faults={"sleep_s": 1.0})
    bid = submit(clean_db, spec)
    victim_task, victim = wait_until(lambda: running_on(clean_db, bid), 60, "a running task")[0]
    killed_at = time.monotonic()
    stack.processes[victim].kill()
    assert stack.wait_for_batch(clean_db, bid, timeout=240) == "COMPLETED"
    report = verify(clean_db, stack.env, bid)
    assert report.lease_expirations >= 1
    detail = repo.get_task_detail(clean_db, victim_task)
    statuses = [(a["worker_id"], a["status"]) for a in detail["attempts"]]
    assert (victim, "LEASE_EXPIRED") in statuses and statuses[-1][1] == "SUCCEEDED"
    assert time.monotonic() - killed_at < 240
    assert victim in ids


def test_sigstop_zombie_cannot_publish_after_reassignment(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    start(stack, clean_db, 2)
    spec = make_spec(lookbacks=list(range(20, 30)), faults={"sleep_s": 1.5})
    bid = submit(clean_db, spec)
    task_id, victim = wait_until(lambda: running_on(clean_db, bid), 60, "a running task")[0]
    proc = stack.processes[victim]
    proc.send(signal.SIGSTOP)
    try:
        # lease (3 s TTL) expires, coordinator reclaims, the other worker re-runs it
        def reassigned() -> bool:
            d = repo.get_task_detail(clean_db, task_id)
            return d["task"]["state"] == "SUCCEEDED" or d["task"]["lease_owner"] not in (
                None,
                victim,
            )

        wait_until(reassigned, 60, "task reassignment")
    finally:
        proc.send(signal.SIGCONT)
    assert stack.wait_for_batch(clean_db, bid, timeout=240) == "COMPLETED"

    # The resumed worker must not publish. Two safe outcomes exist: usually its heartbeat
    # thread learns first that the lease was reclaimed and the task aborts cooperatively;
    # if the compute thread reaches publish first, the fenced finalize rejects it and only
    # records its checksum (covered deterministically in test_repository.py).
    def zombie_settled() -> bool:
        late = any(
            a["late_completion_at"] for a in repo.get_task_detail(clean_db, task_id)["attempts"]
        )
        return late or "lease lost; aborting task" in proc.tail(400)

    wait_until(zombie_settled, 30, "zombie to abort or record a late completion")
    report = verify(clean_db, stack.env, bid)
    detail = repo.get_task_detail(clean_db, task_id)
    first = detail["attempts"][0]
    assert first["status"] == "LEASE_EXPIRED" and first["worker_id"] == victim
    assert detail["artifact"]["attempt"] >= 2  # published by the replacement attempt
    if first["output_sha"] is not None:  # late completion path: checksum must be identical
        assert first["output_sha"] == detail["artifact"]["output_sha"]
    assert report.published_results == 10


def test_transient_faults_are_retried(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    start(stack, clean_db, 2)
    spec = make_spec(
        lookbacks=list(range(20, 32)), faults={"fail_on_attempts": [1], "error_kind": "transient"}
    )
    bid = submit(clean_db, spec)
    assert stack.wait_for_batch(clean_db, bid, timeout=180) == "COMPLETED"
    report = verify(clean_db, stack.env, bid)
    assert report.retry_distribution == {"2": 12}


def test_permanent_fault_then_retry_failed(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    start(stack, clean_db, 2)
    spec = make_spec(
        lookbacks=list(range(20, 26)), faults={"fail_on_attempts": [1], "error_kind": "permanent"}
    )
    bid = submit(clean_db, spec)
    assert stack.wait_for_batch(clean_db, bid, timeout=120) == "COMPLETED_WITH_FAILURES"
    progress = repo.batch_progress(clean_db, bid)
    assert progress["counts"]["FAILED"] == 6
    skipped = repo.retry_failed(clean_db, bid, user_id="e2e")
    assert skipped.retried == 0 and skipped.skipped_not_retryable == 6
    retried = repo.retry_failed(clean_db, bid, user_id="e2e", include_non_retryable=True)
    assert retried.retried == 6
    assert stack.wait_for_batch(clean_db, bid, timeout=120, states=("COMPLETED",)) == "COMPLETED"
    verify(clean_db, stack.env, bid)


def test_cancel_mid_batch_keeps_completed_results(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    start(stack, clean_db, 2)
    spec = make_spec(lookbacks=list(range(20, 60)), faults={"sleep_s": 0.5})
    bid = submit(clean_db, spec)
    wait_until(
        lambda: repo.batch_progress(clean_db, bid)["counts"]["SUCCEEDED"] >= 3, 60, "3 completions"
    )
    report = repo.cancel_batch(clean_db, bid, user_id="e2e")
    assert report.completed_count >= 3
    assert stack.wait_for_batch(clean_db, bid, timeout=60) == "CANCELED"
    final = verify(clean_db, stack.env, bid)
    assert final.counts["SUCCEEDED"] >= report.completed_count
    assert final.counts["SUCCEEDED"] + final.counts["CANCELED"] == 40


def test_identical_resubmission_hits_cache(
    stack: LocalStack, clean_db: Engine, stack_env: dict[str, str]
) -> None:
    start(stack, clean_db, 2)
    spec = make_spec(lookbacks=[20, 30, 40, 50])
    first = submit(clean_db, spec)
    assert stack.wait_for_batch(clean_db, first, timeout=120) == "COMPLETED"
    second = submit(clean_db, spec)  # new idempotency key -> new logical batch, same specs
    assert stack.wait_for_batch(clean_db, second, timeout=120) == "COMPLETED"
    verify(clean_db, stack.env, second)
    a = {r["spec_hash"]: r["output_sha"] for r in repo.list_results(clean_db, first)}
    b = {r["spec_hash"]: r["output_sha"] for r in repo.list_results(clean_db, second)}
    assert a == b
    assert repo.batch_progress(clean_db, second)["cache_hits"] == 4
