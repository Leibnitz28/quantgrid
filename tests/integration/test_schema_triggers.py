"""The database enforces the same state machines as ``quantgrid.domain.states``."""

from __future__ import annotations

import itertools
import uuid

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError

from quantgrid.domain.states import BATCH_TRANSITIONS, TASK_TRANSITIONS, BatchState, TaskState

from .factories import submit, task_ids

pytestmark = pytest.mark.integration


def test_task_transition_table_matches_domain(engine: Engine) -> None:
    with engine.connect() as conn:
        for a, b in itertools.product(TaskState, TaskState):
            if a == b:
                continue
            db_allowed = conn.execute(
                text("SELECT qg_task_transition_allowed(:a, :b)"), {"a": a.value, "b": b.value}
            ).scalar_one()
            assert db_allowed == (b in TASK_TRANSITIONS[a]), (a, b)


def test_batch_transition_table_matches_domain(engine: Engine) -> None:
    with engine.connect() as conn:
        for a, b in itertools.product(BatchState, BatchState):
            if a == b:
                continue
            db_allowed = conn.execute(
                text("SELECT qg_batch_transition_allowed(:a, :b)"), {"a": a.value, "b": b.value}
            ).scalar_one()
            assert db_allowed == (b in BATCH_TRANSITIONS[a]), (a, b)


def test_illegal_transition_and_spec_mutation_rejected(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    with (
        pytest.raises(DBAPIError, match="illegal task transition PENDING -> SUCCEEDED"),
        clean_db.begin() as conn,
    ):
        conn.execute(text("UPDATE tasks SET state = 'SUCCEEDED' WHERE id = :id"), {"id": tid})
    with pytest.raises(DBAPIError, match="immutable"), clean_db.begin() as conn:
        conn.execute(text("UPDATE tasks SET spec = '{}'::jsonb WHERE id = :id"), {"id": tid})
    with pytest.raises(DBAPIError, match="lease_iff_running"), clean_db.begin() as conn:
        conn.execute(text("UPDATE tasks SET state = 'RUNNING' WHERE id = :id"), {"id": tid})


def test_published_results_and_terminal_attempts_are_immutable(clean_db: Engine) -> None:
    from quantgrid.storage import repository as repo

    from .factories import artifact

    res = submit(clean_db, 1)
    tid = task_ids(clean_db, res.batch_id)[0]
    _, c = repo.claim_task(clean_db, tid, "w1", lease_ttl_s=30)
    assert c is not None
    repo.finalize_success(
        clean_db,
        task_id=tid,
        attempt=1,
        worker_id="w1",
        artifact=artifact(),
        stats=repo.AttemptStats(),
        code_sha="c" * 64,
        env_sha="e" * 64,
    )
    with pytest.raises(DBAPIError, match="immutable"), clean_db.begin() as conn:
        conn.execute(
            text("UPDATE result_artifacts SET output_sha = 'x' WHERE task_id = :id"),
            {"id": tid},
        )
    with pytest.raises(DBAPIError, match="terminal"), clean_db.begin() as conn:
        conn.execute(
            text("UPDATE task_attempts SET status = 'FAILED' WHERE task_id = :id"), {"id": tid}
        )
    with (
        pytest.raises(DBAPIError, match="illegal task transition SUCCEEDED -> RETRY_PENDING"),
        clean_db.begin() as conn,
    ):
        conn.execute(text("UPDATE tasks SET state = 'RETRY_PENDING' WHERE id = :id"), {"id": tid})


def test_batch_spec_immutable(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    with pytest.raises(DBAPIError, match="immutable"), clean_db.begin() as conn:
        conn.execute(
            text("UPDATE batches SET spec_sha = 'tampered' WHERE id = :id"),
            {"id": res.batch_id},
        )
    with clean_db.begin() as conn:
        conn.execute(
            text("UPDATE batches SET state = 'CANCELING' WHERE id = :id"), {"id": res.batch_id}
        )
    with (
        pytest.raises(DBAPIError, match="illegal batch transition CANCELING -> COMPLETED"),
        clean_db.begin() as conn,
    ):
        conn.execute(
            text("UPDATE batches SET state = 'COMPLETED' WHERE id = :id"), {"id": res.batch_id}
        )


def test_tasks_must_be_born_pending(clean_db: Engine) -> None:
    res = submit(clean_db, 1)
    with pytest.raises(DBAPIError, match="must be created PENDING"), clean_db.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO tasks (id, batch_id, ordinal, spec, spec_hash, state, max_attempts, "
                "max_runtime_s, max_memory_mb, cache_policy) VALUES (:id, :b, 99, '{}', 'h', "
                "'SUCCEEDED', 1, 1, 64, 'use')"
            ),
            {"id": uuid.uuid4(), "b": res.batch_id},
        )
