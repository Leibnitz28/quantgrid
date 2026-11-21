"""initial schema

Revision ID: 0001
Revises: (none)
Create Date: 2026-11-19 19:23:19.720159+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Database-level invariants; mirror quantgrid/domain/states.py and are checked by
# tests/integration/test_schema_triggers.py. Frozen with this revision.
TRIGGERS_SQL = r"""
CREATE FUNCTION qg_task_transition_allowed(from_state text, to_state text)
RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
  SELECT (from_state, to_state) IN (
    ('PENDING', 'RUNNING'), ('PENDING', 'CANCELED'),
    ('RUNNING', 'SUCCEEDED'), ('RUNNING', 'RETRY_PENDING'), ('RUNNING', 'FAILED'),
    ('RUNNING', 'CANCELED'),
    ('RETRY_PENDING', 'RUNNING'), ('RETRY_PENDING', 'CANCELED'),
    ('FAILED', 'RETRY_PENDING')
  )
$$;

CREATE FUNCTION qg_batch_transition_allowed(from_state text, to_state text)
RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
  SELECT (from_state, to_state) IN (
    ('ACCEPTED', 'RUNNING'), ('ACCEPTED', 'CANCELING'), ('ACCEPTED', 'CANCELED'),
    ('ACCEPTED', 'COMPLETED'), ('ACCEPTED', 'COMPLETED_WITH_FAILURES'),
    ('RUNNING', 'CANCELING'), ('RUNNING', 'CANCELED'), ('RUNNING', 'COMPLETED'),
    ('RUNNING', 'COMPLETED_WITH_FAILURES'),
    ('CANCELING', 'CANCELED'),
    ('COMPLETED_WITH_FAILURES', 'RUNNING')
  )
$$;

CREATE FUNCTION qg_tasks_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.state <> 'PENDING' THEN
      RAISE EXCEPTION 'tasks must be created PENDING (got %)', NEW.state
        USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
  END IF;
  IF NEW.state IS DISTINCT FROM OLD.state
     AND NOT qg_task_transition_allowed(OLD.state, NEW.state) THEN
    RAISE EXCEPTION 'illegal task transition % -> % (task %)', OLD.state, NEW.state, OLD.id
      USING ERRCODE = 'check_violation';
  END IF;
  IF NEW.spec IS DISTINCT FROM OLD.spec OR NEW.spec_hash IS DISTINCT FROM OLD.spec_hash
     OR NEW.batch_id IS DISTINCT FROM OLD.batch_id OR NEW.ordinal IS DISTINCT FROM OLD.ordinal THEN
    RAISE EXCEPTION 'task spec/identity is immutable (task %)', OLD.id
      USING ERRCODE = 'check_violation';
  END IF;
  IF NEW.attempts < OLD.attempts THEN
    RAISE EXCEPTION 'task attempts counter cannot decrease (task %)', OLD.id
      USING ERRCODE = 'check_violation';
  END IF;
  NEW.updated_at := now();
  RETURN NEW;
END
$$;
CREATE TRIGGER qg_tasks_guard BEFORE INSERT OR UPDATE ON tasks
  FOR EACH ROW EXECUTE FUNCTION qg_tasks_guard();

CREATE FUNCTION qg_tasks_log_transition() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'INSERT' OR NEW.state IS DISTINCT FROM OLD.state THEN
    INSERT INTO task_transitions
      (task_id, batch_id, from_state, to_state, attempt, actor, reason, at)
    VALUES (
      NEW.id, NEW.batch_id,
      CASE WHEN TG_OP = 'INSERT' THEN NULL ELSE OLD.state END,
      NEW.state, NEW.attempts,
      coalesce(nullif(current_setting('quantgrid.actor', true), ''), 'unknown'),
      nullif(current_setting('quantgrid.reason', true), ''),
      clock_timestamp()
    );
  END IF;
  RETURN NULL;
END
$$;
CREATE TRIGGER qg_tasks_log_transition AFTER INSERT OR UPDATE OF state ON tasks
  FOR EACH ROW EXECUTE FUNCTION qg_tasks_log_transition();

CREATE FUNCTION qg_attempts_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.status <> 'RUNNING' AND NEW.status <> 'CACHED' THEN
      RAISE EXCEPTION 'attempts must be created RUNNING or CACHED (got %)', NEW.status
        USING ERRCODE = 'check_violation';
    END IF;
    RETURN NEW;
  END IF;
  IF OLD.status <> 'RUNNING' THEN
    -- Terminal attempts are immutable, except that a late ("zombie") completion may be
    -- recorded once: output_sha / late_completion_at / timing columns from NULL.
    IF NEW.status IS DISTINCT FROM OLD.status
       OR NEW.task_id IS DISTINCT FROM OLD.task_id OR NEW.attempt IS DISTINCT FROM OLD.attempt
       OR NEW.worker_id IS DISTINCT FROM OLD.worker_id
       OR NEW.started_at IS DISTINCT FROM OLD.started_at
       OR NEW.ended_at IS DISTINCT FROM OLD.ended_at
       OR NEW.error_code IS DISTINCT FROM OLD.error_code
       OR (OLD.output_sha IS NOT NULL AND NEW.output_sha IS DISTINCT FROM OLD.output_sha)
       OR (OLD.late_completion_at IS NOT NULL
           AND NEW.late_completion_at IS DISTINCT FROM OLD.late_completion_at) THEN
      RAISE EXCEPTION 'attempt % of task % is terminal (%) and immutable',
        OLD.attempt, OLD.task_id, OLD.status USING ERRCODE = 'check_violation';
    END IF;
  ELSIF NEW.status IS DISTINCT FROM OLD.status AND NEW.status = 'RUNNING' THEN
    RAISE EXCEPTION 'invalid attempt status change' USING ERRCODE = 'check_violation';
  END IF;
  RETURN NEW;
END
$$;
CREATE TRIGGER qg_attempts_guard BEFORE INSERT OR UPDATE ON task_attempts
  FOR EACH ROW EXECUTE FUNCTION qg_attempts_guard();

CREATE FUNCTION qg_results_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'published result artifacts are immutable (task %)', OLD.task_id
    USING ERRCODE = 'check_violation';
END
$$;
CREATE TRIGGER qg_results_immutable BEFORE UPDATE ON result_artifacts
  FOR EACH ROW EXECUTE FUNCTION qg_results_immutable();

CREATE FUNCTION qg_batches_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.state IS DISTINCT FROM OLD.state
     AND NOT qg_batch_transition_allowed(OLD.state, NEW.state) THEN
    RAISE EXCEPTION 'illegal batch transition % -> % (batch %)', OLD.state, NEW.state, OLD.id
      USING ERRCODE = 'check_violation';
  END IF;
  IF NEW.spec IS DISTINCT FROM OLD.spec OR NEW.spec_sha IS DISTINCT FROM OLD.spec_sha
     OR NEW.manifest IS DISTINCT FROM OLD.manifest OR NEW.user_id IS DISTINCT FROM OLD.user_id
     OR NEW.idempotency_key IS DISTINCT FROM OLD.idempotency_key
     OR NEW.dataset_sha IS DISTINCT FROM OLD.dataset_sha
     OR NEW.code_sha IS DISTINCT FROM OLD.code_sha THEN
    RAISE EXCEPTION 'batch spec/manifest is immutable (batch %)', OLD.id
      USING ERRCODE = 'check_violation';
  END IF;
  NEW.updated_at := now();
  RETURN NEW;
END
$$;
CREATE TRIGGER qg_batches_guard BEFORE UPDATE ON batches
  FOR EACH ROW EXECUTE FUNCTION qg_batches_guard();
"""

DROP_TRIGGERS_SQL = r"""
DROP TRIGGER IF EXISTS qg_batches_guard ON batches;
DROP TRIGGER IF EXISTS qg_results_immutable ON result_artifacts;
DROP TRIGGER IF EXISTS qg_attempts_guard ON task_attempts;
DROP TRIGGER IF EXISTS qg_tasks_log_transition ON tasks;
DROP TRIGGER IF EXISTS qg_tasks_guard ON tasks;
DROP FUNCTION IF EXISTS qg_batches_guard();
DROP FUNCTION IF EXISTS qg_results_immutable();
DROP FUNCTION IF EXISTS qg_attempts_guard();
DROP FUNCTION IF EXISTS qg_tasks_log_transition();
DROP FUNCTION IF EXISTS qg_tasks_guard();
DROP FUNCTION IF EXISTS qg_batch_transition_allowed(text, text);
DROP FUNCTION IF EXISTS qg_task_transition_allowed(text, text);
"""


def upgrade() -> None:
    op.create_table(
        "batches",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("spec", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("spec_sha", sa.String(length=64), nullable=False),
        sa.Column("dataset_name", sa.String(length=64), nullable=False),
        sa.Column("dataset_sha", sa.String(length=64), nullable=False),
        sa.Column("code_sha", sa.String(length=64), nullable=False),
        sa.Column("env_sha", sa.String(length=64), nullable=False),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("quota", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("task_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("priority", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cache_policy", sa.String(length=16), nullable=False),
        sa.Column("reproduces_batch_id", sa.Uuid(), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "cache_policy IN ('use', 'bypass')", name=op.f("ck_batches_cache_policy_valid")
        ),
        sa.CheckConstraint(
            "state IN ('ACCEPTED', 'RUNNING', 'CANCELING', 'COMPLETED', "
            "'COMPLETED_WITH_FAILURES', 'CANCELED')",
            name=op.f("ck_batches_state_valid"),
        ),
        sa.CheckConstraint("task_count > 0", name=op.f("ck_batches_task_count_positive")),
        sa.ForeignKeyConstraint(
            ["reproduces_batch_id"],
            ["batches.id"],
            name=op.f("fk_batches_reproduces_batch_id_batches"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_batches")),
        sa.UniqueConstraint(
            "user_id", "idempotency_key", name=op.f("uq_batches_user_id_idempotency_key")
        ),
    )
    op.create_index(
        "ix_batches_active",
        "batches",
        ["state"],
        unique=False,
        postgresql_where=sa.text("state IN ('ACCEPTED', 'RUNNING', 'CANCELING')"),
    )
    op.create_index("ix_batches_user_state", "batches", ["user_id", "state"], unique=False)
    op.create_table(
        "worker_heartbeats",
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("hostname", sa.String(length=255), nullable=False),
        sa.Column("pid", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("concurrency", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("inflight_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("resource_profile", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("code_sha", sa.String(length=64), nullable=True),
        sa.Column("env_sha", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "state IN ('ACTIVE', 'STOPPING', 'STOPPED', 'DEAD')",
            name=op.f("ck_worker_heartbeats_state_valid"),
        ),
        sa.PrimaryKeyConstraint("worker_id", name=op.f("pk_worker_heartbeats")),
    )
    op.create_index(
        "ix_worker_heartbeats_last_seen", "worker_heartbeats", ["last_seen"], unique=False
    )
    op.create_table(
        "tasks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("spec", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("spec_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("max_runtime_s", sa.Integer(), nullable=False),
        sa.Column("max_memory_mb", sa.Integer(), nullable=False),
        sa.Column("priority", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cache_policy", sa.String(length=16), nullable=False),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expiry", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "cancel_requested", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "ready_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("not_before", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispatch_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_error_code", sa.String(length=64), nullable=True),
        sa.Column("last_error_message", sa.Text(), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "checksum_mismatch", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(state = 'RUNNING') = (lease_owner IS NOT NULL AND lease_expiry IS NOT NULL)",
            name=op.f("ck_tasks_lease_iff_running"),
        ),
        sa.CheckConstraint(
            "cache_policy IN ('use', 'bypass')", name=op.f("ck_tasks_cache_policy_valid")
        ),
        sa.CheckConstraint(
            "state IN ('PENDING', 'RUNNING', 'RETRY_PENDING', 'SUCCEEDED', 'FAILED', 'CANCELED')",
            name=op.f("ck_tasks_state_valid"),
        ),
        sa.CheckConstraint(
            "attempts >= 0 AND attempts <= max_attempts", name=op.f("ck_tasks_attempts_bounded")
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["batches.id"], name=op.f("fk_tasks_batch_id_batches"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tasks")),
        sa.UniqueConstraint("batch_id", "ordinal", name=op.f("uq_tasks_batch_id_ordinal")),
        sa.UniqueConstraint("batch_id", "spec_hash", name=op.f("uq_tasks_batch_id_spec_hash")),
    )
    op.create_index("ix_tasks_batch_state", "tasks", ["batch_id", "state"], unique=False)
    op.create_index(
        "ix_tasks_dispatch",
        "tasks",
        [sa.literal_column("priority DESC"), "ready_at"],
        unique=False,
        postgresql_where=sa.text("state IN ('PENDING', 'RETRY_PENDING')"),
    )
    op.create_index(
        "ix_tasks_lease_expiry",
        "tasks",
        ["lease_expiry"],
        unique=False,
        postgresql_where=sa.text("state = 'RUNNING'"),
    )
    op.create_index(
        "ix_tasks_lease_owner",
        "tasks",
        ["lease_owner"],
        unique=False,
        postgresql_where=sa.text("state = 'RUNNING'"),
    )
    op.create_table(
        "result_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("spec_hash", sa.String(length=64), nullable=False),
        sa.Column("output_uri", sa.Text(), nullable=False),
        sa.Column("output_sha", sa.String(length=64), nullable=False),
        sa.Column("temp_uri", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("code_sha", sa.String(length=64), nullable=False),
        sa.Column("env_sha", sa.String(length=64), nullable=False),
        sa.Column("cached_from_task_id", sa.Uuid(), nullable=True),
        sa.Column(
            "finalized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["batches.id"],
            name=op.f("fk_result_artifacts_batch_id_batches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name=op.f("fk_result_artifacts_task_id_tasks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_result_artifacts")),
        sa.UniqueConstraint("task_id", name=op.f("uq_result_artifacts_task_id")),
    )
    op.create_index("ix_result_artifacts_batch_id", "result_artifacts", ["batch_id"], unique=False)
    op.create_index(
        "ix_result_artifacts_cache",
        "result_artifacts",
        ["spec_hash", "code_sha", "env_sha"],
        unique=False,
    )
    op.create_table(
        "task_attempts",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queue_wait_ms", sa.BigInteger(), nullable=True),
        sa.Column("wall_ms", sa.BigInteger(), nullable=True),
        sa.Column("cpu_ms", sa.BigInteger(), nullable=True),
        sa.Column("peak_rss_mb", sa.Float(), nullable=True),
        sa.Column("output_sha", sa.String(length=64), nullable=True),
        sa.Column("late_completion_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("code_sha", sa.String(length=64), nullable=True),
        sa.Column("env_sha", sa.String(length=64), nullable=True),
        sa.Column("trace_id", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "status IN ('RUNNING', 'SUCCEEDED', 'CACHED', 'FAILED', 'LEASE_EXPIRED', 'CANCELED')",
            name=op.f("ck_task_attempts_status_valid"),
        ),
        sa.CheckConstraint("attempt >= 1", name=op.f("ck_task_attempts_attempt_positive")),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["batches.id"],
            name=op.f("fk_task_attempts_batch_id_batches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name=op.f("fk_task_attempts_task_id_tasks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_attempts")),
        sa.UniqueConstraint("task_id", "attempt", name=op.f("uq_task_attempts_task_id_attempt")),
    )
    op.create_index("ix_task_attempts_batch_id", "task_attempts", ["batch_id"], unique=False)
    op.create_index(
        "ix_task_attempts_running_worker",
        "task_attempts",
        ["worker_id"],
        unique=False,
        postgresql_where=sa.text("status = 'RUNNING'"),
    )
    op.create_table(
        "task_transitions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=False),
        sa.Column("from_state", sa.String(length=16), nullable=True),
        sa.Column("to_state", sa.String(length=16), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("actor", sa.String(length=160), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "at",
            sa.DateTime(timezone=True),
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name=op.f("fk_task_transitions_task_id_tasks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_transitions")),
    )
    op.create_index("ix_task_transitions_batch_id", "task_transitions", ["batch_id"], unique=False)
    op.create_index(
        "ix_task_transitions_task_id_id", "task_transitions", ["task_id", "id"], unique=False
    )
    op.execute(TRIGGERS_SQL)


def downgrade() -> None:
    op.execute(DROP_TRIGGERS_SQL)
    op.drop_index("ix_task_transitions_task_id_id", table_name="task_transitions")
    op.drop_index("ix_task_transitions_batch_id", table_name="task_transitions")
    op.drop_table("task_transitions")
    op.drop_index(
        "ix_task_attempts_running_worker",
        table_name="task_attempts",
        postgresql_where=sa.text("status = 'RUNNING'"),
    )
    op.drop_index("ix_task_attempts_batch_id", table_name="task_attempts")
    op.drop_table("task_attempts")
    op.drop_index("ix_result_artifacts_cache", table_name="result_artifacts")
    op.drop_index("ix_result_artifacts_batch_id", table_name="result_artifacts")
    op.drop_table("result_artifacts")
    op.drop_index(
        "ix_tasks_lease_owner", table_name="tasks", postgresql_where=sa.text("state = 'RUNNING'")
    )
    op.drop_index(
        "ix_tasks_lease_expiry", table_name="tasks", postgresql_where=sa.text("state = 'RUNNING'")
    )
    op.drop_index(
        "ix_tasks_dispatch",
        table_name="tasks",
        postgresql_where=sa.text("state IN ('PENDING', 'RETRY_PENDING')"),
    )
    op.drop_index("ix_tasks_batch_state", table_name="tasks")
    op.drop_table("tasks")
    op.drop_index("ix_worker_heartbeats_last_seen", table_name="worker_heartbeats")
    op.drop_table("worker_heartbeats")
    op.drop_index("ix_batches_user_state", table_name="batches")
    op.drop_index(
        "ix_batches_active",
        table_name="batches",
        postgresql_where=sa.text("state IN ('ACCEPTED', 'RUNNING', 'CANCELING')"),
    )
    op.drop_table("batches")
