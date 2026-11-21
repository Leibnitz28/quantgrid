"""SQLAlchemy table definitions (schema source of truth together with migration 0001).

Triggers installed by the migration additionally enforce:

* the task state machine (``domain/states.py``) and immutability of task specs,
* immutability of terminal attempts and of published result artifacts,
* the batch state machine,
* an append-only ``task_transitions`` audit log of every task state change.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, cast

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from quantgrid.domain.states import AttemptStatus, BatchState, TaskState

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

WORKER_STATES = ("ACTIVE", "STOPPING", "STOPPED", "DEAD")


def _in(column: str, values: tuple[str, ...] | list[str]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


TASK_STATE_VALUES = tuple(s.value for s in TaskState)
BATCH_STATE_VALUES = tuple(s.value for s in BatchState)
ATTEMPT_STATUS_VALUES = tuple(s.value for s in AttemptStatus)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONB}


TZ = DateTime(timezone=True)


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128))
    idempotency_key: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120))
    state: Mapped[str] = mapped_column(String(32))
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    spec_sha: Mapped[str] = mapped_column(String(64))
    dataset_name: Mapped[str] = mapped_column(String(64))
    dataset_sha: Mapped[str] = mapped_column(String(64))
    code_sha: Mapped[str] = mapped_column(String(64))
    env_sha: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB)
    quota: Mapped[dict[str, Any]] = mapped_column(JSONB)
    task_count: Mapped[int] = mapped_column(Integer)
    max_attempts: Mapped[int] = mapped_column(Integer)
    priority: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    cache_policy: Mapped[str] = mapped_column(String(16))
    reproduces_batch_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("batches.id"), nullable=True
    )
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key"),
        CheckConstraint(_in("state", BATCH_STATE_VALUES), name="state_valid"),
        CheckConstraint("cache_policy IN ('use', 'bypass')", name="cache_policy_valid"),
        CheckConstraint("task_count > 0", name="task_count_positive"),
        Index("ix_batches_user_state", "user_id", "state"),
        Index(
            "ix_batches_active",
            "state",
            postgresql_where=text("state IN ('ACCEPTED', 'RUNNING', 'CANCELING')"),
        ),
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    batch_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("batches.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB)
    spec_hash: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(16))
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(Integer)
    max_runtime_s: Mapped[int] = mapped_column(Integer)
    max_memory_mb: Mapped[int] = mapped_column(Integer)
    priority: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    cache_policy: Mapped[str] = mapped_column(String(16))
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expiry: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    ready_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())
    not_before: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    enqueued_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    dispatch_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    checksum_mismatch: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("batch_id", "spec_hash"),
        UniqueConstraint("batch_id", "ordinal"),
        CheckConstraint(_in("state", TASK_STATE_VALUES), name="state_valid"),
        CheckConstraint(
            "(state = 'RUNNING') = (lease_owner IS NOT NULL AND lease_expiry IS NOT NULL)",
            name="lease_iff_running",
        ),
        CheckConstraint("attempts >= 0 AND attempts <= max_attempts", name="attempts_bounded"),
        CheckConstraint("cache_policy IN ('use', 'bypass')", name="cache_policy_valid"),
        Index(
            "ix_tasks_dispatch",
            text("priority DESC"),
            "ready_at",
            postgresql_where=text("state IN ('PENDING', 'RETRY_PENDING')"),
        ),
        Index("ix_tasks_lease_expiry", "lease_expiry", postgresql_where=text("state = 'RUNNING'")),
        Index("ix_tasks_lease_owner", "lease_owner", postgresql_where=text("state = 'RUNNING'")),
        Index("ix_tasks_batch_state", "batch_id", "state"),
    )


class TaskAttempt(Base):
    __tablename__ = "task_attempts"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    task_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("tasks.id", ondelete="CASCADE"))
    batch_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("batches.id", ondelete="CASCADE"))
    attempt: Mapped[int] = mapped_column(Integer)
    worker_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(TZ)
    ended_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    queue_wait_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    wall_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cpu_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    peak_rss_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    late_completion_at: Mapped[datetime | None] = mapped_column(TZ, nullable=True)
    code_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    env_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("task_id", "attempt"),
        CheckConstraint(_in("status", ATTEMPT_STATUS_VALUES), name="status_valid"),
        CheckConstraint("attempt >= 1", name="attempt_positive"),
        Index("ix_task_attempts_batch_id", "batch_id"),
        Index(
            "ix_task_attempts_running_worker",
            "worker_id",
            postgresql_where=text("status = 'RUNNING'"),
        ),
    )


class ResultArtifact(Base):
    __tablename__ = "result_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("tasks.id", ondelete="CASCADE"), unique=True
    )
    batch_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("batches.id", ondelete="CASCADE"))
    attempt: Mapped[int] = mapped_column(Integer)
    spec_hash: Mapped[str] = mapped_column(String(64))
    output_uri: Mapped[str] = mapped_column(Text)
    output_sha: Mapped[str] = mapped_column(String(64))
    temp_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    row_count: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    code_sha: Mapped[str] = mapped_column(String(64))
    env_sha: Mapped[str] = mapped_column(String(64))
    cached_from_task_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    finalized_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())

    __table_args__ = (
        Index("ix_result_artifacts_batch_id", "batch_id"),
        Index("ix_result_artifacts_cache", "spec_hash", "code_sha", "env_sha"),
    )


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    hostname: Mapped[str] = mapped_column(String(255))
    pid: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(TZ)
    last_seen: Mapped[datetime] = mapped_column(TZ)
    concurrency: Mapped[int] = mapped_column(Integer, server_default=text("1"))
    inflight_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    resource_profile: Mapped[dict[str, Any]] = mapped_column(JSONB)
    code_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    env_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        CheckConstraint(_in("state", WORKER_STATES), name="state_valid"),
        Index("ix_worker_heartbeats_last_seen", "last_seen"),
    )


class TaskTransition(Base):
    """Append-only audit log written by the ``qg_tasks_log_transition`` trigger."""

    __tablename__ = "task_transitions"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    task_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("tasks.id", ondelete="CASCADE"))
    batch_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    from_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_state: Mapped[str] = mapped_column(String(16))
    attempt: Mapped[int] = mapped_column(Integer)
    actor: Mapped[str] = mapped_column(String(160))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    at: Mapped[datetime] = mapped_column(TZ, server_default=text("clock_timestamp()"))

    __table_args__ = (
        Index("ix_task_transitions_task_id_id", "task_id", "id"),
        Index("ix_task_transitions_batch_id", "batch_id"),
    )


batches = cast(Table, Batch.__table__)
tasks = cast(Table, Task.__table__)
task_attempts = cast(Table, TaskAttempt.__table__)
result_artifacts = cast(Table, ResultArtifact.__table__)
worker_heartbeats = cast(Table, WorkerHeartbeat.__table__)
task_transitions = cast(Table, TaskTransition.__table__)
