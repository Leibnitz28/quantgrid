"""HTTP request/response models (typed contracts of the public API).

Conventions: IDs are UUID strings; timestamps are UTC ISO-8601 with a ``Z`` suffix;
list endpoints paginate with an opaque ``next_cursor``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, StringConstraints


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


UtcDatetime = Annotated[datetime, PlainSerializer(_utc, return_type=str)]
IdempotencyKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._:-]{1,255}$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Links(_Model):
    self: str
    tasks: str | None = None
    results: str | None = None
    cancel: str | None = None


class BatchAccepted(_Model):
    batch_id: uuid.UUID
    state: str
    task_count: int
    spec_sha: str
    idempotent_replay: bool
    warnings: list[str] = Field(default_factory=list)
    estimate: dict[str, Any] | None = None
    links: Links


class BatchSummary(_Model):
    batch_id: uuid.UUID
    name: str
    state: str
    task_count: int
    submitted_at: UtcDatetime
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None


class BatchPage(_Model):
    items: list[BatchSummary]
    next_cursor: str | None


class BatchStatus(_Model):
    batch_id: uuid.UUID
    name: str
    state: str
    user_id: str
    task_count: int
    progress: dict[str, Any]
    cost: dict[str, Any]
    cancel_requested: bool
    submitted_at: UtcDatetime
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None
    elapsed_s: float | None
    spec_sha: str
    dataset: dict[str, str]
    code_sha: str
    env_sha: str
    git_commit: str | None
    request_id: str | None
    links: Links


class TaskRow(_Model):
    task_id: uuid.UUID
    ordinal: int
    state: str
    attempts: int
    max_attempts: int
    spec_hash: str
    lease_owner: str | None
    last_error_code: str | None
    cache_hit: bool
    checksum_mismatch: bool
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None


class TaskPage(_Model):
    items: list[TaskRow]
    next_cursor: str | None


class AttemptOut(_Model):
    attempt: int
    worker_id: str
    status: str
    error_type: str | None
    error_code: str | None
    error_message: str | None
    started_at: UtcDatetime
    ended_at: UtcDatetime | None
    queue_wait_ms: int | None
    wall_ms: int | None
    cpu_ms: int | None
    peak_rss_mb: float | None
    output_sha: str | None
    late_completion_at: UtcDatetime | None
    code_sha: str | None
    trace_id: str | None


class ArtifactOut(_Model):
    artifact_id: uuid.UUID
    output_uri: str
    output_sha: str
    size_bytes: int
    row_count: int
    attempt: int
    temp_uri: str | None
    cached_from_task_id: uuid.UUID | None
    finalized_at: UtcDatetime
    metrics: dict[str, Any]


class TransitionOut(_Model):
    from_state: str | None
    to_state: str
    attempt: int
    actor: str
    reason: str | None
    at: UtcDatetime


class TaskDetail(_Model):
    task_id: uuid.UUID
    batch_id: uuid.UUID
    ordinal: int
    state: str
    spec_hash: str
    spec: dict[str, Any]
    attempts_used: int
    max_attempts: int
    lease_owner: str | None
    lease_expiry: UtcDatetime | None
    cancel_requested: bool
    cache_hit: bool
    checksum_mismatch: bool
    last_error: dict[str, Any] | None
    worker_ids: list[str]
    attempts: list[AttemptOut]
    artifact: ArtifactOut | None
    transitions: list[TransitionOut]


class CancelResponse(_Model):
    batch_id: uuid.UUID
    state: str
    already_terminal: bool
    canceled_now: int
    cancel_requested_running: int
    completed_count: int
    failed_count: int
    completed_task_ids: list[uuid.UUID]
    completed_ids_truncated: bool


class RetryRequest(_Model):
    extra_attempts: int = Field(default=1, ge=1, le=5)
    include_non_retryable: bool = False


class RetryResponse(_Model):
    batch_id: uuid.UUID
    state: str
    retried: int
    retried_task_ids: list[uuid.UUID]
    skipped_not_retryable: int
    skipped_attempt_quota: int


class ResultRow(_Model):
    task_id: uuid.UUID
    ordinal: int
    spec_hash: str
    params: dict[str, Any]
    seed: int
    metrics: dict[str, Any]
    output_uri: str
    output_sha: str
    size_bytes: int
    row_count: int
    attempt: int
    cached_from_task_id: uuid.UUID | None
    finalized_at: UtcDatetime


class ResultsPage(_Model):
    batch_id: uuid.UUID
    state: str
    complete: bool
    counts: dict[str, int]
    summary: dict[str, Any]
    manifest: dict[str, Any]
    results: list[ResultRow]
    next_cursor: str | None


class PlannerRequest(_Model):
    goal: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    max_tasks: int | None = Field(default=None, ge=1, le=100_000)
    dataset: str | None = Field(default=None, max_length=64)


class PlannerResponse(_Model):
    status: Literal["ok", "rejected", "invalid_output", "error"]
    spec: dict[str, Any] | None
    allowed: bool
    violations: list[dict[str, Any]]
    warnings: list[str]
    estimate: dict[str, Any] | None
    errors: list[str]
    provider: str
    steps: int
    tool_calls: int
    trace: list[dict[str, Any]]
    next_step: str


class ExplainResponse(_Model):
    batch_id: uuid.UUID
    summary: str
    findings: list[str]
    provider: str
    trace: list[dict[str, Any]]
