"""Batch endpoints: submit (idempotent), status, tasks, cancel, retry-failed, results."""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header, Query, Request, Response
from sqlalchemy import text

from quantgrid.api.context import AppContext, get_ctx, get_principal
from quantgrid.api.schemas import (
    BatchAccepted,
    BatchPage,
    BatchStatus,
    BatchSummary,
    CancelResponse,
    IdempotencyKey,
    Links,
    ResultRow,
    ResultsPage,
    RetryRequest,
    RetryResponse,
    TaskPage,
    TaskRow,
)
from quantgrid.api.security import Principal
from quantgrid.domain.errors import Conflict, QuotaExceeded, ValidationFailed
from quantgrid.domain.specs import ExperimentBatchSpec
from quantgrid.domain.states import TaskState
from quantgrid.manifest import build_manifest
from quantgrid.storage import repository as repo

router = APIRouter(prefix="/v1", tags=["batches"])

Ctx = Annotated[AppContext, Depends(get_ctx)]
User = Annotated[Principal, Depends(get_principal)]


def _links(batch_id: uuid.UUID) -> Links:
    base = f"/v1/batches/{batch_id}"
    return Links(
        self=base, tasks=f"{base}/tasks", results=f"{base}/results", cancel=f"{base}/cancel"
    )


def encode_cursor(data: dict[str, Any]) -> str:
    return (
        base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )


def decode_cursor(cursor: str | None) -> dict[str, Any]:
    if not cursor:
        return {}
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValidationFailed("malformed cursor") from exc
    if not isinstance(value, dict):
        raise ValidationFailed("malformed cursor")
    return value


def _cursor_int(cursor: str | None, key: str = "o") -> int:
    value = decode_cursor(cursor).get(key, -1)
    if not isinstance(value, int):
        raise ValidationFailed("malformed cursor")
    return value


def _metric_inc(name: str, **labels: str) -> None:
    try:
        from quantgrid.observability import metrics
    except ImportError:  # pragma: no cover
        return
    metric = getattr(metrics, name, None)
    if metric is not None:
        (metric.labels(**labels) if labels else metric).inc()


@router.post("/batches", status_code=202, response_model=BatchAccepted)
def submit_batch(
    spec: ExperimentBatchSpec,
    request: Request,
    response: Response,
    ctx: Ctx,
    principal: User,
    idempotency_key: Annotated[IdempotencyKey, Header(alias="Idempotency-Key")],
) -> BatchAccepted:
    """Submit a bounded experiment batch. Idempotent per ``Idempotency-Key``: repeating a
    request returns the original batch; reusing a key for a different spec is a 409."""
    spec_sha = spec.spec_sha()
    existing = repo.find_batch_by_key(ctx.engine, principal.user_id, idempotency_key)
    if existing is not None:
        if existing["spec_sha"] != spec_sha:
            _metric_inc("batches_submitted_total", outcome="conflict")
            raise Conflict(
                "Idempotency-Key already used for a different batch spec",
                details={"batch_id": str(existing["id"])},
            )
        _metric_inc("batches_submitted_total", outcome="replay")
        response.headers["Idempotent-Replayed"] = "true"
        response.headers["Location"] = f"/v1/batches/{existing['id']}"
        return BatchAccepted(
            batch_id=existing["id"],
            state=existing["state"],
            task_count=existing["task_count"],
            spec_sha=spec_sha,
            idempotent_replay=True,
            links=_links(existing["id"]),
        )

    quota = ctx.quota_for(principal.user_id)
    active = repo.active_batch_count(ctx.engine, principal.user_id)
    decision, admitted = ctx.policy.admit(spec, quota, active_batches=active)
    if admitted is None:
        _metric_inc("batches_submitted_total", outcome="rejected")
        details = {
            "violations": [v.model_dump() for v in decision.violations],
            "warnings": decision.warnings,
            "estimate": decision.estimate.model_dump() if decision.estimate else None,
        }
        if decision.error_type() == "QUOTA_EXCEEDED":
            raise QuotaExceeded("batch exceeds quota", details=details)
        raise ValidationFailed("batch failed policy validation", details=details)

    result = repo.create_batch(
        ctx.engine,
        user_id=principal.user_id,
        idempotency_key=idempotency_key,
        spec=spec,
        tasks_expanded=admitted.tasks,
        manifest=build_manifest(admitted.dataset, spec_sha),
        quota=quota,
        dataset=admitted.dataset,
        request_id=getattr(request.state, "request_id", None),
    )
    _metric_inc("batches_submitted_total", outcome="accepted" if result.created else "replay")
    response.headers["Idempotent-Replayed"] = "false" if result.created else "true"
    response.headers["Location"] = f"/v1/batches/{result.batch_id}"
    return BatchAccepted(
        batch_id=result.batch_id,
        state=result.state,
        task_count=result.task_count,
        spec_sha=spec_sha,
        idempotent_replay=not result.created,
        warnings=decision.warnings,
        estimate=decision.estimate.model_dump() if decision.estimate else None,
        links=_links(result.batch_id),
    )


@router.get("/batches", response_model=BatchPage)
def list_batches(
    ctx: Ctx,
    principal: User,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: str | None = None,
) -> BatchPage:
    before_raw = decode_cursor(cursor).get("t")
    before = datetime.fromisoformat(before_raw) if isinstance(before_raw, str) else None
    rows = repo.list_batches(ctx.engine, user_id=principal.user_id, before=before, limit=limit + 1)
    items = [
        BatchSummary(
            batch_id=r["id"],
            name=r["name"],
            state=r["state"],
            task_count=r["task_count"],
            submitted_at=r["submitted_at"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
        )
        for r in rows[:limit]
    ]
    next_cursor = (
        encode_cursor({"t": rows[limit - 1]["submitted_at"].isoformat()})
        if len(rows) > limit
        else None
    )
    return BatchPage(items=items, next_cursor=next_cursor)


@router.get("/batches/{batch_id}", response_model=BatchStatus)
def get_batch(batch_id: uuid.UUID, ctx: Ctx, principal: User) -> BatchStatus:
    """Progress counts, total cost (worker-seconds) and status."""
    b = repo.get_batch(ctx.engine, batch_id, user_id=principal.user_id)
    progress = repo.batch_progress(ctx.engine, batch_id)
    end = b["finished_at"] or datetime.now(UTC)
    code = b["manifest"].get("code", {})
    return BatchStatus(
        batch_id=b["id"],
        name=b["name"],
        state=b["state"],
        user_id=b["user_id"],
        task_count=b["task_count"],
        progress={k: v for k, v in progress.items() if k != "cost"},
        cost=progress["cost"],
        cancel_requested=b["cancel_requested_at"] is not None,
        submitted_at=b["submitted_at"],
        started_at=b["started_at"],
        finished_at=b["finished_at"],
        elapsed_s=round((end - b["submitted_at"]).total_seconds(), 3),
        spec_sha=b["spec_sha"],
        dataset={"name": b["dataset_name"], "sha256": b["dataset_sha"]},
        code_sha=b["code_sha"],
        env_sha=b["env_sha"],
        git_commit=code.get("git_commit"),
        request_id=b["request_id"],
        links=_links(b["id"]),
    )


@router.get("/batches/{batch_id}/tasks", response_model=TaskPage)
def list_batch_tasks(
    batch_id: uuid.UUID,
    ctx: Ctx,
    principal: User,
    state: TaskState | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    cursor: str | None = None,
) -> TaskPage:
    repo.get_batch(ctx.engine, batch_id, user_id=principal.user_id)  # ownership / 404
    rows = repo.list_tasks(
        ctx.engine,
        batch_id,
        state=state.value if state else None,
        after_ordinal=_cursor_int(cursor),
        limit=limit + 1,
    )
    items = [
        TaskRow(
            task_id=r["id"],
            ordinal=r["ordinal"],
            state=r["state"],
            attempts=r["attempts"],
            max_attempts=r["max_attempts"],
            spec_hash=r["spec_hash"],
            lease_owner=r["lease_owner"],
            last_error_code=r["last_error_code"],
            cache_hit=r["cache_hit"],
            checksum_mismatch=r["checksum_mismatch"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
        )
        for r in rows[:limit]
    ]
    next_cursor = encode_cursor({"o": rows[limit - 1]["ordinal"]}) if len(rows) > limit else None
    return TaskPage(items=items, next_cursor=next_cursor)


@router.post("/batches/{batch_id}/cancel", response_model=CancelResponse)
def cancel_batch(batch_id: uuid.UUID, ctx: Ctx, principal: User) -> CancelResponse:
    """Cooperative cancellation; reports the items that had already completed."""
    report = repo.cancel_batch(
        ctx.engine, batch_id, user_id=principal.user_id, actor=f"api:{principal.user_id}"
    )
    return CancelResponse(
        batch_id=report.batch_id,
        state=report.state,
        already_terminal=report.already_terminal,
        canceled_now=report.canceled_now,
        cancel_requested_running=report.cancel_requested_running,
        completed_count=report.completed_count,
        failed_count=report.failed_count,
        completed_task_ids=report.completed_task_ids,
        completed_ids_truncated=report.completed_ids_truncated,
    )


@router.post("/batches/{batch_id}/retry-failed", response_model=RetryResponse)
def retry_failed(
    batch_id: uuid.UUID,
    ctx: Ctx,
    principal: User,
    body: Annotated[RetryRequest | None, Body()] = None,
) -> RetryResponse:
    """Retry eligible FAILED tasks (retryable error codes unless include_non_retryable),
    within the batch's attempt and worker-time quota."""
    req = body or RetryRequest()
    report = repo.retry_failed(
        ctx.engine,
        batch_id,
        user_id=principal.user_id,
        extra_attempts=req.extra_attempts,
        include_non_retryable=req.include_non_retryable,
        actor=f"api:{principal.user_id}",
    )
    return RetryResponse(
        batch_id=report.batch_id,
        state=report.state,
        retried=report.retried,
        retried_task_ids=report.retried_task_ids,
        skipped_not_retryable=report.skipped_not_retryable,
        skipped_attempt_quota=report.skipped_attempt_quota,
    )


@router.get("/batches/{batch_id}/results", response_model=ResultsPage)
def get_results(
    batch_id: uuid.UUID,
    ctx: Ctx,
    principal: User,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    cursor: str | None = None,
) -> ResultsPage:
    """Merged per-task metrics plus the reproducibility manifest. Only published
    (immutable) results are returned; ``complete`` says whether the batch is terminal."""
    from quantgrid.results.export import summarize_metrics

    b = repo.get_batch(ctx.engine, batch_id, user_id=principal.user_id)
    rows = repo.list_results(
        ctx.engine, batch_id, after_ordinal=_cursor_int(cursor), limit=limit + 1
    )
    with ctx.engine.connect() as conn:
        all_metrics = (
            conn.execute(
                text(
                    "SELECT r.task_id, t.ordinal, t.spec_hash, r.metrics FROM result_artifacts r "
                    "JOIN tasks t ON t.id = r.task_id WHERE r.batch_id = :id"
                ),
                {"id": batch_id},
            )
            .mappings()
            .all()
        )
    summary = dict(summarize_metrics([dict(m) for m in all_metrics], key="sharpe"))
    counts = repo.batch_progress(ctx.engine, batch_id)["counts"]
    results = [
        ResultRow(
            task_id=r["task_id"],
            ordinal=r["ordinal"],
            spec_hash=r["spec_hash"],
            params=r["spec"]["params"],
            seed=r["spec"]["seed"],
            metrics=r["metrics"],
            output_uri=r["output_uri"],
            output_sha=r["output_sha"],
            size_bytes=r["size_bytes"],
            row_count=r["row_count"],
            attempt=r["attempt"],
            cached_from_task_id=r["cached_from_task_id"],
            finalized_at=r["finalized_at"],
        )
        for r in rows[:limit]
    ]
    manifest = {
        **b["manifest"],
        "batch_spec": b["spec"],
        "batch_spec_sha": b["spec_sha"],
        "quota": b["quota"],
    }
    next_cursor = encode_cursor({"o": rows[limit - 1]["ordinal"]}) if len(rows) > limit else None
    return ResultsPage(
        batch_id=b["id"],
        state=b["state"],
        complete=b["state"] in ("COMPLETED", "COMPLETED_WITH_FAILURES", "CANCELED"),
        counts=counts,
        summary=summary,
        manifest=manifest,
        results=results,
        next_cursor=next_cursor,
    )
