"""Task endpoint: attempt log, worker ids, artifact ids, audited transitions."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from quantgrid.api.context import AppContext, get_ctx, get_principal
from quantgrid.api.schemas import ArtifactOut, AttemptOut, TaskDetail, TransitionOut
from quantgrid.api.security import Principal
from quantgrid.storage import repository as repo

router = APIRouter(prefix="/v1", tags=["tasks"])


def _error_type(code: str | None) -> str | None:
    return None if code is None else "WORKER_FAILURE"


@router.get("/tasks/{task_id}", response_model=TaskDetail)
def get_task(
    task_id: uuid.UUID,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_principal)],
) -> TaskDetail:
    d = repo.get_task_detail(ctx.engine, task_id, user_id=principal.user_id)
    t: dict[str, Any] = d["task"]
    art = d["artifact"]
    return TaskDetail(
        task_id=t["id"],
        batch_id=t["batch_id"],
        ordinal=t["ordinal"],
        state=t["state"],
        spec_hash=t["spec_hash"],
        spec=t["spec"],
        attempts_used=t["attempts"],
        max_attempts=t["max_attempts"],
        lease_owner=t["lease_owner"],
        lease_expiry=t["lease_expiry"],
        cancel_requested=t["cancel_requested"],
        cache_hit=t["cache_hit"],
        checksum_mismatch=t["checksum_mismatch"],
        last_error=(
            {
                "type": "WORKER_FAILURE",
                "code": t["last_error_code"],
                "message": t["last_error_message"],
            }
            if t["last_error_code"]
            else None
        ),
        worker_ids=sorted({a["worker_id"] for a in d["attempts"]}),
        attempts=[
            AttemptOut(
                attempt=a["attempt"],
                worker_id=a["worker_id"],
                status=a["status"],
                error_type=_error_type(a["error_code"]),
                error_code=a["error_code"],
                error_message=a["error_message"],
                started_at=a["started_at"],
                ended_at=a["ended_at"],
                queue_wait_ms=a["queue_wait_ms"],
                wall_ms=a["wall_ms"],
                cpu_ms=a["cpu_ms"],
                peak_rss_mb=a["peak_rss_mb"],
                output_sha=a["output_sha"],
                late_completion_at=a["late_completion_at"],
                code_sha=a["code_sha"],
                trace_id=a["trace_id"],
            )
            for a in d["attempts"]
        ],
        artifact=(
            ArtifactOut(
                artifact_id=art["id"],
                output_uri=art["output_uri"],
                output_sha=art["output_sha"],
                size_bytes=art["size_bytes"],
                row_count=art["row_count"],
                attempt=art["attempt"],
                temp_uri=art["temp_uri"],
                cached_from_task_id=art["cached_from_task_id"],
                finalized_at=art["finalized_at"],
                metrics=art["metrics"],
            )
            if art
            else None
        ),
        transitions=[TransitionOut(**tr) for tr in d["transitions"]],
    )
