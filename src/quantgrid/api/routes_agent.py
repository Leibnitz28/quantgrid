"""AI endpoints. The planner only *proposes* a spec (never enqueues); the explainer only
reads progress through read-only tools. Deterministic policy checks decide everything."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from quantgrid.api.context import AppContext, get_ctx, get_planner_principal
from quantgrid.api.errors import error_response
from quantgrid.api.schemas import ExplainResponse, PlannerRequest, PlannerResponse
from quantgrid.api.security import Principal
from quantgrid.domain.errors import ErrorCode
from quantgrid.storage import repository as repo


def _count_planner(status: str) -> None:
    from quantgrid.observability.metrics import planner_requests_total

    planner_requests_total.labels(status=status).inc()


router = APIRouter(prefix="/v1", tags=["ai"])


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    return obj


class RepositoryReadModel:
    """Read-only view of scheduler state for the explainer agent's tools."""

    def __init__(self, ctx: AppContext, user_id: str) -> None:
        self._ctx = ctx
        self._user = user_id

    def get_progress(self, batch_id: str) -> dict[str, Any]:
        b = repo.get_batch(self._ctx.engine, batch_id, user_id=self._user)
        progress = repo.batch_progress(self._ctx.engine, batch_id)
        out: dict[str, Any] = _plain(
            {"batch_id": str(b["id"]), "name": b["name"], "state": b["state"], **progress}
        )
        return out

    def list_failed_tasks(self, batch_id: str, limit: int = 20) -> list[dict[str, Any]]:
        repo.get_batch(self._ctx.engine, batch_id, user_id=self._user)
        return [
            {
                "task_id": str(row["id"]),
                "ordinal": row["ordinal"],
                "error_code": row["last_error_code"],
                "error_message": row["last_error_message"],
            }
            for row in repo.failed_tasks(self._ctx.engine, batch_id, limit=min(limit, 50))
        ]

    def get_task_attempts(self, task_id: str) -> list[dict[str, Any]]:
        detail = repo.get_task_detail(self._ctx.engine, task_id, user_id=self._user)
        keep = (
            "attempt",
            "worker_id",
            "status",
            "error_code",
            "error_message",
            "wall_ms",
            "started_at",
            "ended_at",
        )
        return [
            {
                k: _plain(str(a[k]) if k in ("started_at", "ended_at") and a[k] else a[k])
                for k in keep
            }
            for a in detail["attempts"]
        ]


@router.post("/planner/specs", response_model=PlannerResponse)
def plan_spec(
    req: PlannerRequest,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_planner_principal)],
) -> Any:
    """Generate a bounded, typed ExperimentBatchSpec from a natural-language goal.

    The response is a *proposal*: it has passed the same policy checks as a submission,
    but nothing is enqueued. A human reviews it and submits it to ``POST /v1/batches``."""
    from quantgrid.agents.planner import PlannerService

    service = PlannerService(
        provider=ctx.provider_factory(),
        policy=ctx.policy,
        catalog=ctx.catalog,
        settings=ctx.settings,
    )
    result = service.plan(
        req.goal,
        quota=ctx.quota_for(principal.user_id),
        max_tasks=req.max_tasks,
        dataset=req.dataset,
    )
    decision = result.decision
    _count_planner(result.status)
    payload = PlannerResponse(
        status=result.status,
        spec=result.spec.model_dump(mode="json") if result.spec is not None else None,
        allowed=bool(decision and decision.allowed),
        violations=[v.model_dump() for v in decision.violations] if decision else [],
        warnings=list(decision.warnings) if decision else [],
        estimate=decision.estimate.model_dump() if decision and decision.estimate else None,
        errors=list(result.errors),
        provider=str(result.provider),
        steps=int(result.steps),
        tool_calls=int(result.tool_calls),
        trace=[_plain(e) for e in result.trace],
        next_step=(
            "review the spec, then submit it with POST /v1/batches and an Idempotency-Key"
            if result.status == "ok"
            else "adjust the goal or constraints and try again"
        ),
    )
    if result.status == "error":
        return error_response(
            502,
            ErrorCode.PLANNER_ERROR,
            "planner failed",
            None,
            details=payload.model_dump(mode="json"),
        )
    return payload


@router.post("/batches/{batch_id}/explain", response_model=ExplainResponse)
def explain(
    batch_id: uuid.UUID,
    ctx: Annotated[AppContext, Depends(get_ctx)],
    principal: Annotated[Principal, Depends(get_planner_principal)],
) -> ExplainResponse | JSONResponse:
    """Read-only AI diagnosis of a batch's progress and failures."""
    from quantgrid.agents.explainer import explain_batch

    repo.get_batch(ctx.engine, batch_id, user_id=principal.user_id)  # 404 for other users
    provider = ctx.provider_factory()
    exp = explain_batch(str(batch_id), RepositoryReadModel(ctx, principal.user_id), provider)
    return ExplainResponse(
        batch_id=batch_id,
        summary=exp.summary,
        findings=list(exp.findings),
        provider=str(getattr(provider, "name", type(provider).__name__)),
        trace=[_plain(e) for e in exp.trace],
    )
