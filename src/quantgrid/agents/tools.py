"""Read-only tool registry for the planner and explainer roles.

Invariants:
* Every tool is ``read_only``; the registry refuses to hold anything else, and no tool
  here submits, cancels, modifies or deletes. There is no shell, file or network tool.
* Each role has an explicit allowlist; a tool outside it is unreachable even if it
  exists in the registry.
* Arguments are untrusted: size-limited, then validated strictly against a pydantic
  model with ``extra="forbid"``. Handler exceptions become error results; oversized
  results are truncated with a marker.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from quantgrid.agents.llm import ToolResult, ToolSpec
from quantgrid.domain.policy import DatasetCatalog, PolicyService, QuotaPolicy
from quantgrid.domain.specs import (
    ExperimentBatchSpec,
    StrategyRef,
    registered_strategies,
    strategy_params_model,
)

Role = Literal["planner", "explainer"]

MAX_ARGS_BYTES = 32 * 1024
MAX_RESULT_CHARS = 16_000
TRUNCATION_MARKER = "...[truncated by QuantGrid tool result cap]"

PLANNER_TOOLS: frozenset[str] = frozenset(
    {
        "list_datasets",
        "get_dataset_info",
        "list_strategies",
        "get_strategy_schema",
        "get_quota",
        "estimate_cost",
        "validate_spec",
    }
)
EXPLAINER_TOOLS: frozenset[str] = frozenset(
    {"get_batch_progress", "list_failed_tasks", "get_task_attempts"}
)

_Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoArgs(_Args):
    pass


class DatasetArgs(_Args):
    name: Annotated[str, StringConstraints(min_length=1, max_length=64)]


class StrategyArgs(_Args):
    name: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    version: Annotated[str, StringConstraints(min_length=1, max_length=32)]


class SpecArgs(_Args):
    spec: dict[str, Any]


class BatchArgs(_Args):
    batch_id: _Id


class FailedArgs(_Args):
    batch_id: _Id
    limit: int = Field(default=20, ge=1, le=50)


class TaskArgs(_Args):
    task_id: _Id


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], Any]
    read_only: bool = True


class BatchReadModel(Protocol):
    """Read-only view of batch state (plain dicts). The API wires the real one later.

    Expected shapes (extra keys are ignored):
    ``get_progress`` -> ``{"state": str, "total": int, "counts": {STATE: int}}``;
    ``list_failed_tasks`` -> ``[{"task_id", "ordinal", "error_code", "error_message"}]``;
    ``get_task_attempts`` -> ``[{"attempt", "status", "error_code", "worker_id"}]``.
    """

    def get_progress(self, batch_id: str) -> dict[str, Any]: ...

    def list_failed_tasks(self, batch_id: str, limit: int) -> list[dict[str, Any]]: ...

    def get_task_attempts(self, task_id: str) -> list[dict[str, Any]]: ...


class ToolRegistry:
    """A set of read-only tools plus a per-role allowlist."""

    def __init__(self, tools: Sequence[Tool], allowlists: Mapping[Role, frozenset[str]]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if not tool.read_only:
                raise ValueError(f"tool {tool.name!r} is not read-only; refusing to register")
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool {tool.name!r}")
            self._tools[tool.name] = tool
        self._allow = {role: frozenset(names) for role, names in allowlists.items()}
        for role, names in self._allow.items():
            missing = names - set(self._tools)
            if missing:
                raise ValueError(f"allowlist for {role} names unknown tools: {sorted(missing)}")

    def tool_names(self, role: Role) -> list[str]:
        return sorted(self._allow.get(role, frozenset()))

    def specs(self, role: Role) -> list[ToolSpec]:
        out = []
        for name in self.tool_names(role):
            tool = self._tools[name]
            out.append(
                ToolSpec(
                    name=name,
                    description=tool.description,
                    input_schema=tool.args_model.model_json_schema(),
                )
            )
        return out

    def invoke(self, role: Role, name: str, raw_args: Any, *, call_id: str = "") -> ToolResult:
        """Run one tool call. Never raises; a rejected call is an error result."""

        def fail(msg: str) -> ToolResult:
            return ToolResult(call_id=call_id, ok=False, error=msg)

        if name not in self._allow.get(role, frozenset()) or name not in self._tools:
            return fail(f"unknown or not permitted tool {name[:64]!r}")
        if not isinstance(raw_args, dict):
            return fail("arguments must be a JSON object")
        try:
            raw_text = json.dumps(raw_args, default=str)
        except (TypeError, ValueError):
            return fail("arguments are not JSON-serialisable")
        if len(raw_text.encode()) > MAX_ARGS_BYTES:
            return fail(f"arguments exceed {MAX_ARGS_BYTES} bytes")
        tool = self._tools[name]
        try:
            args = tool.args_model.model_validate(raw_args, strict=True)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()[:8]
            )
            return fail(f"invalid arguments: {detail}")
        try:
            content = tool.handler(args)
            text = json.dumps(content, default=str, sort_keys=True)
        except Exception as exc:
            return fail(f"{type(exc).__name__}: {str(exc)[:300]}")
        if len(text) > MAX_RESULT_CHARS:
            content = {"truncated": True, "preview": text[:MAX_RESULT_CHARS] + TRUNCATION_MARKER}
        else:
            content = json.loads(text)
        return ToolResult(call_id=call_id, ok=True, content=content)


# --------------------------------------------------------------------------------------
# Planner tools
# --------------------------------------------------------------------------------------


def _dataset_dict(info: Any) -> dict[str, Any]:
    dumped: dict[str, Any] = info.model_dump(mode="json")
    return dumped


def planner_tools(policy: PolicyService, catalog: DatasetCatalog, quota: QuotaPolicy) -> list[Tool]:
    """Planner tools bound to one (policy, catalog, quota). Dry-run only; no persistence."""

    def list_datasets(_: NoArgs) -> Any:
        rows = []
        for ds in catalog.list():
            if quota.allowed_datasets is not None and ds.name not in quota.allowed_datasets:
                continue
            rows.append(
                {
                    "name": ds.name,
                    "n_bars": ds.n_bars,
                    "bar_interval_s": ds.bar_interval_s,
                    "description": ds.description,
                }
            )
        return sorted(rows, key=lambda r: str(r["name"]))

    def get_dataset_info(args: DatasetArgs) -> Any:
        ds = catalog.get(args.name)
        if ds is None or (
            quota.allowed_datasets is not None and ds.name not in quota.allowed_datasets
        ):
            raise LookupError(f"unknown dataset {args.name!r}")
        return _dataset_dict(ds)

    def list_strategies(_: NoArgs) -> Any:
        return [
            {"name": n, "version": v}
            for n, v in registered_strategies()
            if quota.allowed_strategies is None or n in quota.allowed_strategies
        ]

    def get_strategy_schema(args: StrategyArgs) -> Any:
        model = strategy_params_model(StrategyRef(name=args.name, version=args.version))
        if model is None:
            raise LookupError(f"unknown strategy {args.name}@{args.version}")
        return model.model_json_schema()

    def get_quota(_: NoArgs) -> Any:
        return quota.model_dump(mode="json")

    def _parse(spec: dict[str, Any]) -> tuple[ExperimentBatchSpec | None, list[str]]:
        try:
            return ExperimentBatchSpec.model_validate(spec), []
        except ValidationError as exc:
            return None, [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:10]
            ]

    def validate_spec(args: SpecArgs) -> Any:
        spec, errors = _parse(args.spec)
        if spec is None:
            return {"valid": False, "allowed": False, "errors": errors}
        return {"valid": True, **policy.evaluate(spec, quota).model_dump(mode="json")}

    def estimate_cost(args: SpecArgs) -> Any:
        spec, errors = _parse(args.spec)
        if spec is None:
            return {"valid": False, "errors": errors}
        decision = policy.evaluate(spec, quota)
        est = decision.estimate
        return {
            "valid": True,
            "task_count": decision.task_count,
            "estimate": None if est is None else est.model_dump(mode="json"),
        }

    return [
        Tool("list_datasets", "List datasets available to you.", NoArgs, list_datasets),
        Tool(
            "get_dataset_info",
            "Describe one dataset (incl. sha256).",
            DatasetArgs,
            get_dataset_info,
        ),
        Tool("list_strategies", "List registered strategies.", NoArgs, list_strategies),
        Tool(
            "get_strategy_schema",
            "JSON Schema of a strategy's parameters.",
            StrategyArgs,
            get_strategy_schema,
        ),
        Tool("get_quota", "Your quota limits.", NoArgs, get_quota),
        Tool(
            "estimate_cost",
            "Estimate cost of a draft ExperimentBatchSpec (dry run).",
            SpecArgs,
            estimate_cost,
        ),
        Tool(
            "validate_spec",
            "Dry-run policy check of a draft ExperimentBatchSpec. Nothing is saved.",
            SpecArgs,
            validate_spec,
        ),
    ]


def build_planner_registry(
    policy: PolicyService, catalog: DatasetCatalog, quota: QuotaPolicy
) -> ToolRegistry:
    return ToolRegistry(planner_tools(policy, catalog, quota), {"planner": PLANNER_TOOLS})


# --------------------------------------------------------------------------------------
# Explainer tools
# --------------------------------------------------------------------------------------


def explainer_tools(read_model: BatchReadModel) -> list[Tool]:
    def get_batch_progress(args: BatchArgs) -> Any:
        return read_model.get_progress(args.batch_id)

    def list_failed_tasks(args: FailedArgs) -> Any:
        return read_model.list_failed_tasks(args.batch_id, args.limit)[: args.limit]

    def get_task_attempts(args: TaskArgs) -> Any:
        return read_model.get_task_attempts(args.task_id)

    return [
        Tool(
            "get_batch_progress", "Task counts by state for a batch.", BatchArgs, get_batch_progress
        ),
        Tool(
            "list_failed_tasks",
            "Failed tasks of a batch (limit <= 50).",
            FailedArgs,
            list_failed_tasks,
        ),
        Tool("get_task_attempts", "Attempt history of one task.", TaskArgs, get_task_attempts),
    ]


def build_explainer_registry(read_model: BatchReadModel) -> ToolRegistry:
    return ToolRegistry(explainer_tools(read_model), {"explainer": EXPLAINER_TOOLS})
