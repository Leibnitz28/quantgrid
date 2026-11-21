"""AI experiment planner: proposes a bounded ``ExperimentBatchSpec``; never submits it.

Pipeline (every stage is deterministic code; the model is one untrusted input):
goal length check -> agent loop with read-only tools -> strict single-JSON extraction ->
strict ``ExperimentBatchSpec`` validation (unknown fields rejected) -> ``PolicyService``
evaluation with the effective quota (``max_tasks`` can only *narrow* it). ``plan()``
persists and enqueues nothing; a human submits the returned spec via ``POST /v1/batches``,
which re-runs policy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from quantgrid.agents.fake_llm import FakeLLM
from quantgrid.agents.llm import LLMProvider
from quantgrid.agents.orchestrator import AgentRun, AgentRunner, TraceEvent
from quantgrid.agents.prompts import PLANNER_SYSTEM, planner_user_message
from quantgrid.agents.tools import build_planner_registry
from quantgrid.config import Settings
from quantgrid.domain.policy import DatasetCatalog, PolicyDecision, PolicyService, QuotaPolicy
from quantgrid.domain.specs import ExperimentBatchSpec

MAX_GOAL_CHARS = 4_000
MAX_FINAL_CHARS = 64_000
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_-]*)[ \t]*\n(.*?)```", re.DOTALL)

PlanStatus = Literal["ok", "rejected", "invalid_output", "error"]


@dataclass
class PlannerResult:
    status: PlanStatus
    spec: ExperimentBatchSpec | None
    decision: PolicyDecision | None
    errors: list[str]
    provider: str
    steps: int = 0
    tool_calls: int = 0
    trace: list[TraceEvent] = field(default_factory=list)


def provider_from_settings(settings: Settings) -> LLMProvider:
    """FakeLLM unless ``QG_LLM_PROVIDER=anthropic``."""
    if settings.llm_provider == "anthropic":
        from quantgrid.agents.anthropic_provider import AnthropicProvider

        return AnthropicProvider.from_settings(settings)
    return FakeLLM()


class OutputError(ValueError):
    pass


def extract_single_json(text: str) -> object:
    """The one JSON object in ``text``: the whole text, or exactly one fenced block.

    Raises :class:`OutputError` for no JSON, malformed JSON, or more than one JSON object.
    """
    if len(text) > MAX_FINAL_CHARS:
        raise OutputError("final answer too large")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass  # may be prose-with-braces; fall through to fence handling
    fences = _FENCE.findall(text)
    json_fences = [body for lang, body in fences if lang.lower() in ("json", "")]
    if len(json_fences) > 1:
        raise OutputError("multiple JSON blocks in final answer")
    outside = _FENCE.sub(" ", text)
    decoder = json.JSONDecoder()
    stray = 0
    i = outside.find("{")
    while i != -1:
        try:
            _, end = decoder.raw_decode(outside, i)
            stray += 1
            i = outside.find("{", end)
        except json.JSONDecodeError:
            i = outside.find("{", i + 1)
    if len(json_fences) == 1:
        if stray:
            raise OutputError("multiple JSON objects in final answer")
        try:
            return json.loads(json_fences[0])
        except json.JSONDecodeError as exc:
            raise OutputError(f"malformed JSON: {exc.msg}") from exc
    if stray > 1:
        raise OutputError("multiple JSON objects in final answer")
    if stray == 0:
        raise OutputError("no JSON object in final answer")
    return json.JSONDecoder().raw_decode(outside, outside.find("{"))[0]


class PlannerService:
    def __init__(
        self,
        provider: LLMProvider,
        policy: PolicyService,
        catalog: DatasetCatalog,
        settings: Settings | None = None,
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.catalog = catalog
        self.settings = settings or Settings()

    def plan(
        self,
        goal: str,
        *,
        quota: QuotaPolicy,
        max_tasks: int | None = None,
        dataset: str | None = None,
    ) -> PlannerResult:
        def result(
            status: PlanStatus,
            errors: list[str],
            *,
            spec: ExperimentBatchSpec | None = None,
            decision: PolicyDecision | None = None,
            run: AgentRun | None = None,
        ) -> PlannerResult:
            return PlannerResult(
                status=status,
                spec=spec,
                decision=decision,
                errors=errors,
                provider=self.provider.name,
                steps=run.steps if run else 0,
                tool_calls=run.tool_calls if run else 0,
                trace=run.trace if run else [],
            )

        goal = goal.strip()
        if not goal:
            return result("error", ["goal must not be empty"])
        if len(goal) > MAX_GOAL_CHARS:
            return result("error", [f"goal exceeds {MAX_GOAL_CHARS} characters"])
        if max_tasks is not None and max_tasks < 1:
            return result("error", ["max_tasks must be >= 1"])
        if dataset is not None and self.catalog.get(dataset) is None:
            return result("error", [f"unknown dataset {dataset!r}"])

        # max_tasks may only narrow the quota, never widen it.
        effective = quota
        if max_tasks is not None and max_tasks < quota.max_tasks_per_batch:
            effective = quota.model_copy(update={"max_tasks_per_batch": max_tasks})

        registry = build_planner_registry(self.policy, self.catalog, effective)
        runner = AgentRunner(
            self.provider,
            registry,
            "planner",
            max_steps=self.settings.agent_max_steps,
            max_tool_calls=self.settings.agent_max_tool_calls,
            timeout_s=self.settings.agent_timeout_s,
        )
        run = runner.run(
            PLANNER_SYSTEM,
            planner_user_message(goal, max_tasks=effective.max_tasks_per_batch, dataset=dataset),
        )
        if run.stop_reason != "final":
            return result("error", [f"{run.stop_reason}: {run.error or ''}".strip()], run=run)

        try:
            obj = extract_single_json(run.final_text)
            if not isinstance(obj, dict):
                raise OutputError("final JSON must be an object")
            spec = ExperimentBatchSpec.model_validate(obj)
        except OutputError as exc:
            return result("invalid_output", [str(exc)], run=run)
        except ValidationError as exc:
            errs = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:10]]
            return result("invalid_output", errs, run=run)

        decision = self.policy.evaluate(spec, effective)
        errors = [f"{v.code}: {v.message}" for v in decision.violations]
        if dataset is not None and spec.dataset.name != dataset:
            errors.append(
                f"DATASET_MISMATCH: request pinned {dataset!r}, spec uses {spec.dataset.name!r}"
            )
        if not decision.allowed or errors:
            return result("rejected", errors, spec=spec, decision=decision, run=run)
        return result("ok", [], spec=spec, decision=decision, run=run)
