"""Read-only batch explainer: summarises progress and failures using explainer tools only."""

from __future__ import annotations

from dataclasses import dataclass, field

from quantgrid.agents.llm import LLMProvider
from quantgrid.agents.orchestrator import AgentRunner, TraceEvent
from quantgrid.agents.prompts import EXPLAINER_SYSTEM, explainer_user_message
from quantgrid.agents.tools import BatchReadModel, build_explainer_registry
from quantgrid.config import Settings

MAX_SUMMARY_CHARS = 500
MAX_FINDINGS = 30
MAX_FINDING_CHARS = 300


@dataclass
class Explanation:
    summary: str
    findings: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


def explain_batch(
    batch_id: str,
    read_model: BatchReadModel,
    provider: LLMProvider,
    *,
    settings: Settings | None = None,
) -> Explanation:
    """Explain ``batch_id``. Never raises for model/provider failures; ``ok`` reports them."""
    s = settings or Settings()
    runner = AgentRunner(
        provider,
        build_explainer_registry(read_model),
        "explainer",
        max_steps=s.agent_max_steps,
        max_tool_calls=s.agent_max_tool_calls,
        timeout_s=s.agent_timeout_s,
    )
    run = runner.run(EXPLAINER_SYSTEM, explainer_user_message(batch_id))
    if run.stop_reason != "final":
        return Explanation(
            summary=f"Explanation unavailable ({run.stop_reason}).",
            findings=[],
            trace=run.trace,
            ok=False,
            error=run.error,
        )
    lines = [ln.strip() for ln in run.final_text.splitlines() if ln.strip()]
    summary = (lines[0] if lines else "No explanation produced.")[:MAX_SUMMARY_CHARS]
    findings = [
        ln[2:].strip()[:MAX_FINDING_CHARS] for ln in lines[1:] if ln.startswith(("- ", "* "))
    ][:MAX_FINDINGS]
    return Explanation(summary=summary, findings=findings, trace=run.trace)
