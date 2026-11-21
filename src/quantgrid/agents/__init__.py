"""QuantGrid agents: a small, read-only, policy-bounded experiment-planning layer.

No tool in this package can submit, cancel, modify or delete anything; the planner only
returns a proposed spec that a human submits and the policy service re-checks.
"""

from quantgrid.agents.explainer import Explanation, explain_batch
from quantgrid.agents.fake_llm import FakeLLM, ScriptedLLM
from quantgrid.agents.llm import (
    LLMProvider,
    LLMSession,
    ProviderError,
    ProviderTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from quantgrid.agents.orchestrator import AgentRun, AgentRunner, TraceEvent
from quantgrid.agents.planner import PlannerResult, PlannerService, provider_from_settings
from quantgrid.agents.tools import BatchReadModel, ToolRegistry

__all__ = [
    "AgentRun",
    "AgentRunner",
    "BatchReadModel",
    "Explanation",
    "FakeLLM",
    "LLMProvider",
    "LLMSession",
    "PlannerResult",
    "PlannerService",
    "ProviderError",
    "ProviderTurn",
    "ScriptedLLM",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TraceEvent",
    "explain_batch",
    "provider_from_settings",
]
