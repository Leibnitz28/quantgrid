"""Provider-neutral LLM types. Everything a model returns is untrusted data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

StopReason = Literal["tool_use", "end_turn", "max_tokens", "refusal", "error"]
ProviderErrorKind = Literal[
    "timeout", "rate_limited", "refused", "unavailable", "bad_request", "internal"
]


@dataclass(frozen=True)
class ToolSpec:
    """A tool advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model; ``arguments`` is untrusted."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a tool call; ``error`` is set iff not ok."""

    call_id: str
    ok: bool
    content: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ProviderTurn:
    """One model response: tool calls or final text."""

    tool_calls: tuple[ToolCall, ...] = ()
    text: str = ""
    stop_reason: StopReason = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)


class ProviderError(Exception):
    """A typed provider failure; ``retryable`` is advisory."""

    def __init__(self, kind: ProviderErrorKind, message: str = "", *, retryable: bool = False):
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind: ProviderErrorKind = kind
        self.retryable = retryable
        self.message = message


class LLMSession(Protocol):
    """A conversation. Methods may block; the orchestrator enforces the time budget."""

    def send_user(self, text: str) -> ProviderTurn: ...

    def send_tool_results(self, results: list[ToolResult]) -> ProviderTurn: ...


class LLMProvider(Protocol):
    name: str

    def open_session(self, system: str, tools: list[ToolSpec]) -> LLMSession: ...
