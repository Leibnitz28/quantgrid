"""Bounded agent loop: provider turn -> validated read-only tool calls -> repeat.

Invariants enforced here, independent of what the model says:
* at most ``max_steps`` provider calls and ``max_tool_calls`` tool calls;
* every provider call gets only the remaining wall-clock budget; a stuck provider is
  abandoned (daemon thread) and the run ends with ``timeout``;
* every tool call goes through :class:`ToolRegistry`; rejected calls are returned to the
  model as errors and repeated violations stop the run;
* the model's text is returned as ``final_text`` and never executed.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

from opentelemetry import trace

from quantgrid.agents.llm import (
    LLMProvider,
    LLMSession,
    ProviderError,
    ProviderTurn,
    ToolResult,
)
from quantgrid.agents.tools import Role, ToolRegistry

log = logging.getLogger("quantgrid.agents")
_tracer = trace.get_tracer(__name__)

RunStop = Literal["final", "max_steps", "tool_budget", "timeout", "provider_error", "refused"]
EventKind = Literal["llm_call", "tool_call", "tool_result", "tool_rejected", "final", "error"]

MAX_VIOLATIONS = 3
MAX_TRACE_EVENTS = 500
PREVIEW_CHARS = 240
_SECRET_KEY = re.compile(r"key|token|secret|password|authorization", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]{8,}")

T = TypeVar("T")


def redact(value: Any) -> Any:
    """Mask values under secret-looking keys and token-looking strings."""
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if _SECRET_KEY.search(str(k)) else redact(v) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


def preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    try:
        text = json.dumps(redact(value), default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = _SECRET_VALUE.sub("[REDACTED]", str(value))
    return text if len(text) <= limit else text[:limit] + "..."


def _sha(value: Any) -> str:
    try:
        text = json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    ts: str
    kind: EventKind
    step: int
    tool: str | None = None
    args_sha256: str | None = None
    args_preview: str | None = None
    result_preview: str | None = None
    latency_ms: float | None = None


@dataclass
class AgentRun:
    final_text: str
    stop_reason: RunStop
    steps: int
    tool_calls: int
    trace: list[TraceEvent] = field(default_factory=list)
    error: str | None = None


class _Timeout(Exception):
    pass


def _call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """Run ``fn`` in a daemon thread; raise :class:`_Timeout` on overrun."""
    box: dict[str, Any] = {}
    done = threading.Event()
    ctx = contextvars.copy_context()

    def target() -> None:
        try:
            box["value"] = ctx.run(fn)
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True, name="agent-provider-call").start()
    if not done.wait(max(timeout_s, 0.0)):
        raise _Timeout
    if "error" in box:
        raise box["error"]
    return box["value"]  # type: ignore[no-any-return]


class AgentRunner:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        role: Role,
        *,
        max_steps: int = 8,
        max_tool_calls: int = 16,
        timeout_s: float = 120.0,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.role: Role = role
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.timeout_s = timeout_s

    def run(self, system: str, user: str) -> AgentRun:
        with _tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("agent.role", self.role)
            span.set_attribute("agent.provider", self.provider.name)
            result = self._run(system, user)
            span.set_attribute("agent.stop_reason", result.stop_reason)
            span.set_attribute("agent.steps", result.steps)
            span.set_attribute("agent.tool_calls", result.tool_calls)
            return result

    def _run(self, system: str, user: str) -> AgentRun:
        t_start = time.monotonic()
        deadline = t_start + self.timeout_s
        events: list[TraceEvent] = []
        state = {"steps": 0, "tools": 0, "violations": 0}

        def emit(kind: EventKind, **kw: Any) -> None:
            if len(events) >= MAX_TRACE_EVENTS:
                return
            ev = TraceEvent(
                seq=len(events),
                ts=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                kind=kind,
                step=state["steps"],
                **kw,
            )
            events.append(ev)
            log.info(
                "agent_event",
                extra={
                    "agent_role": self.role,
                    **kw,
                    "kind": kind,
                    "seq": ev.seq,
                    "step": ev.step,
                },
            )

        def finish(stop: RunStop, text: str = "", error: str | None = None) -> AgentRun:
            emit("final" if stop == "final" else "error", result_preview=preview(error or text))
            return AgentRun(
                final_text=text,
                stop_reason=stop,
                steps=state["steps"],
                tool_calls=state["tools"],
                trace=events,
                error=error,
            )

        def timed(fn: Callable[[], T]) -> T:
            return _call_with_timeout(fn, deadline - time.monotonic())

        session: LLMSession
        try:
            session = timed(
                lambda: self.provider.open_session(system, self.registry.specs(self.role))
            )
        except _Timeout:
            return finish("timeout", error="provider timed out opening session")
        except ProviderError as exc:
            return finish(_stop_for(exc), error=str(exc))

        pending: Callable[[], ProviderTurn] = lambda: session.send_user(user)  # noqa: E731
        while True:
            if state["steps"] >= self.max_steps:
                return finish("max_steps", error=f"exceeded max_steps={self.max_steps}")
            if deadline - time.monotonic() <= 0:
                return finish("timeout", error="time budget exhausted")
            state["steps"] += 1
            t0 = time.monotonic()
            try:
                with _tracer.start_as_current_span("agent.step") as sp:
                    sp.set_attribute("agent.step", state["steps"])
                    turn = timed(pending)
            except _Timeout:
                emit(
                    "llm_call", latency_ms=(time.monotonic() - t0) * 1000, result_preview="timeout"
                )
                return finish("timeout", error=f"provider exceeded the {self.timeout_s}s budget")
            except ProviderError as exc:
                emit("llm_call", latency_ms=(time.monotonic() - t0) * 1000, result_preview=str(exc))
                return finish(_stop_for(exc), error=str(exc))
            emit(
                "llm_call",
                latency_ms=round((time.monotonic() - t0) * 1000, 3),
                result_preview=f"stop={turn.stop_reason} tools={len(turn.tool_calls)}",
            )
            if turn.stop_reason == "refusal":
                return finish("refused", error="model refused the request")
            if turn.stop_reason in ("error", "max_tokens"):
                return finish("provider_error", error=f"provider stop_reason={turn.stop_reason}")
            if not turn.tool_calls:
                if turn.stop_reason == "tool_use":
                    return finish("provider_error", error="tool_use without tool calls")
                return finish("final", text=turn.text)
            if state["tools"] + len(turn.tool_calls) > self.max_tool_calls:
                return finish("tool_budget", error=f"exceeded max_tool_calls={self.max_tool_calls}")
            results: list[ToolResult] = []
            for call in turn.tool_calls:
                state["tools"] += 1
                emit(
                    "tool_call",
                    tool=call.name[:64],
                    args_sha256=_sha(call.arguments),
                    args_preview=preview(call.arguments),
                )
                t1 = time.monotonic()
                res = self.registry.invoke(self.role, call.name, call.arguments, call_id=call.id)
                lat = round((time.monotonic() - t1) * 1000, 3)
                if res.ok:
                    emit(
                        "tool_result",
                        tool=call.name[:64],
                        result_preview=preview(res.content),
                        latency_ms=lat,
                    )
                else:
                    state["violations"] += 1
                    emit(
                        "tool_rejected",
                        tool=call.name[:64],
                        result_preview=preview(res.error),
                        latency_ms=lat,
                    )
                results.append(res)
            if state["violations"] >= MAX_VIOLATIONS:
                return finish("tool_budget", error="repeated_tool_violations")
            pending = lambda r=results: session.send_tool_results(r)  # type: ignore[misc]  # noqa: E731


def _stop_for(exc: ProviderError) -> RunStop:
    if exc.kind == "refused":
        return "refused"
    if exc.kind == "timeout":
        return "timeout"
    return "provider_error"
