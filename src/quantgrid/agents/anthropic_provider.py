"""Optional real provider backed by the Anthropic Messages API (SDK 2.10, API 2026-10-21).

Configured only via environment: ``QG_LLM_MODEL`` (default ``claude-opus-5-5``),
``QG_LLM_EFFORT``, ``QG_LLM_TIMEOUT_S``; the SDK itself reads ``ANTHROPIC_API_KEY``.
The module imports without ``anthropic`` installed; the SDK is imported lazily.

Request shape (introspected from the installed SDK, ``client.beta.messages.create``):
one non-streaming call per step with ``max_tokens=16000``, ``system``, ``messages``,
``tools`` (name/description/input_schema), ``output_config={"effort": ...}``,
``betas=["server-side-fallback-2026-07-01"]`` and ``fallbacks="default"``.

Server-side refusal fallback: with the fallback beta enabled the API may transparently
retry a refused request on a fallback model and return a ``fallback`` content block
marking the switch. Echo rule when sending that assistant turn back: drop ``thinking`` /
``redacted_thinking`` / ``tool_use`` blocks that appear BEFORE the last ``fallback``
block (they belong to the abandoned model) and keep the ``fallback`` block in place.
Otherwise ``response.content`` is appended unchanged (thinking blocks included).

Deliberately NOT sent: ``temperature``/``top_p``/``top_k`` (rejected on this API
version), ``thinking`` (adaptive thinking is the default) and ``tool_choice`` (forced
choice returns 400 on claude-opus-5-5). The model's tool calls and text are untrusted
and are only ever returned to the orchestrator, never executed here.
"""

from __future__ import annotations

import json
from typing import Any

from quantgrid.agents.llm import (
    LLMSession,
    ProviderError,
    ProviderTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from quantgrid.config import Settings

BETAS = ["server-side-fallback-2026-07-01"]
MAX_TOKENS = 16_000
MAX_PAUSE_CONTINUATIONS = 5
_ECHO_DROP_BEFORE_FALLBACK = frozenset({"thinking", "redacted_thinking", "tool_use"})


def _btype(block: Any) -> str:
    t = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
    return str(t)


def _bget(block: Any, key: str, default: Any = None) -> Any:
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def echo_content(content: list[Any]) -> list[Any]:
    """Blocks to send back as the assistant turn (see module docstring for the rule)."""
    last_fb = -1
    for i, block in enumerate(content):
        if _btype(block) == "fallback":
            last_fb = i
    if last_fb < 0:
        return list(content)
    return [
        b
        for i, b in enumerate(content)
        if not (i < last_fb and _btype(b) in _ECHO_DROP_BEFORE_FALLBACK)
    ]


def map_sdk_error(exc: Exception, sdk: Any) -> ProviderError | None:
    """Most-specific-first mapping of SDK exceptions; ``None`` if not an SDK error."""
    if isinstance(exc, sdk.RateLimitError):
        return ProviderError("rate_limited", str(exc)[:200], retryable=True)
    if isinstance(exc, sdk.APITimeoutError):  # subclass of APIConnectionError: check first
        return ProviderError("timeout", str(exc)[:200], retryable=True)
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderError("unavailable", str(exc)[:200], retryable=True)
    if isinstance(exc, sdk.APIStatusError):
        status = int(getattr(exc, "status_code", 0) or 0)
        if status >= 500:
            return ProviderError("unavailable", f"HTTP {status}", retryable=True)
        return ProviderError("bad_request", f"HTTP {status}: {str(exc)[:200]}")
    return None


class _AnthropicSession:
    def __init__(self, provider: AnthropicProvider, system: str, tools: list[ToolSpec]) -> None:
        self._p = provider
        self._system = system
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        self._messages: list[dict[str, Any]] = []
        self._pending_ids: set[str] = set()

    def send_user(self, text: str) -> ProviderTurn:
        self._messages.append({"role": "user", "content": text})
        return self._step()

    def send_tool_results(self, results: list[ToolResult]) -> ProviderTurn:
        ids = {r.call_id for r in results}
        if ids != self._pending_ids or len(ids) != len(results):
            raise ProviderError("bad_request", "tool results do not match the pending tool calls")
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": json.dumps(r.content if r.ok else {"error": r.error}, default=str),
                "is_error": not r.ok,
            }
            for r in results
        ]
        self._pending_ids = set()
        self._messages.append({"role": "user", "content": blocks})
        return self._step()

    def _step(self) -> ProviderTurn:
        usage = {"input_tokens": 0, "output_tokens": 0}
        for _ in range(MAX_PAUSE_CONTINUATIONS + 1):
            resp = self._p.create(self._system, self._messages, self._tools)
            u = getattr(resp, "usage", None)
            usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
            usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
            stop = resp.stop_reason  # check BEFORE touching content
            if stop == "refusal":
                details = getattr(resp, "stop_details", None)
                category = getattr(details, "category", None)
                raise ProviderError(
                    "refused",
                    f"model refused (category={category})" if category else "model refused",
                )
            if stop == "max_tokens":
                raise ProviderError("internal", "response truncated at max_tokens")
            if stop not in ("tool_use", "end_turn", "pause_turn"):
                raise ProviderError("internal", f"unexpected stop_reason {stop!r}")
            content = echo_content(list(resp.content))
            self._messages.append({"role": "assistant", "content": content})
            if stop == "pause_turn":
                continue
            if stop == "tool_use":
                calls = tuple(
                    ToolCall(
                        id=str(_bget(b, "id")),
                        name=str(_bget(b, "name")),
                        arguments=_bget(b, "input") if isinstance(_bget(b, "input"), dict) else {},
                    )
                    for b in content
                    if _btype(b) == "tool_use"
                )
                self._pending_ids = {c.id for c in calls}
                return ProviderTurn(tool_calls=calls, stop_reason="tool_use", usage=usage)
            text = "".join(str(_bget(b, "text", "")) for b in content if _btype(b) == "text")
            return ProviderTurn(text=text, stop_reason="end_turn", usage=usage)
        raise ProviderError("internal", "too many pause_turn continuations")


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        timeout_s: float = 60.0,
        sdk: Any = None,
    ) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        if sdk is None:
            import anthropic as sdk  # lazy: optional dependency
        self._sdk = sdk

    @classmethod
    def from_settings(cls, settings: Settings) -> AnthropicProvider:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "unavailable", "the 'anthropic' package is not installed (optional extra)"
            ) from exc
        try:
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY itself
        except anthropic.AnthropicError as exc:
            raise ProviderError("unavailable", f"cannot create Anthropic client: {exc}") from exc
        return cls(
            client,
            model=settings.llm_model,
            effort=settings.llm_effort,
            timeout_s=settings.llm_timeout_s,
            sdk=anthropic,
        )

    def open_session(self, system: str, tools: list[ToolSpec]) -> LLMSession:
        return _AnthropicSession(self, system, tools)

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        try:
            client = self._client.with_options(timeout=self.timeout_s, max_retries=2)
            return client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=list(messages),
                tools=tools,
                output_config={"effort": self.effort},
                betas=list(BETAS),
                fallbacks="default",
            )
        except Exception as exc:
            mapped = map_sdk_error(exc, self._sdk)
            if mapped is not None:
                raise mapped from exc
            raise ProviderError("internal", f"{type(exc).__name__}: {str(exc)[:200]}") from exc
