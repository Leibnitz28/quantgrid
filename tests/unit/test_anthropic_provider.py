"""AnthropicProvider against an injected duck-typed client (no network, no API key)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest

from quantgrid.agents.anthropic_provider import (
    BETAS,
    MAX_TOKENS,
    AnthropicProvider,
    echo_content,
)
from quantgrid.agents.llm import ProviderError, ToolResult, ToolSpec
from quantgrid.config import Settings

anthropic = pytest.importorskip("anthropic")

TOOLS = [ToolSpec("get_quota", "quota", {"type": "object", "properties": {}})]


def text(t: str) -> NS:
    return NS(type="text", text=t)


def tool_use(i: str, name: str, inp: Any) -> NS:
    return NS(type="tool_use", id=i, name=name, input=inp)


def resp(stop: str, *content: Any, **kw: Any) -> NS:
    return NS(
        stop_reason=stop,
        content=list(content),
        usage=NS(input_tokens=10, output_tokens=5),
        stop_details=kw.get("stop_details"),
    )


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.options: list[dict[str, Any]] = []
        self.beta = NS(messages=NS(create=self._create))

    def with_options(self, **kw: Any) -> FakeClient:
        self.options.append(kw)
        return self

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def provider(client: FakeClient, **kw: Any) -> AnthropicProvider:
    return AnthropicProvider(client, model="claude-opus-5-5", effort="high", timeout_s=7.5, **kw)


def test_request_shape_and_forbidden_params():
    c = FakeClient([resp("end_turn", text("hello"))])
    turn = provider(c).open_session("SYS", TOOLS).send_user("hi")
    assert (
        turn.stop_reason == "end_turn" and turn.text == "hello" and turn.usage["output_tokens"] == 5
    )
    (kw,) = c.calls
    assert kw["model"] == "claude-opus-5-5" and kw["max_tokens"] == MAX_TOKENS == 16000
    assert kw["system"] == "SYS" and kw["messages"] == [{"role": "user", "content": "hi"}]
    assert kw["tools"] == [
        {
            "name": "get_quota",
            "description": "quota",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    assert kw["output_config"] == {"effort": "high"}
    assert kw["betas"] == BETAS == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert forbidden not in kw
    assert c.options == [{"timeout": 7.5, "max_retries": 2}]


def test_real_sdk_signature_accepts_our_kwargs():
    import inspect

    from anthropic.resources.beta.messages import Messages

    params = set(inspect.signature(Messages.create).parameters)
    assert {"betas", "fallbacks", "output_config", "tools", "system", "messages"} <= params


def test_tool_use_roundtrip_single_user_message_with_errors():
    tu = [tool_use("t1", "get_quota", {}), tool_use("t2", "get_quota", {"bad": 1})]
    c = FakeClient([resp("tool_use", text("thinking"), *tu), resp("end_turn", text("done"))])
    s = provider(c).open_session("S", TOOLS)
    turn = s.send_user("go")
    assert [t.id for t in turn.tool_calls] == ["t1", "t2"] and turn.stop_reason == "tool_use"
    turn2 = s.send_tool_results(
        [ToolResult("t1", True, {"a": 1}), ToolResult("t2", False, error="nope")]
    )
    assert turn2.text == "done"
    msgs = c.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == [text("thinking"), *tu]  # appended unchanged
    blocks = msgs[2]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["t1", "t2"]
    assert [b["is_error"] for b in blocks] == [False, True]
    assert json.loads(blocks[0]["content"]) == {"a": 1}
    assert json.loads(blocks[1]["content"]) == {"error": "nope"}


def test_thinking_blocks_passed_back_unchanged():
    think = NS(type="thinking", thinking="hmm", signature="sig")
    c = FakeClient(
        [resp("tool_use", think, tool_use("t1", "get_quota", {})), resp("end_turn", text("x"))]
    )
    s = provider(c).open_session("S", TOOLS)
    s.send_user("go")
    s.send_tool_results([ToolResult("t1", True, 1)])
    assert c.calls[1]["messages"][1]["content"][0] is think


def test_tool_results_must_match_pending():
    c = FakeClient([resp("tool_use", tool_use("t1", "get_quota", {}))])
    s = provider(c).open_session("S", TOOLS)
    s.send_user("go")
    with pytest.raises(ProviderError) as ei:
        s.send_tool_results([ToolResult("other", True, 1)])
    assert ei.value.kind == "bad_request"


def test_non_dict_tool_input_becomes_empty_args():
    c = FakeClient([resp("tool_use", tool_use("t1", "get_quota", "oops"))])
    turn = provider(c).open_session("S", TOOLS).send_user("go")
    assert turn.tool_calls[0].arguments == {}


def test_pause_turn_continues_with_assistant_content():
    c = FakeClient([resp("pause_turn", text("part1")), resp("end_turn", text("part2"))])
    turn = provider(c).open_session("S", TOOLS).send_user("go")
    assert turn.text == "part2" and turn.usage["input_tokens"] == 20
    msgs = c.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_endless_pause_turn_bounded():
    c = FakeClient([resp("pause_turn", text("p"))] * 20)
    with pytest.raises(ProviderError) as ei:
        provider(c).open_session("S", TOOLS).send_user("go")
    assert ei.value.kind == "internal" and len(c.calls) <= 7


def test_refusal_checked_before_content_and_includes_category():
    class Explosive:
        stop_reason = "refusal"
        stop_details = NS(category="cyber")
        usage = None

        @property
        def content(self) -> Any:
            raise AssertionError("content must not be read on refusal")

    with pytest.raises(ProviderError) as ei:
        provider(FakeClient([Explosive()])).open_session("S", TOOLS).send_user("go")
    assert ei.value.kind == "refused" and "cyber" in str(ei.value) and not ei.value.retryable


def test_max_tokens_and_unexpected_stop():
    for stop in ("max_tokens", "model_context_window_exceeded"):
        with pytest.raises(ProviderError) as ei:
            provider(FakeClient([resp(stop, text("x"))])).open_session("S", TOOLS).send_user("go")
        assert ei.value.kind == "internal"


def test_fallback_echo_rule():
    think = NS(type="thinking", thinking="a", signature="s")
    red = NS(type="redacted_thinking", data="d")
    old_tool = tool_use("old", "get_quota", {})
    fb = NS(type="fallback", **{"from_": "m1", "to": "m2", "trigger": "refusal"})
    new_think = NS(type="thinking", thinking="b", signature="s2")
    new_tool = tool_use("new", "get_quota", {})
    t = text("keep")
    out = echo_content([think, red, old_tool, t, fb, new_think, new_tool])
    assert out == [t, fb, new_think, new_tool]
    assert echo_content([think, t]) == [think, t]  # no fallback: unchanged
    c = FakeClient([resp("tool_use", think, old_tool, fb, new_think, new_tool)])
    s = provider(c).open_session("S", TOOLS)
    turn = s.send_user("go")
    assert [t.id for t in turn.tool_calls] == ["new"]  # abandoned model's tool_use ignored
    c.script.append(resp("end_turn", text("x")))
    s.send_tool_results([ToolResult("new", True, 1)])
    assert c.calls[1]["messages"][1]["content"] == [fb, new_think, new_tool]


def _status(cls: type, code: int) -> Exception:
    import httpx2

    req = httpx2.Request("POST", "https://x.invalid/v1/messages")
    return cls("err", response=httpx2.Response(code, request=req), body=None)


@pytest.mark.parametrize(
    ("make", "kind", "retryable"),
    [
        (lambda: _status(anthropic.RateLimitError, 429), "rate_limited", True),
        (lambda: anthropic.APITimeoutError(request=_req()), "timeout", True),
        (lambda: anthropic.APIConnectionError(request=_req()), "unavailable", True),
        (lambda: _status(anthropic.InternalServerError, 500), "unavailable", True),
        (lambda: _status(anthropic.APIStatusError, 503), "unavailable", True),
        (lambda: _status(anthropic.BadRequestError, 400), "bad_request", False),
        (lambda: _status(anthropic.AuthenticationError, 401), "bad_request", False),
        (lambda: RuntimeError("weird"), "internal", False),
    ],
)
def test_error_mapping(make, kind, retryable):
    with pytest.raises(ProviderError) as ei:
        provider(FakeClient([make()])).open_session("S", TOOLS).send_user("go")
    assert ei.value.kind == kind and ei.value.retryable is retryable


def _req() -> Any:
    import httpx2

    return httpx2.Request("POST", "https://x.invalid/v1/messages")


def test_timeout_not_misclassified_as_connection_error():
    assert issubclass(anthropic.APITimeoutError, anthropic.APIConnectionError)  # why order matters


def test_from_settings_builds_client_without_network(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    p = AnthropicProvider.from_settings(Settings(llm_model="m", llm_effort="low", llm_timeout_s=3))
    assert (p.model, p.effort, p.timeout_s) == ("m", "low", 3)


def test_module_imports_without_sdk(monkeypatch):
    import importlib

    import quantgrid.agents.anthropic_provider as mod

    monkeypatch.setitem(sys.modules, "anthropic", None)
    importlib.reload(mod)  # must not import the SDK at module level
    with pytest.raises(ProviderError) as ei:
        mod.AnthropicProvider.from_settings(Settings())
    assert ei.value.kind == "unavailable"
    monkeypatch.undo()
    importlib.reload(mod)
