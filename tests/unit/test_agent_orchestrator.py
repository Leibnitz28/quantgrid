from __future__ import annotations

import logging
import time

import pytest

from quantgrid.agents.fake_llm import ScriptedLLM
from quantgrid.agents.llm import ProviderError, ProviderTurn, ToolCall
from quantgrid.agents.orchestrator import AgentRunner, preview, redact
from quantgrid.agents.tools import build_planner_registry
from tests.agent_helpers import call, final, make_policy, quota


def runner(llm, **kw):
    policy, cat = make_policy()
    reg = build_planner_registry(policy, cat, quota())
    kw.setdefault("timeout_s", 5.0)
    return AgentRunner(llm, reg, "planner", **kw)


def test_final_answer_and_trace():
    llm = ScriptedLLM([call("get_quota", {}), final("done")])
    run = runner(llm).run("sys", "hi")
    assert run.stop_reason == "final" and run.final_text == "done"
    assert run.steps == 2 and run.tool_calls == 1
    kinds = [e.kind for e in run.trace]
    assert kinds == ["llm_call", "tool_call", "tool_result", "llm_call", "final"]
    assert [e.seq for e in run.trace] == list(range(len(run.trace)))
    tc = run.trace[1]
    assert tc.tool == "get_quota" and len(tc.args_sha256 or "") == 64
    assert run.trace[0].ts.endswith("Z")
    assert llm.sent[0] == "hi" and llm.sent[1][0].ok  # model received the tool result
    assert llm.systems == ["sys"]


def test_same_args_same_hash():
    llm = ScriptedLLM([call("get_quota", {}), call("get_quota", {}, "c2"), final("x")])
    run = runner(llm).run("s", "u")
    hashes = [e.args_sha256 for e in run.trace if e.kind == "tool_call"]
    assert hashes[0] == hashes[1]


def test_max_steps():
    llm = ScriptedLLM([call("get_quota", {}, f"c{i}") for i in range(10)])
    run = runner(llm, max_steps=3).run("s", "u")
    assert run.stop_reason == "max_steps" and run.steps == 3


def test_tool_budget():
    llm = ScriptedLLM([call("get_quota", {}, f"c{i}") for i in range(10)])
    run = runner(llm, max_tool_calls=2, max_steps=10).run("s", "u")
    assert run.stop_reason == "tool_budget" and run.tool_calls == 2


def test_turn_with_too_many_calls_executes_none():
    turn = ProviderTurn(
        tool_calls=tuple(ToolCall(f"c{i}", "get_quota", {}) for i in range(5)),
        stop_reason="tool_use",
    )
    run = runner(ScriptedLLM([turn]), max_tool_calls=3).run("s", "u")
    assert run.stop_reason == "tool_budget" and run.tool_calls == 0


def test_repeated_violations_stop_run():
    llm = ScriptedLLM([call("shell", {"cmd": "id"}, f"c{i}") for i in range(10)])
    run = runner(llm, max_steps=10).run("s", "u")
    assert run.stop_reason == "tool_budget" and run.error == "repeated_tool_violations"
    assert run.tool_calls == 3
    assert [e.kind for e in run.trace].count("tool_rejected") == 3


def test_invalid_call_is_returned_to_model_as_error():
    llm = ScriptedLLM([call("get_dataset_info", {"name": 1}), final("ok")])
    run = runner(llm).run("s", "u")
    assert run.stop_reason == "final"
    assert llm.sent[1][0].ok is False


def test_provider_timeout_enforced():
    llm = ScriptedLLM([final("late")], delay_s=2.0)
    t0 = time.monotonic()
    run = runner(llm, timeout_s=0.3).run("s", "u")
    assert run.stop_reason == "timeout"
    assert time.monotonic() - t0 < 1.5


def test_time_budget_is_shared_across_steps():
    llm = ScriptedLLM([call("get_quota", {}, f"c{i}") for i in range(20)], delay_s=0.15)
    t0 = time.monotonic()
    run = runner(llm, timeout_s=0.5, max_steps=20, max_tool_calls=50).run("s", "u")
    assert run.stop_reason == "timeout" and time.monotonic() - t0 < 1.5


class RaisingLLM:
    name = "raising"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def open_session(self, system, tools):
        return self

    def send_user(self, text):
        raise self.exc


@pytest.mark.parametrize(
    ("kind", "stop"),
    [
        ("refused", "refused"),
        ("timeout", "timeout"),
        ("unavailable", "provider_error"),
        ("rate_limited", "provider_error"),
        ("bad_request", "provider_error"),
        ("internal", "provider_error"),
    ],
)
def test_provider_error_mapping(kind, stop):
    run = runner(RaisingLLM(ProviderError(kind, "boom"))).run("s", "u")
    assert run.stop_reason == stop and kind in (run.error or "")


def test_refusal_turn_and_error_turns():
    assert (
        runner(ScriptedLLM([ProviderTurn(stop_reason="refusal")])).run("s", "u").stop_reason
        == "refused"
    )
    for sr in ("error", "max_tokens"):
        assert (
            runner(ScriptedLLM([ProviderTurn(stop_reason=sr)])).run("s", "u").stop_reason
            == "provider_error"
        )
    assert (
        runner(ScriptedLLM([ProviderTurn(stop_reason="tool_use")])).run("s", "u").stop_reason
        == "provider_error"
    )


def test_redaction():
    data = {
        "api_key": "sk-ant-abcdefgh1234",
        "nested": [{"Authorization": "Bearer abcdef123456"}],
        "note": "token sk-abcdefgh123456 leaked",
        "n": 3,
    }
    out = redact(data)
    assert out["api_key"] == "[REDACTED]" and out["nested"][0]["Authorization"] == "[REDACTED]"
    assert "sk-abcdefgh123456" not in out["note"] and out["n"] == 3
    assert "sk-ant" not in preview(data)
    assert preview({"a": "x" * 1000}).endswith("...")


def test_secrets_not_in_trace():
    llm = ScriptedLLM([call("get_dataset_info", {"name": "sk-ant-secretsecret123"}), final("x")])
    run = runner(llm).run("s", "u")
    assert all("secretsecret" not in (e.args_preview or "") for e in run.trace)


def test_structured_logging_emitted(caplog):
    with caplog.at_level(logging.INFO, logger="quantgrid.agents"):
        runner(ScriptedLLM([final("x")])).run("s", "u")
    recs = [r for r in caplog.records if r.name == "quantgrid.agents"]
    assert recs and recs[0].agent_role == "planner" and hasattr(recs[0], "seq")  # type: ignore[attr-defined]


def test_otel_spans_created():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import quantgrid.agents.orchestrator as orch

    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    old = orch._tracer
    orch._tracer = tp.get_tracer("test")
    try:
        runner(ScriptedLLM([final("x")])).run("s", "u")
    finally:
        orch._tracer = old
    names = [s.name for s in exporter.get_finished_spans()]
    assert names == ["agent.step", "agent.run"]
