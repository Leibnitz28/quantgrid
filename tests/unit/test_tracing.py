"""Tracing: propagation through inject/extract, ids, and init guards."""

from __future__ import annotations

import re

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from quantgrid.observability import tracing


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(tracing, "_provider", provider)
    return exp


def test_no_active_span_gives_none() -> None:
    assert tracing.current_trace_id() is None
    assert tracing.current_span_id() is None
    assert tracing.inject_context() == {}


def test_span_ids_format_and_attributes(exporter: InMemorySpanExporter) -> None:
    with tracing.span("op", task_id="t1", skip=None, obj=object().__class__):
        tid, sid = tracing.current_trace_id(), tracing.current_span_id()
    assert tid is not None and re.fullmatch(r"[0-9a-f]{32}", tid)
    assert sid is not None and re.fullmatch(r"[0-9a-f]{16}", sid)
    (s,) = exporter.get_finished_spans()
    assert s.attributes is not None
    assert s.attributes["task_id"] == "t1" and "skip" not in s.attributes


def test_inject_extract_round_trip_parent_child(exporter: InMemorySpanExporter) -> None:
    with tracing.span("producer"):
        headers = tracing.inject_context()
        producer_tid = tracing.current_trace_id()
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-0[0-9a-f]", headers["traceparent"])
    assert tracing.current_trace_id() is None
    with tracing.use_context(headers), tracing.span("consumer"):
        assert tracing.current_trace_id() == producer_tid
    assert tracing.current_trace_id() is None
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["consumer"].parent is not None
    assert spans["consumer"].parent.span_id == spans["producer"].context.span_id  # type: ignore[union-attr]
    assert spans["consumer"].context.trace_id == spans["producer"].context.trace_id  # type: ignore[union-attr]


def test_nested_spans_share_trace(exporter: InMemorySpanExporter) -> None:
    with tracing.span("a"), tracing.span("b"):
        pass
    a, b = sorted(exporter.get_finished_spans(), key=lambda s: s.name)
    assert b.parent is not None and b.parent.span_id == a.context.span_id  # type: ignore[union-attr]


def test_use_context_tolerates_empty_and_none() -> None:
    with tracing.use_context(None), tracing.use_context({}):
        assert tracing.current_trace_id() is None


def test_exception_is_recorded_and_reraised(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError), tracing.span("bad"):
        raise RuntimeError("x")
    (s,) = exporter.get_finished_spans()
    assert any(e.name == "exception" for e in s.events)


def test_init_none_is_noop_and_otlp_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_INITIALISED", False)
    monkeypatch.setattr(tracing, "_provider", None)
    tracing.init_tracing("svc", "none")
    assert tracing._provider is None
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="otlp"):
            tracing.init_tracing("svc", "otlp", "http://localhost:4318")
