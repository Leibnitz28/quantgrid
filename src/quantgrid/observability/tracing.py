"""OpenTelemetry tracing helpers with W3C trace-context propagation.

Without :func:`init_tracing` spans are no-ops and ``current_trace_id()`` is ``None``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Tracer

_INITIALISED = False
_provider: TracerProvider | None = None


def init_tracing(
    service_name: str,
    exporter: Literal["none", "console", "otlp"] = "none",
    endpoint: str | None = None,
) -> None:
    """Install the global TracerProvider once; ``"none"`` exports nothing."""
    global _INITIALISED
    if _INITIALISED or exporter == "none":
        return
    if exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as exc:
            raise RuntimeError(
                "OTLP tracing requires the optional extra: pip install 'quantgrid[otlp]' "
                "(opentelemetry-exporter-otlp-proto-http)"
            ) from exc
        span_exporter: Any = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    elif exporter == "console":
        span_exporter = ConsoleSpanExporter()
    else:
        raise ValueError(f"unknown tracing exporter {exporter!r}")
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    global _provider
    _provider = provider
    trace.set_tracer_provider(provider)
    _INITIALISED = True


def get_tracer(name: str = "quantgrid") -> Tracer:
    if _provider is not None:
        return _provider.get_tracer(name)
    return trace.get_tracer(name)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """Start a current span; ``None`` attributes are dropped, invalid types stringified."""
    attrs = {
        k: (v if isinstance(v, str | bool | int | float) else str(v))
        for k, v in attributes.items()
        if v is not None
    }
    with get_tracer().start_as_current_span(name, attributes=attrs) as sp:
        yield sp


def inject_context() -> dict[str, str]:
    """W3C trace-context headers of the current span (empty if none)."""
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_context(headers: Mapping[str, str] | None) -> otel_context.Context:
    return extract(dict(headers or {}))


@contextlib.contextmanager
def use_context(headers: Mapping[str, str] | None) -> Iterator[None]:
    """Make the remote parent described by ``headers`` current for the block."""
    token = otel_context.attach(extract_context(headers))
    try:
        yield
    finally:
        otel_context.detach(token)


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx.is_valid else None


def current_span_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.span_id:016x}" if ctx.is_valid else None
