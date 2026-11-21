"""Structured logging: JSON/console formatters, redaction and contextvars binding.

Invariants: every JSON record is a single line of valid JSON; values under keys that
look secret (key/token/secret/password/authorization) are never emitted; context is
held in :mod:`contextvars`, so it is isolated per thread and per asyncio task.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import traceback
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal

CONTEXT_KEYS = ("request_id", "batch_id", "task_id", "worker_id", "attempt")
REDACTED = "[REDACTED]"
_SECRET_MARKERS = ("key", "token", "secret", "password", "authorization")
_MAX_DEPTH = 8

_context: ContextVar[Mapping[str, Any] | None] = ContextVar("quantgrid_log_context", default=None)


def _current() -> Mapping[str, Any]:
    return _context.get() or {}


# Attributes present on every LogRecord; anything else came from ``extra=``.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def get_context() -> dict[str, Any]:
    """Return a copy of the currently bound context."""
    return dict(_current())


def clear_context() -> None:
    """Drop all bound context in the current execution context."""
    _context.set({})


@contextlib.contextmanager
def bind_context(**kv: Any) -> Iterator[None]:
    """Bind ``kv`` for the duration of the block; nests and restores on exit.

    ``None`` values are ignored so callers can pass optional ids unconditionally.
    """
    merged = {**_current(), **{k: v for k, v in kv.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def context_to_dict() -> dict[str, str]:
    """Serialise the bound context to a plain str->str dict (e.g. Celery headers)."""
    return {k: str(v) for k, v in _current().items()}


@contextlib.contextmanager
def context_from_dict(data: Mapping[str, Any] | None) -> Iterator[None]:
    """Bind context previously produced by :func:`context_to_dict`."""
    with bind_context(**dict(data or {})):
        yield


def _is_secret_key(key: object) -> bool:
    k = str(key).lower()
    return any(m in k for m in _SECRET_MARKERS)


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively redact values of secret-looking keys and make ``value`` JSON-safe."""
    if _depth > _MAX_DEPTH:
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): REDACTED if _is_secret_key(k) else redact(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [redact(v, _depth + 1) for v in value]
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return (
            value if value == value and value not in (float("inf"), float("-inf")) else str(value)
        )
    return str(value)


def _trace_fields() -> dict[str, str]:
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
    except Exception:  # pragma: no cover - defensive
        return {}
    if not ctx.is_valid:
        return {}
    return {"trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}"}


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
    }


def _iso_ts(created: float) -> str:
    dt = datetime.fromtimestamp(created, tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def __init__(self, service: str = "quantgrid") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": _iso_ts(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "service": self.service,
            "pid": os.getpid(),
        }
        doc.update(redact(dict(_current())))
        for k, v in _trace_fields().items():
            doc.setdefault(k, v)
        for k, v in redact(_extras(record)).items():
            doc.setdefault(k, v)
        if record.exc_info and record.exc_info[1] is not None:
            etype, exc, tb = record.exc_info
            doc["exc_type"] = etype.__name__ if etype else type(exc).__name__
            doc["exc_message"] = str(exc)
            doc["traceback"] = "".join(traceback.format_exception(etype, exc, tb))
        return json.dumps(doc, default=str, ensure_ascii=False, allow_nan=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single-line format with ``key=value`` context."""

    def __init__(self, service: str = "quantgrid") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        fields = {**redact(dict(_current())), **_trace_fields(), **redact(_extras(record))}
        tail = " ".join(f"{k}={v}" for k, v in fields.items())
        line = (
            f"{_iso_ts(record.created)} {record.levelname:<8} {record.name}: {record.getMessage()}"
        )
        if tail:
            line += " " + tail
        if record.exc_info and record.exc_info[1] is not None:
            line += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return line


_HANDLER_ATTR = "_quantgrid_handler"


def configure_logging(
    level: str = "INFO",
    fmt: Literal["json", "console"] = "json",
    service: str = "quantgrid",
) -> None:
    """Route the root logger to stderr with the chosen formatter. Idempotent: calling
    again replaces the handler we installed earlier rather than adding another."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _HANDLER_ATTR, False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_ATTR, True)
    handler.setFormatter(JsonFormatter(service) if fmt == "json" else ConsoleFormatter(service))
    root.addHandler(handler)
    root.setLevel(level.upper())
