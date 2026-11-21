"""Structured logging: JSON shape, redaction, context binding, idempotence."""

from __future__ import annotations

import io
import json
import logging
import threading

import pytest

from quantgrid.observability import logging as qlog


def _fmt(msg: str = "hi", **extra: object) -> dict[str, object]:
    rec = logging.LogRecord("t.logger", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    out = qlog.JsonFormatter("svc").format(rec)
    assert "\n" not in out
    return json.loads(out)  # type: ignore[no-any-return]


def test_base_fields() -> None:
    d = _fmt()
    assert d["level"] == "INFO" and d["logger"] == "t.logger" and d["msg"] == "hi"
    assert d["service"] == "svc" and isinstance(d["pid"], int)
    ts = str(d["ts"])
    assert ts.endswith("Z") and len(ts) == len("2024-01-01T00:00:00.000Z")


def test_context_binding_nests_and_restores() -> None:
    qlog.clear_context()
    with qlog.bind_context(request_id="r1", task_id="t1"):
        with qlog.bind_context(task_id="t2", attempt=3):
            d = _fmt()
            assert d["request_id"] == "r1" and d["task_id"] == "t2" and d["attempt"] == 3
        assert qlog.get_context() == {"request_id": "r1", "task_id": "t1"}
    assert qlog.get_context() == {}
    assert "request_id" not in _fmt()


def test_context_is_thread_isolated() -> None:
    seen: list[dict[str, object]] = []
    with qlog.bind_context(batch_id="main"):
        t = threading.Thread(target=lambda: seen.append(qlog.get_context()))
        t.start()
        t.join()
    assert seen == [{}]


def test_context_dict_roundtrip() -> None:
    with qlog.bind_context(task_id="x", attempt=2):
        data = qlog.context_to_dict()
    assert data == {"task_id": "x", "attempt": "2"}
    with qlog.context_from_dict(data):
        assert qlog.get_context()["task_id"] == "x"
    assert qlog.get_context() == {}


def test_extras_and_unserialisable() -> None:
    class Weird:
        def __str__(self) -> str:
            return "weird!"

    d = _fmt(n=1, obj=Weird(), nested={"a": {1, 2}})
    assert d["n"] == 1 and d["obj"] == "weird!" and d["nested"] == {"a": [1, 2]}


def test_redaction_recursive_case_insensitive() -> None:
    d = _fmt(
        API_Key="abc",
        headers={"Authorization": "Bearer z", "ok": "v", "deep": [{"password": "p"}]},
        plain="fine",
    )
    assert d["API_Key"] == qlog.REDACTED
    assert d["headers"] == {
        "Authorization": qlog.REDACTED,
        "ok": "v",
        "deep": [{"password": qlog.REDACTED}],
    }
    assert "abc" not in json.dumps(d) and d["plain"] == "fine"


def test_exception_fields() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = logging.LogRecord("l", logging.ERROR, __file__, 1, "x", None, sys.exc_info())
    d = json.loads(qlog.JsonFormatter().format(rec))
    assert d["exc_type"] == "ValueError" and d["exc_message"] == "boom"
    assert "Traceback" in d["traceback"]


def test_trace_ids_included_inside_span() -> None:
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("s"):
        d = _fmt()
    assert len(str(d["trace_id"])) == 32 and len(str(d["span_id"])) == 16


def test_configure_is_idempotent_and_writes_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = logging.getLogger()
    saved_level, saved = root.level, list(root.handlers)
    try:
        qlog.configure_logging("DEBUG", "json", service="s1")
        qlog.configure_logging("DEBUG", "json", service="s2")
        ours = [h for h in root.handlers if getattr(h, "_quantgrid_handler", False)]
        assert len(ours) == 1
        logging.getLogger("x").info("hello")
        err = capsys.readouterr().err.strip().splitlines()[-1]
        assert json.loads(err)["service"] == "s2"
        qlog.configure_logging("INFO", "console")
        buf = io.StringIO()
        ours = [h for h in root.handlers if getattr(h, "_quantgrid_handler", False)]
        assert isinstance(ours[0].formatter, qlog.ConsoleFormatter)
        del buf
    finally:
        root.handlers[:] = saved
        root.setLevel(saved_level)
