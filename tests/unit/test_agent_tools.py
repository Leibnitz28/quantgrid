from __future__ import annotations

import json
from typing import Any

import pytest

from quantgrid.agents.tools import (
    EXPLAINER_TOOLS,
    MAX_ARGS_BYTES,
    MAX_RESULT_CHARS,
    PLANNER_TOOLS,
    NoArgs,
    Tool,
    ToolRegistry,
    build_explainer_registry,
    build_planner_registry,
)
from tests.agent_helpers import default_catalog, good_spec_dict, make_policy, quota


class ReadModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def get_progress(self, batch_id: str) -> dict[str, Any]:
        self.calls.append(("progress", batch_id))
        return {"state": "RUNNING", "total": 3, "counts": {"SUCCEEDED": 2, "FAILED": 1}}

    def list_failed_tasks(self, batch_id: str, limit: int) -> list[dict[str, Any]]:
        self.calls.append(("failed", limit))
        return [{"task_id": f"t{i}", "error_code": "TIMEOUT"} for i in range(100)]

    def get_task_attempts(self, task_id: str) -> list[dict[str, Any]]:
        return [{"attempt": 1, "status": "FAILED"}]


@pytest.fixture
def planner_reg():
    policy, cat = make_policy()
    return build_planner_registry(policy, cat, quota()), cat


def test_planner_allowlist_is_exact_and_read_only(planner_reg):
    reg, _ = planner_reg
    assert set(reg.tool_names("planner")) == PLANNER_TOOLS
    assert reg.tool_names("explainer") == []
    forbidden = ("submit", "cancel", "delete", "write", "update", "create", "shell", "exec", "run")
    for name in reg.tool_names("planner"):
        assert not any(f in name for f in forbidden)


def test_explainer_allowlist_is_exact():
    reg = build_explainer_registry(ReadModel())
    assert set(reg.tool_names("explainer")) == EXPLAINER_TOOLS
    assert reg.invoke("planner", "get_batch_progress", {"batch_id": "b1"}).ok is False


def test_registry_refuses_write_capable_tool():
    bad = Tool("submit_batch", "x", NoArgs, lambda a: None, read_only=False)
    with pytest.raises(ValueError, match="not read-only"):
        ToolRegistry([bad], {"planner": frozenset({"submit_batch"})})


def test_registry_rejects_duplicates_and_unknown_allowlist():
    t = Tool("a", "x", NoArgs, lambda a: 1)
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([t, t], {"planner": frozenset({"a"})})
    with pytest.raises(ValueError, match="unknown"):
        ToolRegistry([t], {"planner": frozenset({"b"})})


def test_specs_export_json_schema(planner_reg):
    reg, _ = planner_reg
    specs = {s.name: s for s in reg.specs("planner")}
    assert specs["get_dataset_info"].input_schema["required"] == ["name"]
    assert specs["get_dataset_info"].input_schema["additionalProperties"] is False


def test_unknown_tool_and_non_object_args(planner_reg):
    reg, _ = planner_reg
    for name in ("shell", "run_python", "submit_batch", "delete_results"):
        r = reg.invoke("planner", name, {}, call_id="x")
        assert not r.ok and r.call_id == "x" and "not permitted" in (r.error or "")
    assert not reg.invoke("planner", "get_quota", ["not", "a", "dict"]).ok  # type: ignore[arg-type]


def test_strict_argument_validation(planner_reg):
    reg, _ = planner_reg
    assert not reg.invoke("planner", "get_dataset_info", {"name": 5}).ok  # no coercion
    assert not reg.invoke("planner", "get_dataset_info", {}).ok
    r = reg.invoke("planner", "get_dataset_info", {"name": "x", "extra": 1})
    assert not r.ok and "extra" in (r.error or "")
    assert not reg.invoke("planner", "get_quota", {"unexpected": 1}).ok


def test_oversized_args_rejected(planner_reg):
    reg, _ = planner_reg
    r = reg.invoke("planner", "validate_spec", {"spec": {"name": "x" * (MAX_ARGS_BYTES + 1)}})
    assert not r.ok and "exceed" in (r.error or "")


def test_dataset_tools(planner_reg):
    reg, cat = planner_reg
    rows = reg.invoke("planner", "list_datasets", {}).content
    assert [r["name"] for r in rows] == sorted(r["name"] for r in rows)
    info = reg.invoke("planner", "get_dataset_info", {"name": "pairs-ou-small-v1"})
    assert info.ok and info.content["sha256"] == cat.get("pairs-ou-small-v1").sha256
    missing = reg.invoke("planner", "get_dataset_info", {"name": "nope"})
    assert not missing.ok and "unknown dataset" in (missing.error or "")


def test_quota_filters_datasets():
    policy, cat = make_policy()
    reg = build_planner_registry(policy, cat, quota(allowed_datasets=("pairs-ou-small-v1",)))
    assert [r["name"] for r in reg.invoke("planner", "list_datasets", {}).content] == [
        "pairs-ou-small-v1"
    ]
    assert not reg.invoke("planner", "get_dataset_info", {"name": "pairs-ou-medium-v1"}).ok


def test_strategy_tools(planner_reg):
    reg, _ = planner_reg
    assert {"name": "pairs_zscore", "version": "1"} in reg.invoke(
        "planner", "list_strategies", {}
    ).content
    schema = reg.invoke("planner", "get_strategy_schema", {"name": "pairs_zscore", "version": "1"})
    assert "lookback" in schema.content["properties"]
    assert not reg.invoke("planner", "get_strategy_schema", {"name": "x", "version": "1"}).ok


def test_validate_and_estimate(planner_reg):
    reg, cat = planner_reg
    spec = good_spec_dict(cat)
    v = reg.invoke("planner", "validate_spec", {"spec": spec}).content
    assert v["valid"] and v["allowed"] and v["task_count"] == 4
    e = reg.invoke("planner", "estimate_cost", {"spec": spec}).content
    assert e["valid"] and e["estimate"]["task_count"] == 4
    bad = reg.invoke("planner", "validate_spec", {"spec": {**spec, "bogus": 1}}).content
    assert bad["valid"] is False and bad["allowed"] is False and bad["errors"]


def test_validate_reports_policy_violations():
    policy, cat = make_policy()
    reg = build_planner_registry(policy, cat, quota(max_tasks_per_batch=2))
    v = reg.invoke("planner", "validate_spec", {"spec": good_spec_dict(cat)}).content
    assert not v["allowed"] and v["violations"][0]["code"] == "GRID_TOO_LARGE"


def test_handler_exception_becomes_error_result():
    def boom(_: Any) -> Any:
        raise RuntimeError("kaboom")

    reg = ToolRegistry([Tool("boom", "x", NoArgs, boom)], {"planner": frozenset({"boom"})})
    r = reg.invoke("planner", "boom", {})
    assert not r.ok and "RuntimeError" in (r.error or "")


def test_result_truncated_with_marker():
    reg = ToolRegistry(
        [Tool("big", "x", NoArgs, lambda a: ["y" * 100] * 1000)], {"planner": frozenset({"big"})}
    )
    r = reg.invoke("planner", "big", {})
    assert r.ok and r.content["truncated"] is True
    assert r.content["preview"].endswith("truncated by QuantGrid tool result cap]")
    assert len(json.dumps(r.content)) < MAX_RESULT_CHARS + 500


def test_explainer_tools_limit_and_validation():
    rm = ReadModel()
    reg = build_explainer_registry(rm)
    assert reg.invoke("explainer", "get_batch_progress", {"batch_id": "b-1"}).content["total"] == 3
    r = reg.invoke("explainer", "list_failed_tasks", {"batch_id": "b-1", "limit": 5})
    assert len(r.content) == 5
    assert not reg.invoke("explainer", "list_failed_tasks", {"batch_id": "b", "limit": 51}).ok
    assert not reg.invoke("explainer", "get_batch_progress", {"batch_id": "../../etc/passwd"}).ok
    assert not reg.invoke("explainer", "get_task_attempts", {"task_id": "t;DROP"}).ok
    assert reg.invoke("explainer", "get_task_attempts", {"task_id": "t1"}).ok
    assert default_catalog().calls == 0
