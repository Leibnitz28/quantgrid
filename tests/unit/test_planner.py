from __future__ import annotations

import json

import pytest

from quantgrid.agents.fake_llm import FakeLLM, ScriptedLLM, parse_goal
from quantgrid.agents.llm import ProviderError
from quantgrid.agents.planner import (
    MAX_GOAL_CHARS,
    OutputError,
    PlannerService,
    extract_single_json,
    provider_from_settings,
)
from tests.agent_helpers import (
    call,
    fast_settings,
    final,
    good_spec_dict,
    make_policy,
    quota,
)


@pytest.fixture
def svc():
    policy, cat = make_policy()
    return PlannerService(FakeLLM(), policy, cat, fast_settings()), cat


def test_ok_plan_follows_fixed_tool_order(svc):
    s, _ = svc
    r = s.plan("quick sweep of lookback 20-60 and entry z 1.5 to 2.5", quota=quota())
    assert r.status == "ok" and r.errors == [] and r.provider == "fake"
    tools = [e.tool for e in r.trace if e.kind == "tool_call"]
    assert tools == [
        "list_datasets",
        "get_dataset_info",
        "get_strategy_schema",
        "get_quota",
        "validate_spec",
    ]
    assert r.spec is not None and r.decision is not None and r.decision.allowed
    assert r.spec.grid == {"lookback": [20, 60], "entry_z": [1.5, 2.5]}


def test_goal_keywords(svc):
    s, _ = svc
    r = s.plan(
        "medium dataset, 3 seeds, bootstrap 500 samples, exit 0.3, lookback 30", quota=quota()
    )
    assert r.spec is not None
    assert r.spec.dataset.name == "pairs-ou-medium-v1"
    assert r.spec.seeds == [0, 1, 2]
    assert r.spec.workload.bootstrap_samples == 500
    assert r.spec.fixed_params["exit_z"] == 0.3
    assert r.spec.grid["lookback"] == [30]


def test_gappy_and_default_dataset(svc):
    s, _ = svc
    assert s.plan("try the gappy data", quota=quota()).spec.dataset.name == "pairs-ou-gappy-v1"  # type: ignore[union-attr]
    assert s.plan("something", quota=quota()).spec.dataset.name == "pairs-ou-small-v1"  # type: ignore[union-attr]


def test_thorough_vs_quick_grid_sizes():
    q, t = parse_goal("quick"), parse_goal("thorough")
    assert len(q.lookbacks) < len(t.lookbacks) and len(q.entries) < len(t.entries)


def test_at_most_n_tasks_in_goal(svc):
    s, _ = svc
    r = s.plan("thorough sweep, 4 seeds, at most 20 tasks", quota=quota())
    assert r.status == "ok" and r.spec is not None and r.spec.task_count() <= 20


def test_max_tasks_narrows_quota_never_widens(svc):
    s, _ = svc
    r = s.plan("thorough sweep", quota=quota(), max_tasks=6)
    assert r.status == "ok" and r.spec is not None and r.spec.task_count() <= 6
    # asking for more than the quota allows must not widen it
    r = s.plan("thorough sweep with 8 seeds", quota=quota(max_tasks_per_batch=5), max_tasks=10_000)
    assert r.spec is not None and r.spec.task_count() <= 5 and r.status == "ok"


def test_adjustment_after_violation_is_deterministic(svc):
    s, _ = svc
    q = quota(max_bootstrap_samples=100, max_runtime_s_per_task=60)
    r = s.plan("bootstrap 5000", quota=q)
    assert r.status == "ok" and r.spec is not None
    assert r.spec.workload.bootstrap_samples == 100 and r.spec.budget.max_runtime_s == 60
    tools = [e.tool for e in r.trace if e.kind == "tool_call"]
    assert tools.count("validate_spec") == 2


def test_identical_requests_identical_outputs(svc):
    s, _ = svc
    a = s.plan("thorough lookback 10-90, 2 seeds", quota=quota())
    b = s.plan("thorough lookback 10-90, 2 seeds", quota=quota())
    assert a.spec == b.spec and a.status == b.status
    assert [(e.kind, e.tool, e.args_sha256) for e in a.trace] == [
        (e.kind, e.tool, e.args_sha256) for e in b.trace
    ]


def test_pinned_dataset(svc):
    s, _ = svc
    r = s.plan("small please", quota=quota(), dataset="pairs-ou-medium-v1")
    assert r.status == "ok" and r.spec is not None and r.spec.dataset.name == "pairs-ou-medium-v1"
    r = s.plan("x", quota=quota(), dataset="nope")
    assert r.status == "error" and r.steps == 0


def test_pinned_dataset_mismatch_rejected():
    policy, cat = make_policy()
    spec = good_spec_dict(cat, "pairs-ou-small-v1")
    s = PlannerService(ScriptedLLM([final(json.dumps(spec))]), policy, cat, fast_settings())
    r = s.plan("x", quota=quota(), dataset="pairs-ou-medium-v1")
    assert r.status == "rejected" and any("DATASET_MISMATCH" in e for e in r.errors)


def test_goal_validation_before_model_call():
    llm = ScriptedLLM([final("{}")])
    policy, cat = make_policy()
    s = PlannerService(llm, policy, cat, fast_settings())
    for goal in ("", "   ", "x" * (MAX_GOAL_CHARS + 1)):
        assert s.plan(goal, quota=quota()).status == "error"
    assert s.plan("ok", quota=quota(), max_tasks=0).status == "error"
    assert llm.sent == [] and llm.systems == []
    assert s.plan("x" * MAX_GOAL_CHARS, quota=quota()).status == "invalid_output"  # reaches model


def test_policy_runs_even_if_model_says_ok():
    policy, cat = make_policy()
    spec = good_spec_dict(cat, grid={"lookback": list(range(10, 60)), "entry_z": [1.5, 2.0]})
    s = PlannerService(ScriptedLLM([final(json.dumps(spec))]), policy, cat, fast_settings())
    r = s.plan("x", quota=quota(max_tasks_per_batch=10))
    assert r.status == "rejected" and r.decision is not None and not r.decision.allowed
    assert any(e.startswith("GRID_TOO_LARGE") for e in r.errors)


def test_fenced_json_with_prose_ok():
    policy, cat = make_policy()
    spec = good_spec_dict(cat)
    text = "Here is my plan:\n```json\n" + json.dumps(spec) + "\n```\nHope it helps {not json}."
    s = PlannerService(ScriptedLLM([final(text)]), policy, cat, fast_settings())
    assert s.plan("x", quota=quota()).status == "ok"


def test_provider_error_is_error_status():
    class Boom:
        name = "boom"

        def open_session(self, system, tools):
            raise ProviderError("unavailable", "down")

    policy, cat = make_policy()
    r = PlannerService(Boom(), policy, cat, fast_settings()).plan("x", quota=quota())
    assert r.status == "error" and r.spec is None and "provider_error" in r.errors[0]


def test_tool_error_leads_to_error_status_not_crash():
    policy, cat = make_policy()
    llm = ScriptedLLM([call("validate_spec", {"spec": 5})])
    r = PlannerService(llm, policy, cat, fast_settings(agent_max_steps=1)).plan("x", quota=quota())
    assert r.status == "error"


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ('{"a": 1}', True),
        ('  {"a": 1}\n', True),
        ('```json\n{"a": 1}\n```', True),
        ('```\n{"a": 1}\n```', True),
        ('text {"a": 1} text', True),
        ("no json at all", False),
        ('{"a": 1} {"b": 2}', False),
        ('```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```', False),
        ('```json\n{"a": 1}\n```\nand also {"b": 2}', False),
        ('```json\n{"a": \n```', False),
        ("x" * 70_000, False),
    ],
)
def test_extract_single_json(text, ok):
    if ok:
        assert isinstance(extract_single_json(text), dict)
    else:
        with pytest.raises(OutputError):
            extract_single_json(text)


def test_provider_from_settings_defaults_to_fake():
    assert provider_from_settings(fast_settings()).name == "fake"


def test_planner_has_no_persistence_api(svc):
    s, _ = svc
    assert not any(hasattr(s, n) for n in ("submit", "persist", "enqueue", "save"))
