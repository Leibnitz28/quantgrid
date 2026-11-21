"""Skeptical unit tests for the experiment spec contracts."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from quantgrid.domain import specs as specs_mod
from quantgrid.domain.specs import (
    ABSOLUTE_MAX_TASKS,
    MAX_GRID_DIMENSIONS,
    MAX_SEEDS,
    MAX_VALUES_PER_DIMENSION,
    DatasetInfo,
    DatasetRef,
    ExperimentBatchSpec,
    FaultInjection,
    FeeModel,
    GenericScalarParams,
    GridExpansionError,
    PairsZScoreParams,
    ResourceBudget,
    SplitSpec,
    StrategyRef,
    TaskSpec,
    WorkloadSpec,
    expand_grid,
    register_strategy,
    registered_strategies,
    strategy_param_names,
    strategy_params_model,
)

SHA = "a" * 64


def make_spec(**over: Any) -> ExperimentBatchSpec:
    base: dict[str, Any] = {
        "name": "unit",
        "dataset": {"name": "ds", "sha256": SHA},
        "strategy": {"name": "pairs_zscore", "version": "1"},
        "grid": {"lookback": [20, 30]},
        "fixed_params": {"entry_z": 2.0, "exit_z": 0.5},
    }
    base.update(over)
    return ExperimentBatchSpec.model_validate(base)


def ts_split(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "mode": "timestamp",
        "train_fraction": None,
        "train_end": "2020-01-01T00:00:00Z",
        "test_start": "2020-01-02T00:00:00Z",
    }
    base.update(over)
    return base


# ---- extra fields rejected at every nesting level -------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("dataset",),
        ("strategy",),
        ("split",),
        ("fees",),
        ("workload",),
        ("budget",),
        ("faults",),
    ],
)
def test_extra_fields_rejected_everywhere(path: tuple[str, ...]) -> None:
    payload: dict[str, Any] = {
        "name": "x",
        "dataset": {"name": "ds", "sha256": SHA},
        "strategy": {"name": "pairs_zscore", "version": "1"},
        "grid": {"lookback": [20]},
        "split": {},
        "fees": {},
        "workload": {},
        "budget": {},
        "faults": {},
    }
    target = payload
    for part in path:
        target = target[part]
    target["bogus"] = 1
    with pytest.raises(ValidationError, match="bogus"):
        ExperimentBatchSpec.model_validate(payload)


def test_extra_field_in_task_spec_rejected() -> None:
    spec = make_spec()
    task = expand_grid(spec)[0].spec.model_dump()
    task["extra"] = 1
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(task)


def test_models_are_frozen() -> None:
    spec = make_spec()
    with pytest.raises(ValidationError):
        spec.name = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        spec.dataset.name = "other"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["abc", "A" * 64, "a" * 63, "a" * 65, "a" * 64 + "\n"])
def test_dataset_sha_validated(bad: str) -> None:
    with pytest.raises(ValidationError):
        DatasetRef(name="ds", sha256=bad)


@pytest.mark.parametrize("bad", ["", "Upper", "-x", "a b", "a" * 65])
def test_identifier_validated(bad: str) -> None:
    with pytest.raises(ValidationError):
        DatasetRef(name=bad, sha256=SHA)


# ---- SplitSpec ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["2020-01-01T00:00:00", "garbage", "", "2020-01-01", "2020-13-01T00:00:00Z", None]
)
def test_malformed_timestamps_rejected(bad: Any) -> None:
    with pytest.raises(ValidationError):
        SplitSpec.model_validate(ts_split(train_end=bad))
    with pytest.raises(ValidationError):
        SplitSpec.model_validate(ts_split(test_start=bad))


def test_naive_datetime_object_rejected() -> None:
    with pytest.raises(ValidationError):
        SplitSpec.model_validate(ts_split(train_end=datetime(2020, 1, 1)))


def test_numeric_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        SplitSpec.model_validate(ts_split(train_end=20200101))


def test_non_utc_offset_normalised_to_z() -> None:
    s = SplitSpec.model_validate(
        ts_split(train_end="2020-01-01T05:00:00+05:00", test_start="2020-01-01T23:00:00-05:00")
    )
    assert s.train_end == datetime(2020, 1, 1, tzinfo=UTC)
    assert s.train_end is not None and s.train_end.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    dumped = s.model_dump(mode="json")
    assert dumped["train_end"] == "2020-01-01T00:00:00Z"
    assert dumped["test_start"] == "2020-01-02T04:00:00Z"


def test_equivalent_offsets_give_same_hash() -> None:
    a = make_spec(split=ts_split(train_end="2020-01-01T05:00:00+05:00"))
    b = make_spec(split=ts_split(train_end="2020-01-01T00:00:00Z"))
    assert a.spec_sha() == b.spec_sha()


def test_fraction_mode_defaults() -> None:
    s = SplitSpec()
    assert s.mode == "fraction" and s.train_fraction == 0.6 and s.embargo_bars == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "fraction", "train_fraction": None},
        {"mode": "fraction", "train_end": "2020-01-01T00:00:00Z"},
        {"mode": "fraction", "test_start": "2020-01-01T00:00:00Z"},
        {"mode": "timestamp"},  # default train_fraction=0.6 and no timestamps
        {"mode": "timestamp", "train_fraction": None, "train_end": "2020-01-01T00:00:00Z"},
        {"mode": "timestamp", "train_fraction": None, "test_start": "2020-01-01T00:00:00Z"},
        {
            "mode": "timestamp",
            "train_fraction": 0.5,
            "train_end": "2020-01-01T00:00:00Z",
            "test_start": "2020-01-02T00:00:00Z",
        },
        {"mode": "other"},
    ],
)
def test_split_mode_consistency(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SplitSpec.model_validate(payload)


def test_train_end_after_test_start_rejected_but_equal_allowed() -> None:
    with pytest.raises(ValidationError, match="train_end"):
        SplitSpec.model_validate(
            ts_split(train_end="2020-01-02T00:00:01Z", test_start="2020-01-02T00:00:00Z")
        )
    SplitSpec.model_validate(
        ts_split(train_end="2020-01-02T00:00:00Z", test_start="2020-01-02T00:00:00Z")
    )


@pytest.mark.parametrize("frac", [0.05, 0.95, 0.0, 1.0, -1.0, float("nan"), float("inf")])
def test_train_fraction_bounds(frac: float) -> None:
    with pytest.raises(ValidationError):
        SplitSpec(train_fraction=frac)


@pytest.mark.parametrize("emb", [-1, 100_001])
def test_embargo_bounds(emb: int) -> None:
    with pytest.raises(ValidationError):
        SplitSpec(embargo_bars=emb)


# ---- grid / seeds limits --------------------------------------------------------------


def test_grid_dimension_limit() -> None:
    ok = {f"p{i}": [1] for i in range(MAX_GRID_DIMENSIONS)}
    make_spec(grid=ok, fixed_params={})
    ok["p_extra"] = [1]
    with pytest.raises(ValidationError, match="dimensions"):
        make_spec(grid=ok, fixed_params={})


def test_grid_values_per_dimension_limit() -> None:
    make_spec(grid={"lookback": list(range(5, 5 + MAX_VALUES_PER_DIMENSION))})
    with pytest.raises(ValidationError, match="values"):
        make_spec(grid={"lookback": list(range(5, 6 + MAX_VALUES_PER_DIMENSION))})


def test_empty_dimension_and_empty_grid_rejected() -> None:
    with pytest.raises(ValidationError, match="empty"):
        make_spec(grid={"lookback": []})
    with pytest.raises(ValidationError):
        make_spec(grid={})


@pytest.mark.parametrize("name", ["Lookback", "1abc", "has-dash", "", "a" * 49])
def test_param_name_pattern(name: str) -> None:
    with pytest.raises(ValidationError):
        make_spec(grid={name: [1]}, fixed_params={})


def test_seed_limits() -> None:
    make_spec(seeds=list(range(MAX_SEEDS)))
    with pytest.raises(ValidationError, match="too many"):
        make_spec(seeds=list(range(MAX_SEEDS + 1)))
    with pytest.raises(ValidationError, match="unique"):
        make_spec(seeds=[1, 1])
    with pytest.raises(ValidationError):
        make_spec(seeds=[])
    with pytest.raises(ValidationError):
        make_spec(seeds=[-1])
    with pytest.raises(ValidationError):
        make_spec(seeds=[2**63])
    make_spec(seeds=[2**63 - 1])


def test_grid_fixed_overlap_rejected() -> None:
    with pytest.raises(ValidationError, match="both in grid and fixed_params"):
        make_spec(grid={"lookback": [20]}, fixed_params={"lookback": 20, "entry_z": 2, "exit_z": 1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_attempts", 0),
        ("max_attempts", 21),
        ("priority", 11),
        ("priority", -11),
        ("cache_policy", "maybe"),
        ("name", ""),
        ("name", "x" * 121),
        ("name", "bad<name>"),
        ("description", "x" * 2001),
    ],
)
def test_scalar_field_bounds(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        make_spec(**{field: value})


def test_tags_limits() -> None:
    make_spec(tags={f"k{i}": "v" for i in range(16)})
    with pytest.raises(ValidationError):
        make_spec(tags={f"k{i}": "v" for i in range(17)})
    with pytest.raises(ValidationError):
        make_spec(tags={"Bad": "v"})
    with pytest.raises(ValidationError):
        make_spec(tags={"k": "v" * 129})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [1], {"a": 1}, "x" * 65])
def test_non_scalar_or_nonfinite_param_values_rejected(value: Any) -> None:
    with pytest.raises(ValidationError):
        make_spec(grid={"lookback": [value]})
    with pytest.raises(ValidationError):
        make_spec(fixed_params={"entry_z": value, "exit_z": 0.5})


def test_misc_model_bounds() -> None:
    for model, kw in [
        (FeeModel, {"commission_bps": -1}),
        (FeeModel, {"slippage_bps": 101}),
        (FeeModel, {"commission_bps": float("nan")}),
        (WorkloadSpec, {"bootstrap_samples": -1}),
        (WorkloadSpec, {"bootstrap_samples": 50_001}),
        (WorkloadSpec, {"bootstrap_block": 0}),
        (ResourceBudget, {"max_runtime_s": 0}),
        (ResourceBudget, {"max_memory_mb": 63}),
        (FaultInjection, {"fail_on_attempts": [0]}),
        (FaultInjection, {"fail_on_attempts": [101]}),
        (FaultInjection, {"error_kind": "weird"}),
        (FaultInjection, {"sleep_s": 121}),
    ]:
        with pytest.raises(ValidationError):
            model(**kw)


def test_fault_injection_tuple_serialises_as_list() -> None:
    f = FaultInjection(fail_on_attempts=[1, 2])
    assert f.model_dump(mode="json")["fail_on_attempts"] == [1, 2]


# ---- task_count / expansion limits ----------------------------------------------------


def test_task_count_without_expansion() -> None:
    spec = make_spec(
        grid={"lookback": [20, 30, 40], "entry_z": [2.0, 3.0]},
        fixed_params={"exit_z": 0.5},
        seeds=[0, 1],
    )
    assert spec.task_count() == 12
    assert len(expand_grid(spec)) == 12


def test_huge_grid_rejected_fast_without_materialising() -> None:
    grid = {f"p{i}": list(range(MAX_VALUES_PER_DIMENSION)) for i in range(MAX_GRID_DIMENSIONS)}
    spec = make_spec(grid=grid, fixed_params={}, seeds=list(range(MAX_SEEDS)))
    assert spec.task_count() == 256**8 * 64
    t0 = time.perf_counter()
    with pytest.raises(GridExpansionError, match="max"):
        expand_grid(spec)
    assert time.perf_counter() - t0 < 0.5
    # 10^9-ish tasks
    spec2 = make_spec(grid={f"p{i}": list(range(32)) for i in range(6)}, fixed_params={})
    assert spec2.task_count() == 32**6 > 10**9
    t0 = time.perf_counter()
    with pytest.raises(GridExpansionError):
        expand_grid(spec2)
    assert time.perf_counter() - t0 < 0.5


def test_max_tasks_boundary() -> None:
    spec = make_spec(grid={"lookback": list(range(5, 15))}, seeds=[0, 1])
    assert len(expand_grid(spec, max_tasks=20)) == 20
    with pytest.raises(GridExpansionError):
        expand_grid(spec, max_tasks=19)
    assert ABSOLUTE_MAX_TASKS == 100_000


# ---- expansion ordering, ordinals, duplicates ------------------------------------------


def test_deterministic_order_and_ordinals() -> None:
    spec = make_spec(
        grid={"lookback": [30, 20], "entry_z": [3.0, 2.0]},
        fixed_params={"exit_z": 0.5},
        seeds=[7, 3],
    )
    tasks = expand_grid(spec)
    assert [t.ordinal for t in tasks] == list(range(8))
    # params sorted alphabetically (entry_z outer, lookback next), seeds innermost, values in given order
    got = [(t.spec.params["entry_z"], t.spec.params["lookback"], t.spec.seed) for t in tasks]
    assert got == [
        (3.0, 30, 7), (3.0, 30, 3), (3.0, 20, 7), (3.0, 20, 3),
        (2.0, 30, 7), (2.0, 30, 3), (2.0, 20, 7), (2.0, 20, 3),
    ]  # fmt: skip
    again = expand_grid(spec)
    assert [t.spec_hash for t in tasks] == [t.spec_hash for t in again]
    assert len({t.spec_hash for t in tasks}) == 8
    for t in tasks:
        assert t.spec_hash == t.spec.spec_hash()


def test_grid_key_order_does_not_change_expansion() -> None:
    a = make_spec(grid={"lookback": [20, 30], "entry_z": [2.0, 3.0]}, fixed_params={"exit_z": 0.5})
    b = make_spec(grid={"entry_z": [2.0, 3.0], "lookback": [20, 30]}, fixed_params={"exit_z": 0.5})
    assert [t.spec_hash for t in expand_grid(a)] == [t.spec_hash for t in expand_grid(b)]
    assert a.spec_sha() == b.spec_sha()


def test_duplicate_task_specs_detected() -> None:
    spec = make_spec(grid={"lookback": [20, 20]})
    with pytest.raises(GridExpansionError, match="invalid grid") as ei:
        expand_grid(spec)
    assert any("duplicate" in p for p in ei.value.problems)


def test_duplicates_after_normalisation_detected() -> None:
    spec = make_spec(grid={"lookback": [20, "20", 20.0]})
    with pytest.raises(GridExpansionError) as ei:
        expand_grid(spec)
    assert sum("duplicate" in p for p in ei.value.problems) == 2


def test_invalid_combinations_report_offending_point() -> None:
    spec = make_spec(
        grid={"exit_z": [0.5, 2.0, 3.0]}, fixed_params={"lookback": 20, "entry_z": 2.0}
    )
    with pytest.raises(GridExpansionError) as ei:
        expand_grid(spec)
    problems = ei.value.problems
    assert len(problems) == 2
    assert any("'exit_z': 2.0" in p and "exit_z must be < entry_z" in p for p in problems)
    assert any("'exit_z': 3.0" in p for p in problems)
    assert not any("'exit_z': 0.5" in p for p in problems)


def test_stop_z_not_above_entry_reported() -> None:
    spec = make_spec(
        grid={"stop_z": [1.0, 2.0, 2.5]},
        fixed_params={"lookback": 20, "entry_z": 2.0, "exit_z": 0.5},
    )
    with pytest.raises(GridExpansionError) as ei:
        expand_grid(spec)
    assert len(ei.value.problems) == 2
    assert all("stop_z must be > entry_z" in p for p in ei.value.problems)
    assert any("'stop_z': 1.0" in p for p in ei.value.problems)


def test_problem_list_is_capped_but_all_failures_raise() -> None:
    spec = make_spec(
        grid={"lookback": list(range(-100, 0))}, fixed_params={"entry_z": 2.0, "exit_z": 0.5}
    )
    with pytest.raises(GridExpansionError) as ei:
        expand_grid(spec)
    assert len(ei.value.problems) == 20


def test_missing_required_param_reported() -> None:
    spec = make_spec(grid={"lookback": [20]}, fixed_params={"entry_z": 2.0})
    with pytest.raises(GridExpansionError) as ei:
        expand_grid(spec)
    assert any("exit_z" in p for p in ei.value.problems)


def test_unknown_params_rejected() -> None:
    spec = make_spec(fixed_params={"entry_z": 2.0, "exit_z": 0.5, "bogus": 1})
    with pytest.raises(GridExpansionError, match="unknown parameters"):
        expand_grid(spec)
    spec = make_spec(grid={"lookback": [20], "bogus": [1]})
    with pytest.raises(GridExpansionError, match="bogus"):
        expand_grid(spec)


def test_unknown_strategy_rejected() -> None:
    spec = make_spec(strategy={"name": "nope", "version": "1"})
    with pytest.raises(GridExpansionError, match="unknown strategy"):
        expand_grid(spec)
    spec = make_spec(strategy={"name": "pairs_zscore", "version": "2"})
    with pytest.raises(GridExpansionError, match="unknown strategy"):
        expand_grid(spec)


def test_out_of_range_params_rejected() -> None:
    for grid in ({"lookback": [4]}, {"lookback": [5001]}, {"lookback": [True]}):
        with pytest.raises(GridExpansionError):
            expand_grid(make_spec(grid=grid))


def test_fractional_lookback_rejected() -> None:
    with pytest.raises(GridExpansionError):
        expand_grid(make_spec(grid={"lookback": [20.5]}))


# ---- normalisation / hashing ----------------------------------------------------------


def test_int_float_str_normalisation_same_hash() -> None:
    hashes = set()
    for lookback in (20, "20", 20.0):
        spec = make_spec(
            grid={"lookback": [lookback]}, fixed_params={"entry_z": "2", "exit_z": 0.5}
        )
        (task,) = expand_grid(spec)
        assert task.spec.params["lookback"] == 20 and isinstance(task.spec.params["lookback"], int)
        assert task.spec.params["entry_z"] == 2.0 and isinstance(task.spec.params["entry_z"], float)
        hashes.add(task.spec_hash)
    assert len(hashes) == 1


def test_null_optional_params_equal_omitted() -> None:
    a = expand_grid(make_spec(fixed_params={"entry_z": 2.0, "exit_z": 0.5, "stop_z": None}))
    b = expand_grid(make_spec(fixed_params={"entry_z": 2.0, "exit_z": 0.5}))
    assert [t.spec_hash for t in a] == [t.spec_hash for t in b]
    assert a[0].spec.params["stop_z"] is None
    assert a[0].spec.params["max_holding_bars"] is None


def test_task_params_are_fully_materialised() -> None:
    (task,) = expand_grid(make_spec(grid={"lookback": [20]}))
    assert set(task.spec.params) == set(PairsZScoreParams.model_fields)


def test_spec_hash_changes_with_every_result_relevant_field() -> None:
    base = expand_grid(make_spec(grid={"lookback": [20]}))[0].spec
    variants = [
        make_spec(grid={"lookback": [21]}),
        make_spec(grid={"lookback": [20]}, fees={"commission_bps": 2.0}),
        make_spec(grid={"lookback": [20]}, workload={"bootstrap_samples": 10}),
        make_spec(grid={"lookback": [20]}, split={"train_fraction": 0.7}),
        make_spec(grid={"lookback": [20]}, seeds=[1]),
        make_spec(grid={"lookback": [20]}, dataset={"name": "ds", "sha256": "b" * 64}),
        make_spec(grid={"lookback": [20]}, faults={"fail_on_attempts": [1]}),
    ]
    hashes = {expand_grid(v)[0].spec_hash for v in variants} | {base.spec_hash()}
    assert len(hashes) == len(variants) + 1


def test_spec_hash_ignores_execution_limits() -> None:
    a = expand_grid(make_spec(grid={"lookback": [20]}))[0].spec_hash
    b = expand_grid(
        make_spec(
            grid={"lookback": [20]},
            budget={"max_runtime_s": 5},
            max_attempts=1,
            priority=3,
            name="other",
            cache_policy="bypass",
            tags={"a": "b"},
        )
    )[0].spec_hash
    assert a == b


def test_golden_hashes() -> None:
    spec = make_spec()
    assert spec.spec_sha() == "31bd0f30602f83221d5b1e9185f504ce6b74e3102f787ec785458b8b1c417d29"
    tasks = expand_grid(spec)
    assert tasks[0].spec_hash == "a582562c0b17da3e17dce3b018291791d6c529062be33fd7ea0dcfb123b50ed0"


def test_spec_sha_roundtrip_stable() -> None:
    spec = make_spec(split=ts_split(train_end="2020-01-01T05:00:00+05:00"), seeds=[3, 1, 2])
    again = ExperimentBatchSpec.model_validate(spec.canonical())
    assert again == spec and again.spec_sha() == spec.spec_sha()
    ((t,),) = [expand_grid(make_spec(grid={"lookback": [20]}))]
    assert TaskSpec.model_validate(t.spec.canonical()).spec_hash() == t.spec_hash


def test_seed_order_matters_to_batch_sha_not_task_hashes() -> None:
    a = make_spec(seeds=[1, 2])
    b = make_spec(seeds=[2, 1])
    assert a.spec_sha() != b.spec_sha()
    assert {t.spec_hash for t in expand_grid(a)} == {t.spec_hash for t in expand_grid(b)}


# ---- strategy registry ----------------------------------------------------------------


@pytest.fixture
def restore_registry() -> Any:
    saved = dict(specs_mod._STRATEGY_PARAMS)
    yield
    specs_mod._STRATEGY_PARAMS.clear()
    specs_mod._STRATEGY_PARAMS.update(saved)


def test_builtin_strategy_registered() -> None:
    assert ("pairs_zscore", "1") in registered_strategies()
    assert strategy_params_model(StrategyRef(name="pairs_zscore", version="1")) is PairsZScoreParams
    assert strategy_params_model(StrategyRef(name="zzz", version="1")) is None
    assert strategy_param_names(StrategyRef(name="zzz", version="1")) == []
    assert "lookback" in strategy_param_names(StrategyRef(name="pairs_zscore", version="1"))


def test_register_generic_strategy(restore_registry: None) -> None:
    register_strategy("ext_bt", "2", GenericScalarParams)
    spec = make_spec(
        strategy={"name": "ext_bt", "version": "2"},
        grid={"fast": [3, 5], "mode": ["a", "b"]},
        fixed_params={"flag": True, "opt": None},
    )
    tasks = expand_grid(spec)
    assert len(tasks) == 4
    assert tasks[0].spec.params == {"flag": True, "opt": None, "fast": 3, "mode": "a"}
    assert ("ext_bt", "2") in registered_strategies()
    # unknown names are allowed for generic models; values are not normalised
    assert len({t.spec_hash for t in tasks}) == 4


def test_register_custom_model(restore_registry: None) -> None:
    class P(BaseModel):
        model_config = {"extra": "forbid", "frozen": True}
        a: int

    register_strategy("custom", "1", P)
    ok = make_spec(
        strategy={"name": "custom", "version": "1"}, grid={"a": [1, "2"]}, fixed_params={}
    )
    assert [t.spec.params for t in expand_grid(ok)] == [{"a": 1}, {"a": 2}]
    with pytest.raises(GridExpansionError, match="unknown parameters"):
        expand_grid(
            make_spec(strategy={"name": "custom", "version": "1"}, grid={"b": [1]}, fixed_params={})
        )


def test_generic_params_reject_nested_values() -> None:
    with pytest.raises(ValidationError):
        GenericScalarParams.model_validate({"a": [1]})
    with pytest.raises(ValidationError):
        GenericScalarParams.model_validate({"a": float("nan")})
    with pytest.raises(ValidationError):
        GenericScalarParams.model_validate([1])


# ---- DatasetInfo ----------------------------------------------------------------------


def test_dataset_info_validation() -> None:
    kw: dict[str, Any] = {
        "name": "ds", "sha256": SHA, "n_bars": 10, "start": "2020-01-01T00:00:00Z",
        "end": "2020-01-02T00:00:00Z", "bar_interval_s": 60, "columns": ["a"], "synthetic": True,
        "license": "x", "description": "d",
    }  # fmt: skip
    info = DatasetInfo.model_validate(kw)
    assert info.columns == ("a",)
    with pytest.raises(ValidationError):
        DatasetInfo.model_validate({**kw, "start": "2020-01-01T00:00:00"})
    with pytest.raises(ValidationError):
        DatasetInfo.model_validate({**kw, "extra": 1})


def test_task_hash_matches_hand_built_canonical_json() -> None:
    import hashlib

    task = expand_grid(make_spec(grid={"lookback": [20]}))[0].spec
    text = (
        '{"dataset":{"name":"ds","sha256":"' + SHA + '"},"faults":null,'
        '"fees":{"commission_bps":1.0,"slippage_bps":0.5},'
        '"params":{"entry_z":2.0,"exit_z":0.5,"lookback":20,"max_holding_bars":null,"stop_z":null},'
        '"result_schema_version":1,"schema_version":1,"seed":0,'
        '"split":{"embargo_bars":0,"mode":"fraction","test_start":null,"train_end":null,"train_fraction":0.6},'
        '"strategy":{"name":"pairs_zscore","version":"1"},'
        '"workload":{"bootstrap_block":20,"bootstrap_samples":0}}'
    )
    assert task.spec_hash() == hashlib.sha256(text.encode()).hexdigest()
