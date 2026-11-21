"""Task engine: contract, purity, faults, cancellation, dispatch, golden regression fixture.

The golden fixture ``fixtures/golden/engine_golden.json`` pins the exact metrics (and a
digest of every output series) of a few fixed task specs. Any numerical change to the
generator, strategy, metrics or bootstrap fails :func:`test_golden_fixture` with an exact
comparison. After an *intended* change, regenerate and review the diff:

    QG_REGEN_GOLDEN=1 .venv/bin/pytest tests/unit/test_engine.py -k golden
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel

from quantgrid.domain import specs as specs_module
from quantgrid.domain.bootstrap import BOOTSTRAP_METRIC_KEYS, stationary_bootstrap_sharpe
from quantgrid.domain.datasets import DATASET_DEFINITIONS, dataset_ref, generate_bars
from quantgrid.domain.engine import (
    ENGINE_VERSION,
    get_engine,
    register_engine,
    registered_engines,
    run_pairs_zscore_task,
    run_task_spec,
    unregister_engine,
)
from quantgrid.domain.errors import (
    DataIntegrityError,
    ErrorCode,
    PermanentTaskError,
    SpecError,
    TaskAborted,
    TaskTimeout,
    TransientTaskError,
)
from quantgrid.domain.hashing import canonical_json
from quantgrid.domain.metrics import CORE_METRIC_KEYS, periods_per_year
from quantgrid.domain.specs import (
    FaultInjection,
    FeeModel,
    GenericScalarParams,
    SplitSpec,
    StrategyRef,
    TaskSpec,
    WorkloadSpec,
)
from quantgrid.domain.types import (
    Bars,
    CancelToken,
    DeadlineCancelToken,
    NullCancelToken,
    TaskOutput,
)

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "golden" / "engine_golden.json"
STRATEGY = StrategyRef(name="pairs_zscore", version="1")
OU_DATASETS = ["pairs-ou-small-v1", "pairs-ou-medium-v1", "pairs-ou-large-v1", "pairs-ou-gappy-v1"]


class CountingToken:
    def __init__(self, abort_at: int | None = None) -> None:
        self.checks = 0
        self.abort_at = abort_at

    def check(self) -> None:
        self.checks += 1
        if self.abort_at is not None and self.checks >= self.abort_at:
            raise TaskAborted("CANCELED")


@pytest.fixture(scope="module")
def bars() -> dict[str, Bars]:
    return {name: generate_bars(name) for name in DATASET_DEFINITIONS}


def make_spec(dataset: str = "pairs-ou-small-v1", **overrides: Any) -> TaskSpec:
    fields: dict[str, Any] = {
        "dataset": dataset_ref(dataset),
        "strategy": STRATEGY,
        "params": {"lookback": 30, "entry_z": 2.0, "exit_z": 0.5},
        "split": SplitSpec(train_fraction=0.6),
        "fees": FeeModel(),
        "seed": 3,
        "workload": WorkloadSpec(),
    }
    fields.update(overrides)
    return TaskSpec(**fields)


def run(
    spec: TaskSpec, data: Bars, cancel: CancelToken | None = None, attempt: int = 1
) -> TaskOutput:
    return run_task_spec(spec, data, cancel=cancel or NullCancelToken(), attempt=attempt)


def assert_same_output(a: TaskOutput, b: TaskOutput) -> None:
    assert list(a.series) == list(b.series)
    for key in a.series:
        assert a.series[key].dtype == b.series[key].dtype, key
        assert np.array_equal(a.series[key], b.series[key], equal_nan=True), key
    assert dict(a.metrics) == dict(b.metrics)
    assert [type(v) for v in a.metrics.values()] == [type(v) for v in b.metrics.values()]
    assert dict(a.diagnostics) == dict(b.diagnostics)


# --------------------------------------------------------------------------------------
# Output contract and purity
# --------------------------------------------------------------------------------------


class TestContract:
    def test_output_shapes_types_and_keys(self, bars: dict[str, Bars]) -> None:
        out = run(make_spec(), bars["pairs-ou-small-v1"])
        assert tuple(out.metrics) == CORE_METRIC_KEYS + BOOTSTRAP_METRIC_KEYS
        n_test = out.metrics["n_test_bars"]
        assert n_test == 2000 - 1200
        assert list(out.series) == ["ts", "spread", "zscore", "position", "ret", "equity"]
        for key, arr in out.series.items():
            assert arr.shape == (n_test,), key
            assert arr.flags.c_contiguous
        assert out.series["ts"].dtype == np.int64
        assert out.series["position"].dtype == np.int64
        for key in ("spread", "zscore", "ret", "equity"):
            assert out.series[key].dtype == np.float64
        assert out.diagnostics["engine_version"] == ENGINE_VERSION == "pairs_zscore/1"
        assert out.diagnostics["test_start_index"] == 1200
        assert all(isinstance(v, int | float | str) for v in out.diagnostics.values())

    def test_metrics_are_finite_json(self, bars: dict[str, Bars]) -> None:
        for name in DATASET_DEFINITIONS:
            out = run(make_spec(name), bars[name])
            for key, value in out.metrics.items():
                assert value is None or math.isfinite(value), (name, key)
            canonical_json(dict(out.metrics))
            canonical_json(dict(out.diagnostics))

    def test_pure_function_bit_identical(self, bars: dict[str, Bars]) -> None:
        spec = make_spec("pairs-ou-medium-v1", workload=WorkloadSpec(bootstrap_samples=120))
        first = run(spec, bars["pairs-ou-medium-v1"])
        second = run(spec, bars["pairs-ou-medium-v1"], cancel=CountingToken(), attempt=3)
        assert_same_output(first, second)
        # a deep copy of the input bars gives the same result as well
        b = bars["pairs-ou-medium-v1"]
        copied = Bars(
            name=b.name,
            ts=b.ts.copy(),
            close_x=b.close_x.copy(),
            close_y=b.close_y.copy(),
            sha256=b.sha256,
        )
        assert_same_output(first, run(spec, copied))

    def test_input_bars_are_not_modified(self, bars: dict[str, Bars]) -> None:
        b = bars["pairs-ou-small-v1"]
        before = (b.ts.copy(), b.close_x.copy(), b.close_y.copy())
        run(make_spec(), b)
        assert np.array_equal(b.ts, before[0])
        assert np.array_equal(b.close_x, before[1])
        assert np.array_equal(b.close_y, before[2])

    @pytest.mark.parametrize("name", OU_DATASETS)
    def test_strategy_trades_on_every_ou_dataset(self, name: str, bars: dict[str, Bars]) -> None:
        spec = make_spec(name, params={"lookback": 50, "entry_z": 2.0, "exit_z": 0.5})
        out = run(spec, bars[name])
        assert int(out.metrics["n_trades"] or 0) >= 10

    def test_medium_dataset_runs_fast(self, bars: dict[str, Bars]) -> None:
        spec = make_spec("pairs-ou-medium-v1")
        run(spec, bars["pairs-ou-medium-v1"])  # warm-up
        start = time.perf_counter()
        run(spec, bars["pairs-ou-medium-v1"])
        assert time.perf_counter() - start < 1.0  # measured ~5 ms; generous for loaded CI


class TestBootstrapWorkload:
    def test_keys_are_none_without_samples(self, bars: dict[str, Bars]) -> None:
        out = run(make_spec(), bars["pairs-ou-small-v1"])
        assert out.metrics["bootstrap_samples"] == 0
        for key in BOOTSTRAP_METRIC_KEYS[:-1]:
            assert out.metrics[key] is None

    def test_bootstrap_metrics_match_direct_call(self, bars: dict[str, Bars]) -> None:
        spec = make_spec(workload=WorkloadSpec(bootstrap_samples=200, bootstrap_block=7), seed=41)
        out = run(spec, bars["pairs-ou-small-v1"])
        direct = stationary_bootstrap_sharpe(
            out.series["ret"],  # type: ignore[arg-type]
            samples=200,
            mean_block=7,
            seed=41,
            periods_per_year=periods_per_year(out.series["ts"]),  # type: ignore[arg-type]
        )
        for key, value in direct.as_metrics().items():
            assert out.metrics[key] == value
        assert out.metrics["sharpe_ci_low"] <= out.metrics["sharpe_ci_high"]  # type: ignore[operator]

    def test_seed_only_affects_the_bootstrap(self, bars: dict[str, Bars]) -> None:
        workload = WorkloadSpec(bootstrap_samples=50)
        a = run(make_spec(seed=1, workload=workload), bars["pairs-ou-small-v1"])
        b = run(make_spec(seed=2, workload=workload), bars["pairs-ou-small-v1"])
        for key in CORE_METRIC_KEYS:
            assert a.metrics[key] == b.metrics[key]
        assert a.metrics["sharpe_boot_mean"] != b.metrics["sharpe_boot_mean"]


# --------------------------------------------------------------------------------------
# Integrity, params, faults
# --------------------------------------------------------------------------------------


class TestIntegrityAndParams:
    def test_dataset_signature_mismatch(self, bars: dict[str, Bars]) -> None:
        with pytest.raises(DataIntegrityError, match="signature") as info:
            run(make_spec("pairs-ou-medium-v1"), bars["pairs-ou-small-v1"])
        assert info.value.code == ErrorCode.DATA_INTEGRITY_ERROR
        assert info.value.retryable is False

    def test_invalid_bars_are_rejected(self, bars: dict[str, Bars]) -> None:
        b = bars["pairs-ou-small-v1"]
        cx = b.close_x.copy()
        cx[500] = np.nan
        broken = Bars(name=b.name, ts=b.ts, close_x=cx, close_y=b.close_y, sha256=b.sha256)
        with pytest.raises(DataIntegrityError, match="non-finite"):
            run(make_spec(), broken)

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"lookback": 30, "entry_z": 0.5, "exit_z": 0.5}, "exit_z must be < entry_z"),
            ({"lookback": 30, "entry_z": 2.0}, "exit_z"),
            ({"lookback": 30, "entry_z": 2.0, "exit_z": 0.5, "bogus": 1}, "bogus"),
            ({"lookback": 2, "entry_z": 2.0, "exit_z": 0.5}, "lookback"),
        ],
    )
    def test_invalid_params_raise_spec_error(
        self, params: dict[str, Any], match: str, bars: dict[str, Bars]
    ) -> None:
        with pytest.raises(SpecError, match=match) as info:
            run(make_spec(params=params), bars["pairs-ou-small-v1"])
        assert info.value.code == ErrorCode.SPEC_ERROR
        assert info.value.details["errors"]

    def test_protocol_violation_raises_spec_error(self, bars: dict[str, Bars]) -> None:
        spec = make_spec(params={"lookback": 500, "entry_z": 2.0, "exit_z": 0.5})
        with pytest.raises(SpecError, match="lookback 500"):
            run(spec, bars["pairs-ou-small-v1"])

    def test_unknown_strategy(self, bars: dict[str, Bars]) -> None:
        spec = make_spec(strategy=StrategyRef(name="nope", version="1"))
        with pytest.raises(SpecError, match="unknown strategy"):
            run(spec, bars["pairs-ou-small-v1"])

    def test_strategy_without_engine(
        self, bars: dict[str, Bars], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(specs_module._STRATEGY_PARAMS, ("orphan", "1"), GenericScalarParams)
        spec = make_spec(strategy=StrategyRef(name="orphan", version="1"), params={"a": 1})
        with pytest.raises(SpecError, match="no engine registered"):
            run(spec, bars["pairs-ou-small-v1"])

    def test_attempt_numbers_are_one_based(self, bars: dict[str, Bars]) -> None:
        with pytest.raises(ValueError, match="1-based"):
            run(make_spec(), bars["pairs-ou-small-v1"], attempt=0)


class TestFaultInjection:
    def test_transient_fault_then_identical_success(self, bars: dict[str, Bars]) -> None:
        data = bars["pairs-ou-small-v1"]
        spec = make_spec(faults=FaultInjection(fail_on_attempts=(1, 2), error_kind="transient"))
        for attempt in (1, 2):
            with pytest.raises(TransientTaskError, match=f"attempt {attempt}") as info:
                run(spec, data, attempt=attempt)
            assert info.value.retryable is True
            assert info.value.code == ErrorCode.TRANSIENT_ERROR
        recovered = run(spec, data, attempt=3)
        baseline = run(make_spec(), data)
        assert dict(recovered.metrics) == dict(baseline.metrics)
        assert np.array_equal(recovered.series["ret"], baseline.series["ret"])

    def test_permanent_fault(self, bars: dict[str, Bars]) -> None:
        spec = make_spec(faults=FaultInjection(fail_on_attempts=(2,), error_kind="permanent"))
        run(spec, bars["pairs-ou-small-v1"], attempt=1)
        with pytest.raises(PermanentTaskError) as info:
            run(spec, bars["pairs-ou-small-v1"], attempt=2)
        assert info.value.retryable is False
        assert info.value.details["injected"] is True

    def test_fault_sleep_takes_wall_time(self, bars: dict[str, Bars]) -> None:
        spec = make_spec(faults=FaultInjection(sleep_s=0.12))
        start = time.monotonic()
        run(spec, bars["pairs-ou-small-v1"])
        assert time.monotonic() - start >= 0.12

    def test_fault_sleep_is_cooperative(self, bars: dict[str, Bars]) -> None:
        spec = make_spec(faults=FaultInjection(sleep_s=30.0))
        token = CountingToken(abort_at=3)
        start = time.monotonic()
        with pytest.raises(TaskAborted):
            run(spec, bars["pairs-ou-small-v1"], cancel=token)
        assert time.monotonic() - start < 1.0  # 3 checks, <= 50 ms apart
        with pytest.raises(TaskTimeout):
            run(spec, bars["pairs-ou-small-v1"], cancel=DeadlineCancelToken(max_runtime_s=0.1))


# --------------------------------------------------------------------------------------
# Cancellation and deadlines
# --------------------------------------------------------------------------------------


class TestCancellation:
    def test_abort_mid_run_raises_task_aborted(self, bars: dict[str, Bars]) -> None:
        spec = make_spec("pairs-ou-medium-v1", workload=WorkloadSpec(bootstrap_samples=5000))
        token = CountingToken(abort_at=25)
        with pytest.raises(TaskAborted) as info:
            run(spec, bars["pairs-ou-medium-v1"], cancel=token)
        assert info.value.reason == "CANCELED"
        assert token.checks == 25

    def test_abort_via_deadline_token(self, bars: dict[str, Bars]) -> None:
        token = DeadlineCancelToken()
        token.abort("LEASE_LOST")
        with pytest.raises(TaskAborted) as info:
            run(make_spec(), bars["pairs-ou-small-v1"], cancel=token)
        assert info.value.reason == "LEASE_LOST"

    def test_deadline_times_out_a_long_bootstrap(self, bars: dict[str, Bars]) -> None:
        spec = make_spec("pairs-ou-medium-v1", workload=WorkloadSpec(bootstrap_samples=50_000))
        token = DeadlineCancelToken(max_runtime_s=0.25)
        start = time.monotonic()
        with pytest.raises(TaskTimeout, match=r"max_runtime_s=0\.25") as info:
            run(spec, bars["pairs-ou-medium-v1"], cancel=token)
        elapsed = time.monotonic() - start
        assert 0.25 <= elapsed < 3.0  # full run would take ~10 s
        assert info.value.code == ErrorCode.TASK_TIMEOUT
        assert info.value.retryable is True

    def test_long_strategy_loops_check_cancel(self, bars: dict[str, Bars]) -> None:
        token = CountingToken()
        run(make_spec("pairs-ou-large-v1"), bars["pairs-ou-large-v1"], cancel=token)
        # 80,000 test bars: >= 79 rolling blocks + >= 19 position-loop checks
        assert token.checks >= 98


# --------------------------------------------------------------------------------------
# Engine registry
# --------------------------------------------------------------------------------------


class ToyParams(GenericScalarParams):
    pass


def toy_engine(spec: TaskSpec, params: BaseModel, data: Bars, cancel: CancelToken) -> TaskOutput:
    cancel.check()
    return TaskOutput(
        metrics={"n_bars": len(data), "k": params.model_dump()["k"]},
        series={"ts": data.ts.copy()},
        diagnostics={"engine_version": "toy/1"},
    )


class TestRegistry:
    def test_builtin_registration(self) -> None:
        assert ("pairs_zscore", "1") in registered_engines()
        reg = get_engine("pairs_zscore", "1")
        assert reg is not None
        assert reg.fn is run_pairs_zscore_task
        assert reg.engine_version == ENGINE_VERSION

    def test_external_engine_dispatch(
        self, bars: dict[str, Bars], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(specs_module._STRATEGY_PARAMS, ("toy", "1"), ToyParams)
        register_engine("toy", "1", toy_engine)
        try:
            spec = make_spec(strategy=StrategyRef(name="toy", version="1"), params={"k": 7})
            out = run(spec, bars["pairs-ou-small-v1"])
            assert dict(out.metrics) == {"n_bars": 2000, "k": 7}
            register_engine("toy", "1", toy_engine)  # idempotent for the same function
        finally:
            unregister_engine("toy", "1")
        assert get_engine("toy", "1") is None

    def test_cannot_shadow_an_engine_by_accident(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_engine("pairs_zscore", "1", toy_engine)
        register_engine("pairs_zscore", "1", toy_engine, replace=True)
        try:
            reg = get_engine("pairs_zscore", "1")
            assert reg is not None
            assert reg.fn is toy_engine
        finally:
            register_engine("pairs_zscore", "1", run_pairs_zscore_task, replace=True)


# --------------------------------------------------------------------------------------
# Golden regression fixture
# --------------------------------------------------------------------------------------


def golden_specs() -> list[tuple[str, TaskSpec]]:
    return [
        (
            "small-baseline",
            make_spec(
                "pairs-ou-small-v1",
                params={"lookback": 30, "entry_z": 2.0, "exit_z": 0.5},
                seed=11,
            ),
        ),
        (
            "medium-bootstrap",
            make_spec(
                "pairs-ou-medium-v1",
                params={"lookback": 48, "entry_z": 1.8, "exit_z": 0.3},
                split=SplitSpec(train_fraction=0.6, embargo_bars=24),
                fees=FeeModel(commission_bps=1.0, slippage_bps=1.0),
                seed=2024,
                workload=WorkloadSpec(bootstrap_samples=300, bootstrap_block=24),
            ),
        ),
        (
            "large-stops",
            make_spec(
                "pairs-ou-large-v1",
                params={
                    "lookback": 120,
                    "entry_z": 2.2,
                    "exit_z": 0.4,
                    "stop_z": 4.0,
                    "max_holding_bars": 240,
                },
                split=SplitSpec(train_fraction=0.5),
                fees=FeeModel(commission_bps=0.5, slippage_bps=0.5),
                seed=7,
            ),
        ),
        (
            "gappy-timestamp-split",
            make_spec(
                "pairs-ou-gappy-v1",
                params={
                    "lookback": 24,
                    "entry_z": 2.0,
                    "exit_z": 0.0,
                    "stop_z": 3.5,
                    "max_holding_bars": 48,
                },
                split=SplitSpec(
                    mode="timestamp",
                    train_fraction=None,
                    train_end="2019-04-15T00:00:00Z",
                    test_start="2019-04-16T00:00:00Z",
                    embargo_bars=6,
                ),
                seed=99,
                workload=WorkloadSpec(bootstrap_samples=200, bootstrap_block=10),
            ),
        ),
    ]


def series_digest(series: Mapping[str, npt.NDArray[np.generic]]) -> str:
    digest = hashlib.sha256()
    for key, arr in series.items():
        digest.update(f"{key}\0{arr.dtype.str}\0{arr.shape[0]}\0".encode())
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def golden_record(case_id: str, spec: TaskSpec, data: Bars) -> dict[str, Any]:
    out = run(spec, data)
    return {
        "id": case_id,
        "spec": spec.canonical(),
        "spec_hash": spec.spec_hash(),
        "metrics": dict(out.metrics),
        "series_sha256": series_digest(out.series),
        "diagnostics": dict(out.diagnostics),
    }


def write_golden(data: dict[str, Bars]) -> None:
    document = {
        "format": "quantgrid.engine-golden/1",
        "engine_version": ENGINE_VERSION,
        "recorded_with": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "regenerate": "QG_REGEN_GOLDEN=1 .venv/bin/pytest tests/unit/test_engine.py -k golden",
        "cases": [
            golden_record(case_id, spec, data[spec.dataset.name])
            for case_id, spec in golden_specs()
        ],
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def test_golden_fixture(bars: dict[str, Bars]) -> None:
    if os.environ.get("QG_REGEN_GOLDEN") == "1":
        write_golden(bars)
    document = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert document["engine_version"] == ENGINE_VERSION
    cases = {case["id"]: case for case in document["cases"]}
    assert set(cases) == {case_id for case_id, _ in golden_specs()}
    for case_id, spec in golden_specs():
        recorded = cases[case_id]
        stored_spec = TaskSpec.model_validate(recorded["spec"])
        assert stored_spec == spec, case_id
        assert stored_spec.spec_hash() == recorded["spec_hash"], case_id
        fresh = golden_record(case_id, stored_spec, bars[stored_spec.dataset.name])
        for key, value in recorded["metrics"].items():
            got = fresh["metrics"][key]
            assert type(got) is type(value), (case_id, key, got, value)
            assert got == value, (case_id, key, got, value)
        assert list(fresh["metrics"]) == list(recorded["metrics"]), case_id
        assert fresh["series_sha256"] == recorded["series_sha256"], case_id
        assert fresh["diagnostics"] == recorded["diagnostics"], case_id


def test_golden_fixture_covers_the_required_cases() -> None:
    specs = dict(golden_specs())
    datasets = {spec.dataset.name for spec in specs.values()}
    assert {"pairs-ou-small-v1", "pairs-ou-medium-v1", "pairs-ou-large-v1"} <= datasets
    assert any(spec.workload.bootstrap_samples > 0 for spec in specs.values())
    assert any(
        spec.params.get("stop_z") is not None and spec.params.get("max_holding_bars") is not None
        for spec in specs.values()
    )


def test_engine_rejects_bars_whose_content_does_not_match_their_claimed_signature(
    bars: dict[str, Bars],
) -> None:
    """Review finding: a Bars object carrying the right sha256 label but tampered prices
    (e.g. loaded with verify=False, or a corrupted in-process cache) must not be run."""
    good = bars["pairs-ou-small-v1"]
    cx = good.close_x.copy()
    cx[1234] *= 1.01
    forged = Bars(good.name, good.ts, cx, good.close_y, good.sha256)
    with pytest.raises(DataIntegrityError, match="content"):
        run(make_spec(), forged)
