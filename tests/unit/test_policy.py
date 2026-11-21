"""Unit tests for PolicyService admission control."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from quantgrid.config import Settings
from quantgrid.domain.policy import (
    MIN_TEST_BARS,
    MIN_TRAIN_BARS,
    QUOTA_VIOLATIONS,
    TEST_BARS_PER_LOOKBACK,
    CostModel,
    PolicyDecision,
    PolicyService,
    QuotaPolicy,
    Violation,
    estimate_split,
    violation_messages,
)
from quantgrid.domain.specs import DatasetInfo, ExperimentBatchSpec, expand_grid

SHA = "a" * 64
START = datetime(2020, 1, 1, tzinfo=UTC)
N_BARS = 10_000
INTERVAL = 60
END = START + timedelta(seconds=INTERVAL * (N_BARS - 1))


def make_ds(name: str = "ds", sha: str = SHA, n_bars: int = N_BARS) -> DatasetInfo:
    return DatasetInfo(
        name=name,
        sha256=sha,
        n_bars=n_bars,
        start=START,
        end=START + timedelta(seconds=INTERVAL * (n_bars - 1)),
        bar_interval_s=INTERVAL,
        columns=("ts", "a", "b"),
        synthetic=True,
        license="synthetic",
        description="test",
    )


class FakeCatalog:
    def __init__(self, *infos: DatasetInfo) -> None:
        self._d = {i.name: i for i in infos}

    def get(self, name: str) -> DatasetInfo | None:
        return self._d.get(name)

    def list(self) -> Sequence[DatasetInfo]:
        return list(self._d.values())


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


@pytest.fixture
def svc() -> PolicyService:
    return PolicyService(FakeCatalog(make_ds()))


def codes(decision: PolicyDecision) -> set[str]:
    return {v.code for v in decision.violations}


def test_baseline_allowed(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(), QuotaPolicy())
    assert d.allowed and not d.violations and d.task_count == 2
    assert d.estimate is not None


def test_estimate_fields(svc: PolicyService) -> None:
    spec = make_spec(
        workload={"bootstrap_samples": 1000}, budget={"max_runtime_s": 60, "max_memory_mb": 512}
    )
    d = svc.evaluate(spec, QuotaPolicy())
    est = d.estimate
    assert est is not None
    cm = CostModel()
    assert est.task_count == 2
    assert est.n_train_bars_est == 6000 and est.n_test_bars_est == 4000
    exp_s = cm.seconds_per_task(N_BARS, 4000, 1000)
    exp_mb = cm.memory_mb_per_task(N_BARS, 4000, 1000)
    assert est.est_seconds_per_task == pytest.approx(exp_s, abs=1e-6)
    assert est.est_total_worker_seconds == pytest.approx(exp_s * 2, abs=1e-3)
    assert est.est_memory_mb_per_task == pytest.approx(exp_mb, abs=0.05)
    assert est.budget_worker_seconds == 2 * 60


def test_cost_model_monotone_and_chunked() -> None:
    cm = CostModel()
    assert cm.seconds_per_task(1000, 500, 0) == pytest.approx(0.05 + 2e-6 * 1000)
    assert cm.seconds_per_task(1000, 500, 100) > cm.seconds_per_task(1000, 500, 0)
    # bootstrap memory is chunked: capped at bootstrap_chunk samples
    assert cm.memory_mb_per_task(1000, 500, 10**6) == cm.memory_mb_per_task(1000, 500, 256)
    assert cm.memory_mb_per_task(1000, 500, 0) == pytest.approx(160 + 256 * 1000 / 1e6)


def test_strategy_unregistered(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(strategy={"name": "nope", "version": "1"}), QuotaPolicy())
    assert codes(d) == {"STRATEGY_NOT_ALLOWED"} and not d.allowed


def test_strategy_version_unregistered(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(strategy={"name": "pairs_zscore", "version": "9"}), QuotaPolicy())
    assert "STRATEGY_NOT_ALLOWED" in codes(d)


def test_strategy_not_in_allowlist(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(), QuotaPolicy(allowed_strategies=("other",)))
    assert codes(d) == {"STRATEGY_NOT_ALLOWED"}
    assert svc.evaluate(make_spec(), QuotaPolicy(allowed_strategies=("pairs_zscore",))).allowed


def test_dataset_unknown(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(dataset={"name": "missing", "sha256": SHA}), QuotaPolicy())
    assert codes(d) == {"DATASET_NOT_ALLOWED"}
    assert d.estimate is None


def test_dataset_not_in_allowlist(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(), QuotaPolicy(allowed_datasets=("other",)))
    assert codes(d) == {"DATASET_NOT_ALLOWED"}
    assert svc.evaluate(make_spec(), QuotaPolicy(allowed_datasets=("ds",))).allowed


def test_dataset_signature_mismatch(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(dataset={"name": "ds", "sha256": "b" * 64}), QuotaPolicy())
    assert codes(d) == {"DATASET_SIGNATURE_MISMATCH"}
    assert SHA in d.violations[0].message and "b" * 64 in d.violations[0].message


def test_grid_too_large(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(), QuotaPolicy(max_tasks_per_batch=1))
    assert codes(d) == {"GRID_TOO_LARGE"}
    assert d.task_count == 2


def test_runtime_budget_exceeds_policy(svc: PolicyService) -> None:
    d = svc.evaluate(
        make_spec(budget={"max_runtime_s": 100}), QuotaPolicy(max_runtime_s_per_task=99)
    )
    assert codes(d) == {"RUNTIME_BUDGET_EXCEEDS_POLICY"}


def test_memory_budget_exceeds_policy(svc: PolicyService) -> None:
    d = svc.evaluate(
        make_spec(budget={"max_memory_mb": 1024}), QuotaPolicy(max_memory_mb_per_task=512)
    )
    assert codes(d) == {"MEMORY_BUDGET_EXCEEDS_POLICY"}


def test_memory_estimate_exceeds_budget(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(budget={"max_memory_mb": 64}), QuotaPolicy())
    assert codes(d) == {"MEMORY_ESTIMATE_EXCEEDS_BUDGET"}
    assert d.estimate is not None and d.estimate.est_memory_mb_per_task > 64
    assert d.error_type() == "VALIDATION_ERROR"


def test_worker_time_quota(svc: PolicyService) -> None:
    d = svc.evaluate(
        make_spec(budget={"max_runtime_s": 100}), QuotaPolicy(max_worker_seconds_per_batch=199)
    )
    assert codes(d) == {"WORKER_TIME_QUOTA_EXCEEDED"}
    assert svc.evaluate(
        make_spec(budget={"max_runtime_s": 100}), QuotaPolicy(max_worker_seconds_per_batch=200)
    ).allowed


def test_max_attempts_exceeds_policy(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(max_attempts=5), QuotaPolicy(max_attempts=4))
    assert codes(d) == {"MAX_ATTEMPTS_EXCEEDS_POLICY"}


def test_workload_exceeds_policy(svc: PolicyService) -> None:
    d = svc.evaluate(
        make_spec(workload={"bootstrap_samples": 101}), QuotaPolicy(max_bootstrap_samples=100)
    )
    assert codes(d) == {"WORKLOAD_EXCEEDS_POLICY"}
    assert svc.evaluate(
        make_spec(workload={"bootstrap_samples": 100}), QuotaPolicy(max_bootstrap_samples=100)
    ).allowed


def test_fault_injection_disabled(svc: PolicyService) -> None:
    spec = make_spec(faults={"fail_on_attempts": [1]})
    assert codes(svc.evaluate(spec, QuotaPolicy())) == {"FAULT_INJECTION_DISABLED"}
    assert svc.evaluate(spec, QuotaPolicy(allow_fault_injection=True)).allowed


def test_active_batch_limit(svc: PolicyService) -> None:
    q = QuotaPolicy(max_active_batches=3)
    assert svc.evaluate(make_spec(), q, active_batches=2).allowed
    d = svc.evaluate(make_spec(), q, active_batches=3)
    assert codes(d) == {"ACTIVE_BATCH_LIMIT"}


def test_protocol_train_too_short(svc: PolicyService) -> None:
    small = PolicyService(FakeCatalog(make_ds(n_bars=150)))
    d = small.evaluate(
        make_spec(split={"train_fraction": 0.4}, grid={"lookback": [5]}), QuotaPolicy()
    )
    assert "PROTOCOL_VIOLATION" in codes(d)
    assert any(f"< {MIN_TRAIN_BARS}" in v.message for v in d.violations)


def test_protocol_test_too_short() -> None:
    small = PolicyService(FakeCatalog(make_ds(n_bars=150)))
    d = small.evaluate(
        make_spec(split={"train_fraction": 0.9}, grid={"lookback": [5]}), QuotaPolicy()
    )
    assert any(f"< {MIN_TEST_BARS}" in v.message for v in d.violations)


def test_protocol_embargo_eats_test() -> None:
    d = PolicyService(FakeCatalog(make_ds())).evaluate(
        make_spec(split={"train_fraction": 0.6, "embargo_bars": 3950}), QuotaPolicy()
    )
    assert "PROTOCOL_VIOLATION" in codes(d)
    assert d.estimate is not None and d.estimate.n_test_bars_est == 50


def test_protocol_test_shorter_than_lookback_multiple(svc: PolicyService) -> None:
    # test segment = 4000 bars; lookback 2001 needs 4002
    d = svc.evaluate(make_spec(grid={"lookback": [20, 2001]}), QuotaPolicy())
    assert codes(d) == {"PROTOCOL_VIOLATION"}
    assert any(f"{TEST_BARS_PER_LOOKBACK}x" in v.message for v in d.violations)
    assert svc.evaluate(make_spec(grid={"lookback": [20, 2000]}), QuotaPolicy()).allowed


def test_protocol_lookback_from_fixed_params(svc: PolicyService) -> None:
    spec = make_spec(grid={"entry_z": [2.0]}, fixed_params={"lookback": 4000, "exit_z": 0.5})
    assert "PROTOCOL_VIOLATION" in codes(svc.evaluate(spec, QuotaPolicy()))


def test_protocol_timestamps_outside_range(svc: PolicyService) -> None:
    before = (START - timedelta(days=1)).isoformat()
    after = (END + timedelta(days=1)).isoformat()
    spec = make_spec(
        split={
            "mode": "timestamp",
            "train_fraction": None,
            "train_end": before,
            "test_start": after,
        }
    )
    d = svc.evaluate(spec, QuotaPolicy())
    msgs = [v.message for v in d.violations if v.code == "PROTOCOL_VIOLATION"]
    assert sum("outside dataset range" in m for m in msgs) == 2


def test_protocol_timestamp_split_ok(svc: PolicyService) -> None:
    te = START + timedelta(seconds=INTERVAL * 6000)
    ts = START + timedelta(seconds=INTERVAL * 6010)
    spec = make_spec(
        split={
            "mode": "timestamp",
            "train_fraction": None,
            "train_end": te.isoformat(),
            "test_start": ts.isoformat(),
        }
    )
    d = svc.evaluate(spec, QuotaPolicy())
    assert d.allowed, d.violations
    assert d.estimate is not None
    assert d.estimate.n_train_bars_est == 6000 and d.estimate.n_test_bars_est == N_BARS - 6010


def test_timestamp_split_with_embargo_uses_larger_start() -> None:
    ds = make_ds()
    te = START + timedelta(seconds=INTERVAL * 6000)
    spec = make_spec(
        split={
            "mode": "timestamp",
            "train_fraction": None,
            "train_end": te.isoformat(),
            "test_start": te.isoformat(),
            "embargo_bars": 100,
        }
    )
    n_train, n_test, problems = estimate_split(spec, ds)
    assert (n_train, n_test, problems) == (6000, N_BARS - 6100, [])


def test_invalid_params(svc: PolicyService) -> None:
    d = svc.evaluate(
        make_spec(grid={"exit_z": [0.5, 3.0]}, fixed_params={"lookback": 20, "entry_z": 2.0}),
        QuotaPolicy(),
    )
    assert codes(d) == {"INVALID_PARAMS"}
    assert "exit_z" in d.violations[0].message
    d = svc.evaluate(
        make_spec(fixed_params={"entry_z": 2.0, "exit_z": 0.5, "bogus": 1}), QuotaPolicy()
    )
    assert codes(d) == {"INVALID_PARAMS"}


def test_duplicate_grid_points_are_invalid_params(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(grid={"lookback": [20, 20]}), QuotaPolicy())
    assert codes(d) == {"INVALID_PARAMS"}


def test_multiple_violations_accumulate(svc: PolicyService) -> None:
    spec = make_spec(max_attempts=9, budget={"max_runtime_s": 9999}, faults={"sleep_s": 1.0})
    d = svc.evaluate(spec, QuotaPolicy(max_attempts=1, max_runtime_s_per_task=10))
    assert {
        "MAX_ATTEMPTS_EXCEEDS_POLICY",
        "RUNTIME_BUDGET_EXCEEDS_POLICY",
        "FAULT_INJECTION_DISABLED",
    } <= codes(d)


@pytest.mark.parametrize("code", sorted(QUOTA_VIOLATIONS))
def test_error_type_quota(code: str) -> None:
    d = PolicyDecision(
        allowed=False, task_count=1, violations=[Violation(code=code, field="f", message="m")]
    )  # type: ignore[arg-type]
    assert d.error_type() == "QUOTA_EXCEEDED"


@pytest.mark.parametrize(
    "code",
    ["STRATEGY_NOT_ALLOWED", "DATASET_NOT_ALLOWED", "DATASET_SIGNATURE_MISMATCH",
     "MEMORY_ESTIMATE_EXCEEDS_BUDGET", "FAULT_INJECTION_DISABLED", "PROTOCOL_VIOLATION", "INVALID_PARAMS"],
)  # fmt: skip
def test_error_type_validation(code: str) -> None:
    d = PolicyDecision(
        allowed=False, task_count=1, violations=[Violation(code=code, field="f", message="m")]
    )  # type: ignore[arg-type]
    assert d.error_type() == "VALIDATION_ERROR"


def test_error_type_mixed_and_empty() -> None:
    mixed = PolicyDecision(
        allowed=False,
        task_count=1,
        violations=[Violation(code="GRID_TOO_LARGE", field="g", message="m"),
                    Violation(code="PROTOCOL_VIOLATION", field="s", message="m")],
    )  # fmt: skip
    assert mixed.error_type() == "VALIDATION_ERROR"
    assert PolicyDecision(allowed=True, task_count=0).error_type() == "VALIDATION_ERROR"


def test_every_violation_code_is_classified() -> None:
    from typing import get_args

    from quantgrid.domain.policy import ViolationCode

    assert set(get_args(ViolationCode)) >= QUOTA_VIOLATIONS


def test_admit_returns_tasks_when_allowed(svc: PolicyService) -> None:
    spec = make_spec(seeds=[0, 1])
    decision, admitted = svc.admit(spec, QuotaPolicy())
    assert decision.allowed and admitted is not None
    assert [t.spec_hash for t in admitted.tasks] == [t.spec_hash for t in expand_grid(spec)]
    assert len(admitted.tasks) == decision.task_count == 4
    assert admitted.dataset.name == "ds" and admitted.spec is spec


def test_admit_none_when_rejected(svc: PolicyService) -> None:
    decision, admitted = svc.admit(make_spec(max_attempts=5), QuotaPolicy(max_attempts=1))
    assert not decision.allowed and admitted is None


def test_huge_grid_not_expanded(svc: PolicyService, monkeypatch: pytest.MonkeyPatch) -> None:
    import quantgrid.domain.policy as pol

    calls: list[int] = []
    real = pol.expand_grid

    def spy(spec: ExperimentBatchSpec, *, max_tasks: int = 0) -> Any:
        calls.append(max_tasks)
        return real(spec, max_tasks=max_tasks)

    monkeypatch.setattr(pol, "expand_grid", spy)
    spec = make_spec(
        grid={f"p{i}": list(range(256)) for i in range(8)}, fixed_params={}, seeds=list(range(64))
    )
    t0 = time.perf_counter()
    d, admitted = svc.admit(spec, QuotaPolicy())
    assert time.perf_counter() - t0 < 0.5
    assert admitted is None and "GRID_TOO_LARGE" in codes(d)
    assert calls == []
    assert d.task_count == 256**8 * 64
    assert d.error_type() in {"QUOTA_EXCEEDED", "VALIDATION_ERROR"}


def test_unregistered_strategy_skips_expansion(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(strategy={"name": "nope", "version": "1"}), QuotaPolicy())
    assert codes(d) == {"STRATEGY_NOT_ALLOWED"}  # no INVALID_PARAMS noise


def test_evaluate_has_no_side_effects(svc: PolicyService) -> None:
    spec = make_spec()
    a = svc.evaluate(spec, QuotaPolicy())
    b = svc.evaluate(spec, QuotaPolicy())
    assert a == b


def test_violation_messages() -> None:
    v = Violation(code="GRID_TOO_LARGE", field="grid", message="x")
    assert violation_messages([v]) == [{"code": "GRID_TOO_LARGE", "field": "grid", "message": "x"}]


def test_runtime_warning_when_estimate_exceeds_budget() -> None:
    svc = PolicyService(FakeCatalog(make_ds()))
    spec = make_spec(
        workload={"bootstrap_samples": 50_000}, budget={"max_runtime_s": 1, "max_memory_mb": 4096}
    )
    d = svc.evaluate(spec, QuotaPolicy(max_bootstrap_samples=50_000))
    assert any("may time out" in w for w in d.warnings)


def test_quota_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QG_QUOTA_MAX_TASKS_PER_BATCH", "7")
    monkeypatch.setenv("QG_QUOTA_ALLOWED_DATASETS", "a, b")
    monkeypatch.setenv("QG_ALLOW_FAULT_INJECTION", "true")
    q = QuotaPolicy.from_settings(Settings())
    assert q.max_tasks_per_batch == 7
    assert q.allowed_datasets == ("a", "b")
    assert q.allow_fault_injection is True
    assert q.allowed_strategies is None


def test_quota_is_frozen_and_forbids_extra() -> None:
    q = QuotaPolicy()
    with pytest.raises(Exception):  # noqa: B017, PT011
        q.max_attempts = 1  # type: ignore[misc]
    with pytest.raises(Exception):  # noqa: B017, PT011
        QuotaPolicy(bogus=1)  # type: ignore[call-arg]


def test_string_lookback_cannot_bypass_protocol_check(svc: PolicyService) -> None:
    d = svc.evaluate(make_spec(grid={"lookback": ["2500"]}), QuotaPolicy())
    assert "PROTOCOL_VIOLATION" in codes(d)
