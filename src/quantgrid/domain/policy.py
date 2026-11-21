"""Quota and policy service: deterministic admission control for experiment batches.

Every batch - whether hand-written or proposed by the AI planner - must pass
:meth:`PolicyService.evaluate` before anything is persisted or enqueued. The planner can
*call* this check through a read-only tool, but cannot bypass it: the API runs it again
on submission. Nothing here depends on an LLM.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from quantgrid.config import Settings
from quantgrid.domain.specs import (
    DatasetInfo,
    ExpandedTask,
    ExperimentBatchSpec,
    GridExpansionError,
    expand_grid,
    registered_strategies,
)

ViolationCode = Literal[
    "STRATEGY_NOT_ALLOWED",
    "DATASET_NOT_ALLOWED",
    "DATASET_SIGNATURE_MISMATCH",
    "GRID_TOO_LARGE",
    "RUNTIME_BUDGET_EXCEEDS_POLICY",
    "MEMORY_BUDGET_EXCEEDS_POLICY",
    "MEMORY_ESTIMATE_EXCEEDS_BUDGET",
    "WORKER_TIME_QUOTA_EXCEEDED",
    "MAX_ATTEMPTS_EXCEEDS_POLICY",
    "WORKLOAD_EXCEEDS_POLICY",
    "FAULT_INJECTION_DISABLED",
    "ACTIVE_BATCH_LIMIT",
    "PROTOCOL_VIOLATION",
    "INVALID_PARAMS",
]

#: Violations that represent resource/quota limits (API type QUOTA_EXCEEDED) as opposed
#: to malformed or scientifically invalid requests (API type VALIDATION_ERROR).
QUOTA_VIOLATIONS: frozenset[str] = frozenset(
    {
        "GRID_TOO_LARGE",
        "RUNTIME_BUDGET_EXCEEDS_POLICY",
        "MEMORY_BUDGET_EXCEEDS_POLICY",
        "WORKER_TIME_QUOTA_EXCEEDED",
        "MAX_ATTEMPTS_EXCEEDS_POLICY",
        "WORKLOAD_EXCEEDS_POLICY",
        "ACTIVE_BATCH_LIMIT",
    }
)

#: Minimum segment sizes for the research protocol.
MIN_TRAIN_BARS = 100
MIN_TEST_BARS = 100
TEST_BARS_PER_LOOKBACK = 2  # test segment must hold at least 2x the longest lookback


class DatasetCatalog(Protocol):
    """Read-only registry of datasets the platform may run on."""

    def get(self, name: str) -> DatasetInfo | None: ...

    def list(self) -> Sequence[DatasetInfo]: ...


class QuotaPolicy(BaseModel):
    """Per-user limits. A snapshot is stored on every batch row (``batches.quota``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tasks_per_batch: int = 5_000
    max_runtime_s_per_task: int = 900
    max_memory_mb_per_task: int = 2_048
    max_worker_seconds_per_batch: int = 12 * 3600
    max_active_batches: int = 20
    max_attempts: int = 10
    max_bootstrap_samples: int = 20_000
    allowed_datasets: tuple[str, ...] | None = None  # None = all registered datasets
    allowed_strategies: tuple[str, ...] | None = None  # None = all registered strategies
    allow_fault_injection: bool = False

    @classmethod
    def from_settings(cls, settings: Settings) -> QuotaPolicy:
        return cls(
            max_tasks_per_batch=settings.quota_max_tasks_per_batch,
            max_runtime_s_per_task=settings.quota_max_runtime_s_per_task,
            max_memory_mb_per_task=settings.quota_max_memory_mb_per_task,
            max_worker_seconds_per_batch=settings.quota_max_worker_seconds_per_batch,
            max_active_batches=settings.quota_max_active_batches,
            max_attempts=settings.quota_max_attempts,
            max_bootstrap_samples=settings.quota_max_bootstrap_samples,
            allowed_datasets=settings.allowed_datasets(),
            allow_fault_injection=settings.allow_fault_injection,
        )


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ViolationCode
    field: str
    message: str


@dataclass(frozen=True)
class CostModel:
    """Rough, documented per-task cost model used for admission and planning.

    Constants are conservative upper-bound style estimates for the built-in engine on a
    ~2 GHz x86 core; ``benchmarks/`` documents how they were checked. They only drive
    *estimates*; hard limits are the declared budgets.
    """

    overhead_s: float = 0.05
    seconds_per_bar: float = 2.0e-6
    seconds_per_bootstrap_bar: float = 2.0e-8
    base_memory_mb: float = 160.0
    bytes_per_bar: float = 8.0 * 32
    bootstrap_chunk: int = 256

    def seconds_per_task(self, n_bars: int, n_test_bars: int, bootstrap_samples: int) -> float:
        return (
            self.overhead_s
            + self.seconds_per_bar * n_bars
            + self.seconds_per_bootstrap_bar * n_test_bars * bootstrap_samples
        )

    def memory_mb_per_task(self, n_bars: int, n_test_bars: int, bootstrap_samples: int) -> float:
        chunk = min(self.bootstrap_chunk, bootstrap_samples)
        return (
            self.base_memory_mb
            + self.bytes_per_bar * n_bars / 1e6
            + 8.0 * 3 * n_test_bars * chunk / 1e6
        )


class CostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_count: int
    est_seconds_per_task: float
    est_total_worker_seconds: float
    budget_worker_seconds: int = Field(description="task_count * max_runtime_s (admission bound)")
    est_memory_mb_per_task: float
    n_train_bars_est: int
    n_test_bars_est: int


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    task_count: int
    violations: list[Violation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimate: CostEstimate | None = None

    def error_type(self) -> Literal["QUOTA_EXCEEDED", "VALIDATION_ERROR"]:
        """API error type for a rejected decision."""
        codes = {v.code for v in self.violations}
        return "QUOTA_EXCEEDED" if codes and codes <= QUOTA_VIOLATIONS else "VALIDATION_ERROR"


@dataclass(frozen=True)
class AdmittedBatch:
    """A spec that passed policy, with its deterministic task expansion."""

    spec: ExperimentBatchSpec
    tasks: list[ExpandedTask]
    decision: PolicyDecision
    dataset: DatasetInfo


def _max_lookback(spec: ExperimentBatchSpec) -> int:
    values: list[Any] = list(spec.grid.get("lookback", []))
    if "lookback" in spec.fixed_params:
        values.append(spec.fixed_params["lookback"])
    best = 0
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        try:  # same coercion pydantic applies when the grid is expanded ("2500" -> 2500)
            best = max(best, int(float(v)))
        except (TypeError, ValueError):
            continue
    return best


def estimate_split(spec: ExperimentBatchSpec, ds: DatasetInfo) -> tuple[int, int, list[str]]:
    """Approximate (n_train, n_test) bars and protocol problems for admission.

    Exact indices are resolved by the engine against the recorded timestamps; datasets
    with gaps can differ slightly, in which case the engine raises a non-retryable
    ``SPEC_ERROR`` at execution time.
    """
    problems: list[str] = []
    n = ds.n_bars
    split = spec.split
    if split.mode == "fraction":
        if split.train_fraction is None:  # guaranteed by SplitSpec validation
            raise ValueError("fraction split without train_fraction")
        n_train = math.floor(n * split.train_fraction)
        test_start = n_train + split.embargo_bars
    else:
        if split.train_end is None or split.test_start is None:  # guaranteed by SplitSpec
            raise ValueError("timestamp split without train_end/test_start")
        if not (ds.start <= split.train_end <= ds.end):
            problems.append(
                f"train_end {split.train_end.isoformat()} outside dataset range "
                f"[{ds.start.isoformat()}, {ds.end.isoformat()}]"
            )
        if not (ds.start <= split.test_start <= ds.end):
            problems.append(
                f"test_start {split.test_start.isoformat()} outside dataset range "
                f"[{ds.start.isoformat()}, {ds.end.isoformat()}]"
            )
        interval = max(ds.bar_interval_s, 1)
        n_train = max(0, int((split.train_end - ds.start).total_seconds() // interval))
        test_idx = max(0, math.ceil((split.test_start - ds.start).total_seconds() / interval))
        test_start = max(test_idx, n_train + split.embargo_bars)
    n_test = max(0, n - test_start)
    if n_train < MIN_TRAIN_BARS:
        problems.append(f"train segment has ~{n_train} bars (< {MIN_TRAIN_BARS})")
    if n_test < MIN_TEST_BARS:
        problems.append(f"test segment has ~{n_test} bars (< {MIN_TEST_BARS})")
    lookback = _max_lookback(spec)
    if lookback and n_test < TEST_BARS_PER_LOOKBACK * lookback:
        problems.append(
            f"test segment (~{n_test} bars) shorter than {TEST_BARS_PER_LOOKBACK}x the "
            f"longest lookback ({lookback})"
        )
    return n_train, n_test, problems


class PolicyService:
    def __init__(self, catalog: DatasetCatalog, cost_model: CostModel | None = None) -> None:
        self.catalog = catalog
        self.cost_model = cost_model or CostModel()

    def evaluate(
        self,
        spec: ExperimentBatchSpec,
        quota: QuotaPolicy,
        *,
        active_batches: int = 0,
    ) -> PolicyDecision:
        """Check ``spec`` against ``quota`` and the research protocol (no side effects)."""
        violations: list[Violation] = []
        warnings: list[str] = []
        task_count = spec.task_count()

        def add(code: ViolationCode, field: str, message: str) -> None:
            violations.append(Violation(code=code, field=field, message=message))

        strategy_key = (spec.strategy.name, spec.strategy.version)
        if strategy_key not in registered_strategies():
            add(
                "STRATEGY_NOT_ALLOWED",
                "strategy",
                f"strategy {spec.strategy.name}@{spec.strategy.version} is not registered",
            )
        elif (
            quota.allowed_strategies is not None
            and spec.strategy.name not in quota.allowed_strategies
        ):
            add("STRATEGY_NOT_ALLOWED", "strategy", f"strategy {spec.strategy.name} not allowed")

        ds = self.catalog.get(spec.dataset.name)
        if ds is None:
            add("DATASET_NOT_ALLOWED", "dataset.name", f"unknown dataset {spec.dataset.name!r}")
        elif quota.allowed_datasets is not None and ds.name not in quota.allowed_datasets:
            add("DATASET_NOT_ALLOWED", "dataset.name", f"dataset {ds.name!r} not allowed for user")
        elif ds.sha256 != spec.dataset.sha256:
            add(
                "DATASET_SIGNATURE_MISMATCH",
                "dataset.sha256",
                f"dataset {ds.name!r} has signature {ds.sha256}, request expected "
                f"{spec.dataset.sha256}",
            )

        if task_count > quota.max_tasks_per_batch:
            add(
                "GRID_TOO_LARGE",
                "grid",
                f"grid expands to {task_count} tasks (quota {quota.max_tasks_per_batch})",
            )
        if spec.budget.max_runtime_s > quota.max_runtime_s_per_task:
            add(
                "RUNTIME_BUDGET_EXCEEDS_POLICY",
                "budget.max_runtime_s",
                f"{spec.budget.max_runtime_s}s > policy {quota.max_runtime_s_per_task}s",
            )
        if spec.budget.max_memory_mb > quota.max_memory_mb_per_task:
            add(
                "MEMORY_BUDGET_EXCEEDS_POLICY",
                "budget.max_memory_mb",
                f"{spec.budget.max_memory_mb}MB > policy {quota.max_memory_mb_per_task}MB",
            )
        if spec.max_attempts > quota.max_attempts:
            add(
                "MAX_ATTEMPTS_EXCEEDS_POLICY",
                "max_attempts",
                f"{spec.max_attempts} > policy {quota.max_attempts}",
            )
        if spec.workload.bootstrap_samples > quota.max_bootstrap_samples:
            add(
                "WORKLOAD_EXCEEDS_POLICY",
                "workload.bootstrap_samples",
                f"{spec.workload.bootstrap_samples} > policy {quota.max_bootstrap_samples}",
            )
        budget_worker_seconds = task_count * spec.budget.max_runtime_s
        if budget_worker_seconds > quota.max_worker_seconds_per_batch:
            add(
                "WORKER_TIME_QUOTA_EXCEEDED",
                "budget.max_runtime_s",
                f"task_count x max_runtime_s = {budget_worker_seconds}s > quota "
                f"{quota.max_worker_seconds_per_batch}s",
            )
        if spec.faults is not None and not quota.allow_fault_injection:
            add("FAULT_INJECTION_DISABLED", "faults", "fault injection is disabled by policy")
        if active_batches >= quota.max_active_batches:
            add(
                "ACTIVE_BATCH_LIMIT",
                "batch",
                f"user already has {active_batches} active batches "
                f"(quota {quota.max_active_batches})",
            )

        estimate: CostEstimate | None = None
        if ds is not None:
            n_train, n_test, problems = estimate_split(spec, ds)
            for problem in problems:
                add("PROTOCOL_VIOLATION", "split", problem)
            est_s = self.cost_model.seconds_per_task(
                ds.n_bars, n_test, spec.workload.bootstrap_samples
            )
            est_mb = self.cost_model.memory_mb_per_task(
                ds.n_bars, n_test, spec.workload.bootstrap_samples
            )
            estimate = CostEstimate(
                task_count=task_count,
                est_seconds_per_task=round(est_s, 6),
                est_total_worker_seconds=round(est_s * task_count, 3),
                budget_worker_seconds=budget_worker_seconds,
                est_memory_mb_per_task=round(est_mb, 1),
                n_train_bars_est=n_train,
                n_test_bars_est=n_test,
            )
            if est_mb > spec.budget.max_memory_mb:
                add(
                    "MEMORY_ESTIMATE_EXCEEDS_BUDGET",
                    "budget.max_memory_mb",
                    f"estimated {est_mb:.0f}MB per task > budget {spec.budget.max_memory_mb}MB",
                )
            if est_s > spec.budget.max_runtime_s:
                warnings.append(
                    f"estimated {est_s:.1f}s per task exceeds max_runtime_s="
                    f"{spec.budget.max_runtime_s}; tasks may time out"
                )

        # Only expand when the grid is small enough to do so safely.
        if task_count <= quota.max_tasks_per_batch and not any(
            v.code == "STRATEGY_NOT_ALLOWED" for v in violations
        ):
            try:
                expand_grid(spec, max_tasks=quota.max_tasks_per_batch)
            except GridExpansionError as exc:
                add("INVALID_PARAMS", "grid", f"{exc}: {exc.problems[:5]}")

        return PolicyDecision(
            allowed=not violations,
            task_count=task_count,
            violations=violations,
            warnings=warnings,
            estimate=estimate,
        )

    def admit(
        self,
        spec: ExperimentBatchSpec,
        quota: QuotaPolicy,
        *,
        active_batches: int = 0,
    ) -> tuple[PolicyDecision, AdmittedBatch | None]:
        """Evaluate and, if allowed, return the expanded immutable task specs."""
        decision = self.evaluate(spec, quota, active_batches=active_batches)
        if not decision.allowed:
            return decision, None
        ds = self.catalog.get(spec.dataset.name)
        if ds is None:  # evaluate() already rejected unknown datasets
            raise ValueError(f"dataset {spec.dataset.name!r} vanished from the catalog")
        tasks = expand_grid(spec, max_tasks=quota.max_tasks_per_batch)
        return decision, AdmittedBatch(spec=spec, tasks=tasks, decision=decision, dataset=ds)


def violation_messages(violations: Iterable[Violation]) -> list[dict[str, str]]:
    return [v.model_dump() for v in violations]
