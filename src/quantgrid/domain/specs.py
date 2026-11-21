"""Typed, immutable experiment contracts.

``ExperimentBatchSpec`` is what a client or the planner submits. ``TaskSpec`` is one child
task from deterministic grid expansion. Everything that can change a task's numerical result
lives in the TaskSpec and is covered by ``spec_hash``; execution limits (runtime and memory
budget, max attempts) do not change results and are stored as task columns. Models are frozen
and reject unknown fields, so unsupported fields fail validation instead of being ignored.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, ClassVar, Final, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from quantgrid.domain.hashing import canonical_sha256

RESULT_SCHEMA_VERSION: Final = 1
TASK_SPEC_SCHEMA_VERSION: Final = 1

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")]
ParamName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,47}$")]
Scalar = bool | int | float | Annotated[str, StringConstraints(max_length=64)]
#: A parameter value; ``None`` means "use the strategy default / feature disabled".
ParamValue = Scalar | None

#: Hard structural limits (independent of per-user quotas, which are usually tighter).
MAX_GRID_DIMENSIONS = 8
MAX_VALUES_PER_DIMENSION = 256
MAX_SEEDS = 64
ABSOLUTE_MAX_TASKS = 100_000


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DatasetRef(_Frozen):
    """Dataset name plus the content SHA-256 the client expects; admission rejects a mismatch."""

    name: Identifier
    sha256: Sha256Hex


class StrategyRef(_Frozen):
    name: Identifier
    version: Annotated[str, StringConstraints(pattern=r"^[0-9A-Za-z._-]{1,32}$")]


class SplitSpec(_Frozen):
    """Chronological train/test split.

    ``mode="fraction"``: the first ``floor(n * train_fraction)`` bars are train.
    ``mode="timestamp"``: bars with ``ts < train_end`` are train; test starts at the first
    bar with ``ts >= test_start``. Timestamps must be timezone-aware ISO-8601 and are
    normalised to UTC. ``embargo_bars`` additional bars after the train segment are
    skipped before the test segment begins (purging/embargo against leakage).
    """

    mode: Literal["fraction", "timestamp"] = "fraction"
    train_fraction: float | None = Field(default=0.6, gt=0.05, lt=0.95)
    train_end: AwareDatetime | None = None
    test_start: AwareDatetime | None = None
    embargo_bars: int = Field(default=0, ge=0, le=100_000)

    @field_validator("train_end", "test_start", mode="before")
    @classmethod
    def _reject_numeric_timestamps(cls, value: object) -> object:
        if value is not None and not isinstance(value, str | datetime):
            raise ValueError("timestamps must be ISO-8601 strings with a UTC offset")
        return value

    @field_validator("train_end", "test_start")
    @classmethod
    def _to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_mode(self) -> SplitSpec:
        if self.mode == "fraction":
            if self.train_fraction is None:
                raise ValueError("mode='fraction' requires train_fraction")
            if self.train_end is not None or self.test_start is not None:
                raise ValueError("mode='fraction' must not set train_end/test_start")
        else:
            if self.train_end is None or self.test_start is None:
                raise ValueError("mode='timestamp' requires train_end and test_start")
            if self.train_fraction is not None:
                raise ValueError("mode='timestamp' must not set train_fraction (use null)")
            if not self.train_end <= self.test_start:
                raise ValueError("train_end must be <= test_start (no train/test overlap)")
        return self


class FeeModel(_Frozen):
    commission_bps: float = Field(default=1.0, ge=0.0, le=100.0)
    slippage_bps: float = Field(default=0.5, ge=0.0, le=100.0)


class WorkloadSpec(_Frozen):
    """CPU-bound bootstrap work per task; 0 samples gives a short task."""

    bootstrap_samples: int = Field(default=0, ge=0, le=50_000)
    bootstrap_block: int = Field(default=20, ge=1, le=10_000)


class FaultInjection(_Frozen):
    """Failure injection for tests and demos (policy-gated).

    ``fail_on_attempts`` are 1-based attempt numbers, ``error_kind`` is ``transient`` or
    ``permanent``, ``sleep_s`` is extra cooperative wall time per attempt.
    """

    fail_on_attempts: tuple[Annotated[int, Field(ge=1, le=100)], ...] = ()
    error_kind: Literal["transient", "permanent"] = "transient"
    sleep_s: float = Field(default=0.0, ge=0.0, le=120.0)


class ResourceBudget(_Frozen):
    max_runtime_s: int = Field(default=120, ge=1, le=24 * 3600)
    max_memory_mb: int = Field(default=512, ge=64, le=65_536)


class PairsZScoreParams(_Frozen):
    lookback: int = Field(ge=5, le=5_000, description="rolling window (bars) for spread mean/std")
    entry_z: float = Field(gt=0.0, le=10.0, description="|z| that opens a position")
    exit_z: float = Field(ge=0.0, le=10.0, description="|z| at/below which a position closes")
    stop_z: float | None = Field(default=None, gt=0.0, le=50.0, description="|z| stop-loss")
    max_holding_bars: int | None = Field(default=None, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def _check_thresholds(self) -> PairsZScoreParams:
        if not self.exit_z < self.entry_z:
            raise ValueError("exit_z must be < entry_z")
        if self.stop_z is not None and not self.stop_z > self.entry_z:
            raise ValueError("stop_z must be > entry_z")
        return self


class GenericScalarParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _scalars_only(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            raise ValueError("params must be an object")
        for key, value in data.items():
            if not isinstance(key, str) or not key:
                raise ValueError("param names must be non-empty strings")
            if value is not None and not isinstance(value, bool | int | float | str):
                raise ValueError(f"param {key!r} must be a scalar or null")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"param {key!r} must be finite")
        return data


#: (strategy name, version) -> params model. Extended by adapters via register_strategy().
_STRATEGY_PARAMS: dict[tuple[str, str], type[BaseModel]] = {
    ("pairs_zscore", "1"): PairsZScoreParams,
}


def register_strategy(name: str, version: str, params_model: type[BaseModel]) -> None:
    _STRATEGY_PARAMS[(name, version)] = params_model


def strategy_params_model(ref: StrategyRef) -> type[BaseModel] | None:
    return _STRATEGY_PARAMS.get((ref.name, ref.version))


def registered_strategies() -> list[tuple[str, str]]:
    return sorted(_STRATEGY_PARAMS)


def strategy_param_names(ref: StrategyRef) -> list[str]:
    model = strategy_params_model(ref)
    return [] if model is None else sorted(model.model_fields)


class TaskSpec(_Frozen):
    schema_version: Literal[1] = TASK_SPEC_SCHEMA_VERSION
    dataset: DatasetRef
    strategy: StrategyRef
    params: dict[str, ParamValue]
    split: SplitSpec
    fees: FeeModel
    seed: int = Field(ge=0, le=2**63 - 1)
    workload: WorkloadSpec
    faults: FaultInjection | None = None
    result_schema_version: Literal[1] = RESULT_SCHEMA_VERSION

    def canonical(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def spec_hash(self) -> str:
        return canonical_sha256(self.canonical())


class ExpandedTask(_Frozen):
    ordinal: int
    spec: TaskSpec
    spec_hash: Sha256Hex


class ExperimentBatchSpec(_Frozen):
    MAX_NAME: ClassVar[int] = 120

    name: Annotated[
        str, StringConstraints(min_length=1, max_length=120, pattern=r"^[\w .:/()+-]+$")
    ]
    description: Annotated[str, StringConstraints(max_length=2000)] = ""
    dataset: DatasetRef
    strategy: StrategyRef
    grid: dict[ParamName, list[ParamValue]] = Field(min_length=1)
    fixed_params: dict[ParamName, ParamValue] = Field(default_factory=dict)
    seeds: list[Annotated[int, Field(ge=0, le=2**63 - 1)]] = Field(default=[0], min_length=1)
    split: SplitSpec = Field(default_factory=SplitSpec)
    fees: FeeModel = Field(default_factory=FeeModel)
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    budget: ResourceBudget = Field(default_factory=ResourceBudget)
    max_attempts: int = Field(default=3, ge=1, le=20)
    cache_policy: Literal["use", "bypass"] = "use"
    priority: int = Field(default=0, ge=-10, le=10)
    faults: FaultInjection | None = None
    tags: dict[
        Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_.-]{0,31}$")],
        Annotated[str, StringConstraints(max_length=128)],
    ] = Field(default_factory=dict, max_length=16)

    @field_validator("grid")
    @classmethod
    def _check_grid_shape(cls, grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if len(grid) > MAX_GRID_DIMENSIONS:
            raise ValueError(f"grid has {len(grid)} dimensions (max {MAX_GRID_DIMENSIONS})")
        for name, values in grid.items():
            if not values:
                raise ValueError(f"grid[{name!r}] is empty")
            if len(values) > MAX_VALUES_PER_DIMENSION:
                raise ValueError(
                    f"grid[{name!r}] has {len(values)} values (max {MAX_VALUES_PER_DIMENSION})"
                )
        return grid

    @field_validator("seeds")
    @classmethod
    def _check_seeds(cls, seeds: list[int]) -> list[int]:
        if len(seeds) > MAX_SEEDS:
            raise ValueError(f"too many seeds ({len(seeds)} > {MAX_SEEDS})")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        return seeds

    @model_validator(mode="after")
    def _check_params_disjoint(self) -> ExperimentBatchSpec:
        overlap = set(self.grid) & set(self.fixed_params)
        if overlap:
            raise ValueError(f"params both in grid and fixed_params: {sorted(overlap)}")
        return self

    def task_count(self) -> int:
        return math.prod(len(v) for v in self.grid.values()) * len(self.seeds)

    def canonical(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def spec_sha(self) -> str:
        return canonical_sha256(self.canonical())


class GridExpansionError(ValueError):
    def __init__(self, message: str, problems: Sequence[str]) -> None:
        super().__init__(message)
        self.problems = list(problems)


def _normalise_params(
    model: type[BaseModel], raw: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = model.model_validate(dict(raw))
    except ValidationError as exc:
        msgs = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'params'}: {err['msg']}"
            for err in exc.errors()
        )
        return None, msgs
    return parsed.model_dump(mode="json", exclude_none=False), None


def expand_grid(
    spec: ExperimentBatchSpec, *, max_tasks: int = ABSOLUTE_MAX_TASKS
) -> list[ExpandedTask]:
    """Deterministically expand a batch spec into immutable task specs.

    Order: parameter names sorted alphabetically, values in the given order, cartesian
    product, then seeds (innermost). ``ordinal`` is the 0-based position. Duplicate
    task specs (e.g. a grid listing the same value twice) are rejected because
    ``spec_hash`` must be unique per batch.

    Raises :class:`GridExpansionError` if the strategy is unknown, the grid is too large,
    a parameter is unknown, or any grid point fails the strategy's params schema.
    """
    count = spec.task_count()
    if count > max_tasks:
        raise GridExpansionError(
            f"grid expands to {count} tasks (max {max_tasks})", [f"task_count={count}"]
        )
    model = strategy_params_model(spec.strategy)
    if model is None:
        raise GridExpansionError(
            f"unknown strategy {spec.strategy.name}@{spec.strategy.version}",
            [f"registered: {registered_strategies()}"],
        )
    allowed = set(model.model_fields)
    if not issubclass(model, GenericScalarParams):
        unknown = sorted((set(spec.grid) | set(spec.fixed_params)) - allowed)
        if unknown:
            raise GridExpansionError(
                f"unknown parameters for {spec.strategy.name}: {unknown}",
                [f"allowed: {sorted(allowed)}"],
            )

    names = sorted(spec.grid)
    problems: list[str] = []
    tasks: list[ExpandedTask] = []
    seen: dict[str, int] = {}
    ordinal = 0
    for combo in itertools.product(*(spec.grid[n] for n in names)):
        raw = dict(spec.fixed_params)
        raw.update(zip(names, combo, strict=True))
        params, err = _normalise_params(model, raw)
        if err is not None or params is None:
            if len(problems) < 20:
                problems.append(f"{dict(zip(names, combo, strict=True))}: {err}")
            ordinal += len(spec.seeds)
            continue
        for seed in spec.seeds:
            task = TaskSpec(
                dataset=spec.dataset,
                strategy=spec.strategy,
                params=params,
                split=spec.split,
                fees=spec.fees,
                seed=seed,
                workload=spec.workload,
                faults=spec.faults,
            )
            digest = task.spec_hash()
            if digest in seen:
                problems.append(
                    f"duplicate task spec at ordinals {seen[digest]} and {ordinal} "
                    f"(params={params}, seed={seed})"
                )
            else:
                seen[digest] = ordinal
                tasks.append(ExpandedTask(ordinal=ordinal, spec=task, spec_hash=digest))
            ordinal += 1
    if problems:
        raise GridExpansionError(
            f"{len(problems)} invalid grid point(s) for {spec.strategy.name}", problems
        )
    return tasks


class DatasetInfo(_Frozen):
    name: Identifier
    sha256: Sha256Hex
    n_bars: int
    start: AwareDatetime
    end: AwareDatetime
    bar_interval_s: int
    columns: tuple[str, ...]
    synthetic: bool
    license: str
    description: str
    generator: dict[str, Any] = Field(default_factory=dict)
