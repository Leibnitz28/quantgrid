"""Deterministic task engine: ``run_task_spec(spec, bars) -> TaskOutput``.

Invariants:

* The output is a function of ``(spec, bars)`` alone. ``attempt`` only selects injected faults.
  Diagnostics hold no wall-clock values, so they can be part of output checksums.
* Checks run in this order: bars carry the signature the spec names (``DataIntegrityError``),
  fault injection, params re-validated (``SpecError``), dispatch on ``(strategy, version)``.
  The bars are re-hashed so an unverified ``Bars`` label is still rejected.
* Long loops call ``cancel.check()`` within about 50 ms of work.

External backtesters plug in with :func:`register_engine` after ``specs.register_strategy``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ValidationError

from quantgrid.domain.bootstrap import empty_bootstrap_metrics, stationary_bootstrap_sharpe
from quantgrid.domain.datasets import dataset_content_sha256, validate_bars
from quantgrid.domain.errors import (
    DataIntegrityError,
    PermanentTaskError,
    SpecError,
    TransientTaskError,
)
from quantgrid.domain.metrics import compute_metrics, periods_per_year
from quantgrid.domain.specs import (
    FaultInjection,
    PairsZScoreParams,
    TaskSpec,
    strategy_params_model,
)
from quantgrid.domain.strategy import STRATEGY_NAME, STRATEGY_VERSION, run_pairs_zscore
from quantgrid.domain.types import Bars, CancelToken, TaskOutput

ENGINE_VERSION: Final = "pairs_zscore/1"
#: Cooperative fault-injection sleeps in slices of at most this many seconds.
FAULT_SLEEP_SLICE_S: Final = 0.05

#: ``fn(spec, validated_params, bars, cancel) -> TaskOutput``; must be pure (see module doc).
EngineFn = Callable[[TaskSpec, BaseModel, Bars, CancelToken], TaskOutput]


@dataclass(frozen=True)
class EngineRegistration:
    name: str
    version: str
    fn: EngineFn

    @property
    def engine_version(self) -> str:
        return f"{self.name}/{self.version}"


_REGISTRY: dict[tuple[str, str], EngineRegistration] = {}
_REGISTRY_LOCK = threading.Lock()


def register_engine(name: str, version: str, fn: EngineFn, *, replace: bool = False) -> None:
    """Register the engine for strategy ``name``/``version``.

    Re-registering an existing key raises :class:`ValueError` unless ``replace`` is true,
    so an adapter cannot shadow the built-in engine by accident.
    """
    key = (name, version)
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and existing.fn is not fn and not replace:
            raise ValueError(f"an engine is already registered for {name}@{version}")
        _REGISTRY[key] = EngineRegistration(name=name, version=version, fn=fn)


def unregister_engine(name: str, version: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.pop((name, version), None)


def get_engine(name: str, version: str) -> EngineRegistration | None:
    with _REGISTRY_LOCK:
        return _REGISTRY.get((name, version))


def registered_engines() -> list[tuple[str, str]]:
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY)


def _inject_faults(faults: FaultInjection, attempt: int, cancel: CancelToken) -> None:
    """Cooperative sleep (``<= 50 ms`` slices, checking ``cancel``), then maybe fail."""
    if faults.sleep_s > 0.0:
        deadline = time.monotonic() + faults.sleep_s
        while True:
            cancel.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(FAULT_SLEEP_SLICE_S, remaining))
    if attempt in faults.fail_on_attempts:
        message = f"injected {faults.error_kind} fault on attempt {attempt}"
        details = {"injected": True, "attempt": attempt, "error_kind": faults.error_kind}
        if faults.error_kind == "permanent":
            raise PermanentTaskError(message, details=details)
        raise TransientTaskError(message, details=details)


def _parse_params(spec: TaskSpec) -> BaseModel:
    model = strategy_params_model(spec.strategy)
    if model is None:
        raise SpecError(f"unknown strategy {spec.strategy.name}@{spec.strategy.version}")
    try:
        return model.model_validate(spec.params)
    except ValidationError as exc:
        problems = [
            {"loc": ".".join(str(p) for p in err["loc"]) or "params", "msg": err["msg"]}
            for err in exc.errors(include_url=False)
        ]
        raise SpecError(
            f"invalid params for {spec.strategy.name}@{spec.strategy.version}: "
            + "; ".join(f"{p['loc']}: {p['msg']}" for p in problems),
            details={"errors": problems},
        ) from exc


def run_task_spec(
    spec: TaskSpec, bars: Bars, *, cancel: CancelToken, attempt: int = 1
) -> TaskOutput:
    if attempt < 1:
        raise ValueError("attempt numbers are 1-based")
    if bars.sha256 != spec.dataset.sha256:
        raise DataIntegrityError(
            f"bars {bars.name!r} have signature {bars.sha256}, task expects "
            f"{spec.dataset.name!r} with {spec.dataset.sha256}",
            details={"expected": spec.dataset.sha256, "actual": bars.sha256},
        )
    validate_bars(bars.ts, bars.close_x, bars.close_y)
    actual = dataset_content_sha256(bars.ts, bars.close_x, bars.close_y)
    if actual != bars.sha256:
        raise DataIntegrityError(
            f"bars {bars.name!r}: content hashes to {actual}, not the claimed {bars.sha256} "
            "(tampered or corrupted in memory)",
            details={"claimed": bars.sha256, "actual": actual},
        )
    if spec.faults is not None:
        _inject_faults(spec.faults, attempt, cancel)
    params = _parse_params(spec)
    registration = get_engine(spec.strategy.name, spec.strategy.version)
    if registration is None:
        raise SpecError(
            f"no engine registered for {spec.strategy.name}@{spec.strategy.version}",
            details={"registered": registered_engines()},
        )
    cancel.check()
    return registration.fn(spec, params, bars, cancel)


def run_pairs_zscore_task(
    spec: TaskSpec, params: BaseModel, bars: Bars, cancel: CancelToken
) -> TaskOutput:
    if not isinstance(params, PairsZScoreParams):
        raise SpecError(f"expected PairsZScoreParams, got {type(params).__name__}")
    result = run_pairs_zscore(bars, params, spec.split, spec.fees, cancel=cancel)
    metrics = compute_metrics(result)
    workload = spec.workload
    if workload.bootstrap_samples > 0:
        boot = stationary_bootstrap_sharpe(
            result.ret,
            samples=workload.bootstrap_samples,
            mean_block=workload.bootstrap_block,
            seed=spec.seed,
            periods_per_year=periods_per_year(result.ts),
            cancel=cancel,
        )
        metrics.update(boot.as_metrics())
    else:
        metrics.update(empty_bootstrap_metrics())
    split = result.split
    diagnostics: dict[str, float | int | str] = {
        "engine_version": ENGINE_VERSION,
        "strategy": f"{spec.strategy.name}@{spec.strategy.version}",
        "dataset": bars.name,
        "dataset_sha256": bars.sha256,
        "n_bars": split.n_bars,
        "train_end_index": split.train_end,
        "test_start_index": split.test_start,
        "test_start_ts_ns": int(result.ts[0]),
        "test_end_ts_ns": int(result.ts[-1]),
        "first_zscore_index": max(split.test_start, result.lookback - 1),
        "rolling_recomputed_windows": result.recomputed_windows,
        "bootstrap_block": workload.bootstrap_block,
    }
    return TaskOutput(metrics=metrics, series=result.series(), diagnostics=diagnostics)


register_engine(STRATEGY_NAME, STRATEGY_VERSION, run_pairs_zscore_task)
