"""Pure helpers for benchmark reports (speedup, efficiency, Karp-Flatt, percentiles).

``percentile`` matches ``numpy.percentile(..., method="linear")`` bit for bit (same operation
order). A negative serial fraction means superlinear speedup, usually a measurement artefact
that is reported rather than hidden. Empty, non-finite or out-of-range input raises ValueError.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypedDict


class Summary(TypedDict):
    n: int
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float


class ScalingRow(TypedDict):
    workers: int
    t1_s: float
    tp_s: float
    speedup: float
    efficiency: float
    karp_flatt: float | None


def _positive_finite(name: str, value: float) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return v


def _workers(p: int) -> int:
    if isinstance(p, bool) or not isinstance(p, int) or p < 1:
        raise ValueError(f"worker count must be an integer >= 1, got {p!r}")
    return p


def speedup(t1: float, tp: float) -> float:
    return _positive_finite("t1", t1) / _positive_finite("tp", tp)


def efficiency(t1: float, tp: float, p: int) -> float:
    return speedup(t1, tp) / _workers(p)


def karp_flatt_serial_fraction(speedup_value: float, p: int) -> float:
    """Karp-Flatt metric ``e = (1/S - 1/p) / (1 - 1/p)``; defined only for ``p > 1``."""
    s = _positive_finite("speedup", speedup_value)
    workers = _workers(p)
    if workers == 1:
        raise ValueError("the Karp-Flatt metric is undefined for p = 1")
    return (1.0 / s - 1.0 / workers) / (1.0 - 1.0 / workers)


def scaling_row(t1: float, tp: float, p: int) -> ScalingRow:
    """Speedup, efficiency and Karp-Flatt fraction (``None`` for ``p = 1``) in one row."""
    s = speedup(t1, tp)
    workers = _workers(p)
    return {
        "workers": workers,
        "t1_s": float(t1),
        "tp_s": float(tp),
        "speedup": s,
        "efficiency": s / workers,
        "karp_flatt": karp_flatt_serial_fraction(s, workers) if workers > 1 else None,
    }


def _sorted_finite(values: Iterable[float]) -> list[float]:
    data = [float(v) for v in values]
    if not data:
        raise ValueError("values must not be empty")
    if not all(math.isfinite(v) for v in data):
        raise ValueError("values must be finite")
    data.sort()
    return data


def _percentile_sorted(data: list[float], q: float) -> float:
    quantile = q / 100.0
    h = (len(data) - 1) * quantile
    if h >= len(data) - 1:
        return data[-1]
    lo = math.floor(h)
    gamma = h - lo
    a = data[lo]
    b = data[lo + 1]
    diff = b - a
    # numpy's _lerp: interpolate from the nearer end for numerical symmetry.
    return b - diff * (1.0 - gamma) if gamma >= 0.5 else a + diff * gamma


def percentile(values: Iterable[float], q: float) -> float:
    qf = float(q)
    if not (math.isfinite(qf) and 0.0 <= qf <= 100.0):
        raise ValueError(f"q must be within [0, 100], got {q!r}")
    return _percentile_sorted(_sorted_finite(values), qf)


def summarize(values: Iterable[float]) -> Summary:
    """``n``, ``mean`` (``math.fsum``-exact sum), ``p50``/``p95``/``p99``, ``min``, ``max``."""
    data = _sorted_finite(values)
    return {
        "n": len(data),
        "mean": math.fsum(data) / len(data),
        "p50": _percentile_sorted(data, 50.0),
        "p95": _percentile_sorted(data, 95.0),
        "p99": _percentile_sorted(data, 99.0),
        "min": data[0],
        "max": data[-1],
    }
