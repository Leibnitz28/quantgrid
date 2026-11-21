"""benchmark_math: speedup / efficiency / Karp-Flatt / percentile / summarize."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantgrid.domain.benchmark_math import (
    efficiency,
    karp_flatt_serial_fraction,
    percentile,
    scaling_row,
    speedup,
    summarize,
)


def test_speedup_and_efficiency_hand_values() -> None:
    assert speedup(100.0, 25.0) == 4.0
    assert efficiency(100.0, 25.0, 8) == 0.5
    assert efficiency(10.0, 10.0, 1) == 1.0


def test_karp_flatt_known_values() -> None:
    assert karp_flatt_serial_fraction(4.0, 4) == pytest.approx(0.0)  # perfect scaling
    assert karp_flatt_serial_fraction(1.0, 4) == pytest.approx(1.0)  # no speedup
    # Amdahl with serial fraction f=0.1, p=8: S = 1/(f + (1-f)/p)
    f, p = 0.1, 8
    s = 1.0 / (f + (1 - f) / p)
    assert karp_flatt_serial_fraction(s, p) == pytest.approx(f)
    assert karp_flatt_serial_fraction(10.0, 8) < 0  # superlinear is reported, not hidden


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_rejects_bad_times(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        speedup(bad, 1.0)
    with pytest.raises(ValueError, match=r"\w"):
        speedup(1.0, bad)


@pytest.mark.parametrize("p", [0, -2, True, 1.5])
def test_rejects_bad_worker_counts(p: object) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        efficiency(1.0, 1.0, p)  # type: ignore[arg-type]


def test_karp_flatt_undefined_for_one_worker() -> None:
    with pytest.raises(ValueError, match=r"\w"):
        karp_flatt_serial_fraction(1.0, 1)


def test_scaling_row() -> None:
    row = scaling_row(80.0, 20.0, 4)
    assert (
        row["speedup"] == 4.0
        and row["efficiency"] == 1.0
        and row["karp_flatt"] == pytest.approx(0.0)
    )
    assert scaling_row(5.0, 5.0, 1)["karp_flatt"] is None


def test_percentile_matches_numpy_exactly() -> None:
    rng = np.random.default_rng(5)
    for n in (1, 2, 3, 10, 101):
        vals = rng.normal(size=n).tolist()
        for q in (0, 1, 25, 50, 95, 99, 99.9, 100):
            assert percentile(vals, q) == float(np.percentile(vals, q, method="linear"))


def test_percentile_small_hand_values() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([4, 1, 3, 2], 0) == 1.0  # unsorted input
    assert percentile([4, 1, 3, 2], 100) == 4.0
    assert percentile([7], 99) == 7.0


@pytest.mark.parametrize("q", [-0.1, 100.1, math.nan])
def test_percentile_rejects_bad_q(q: float) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        percentile([1.0], q)


@pytest.mark.parametrize("vals", [[], [1.0, math.nan], [math.inf]])
def test_rejects_empty_or_nonfinite(vals: list[float]) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        percentile(vals, 50)
    with pytest.raises(ValueError, match=r"\w"):
        summarize(vals)


def test_summarize() -> None:
    s = summarize(range(1, 101))
    assert s["n"] == 100 and s["mean"] == 50.5 and s["min"] == 1.0 and s["max"] == 100.0
    assert s["p50"] == 50.5
    assert s["p95"] == pytest.approx(95.05) and s["p99"] == pytest.approx(99.01)
    one = summarize([3.0])
    assert one["p50"] == one["p95"] == one["p99"] == one["mean"] == 3.0
