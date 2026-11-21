"""Metrics: hand-computed values, undefined cases (``None``), trades and invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hst

from quantgrid.domain.hashing import canonical_json
from quantgrid.domain.metrics import (
    CORE_METRIC_KEYS,
    STD_FLOOR,
    Trade,
    annualized_return,
    compute_metrics,
    extract_trades,
    max_drawdown,
    performance_metrics,
    periods_per_year,
    sharpe_ratio,
    sortino_ratio,
)
from quantgrid.domain.strategy import SplitIndices, StrategyResult

DAY_NS = 86_400 * 1_000_000_000


def daily(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) * DAY_NS


def metrics_for(
    position: list[int], ret: list[float], *, step_ns: int = DAY_NS, cost: float = 0.0005
):
    r = np.array(ret, dtype=np.float64)
    return performance_metrics(
        np.arange(len(ret), dtype=np.int64) * step_ns,
        np.array(position, dtype=np.int64),
        r,
        np.cumprod(1.0 + r),
        cost_rate=cost,
        hedge_ratio=1.25,
        alpha=-0.5,
        n_train_bars=321,
    )


class TestHandComputedExample:
    POS = [0, 1, 1, 0, -1, -1, 0]
    RET = [-0.001, 0.010, -0.004, 0.006, -0.0015, -0.003, 0.0005]

    def test_every_metric(self) -> None:
        m = metrics_for(self.POS, self.RET)
        assert tuple(m) == CORE_METRIC_KEYS
        r = self.RET
        n = len(r)
        ppy = 365.25
        mean = sum(r) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in r) / (n - 1))
        downside = math.sqrt(sum(min(v, 0.0) ** 2 for v in r) / n)
        equity = list(np.cumprod(1.0 + np.array(r)))
        peak, mdd = 1.0, 0.0
        for e in equity:
            peak = max(peak, e)
            mdd = max(mdd, 1.0 - e / peak)
        approx = {"rel": 1e-12, "abs": 1e-15}
        assert m["periods_per_year"] == ppy
        assert m["total_return"] == pytest.approx(math.prod(1 + v for v in r) - 1, **approx)
        assert m["ann_return"] == pytest.approx(equity[-1] ** (ppy / n) - 1, **approx)
        assert m["ann_vol"] == pytest.approx(sd * math.sqrt(ppy), **approx)
        assert m["sharpe"] == pytest.approx(mean / sd * math.sqrt(ppy), **approx)
        assert m["sortino"] == pytest.approx(mean / downside * math.sqrt(ppy), **approx)
        assert m["max_drawdown"] == pytest.approx(mdd, **approx)
        assert m["turnover"] == 4
        assert m["n_trades"] == 2
        assert m["exposure"] == 4 / 7
        assert m["fees_paid"] == pytest.approx(4 * 0.0005, **approx)
        assert m["hit_rate"] == 0.5  # +0.012 and -0.004
        assert m["avg_holding_bars"] == 2.0
        assert (m["hedge_ratio"], m["alpha"], m["n_train_bars"], m["n_test_bars"]) == (
            1.25,
            -0.5,
            321,
            7,
        )

    def test_trades(self) -> None:
        trades = extract_trades(np.array(self.POS), np.array(self.RET))
        assert trades == [
            Trade(entry_index=1, exit_index=3, side=1, pnl=math.fsum(self.RET[1:4])),
            Trade(entry_index=4, exit_index=6, side=-1, pnl=math.fsum(self.RET[4:7])),
        ]
        assert [t.holding_bars for t in trades] == [2, 2]

    def test_compute_metrics_wraps_a_strategy_result(self) -> None:
        r = np.array(self.RET)
        result = StrategyResult(
            split=SplitIndices(n_bars=400, train_end=300, test_start=393),
            lookback=20,
            beta=1.25,
            alpha=-0.5,
            cost_rate=0.0005,
            ts=daily(7),
            spread=np.zeros(7),
            zscore=np.zeros(7),
            position=np.array(self.POS, dtype=np.int64),
            ret=r,
            equity=np.cumprod(1.0 + r),
            recomputed_windows=0,
        )
        m = compute_metrics(result)
        assert m == {**metrics_for(self.POS, self.RET), "n_train_bars": 300}


class TestUndefinedValues:
    def test_flat_zero_returns(self) -> None:
        m = metrics_for([0] * 10, [0.0] * 10)
        assert m["sharpe"] is None
        assert m["sortino"] is None
        assert m["hit_rate"] is None
        assert m["avg_holding_bars"] is None
        assert (m["ann_vol"], m["total_return"], m["ann_return"], m["max_drawdown"]) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert (m["n_trades"], m["turnover"], m["exposure"], m["fees_paid"]) == (0, 0, 0.0, 0.0)

    def test_no_downside_means_no_sortino(self) -> None:
        m = metrics_for([0, 1, 1, 0], [0.001, 0.002, 0.003, 0.001])
        assert m["sortino"] is None
        assert m["sharpe"] is not None and m["sharpe"] > 0

    def test_tiny_dispersion_is_treated_as_zero(self) -> None:
        r = np.full(50, 1e-3)
        r[7] += 1e-17
        assert float(np.std(r, ddof=1)) <= STD_FLOOR
        assert sharpe_ratio(r, 252.0) is None
        assert sortino_ratio(-r, 252.0) is not None

    def test_non_positive_equity(self) -> None:
        m = metrics_for([0, 1, 1, 0], [0.0, -1.5, 0.1, 0.0])
        assert m["ann_return"] is None
        assert m["total_return"] == pytest.approx(-1.55)
        assert m["max_drawdown"] == pytest.approx(1.55)  # > 1 only with non-positive equity

    def test_annualisation_overflow_is_none(self) -> None:
        m = metrics_for([0, 0], [0.05, 0.05], step_ns=1_000_000_000)
        assert m["periods_per_year"] == 365.25 * 86400
        assert m["ann_return"] is None
        assert annualized_return(1.1, 1e9, 1) is None
        assert annualized_return(0.5, 1e9, 1) == -1.0  # underflow to 0 -> -100%
        assert annualized_return(1.0, 10.0, 0) is None

    def test_single_bar_cannot_be_annualised(self) -> None:
        with pytest.raises(ValueError, match="two timestamps"):
            metrics_for([0], [0.0])

    def test_series_must_align(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            performance_metrics(
                daily(3),
                np.zeros(2, dtype=np.int64),
                np.zeros(3),
                np.ones(3),
                cost_rate=0.0,
                hedge_ratio=1.0,
                alpha=0.0,
                n_train_bars=1,
            )


class TestBuildingBlocks:
    def test_periods_per_year_uses_the_median_interval(self) -> None:
        minute = 60 * 1_000_000_000
        assert periods_per_year(np.arange(10, dtype=np.int64) * minute) == 525_960.0
        gappy = np.cumsum(np.array([0, 1, 1, 2, 1, 5, 1], dtype=np.int64)) * minute
        assert periods_per_year(gappy) == 525_960.0
        assert periods_per_year(np.array([0, 60, 180], dtype=np.int64) * 1_000_000_000) == (
            365.25 * 86400 / 90
        )
        with pytest.raises(ValueError, match="increasing"):
            periods_per_year(np.array([5, 5, 5], dtype=np.int64))

    def test_max_drawdown_running_max_starts_at_one(self) -> None:
        assert max_drawdown(np.array([0.99, 0.98, 0.985])) == pytest.approx(0.02)
        assert max_drawdown(np.array([1.1, 1.21, 0.968, 1.3])) == pytest.approx(0.2)
        assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0
        assert max_drawdown(np.array([])) == 0.0

    def test_extract_trades_rejects_invalid_paths(self) -> None:
        with pytest.raises(ValueError, match="reversal"):
            extract_trades(np.array([0, 1, -1, 0]), np.zeros(4))
        with pytest.raises(ValueError, match="open trade"):
            extract_trades(np.array([0, 1, 1]), np.zeros(3))
        with pytest.raises(ValueError, match="shape"):
            extract_trades(np.array([0, 1, 0]), np.zeros(4))


@hst.composite
def position_paths(draw: hst.DrawFn) -> tuple[list[int], list[float]]:
    raw = draw(hst.lists(hst.sampled_from([-1, 0, 0, 1]), min_size=2, max_size=200))
    pos: list[int] = []
    prev = 0
    for p in raw:
        cur = 0 if prev != 0 and p == -prev else p  # exits go flat first
        pos.append(cur)
        prev = cur
    pos[-1] = 0
    rets = draw(
        hst.lists(
            hst.floats(min_value=-0.2, max_value=0.2, allow_nan=False),
            min_size=len(pos),
            max_size=len(pos),
        )
    )
    return pos, rets


@settings(max_examples=150, deadline=None, derandomize=True, database=None)
@given(position_paths())
def test_metrics_are_always_finite_or_none(path: tuple[list[int], list[float]]) -> None:
    pos, rets = path
    m = metrics_for(pos, rets)
    assert tuple(m) == CORE_METRIC_KEYS
    for key, value in m.items():
        assert value is None or (isinstance(value, int | float) and math.isfinite(value)), key
    canonical_json(m)  # rejects NaN/inf
    n_trades = m["n_trades"]
    assert isinstance(n_trades, int)
    assert m["turnover"] == 2 * n_trades
    assert 0.0 <= float(m["exposure"] or 0.0) <= 1.0
    assert float(m["max_drawdown"] or 0.0) >= 0.0
    if n_trades:
        assert 0.0 <= float(m["hit_rate"] or 0.0) <= 1.0
        assert float(m["avg_holding_bars"] or 0.0) >= 1.0
