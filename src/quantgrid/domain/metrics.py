"""Performance metrics of one run, computed on the test segment only.

Every value is a finite float or int, or ``None`` when undefined, so metrics can be stored
as canonical JSON. Definitions are in docs/strategy.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from quantgrid.domain.datasets import SECONDS_PER_YEAR
from quantgrid.domain.strategy import StrategyResult
from quantgrid.domain.types import FloatArray, IntArray, MetricValue

#: Keys produced by :func:`compute_metrics`, in output order.
CORE_METRIC_KEYS: Final[tuple[str, ...]] = (
    "total_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "turnover",
    "n_trades",
    "exposure",
    "fees_paid",
    "hit_rate",
    "avg_holding_bars",
    "hedge_ratio",
    "alpha",
    "n_train_bars",
    "n_test_bars",
    "periods_per_year",
)
#: Dispersion at or below this is treated as zero (ratio undefined).
STD_FLOOR: Final = 1e-15


@dataclass(frozen=True)
class Trade:
    """A closed round trip. Indices are test-segment offsets: ``p`` becomes ``side`` at
    ``entry_index`` and is flat again at ``exit_index``."""

    entry_index: int
    exit_index: int
    side: int
    pnl: float

    @property
    def holding_bars(self) -> int:
        return self.exit_index - self.entry_index


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _previous_positions(position: IntArray) -> IntArray:
    prev = np.zeros_like(position)
    prev[1:] = position[:-1]
    return prev


def periods_per_year(ts: IntArray) -> float:
    """``365.25*86400 / (median(diff(ts))/1e9)`` (needs >= 2 strictly increasing stamps)."""
    if ts.shape[0] < 2:
        raise ValueError("periods_per_year needs at least two timestamps")
    median_ns = float(np.median(np.diff(ts)))
    if not median_ns > 0.0:
        raise ValueError("timestamps must be strictly increasing")
    return SECONDS_PER_YEAR / (median_ns / 1e9)


def sharpe_ratio(r: FloatArray, ppy: float) -> float | None:
    if r.shape[0] < 2:
        return None
    sd = float(np.std(r, ddof=1))
    if not sd > STD_FLOOR:
        return None
    return _finite_or_none(float(np.mean(r)) / sd * math.sqrt(ppy))


def sortino_ratio(r: FloatArray, ppy: float) -> float | None:
    if r.shape[0] == 0:
        return None
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    if not downside > STD_FLOOR:
        return None
    return _finite_or_none(float(np.mean(r)) / downside * math.sqrt(ppy))


def max_drawdown(equity: FloatArray) -> float:
    if equity.shape[0] == 0:
        return 0.0
    running_max = np.maximum.accumulate(np.maximum(equity, 1.0))
    return float(np.max(1.0 - equity / running_max))


def annualized_return(equity_last: float, ppy: float, n: int) -> float | None:
    """``equity_last**(ppy/n) - 1``, or ``None`` when undefined or not representable."""
    if n <= 0 or not equity_last > 0.0:
        return None
    try:
        value = equity_last ** (ppy / n) - 1.0
    except OverflowError:
        return None
    return _finite_or_none(value)


def extract_trades(position: IntArray, ret: FloatArray) -> list[Trade]:
    """Round trips of a v1 position path (no reversals; flat at the end).

    Raises :class:`ValueError` for a same-bar reversal or a trade left open.
    """
    if position.shape != ret.shape:
        raise ValueError("position and ret must have the same shape")
    prev = _previous_positions(position)
    if bool(np.any(np.abs(position - prev) > 1)):
        raise ValueError("position path contains a same-bar reversal")
    entries = np.flatnonzero((prev == 0) & (position != 0))
    exits = np.flatnonzero((prev != 0) & (position == 0))
    if entries.shape[0] != exits.shape[0]:
        raise ValueError("position path ends with an open trade")
    trades: list[Trade] = []
    for e, x in zip(entries.tolist(), exits.tolist(), strict=True):
        trades.append(
            Trade(
                entry_index=e,
                exit_index=x,
                side=int(position[e]),
                pnl=math.fsum(ret[e : x + 1].tolist()),
            )
        )
    return trades


def performance_metrics(
    ts: IntArray,
    position: IntArray,
    ret: FloatArray,
    equity: FloatArray,
    *,
    cost_rate: float,
    hedge_ratio: float,
    alpha: float,
    n_train_bars: int,
) -> dict[str, MetricValue]:
    n = int(ret.shape[0])
    if n == 0 or not (ts.shape == position.shape == ret.shape == equity.shape):
        raise ValueError("series must be non-empty and of equal length")
    ppy = periods_per_year(ts)
    prev = _previous_positions(position)
    abs_dp = np.abs(position - prev)
    trades = extract_trades(position, ret)
    equity_last = float(equity[-1])
    sd = float(np.std(ret, ddof=1)) if n >= 2 else None
    n_trades = len(trades)
    return {
        "total_return": _finite_or_none(equity_last - 1.0),
        "ann_return": annualized_return(equity_last, ppy, n),
        "ann_vol": None if sd is None else _finite_or_none(sd * math.sqrt(ppy)),
        "sharpe": sharpe_ratio(ret, ppy),
        "sortino": sortino_ratio(ret, ppy),
        "max_drawdown": _finite_or_none(max_drawdown(equity)),
        "turnover": int(abs_dp.sum()),
        "n_trades": n_trades,
        "exposure": int(np.count_nonzero(prev)) / n,
        "fees_paid": _finite_or_none(float(np.sum(cost_rate * abs_dp))),
        "hit_rate": (sum(1 for t in trades if t.pnl > 0.0) / n_trades) if n_trades else None,
        "avg_holding_bars": (sum(t.holding_bars for t in trades) / n_trades if n_trades else None),
        "hedge_ratio": _finite_or_none(hedge_ratio),
        "alpha": _finite_or_none(alpha),
        "n_train_bars": n_train_bars,
        "n_test_bars": n,
        "periods_per_year": _finite_or_none(ppy),
    }


def compute_metrics(result: StrategyResult) -> dict[str, MetricValue]:
    return performance_metrics(
        result.ts,
        result.position,
        result.ret,
        result.equity,
        cost_rate=result.cost_rate,
        hedge_ratio=result.beta,
        alpha=result.alpha,
        n_train_bars=result.split.n_train,
    )
