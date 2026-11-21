"""Slow pure-Python reference implementation of ``pairs_zscore`` v1, for differential tests only.

Written independently of ``strategy`` and ``metrics`` from the same spec text, using explicit
loops, ``math.fsum`` for every sum and no vectorised numpy. The implementation choices differ
on purpose (linear-scan split boundaries, two-pass window statistics, a ``held`` counter) so
that agreement means something. O(n * lookback): production code must not call it.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from quantgrid.domain.errors import SpecError
from quantgrid.domain.policy import MIN_TEST_BARS, MIN_TRAIN_BARS, TEST_BARS_PER_LOOKBACK
from quantgrid.domain.specs import FeeModel, PairsZScoreParams, SplitSpec

NAN = float("nan")
_SD_FLOOR = 1e-12
_STD_FLOOR = 1e-15
_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass
class ReferenceTrade:
    entry: int  # absolute bar index
    exit: int  # absolute bar index
    side: int
    pnl: float


@dataclass
class ReferenceResult:
    """Everything the reference computes. Series cover the test segment ``[te_start, n)``."""

    tr_end: int
    te_start: int
    beta: float
    alpha: float
    cost_rate: float
    ts: list[int]
    spread: list[float]
    zscore: list[float]
    rolling_sd: list[float]  # NaN where the window is incomplete
    position: list[int]
    ret: list[float]
    equity: list[float]
    trades: list[ReferenceTrade]
    metrics: dict[str, float | int | None] = field(default_factory=dict)


def _to_ns(value: datetime) -> int:
    utc = value.astimezone(UTC)
    return calendar.timegm(utc.utctimetuple()) * 1_000_000_000 + utc.microsecond * 1_000


def _first_index_at_or_after(ts: Sequence[int], target: int) -> int:
    for i in range(len(ts)):
        if ts[i] >= target:
            return i
    return len(ts)


def reference_split(ts: Sequence[int], split: SplitSpec, lookback: int) -> tuple[int, int]:
    """``(tr_end, te_start)``; raises :class:`SpecError` on protocol violations."""
    n = len(ts)
    if split.mode == "fraction":
        if split.train_fraction is None:
            raise SpecError("fraction split without train_fraction")
        tr_end = math.floor(n * split.train_fraction)
        te_start = tr_end + split.embargo_bars
    else:
        if split.train_end is None or split.test_start is None:
            raise SpecError("timestamp split without boundaries")
        tr_end = _first_index_at_or_after(ts, _to_ns(split.train_end))
        te_start = _first_index_at_or_after(ts, _to_ns(split.test_start))
        if te_start < tr_end + split.embargo_bars:
            te_start = tr_end + split.embargo_bars
    n_test = n - te_start
    if tr_end < MIN_TRAIN_BARS:
        raise SpecError(f"reference: train segment too short ({tr_end})")
    if n_test < MIN_TEST_BARS:
        raise SpecError(f"reference: test segment too short ({n_test})")
    if n_test < TEST_BARS_PER_LOOKBACK * lookback:
        raise SpecError(f"reference: test segment shorter than 2x lookback ({n_test})")
    return tr_end, te_start


def reference_hedge_ratio(
    lx: Sequence[float], ly: Sequence[float], tr_end: int
) -> tuple[float, float]:
    """OLS with population moments on ``[0, tr_end)``: ``(beta, alpha)``."""
    constant = True
    for i in range(1, tr_end):
        if lx[i] != lx[0]:
            constant = False
            break
    if constant:
        raise SpecError("reference: var(lx) == 0")
    sx: list[float] = []
    sy: list[float] = []
    for i in range(tr_end):
        sx.append(lx[i])
        sy.append(ly[i])
    mean_x = math.fsum(sx) / tr_end
    mean_y = math.fsum(sy) / tr_end
    sq: list[float] = []
    cross: list[float] = []
    for i in range(tr_end):
        dx = sx[i] - mean_x
        sq.append(dx * dx)
        cross.append(dx * (sy[i] - mean_y))
    var_x = math.fsum(sq) / tr_end
    if var_x == 0.0:
        raise SpecError("reference: var(lx) == 0")
    beta = math.fsum(cross) / tr_end / var_x
    alpha = mean_y - beta * mean_x
    return beta, alpha


def reference_window_stats(s: Sequence[float], i: int, lookback: int) -> tuple[float, float]:
    """Two-pass mean and sample std of ``s[i-lookback+1 .. i]``."""
    window: list[float] = []
    for k in range(i - lookback + 1, i + 1):
        window.append(s[k])
    mean = math.fsum(window) / lookback
    sq: list[float] = []
    for v in window:
        sq.append((v - mean) * (v - mean))
    return mean, math.sqrt(math.fsum(sq) / (lookback - 1))


def reference_positions(z: Sequence[float], te_start: int, params: PairsZScoreParams) -> list[int]:
    """Absolute-index positions ``p[0..n-1]`` (``z`` indexed by absolute bar)."""
    n = len(z)
    p = [0] * n
    held = 0
    for i in range(te_start, n - 1):
        prev = p[i - 1]
        zi = z[i]
        if math.isnan(zi):
            new = 0
        elif prev == 0:
            if zi > params.entry_z:
                new = -1
            elif zi < -params.entry_z:
                new = 1
            else:
                new = 0
        else:
            timed_out = params.max_holding_bars is not None and held >= params.max_holding_bars
            if prev == 1:
                stopped = params.stop_z is not None and zi <= -params.stop_z
                new = 0 if (zi >= -params.exit_z or stopped or timed_out) else 1
            else:
                stopped = params.stop_z is not None and zi >= params.stop_z
                new = 0 if (zi <= params.exit_z or stopped or timed_out) else -1
        if new != 0 and prev == 0:
            held = 1
        elif new != 0:
            held += 1
        else:
            held = 0
        p[i] = new
    p[n - 1] = 0
    return p


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    m = _mean(values)
    sq: list[float] = []
    for v in values:
        sq.append((v - m) * (v - m))
    return math.sqrt(math.fsum(sq) / (len(values) - 1))


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def reference_metrics(res: ReferenceResult) -> dict[str, float | int | None]:
    r = res.ret
    n = len(r)
    diffs: list[int] = []
    for k in range(1, len(res.ts)):
        diffs.append(res.ts[k] - res.ts[k - 1])
    diffs.sort()
    mid = len(diffs) // 2
    median_ns = float(diffs[mid]) if len(diffs) % 2 == 1 else (diffs[mid - 1] + diffs[mid]) / 2
    ppy = _SECONDS_PER_YEAR / (median_ns / 1e9)

    equity_last = res.equity[-1]
    ann_return: float | None = None
    if equity_last > 0.0:
        try:
            ann_return = _finite(equity_last ** (ppy / n) - 1.0)
        except OverflowError:
            ann_return = None

    sd = _sample_std(r) if n >= 2 else None
    mean_r = _mean(r)
    sharpe = None
    if sd is not None and sd > _STD_FLOOR:
        sharpe = _finite(mean_r / sd * math.sqrt(ppy))
    down_sq: list[float] = []
    for v in r:
        d = v if v < 0.0 else 0.0
        down_sq.append(d * d)
    downside = math.sqrt(math.fsum(down_sq) / n)
    sortino = _finite(mean_r / downside * math.sqrt(ppy)) if downside > _STD_FLOOR else None

    peak = 1.0
    mdd = 0.0
    for eq in res.equity:
        if eq > peak:
            peak = eq
        dd = 1.0 - eq / peak
        if dd > mdd:
            mdd = dd

    turnover = 0
    exposure_bars = 0
    fees: list[float] = []
    prev = 0
    for pos in res.position:
        change = abs(pos - prev)
        turnover += change
        fees.append(res.cost_rate * change)
        if prev != 0:
            exposure_bars += 1
        prev = pos

    n_trades = len(res.trades)
    wins = 0
    hold_total = 0
    for trade in res.trades:
        if trade.pnl > 0.0:
            wins += 1
        hold_total += trade.exit - trade.entry

    return {
        "total_return": _finite(equity_last - 1.0),
        "ann_return": ann_return,
        "ann_vol": None if sd is None else _finite(sd * math.sqrt(ppy)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "turnover": turnover,
        "n_trades": n_trades,
        "exposure": exposure_bars / n,
        "fees_paid": math.fsum(fees),
        "hit_rate": wins / n_trades if n_trades else None,
        "avg_holding_bars": hold_total / n_trades if n_trades else None,
        "hedge_ratio": res.beta,
        "alpha": res.alpha,
        "n_train_bars": res.tr_end,
        "n_test_bars": n,
        "periods_per_year": ppy,
    }


def reference_run(
    ts: Sequence[int],
    close_x: Sequence[float],
    close_y: Sequence[float],
    params: PairsZScoreParams,
    split: SplitSpec,
    fees: FeeModel,
) -> ReferenceResult:
    ts_list = [int(t) for t in ts]
    n = len(ts_list)
    lookback = params.lookback
    tr_end, te_start = reference_split(ts_list, split, lookback)
    lx: list[float] = []
    ly: list[float] = []
    for i in range(n):
        lx.append(math.log(float(close_x[i])))
        ly.append(math.log(float(close_y[i])))
    beta, alpha = reference_hedge_ratio(lx, ly, tr_end)
    s: list[float] = []
    for i in range(n):
        s.append(ly[i] - beta * lx[i] - alpha)

    z = [NAN] * n
    sds = [NAN] * n
    for i in range(te_start, n):
        if i < lookback - 1:
            continue
        mean, sd = reference_window_stats(s, i, lookback)
        sds[i] = sd
        z[i] = (s[i] - mean) / sd if sd > _SD_FLOOR else 0.0

    p = reference_positions(z, te_start, params)
    cost_rate = (fees.commission_bps + fees.slippage_bps) / 1e4
    rets: list[float] = []
    equity: list[float] = []
    level = 1.0
    for i in range(te_start, n):
        g = (ly[i] - ly[i - 1]) - beta * (lx[i] - lx[i - 1])
        r = p[i - 1] * g / (1.0 + abs(beta)) - cost_rate * abs(p[i] - p[i - 1])
        rets.append(r)
        level = level * (1.0 + r)
        equity.append(level)

    trades: list[ReferenceTrade] = []
    entry = -1
    for i in range(te_start, n):
        if p[i - 1] == 0 and p[i] != 0:
            entry = i
        elif p[i - 1] != 0 and p[i] == 0:
            pnl_terms: list[float] = []
            for k in range(entry, i + 1):
                pnl_terms.append(rets[k - te_start])
            trades.append(
                ReferenceTrade(entry=entry, exit=i, side=p[entry], pnl=math.fsum(pnl_terms))
            )

    res = ReferenceResult(
        tr_end=tr_end,
        te_start=te_start,
        beta=beta,
        alpha=alpha,
        cost_rate=cost_rate,
        ts=ts_list[te_start:],
        spread=s[te_start:],
        zscore=z[te_start:],
        rolling_sd=sds[te_start:],
        position=p[te_start:],
        ret=rets,
        equity=equity,
        trades=trades,
    )
    res.metrics = reference_metrics(res)
    return res
