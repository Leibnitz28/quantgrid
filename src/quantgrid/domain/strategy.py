"""Pairs z-score mean reversion, strategy ``pairs_zscore`` version ``"1"``.

The definition is in docs/strategy.md. ``reference.py`` implements the same text
independently; keep the two in step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from quantgrid.domain.errors import SpecError
from quantgrid.domain.policy import MIN_TEST_BARS, MIN_TRAIN_BARS, TEST_BARS_PER_LOOKBACK
from quantgrid.domain.specs import FeeModel, PairsZScoreParams, SplitSpec
from quantgrid.domain.types import Bars, CancelToken, FloatArray, IntArray, NullCancelToken

STRATEGY_NAME: Final = "pairs_zscore"
STRATEGY_VERSION: Final = "1"

#: ``z_i = 0`` when the rolling sample std is not above this floor.
SD_FLOOR: Final = 1e-12
#: Python loops over bars call ``cancel.check()`` at least this often (a few ms of work).
CANCEL_CHECK_EVERY: Final = 4096
#: Windows per block of the rolling-statistics kernel (fixed: results depend on it at the
#: ulp level, so it must never depend on runtime conditions).
ROLLING_BLOCK: Final = 1024
#: Recompute a window exactly when ``M2 <= ROLLING_RECOMPUTE_RATIO * S2``.
ROLLING_RECOMPUTE_RATIO: Final = 1e-3
#: Upper bound on elements materialised at once by the exact fallback.
_FALLBACK_MAX_ELEMENTS: Final = 1 << 20

_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_NULL_CANCEL: Final = NullCancelToken()


@dataclass(frozen=True)
class SplitIndices:
    """Resolved chronological split: train ``[0, train_end)``, test ``[test_start, n_bars)``.

    Invariant (enforced by :func:`resolve_split`): ``train_end >= MIN_TRAIN_BARS``,
    ``train_end <= test_start`` and ``n_test >= max(MIN_TEST_BARS, 2*lookback)``.
    """

    n_bars: int
    train_end: int
    test_start: int

    @property
    def n_train(self) -> int:
        return self.train_end

    @property
    def n_test(self) -> int:
        return self.n_bars - self.test_start


def datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _searchsorted_left(ts: IntArray, value: int) -> int:
    """``numpy.searchsorted(ts, value, "left")`` for arbitrary Python ints (no int64 overflow)."""
    n = int(ts.shape[0])
    if n == 0 or value <= int(ts[0]):
        return 0
    if value > int(ts[-1]):
        return n
    return int(np.searchsorted(ts, np.int64(value), side="left"))


def resolve_split(ts: IntArray, split: SplitSpec, lookback: int) -> SplitIndices:
    """Resolve ``split`` against the recorded timestamps (see module docstring).

    Raises :class:`SpecError` when the protocol minimums are not met.
    """
    n = int(ts.shape[0])
    if split.mode == "fraction":
        if split.train_fraction is None:
            raise SpecError("split mode 'fraction' requires train_fraction")
        train_end = math.floor(n * split.train_fraction)
        test_start = train_end + split.embargo_bars
    else:
        if split.train_end is None or split.test_start is None:
            raise SpecError("split mode 'timestamp' requires train_end and test_start")
        train_end = _searchsorted_left(ts, datetime_to_ns(split.train_end))
        test_start = max(
            _searchsorted_left(ts, datetime_to_ns(split.test_start)),
            train_end + split.embargo_bars,
        )
    n_test = n - test_start
    problems: list[str] = []
    if train_end < MIN_TRAIN_BARS:
        problems.append(f"train segment has {train_end} bars (< {MIN_TRAIN_BARS})")
    if n_test < MIN_TEST_BARS:
        problems.append(f"test segment has {n_test} bars (< {MIN_TEST_BARS})")
    if n_test < TEST_BARS_PER_LOOKBACK * lookback:
        problems.append(
            f"test segment has {n_test} bars (< {TEST_BARS_PER_LOOKBACK} x lookback {lookback})"
        )
    if problems:
        raise SpecError(
            "invalid train/test split: " + "; ".join(problems),
            details={
                "n_bars": n,
                "train_end": train_end,
                "test_start": test_start,
                "lookback": lookback,
            },
        )
    return SplitIndices(n_bars=n, train_end=train_end, test_start=test_start)


def log_prices(prices: FloatArray) -> FloatArray:
    n = int(prices.shape[0])
    return np.fromiter(map(math.log, prices.tolist()), dtype=np.float64, count=n)


def fit_hedge_ratio(lx: FloatArray, ly: FloatArray, train_end: int) -> tuple[float, float]:
    """OLS ``ly ~ alpha + beta*lx`` on ``[0, train_end)`` with population moments.

    Returns ``(beta, alpha)``. Raises :class:`SpecError` if ``lx`` is constant on the
    train segment (zero variance) or the fit is not finite.
    """
    if train_end < 2:
        raise SpecError(f"hedge-ratio fit needs >= 2 train bars, got {train_end}")
    x = lx[:train_end]
    y = ly[:train_end]
    if float(np.max(x)) == float(np.min(x)):
        raise SpecError("var(log close_x) is zero on the train segment; hedge ratio undefined")
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    dx = x - mean_x
    var_x = float(np.mean(dx * dx))
    cov_xy = float(np.mean(dx * (y - mean_y)))
    if not var_x > 0.0:
        raise SpecError("var(log close_x) is zero on the train segment; hedge ratio undefined")
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    if not (math.isfinite(beta) and math.isfinite(alpha)):
        raise SpecError(f"hedge-ratio fit is not finite (beta={beta}, alpha={alpha})")
    return beta, alpha


@dataclass(frozen=True)
class RollingStats:
    """Rolling mean / sample std (ddof=1) of windows ending at ``first .. n-1``."""

    first: int
    mean: FloatArray
    sd: FloatArray
    recomputed_windows: int


def rolling_mean_std(
    values: FloatArray, lookback: int, first: int, *, cancel: CancelToken = _NULL_CANCEL
) -> RollingStats:
    """Mean and sample std over ``values[i-lookback+1 .. i]`` for ``i`` in ``[first, n)``.

    Requires ``2 <= lookback`` and ``lookback - 1 <= first <= n``. See the module
    docstring for the algorithm and its accuracy guarantee.
    """
    n = int(values.shape[0])
    if lookback < 2:
        raise ValueError("lookback must be >= 2 for a sample standard deviation")
    if not lookback - 1 <= first <= n:
        raise ValueError(f"first={first} must be in [lookback-1, n] = [{lookback - 1}, {n}]")
    count = n - first
    mean = np.empty(count, dtype=np.float64)
    sd = np.empty(count, dtype=np.float64)
    recomputed = 0
    lag = lookback - 1
    for block_start in range(first, n, ROLLING_BLOCK):
        cancel.check()
        block_end = min(n, block_start + ROLLING_BLOCK)
        seg = values[block_start - lag : block_end]
        anchor = float(np.mean(seg))
        dev = seg - anchor
        c1 = np.zeros(seg.shape[0] + 1, dtype=np.float64)
        c2 = np.zeros(seg.shape[0] + 1, dtype=np.float64)
        np.cumsum(dev, out=c1[1:])
        np.cumsum(dev * dev, out=c2[1:])
        s1 = c1[lookback:] - c1[:-lookback]
        s2 = c2[lookback:] - c2[:-lookback]
        block_mean = anchor + s1 / lookback
        m2 = s2 - s1 * (s1 / lookback)
        bad = np.flatnonzero(~(m2 > ROLLING_RECOMPUTE_RATIO * s2))
        if bad.size:
            recomputed += int(bad.size)
            windows = sliding_window_view(seg, lookback)
            step = max(1, _FALLBACK_MAX_ELEMENTS // lookback)
            for k in range(0, int(bad.size), step):
                idx = bad[k : k + step]
                w = windows[idx]
                w_mean = w.mean(axis=1)
                w_dev = w - w_mean[:, None]
                block_mean[idx] = w_mean
                m2[idx] = (w_dev * w_dev).sum(axis=1)
        out = slice(block_start - first, block_end - first)
        mean[out] = block_mean
        sd[out] = np.sqrt(np.maximum(m2, 0.0) / (lookback - 1))
    return RollingStats(first=first, mean=mean, sd=sd, recomputed_windows=recomputed)


def segment_zscores(
    spread: FloatArray, lookback: int, test_start: int, *, cancel: CancelToken = _NULL_CANCEL
) -> tuple[FloatArray, int]:
    """Z-scores for bars ``test_start .. n-1`` (NaN where ``i < lookback-1``).

    Returns ``(z, recomputed_windows)``.
    """
    n = int(spread.shape[0])
    z = np.full(n - test_start, np.nan, dtype=np.float64)
    first = max(test_start, lookback - 1)
    if first >= n:
        return z, 0
    stats = rolling_mean_std(spread, lookback, first, cancel=cancel)
    num = spread[first:] - stats.mean
    ok = stats.sd > SD_FLOOR
    zz = np.zeros(n - first, dtype=np.float64)
    np.divide(num, stats.sd, out=zz, where=ok)
    z[first - test_start :] = zz
    return z, stats.recomputed_windows


def compute_positions(
    z: FloatArray, params: PairsZScoreParams, *, cancel: CancelToken = _NULL_CANCEL
) -> IntArray:
    """Run the position state machine over the test segment.

    ``z[k]`` is the z-score of test bar ``k`` (bar ``te_start + k``). Returns ``p`` with
    ``p[k]`` in ``{-1, 0, +1}``. The last element is forced to 0 and the position before
    ``k = 0`` is flat.
    """
    n_test = int(z.shape[0])
    positions = [0] * n_test
    entry = float(params.entry_z)
    exit_z = float(params.exit_z)
    stop = None if params.stop_z is None else float(params.stop_z)
    max_hold = params.max_holding_bars
    zs = z.tolist()
    prev = 0
    entered_at = 0
    for k in range(n_test - 1):
        if k % CANCEL_CHECK_EVERY == 0:
            cancel.check()
        zk = zs[k]
        if zk != zk:  # NaN: undefined z-score -> flat
            cur = 0
        elif prev == 0:
            if zk > entry:
                cur = -1
            elif zk < -entry:
                cur = 1
            else:
                cur = 0
            if cur != 0:
                entered_at = k
        elif prev == 1:
            if (
                zk >= -exit_z
                or (stop is not None and zk <= -stop)
                or (max_hold is not None and k - entered_at >= max_hold)
            ):
                cur = 0
            else:
                cur = 1
        else:
            if (
                zk <= exit_z
                or (stop is not None and zk >= stop)
                or (max_hold is not None and k - entered_at >= max_hold)
            ):
                cur = 0
            else:
                cur = -1
        positions[k] = cur
        prev = cur
    return np.array(positions, dtype=np.int64)


@dataclass(frozen=True)
class StrategyResult:
    """Deterministic output of one strategy run. Arrays cover the test segment only."""

    split: SplitIndices
    lookback: int
    beta: float
    alpha: float
    cost_rate: float
    ts: IntArray
    spread: FloatArray
    zscore: FloatArray
    position: IntArray
    ret: FloatArray
    equity: FloatArray
    recomputed_windows: int

    def series(self) -> dict[str, npt.NDArray[np.generic]]:
        return {
            "ts": self.ts,
            "spread": self.spread,
            "zscore": self.zscore,
            "position": self.position,
            "ret": self.ret,
            "equity": self.equity,
        }


def run_pairs_zscore(
    bars: Bars,
    params: PairsZScoreParams,
    split: SplitSpec,
    fees: FeeModel,
    *,
    cancel: CancelToken = _NULL_CANCEL,
) -> StrategyResult:
    split_idx = resolve_split(bars.ts, split, params.lookback)
    lx = log_prices(bars.close_x)
    ly = log_prices(bars.close_y)
    cancel.check()
    beta, alpha = fit_hedge_ratio(lx, ly, split_idx.train_end)
    spread = ly - beta * lx - alpha
    te = split_idx.test_start
    z, recomputed = segment_zscores(spread, params.lookback, te, cancel=cancel)
    position = compute_positions(z, params, cancel=cancel)
    g = (ly[te:] - ly[te - 1 : -1]) - beta * (lx[te:] - lx[te - 1 : -1])
    prev = np.zeros_like(position)
    prev[1:] = position[:-1]
    cost_rate = (fees.commission_bps + fees.slippage_bps) / 1e4
    ret = prev * g / (1.0 + abs(beta)) - cost_rate * np.abs(position - prev)
    equity = np.cumprod(1.0 + ret)
    return StrategyResult(
        split=split_idx,
        lookback=params.lookback,
        beta=beta,
        alpha=alpha,
        cost_rate=cost_rate,
        ts=bars.ts[te:].copy(),
        spread=spread[te:].copy(),
        zscore=z,
        position=position,
        ret=ret,
        equity=equity,
        recomputed_windows=recomputed,
    )
