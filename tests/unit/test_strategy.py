"""Strategy ``pairs_zscore`` v1: split, hedge ratio, rolling statistics, state machine, returns."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quantgrid.domain import strategy as st
from quantgrid.domain.datasets import (
    dataset_content_sha256,
    dataset_ref,
    generate_bars,
    get_catalog,
)
from quantgrid.domain.errors import SpecError, TaskAborted
from quantgrid.domain.policy import estimate_split
from quantgrid.domain.specs import (
    ExperimentBatchSpec,
    FeeModel,
    PairsZScoreParams,
    SplitSpec,
    StrategyRef,
)
from quantgrid.domain.strategy import (
    ROLLING_BLOCK,
    SD_FLOOR,
    compute_positions,
    datetime_to_ns,
    fit_hedge_ratio,
    resolve_split,
    rolling_mean_std,
    run_pairs_zscore,
    segment_zscores,
)
from quantgrid.domain.types import Bars, NullCancelToken

HOUR_NS = 3_600 * 1_000_000_000
T0 = datetime(2020, 1, 1, tzinfo=UTC)


class CountingToken:
    """Cancel token that counts checks and aborts on the ``abort_at``-th check."""

    def __init__(self, abort_at: int | None = None) -> None:
        self.checks = 0
        self.abort_at = abort_at

    def check(self) -> None:
        self.checks += 1
        if self.abort_at is not None and self.checks >= self.abort_at:
            raise TaskAborted("CANCELED")


def hourly_ts(n: int) -> np.ndarray:
    return datetime_to_ns(T0) + np.arange(n, dtype=np.int64) * HOUR_NS


def params(**kw: object) -> PairsZScoreParams:
    base: dict[str, object] = {"lookback": 20, "entry_z": 2.0, "exit_z": 0.5}
    base.update(kw)
    return PairsZScoreParams.model_validate(base)


def naive_rolling(values: np.ndarray, lookback: int, first: int) -> tuple[np.ndarray, np.ndarray]:
    means, sds = [], []
    for i in range(first, values.shape[0]):
        w = values[i - lookback + 1 : i + 1]
        m = math.fsum(w.tolist()) / lookback
        means.append(m)
        sds.append(math.sqrt(math.fsum(((w - m) ** 2).tolist()) / (lookback - 1)))
    return np.array(means), np.array(sds)


def synthetic_bars(n: int, seed: int = 0) -> Bars:
    rng = np.random.default_rng(seed)
    lx = math.log(50.0) + np.cumsum(rng.normal(0.0, 0.01, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0.0, 0.01)
    ly = 0.1 + 1.3 * lx + spread
    ts = hourly_ts(n)
    cx, cy = np.exp(lx), np.exp(ly)
    return Bars(
        name="synthetic", ts=ts, close_x=cx, close_y=cy, sha256=dataset_content_sha256(ts, cx, cy)
    )


class TestResolveSplit:
    def test_fraction_mode_with_embargo(self) -> None:
        idx = resolve_split(hourly_ts(1000), SplitSpec(train_fraction=0.6, embargo_bars=10), 20)
        assert (idx.n_bars, idx.train_end, idx.test_start) == (1000, 600, 610)
        assert (idx.n_train, idx.n_test) == (600, 390)

    def test_fraction_uses_floor(self) -> None:
        idx = resolve_split(hourly_ts(999), SplitSpec(train_fraction=0.5), 20)
        assert idx.train_end == 499  # floor(499.5)

    def test_timestamp_mode_left_side_semantics(self) -> None:
        ts = hourly_ts(1000)
        exact = SplitSpec(
            mode="timestamp",
            train_fraction=None,
            train_end=T0 + timedelta(hours=400),  # == ts[400] -> bar 400 is NOT train
            test_start=T0 + timedelta(hours=410, minutes=1),  # between bars -> next bar 411
        )
        idx = resolve_split(ts, exact, 20)
        assert (idx.train_end, idx.test_start) == (400, 411)

    def test_timestamp_mode_embargo_pushes_test_start(self) -> None:
        split = SplitSpec(
            mode="timestamp",
            train_fraction=None,
            train_end=T0 + timedelta(hours=400),
            test_start=T0 + timedelta(hours=400),
            embargo_bars=25,
        )
        idx = resolve_split(hourly_ts(1000), split, 20)
        assert (idx.train_end, idx.test_start) == (400, 425)

    def test_timestamp_mode_handles_far_dates_without_overflow(self) -> None:
        far = SplitSpec(
            mode="timestamp",
            train_fraction=None,
            train_end=datetime(1700, 1, 1, tzinfo=UTC),
            test_start=datetime(9999, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(SpecError, match=r"train segment has 0 bars.*test segment has 0 bars"):
            resolve_split(hourly_ts(1000), far, 20)

    def test_timestamp_mode_respects_gaps(self) -> None:
        ts = hourly_ts(1000)
        gappy = np.concatenate([ts[:300], ts[310:]])  # 10 missing hours
        split = SplitSpec(
            mode="timestamp",
            train_fraction=None,
            train_end=T0 + timedelta(hours=305),  # inside the gap
            test_start=T0 + timedelta(hours=305),
        )
        idx = resolve_split(gappy, split, 20)
        assert (idx.train_end, idx.test_start) == (300, 300)

    @pytest.mark.parametrize(
        ("n", "split", "lookback", "match"),
        [
            (1000, SplitSpec(train_fraction=0.06), 10, "train segment has 60 bars"),
            (1000, SplitSpec(train_fraction=0.6, embargo_bars=320), 10, "test segment has 80 bars"),
            (1000, SplitSpec(train_fraction=0.6), 201, r"< 2 x lookback 201"),
        ],
    )
    def test_protocol_violations_raise_spec_error(
        self, n: int, split: SplitSpec, lookback: int, match: str
    ) -> None:
        with pytest.raises(SpecError, match=match) as info:
            resolve_split(hourly_ts(n), split, lookback)
        assert info.value.retryable is False

    def test_boundary_values_are_allowed(self) -> None:
        # exactly MIN_TRAIN_BARS train bars and exactly 2 x lookback test bars
        idx = resolve_split(hourly_ts(250), SplitSpec(train_fraction=0.4), 75)
        assert (idx.train_end, idx.n_test) == (100, 150)
        with pytest.raises(SpecError, match="lookback 76"):
            resolve_split(hourly_ts(250), SplitSpec(train_fraction=0.4), 76)
        idx = resolve_split(hourly_ts(400), SplitSpec(train_fraction=0.5, embargo_bars=100), 10)
        assert idx.n_test == 100  # exactly MIN_TEST_BARS

    @pytest.mark.parametrize(
        "split",
        [
            SplitSpec(train_fraction=0.6),
            SplitSpec(train_fraction=0.37, embargo_bars=17),
            SplitSpec(
                mode="timestamp",
                train_fraction=None,
                train_end=datetime(2018, 1, 1, tzinfo=UTC),
                test_start=datetime(2018, 2, 3, 12, tzinfo=UTC),
                embargo_bars=3,
            ),
        ],
    )
    def test_agrees_with_admission_estimate_on_regular_grids(self, split: SplitSpec) -> None:
        info = get_catalog().get("pairs-ou-small-v1")
        assert info is not None
        bars = generate_bars(info.name)
        spec = ExperimentBatchSpec(
            name="split check",
            dataset=dataset_ref(info.name),
            strategy=StrategyRef(name="pairs_zscore", version="1"),
            grid={"lookback": [20]},
            fixed_params={"entry_z": 2.0, "exit_z": 0.5},
            split=split,
        )
        n_train, n_test, problems = estimate_split(spec, info)
        assert problems == []
        idx = resolve_split(bars.ts, split, 20)
        assert (idx.n_train, idx.n_test) == (n_train, n_test)

    def test_datetime_to_ns_is_exact(self) -> None:
        dt = datetime(2262, 4, 11, 23, 47, 16, 854775, tzinfo=UTC)
        assert datetime_to_ns(dt) == 9_223_372_036_854_775_000
        assert datetime_to_ns(datetime(1969, 12, 31, 23, 59, 59, tzinfo=UTC)) == -1_000_000_000
        with pytest.raises(ValueError, match="timezone-aware"):
            datetime_to_ns(datetime(2020, 1, 1))  # naive


class TestHedgeRatio:
    def test_recovers_an_exact_linear_relation(self) -> None:
        lx = np.log(np.linspace(20.0, 80.0, 300))
        ly = -0.7 + 1.6 * lx
        beta, alpha = fit_hedge_ratio(lx, ly, 300)
        assert beta == pytest.approx(1.6, rel=1e-12)
        assert alpha == pytest.approx(-0.7, rel=1e-11)

    def test_population_moments(self) -> None:
        rng = np.random.default_rng(5)
        lx = rng.normal(size=500)
        ly = 0.3 * lx + rng.normal(size=500)
        beta, alpha = fit_hedge_ratio(lx, ly, 500)
        expected = np.mean((lx - lx.mean()) * (ly - ly.mean())) / np.var(lx)
        assert beta == pytest.approx(expected, rel=1e-13)
        assert alpha == pytest.approx(ly.mean() - expected * lx.mean(), rel=1e-12)

    def test_uses_train_segment_only(self) -> None:
        rng = np.random.default_rng(1)
        lx = rng.normal(size=400)
        ly = 2.0 * lx + rng.normal(scale=0.1, size=400)
        before = fit_hedge_ratio(lx, ly, 250)
        ly2 = ly.copy()
        ly2[250:] = -ly2[250:] * 7
        assert fit_hedge_ratio(lx, ly2, 250) == before

    def test_constant_log_x_raises(self) -> None:
        lx = np.full(200, math.log(37.0))
        with pytest.raises(SpecError, match="zero"):
            fit_hedge_ratio(lx, np.linspace(0, 1, 200), 200)


class TestRollingStats:
    @pytest.mark.parametrize(
        ("n", "lookback", "first"),
        [(500, 5, 4), (500, 20, 100), (3 * ROLLING_BLOCK + 17, 50, 49), (5000, 777, 1200)],
    )
    def test_matches_naive_two_pass(self, n: int, lookback: int, first: int) -> None:
        rng = np.random.default_rng(n + lookback)
        values = np.cumsum(rng.normal(0, 0.01, n)) + 0.3
        stats = rolling_mean_std(values, lookback, first)
        mean, sd = naive_rolling(values, lookback, first)
        np.testing.assert_allclose(stats.mean, mean, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(stats.sd, sd, rtol=1e-10, atol=1e-15)
        assert stats.first == first

    def test_constant_windows_have_zero_dispersion(self) -> None:
        values = np.concatenate(
            [np.full(300, 0.123456789), np.linspace(0, 1, 100), np.full(300, -7.5)]
        )
        stats = rolling_mean_std(values, 30, 29)
        sd = stats.sd
        assert np.all(sd[: 300 - 29] <= 1e-15)  # first constant regime
        assert np.all(sd[-(300 - 29) :] <= 1e-14)  # second constant regime (|level| 7.5)
        assert stats.recomputed_windows >= 2 * (300 - 29)

    def test_large_offset_keeps_relative_accuracy(self) -> None:
        rng = np.random.default_rng(9)
        values = 1e6 + rng.normal(0, 1e-3, 3000)
        stats = rolling_mean_std(values, 40, 39)
        mean, sd = naive_rolling(values, 40, 39)
        np.testing.assert_allclose(stats.sd, sd, rtol=1e-9)
        np.testing.assert_allclose(stats.mean, mean, rtol=1e-15)

    def test_trending_series(self) -> None:
        values = np.linspace(-5.0, 5.0, 4000) ** 3
        stats = rolling_mean_std(values, 64, 100)
        mean, sd = naive_rolling(values, 64, 100)
        np.testing.assert_allclose(stats.sd, sd, rtol=1e-9)
        np.testing.assert_allclose(stats.mean, mean, rtol=1e-12, atol=1e-12)

    def test_is_deterministic_and_checks_cancel(self) -> None:
        values = np.random.default_rng(3).normal(size=5 * ROLLING_BLOCK)
        token = CountingToken()
        a = rolling_mean_std(values, 30, 29, cancel=token)
        b = rolling_mean_std(values, 30, 29)
        assert np.array_equal(a.mean, b.mean)
        assert np.array_equal(a.sd, b.sd)
        assert token.checks >= 5

    @pytest.mark.parametrize(("lookback", "first"), [(1, 0), (10, 8), (10, 501)])
    def test_argument_validation(self, lookback: int, first: int) -> None:
        with pytest.raises(ValueError, match=r"lookback|first"):
            rolling_mean_std(np.zeros(500), lookback, first)

    def test_zscores_nan_before_first_full_window_and_zero_on_flat(self) -> None:
        spread = np.concatenate([np.random.default_rng(0).normal(size=150), np.full(100, 0.25)])
        z, _ = segment_zscores(spread, 60, 30)  # test starts before the first full window
        assert z.shape == (220,)
        assert np.all(np.isnan(z[: 59 - 30]))
        assert not np.any(np.isnan(z[59 - 30 :]))
        assert np.all(z[-40:] == 0.0)  # windows fully inside the flat tail
        stats = rolling_mean_std(spread, 60, 59)
        i = 120
        expected = (spread[i] - stats.mean[i - 59]) / stats.sd[i - 59]
        assert z[i - 30] == expected
        assert SD_FLOOR == 1e-12


def positions(z: list[float], **kw: object) -> list[int]:
    return compute_positions(np.array(z, dtype=np.float64), params(**kw)).tolist()


class TestPositions:
    def test_short_spread_entry_and_exit(self) -> None:
        assert positions([0.0, 2.5, 1.0, 0.6, 0.5, 2.1, 0.0]) == [0, -1, -1, -1, 0, -1, 0]

    def test_long_spread_entry_and_exit(self) -> None:
        assert positions([0.0, -2.5, -1.0, -0.5, -3.0, 0.0]) == [0, 1, 1, 0, 1, 0]

    def test_threshold_boundaries(self) -> None:
        assert positions([2.0, -2.0, 0.0]) == [0, 0, 0]  # entry needs |z| > entry_z
        assert positions([-2.01, -0.5, 0.0]) == [1, 0, 0]  # long exits at z >= -exit_z
        assert positions([2.01, 0.5, 0.0]) == [-1, 0, 0]  # short exits at z <= exit_z
        assert positions([-2.01, -0.5000001, 0.0]) == [1, 1, 0]

    def test_no_same_bar_reversal(self) -> None:
        # short open, z flips through -entry: exit to flat first, re-enter long next bar
        assert positions([2.5, -2.5, -2.5, -2.5, 0.0]) == [-1, 0, 1, 1, 0]

    def test_stop_loss(self) -> None:
        z = [-2.5, -3.5, -4.1, -4.5, 0.0]
        assert positions(z, stop_z=4.0) == [1, 1, 0, 1, 0]  # stop at <= -4, then re-entry
        assert positions(z) == [1, 1, 1, 1, 0]  # no stop
        assert positions([2.5, 4.0, 0.0], stop_z=4.0) == [-1, 0, 0]  # short stop at z >= 4

    def test_max_holding_bars(self) -> None:
        z = [0.0, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, 0.0]
        # enter at k=1; forced flat at k = 1 + 3 = 4; re-enter at 5; flat at 8 (last bar)
        assert positions(z, max_holding_bars=3) == [0, 1, 1, 1, 0, 1, 1, 1, 0]
        assert positions(z, max_holding_bars=1) == [0, 1, 0, 1, 0, 1, 0, 1, 0]
        assert positions(z) == [0, 1, 1, 1, 1, 1, 1, 1, 0]

    def test_last_bar_is_forced_flat_and_nan_is_flat(self) -> None:
        assert positions([2.5, 2.5, 2.5]) == [-1, -1, 0]
        assert positions([math.nan, 2.5, math.nan, 2.5, 0.0]) == [0, -1, 0, -1, 0]
        assert positions([3.0]) == [0]

    def test_exit_z_zero(self) -> None:
        assert positions([-2.5, -0.1, 0.0, 0.0], exit_z=0.0) == [1, 1, 0, 0]

    def test_checks_cancel_and_can_abort(self) -> None:
        z = np.zeros(3 * st.CANCEL_CHECK_EVERY + 1)
        token = CountingToken()
        compute_positions(z, params(), cancel=token)
        assert token.checks >= 3
        with pytest.raises(TaskAborted):
            compute_positions(z, params(), cancel=CountingToken(abort_at=2))


class TestRunPairsZScore:
    def test_output_contract_and_return_formula(self) -> None:
        bars = synthetic_bars(800, seed=4)
        p = params(lookback=24, entry_z=1.5, exit_z=0.2)
        fees = FeeModel(commission_bps=3.0, slippage_bps=2.0)
        res = run_pairs_zscore(bars, p, SplitSpec(train_fraction=0.5, embargo_bars=7), fees)
        te = 407
        assert (res.split.train_end, res.split.test_start) == (400, te)
        n_test = 800 - te
        series = res.series()
        assert list(series) == ["ts", "spread", "zscore", "position", "ret", "equity"]
        for key, arr in series.items():
            assert arr.shape == (n_test,), key
        assert series["ts"].dtype == np.int64
        assert series["position"].dtype == np.int64
        assert np.array_equal(series["ts"], bars.ts[te:])
        assert res.position[-1] == 0
        assert set(np.unique(res.position).tolist()) <= {-1, 0, 1}
        assert int(np.count_nonzero(np.diff(res.position))) > 4  # it trades

        lx, ly = st.log_prices(bars.close_x), st.log_prices(bars.close_y)
        beta, alpha = fit_hedge_ratio(lx, ly, 400)
        assert (res.beta, res.alpha) == (beta, alpha)
        np.testing.assert_allclose(res.spread, (ly - beta * lx - alpha)[te:], rtol=0, atol=1e-15)
        p_prev = np.concatenate([[0], res.position[:-1]])
        g = np.diff(ly)[te - 1 :] - beta * np.diff(lx)[te - 1 :]
        c = 5.0 / 1e4
        assert res.cost_rate == c
        expected = p_prev * g / (1 + abs(beta)) - c * np.abs(res.position - p_prev)
        np.testing.assert_allclose(res.ret, expected, rtol=1e-12, atol=1e-16)
        assert np.array_equal(res.equity, np.cumprod(1.0 + res.ret))

    def test_outputs_do_not_alias_inputs(self) -> None:
        bars = synthetic_bars(400)
        res = run_pairs_zscore(bars, params(), SplitSpec(train_fraction=0.5), FeeModel())
        assert not np.shares_memory(res.ts, bars.ts)

    def test_zero_cost_flat_strategy_has_unit_equity(self) -> None:
        bars = synthetic_bars(400)
        res = run_pairs_zscore(
            bars,
            params(entry_z=9.9, exit_z=0.0),
            SplitSpec(train_fraction=0.5),
            FeeModel(commission_bps=0.0, slippage_bps=0.0),
        )
        assert not res.position.any()
        assert np.all(res.ret == 0.0)
        assert np.all(res.equity == 1.0)

    def test_short_split_raises_spec_error(self) -> None:
        bars = synthetic_bars(300)
        with pytest.raises(SpecError):
            run_pairs_zscore(bars, params(lookback=80), SplitSpec(train_fraction=0.6), FeeModel())

    def test_null_token_default_and_identifiers(self) -> None:
        bars = synthetic_bars(300)
        a = run_pairs_zscore(bars, params(), SplitSpec(train_fraction=0.5), FeeModel())
        b = run_pairs_zscore(
            bars, params(), SplitSpec(train_fraction=0.5), FeeModel(), cancel=NullCancelToken()
        )
        for key, arr in a.series().items():
            assert np.array_equal(arr, b.series()[key], equal_nan=True), key
        assert (st.STRATEGY_NAME, st.STRATEGY_VERSION) == ("pairs_zscore", "1")


def _spike_bars(extra: dict[int, float] | None = None) -> Bars:
    """n=260. log x = ln100 + 0.001 i; log y = log x + s with s = 0 except two one-bar spikes."""
    n = 260
    i = np.arange(n)
    lx = math.log(100.0) + 0.001 * i
    s = np.zeros(n)
    s[200] = 0.1
    s[230] = -0.2
    for k, v in (extra or {}).items():
        s[k] = v
    ts = hourly_ts(n)
    return Bars("hand", ts, np.exp(lx), np.exp(lx + s), "0" * 64)


class TestHandComputedExample:
    """beta=1, alpha=0 (train spread is exactly 0). With lookback 5 a lone spike d gives
    window [0,0,0,0,d]: mean d/5, sample sd d*sqrt(0.2), z = 4/sqrt(5) = 1.7889 -> entry at
    the spike bar; the next bar is [0,0,0,d,0] with z = -0.4472, inside the exit band, so the
    position is closed one bar later. Test segment is bars 156..259 (floor(260*0.6) = 156).
    """

    def run(self, bars: Bars) -> st.StrategyResult:
        params = PairsZScoreParams(lookback=5, entry_z=1.5, exit_z=0.5)
        return run_pairs_zscore(
            bars, params, SplitSpec(train_fraction=0.6), FeeModel(commission_bps=3, slippage_bps=2)
        )

    def test_end_to_end(self) -> None:
        res = self.run(_spike_bars())
        assert (res.split.train_end, res.split.test_start, res.split.n_test) == (156, 156, 104)
        assert res.beta == pytest.approx(1.0, abs=1e-12)
        assert res.alpha == pytest.approx(0.0, abs=1e-12)
        c = 5e-4
        assert res.cost_rate == pytest.approx(c)
        assert res.zscore[200 - 156] == pytest.approx(4 / math.sqrt(5), rel=1e-9)
        assert res.zscore[201 - 156] == pytest.approx(-1 / math.sqrt(5), rel=1e-9)
        assert res.zscore[230 - 156] == pytest.approx(-4 / math.sqrt(5), rel=1e-9)
        assert res.zscore[231 - 156] == pytest.approx(1 / math.sqrt(5), rel=1e-9)
        expected_pos = np.zeros(104, dtype=np.int64)
        expected_pos[200 - 156] = -1  # short spread after the positive spike
        expected_pos[230 - 156] = +1  # long spread after the negative spike
        assert res.position.tolist() == expected_pos.tolist()
        expected_ret = np.zeros(104)
        expected_ret[200 - 156] = -c  # entry cost on the entry bar, no position held yet
        expected_ret[201 - 156] = 0.05 - c  # -1 * (-0.1) / (1+1) minus exit cost
        expected_ret[230 - 156] = -c
        expected_ret[231 - 156] = 0.10 - c  # +1 * (+0.2) / 2 minus exit cost
        np.testing.assert_allclose(res.ret, expected_ret, rtol=0, atol=1e-12)
        eq = (1 - c) * (1.05 - c) * (1 - c) * (1.10 - c)
        assert res.equity[-1] == pytest.approx(eq, rel=1e-12)
        from quantgrid.domain.metrics import compute_metrics

        m = compute_metrics(res)
        assert m["n_trades"] == 2 and m["turnover"] == 4
        assert m["fees_paid"] == pytest.approx(4 * c)
        assert m["exposure"] == pytest.approx(2 / 104)
        assert m["hit_rate"] == 1.0 and m["avg_holding_bars"] == 1.0
        assert m["total_return"] == pytest.approx(eq - 1, rel=1e-12)

    def test_no_lookahead_future_bars_do_not_change_past_decisions(self) -> None:
        base = self.run(_spike_bars())
        # Perturb only bars strictly after 215 (after the first trade, before the second).
        pert = self.run(_spike_bars({250: 0.3, 251: -0.3}))
        k = 215 - 156
        assert np.array_equal(base.position[: k + 1], pert.position[: k + 1])
        assert np.array_equal(base.zscore[: k + 1], pert.zscore[: k + 1], equal_nan=True)
        assert np.array_equal(base.ret[: k + 1], pert.ret[: k + 1])
        assert base.beta == pert.beta and base.alpha == pert.alpha

    def test_hedge_ratio_ignores_test_bars(self) -> None:
        a = _spike_bars()
        y2 = a.close_y.copy()
        y2[156:] *= 1.5  # wreck the whole test segment
        b = Bars("hand2", a.ts, a.close_x, y2, "0" * 64)
        assert self.run(a).beta == self.run(b).beta
        assert self.run(a).alpha == self.run(b).alpha
