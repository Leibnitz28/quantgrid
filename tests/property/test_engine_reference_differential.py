"""Differential test: vectorised engine vs the slow pure-Python reference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quantgrid.domain.errors import SpecError
from quantgrid.domain.metrics import compute_metrics
from quantgrid.domain.reference import reference_run
from quantgrid.domain.specs import FeeModel, PairsZScoreParams, SplitSpec
from quantgrid.domain.strategy import run_pairs_zscore
from quantgrid.domain.types import Bars

DAY_NS = 86_400 * 1_000_000_000


def make_bars(close_x: np.ndarray, close_y: np.ndarray) -> Bars:
    n = len(close_x)
    ts = np.arange(n, dtype=np.int64) * DAY_NS + 1_420_070_400 * 1_000_000_000
    return Bars("hyp", ts, close_x.astype(np.float64), close_y.astype(np.float64), "0" * 64)


@st.composite
def scenarios(draw: st.DrawFn) -> dict[str, Any]:
    n = draw(st.integers(260, 600))
    kind = draw(st.sampled_from(["ou", "walk", "raw", "flat_y"]))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    if kind == "raw":  # arbitrary positive prices, wide dynamic range
        x = np.exp(rng.uniform(-3, 3, n))
        y = np.exp(rng.uniform(-3, 3, n))
    else:
        lx = np.log(100.0) + np.cumsum(rng.normal(0, 0.01, n))
        if kind == "ou":
            s = np.zeros(n)
            for i in range(1, n):
                s[i] = 0.9 * s[i - 1] + rng.normal(0, 0.01)
            ly = 0.1 + 1.3 * lx + s
        elif kind == "walk":
            ly = np.log(50.0) + np.cumsum(rng.normal(0, 0.01, n))
        else:  # y is an exact multiple: spread constant after fit -> sd floor path
            ly = 0.5 * lx + 1.0
        x, y = np.exp(lx), np.exp(ly)
    lookback = draw(st.integers(5, 60))
    entry = draw(st.floats(0.3, 3.0))
    exit_z = draw(st.floats(0.0, 0.95)) * entry
    stop = draw(st.one_of(st.none(), st.floats(1.0, 4.0).map(lambda m: entry * m + 0.01)))
    max_hold = draw(st.one_of(st.none(), st.integers(1, 40)))
    embargo = draw(st.integers(0, 20))
    frac = draw(st.floats(0.3, 0.7))
    fees = FeeModel(commission_bps=draw(st.floats(0, 20)), slippage_bps=draw(st.floats(0, 20)))
    return {
        "x": x,
        "y": y,
        "params": PairsZScoreParams(
            lookback=lookback, entry_z=entry, exit_z=exit_z, stop_z=stop, max_holding_bars=max_hold
        ),
        "split": SplitSpec(train_fraction=frac, embargo_bars=embargo),
        "fees": fees,
    }


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(scenarios())
def test_engine_matches_reference(sc: dict[str, Any]) -> None:
    bars = make_bars(sc["x"], sc["y"])
    try:
        ref = reference_run(
            bars.ts.tolist(),
            bars.close_x.tolist(),
            bars.close_y.tolist(),
            sc["params"],
            sc["split"],
            sc["fees"],
        )
    except SpecError:
        with pytest.raises(SpecError):
            run_pairs_zscore(bars, sc["params"], sc["split"], sc["fees"])
        return
    res = run_pairs_zscore(bars, sc["params"], sc["split"], sc["fees"])
    assert res.split.train_end == ref.tr_end and res.split.test_start == ref.te_start
    assert close(res.beta, ref.beta) and close(res.alpha, ref.alpha)
    assert res.position.tolist() == ref.position
    assert res.ts.tolist() == ref.ts
    for name, got, want in (
        ("spread", res.spread, ref.spread),
        ("ret", res.ret, ref.ret),
        ("equity", res.equity, ref.equity),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, err_msg=name)
    np.testing.assert_allclose(res.zscore, ref.zscore, rtol=1e-9, atol=1e-12, equal_nan=True)
    metrics = compute_metrics(res)
    for key, mwant in ref.metrics.items():
        mgot = metrics[key]
        if mwant is None or mgot is None:
            assert (mwant is None and mgot is None) or (
                # a metric on a std knife edge (~1e-15) may legitimately flip to None
                key in {"sharpe", "sortino"}
            ), key
        elif isinstance(mwant, int) and not isinstance(mwant, bool):
            assert mgot == mwant, key
        else:
            assert close(float(mgot), float(mwant)), (key, mgot, mwant)


def test_scenarios_are_not_vacuous() -> None:
    """A fixed OU scenario must actually trade, so position equality means something."""
    rng = np.random.default_rng(1)
    n = 500
    lx = np.log(100.0) + np.cumsum(rng.normal(0, 0.01, n))
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.9 * s[i - 1] + rng.normal(0, 0.01)
    bars = make_bars(np.exp(lx), np.exp(0.1 + 1.3 * lx + s))
    params = PairsZScoreParams(
        lookback=20, entry_z=1.5, exit_z=0.3, stop_z=4.0, max_holding_bars=15
    )
    res = run_pairs_zscore(bars, params, SplitSpec(), FeeModel())
    assert int(np.count_nonzero(res.position)) > 20
    assert (
        len(
            reference_run(
                bars.ts.tolist(),
                bars.close_x.tolist(),
                bars.close_y.tolist(),
                params,
                SplitSpec(),
                FeeModel(),
            ).trades
        )
        > 3
    )
