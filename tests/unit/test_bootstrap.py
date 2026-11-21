"""Stationary block bootstrap: draw order, index construction, chunk invariance, cancellation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantgrid.domain.bootstrap import (
    BOOTSTRAP_METRIC_KEYS,
    BOOTSTRAP_SEED_SALT,
    MAX_CHUNK_SAMPLES,
    bootstrap_sharpes,
    default_chunk_size,
    empty_bootstrap_metrics,
    stationary_bootstrap_indices,
    stationary_bootstrap_sharpe,
)
from quantgrid.domain.errors import TaskAborted, TaskTimeout
from quantgrid.domain.types import DeadlineCancelToken

PPY = 8766.0  # hourly bars


class CountingToken:
    def __init__(self, abort_at: int | None = None) -> None:
        self.checks = 0
        self.abort_at = abort_at

    def check(self) -> None:
        self.checks += 1
        if self.abort_at is not None and self.checks >= self.abort_at:
            raise TaskAborted("LEASE_LOST")


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(2e-4, 3e-3, 1500)


def naive_indices(u: np.ndarray, starts: np.ndarray, mean_block: float) -> np.ndarray:
    """Literal transcription of the definition (explicit loop)."""
    n = u.shape[0]
    out = np.empty(n, dtype=np.int64)
    j = 0
    for t in range(n):
        if t == 0 or u[t] < 1.0 / mean_block:
            j = t
        out[t] = (starts[j] + (t - j)) % n
    return out


def naive_sharpe(x: np.ndarray, ppy: float) -> float:
    sd = float(np.std(x, ddof=1))
    return 0.0 if sd <= 1e-15 else float(np.mean(x)) / sd * math.sqrt(ppy)


class TestIndices:
    @pytest.mark.parametrize("mean_block", [1, 2, 7.5, 20, 400])
    def test_vectorised_indices_match_definition(self, mean_block: float) -> None:
        rng = np.random.default_rng(int(mean_block * 10))
        u = rng.random((5, 300))
        starts = rng.integers(0, 300, size=(5, 300))
        stacked = stationary_bootstrap_indices(u, starts, mean_block)
        for row in range(5):
            expected = naive_indices(u[row], starts[row], mean_block)
            assert np.array_equal(stacked[row], expected)
            assert np.array_equal(
                stationary_bootstrap_indices(u[row], starts[row], mean_block), expected
            )

    def test_block_length_one_is_the_iid_bootstrap(self) -> None:
        rng = np.random.default_rng(0)
        u, starts = rng.random(100), rng.integers(0, 100, 100)
        assert np.array_equal(stationary_bootstrap_indices(u, starts, 1), starts)

    def test_single_block_is_a_circular_shift(self) -> None:
        u = np.full(50, 0.5)
        starts = np.full(50, 37)
        idx = stationary_bootstrap_indices(u, starts, 1e9)
        assert np.array_equal(idx, (37 + np.arange(50)) % 50)

    def test_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="same"):
            stationary_bootstrap_indices(np.zeros(5), np.zeros(4, dtype=np.int64), 3)


class TestDrawOrderAndDeterminism:
    def test_samples_use_the_documented_stream(self, returns: np.ndarray) -> None:
        n = returns.shape[0]
        rng = np.random.default_rng([123, BOOTSTRAP_SEED_SALT])
        expected = []
        for _ in range(3):
            u = rng.random(n)
            starts = rng.integers(0, n, size=n)
            expected.append(naive_sharpe(returns[naive_indices(u, starts, 15)], PPY))
        got = bootstrap_sharpes(returns, samples=3, mean_block=15, seed=123, periods_per_year=PPY)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)
        assert BOOTSTRAP_SEED_SALT == 0x5EED

    @pytest.mark.parametrize("chunk", [1, 2, 7, 33, MAX_CHUNK_SAMPLES])
    def test_results_do_not_depend_on_chunking(self, returns: np.ndarray, chunk: int) -> None:
        kwargs = {"samples": 150, "mean_block": 12, "seed": 9, "periods_per_year": PPY}
        reference = bootstrap_sharpes(returns, **kwargs)  # default chunking
        chunked = bootstrap_sharpes(returns, chunk_size=chunk, **kwargs)
        assert np.array_equal(reference, chunked)
        assert stationary_bootstrap_sharpe(returns, chunk_size=chunk, **kwargs) == (
            stationary_bootstrap_sharpe(returns, **kwargs)
        )

    def test_prefix_property(self, returns: np.ndarray) -> None:
        kwargs = {"mean_block": 5, "seed": 77, "periods_per_year": PPY}
        long = bootstrap_sharpes(returns, samples=100, **kwargs)
        short = bootstrap_sharpes(returns, samples=37, **kwargs)
        assert np.array_equal(long[:37], short)

    def test_seed_sensitivity(self, returns: np.ndarray) -> None:
        a = bootstrap_sharpes(returns, samples=20, mean_block=5, seed=1, periods_per_year=PPY)
        b = bootstrap_sharpes(returns, samples=20, mean_block=5, seed=2, periods_per_year=PPY)
        assert not np.array_equal(a, b)


class TestSummary:
    def test_summary_statistics(self, returns: np.ndarray) -> None:
        kwargs = {"samples": 400, "mean_block": 10, "seed": 5, "periods_per_year": PPY}
        sharpes = bootstrap_sharpes(returns, **kwargs)
        res = stationary_bootstrap_sharpe(returns, **kwargs)
        low, high = np.percentile(sharpes, [2.5, 97.5], method="linear")
        assert (res.sharpe_ci_low, res.sharpe_ci_high) == (float(low), float(high))
        assert res.sharpe_boot_mean == float(np.mean(sharpes))
        assert res.p_sharpe_le_0 == float(np.mean(sharpes <= 0.0))
        assert res.samples == 400
        metrics = res.as_metrics()
        assert tuple(metrics) == BOOTSTRAP_METRIC_KEYS
        assert metrics["bootstrap_samples"] == 400
        sample_sharpe = naive_sharpe(returns, PPY)
        assert res.sharpe_ci_low < sample_sharpe < res.sharpe_ci_high
        assert res.sharpe_ci_low < res.sharpe_boot_mean < res.sharpe_ci_high

    def test_zero_returns(self) -> None:
        res = stationary_bootstrap_sharpe(
            np.zeros(200), samples=50, mean_block=4, seed=0, periods_per_year=PPY
        )
        assert (res.sharpe_ci_low, res.sharpe_ci_high, res.sharpe_boot_mean) == (0.0, 0.0, 0.0)
        assert res.p_sharpe_le_0 == 1.0

    def test_empty_metrics(self) -> None:
        empty = empty_bootstrap_metrics()
        assert tuple(empty) == BOOTSTRAP_METRIC_KEYS
        assert empty["bootstrap_samples"] == 0
        assert all(v is None for k, v in empty.items() if k != "bootstrap_samples")


class TestCancellationAndLimits:
    def test_cancel_is_checked_at_least_every_64_samples(self, returns: np.ndarray) -> None:
        token = CountingToken()
        bootstrap_sharpes(
            returns, samples=640, mean_block=8, seed=1, periods_per_year=PPY, cancel=token
        )
        assert token.checks >= 640 // MAX_CHUNK_SAMPLES

    def test_abort_propagates(self, returns: np.ndarray) -> None:
        with pytest.raises(TaskAborted) as info:
            bootstrap_sharpes(
                returns,
                samples=10_000,
                mean_block=8,
                seed=1,
                periods_per_year=PPY,
                cancel=CountingToken(abort_at=3),
            )
        assert info.value.reason == "LEASE_LOST"

    def test_deadline_token_times_out(self, returns: np.ndarray) -> None:
        token = DeadlineCancelToken(max_runtime_s=0.0)
        with pytest.raises(TaskTimeout):
            bootstrap_sharpes(
                returns, samples=1000, mean_block=8, seed=1, periods_per_year=PPY, cancel=token
            )

    def test_default_chunk_size_bounds_memory(self) -> None:
        assert default_chunk_size(8_000) == MAX_CHUNK_SAMPLES
        assert default_chunk_size(80_000) == 13
        assert default_chunk_size(10_000_000) == 1
        assert default_chunk_size(0) == MAX_CHUNK_SAMPLES

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"returns": np.zeros(1)}, "at least 2"),
            ({"returns": np.array([0.0, np.nan, 1.0])}, "finite"),
            ({"samples": 0}, "samples"),
            ({"mean_block": 0}, "mean_block"),
            ({"periods_per_year": 0.0}, "periods_per_year"),
            ({"periods_per_year": math.inf}, "periods_per_year"),
            ({"seed": -1}, "seed"),
            ({"chunk_size": MAX_CHUNK_SAMPLES + 1}, "chunk_size"),
            ({"chunk_size": 0}, "chunk_size"),
        ],
    )
    def test_argument_validation(self, kwargs: dict[str, object], match: str) -> None:
        args: dict[str, object] = {
            "returns": np.zeros(10),
            "samples": 5,
            "mean_block": 3,
            "seed": 0,
            "periods_per_year": PPY,
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            bootstrap_sharpes(**args)  # type: ignore[arg-type]
