"""Stationary block bootstrap (Politis & Romano, 1994) of the test-period Sharpe ratio.

This is the per-task CPU-bound workload. ``WorkloadSpec.bootstrap_samples`` resamples of
the net test returns, with mean block length ``WorkloadSpec.bootstrap_block``, yield a
confidence interval for the annualised Sharpe ratio. It is a description of sampling
uncertainty under the stationary-bootstrap assumptions, not a claim of real-world
profitability.

Algorithm (normative; results are a pure function of the inputs):

* ``rng = numpy.random.default_rng([seed, 0x5EED])``.
* For each sample ``b = 0 .. samples-1``, IN ORDER: ``u = rng.random(n)`` and then
  ``starts = rng.integers(0, n, size=n)``. Set ``new_block = u < 1/L`` with
  ``new_block[0] = True``. Let ``j(t)`` be the most recent ``t' <= t`` with
  ``new_block[t']``; then ``idx_t = (starts[j(t)] + (t - j(t))) mod n``, so blocks wrap
  circularly. Resample ``x = returns[idx]``; its Sharpe is
  ``mean(x)/std(x, ddof=1)*sqrt(ppy)``, or ``0.0`` if ``std <= 1e-15``.
* ``sharpe_ci_low`` and ``sharpe_ci_high`` are
  ``numpy.percentile(sharpes, [2.5, 97.5], method="linear")``. ``sharpe_boot_mean`` is
  the mean of the resampled Sharpes, and ``p_sharpe_le_0`` the share that are ``<= 0``.

Samples are processed in chunks of at most 64, with ``cancel.check()`` before each chunk.
Each sample's draws are made in order whatever the chunking, and statistics are computed
row by row, so results are bit-identical for every chunk size (tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from quantgrid.domain.types import CancelToken, FloatArray, IntArray, MetricValue, NullCancelToken

BOOTSTRAP_SEED_SALT: Final = 0x5EED
#: Samples per chunk never exceed this (cancellation latency bound).
MAX_CHUNK_SAMPLES: Final = 64
#: Default chunks hold at most this many resampled elements (memory bound ~ 50 MB).
MAX_CHUNK_ELEMENTS: Final = 1 << 20
STD_FLOOR: Final = 1e-15
BOOTSTRAP_METRIC_KEYS: Final[tuple[str, ...]] = (
    "sharpe_ci_low",
    "sharpe_ci_high",
    "sharpe_boot_mean",
    "p_sharpe_le_0",
    "bootstrap_samples",
)

_NULL_CANCEL: Final = NullCancelToken()


@dataclass(frozen=True)
class BootstrapResult:
    samples: int
    mean_block: int
    sharpe_ci_low: float
    sharpe_ci_high: float
    sharpe_boot_mean: float
    p_sharpe_le_0: float

    def as_metrics(self) -> dict[str, MetricValue]:
        return {
            "sharpe_ci_low": self.sharpe_ci_low,
            "sharpe_ci_high": self.sharpe_ci_high,
            "sharpe_boot_mean": self.sharpe_boot_mean,
            "p_sharpe_le_0": self.p_sharpe_le_0,
            "bootstrap_samples": self.samples,
        }


def empty_bootstrap_metrics() -> dict[str, MetricValue]:
    """Bootstrap metric keys for a task that ran no resamples."""
    return {
        "sharpe_ci_low": None,
        "sharpe_ci_high": None,
        "sharpe_boot_mean": None,
        "p_sharpe_le_0": None,
        "bootstrap_samples": 0,
    }


def default_chunk_size(n: int) -> int:
    """Samples per chunk for series length ``n`` (1 .. :data:`MAX_CHUNK_SAMPLES`)."""
    return max(1, min(MAX_CHUNK_SAMPLES, MAX_CHUNK_ELEMENTS // max(n, 1)))


def stationary_bootstrap_indices(u: FloatArray, starts: IntArray, mean_block: float) -> IntArray:
    """Resample indices for one sample (1-D inputs) or a stack of samples (2-D, row-wise).

    ``u`` are the uniforms that open new blocks and ``starts`` the candidate block starts.
    Fully vectorised: ``j(t)`` comes from a running maximum of block-start positions.
    """
    u2 = np.atleast_2d(u)
    s2 = np.atleast_2d(starts)
    if u2.shape != s2.shape or u2.ndim != 2:
        raise ValueError("u and starts must have the same 1-D or 2-D shape")
    n = int(u2.shape[1])
    new_block = u2 < (1.0 / mean_block)
    new_block[:, 0] = True
    t = np.arange(n, dtype=np.int64)
    last_start = np.where(new_block, t, np.int64(0))
    np.maximum.accumulate(last_start, axis=1, out=last_start)
    idx = np.take_along_axis(s2, last_start, axis=1)
    idx += t
    idx -= last_start
    idx[idx >= n] -= n  # starts < n and t - j < n, so one wrap suffices
    return idx if np.ndim(u) == 2 else idx[0]


def _row_sharpes(x: FloatArray, scale: float) -> FloatArray:
    mean = x.mean(axis=1)
    sd = x.std(axis=1, ddof=1)
    out = np.zeros(x.shape[0], dtype=np.float64)
    np.divide(mean, sd, out=out, where=sd > STD_FLOOR)
    out *= scale
    return out


def bootstrap_sharpes(
    returns: FloatArray,
    *,
    samples: int,
    mean_block: int,
    seed: int,
    periods_per_year: float,
    cancel: CancelToken = _NULL_CANCEL,
    chunk_size: int | None = None,
) -> FloatArray:
    """The resampled Sharpe ratios, in sample order (see module docstring)."""
    r = np.ascontiguousarray(returns, dtype=np.float64)
    if r.ndim != 1 or r.shape[0] < 2:
        raise ValueError("bootstrap needs a 1-D return series with at least 2 values")
    if not bool(np.all(np.isfinite(r))):
        raise ValueError("bootstrap returns must be finite")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1")
    if not (math.isfinite(periods_per_year) and periods_per_year > 0.0):
        raise ValueError("periods_per_year must be finite and > 0")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    n = int(r.shape[0])
    chunk = default_chunk_size(n) if chunk_size is None else chunk_size
    if not 1 <= chunk <= MAX_CHUNK_SAMPLES:
        raise ValueError(f"chunk_size must be in [1, {MAX_CHUNK_SAMPLES}]")
    rng = np.random.default_rng([seed, BOOTSTRAP_SEED_SALT])
    scale = math.sqrt(periods_per_year)
    sharpes = np.empty(samples, dtype=np.float64)
    u_buf = np.empty((chunk, n), dtype=np.float64)
    s_buf = np.empty((chunk, n), dtype=np.int64)
    for b0 in range(0, samples, chunk):
        cancel.check()
        k = min(chunk, samples - b0)
        for j in range(k):  # per-sample draw order: u, then starts
            rng.random(out=u_buf[j])
            s_buf[j] = rng.integers(0, n, size=n)
        idx = stationary_bootstrap_indices(u_buf[:k], s_buf[:k], mean_block)
        sharpes[b0 : b0 + k] = _row_sharpes(r[idx], scale)
    cancel.check()
    return sharpes


def stationary_bootstrap_sharpe(
    returns: FloatArray,
    *,
    samples: int,
    mean_block: int,
    seed: int,
    periods_per_year: float,
    cancel: CancelToken = _NULL_CANCEL,
    chunk_size: int | None = None,
) -> BootstrapResult:
    """Summarise :func:`bootstrap_sharpes` (CI, mean, share <= 0)."""
    sharpes = bootstrap_sharpes(
        returns,
        samples=samples,
        mean_block=mean_block,
        seed=seed,
        periods_per_year=periods_per_year,
        cancel=cancel,
        chunk_size=chunk_size,
    )
    low, high = np.percentile(sharpes, [2.5, 97.5], method="linear")
    return BootstrapResult(
        samples=samples,
        mean_block=mean_block,
        sharpe_ci_low=float(low),
        sharpe_ci_high=float(high),
        sharpe_boot_mean=float(np.mean(sharpes)),
        p_sharpe_le_0=int(np.count_nonzero(sharpes <= 0.0)) / samples,
    )
