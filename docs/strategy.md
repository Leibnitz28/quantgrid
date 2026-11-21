# Strategy, metrics and bootstrap

Normative definitions for the built-in strategy `pairs_zscore`, version `1`. The engine in `src/quantgrid/domain/strategy.py` implements them; the independent pure-Python implementation in `reference.py` follows the same text, and a property test compares the two.

## Strategy: pairs z-score mean reversion

* Bars `i = 0..n-1`; `lx_i = ln close_x_i` and `ly_i = ln close_y_i`.
* Split. In fraction mode, `tr_end = floor(n * train_fraction)` and
  `te_start = tr_end + embargo_bars`. In timestamp mode,
  `tr_end = searchsorted(ts, train_end_ns, "left")` and
  `te_start = max(searchsorted(ts, test_start_ns, "left"), tr_end + embargo_bars)`.
  The split requires `tr_end >= MIN_TRAIN_BARS`, `n - te_start >= MIN_TEST_BARS` and
  `n - te_start >= TEST_BARS_PER_LOOKBACK * lookback`; otherwise `SpecError`.
* Hedge ratio by OLS on the TRAIN segment `[0, tr_end)` only:
  `beta = cov(lx, ly) / var(lx)` with population moments (ddof=0), and
  `alpha = mean(ly) - beta*mean(lx)`. `var(lx) == 0` (log x constant on the train
  segment) raises `SpecError`.
* Spread `s_i = ly_i - beta*lx_i - alpha` for all `i`. Causal rolling statistics for
  `i >= lookback-1` over `s[i-lookback+1 .. i]`: mean `m_i` and SAMPLE std `sd_i`
  (ddof=1). `z_i = (s_i - m_i)/sd_i` if `sd_i > 1e-12`, else `0.0`.
  `z_i = NaN` for `i < lookback-1`.
* Positions `p_i` are in `{-1, 0, +1}`, decided at the close of bar `i` from
  `z_i` and held over `(i, i+1]`. `p_i = 0` for `i < te_start`, and
  `p_{n-1} = 0` (forced flat on the last bar). For `te_start <= i <= n-2`, with
  `prev = p_{i-1}` (and `p_{te_start-1} = 0`):

  - `z_i` NaN gives 0.
  - `prev == 0`: `z_i > entry_z` gives -1 (short spread), `z_i < -entry_z` gives
    +1, else 0.
  - `prev == +1`: go to 0 if `z_i >= -exit_z`, or (`stop_z` set and
    `z_i <= -stop_z`), or `held >= max_holding_bars`; else stay +1.
  - `prev == -1`: go to 0 if `z_i <= exit_z`, or (`stop_z` set and
    `z_i >= stop_z`), or `held >= max_holding_bars`; else stay -1.

  Exits go to flat only (no same-bar reversal). `held` is the number of bars since
  entry: entering at bar `e` gives `held = 1` at bar `e`, so the position is forced
  flat at bar `e + max_holding_bars`. In other words, the check at bar `i` uses
  `held = i - e`.
* Spread return `g_i = (ly_i - ly_{i-1}) - beta*(lx_i - lx_{i-1})`. With cost rate
  `c = (commission_bps + slippage_bps)/1e4`, the net return is
  `r_i = p_{i-1}*g_i/(1 + |beta|) - c*|p_i - p_{i-1}|` for every test bar `i` in
  `[te_start, n-1]`, and `equity_i = prod_{k=te_start..i} (1 + r_k)`.
* Output series for the test segment `[te_start, n-1]`: `ts` (int64 ns), `spread`,
  `zscore`, `position` (int64), `ret` and `equity`.

### Numerical notes

* Logs use `math.log` (libm), not `numpy.log`. numpy dispatches transcendental
  functions to CPU-specific SIMD kernels, so `math.log` keeps results identical across
  workers with different instruction sets. It costs about 12 ms per 200k prices.
* Rolling statistics use block-local cumulative sums of the spread, re-centred on each
  block's mean (O(n)). A window whose computed sum of squared deviations `M2` is not
  above `1e-3 * S2` would lose more than three digits to cancellation (`S2` is the
  sum of squares around the block anchor). Such a window, which includes every constant
  window, is recomputed exactly with a two-pass formula. Accuracy therefore matches a
  direct two-pass computation, and a constant window yields `sd` at ulp level, which the
  1e-12 floor maps to `z = 0`.
* All results are a pure function of the inputs: no randomness, no threading, and fixed
  block sizes that never depend on runtime conditions.

## Metrics

Every value is a finite Python `float`/`int` or `None` (undefined), never NaN or inf,
so metrics can be stored as canonical JSON (`result_artifacts.metrics`).

Definitions. `r` are the net returns of the test bars, `p` the positions, `equity` the
running product of `1 + r`, `n = n_test` and `dp_i = p_i - p_{i-1}` (with the
position before the first test bar flat). `c` is the cost rate.

* `periods_per_year = 365.25*86400 / bar_interval_s`, where
  `bar_interval_s = median(diff(ts))/1e9` over the test-segment timestamps. The median
  keeps it stable under missing bars.
* `total_return = equity_last - 1`. `ann_return = equity_last**(ppy/n) - 1`, or
  `None` if `equity_last <= 0` or the power overflows.
* `ann_vol = std(r, ddof=1)*sqrt(ppy)`. `sharpe = mean(r)/std(r, ddof=1)*sqrt(ppy)`,
  `None` if `n < 2` or `std <= 1e-15`. `sortino = mean(r)/downside*sqrt(ppy)` with
  `downside = sqrt(mean(min(r, 0)^2))`, `None` if `downside <= 1e-15`.
* `max_drawdown = max_i (1 - equity_i/running_max_i)`, where the running maximum
  starts at 1.0. It is a non-negative fraction, and exceeds 1 only if equity goes
  non-positive.
* `turnover = sum |dp|` (int). `n_trades` = number of entries (0 -> +/-1).
  `exposure` = share of test bars with `p_{i-1} != 0`. `fees_paid = sum c*|dp|`.
* A trade enters at bar `e` and exits (back to flat) at bar `x`, and its
  `pnl = sum(r_e .. r_x)` (`math.fsum`), so it includes both transaction costs.
  `hit_rate` = share of closed trades with `pnl > 0` and `avg_holding_bars` =
  mean of `x - e`. Both are `None` without trades. Every trade is closed, because the
  strategy is flat on the last bar.
* Pass-through values: `hedge_ratio`, `alpha`, `n_train_bars`, `n_test_bars` and
  `periods_per_year`.

## Bootstrap workload

`WorkloadSpec.bootstrap_samples` resamples of the net test returns, with mean block length
`WorkloadSpec.bootstrap_block`, give a confidence interval for the annualised Sharpe ratio.
Results are a pure function of the inputs; the algorithm below is normative.

* `rng = numpy.random.default_rng([seed, 0x5EED])`.
* For each sample `b = 0 .. samples-1`, in order: `u = rng.random(n)` then
  `starts = rng.integers(0, n, size=n)`. Set `new_block = u < 1/L` with
  `new_block[0] = True`. Let `j(t)` be the most recent `t' <= t` with
  `new_block[t']`; then `idx_t = (starts[j(t)] + (t - j(t))) mod n`. Resample
  `x = returns[idx]`; its Sharpe is `mean(x)/std(x, ddof=1)*sqrt(ppy)`, or `0.0` if
  `std <= 1e-15`.
* `sharpe_ci_low` and `sharpe_ci_high` are
  `numpy.percentile(sharpes, [2.5, 97.5], method="linear")`. `sharpe_boot_mean` is the
  mean of the resampled Sharpes and `p_sharpe_le_0` the share that are `<= 0`.

Samples are processed in chunks of at most 64 with `cancel.check()` before each chunk.
Results are bit-identical for every chunk size.
