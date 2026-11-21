# Synthetic datasets

Every dataset is recorded: an entry in `DATASET_DEFINITIONS`
(`src/quantgrid/domain/datasets.py`) pins the generator, every generator parameter
(including the seed) and the logical SHA-256 of the bars it produces. Workers materialise
the bars to Parquet once and recompute the logical hash on every load, so a task cannot
run on different data without raising `DATA_INTEGRITY_ERROR`.

## Generator (`generator_version` 1)

For a definition with `n_grid` bars, `bar_interval_s` and `seed`:

- `dt = bar_interval_s / (365.25 * 86400)` (years) and `rng = numpy.random.default_rng(seed)`.
- Shocks: `e_x = rng.standard_normal(n_grid)` is drawn first, then
  `e_2 = rng.standard_normal(n_grid)` (spread shocks for `cointegrated_ou`, `log y` shocks
  for the `independent_rw` control). Element 0 of each is drawn but unused, because bar 0
  is the fixed initial state.
- `log x` is a random walk with drift: `lx[0] = ln 100` and
  `lx[t] = lx[t-1] + (mu_x*dt + sigma_x*sqrt(dt)*e_x[t])`, computed as strictly sequential
  sums.
- `cointegrated_ou`: the spread is an Ornstein-Uhlenbeck process, `s[0] = 0` and
  `s[t] = s[t-1] - theta*s[t-1]*dt + sigma_s*sqrt(dt)*e_2[t]` with
  `theta = ln 2 / half_life_years`. Then `ly = alpha_true + beta_true*lx + s`.
- `independent_rw` (negative control): `ly[0] = ln 100 + y_log_offset` and
  `ly[t] = ly[t-1] + (mu_y*dt + sigma_y*sqrt(dt)*e_2[t])`. y is independent of x, so any
  edge a pairs strategy shows on this dataset is noise.
- `close = numpy.round(exp(l), price_decimals)`, using `math.exp` and `price_decimals = 6`.
  The 1e-6 tick (at most 5e-7 absolute, about 5e-9 relative) is far below any modelled
  effect. It makes the recorded signature stable against last-ulp differences between
  `exp` implementations: numpy's AVX-512 `exp` differs from glibc in about 5% of inputs,
  and without the rounding the recorded hash would depend on the worker's CPU or libm.
- Timestamps form a regular UTC grid, `start + k*bar_interval_s`, as int64 nanoseconds.
- Optional deterministic gaps: `default_rng(seed + 1).random(n_grid) < gap_fraction` marks
  bars to drop. The first and last bar are always kept. The latent process keeps evolving
  through a gap, so the next observed bar carries the accumulated move, as with a real
  missing bar.

### Parameter choices

The OU half-lives are tens of bars (10 days on daily bars, 2 days on hourly bars, 2 hours
on one-minute bars, 1 day on the gappy hourly set), so the z-score strategy trades.
Typical lookbacks of 20 to 200 bars see many mean-reversion cycles, and the stationary
spread standard deviation `sigma_s / sqrt(2*theta)` is 1% to 3.5% of price, well above the
default 1.5 bps cost.

## Logical content hash

`dataset_content_sha256` hashes a canonical byte layout:

1. The canonical JSON header
   `{"columns": ["ts","close_x","close_y"], "dtypes": ["<i8","<f8","<f8"], "length": n}`.
2. A single `\0` byte.
3. The raw little-endian bytes of `ts`, `close_x` and `close_y`, in that order.

The hash does not depend on the Parquet encoding (compression, row groups, writer version).
