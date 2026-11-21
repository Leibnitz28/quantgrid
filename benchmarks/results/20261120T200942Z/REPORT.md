# QuantGrid scaling benchmark — run 20261120T200942Z

All numbers below were measured by `scripts/benchmark.py` in this run; raw data is in
`runs.csv`, `attempts.csv` and `samples.csv` next to this file.

## Environment

| item | value |
|---|---|
| git commit | `None` (dirty: None) |
| code_sha | `62818535135c65f3…` |
| OS / kernel | Linux-6.18.44-fc-v726-x86_64-with-glibc2.39 / 6.18.44-fc-v726 |
| CPU | Intel(R) Xeon(R) Processor @ 2.10GHz × 4 logical CPUs |
| RAM | 15.7 GB |
| Python / numpy / pyarrow | 3.12.3 / 2.5.5 / 26.0.1 |
| PostgreSQL / Redis | PostgreSQL 16.15 (Ubuntu 16.15-0ubuntu0.24.04.1) / 7.0.15 |
| build | CPython (no debug build); numpy wheels from PyPI |
| dataset | `pairs-ou-medium-v1`, 20000 records, sha256 `275180684b103db7…` |
| scheduler | {'lease_ttl_s': 15.0, 'heartbeat_interval_s': 3.0, 'coordinator_tick_s': 0.5, 'dispatch_window_factor': 2.0, 'dispatch_min_window': 4} |
| workload | 32 long tasks (3000 bootstrap samples) + 64 short tasks (no bootstrap) per repetition; 3 repetitions; 1 warm-up batch per worker count (excluded) |

## Scaling (main workload)

| workers | reps | makespan mean (min–max) s | speedup T1/Tp | efficiency | Karp–Flatt e | tasks/min | queue wait p50 / p95 / p99 ms | utilisation | inter-task gap p50 ms | worker CPU % (mean) | worker RSS max MB | host CPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 18.689 (18.472–18.977) | 1.0 | 1.0 | None | 308.2 | 17849.7 / 18583.5 / 18644.5 | 0.954 | 5.4 | 90.9 | 202.1 | 27.2 |
| 2 | 3 | 9.903 (9.762–10.025) | 1.887 | 0.944 | 0.06 | 581.7 | 9420.2 / 9827.3 / 9864.3 | 0.947 | 5.7 | 92.1 | 202.0 | 52.5 |
| 4 | 3 | 5.189 (4.692–5.484) | 3.602 | 0.9 | 0.037 | 1115.5 | 4816.7 / 5123.2 / 5151.2 | 0.939 | 6.3 | 90.2 | 202.0 | 96.9 |

![speedup](speedup.svg)

## Overhead at low task counts

| tasks | workers | T1 s | Tp s | speedup | efficiency |
|---|---|---|---|---|---|
| 4 | 2 | 0.524 | 0.329 | 1.593 | 0.796 |
| 4 | 4 | 0.524 | 0.218 | 2.401 | 0.6 |
| 8 | 2 | 1.127 | 0.614 | 1.835 | 0.918 |
| 8 | 4 | 1.127 | 0.413 | 2.727 | 0.682 |
| 16 | 2 | 2.273 | 1.165 | 1.951 | 0.976 |
| 16 | 4 | 2.273 | 0.773 | 2.942 | 0.736 |
| 32 | 2 | 4.501 | 2.404 | 1.872 | 0.936 |
| 32 | 4 | 4.501 | 1.411 | 3.189 | 0.797 |
| 64 | 2 | 8.348 | 4.814 | 1.734 | 0.867 |
| 64 | 4 | 8.348 | 2.544 | 3.282 | 0.82 |

![efficiency vs batch size](efficiency_vs_tasks.svg)

## Method

* Makespan T_p = (last task `finished_at`) − (DB `now()` immediately before submission); both from PostgreSQL's clock.
* Queue wait = claim time − time the task became eligible (`ready_at`), so it includes time waiting in the database for a dispatch slot.
* Utilisation = Σ task wall time / (p × T_p). Inter-task gap = time between a worker publishing one task and starting the next (scheduler round trip).
* Speedup S_p = mean(T1)/mean(Tp); efficiency E_p = S_p/p; Karp–Flatt serial fraction e = (1/S − 1/p)/(1 − 1/p).
* Each worker count gets fresh processes and one excluded warm-up batch; every run uses new seeds, so every artifact is computed and written (cache bypassed, no content-address dedup).

## Measurement bias and caveats

* Coordinator, PostgreSQL, Redis, the benchmark driver and all workers share the same host; at 4 workers on a 4-vCPU machine they compete for CPU, so T4 understates what dedicated worker hosts would achieve.
* The host is a virtual machine: CPU frequency, turbo and steal time are not controlled; spread across repetitions is reported (min–max) instead of a single number.
* Short tasks are deliberately included: their cost is dominated by per-task scheduler overhead (claim, publish, NOTIFY → dispatch), which bounds efficiency at small batch sizes.
* Numerical libraries run single-threaded per process (OMP/OpenBLAS/MKL threads = 1); results do not transfer to multi-threaded configurations.
* Artifacts are written to a local filesystem with fsync; network/object storage would add latency per task.
