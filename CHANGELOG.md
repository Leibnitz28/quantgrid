# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-21

First release.

### Added
- Batch API: submit with an idempotency key, status, task detail with attempt log, cancel, retry-failed, merged results with a reproducibility manifest.
- Scheduler: PostgreSQL leases with a fencing token, bounded dispatch window, lease reconciliation, retries with backoff, cooperative cancellation, leader-elected coordinator.
- Database triggers enforcing the task and batch state machines, immutable specs, results and terminal attempts, and an audit log of every task transition.
- Content-addressed Parquet artifacts with staged commit, byte-identical for identical specs.
- Deterministic pairs z-score backtest engine, a pure-Python reference implementation, and five recorded synthetic datasets with checksum verification.
- Batch verifier (`quantgrid verify`) proving one published result per expected task.
- Read-only AI planner and explainer with a deterministic default provider and an optional Claude provider.
- Prometheus metrics, JSON logs with correlation ids, OpenTelemetry tracing.
- Chaos script (SIGKILL and SIGSTOP of a worker), scaling benchmark, demo script.
