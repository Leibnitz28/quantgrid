# Operations

## Processes

| Command | Role | Count |
| --- | --- | --- |
| `quantgrid migrate` | Apply database migrations, then exit | once per release |
| `quantgrid api --host 0.0.0.0 --port 8000` | HTTP API | any, stateless |
| `quantgrid coordinator` | Scheduler loop | 1 active; run 2 for failover |
| `quantgrid worker --worker-id NAME` | One execution slot | as many as CPU cores |

A worker is a single-process Celery `solo` pool. Parallelism comes from more worker processes, not threads. Set `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) so numeric libraries stay single-threaded.

Local PostgreSQL and Redis without Docker: `scripts/dev_services.sh start` (ports 5433 and 6380). Full stack in containers: `docker compose up --build --scale worker=4`.

## Configuration

All settings are environment variables prefixed `QG_`; `.env.example` lists every one with its default.

| Variable | Default | Effect |
| --- | --- | --- |
| `QG_DATABASE_URL` | local dev URL | SQLAlchemy URL for PostgreSQL |
| `QG_REDIS_URL` | `redis://127.0.0.1:6380/0` | Celery broker and rate limiter |
| `QG_ARTIFACT_ROOT` | `var/artifacts` | Content-addressed result store |
| `QG_DATA_DIR` | `var/data` | Materialised datasets |
| `QG_LEASE_TTL_S` | 15 | Time a worker can be silent before its task is reclaimed |
| `QG_HEARTBEAT_INTERVAL_S` | 3 | Lease extension period; keep at most TTL / 3 |
| `QG_COORDINATOR_TICK_S` | 0.5 | Coordinator wake-up interval when idle |
| `QG_DISPATCH_WINDOW_FACTOR`, `QG_DISPATCH_MIN_WINDOW` | 2, 4 | Unclaimed messages allowed = max(min window, live worker slots x factor) |
| `QG_DISPATCH_LOST_AFTER_S` | 120 | Re-dispatch a task unclaimed this long |
| `QG_RETRY_BACKOFF_BASE_S`, `QG_RETRY_BACKOFF_MAX_S` | 1, 30 | Exponential retry backoff |
| `QG_API_KEYS` | `dev-key:dev-user` | Comma-separated `key:user` pairs. Change it. |
| `QG_QUOTA_*` | see `.env.example` | Per-user limits applied to every batch |
| `QG_ALLOW_FAULT_INJECTION` | `false` | Allow requests that make tasks fail on purpose |
| `QG_LLM_PROVIDER` | `fake` | `fake` (local, deterministic) or `anthropic` |
| `QG_LOG_FORMAT`, `QG_LOG_LEVEL` | `json`, `INFO` | |
| `QG_OTEL_EXPORTER` | `none` | `none`, `console` or `otlp` (needs the `otlp` extra) |

The lease TTL is the main recovery trade-off. After a worker dies its task waits for the remaining TTL before it is retried. Measured recovery with the default 15 s TTL was about 15.4 s (see the README). A shorter TTL recovers faster but a worker that is only slow (a long GC pause, a loaded host) may lose its lease and have its result discarded.

## Health and metrics

- `GET /healthz`: process is up. `GET /readyz`: PostgreSQL and Redis reachable, otherwise 503.
- `GET /metrics` on the API. The coordinator and workers expose metrics on `QG_COORDINATOR_METRICS_PORT` and `QG_WORKER_METRICS_PORT` when set above 0.

| Metric | Type | Labels |
| --- | --- | --- |
| `quantgrid_tasks_by_state` | gauge | `state` |
| `quantgrid_live_workers` | gauge | |
| `quantgrid_task_attempts_total` | counter | `status`, `error_code` |
| `quantgrid_task_duration_seconds` | histogram | `outcome` |
| `quantgrid_queue_wait_seconds` | histogram | |
| `quantgrid_leases_reclaimed_total` | counter | `outcome` |
| `quantgrid_lease_lost_total`, `quantgrid_late_completions_total` | counter | `checksum` on the second |
| `quantgrid_tasks_dispatched_total`, `quantgrid_dispatch_redelivered_total` | counter | |
| `quantgrid_artifact_commits_total`, `quantgrid_artifact_bytes_written_total` | counter | `outcome` on the first |
| `quantgrid_cache_hits_total`, `quantgrid_heartbeat_failures_total` | counter | |
| `quantgrid_coordinator_tick_seconds` | histogram | |
| `quantgrid_http_requests_total`, `quantgrid_http_request_duration_seconds` | counter, histogram | `method`, `route`, `status` |
| `quantgrid_batches_submitted_total`, `quantgrid_planner_requests_total` | counter | `outcome`, `status` |

Labels are bounded; task and batch ids are never labels.

Useful alerts: `quantgrid_live_workers == 0` while tasks are pending; a rising `quantgrid_leases_reclaimed_total`; any `quantgrid_late_completions_total{checksum="mismatch"}` (a retry produced different bytes); `quantgrid_heartbeat_failures_total` increasing (database trouble).

## Logs

JSON, one object per line, to stderr: `ts`, `level`, `logger`, `msg`, `service`, plus `request_id`, `batch_id`, `task_id`, `worker_id`, `attempt`, `trace_id` when set. Keys containing `key`, `token`, `secret`, `password` or `authorization` are redacted.

## Shutdown

- **Worker, SIGTERM:** finishes the current task, marks itself stopping, exits. If it is killed instead, its task is retried after the lease expires.
- **Coordinator, SIGTERM:** finishes the current tick, releases the leader lock, exits.
- **API:** closes its connection pool.

## Checking a batch

```sh
quantgrid verify <batch-id>             # exits 1 if any check fails
quantgrid verify <batch-id> --recompute 5   # also re-run 5 tasks and compare bytes
```

It checks the stored spec and task set, one result per succeeded task, artifact checksums and contents, an independent DuckDB scan, the attempt log, every audited state transition, and that no retry produced different bytes. Orphaned artifacts (for example from canceled tasks) are reported as a warning; `quantgrid gc` removes stale temporary files only.

## Upgrades and backups

- Run `quantgrid migrate` before starting the new version. Migrations are forward and backward tested in CI (`alembic upgrade head`, `alembic check`, `alembic downgrade base`).
- Back up PostgreSQL and the artifact directory together. Redis holds no state that cannot be rebuilt: flushing it loses nothing, and the coordinator re-dispatches pending tasks.
- Changing the code or any installed dependency changes `code_sha` and `env_sha`, so cached results from earlier runs are no longer reused. That is intended.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Batch stays `ACCEPTED` | Is a coordinator running? `quantgrid_live_workers` above 0? Is Redis reachable (`/readyz`)? |
| Tasks `PENDING` but nothing runs | Workers must use the same `QG_QUEUE_NAME` and Redis as the coordinator |
| `LEASE_EXPIRED` attempts on healthy workers | Lease TTL too short for the host's load, or database latency; raise `QG_LEASE_TTL_S` |
| `DATA_INTEGRITY_ERROR` | Dataset file corrupt or a different checksum than the request; delete the file under `QG_DATA_DIR` to regenerate |
| `429` | Per-user rate limit; see `Retry-After` |
