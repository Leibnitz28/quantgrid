# API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs`, OpenAPI at `/openapi.json`.

## Conventions

- **Auth:** `X-API-Key: <key>` on every `/v1` request. Keys come from `QG_API_KEYS`.
- **IDs** are UUIDs. **Timestamps** are UTC ISO-8601 with a `Z` suffix.
- **Correlation:** send `X-Request-ID` (letters, digits, `._:-`, up to 128 chars) or one is generated. It is echoed in the response header and in every error body, and appears in JSON logs.
- **Pagination:** list endpoints take `limit` and `cursor`; a response with `next_cursor` has more. Cursors are opaque.
- **Rate limits:** per user, per minute (`QG_RATE_LIMIT_PER_MINUTE`, and a separate `QG_PLANNER_RATE_LIMIT_PER_MINUTE` for AI endpoints). Exceeding returns `429` with `Retry-After`.
- **Body limit:** `QG_MAX_REQUEST_BYTES` (default 256 KiB), otherwise `413`.

## Errors

```json
{"error": {"type": "VALIDATION_ERROR", "message": "request validation failed",
           "request_id": "9f1c...", "details": [{"loc": ["body", "grid"], "msg": "...", "type": "..."}]}}
```

| `type` | Status | Meaning |
| --- | --- | --- |
| `VALIDATION_ERROR` | 422 | Schema or policy violation (`details.violations` lists policy codes) |
| `QUOTA_EXCEEDED` | 422 | Only quota limits were violated (grid size, runtime, worker-seconds, active batches) |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `NOT_FOUND` | 404 | Unknown id, or it belongs to another user |
| `CONFLICT` | 409 | Idempotency key reused with a different spec, or retry after cancel |
| `PAYLOAD_TOO_LARGE` | 413 | Body over the limit |
| `RATE_LIMITED` | 429 | Rate limit hit |
| `PLANNER_ERROR` | 502 | The planner failed (provider error, timeout, refusal) |
| `INTERNAL_ERROR` | 500 | Unexpected failure |

Task failures are recorded per attempt with a specific `error_code` (for example `WORKER_LOST`, `TASK_TIMEOUT`, `DATA_INTEGRITY_ERROR`) and reported under the umbrella type `WORKER_FAILURE`.

## Endpoints

### `POST /v1/batches`
Submit a batch. Requires an `Idempotency-Key` header (`[A-Za-z0-9._:-]`, 1 to 255). Returns `202`.

Repeating the same key and spec returns the original batch with `idempotent_replay: true`. The same key with a different spec returns `409`.

Request body (`ExperimentBatchSpec`):

| Field | Type | Notes |
| --- | --- | --- |
| `name` | string | 1 to 120 chars |
| `dataset` | `{name, sha256}` | The signature must match the registry |
| `strategy` | `{name, version}` | Built in: `pairs_zscore` `1` |
| `grid` | object of lists | Parameter name to values; cartesian product |
| `fixed_params` | object | Parameters not swept |
| `seeds` | list of int | Default `[0]`, unique, up to 64 |
| `split` | object | `fraction` (default 0.6) or `timestamp` mode; optional `embargo_bars` |
| `fees` | object | `commission_bps`, `slippage_bps` |
| `workload` | object | `bootstrap_samples` (0 to 50,000), `bootstrap_block` |
| `budget` | object | `max_runtime_s`, `max_memory_mb` per task |
| `max_attempts` | int | 1 to 20 |
| `cache_policy` | `use` or `bypass` | |
| `faults` | object | Failure injection; rejected unless `QG_ALLOW_FAULT_INJECTION=true` |

Unknown fields are rejected at every level. Tasks per batch = product of grid sizes times number of seeds.

Response:

```json
{"batch_id": "198a5f33-...", "state": "ACCEPTED", "task_count": 144, "spec_sha": "9198883d...",
 "idempotent_replay": false, "warnings": [],
 "estimate": {"task_count": 144, "est_seconds_per_task": 0.102, "est_total_worker_seconds": 14.688,
              "budget_worker_seconds": 17280, "est_memory_mb_per_task": 165.4,
              "n_train_bars_est": 1200, "n_test_bars_est": 800},
 "links": {"self": "/v1/batches/198a5f33-...", "tasks": ".../tasks", "results": ".../results", "cancel": ".../cancel"}}
```

### `GET /v1/batches`
Your batches, newest first. Query: `limit` (1 to 200), `cursor`.

### `GET /v1/batches/{id}`
Status: `state`, `progress.counts` per task state, attempts, retries, lease expirations, cache hits, `cost` (worker-seconds), timing, and the identifiers that pin the run (`spec_sha`, `dataset`, `code_sha`, `env_sha`, `git_commit`).

### `GET /v1/batches/{id}/tasks`
Tasks in ordinal order. Query: `state`, `limit` (1 to 1000), `cursor`.

### `GET /v1/tasks/{id}`
One task: spec, state, `attempts[]` (worker, status, error code, queue wait, wall and CPU time, peak RSS, output checksum, late-completion time), `worker_ids`, `artifact` (URI, checksum, size, rows, metrics), and `transitions[]` from the audit log.

### `POST /v1/batches/{id}/cancel`
Cooperative cancel. Returns `canceled_now`, `cancel_requested_running`, `completed_count`, `completed_task_ids` and `state`. Calling it on a finished batch returns `already_terminal: true`.

### `POST /v1/batches/{id}/retry-failed`
Optional body `{"extra_attempts": 1, "include_non_retryable": false}`. Moves eligible `FAILED` tasks back to `RETRY_PENDING` within the batch's quota snapshot. Returns what was retried and what was skipped and why. `409` if the batch was canceled; `422 QUOTA_EXCEEDED` if retrying would exceed the worker-time quota.

### `GET /v1/batches/{id}/results`
Published results only (they are immutable): per task `params`, `seed`, `metrics`, `output_uri`, `output_sha`; plus `complete`, per-state `counts`, a `summary` of the Sharpe distribution, and the full reproducibility `manifest`. Query: `limit` (1 to 1000), `cursor`.

### `POST /v1/planner/specs`
Body `{"goal": "...", "max_tasks": 100, "dataset": "pairs-ou-small-v1"}` (`goal` up to 4,000 chars). Returns `status` (`ok`, `rejected`, `invalid_output`, `error`), the proposed `spec`, the policy `violations` and `estimate`, and the agent trace. Nothing is stored or enqueued. Submit the returned `spec` to `POST /v1/batches` to run it.

### `POST /v1/batches/{id}/explain`
Read-only diagnosis of a batch's progress and failures: `summary`, `findings`, `trace`.

### `GET /v1/datasets`
Registered datasets with their checksums.

### Operational

| Path | Purpose |
| --- | --- |
| `GET /healthz` | Liveness |
| `GET /readyz` | `200` when PostgreSQL and Redis answer, otherwise `503` |
| `GET /metrics` | Prometheus text format |

## Results columns

Metrics stored per task: `total_return`, `ann_return`, `ann_vol`, `sharpe`, `sortino`, `max_drawdown`, `turnover`, `n_trades`, `exposure`, `fees_paid`, `hit_rate`, `avg_holding_bars`, `hedge_ratio`, `alpha`, `n_train_bars`, `n_test_bars`, `periods_per_year`, `bootstrap_samples`, and, when it is above 0, `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_boot_mean`, `p_sharpe_le_0`. A value is `null` when it is undefined (for example Sharpe of a flat series).
