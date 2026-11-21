# Correctness

Delivery is at-least-once: a task can run more than once. The effect, one published result per task, happens at most once, and exactly once for every task that succeeds. The project does not claim exactly-once execution.

| Invariant | Mechanism | Test |
| --- | --- | --- |
| A task has at most one published result | `result_artifacts.task_id` is unique; the insert and the move to `SUCCEEDED` share one transaction that holds the task row lock | `test_publish_once_and_transitions_logged`; verifier check `one_result_per_success` |
| A worker that lost its lease cannot publish | The lock holder re-checks `(lease_owner, attempts)` before publishing | `test_zombie_worker_cannot_publish_after_reassignment`; end-to-end SIGSTOP test |
| One of N concurrent claims wins | Conditional `UPDATE ... WHERE state IN (PENDING, RETRY_PENDING)` | `test_exactly_one_concurrent_claim_wins` |
| Cancel and publish cannot both win | Both take the task row lock | `test_cancel_vs_publish_race_is_consistent` |
| A repeated idempotency key never creates a second batch | `UNIQUE(user_id, idempotency_key)` with `ON CONFLICT DO NOTHING` | `test_concurrent_identical_submissions_create_one_batch` |
| Illegal state transitions are impossible | PL/pgSQL trigger mirrors `domain/states.py` | `test_task_transition_table_matches_domain` |
| A retry cannot silently change a result | Content-addressed artifacts; late checksums stored and compared; `checksum_mismatch` flag | verifier check `retries_consistent` |
| A lost wake-up message strands nothing | The database is the queue; unclaimed tasks are re-dispatched | `test_dispatch_respects_window_and_redelivers_lost_messages` |
| Clock skew between hosts is harmless | Lease arithmetic uses PostgreSQL `now()` only | |
| Results are reproducible | Canonical spec hashing; dataset checksums; deterministic engine | engine determinism tests; `quantgrid verify --recompute` |

## Bugs found by the tests

- Explicit `FOR UPDATE` on the batch row deadlocked cancel against publish, because foreign-key checks take `FOR KEY SHARE` on it. All explicit locks are `FOR NO KEY UPDATE`, taken batch first, then task.
- `retry-failed` re-queued permanent failures because they shared an error code with retryable ones. They now use `PERMANENT_TASK_ERROR`.
- The engine trusted a dataset's checksum label without hashing the arrays. It now recomputes the hash on every run.
- A string `lookback` such as `"2500"` skipped the test-segment length check. The policy now coerces it the way the engine does.
- Numeric split timestamps were accepted as Unix seconds. They are now rejected.
- `is_sha256_hex` accepted a trailing newline.

## Known limits

- A worker stuck inside native code cannot be interrupted. The heartbeat stops extending its lease after `max_runtime_s + lease_ttl_s`, so the task is reclaimed, but the process keeps running until something kills it.
- One coordinator is active at a time. Failover takes up to about 2 s plus one tick.
- Bitwise reproducibility holds for the same code, dependencies and CPU architecture only.
