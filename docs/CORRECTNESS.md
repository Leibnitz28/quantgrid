# Correctness notes

What the scheduler guarantees, why, and which test would catch a violation.
Execution is **at-least-once**; the effect (one published result per task) is **at-most-once**,
and exactly-once for every task that succeeds. We do not claim exactly-once execution.

| # | Invariant | Mechanism | Falsified by |
|---|---|---|---|
| 1 | A task has at most one published result | `result_artifacts.task_id UNIQUE`; the insert and `-> SUCCEEDED` happen in one transaction holding the task row lock | `test_publish_once_and_transitions_logged`, verifier check `one_result_per_success` |
| 2 | A worker that lost its lease cannot publish | fencing token `(lease_owner, attempts)` re-checked under the row lock in `finalize_success` | `test_zombie_worker_cannot_publish_after_reassignment`, e2e SIGSTOP test |
| 3 | Two workers cannot claim one task | conditional `UPDATE ... WHERE state IN (PENDING, RETRY_PENDING)` | `test_exactly_one_concurrent_claim_wins` (10 threads) |
| 4 | Cancel and publish are mutually exclusive | both take the task row lock (`FOR NO KEY UPDATE`); a task is reported completed by cancel or can never publish | `test_cancel_vs_publish_race_is_consistent` |
| 5 | Re-submitting an idempotency key never duplicates a batch | `UNIQUE(user_id, idempotency_key)` + `ON CONFLICT DO NOTHING` | `test_concurrent_identical_submissions_create_one_batch` |
| 6 | Illegal state transitions are impossible | PL/pgSQL trigger mirrors `domain/states.py`; both tables compared in a test | `test_task_transition_table_matches_domain` |
| 7 | A retry cannot silently change a result | artifacts are content-addressed; late/zombie checksums are stored and compared; `checksum_mismatch` is flagged | verifier check `retries_consistent` |
| 8 | Lost wake-up messages never strand work | DB is the queue; unclaimed dispatched tasks are re-dispatched after `QG_DISPATCH_LOST_AFTER_S` | `test_dispatch_respects_window_and_redelivers_lost_messages` |
| 9 | Lease arithmetic is immune to worker clock skew | all lease times use PostgreSQL `now()` | by construction |
| 10 | Results are reproducible | specs hashed canonically; datasets verified by logical SHA-256; bit-identical reruns | engine determinism tests, `quantgrid verify --recompute` |

## Races found and fixed while building it
* **Deadlock between cancel and publish.** Explicit `FOR UPDATE` on the batch row conflicts with the
  `FOR KEY SHARE` taken by foreign-key checks on `result_artifacts`; the race test deadlocked.
  All explicit locks are now `FOR NO KEY UPDATE`, ordered batch -> task.
* **`retry-failed` re-queued permanent failures** because permanent errors shared the `TASK_ERROR`
  code with retryable ones. They now carry `PERMANENT_TASK_ERROR`.
* **Engine trusted a dataset's hash label** without hashing the arrays; it now recomputes the content
  hash on every run.

## Honest limits
* A worker that hangs *inside native code* cannot be interrupted cooperatively; the heartbeat watchdog
  stops extending its lease after `max_runtime_s + lease_ttl_s`, so the coordinator reclaims the task,
  but the stuck process keeps its CPU until the supervisor kills it.
* One coordinator is active at a time (PostgreSQL advisory lock). Failover takes up to ~2 s plus a tick.
* Bitwise reproducibility holds for identical code + dependencies + CPU architecture only.
