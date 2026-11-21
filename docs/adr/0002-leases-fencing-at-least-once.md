# ADR 0002 — Leases with fencing tokens; at-least-once execution, at-most-once publication

**Status:** accepted

## Context
Exactly-once *execution* is not achievable with ordinary task queues: a worker can crash
after doing work but before recording it, or stall (GC pause, SIGSTOP, network partition)
long enough to be presumed dead and then resume.

## Decision
* A claim sets `state=RUNNING, lease_owner=<worker>, lease_expiry=now()+TTL` and increments
  `attempts`. The pair `(lease_owner, attempts)` is the **fencing token**.
* Workers extend leases from a heartbeat thread (`QG_HEARTBEAT_INTERVAL_S`, default 3 s,
  TTL 15 s). Every worker write (heartbeat, publish, failure) is conditional on the fence.
* Only the coordinator's reconciler takes leases away: expired RUNNING tasks become
  RETRY_PENDING (or FAILED when attempts are exhausted, CANCELED when cancel is pending),
  closing the attempt as `LEASE_EXPIRED / WORKER_LOST`.
* A resumed "zombie" sees a changed fence: it cannot publish; its output checksum is kept
  on its (closed) attempt row and compared with the published one.
* All lease arithmetic uses the database clock.

## Consequences
* Work may execute more than once (at-least-once). The *effect* — one published result
  per task — happens at most once, and exactly once for tasks that succeed.
* Recovery latency after a crash ≈ remaining TTL + reconcile tick + dispatch + queue wait
  (measured by `scripts/chaos_kill_worker.py`).
* We never claim exactly-once execution.
