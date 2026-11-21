# ADR 0001 — PostgreSQL is the source of truth; Celery/Redis is only a transport

**Status:** accepted

## Context
The product must track every child task of a parameter sweep, survive worker crashes and
recover without double-counting results. Celery offers retries and `acks_late`, but its
state lives in the broker/result backend, which is neither queryable for progress nor
transactional with our results. "Fire-and-forget Celery tasks" cannot prove that each
logical task produced exactly one published result.

## Decision
* Every batch, task, attempt, lease, transition and published result lives in PostgreSQL.
* A Celery message is a **wake-up token** carrying only a `task_id`. Workers *claim* the
  task with a conditional `UPDATE ... WHERE state IN ('PENDING','RETRY_PENDING')`; a
  duplicate or stale message simply fails to claim and is acknowledged.
* Lost messages are tolerated: the coordinator re-dispatches tasks that stay unclaimed
  for `QG_DISPATCH_LOST_AFTER_S`.
* Celery runs with `acks_late`, `reject_on_worker_lost`, prefetch 1 and no result backend;
  recovery of crashed work never depends on broker redelivery (whose Redis visibility
  timeout is far too long to rely on).

## Consequences
* Progress, costs and attempt history are SQL queries; the verifier can prove invariants.
* One extra DB round-trip per task (claim) — measured in `benchmarks/` as scheduler overhead.
* Redis can be flushed without losing work (tasks are re-dispatched from the DB).
