# ADR 0005 — Bounded dispatch window instead of enqueueing the whole batch

**Status:** accepted

## Decision
The coordinator keeps at most `max(QG_DISPATCH_MIN_WINDOW, live_worker_slots ×
QG_DISPATCH_WINDOW_FACTOR)` unclaimed wake-up messages outstanding, choosing tasks by
`priority DESC, ready_at, batch, ordinal` with `FOR NO KEY UPDATE SKIP LOCKED`. It wakes on
PostgreSQL `NOTIFY` (task finished / batch submitted) or every `QG_COORDINATOR_TICK_S`.

## Rationale
* Queue memory stays bounded for 5,000-task batches; cancellation takes effect immediately
  for undispatched work; later batches are not starved behind a huge earlier batch.
* "Queue wait" is defined as claim time − time the task became eligible, which includes
  time spent waiting in the database — the user-visible wait.

## Consequences
* Each completion → NOTIFY → dispatch → claim adds scheduler latency; benchmarks report it
  as overhead, especially visible with very short tasks.
