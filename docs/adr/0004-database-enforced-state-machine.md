# ADR 0004 — The database enforces the state machines and logs every transition

**Status:** accepted

## Decision
Migration `0001` installs PL/pgSQL triggers that:
* reject task transitions not in the table of `quantgrid/domain/states.py`
  (e.g. `PENDING -> SUCCEEDED`), and require tasks to be created PENDING;
* make task specs, batch specs/manifests, terminal attempts and published results immutable;
* append every task state change to `task_transitions` with the acting component
  (`api`, `worker:<id>`, `coordinator`) supplied via a transaction-local GUC.
A check constraint ties `state = RUNNING` to the presence of a lease.

## Consequences
* Application bugs cannot silently corrupt state; they fail loudly.
* An integration test asserts the SQL and Python transition tables are identical.
* The audit log lets the verifier prove that every observed transition was legal.
