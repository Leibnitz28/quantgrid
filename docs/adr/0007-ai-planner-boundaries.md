# ADR 0007 — The AI planner proposes typed specs; deterministic code decides

**Status:** accepted

## Decision
* A new, minimal Python orchestrator (explicit tool registry with per-role allowlists,
  strict Pydantic argument schemas, max steps, tool-call budget, wall-clock timeout,
  traced execution) — no reuse of earlier agent code.
* Tools are read-only (catalog, schema, quota, cost estimate, dry-run validation, batch
  progress). There is no submit/cancel/delete/shell/code tool.
* The planner's output must parse as a strict `ExperimentBatchSpec` and pass the same
  `PolicyService` the API uses. `POST /v1/planner/specs` never enqueues; a human submits.
* `FakeLLM` (deterministic) is the default provider and is used by CI; the optional Claude
  provider is enabled only via environment (`QG_LLM_PROVIDER=anthropic`).

## Consequences
* A fully compromised model (prompt injection through a goal or tool output) can at worst
  propose a spec that the policy service rejects — covered by adversarial tests.
