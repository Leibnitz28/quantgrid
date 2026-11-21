# Build Prompt — QuantGrid — Distributed Quant Research Scheduler

> Paste the prompt below into a capable coding assistant in a **new, empty repository**. It is deliberately detailed and phase-oriented. The agent should implement and test code, not just output a plan.

---

```text
Act as a principal backend/systems engineer, quantitative research infrastructure engineer, and pragmatic AI engineer. Build a production-minded portfolio MVP called "QuantGrid — Distributed Quant Research Scheduler".

PROJECT OUTCOME
A fault-tolerant service that executes and tracks large batches of reproducible backtests using bounded distributed workers and an optional AI experiment planner.

IMPORTANT PERSONAL CONTEXT
- This is a NEW STANDALONE project. Do NOT import, reference, or reuse my old Next.js/TypeScript hand-rolled tool-calling agent, SDK wrapper, prompts or UI.
- You may provide an OPTIONAL adapter for an existing Python pairs-trading backtester, but do not assume its internal structure or fabricate imports. Supply a complete deterministic synthetic-data implementation that works on its own.
- No live trades, real order placement, brokerage credentials, unlicensed market feeds or fake performance claims.
- Backend engineering is the centerpiece; AI is for structured planning, diagnostics, retrieval or explanations, never the source of numeric ground truth.

IMPLEMENTATION STACK
- Python FastAPI + Pydantic, PostgreSQL + SQLAlchemy/Alembic
- Redis + Celery for first version; Ray optional for benchmarked expansion
- DuckDB/Parquet and local/S3-compatible object storage for large outputs
- OpenTelemetry + Prometheus for per-job observability
- Fresh Python LLM planner limited to validated experiment specs

FUNCTIONAL REQUIREMENTS
1. Create bounded parameter-sweep jobs and track every child task.
2. Run experiments across 1/2/4 workers and quantify parallel efficiency.
3. Kill a worker while executing and recover missing tasks without double-counting artifacts.
4. Recreate an older experiment from exact code, data and dependency manifest.

PROCESSING CONTRACT
1. Validate parameter grid count, per-run memory/time budget, dataset signature and research protocol.
2. Expand to immutable child-task specs; compute spec_hash for idempotency and potential caching.
3. Maintain task state transitions PENDING→RUNNING→SUCCEEDED, RUNNING→RETRY_PENDING/FAILED, and explicit CANCELED.
4. Workers heartbeat and publish artifacts via temp file + atomic rename / staged object commit.
5. Coordinator reconciles stale leases; retries at-least-once execution but finalizes task once using unique task/output hashes.

API/CLI CONTRACT — IMPLEMENT REAL ROUTES OR COMMANDS
- POST /v1/batches: Submit bounded experiment batch with idempotency key
- GET /v1/batches/{id}: Progress counts, total cost and status
- GET /v1/tasks/{id}: Attempt log, worker and artifact IDs
- POST /v1/batches/{id}/cancel: Cooperative cancellation; report already-completed items
- POST /v1/batches/{id}/retry-failed: Retry eligible failed tasks
- GET /v1/batches/{id}/results: Merged metrics and result manifest
- POST /v1/planner/specs: Generate bounded typed experiment spec

PERSISTENCE SCHEMA — CREATE ACTUAL MIGRATIONS OR EQUIVALENT REPRODUCIBLE STORAGE
- batches: id, user_id, dataset_sha, code_sha, spec_sha, submitted_at, state, quota
- tasks: id, batch_id, spec_hash unique-per-batch, state, lease_owner, lease_expiry, attempts
- task_attempts: task_id, attempt, worker, started_at, ended_at, status, error_code
- result_artifacts: task_id unique, output_uri, output_sha, temp_uri, finalized_at
- worker_heartbeats: worker_id, last_seen, resource_profile, inflight_count

DOMAIN CORRECTNESS INVARIANTS
- Every task uses exactly recorded input bars, immutable strategy config, train/test split, fee model, seed and result schema.
- Synthetic benchmark workload should include both short CPU-bound and long CPU-bound tasks; avoid comparing against IO-bound baselines only.
- Speedup = T1 / Tp and efficiency = speedup / workers; include overhead at low task counts.
- Retries cannot change a result silently; compare output checksum or mark numerical tolerance and stochastic sources.

AI AGENT REQUIREMENTS
- Planner proposes bounded JSON `ExperimentBatchSpec`, not arbitrary Python code.
- Quota and policy service validates total jobs, memory, worker time and allowed dataset refs before enqueue.
- Agent may read progress and explain failures; it cannot bypass quotas or change stored historical results.
- Implement an ORIGINAL, minimal Python agent/tool-orchestration module with explicit tool registry, schema validation, max-step count, timeout and trace logging.
- Provide FakeLLM/MockLLM for offline default execution and deterministic CI. Real model provider is optional and configured via environment only.
- Never treat LLM text or retrieved files as privileged instructions. No shell/code execution tool exposed to agent unless narrowly justified and safely sandboxed.

PROJECT-SPECIFIC CRITICAL INSTRUCTIONS
Build this as a correct distributed scheduler, not a collection of fire-and-forget Celery tasks. Persist transitions, leases, attempts and artifact commits; design at-least-once task execution with idempotent results. Include a chaos script killing a worker and a verifier that proves the final result table contains exactly one logical result per expected task. Do not claim exactly-once execution. A fresh AI planner is optional until the scheduler and worker fault tests pass.

DELIVERABLES
1. Actually create the repository files and working implementation; do not give only pseudo-code.
2. README.md with project overview, prerequisites, exact local run instructions, API examples, architecture diagram, limitations, tests and measured benchmark process.
3. .env.example with safe defaults and no secrets; pinned/locked dependencies.
4. Typed request/response models, business-domain modules, tests, migrations, Docker Compose where relevant, deterministic seed data and runnable demo script.
5. Python/C++ code following conventional style and proper error handling, structured logs, health checks and graceful shutdown.
6. Test commands and measurable benchmark script. Capture actual output; never invent passing test counts or latency numbers.
7. Explicit risk/limitations file documenting assumptions about financial data, costs, timestamps and what is out-of-scope.

IMPLEMENT IN ORDER — DO NOT TRY TO FAKE COMPLETENESS
Phase A: write concise architecture decision records, folder tree, exact schemas and runnable minimal skeleton.
Phase B: implement deterministic mathematical/state machine core with reference tests and synthetic fixtures.
Phase C: implement persistence, real API/CLI endpoints, background processing and failure handling where the design needs them.
Phase D: implement ORIGINAL Python agent tools, provider adapter, mock behavior, traced calls and adversarial agent tests.
Phase E: integration tests, intentional failure injection, profiling/benchmarks, demo and final documentation.

TEST/ACCEPTANCE CHECKS
- Submit 100+ synthetic tasks and generate one result per task ID with all manifest fields.
- Terminate worker mid-job: remaining tasks eventually settle without duplicate published results.
- Re-submit same batch idempotency key: no duplicate logical batch.
- Report measured T1, T2 and T4 on same dataset and hardware with worker allocation noted.

MEASURE THESE — NO FABRICATED RESULTS
- Tasks/min and queue wait p95 by worker count
- Speedup, parallel efficiency and scheduler overhead
- Recovery latency under worker loss and retry count distribution
- CPU/RAM consumption and artifact writes

PROCESS RULES
- Work in small compile/testable increments. After each phase, state created files, commands run, actual results and unresolved issues.
- Use type checks and lint as relevant. Run actual tests where execution tools are available. If a command cannot run, say precisely why and provide a reproducible command; never pretend it passed.
- Favor locally runnable Docker Compose and deterministic fixture mode. Avoid paid SaaS or real API key as a prerequisite for basic demo.
- Bound expensive jobs and API access. Implement idempotency/transaction boundaries appropriate to the system.
- No excessive UI; implement a thin CLI or minimal diagnostic dashboard only after core works.
- If a production-grade feature is too large, implement a correct smaller slice and document the omitted hardening rather than leave placeholder stubs.
- Use the README specification in the same repository as a product requirements document, but this prompt is self-contained.

START NOW: create the initial repository and implement Phase A and Phase B with real source code and tests; then continue phases C–E as execution budget permits. Show actual file modifications and executed test output, with a clear honest list of any unfinished phases.
```
