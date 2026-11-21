# Security

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability") with the steps to reproduce.

## Scope and threat model

QuantGrid is offline research infrastructure. It has no broker or exchange connectivity and places no orders.

What the code defends against:

- **Malformed or hostile requests.** All input is validated by typed schemas that reject unknown fields. Bodies are capped (`QG_MAX_REQUEST_BYTES`), requests are rate limited per user and per endpoint, and grids are bounded before they are expanded.
- **Resource abuse.** Every batch passes the quota policy (tasks, runtime, memory, worker-seconds, active batches, bootstrap samples, dataset allow-list) before anything is stored. The policy runs again on submission even when a spec came from the AI planner.
- **LLM output and injected text.** The planner has seven read-only tools, no submit, cancel, delete, shell or file tool, and its output must parse as a strict spec and pass the same policy. Tool results and the user's goal are treated as data.
- **Cross-user access.** Batches, tasks and results are scoped to the authenticated user; other users get `404`.
- **Path traversal in artifacts.** Artifact URIs must match `artifact://sha256/<aa>/<64 hex>.parquet` exactly.
- **Result tampering.** Published results are immutable (database triggers), content-addressed, and re-hashed on read.

What it does not do:

- Authentication is static API keys from `QG_API_KEYS`. There is no key rotation, scoping or expiry.
- Traffic is plain HTTP. Put it behind a TLS-terminating proxy.
- PostgreSQL and Redis are trusted and unauthenticated in the local setup. Set passwords and network rules before exposing them.
- The default dev key `dev-key` is for local use only. Change `QG_API_KEYS` in any shared environment.
- `QG_ALLOW_FAULT_INJECTION` lets a request make tasks fail on purpose. Leave it `false` outside tests and demos.
- The optional Claude provider sends planner prompts and tool results to the Anthropic API when `QG_LLM_PROVIDER=anthropic`. The default provider is local and sends nothing.

See [docs/limitations.md](docs/limitations.md) for the full list of omitted hardening.
