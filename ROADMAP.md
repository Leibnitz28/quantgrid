# Roadmap

Planned work, in rough priority order. Nothing here is built.

## 1. Deployment
- Container image published from CI, and a Helm chart or equivalent only once there is a real deployment to run.
- Per-service health and readiness checks wired into the Compose file for all roles.

## 2. Scheduler
- Per-user fair-share dispatch so one large batch cannot delay another user's small one.
- Task-level timeouts enforced by a supervisor that can kill a worker stuck in native code.
- Lease TTL that adapts to measured heartbeat jitter.

## 3. Storage
- S3-compatible artifact store behind the existing `ArtifactStore` protocol.
- Garbage collection for orphaned artifacts, with a dry-run report.

## 4. Research workflow
- External strategy adapter, registered through `register_strategy` and the engine registry.
- Multiple-testing correction across a parameter grid in the results summary.
- Real, licensed datasets with a documented provenance record per dataset.

## 5. Security
- Per-user API keys with rotation and scopes, TLS in the reference deployment, and an authenticated PostgreSQL and Redis setup.
