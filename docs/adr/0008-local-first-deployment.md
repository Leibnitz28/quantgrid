# ADR 0008 — Local-first deployment: native dev services or Docker Compose; no Kubernetes

**Status:** accepted

## Decision
* `scripts/dev_services.sh` runs PostgreSQL 16 + Redis 7 natively (non-default ports) for
  development and CI without Docker; `docker-compose.yml` runs the full stack (postgres,
  redis, migrate, api, coordinator, scaled workers) in containers.
* Single coordinator elected with a PostgreSQL advisory lock; any number of workers.
* Kubernetes, service meshes and microservice splits are out of scope until profiling
  shows a need (see docs/LIMITATIONS.md for production hardening that is deliberately omitted).
