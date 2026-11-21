# Contributing

## Set up

Python 3.12, [uv](https://docs.astral.sh/uv/), and PostgreSQL 16 and Redis 7 binaries (or Docker).

```sh
make install     # uv sync --all-extras
make services    # native PostgreSQL + Redis on ports 5433 and 6380, no Docker
make migrate
make test        # unit, property, integration and end-to-end tests
make check       # ruff, ruff format --check, mypy --strict
```

`make test-fast` skips everything that needs PostgreSQL and Redis.

## Where things live

| Path | What it is |
| --- | --- |
| `src/quantgrid/domain` | Specs, hashing, state machines, policy, backtest engine. Pure code, no I/O. |
| `src/quantgrid/storage` | Schema, repository (the only code that changes scheduler state), artifact store |
| `src/quantgrid/api` | FastAPI application |
| `src/quantgrid/workers`, `coordinator` | Celery worker, lease heartbeat, coordinator loop |
| `src/quantgrid/agents` | Read-only AI planner and explainer |
| `src/quantgrid/results` | Result codec, DuckDB export, batch verifier |
| `migrations/` | Alembic migrations, including the state-machine triggers |
| `tests/` | `unit`, `property`, `integration`, `adversarial` |
| `benchmarks/results/` | Raw benchmark and chaos output, one directory per run |

## Making a change

- Work on a branch and open a pull request. CI runs lint, types and the full test suite.
- A change to scheduler state goes through `storage/repository.py` and gets a test that fails without it. Races get a test with real threads against PostgreSQL.
- A change to the schema is a new Alembic migration. `alembic check` must report no drift.
- A change to engine output updates `fixtures/golden/engine_golden.json` in the same commit, with the reason in the message.
- Numbers in the README come from a file under `benchmarks/results/`. Change the source, not the copy.
- Never commit secrets. Configuration comes from environment variables; keep `.env.example` current.
- Report a vulnerability privately, as [SECURITY.md](SECURITY.md) explains.
