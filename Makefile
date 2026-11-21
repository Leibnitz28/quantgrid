.PHONY: install services migrate test test-fast check fmt demo chaos bench clean

PY := .venv/bin/python

install:
	uv sync --all-extras

services:
	scripts/dev_services.sh start

migrate:
	.venv/bin/alembic upgrade head

test:
	.venv/bin/pytest

test-fast:
	.venv/bin/pytest -m "not integration"

check:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .
	.venv/bin/mypy src/quantgrid

fmt:
	.venv/bin/ruff format .
	.venv/bin/ruff check --fix .

demo:
	scripts/demo.sh

chaos:
	$(PY) scripts/chaos_kill_worker.py --workers 4 --signal KILL --restart
	$(PY) scripts/chaos_kill_worker.py --workers 4 --signal STOP

bench:
	$(PY) scripts/benchmark.py --workers 1,2,4 --reps 3

clean:
	rm -rf var .pytest_cache .mypy_cache .ruff_cache .hypothesis
