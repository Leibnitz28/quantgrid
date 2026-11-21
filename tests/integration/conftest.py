"""Integration-test fixtures: a throwaway PostgreSQL database per pytest session.

Each session creates ``qg_it_<random>``, applies the migrations and drops it afterwards.
Tests are skipped when no PostgreSQL is reachable (``scripts/dev_services.sh start`` or
set ``QG_TEST_DATABASE_URL``).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, make_url, text
from sqlalchemy.exc import OperationalError

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = os.environ.get(
    "QG_TEST_DATABASE_URL",
    "postgresql+psycopg://quantgrid:quantgrid@127.0.0.1:5433/quantgrid_test",
)
TABLES = (
    "task_transitions",
    "result_artifacts",
    "task_attempts",
    "tasks",
    "worker_heartbeats",
    "batches",
)


def _admin_engine() -> Engine:
    return create_engine(BASE_URL, isolation_level="AUTOCOMMIT", pool_pre_ping=True)


@pytest.fixture(scope="session")
def db_url() -> Iterator[str]:
    admin = _admin_engine()
    try:
        with admin.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:  # pragma: no cover - environment dependent
        admin.dispose()
        pytest.skip(f"PostgreSQL not reachable at {BASE_URL}: {exc.__class__.__name__}")
    name = f"qg_it_{uuid.uuid4().hex[:10]}"
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(BASE_URL).set(database=name).render_as_string(hide_password=False)

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["dburl"] = url
    command.upgrade(cfg, "head")
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="session")
def engine(db_url: str) -> Iterator[Engine]:
    eng = create_engine(
        db_url,
        pool_size=20,
        max_overflow=20,
        connect_args={"options": "-c timezone=UTC -c statement_timeout=30000"},
    )
    yield eng
    eng.dispose()


@pytest.fixture
def clean_db(engine: Engine) -> Engine:
    """Empty all tables before each test."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))
    return engine
