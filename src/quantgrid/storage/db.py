"""Database engine and transaction helpers.

State changes run in short READ COMMITTED transactions using conditional UPDATEs and row
locks. Lease times use the database clock (``now()``), never a worker's.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Connection, Engine, create_engine, text

from quantgrid.config import get_settings

NOTIFY_CHANNEL = "quantgrid_events"


def make_engine(
    url: str | None = None,
    *,
    pool_size: int | None = None,
    application_name: str = "quantgrid",
    statement_timeout_ms: int | None = None,
) -> Engine:
    settings = get_settings()
    timeout = statement_timeout_ms or settings.db_statement_timeout_ms
    return create_engine(
        url or settings.database_url,
        pool_size=pool_size or settings.db_pool_size,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={
            "application_name": application_name[:63],
            "options": f"-c statement_timeout={int(timeout)} -c timezone=UTC",
        },
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


def dispose_engine() -> None:
    """Dispose the cached engine (after fork, on shutdown, in tests)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
        get_engine.cache_clear()


@contextmanager
def transaction(engine: Engine | None = None) -> Iterator[Connection]:
    """One transaction; rolls back on exception."""
    with (engine or get_engine()).begin() as conn:
        yield conn


def set_actor(conn: Connection, actor: str, reason: str | None = None) -> None:
    """Set the actor recorded by the task-transition audit trigger."""
    conn.execute(
        text(
            "SELECT set_config('quantgrid.actor', :actor, true), "
            "set_config('quantgrid.reason', :reason, true)"
        ),
        {"actor": actor[:160], "reason": (reason or "")[:500]},
    )


def set_reason(conn: Connection, reason: str | None) -> None:
    conn.execute(
        text("SELECT set_config('quantgrid.reason', :reason, true)"),
        {"reason": (reason or "")[:500]},
    )


def notify(conn: Connection, payload: str = "tasks") -> None:
    """Wake the coordinator (delivered on COMMIT; identical payloads dedupe per transaction)."""
    conn.execute(
        text("SELECT pg_notify(:channel, :payload)"),
        {"channel": NOTIFY_CHANNEL, "payload": payload},
    )
