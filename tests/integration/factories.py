"""Small builders shared by integration tests (no I/O beyond the given engine)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine

from quantgrid.domain.hashing import sha256_hex
from quantgrid.domain.policy import QuotaPolicy
from quantgrid.domain.specs import (
    DatasetInfo,
    DatasetRef,
    ExperimentBatchSpec,
    StrategyRef,
    expand_grid,
)
from quantgrid.manifest import build_manifest
from quantgrid.storage import repository as repo

FAKE_SHA = "a" * 64
FAKE_DATASET = DatasetInfo(
    name="pairs-test-v1",
    sha256=FAKE_SHA,
    n_bars=2_000,
    start=datetime(2015, 1, 1, tzinfo=UTC),
    end=datetime(2020, 6, 22, tzinfo=UTC),
    bar_interval_s=86_400,
    columns=("ts", "close_x", "close_y"),
    synthetic=True,
    license="synthetic",
    description="fake dataset for repository tests",
)


def make_spec(
    n_tasks: int = 4, *, max_attempts: int = 3, name: str = "it-batch", **overrides: Any
) -> ExperimentBatchSpec:
    data: dict[str, Any] = {
        "name": name,
        "dataset": DatasetRef(name=FAKE_DATASET.name, sha256=FAKE_SHA),
        "strategy": StrategyRef(name="pairs_zscore", version="1"),
        "grid": {"lookback": [20 + i for i in range(n_tasks)]},
        "fixed_params": {"entry_z": 2.0, "exit_z": 0.5},
        "max_attempts": max_attempts,
    }
    data.update(overrides)
    return ExperimentBatchSpec(**data)


def submit(
    engine: Engine,
    n_tasks: int = 4,
    *,
    user_id: str = "user-1",
    key: str | None = None,
    spec: ExperimentBatchSpec | None = None,
    quota: QuotaPolicy | None = None,
    **spec_overrides: Any,
) -> repo.CreateBatchResult:
    spec = spec or make_spec(n_tasks, **spec_overrides)
    return repo.create_batch(
        engine,
        user_id=user_id,
        idempotency_key=key or uuid.uuid4().hex,
        spec=spec,
        tasks_expanded=expand_grid(spec),
        manifest=build_manifest(FAKE_DATASET, spec.spec_sha()),
        quota=quota or QuotaPolicy(),
        dataset=FAKE_DATASET,
    )


def artifact(tag: str = "x", *, sha: str | None = None) -> repo.ArtifactRecord:
    digest = sha or sha256_hex(tag)
    return repo.ArtifactRecord(
        output_uri=f"artifact://sha256/{digest[:2]}/{digest}.parquet",
        output_sha=digest,
        size_bytes=123,
        row_count=10,
        metrics={"sharpe": 1.0, "n_trades": 3},
        temp_uri=f"tmp/{tag}.partial",
    )


def task_ids(engine: Engine, batch_id: uuid.UUID) -> list[uuid.UUID]:
    return [t["id"] for t in repo.list_tasks(engine, batch_id, limit=10_000)]
