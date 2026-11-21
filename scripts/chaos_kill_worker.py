#!/usr/bin/env python
"""Chaos test: kill or freeze a worker mid-batch and check the batch recovers without duplicates.

Starts a coordinator and --workers worker processes, submits a 144-task grid, and once the
victim worker is running a task:
* --signal KILL: SIGKILL the worker.
* --signal STOP: SIGSTOP it for --stop-seconds (default 2 x lease TTL), then SIGCONT.

After the batch settles it prints recovery latencies (database clock) and the batch
verifier summary, and writes a JSON report (--out, default benchmarks/results/).
Exit code is 0 only if the verifier finds exactly one published result per task.

Run: scripts/dev_services.sh start && eval "$(scripts/dev_services.sh env)"
     .venv/bin/alembic upgrade head
     .venv/bin/python scripts/chaos_kill_worker.py --workers 4 --signal KILL --restart
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantgrid.config import get_settings  # noqa: E402
from quantgrid.devstack import LocalStack  # noqa: E402
from quantgrid.domain.datasets import get_catalog  # noqa: E402
from quantgrid.domain.policy import PolicyService, QuotaPolicy  # noqa: E402
from quantgrid.domain.specs import DatasetRef, ExperimentBatchSpec, StrategyRef  # noqa: E402
from quantgrid.manifest import build_manifest, code_digest, git_info  # noqa: E402
from quantgrid.results.verify import verify_batch  # noqa: E402
from quantgrid.storage import repository as repo  # noqa: E402
from quantgrid.storage.artifacts import LocalArtifactStore  # noqa: E402
from quantgrid.storage.db import make_engine  # noqa: E402

DEMO_GRID = {
    "lookback": [20, 30, 40, 60, 80, 120],
    "entry_z": [1.5, 2.0, 2.5, 3.0],
    "exit_z": [0.0, 0.5, 1.0],
}
DEMO_SEEDS = [7, 11]  # 6 * 4 * 3 * 2 = 144 tasks


def build_spec(args: argparse.Namespace) -> ExperimentBatchSpec:
    info = get_catalog().get(args.dataset)
    if info is None:
        raise SystemExit(f"unknown dataset {args.dataset}")
    return ExperimentBatchSpec(
        name=f"chaos-{args.signal.lower()}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
        description="chaos run: worker failure injected mid-batch",
        dataset=DatasetRef(name=info.name, sha256=info.sha256),
        strategy=StrategyRef(name="pairs_zscore", version="1"),
        grid=DEMO_GRID,
        seeds=DEMO_SEEDS,
        workload={"bootstrap_samples": args.bootstrap, "bootstrap_block": 20},
        budget={"max_runtime_s": 300, "max_memory_mb": 1024},
        max_attempts=5,
        cache_policy="bypass",
        tags={"kind": "chaos"},
    )


def submit(engine: Engine, spec: ExperimentBatchSpec, quota: QuotaPolicy) -> uuid.UUID:
    decision, admitted = PolicyService(get_catalog()).admit(spec, quota)
    if admitted is None:
        raise SystemExit(f"policy rejected the chaos spec: {decision.violations}")
    res = repo.create_batch(
        engine,
        user_id="chaos",
        idempotency_key=f"chaos-{uuid.uuid4().hex}",
        spec=spec,
        tasks_expanded=admitted.tasks,
        manifest=build_manifest(admitted.dataset, spec.spec_sha()),
        quota=quota,
        dataset=admitted.dataset,
    )
    return res.batch_id


def db_now(engine: Engine) -> datetime:
    with engine.connect() as conn:
        value: datetime = conn.execute(text("SELECT now()")).scalar_one()
    return value


def running(engine: Engine, batch_id: uuid.UUID) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, lease_owner FROM tasks WHERE batch_id = :b AND state = 'RUNNING'"),
            {"b": batch_id},
        ).all()
    return [(str(r.id), str(r.lease_owner)) for r in rows]


def recovery_latencies(
    engine: Engine, batch_id: uuid.UUID, victim: str, t_kill: datetime
) -> list[dict[str, Any]]:
    """For each attempt the victim held at kill time: reclaim / restart / publish delays."""
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT a.task_id, a.attempt, a.ended_at AS reclaimed_at,
                       n.started_at AS next_started_at, t.finished_at, t.state, t.attempts
                FROM task_attempts a
                JOIN tasks t ON t.id = a.task_id
                LEFT JOIN task_attempts n ON n.task_id = a.task_id AND n.attempt = a.attempt + 1
                WHERE a.batch_id = :b AND a.worker_id = :w AND a.status = 'LEASE_EXPIRED'
                ORDER BY a.ended_at
                """
                ),
                {"b": batch_id, "w": victim},
            )
            .mappings()
            .all()
        )

    def delta(ts: datetime | None) -> float | None:
        return None if ts is None else round((ts - t_kill).total_seconds(), 3)

    return [
        {
            "task_id": str(r["task_id"]),
            "attempt": r["attempt"],
            "reclaim_s": delta(r["reclaimed_at"]),
            "restart_s": delta(r["next_started_at"]),
            "published_s": delta(r["finished_at"]) if r["state"] == "SUCCEEDED" else None,
            "final_state": r["state"],
            "final_attempts": r["attempts"],
        }
        for r in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--signal", choices=["KILL", "STOP"], default="KILL")
    ap.add_argument(
        "--stop-seconds", type=float, default=None, help="SIGSTOP duration (default: 2 x lease TTL)"
    )
    ap.add_argument(
        "--restart", action="store_true", help="start a replacement worker after the kill"
    )
    ap.add_argument("--dataset", default="pairs-ou-small-v1")
    ap.add_argument(
        "--bootstrap", type=int, default=3000, help="bootstrap samples per task (CPU load)"
    )
    ap.add_argument(
        "--min-completed",
        type=int,
        default=5,
        help="inject the failure after this many tasks completed",
    )
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = ap.parse_args()

    settings = get_settings()
    engine = make_engine(pool_size=4, application_name="qg-chaos")
    quota = QuotaPolicy.from_settings(settings)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "var" / "runs" / f"chaos-{run_id}"
    env = {"QG_QUEUE_NAME": f"qg.chaos.{uuid.uuid4().hex[:6]}"}
    stop_seconds = args.stop_seconds or 2 * settings.lease_ttl_s

    print(
        f"[chaos] lease_ttl={settings.lease_ttl_s}s heartbeat={settings.heartbeat_interval_s}s "
        f"workers={args.workers} signal={args.signal} logs={run_dir}"
    )
    with LocalStack(log_dir=run_dir / "logs", env=env) as stack:
        stack.start_coordinator()
        ids = [f"chaos-w{i}" for i in range(args.workers)]
        for wid in ids:
            stack.start_worker(wid)
        stack.wait_for_workers(engine, ids, timeout=90)
        spec = build_spec(args)
        t_submit = time.monotonic()
        batch_id = submit(engine, spec, quota)
        print(f"[chaos] submitted batch {batch_id} with {spec.task_count()} tasks")

        deadline = time.monotonic() + args.timeout
        victim = victim_task = None
        while time.monotonic() < deadline:
            done = repo.batch_progress(engine, batch_id)["counts"]["SUCCEEDED"]
            run = running(engine, batch_id)
            if done >= args.min_completed and run:
                victim_task, victim = run[0]
                break
            time.sleep(0.1)
        if victim is None:
            raise SystemExit("no running task found to interrupt")
        held = [t for t, w in running(engine, batch_id) if w == victim]
        proc = stack.processes[victim]
        if args.signal == "KILL":
            proc.kill()
        else:
            proc.send(signal.SIGSTOP)
        t_kill = db_now(engine)
        print(
            f"[chaos] {args.signal} {victim} (pid {proc.pid}) while running {held} at {t_kill.isoformat()}"
        )
        if args.signal == "STOP":
            time.sleep(stop_seconds)
            proc.send(signal.SIGCONT)
            print(f"[chaos] SIGCONT {victim} after {stop_seconds:.1f}s (worker resumes)")
        if args.restart and args.signal == "KILL":
            stack.start_worker(f"{victim}-replacement")
            print(f"[chaos] started replacement worker {victim}-replacement")

        final = stack.wait_for_batch(
            engine, batch_id, timeout=max(1.0, deadline - time.monotonic())
        )
        elapsed = time.monotonic() - t_submit
        # a resumed worker records its late finish
        time.sleep(settings.heartbeat_interval_s * 2)

    latencies = recovery_latencies(engine, batch_id, victim, t_kill)
    report = verify_batch(engine, LocalArtifactStore.from_settings(settings), batch_id)
    print(report.summary())
    for lat in latencies:
        print(
            f"[chaos] recovery task {lat['task_id'][:8]}: lease reclaimed +{lat['reclaim_s']}s, "
            f"re-started +{lat['restart_s']}s, published +{lat['published_s']}s"
        )
    out = {
        "run_id": run_id,
        "batch_id": str(batch_id),
        "final_state": final,
        "signal": args.signal,
        "victim": victim,
        "victim_tasks_at_kill": held,
        "victim_example_task": victim_task,
        "stop_seconds": stop_seconds if args.signal == "STOP" else None,
        "workers": args.workers,
        "restart": args.restart,
        "elapsed_s": round(elapsed, 3),
        "lease_ttl_s": settings.lease_ttl_s,
        "heartbeat_interval_s": settings.heartbeat_interval_s,
        "coordinator_tick_s": settings.coordinator_tick_s,
        "recovery": latencies,
        "verification": report.to_dict(),
        "environment": {
            "code_sha": code_digest(),
            "git": git_info(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
        },
    }
    path = (
        args.out or ROOT / "benchmarks" / "results" / f"chaos-{args.signal.lower()}-{run_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[chaos] report written to {path}")
    engine.dispose()
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
