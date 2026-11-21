#!/usr/bin/env python
"""Scaling benchmark for QuantGrid: measured T1/T2/T4, speedup, efficiency, overhead.

For each worker count p (default 1, 2, 4) a FRESH local stack (coordinator + p worker
processes, one numeric thread each) is started against the configured PostgreSQL/Redis,
a warm-up batch runs, then ``--reps`` repetitions of a mixed CPU-bound workload:

* ``--long-tasks``  tasks with ``--long-bootstrap`` bootstrap resamples (long, CPU-bound)
* ``--short-tasks`` tasks without bootstrap (short; dominated by scheduler overhead)

submitted together (long batch first). Every run uses fresh seeds so every artifact is
newly computed and written (no cache, no content-address deduplication). A task-count
sweep (``--sweep-counts``) at each p shows how overhead dominates at low task counts.

Measured per run (from PostgreSQL timestamps, i.e. one clock):
  makespan T_p (submission -> last task published), tasks/min, queue wait p50/p95/p99,
  task wall time p50/p95, worker utilisation, inter-task gap per worker (scheduler
  round-trip), CPU% / RSS of worker processes (psutil sampling), artifact writes.
Derived: speedup S_p = T1/Tp, efficiency E_p = S_p/p, Karp-Flatt serial fraction.

Outputs in ``--out-dir`` (default benchmarks/results/<UTC timestamp>/): runs.csv,
attempts.csv, samples.csv, summary.json, REPORT.md, speedup.svg, efficiency_vs_tasks.svg.
Nothing is pre-filled: every number in the report comes from this run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import psutil
from sqlalchemy import Engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantgrid.config import get_settings  # noqa: E402
from quantgrid.devstack import LocalStack  # noqa: E402
from quantgrid.domain.datasets import get_catalog  # noqa: E402
from quantgrid.domain.policy import PolicyService, QuotaPolicy  # noqa: E402
from quantgrid.domain.specs import DatasetInfo, DatasetRef, ExperimentBatchSpec, StrategyRef  # noqa: E402
from quantgrid.manifest import build_manifest, code_digest, environment_manifest, git_info  # noqa: E402
from quantgrid.storage import repository as repo  # noqa: E402
from quantgrid.storage.db import make_engine  # noqa: E402


# ------------------------------------------------------------------------------ stats
def pct(values: list[float], q: float) -> float | None:
    """Percentile with linear interpolation (numpy 'linear' / Excel PERCENTILE.INC)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def r(value: float | None, nd: int = 3) -> float | None:
    return None if value is None else round(value, nd)


# ------------------------------------------------------------------------------ sampling
class Sampler(threading.Thread):
    """Samples CPU% and RSS of the given processes (and the whole host) periodically."""

    def __init__(self, pids: dict[str, int], interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.procs = {}
        for name, pid in pids.items():
            try:
                p = psutil.Process(pid)
                p.cpu_percent(None)
                self.procs[name] = p
            except psutil.Error:
                pass
        self.samples: list[dict[str, Any]] = []
        self._halt = threading.Event()
        psutil.cpu_percent(None)

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            t = time.time()
            host = psutil.cpu_percent(None)
            for name, p in self.procs.items():
                try:
                    self.samples.append(
                        {
                            "t": t,
                            "proc": name,
                            "cpu_pct": p.cpu_percent(None),
                            "rss_mb": p.memory_info().rss / 1e6,
                            "host_cpu_pct": host,
                        }
                    )
                except psutil.Error:
                    continue

    def stop(self) -> list[dict[str, Any]]:
        self._halt.set()
        self.join(5)
        return self.samples


# ------------------------------------------------------------------------------ workload
QUOTA = QuotaPolicy(
    max_tasks_per_batch=5000,
    max_worker_seconds_per_batch=10**7,
    max_active_batches=1000,
    max_bootstrap_samples=50_000,
)


def spec_for(ds: DatasetInfo, n: int, bootstrap: int, seed: int, label: str) -> ExperimentBatchSpec:
    return ExperimentBatchSpec(
        name=f"bench-{label}",
        dataset=DatasetRef(name=ds.name, sha256=ds.sha256),
        strategy=StrategyRef(name="pairs_zscore", version="1"),
        grid={"lookback": [20 + 2 * i for i in range(n)]},
        fixed_params={"entry_z": 2.0, "exit_z": 0.5},
        seeds=[seed],
        workload={"bootstrap_samples": bootstrap, "bootstrap_block": 20},
        budget={"max_runtime_s": 600, "max_memory_mb": 2048},
        max_attempts=3,
        cache_policy="bypass",
        tags={"kind": "benchmark", "label": label[:120]},
    )


def submit(engine: Engine, spec: ExperimentBatchSpec) -> uuid.UUID:
    decision, admitted = PolicyService(get_catalog()).admit(spec, QUOTA)
    if admitted is None:
        raise SystemExit(f"benchmark spec rejected: {decision.violations}")
    return repo.create_batch(
        engine,
        user_id="benchmark",
        idempotency_key=uuid.uuid4().hex,
        spec=spec,
        tasks_expanded=admitted.tasks,
        manifest=build_manifest(admitted.dataset, spec.spec_sha()),
        quota=QUOTA,
        dataset=admitted.dataset,
    ).batch_id


def db_now(engine: Engine) -> datetime:
    with engine.connect() as conn:
        value: datetime = conn.execute(text("SELECT now()")).scalar_one()
    return value


@dataclass
class RunResult:
    phase: str
    workers: int
    rep: int
    n_tasks: int
    batch_ids: list[str]
    makespan_s: float
    row: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)


def measure(
    engine: Engine,
    stack: LocalStack,
    phase: str,
    p: int,
    rep: int,
    batches: list[tuple[int, int, int, str]],
    ds: DatasetInfo,
    timeout: float,
) -> RunResult:
    """batches: (n_tasks, bootstrap, seed, label) submitted back-to-back."""
    pids = {name: proc.pid for name, proc in stack.processes.items()}
    sampler = Sampler(pids)
    sampler.start()
    t_submit = db_now(engine)
    ids = [submit(engine, spec_for(ds, n, b, seed, label)) for n, b, seed, label in batches]
    for bid in ids:
        state = stack.wait_for_batch(engine, bid, timeout=timeout, poll_s=0.1)
        if state != "COMPLETED":
            raise SystemExit(f"benchmark batch {bid} ended {state}")
    samples = sampler.stop()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT a.task_id, a.batch_id, a.attempt, a.worker_id, a.status, a.started_at,
                       a.ended_at, a.queue_wait_ms, a.wall_ms, a.cpu_ms, a.peak_rss_mb,
                       t.finished_at AS task_finished_at
                FROM task_attempts a JOIN tasks t ON t.id = a.task_id
                WHERE a.batch_id = ANY(CAST(:ids AS uuid[]))
                ORDER BY a.started_at
                """
                ),
                {"ids": ids},
            )
            .mappings()
            .all()
        )
        art = (
            conn.execute(
                text(
                    "SELECT count(*) AS n, coalesce(sum(size_bytes), 0) AS bytes, "
                    "count(DISTINCT output_sha) AS distinct_sha FROM result_artifacts "
                    "WHERE batch_id = ANY(CAST(:ids AS uuid[]))"
                ),
                {"ids": ids},
            )
            .mappings()
            .one()
        )
    attempts = [dict(x) for x in rows]
    n_tasks = sum(b[0] for b in batches)
    last = max(a["task_finished_at"] for a in attempts)
    makespan = (last - t_submit).total_seconds()
    waits = [float(a["queue_wait_ms"]) for a in attempts if a["queue_wait_ms"] is not None]
    walls = [float(a["wall_ms"]) for a in attempts if a["wall_ms"] is not None]
    gaps: list[float] = []
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for a in attempts:
        by_worker.setdefault(a["worker_id"], []).append(a)
    for seq in by_worker.values():
        seq.sort(key=lambda a: a["started_at"])
        gaps += [
            (b["started_at"] - a["ended_at"]).total_seconds() * 1000
            for a, b in pairwise(seq)
            if a["ended_at"] is not None
        ]
    worker_samples = [s for s in samples if s["proc"] != "coordinator"]
    coord_samples = [s for s in samples if s["proc"] == "coordinator"]
    row = {
        "phase": phase,
        "workers": p,
        "rep": rep,
        "n_tasks": n_tasks,
        "attempts": len(attempts),
        "makespan_s": r(makespan),
        "tasks_per_min": r(n_tasks / makespan * 60, 1),
        "queue_wait_p50_ms": r(pct(waits, 50), 1),
        "queue_wait_p95_ms": r(pct(waits, 95), 1),
        "queue_wait_p99_ms": r(pct(waits, 99), 1),
        "task_wall_p50_ms": r(pct(walls, 50), 1),
        "task_wall_p95_ms": r(pct(walls, 95), 1),
        "sum_task_wall_s": r(sum(walls) / 1000),
        "utilization": r(sum(walls) / 1000 / (p * makespan)),
        "intertask_gap_p50_ms": r(pct(gaps, 50), 1),
        "intertask_gap_p95_ms": r(pct(gaps, 95), 1),
        "worker_cpu_pct_mean": r(mean([s["cpu_pct"] for s in worker_samples]), 1),
        "worker_rss_mb_max": r(max((s["rss_mb"] for s in worker_samples), default=0.0), 1),
        "coordinator_cpu_pct_mean": r(mean([s["cpu_pct"] for s in coord_samples]), 1),
        "host_cpu_pct_mean": r(mean([s["host_cpu_pct"] for s in samples]), 1),
        "artifacts_written": int(art["n"]),
        "artifact_bytes": int(art["bytes"]),
        "distinct_outputs": int(art["distinct_sha"]),
    }
    print(
        f"  [{phase} p={p} rep={rep}] tasks={n_tasks} T={makespan:.2f}s "
        f"util={row['utilization']} qwait95={row['queue_wait_p95_ms']}ms "
        f"gap50={row['intertask_gap_p50_ms']}ms",
        flush=True,
    )
    return RunResult(
        phase, p, rep, n_tasks, [str(i) for i in ids], makespan, row, attempts, samples
    )


# ------------------------------------------------------------------------------ charts
SVG_STYLE = """
<style>
  .viz { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
         --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; }
  @media (prefers-color-scheme: dark) {
    .viz { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
           --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; }
  }
  .viz text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .title { fill: var(--ink); font-size: 15px; font-weight: 600; }
  .sub { fill: var(--ink2); font-size: 12px; }
  .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .label { fill: var(--ink2); font-size: 12px; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .series { stroke: var(--s1); stroke-width: 2; fill: none; stroke-linejoin: round; stroke-linecap: round; }
  .ref { stroke: var(--muted); stroke-width: 1.5; fill: none; }
  .whisker { stroke: var(--s1); stroke-width: 1; opacity: 0.55; }
  .dot { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
</style>"""


def _svg_frame(width: int, height: int, title: str, subtitle: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="viz" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">{SVG_STYLE}'
        f'<rect width="100%" height="100%" fill="var(--surface)"/>'
        f'<text class="title" x="20" y="26">{title}</text>'
        f'<text class="sub" x="20" y="45">{subtitle}</text>{body}</svg>\n'
    )


def speedup_svg(points: list[dict[str, Any]], path: Path, caption: str) -> None:
    """points: workers, speedup_mean, speedup_min, speedup_max, efficiency, makespan_mean."""
    w, h, left, right, top, bottom = 640, 380, 56, 150, 84, 46
    pw, ph = w - left - right, h - top - bottom
    max_p = max(pt["workers"] for pt in points)
    y_max = max(max_p, max(pt["speedup_max"] for pt in points)) * 1.05

    def x(v: float) -> float:
        return left + (v - 1) / (max_p - 1) * pw if max_p > 1 else left + pw / 2

    def y(v: float) -> float:
        return top + ph - v / y_max * ph

    parts = []
    for tick in range(0, int(y_max) + 1):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick}×</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
    )
    for pt in points:
        parts.append(
            f'<text class="tick" x="{x(pt["workers"]):.1f}" y="{top + ph + 18}" text-anchor="middle">{pt["workers"]}</text>'
        )
    parts.append(
        f'<text class="label" x="{left + pw / 2:.1f}" y="{h - 8}" text-anchor="middle">worker processes</text>'
    )
    parts.append(
        f'<polyline class="ref" points="{x(1):.1f},{y(1):.1f} {x(max_p):.1f},{y(max_p):.1f}"/>'
    )
    coords = " ".join(f"{x(pt['workers']):.1f},{y(pt['speedup_mean']):.1f}" for pt in points)
    parts.append(f'<polyline class="series" points="{coords}"/>')
    for pt in points:
        cx = x(pt["workers"])
        parts.append(
            f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(pt["speedup_min"]):.1f}" y2="{y(pt["speedup_max"]):.1f}"/>'
        )
        tip = (
            f"{pt['workers']} worker(s): speedup {pt['speedup_mean']:.2f}× "
            f"(efficiency {pt['efficiency'] * 100:.0f}%), mean makespan {pt['makespan_mean']:.2f} s"
        )
        parts.append(
            f'<circle class="dot" cx="{cx:.1f}" cy="{y(pt["speedup_mean"]):.1f}" r="4.5"><title>{tip}</title></circle>'
        )
    end = points[-1]
    parts.append(
        f'<text class="label" x="{x(end["workers"]) + 10:.1f}" y="{y(end["speedup_mean"]) + 4:.1f}">'
        f"{end['speedup_mean']:.2f}× ({end['efficiency'] * 100:.0f}% eff.)</text>"
    )
    parts.append(
        f'<text class="label" x="{x(max_p) + 10:.1f}" y="{y(max_p) + 4:.1f}">ideal {max_p}×</text>'
    )
    # legend (2 entries: measured series + reference line)
    lx, ly = left, 64
    parts.append(
        f'<line class="series" x1="{lx}" x2="{lx + 18}" y1="{ly}" y2="{ly}"/>'
        f'<circle class="dot" cx="{lx + 9}" cy="{ly}" r="4"/>'
        f'<text class="label" x="{lx + 26}" y="{ly + 4}">measured speedup T1/Tp (mean, min–max whiskers)</text>'
    )
    parts.append(
        f'<line class="ref" x1="{lx + 330}" x2="{lx + 348}" y1="{ly}" y2="{ly}"/>'
        f'<text class="label" x="{lx + 356}" y="{ly + 4}">ideal linear</text>'
    )
    path.write_text(_svg_frame(w, h, "Parallel speedup by worker count", caption, "".join(parts)))


def efficiency_svg(points: list[dict[str, Any]], path: Path, caption: str, p: int) -> None:
    """points: n_tasks, efficiency (fraction). Single series -> no legend box."""
    w, h, left, right, top, bottom = 640, 360, 56, 130, 64, 46
    pw, ph = w - left - right, h - top - bottom
    ns = [pt["n_tasks"] for pt in points]

    import math

    lo, hi = math.log2(min(ns)), math.log2(max(ns))

    def x(n: float) -> float:
        return left + (math.log2(n) - lo) / (hi - lo) * pw if hi > lo else left + pw / 2

    def y(e: float) -> float:
        return top + ph - min(e, 1.1) / 1.1 * ph

    parts = []
    for tick in (0, 25, 50, 75, 100):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{left + pw}" y1="{y(tick / 100):.1f}" y2="{y(tick / 100):.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y(tick / 100) + 4:.1f}" text-anchor="end">{tick}%</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" x2="{left + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>'
    )
    for n in ns:
        parts.append(
            f'<text class="tick" x="{x(n):.1f}" y="{top + ph + 18}" text-anchor="middle">{n}</text>'
        )
    parts.append(
        f'<text class="label" x="{left + pw / 2:.1f}" y="{h - 8}" text-anchor="middle">tasks in batch (log scale)</text>'
    )
    coords = " ".join(f"{x(pt['n_tasks']):.1f},{y(pt['efficiency']):.1f}" for pt in points)
    parts.append(f'<polyline class="series" points="{coords}"/>')
    for pt in points:
        tip = (
            f"{pt['n_tasks']} tasks: efficiency {pt['efficiency'] * 100:.0f}% "
            f"(T1 {pt['t1']:.2f} s, T{p} {pt['tp']:.2f} s)"
        )
        parts.append(
            f'<circle class="dot" cx="{x(pt["n_tasks"]):.1f}" cy="{y(pt["efficiency"]):.1f}" r="4.5"><title>{tip}</title></circle>'
        )
    first, last = points[0], points[-1]
    parts.append(
        f'<text class="label" x="{x(last["n_tasks"]) + 10:.1f}" y="{y(last["efficiency"]) + 4:.1f}">{last["efficiency"] * 100:.0f}%</text>'
    )
    parts.append(
        f'<text class="label" x="{x(first["n_tasks"]) + 8:.1f}" y="{y(first["efficiency"]) - 10:.1f}">{first["efficiency"] * 100:.0f}%</text>'
    )
    path.write_text(
        _svg_frame(
            w, h, f"Parallel efficiency at {p} workers vs batch size", caption, "".join(parts)
        )
    )


# ------------------------------------------------------------------------------ environment
def environment(engine: Engine, ds: DatasetInfo) -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    with engine.connect() as conn:
        pg = conn.execute(text("SELECT version()")).scalar_one()
    settings = get_settings()
    try:
        import redis

        redis_version = redis.Redis.from_url(settings.redis_url).info("server").get("redis_version")
    except Exception:
        redis_version = None
    env = environment_manifest()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git": git_info(),
        "code_sha": code_digest(),
        "os": platform.platform(),
        "kernel": os.uname().release,
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "numpy": env["packages"].get("numpy"),
        "pyarrow": env["packages"].get("pyarrow"),
        "postgres": pg.split(" on ")[0],
        "redis": redis_version,
        "build": "CPython (no debug build); numpy wheels from PyPI",
        "dataset": {"name": ds.name, "records": ds.n_bars, "sha256": ds.sha256},
        "scheduler": {
            "lease_ttl_s": settings.lease_ttl_s,
            "heartbeat_interval_s": settings.heartbeat_interval_s,
            "coordinator_tick_s": settings.coordinator_tick_s,
            "dispatch_window_factor": settings.dispatch_window_factor,
            "dispatch_min_window": settings.dispatch_min_window,
        },
    }


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dataset", default="pairs-ou-medium-v1")
    ap.add_argument("--short-tasks", type=int, default=64)
    ap.add_argument("--long-tasks", type=int, default=32)
    ap.add_argument("--long-bootstrap", type=int, default=3000)
    ap.add_argument("--sweep-counts", default="4,8,16,32,64")
    ap.add_argument("--sweep-bootstrap", type=int, default=600)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    worker_counts = [int(x) for x in args.workers.split(",")]
    sweep_counts = [] if args.no_sweep else [int(x) for x in args.sweep_counts.split(",")]
    ds = get_catalog().get(args.dataset)
    if ds is None:
        raise SystemExit(f"unknown dataset {args.dataset}")
    engine = make_engine(pool_size=4, application_name="qg-benchmark")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir or ROOT / "benchmarks" / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    env_info = environment(engine, ds)
    print(
        f"[bench] {env_info['cpu_model']} x{env_info['logical_cpus']}, {env_info['ram_gb']} GB, "
        f"dataset {ds.name} ({ds.n_bars} bars); output {out}",
        flush=True,
    )

    results: list[RunResult] = []
    seed = 1_000_000
    for p in worker_counts:
        print(f"[bench] === {p} worker(s) ===", flush=True)
        with LocalStack(
            log_dir=out / "logs" / f"p{p}",
            env={"QG_QUEUE_NAME": f"qg.bench.{uuid.uuid4().hex[:6]}"},
        ) as stack:
            stack.start_coordinator()
            ids = [f"bench-p{p}-w{i}" for i in range(p)]
            for wid in ids:
                stack.start_worker(wid)
            stack.wait_for_workers(engine, ids, timeout=120)
            seed += 1
            measure(
                engine,
                stack,
                "warmup",
                p,
                0,
                [
                    (max(2, p), args.long_bootstrap // 4, seed, "warmup-long"),
                    (2 * p, 0, seed, "warmup-short"),
                ],
                ds,
                args.timeout,
            )
            for rep in range(1, args.reps + 1):
                seed += 1
                results.append(
                    measure(
                        engine,
                        stack,
                        "main",
                        p,
                        rep,
                        [
                            (args.long_tasks, args.long_bootstrap, seed, f"main-long-p{p}-r{rep}"),
                            (args.short_tasks, 0, seed, f"main-short-p{p}-r{rep}"),
                        ],
                        ds,
                        args.timeout,
                    )
                )
            for n in sweep_counts:
                seed += 1
                results.append(
                    measure(
                        engine,
                        stack,
                        "sweep",
                        p,
                        n,
                        [(n, args.sweep_bootstrap, seed, f"sweep-{n}-p{p}")],
                        ds,
                        args.timeout,
                    )
                )

    # ---------------------------------------------------------------- aggregate
    main_runs = [x for x in results if x.phase == "main"]
    t_by_p = {p: [x.makespan_s for x in main_runs if x.workers == p] for p in worker_counts}
    t1 = mean(t_by_p.get(1, [])) if 1 in t_by_p else None
    table = []
    for p in worker_counts:
        ts = t_by_p[p]
        tp = mean(ts)
        assert tp is not None
        s_mean = t1 / tp if t1 else None
        rows = [x.row for x in main_runs if x.workers == p]
        table.append(
            {
                "workers": p,
                "reps": len(ts),
                "makespan_mean_s": r(tp),
                "makespan_min_s": r(min(ts)),
                "makespan_max_s": r(max(ts)),
                "makespan_stdev_s": r(statistics.stdev(ts)) if len(ts) > 1 else None,
                "speedup": r(s_mean),
                "efficiency": r(s_mean / p) if s_mean else None,
                "karp_flatt": r((1 / s_mean - 1 / p) / (1 - 1 / p)) if s_mean and p > 1 else None,
                "tasks_per_min": r(mean([x["tasks_per_min"] for x in rows]), 1),
                "queue_wait_p95_ms": r(mean([x["queue_wait_p95_ms"] for x in rows]), 1),
                "queue_wait_p50_ms": r(mean([x["queue_wait_p50_ms"] for x in rows]), 1),
                "queue_wait_p99_ms": r(mean([x["queue_wait_p99_ms"] for x in rows]), 1),
                "utilization": r(mean([x["utilization"] for x in rows])),
                "intertask_gap_p50_ms": r(
                    mean(
                        [
                            x["intertask_gap_p50_ms"]
                            for x in rows
                            if x["intertask_gap_p50_ms"] is not None
                        ]
                    ),
                    1,
                ),
                "worker_cpu_pct_mean": r(
                    mean(
                        [
                            x["worker_cpu_pct_mean"]
                            for x in rows
                            if x["worker_cpu_pct_mean"] is not None
                        ]
                    ),
                    1,
                ),
                "worker_rss_mb_max": max((x["worker_rss_mb_max"] for x in rows), default=None),
                "host_cpu_pct_mean": r(
                    mean(
                        [x["host_cpu_pct_mean"] for x in rows if x["host_cpu_pct_mean"] is not None]
                    ),
                    1,
                ),
                "artifacts_written_per_rep": rows[0]["artifacts_written"] if rows else None,
                "artifact_bytes_per_rep": rows[0]["artifact_bytes"] if rows else None,
            }
        )
    sweep = []
    for n in sweep_counts:
        runs = {x.workers: x for x in results if x.phase == "sweep" and x.rep == n}
        if 1 in runs:
            for p in worker_counts[1:]:
                if p in runs:
                    t1n, tpn = runs[1].makespan_s, runs[p].makespan_s
                    sweep.append(
                        {
                            "n_tasks": n,
                            "workers": p,
                            "t1": r(t1n),
                            "tp": r(tpn),
                            "speedup": r(t1n / tpn),
                            "efficiency": r(t1n / tpn / p),
                        }
                    )

    # ---------------------------------------------------------------- write outputs
    with (out / "runs.csv").open("w", newline="") as f:
        rows_all = [x.row for x in results]
        writer = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        writer.writeheader()
        writer.writerows(rows_all)
    with (out / "attempts.csv").open("w", newline="") as f:
        fields = [
            "phase",
            "workers",
            "rep",
            "task_id",
            "batch_id",
            "attempt",
            "worker_id",
            "status",
            "started_at",
            "ended_at",
            "queue_wait_ms",
            "wall_ms",
            "cpu_ms",
            "peak_rss_mb",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for x in results:
            for a in x.attempts:
                writer.writerow(
                    {
                        "phase": x.phase,
                        "workers": x.workers,
                        "rep": x.rep,
                        **{k: a.get(k) for k in fields[3:]},
                    }
                )
    with (out / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "workers",
                "rep",
                "t",
                "proc",
                "cpu_pct",
                "rss_mb",
                "host_cpu_pct",
            ],
        )
        writer.writeheader()
        for x in results:
            for s in x.samples:
                writer.writerow({"phase": x.phase, "workers": x.workers, "rep": x.rep, **s})
    workload = {
        "short_tasks": args.short_tasks,
        "long_tasks": args.long_tasks,
        "long_bootstrap_samples": args.long_bootstrap,
        "sweep_counts": sweep_counts,
        "sweep_bootstrap_samples": args.sweep_bootstrap,
        "reps": args.reps,
        "warmup": "1 warm-up batch per worker count (excluded)",
    }
    summary = {
        "run_id": run_id,
        "environment": env_info,
        "workload": workload,
        "scaling": table,
        "sweep": sweep,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if t1:
        pts = []
        for row in table:
            ts = t_by_p[row["workers"]]
            pts.append(
                {
                    "workers": row["workers"],
                    "speedup_mean": t1 / mean(ts),
                    "speedup_min": t1 / max(ts),
                    "speedup_max": t1 / min(ts),
                    "efficiency": t1 / mean(ts) / row["workers"],
                    "makespan_mean": mean(ts),
                }
            )
        speedup_svg(
            pts,
            out / "speedup.svg",
            f"{args.long_tasks} long + {args.short_tasks} short CPU-bound tasks · {ds.name} · "
            f"{env_info['logical_cpus']} vCPU host · {args.reps} reps",
        )
    pmax = max(worker_counts)
    sweep_pts = [s for s in sweep if s["workers"] == pmax]
    if len(sweep_pts) >= 2:
        efficiency_svg(
            sweep_pts,
            out / "efficiency_vs_tasks.svg",
            f"tasks with {args.sweep_bootstrap} bootstrap samples · efficiency = T1 / (p × Tp)",
            pmax,
        )
    write_report(out, summary)
    print(f"[bench] done: {out}", flush=True)
    engine.dispose()
    return 0


def write_report(out: Path, s: dict[str, Any]) -> None:
    e, w = s["environment"], s["workload"]
    lines = [
        f"# QuantGrid scaling benchmark — run {s['run_id']}",
        "",
        "All numbers below were measured by `scripts/benchmark.py` in this run; raw data is in",
        "`runs.csv`, `attempts.csv` and `samples.csv` next to this file.",
        "",
        "## Environment",
        "",
        "| item | value |",
        "|---|---|",
        f"| git commit | `{e['git'].get('git_commit')}` (dirty: {e['git'].get('git_dirty')}) |",
        f"| code_sha | `{e['code_sha'][:16]}…` |",
        f"| OS / kernel | {e['os']} / {e['kernel']} |",
        f"| CPU | {e['cpu_model']} × {e['logical_cpus']} logical CPUs |",
        f"| RAM | {e['ram_gb']} GB |",
        f"| Python / numpy / pyarrow | {e['python']} / {e['numpy']} / {e['pyarrow']} |",
        f"| PostgreSQL / Redis | {e['postgres']} / {e['redis']} |",
        f"| build | {e['build']} |",
        f"| dataset | `{e['dataset']['name']}`, {e['dataset']['records']} records, sha256 `{e['dataset']['sha256'][:16]}…` |",
        f"| scheduler | {e['scheduler']} |",
        f"| workload | {w['long_tasks']} long tasks ({w['long_bootstrap_samples']} bootstrap samples) + {w['short_tasks']} short tasks (no bootstrap) per repetition; {w['reps']} repetitions; {w['warmup']} |",
        "",
        "## Scaling (main workload)",
        "",
        "| workers | reps | makespan mean (min–max) s | speedup T1/Tp | efficiency | Karp–Flatt e | tasks/min | queue wait p50 / p95 / p99 ms | utilisation | inter-task gap p50 ms | worker CPU % (mean) | worker RSS max MB | host CPU % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in s["scaling"]:
        lines.append(
            f"| {t['workers']} | {t['reps']} | {t['makespan_mean_s']} ({t['makespan_min_s']}–{t['makespan_max_s']}) | "
            f"{t['speedup']} | {t['efficiency']} | {t['karp_flatt']} | {t['tasks_per_min']} | "
            f"{t['queue_wait_p50_ms']} / {t['queue_wait_p95_ms']} / {t['queue_wait_p99_ms']} | {t['utilization']} | "
            f"{t['intertask_gap_p50_ms']} | {t['worker_cpu_pct_mean']} | {t['worker_rss_mb_max']} | {t['host_cpu_pct_mean']} |"
        )
    lines += ["", "![speedup](speedup.svg)", ""]
    if s["sweep"]:
        lines += [
            "## Overhead at low task counts",
            "",
            "| tasks | workers | T1 s | Tp s | speedup | efficiency |",
            "|---|---|---|---|---|---|",
        ]
        for t in s["sweep"]:
            lines.append(
                f"| {t['n_tasks']} | {t['workers']} | {t['t1']} | {t['tp']} | {t['speedup']} | {t['efficiency']} |"
            )
        lines += ["", "![efficiency vs batch size](efficiency_vs_tasks.svg)", ""]
    lines += [
        "## Method",
        "",
        "* Makespan T_p = (last task `finished_at`) − (DB `now()` immediately before submission); both from PostgreSQL's clock.",
        "* Queue wait = claim time − time the task became eligible (`ready_at`), so it includes time waiting in the database for a dispatch slot.",
        "* Utilisation = Σ task wall time / (p × T_p). Inter-task gap = time between a worker publishing one task and starting the next (scheduler round trip).",
        "* Speedup S_p = mean(T1)/mean(Tp); efficiency E_p = S_p/p; Karp–Flatt serial fraction e = (1/S − 1/p)/(1 − 1/p).",
        "* Each worker count gets fresh processes and one excluded warm-up batch; every run uses new seeds, so every artifact is computed and written (cache bypassed, no content-address dedup).",
        "",
        "## Measurement bias and caveats",
        "",
        "* Coordinator, PostgreSQL, Redis, the benchmark driver and all workers share the same host; at 4 workers on a 4-vCPU machine they compete for CPU, so T4 understates what dedicated worker hosts would achieve.",
        "* The host is a virtual machine: CPU frequency, turbo and steal time are not controlled; spread across repetitions is reported (min–max) instead of a single number.",
        "* Short tasks are deliberately included: their cost is dominated by per-task scheduler overhead (claim, publish, NOTIFY → dispatch), which bounds efficiency at small batch sizes.",
        "* Numerical libraries run single-threaded per process (OMP/OpenBLAS/MKL threads = 1); results do not transfer to multi-threaded configurations.",
        "* Artifacts are written to a local filesystem with fsync; network/object storage would add latency per task.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
