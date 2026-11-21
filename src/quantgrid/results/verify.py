"""Batch verifier: checks that the result table holds exactly one result per expected task
and that the scheduler's audit trail is consistent. ``error`` findings fail verification,
``warning`` ones are informational. Checks:

1. batch_spec_integrity       stored batch spec re-hashes to ``batches.spec_sha``
2. expected_task_set          re-expanding the stored spec yields exactly the stored
                              (ordinal, spec_hash) set; every stored task spec re-hashes
3. all_terminal               no task is still PENDING/RUNNING/RETRY_PENDING (optional)
4. one_result_per_success     each SUCCEEDED task has exactly one result row, and no other
                              task has one; no spec_hash is published twice
5. artifacts_verified         every referenced artifact exists, matches its SHA-256 and size,
                              decodes, and embeds the task's spec_hash and metrics
6. duckdb_cross_check         DuckDB scan of the artifacts agrees on row counts/spec hashes
7. attempt_log_consistent     attempts are numbered 1..n, none left RUNNING on terminal
                              tasks, exactly one SUCCEEDED/CACHED attempt per success and it
                              is the attempt that published
8. transitions_legal          every audited transition is legal and each task's path ends
                              in its current state with one ->RUNNING per attempt
9. retries_consistent         every attempt that produced output (incl. late "zombie"
                              completions) produced the published checksum
10. orphan_artifacts          (warning) stored artifacts referenced by no result row
11. recompute_sample          (optional) re-running N tasks reproduces the published bytes
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import Engine, text

from quantgrid.domain.specs import ExperimentBatchSpec, TaskSpec, expand_grid
from quantgrid.domain.states import TaskState, can_transition

Severity = Literal["error", "warning"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    severity: Severity = "error"
    examples: list[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    batch_id: str
    batch_state: str
    expected_tasks: int
    task_rows: int
    counts: dict[str, int]
    published_results: int
    attempts: int
    retry_distribution: dict[str, int]
    lease_expirations: int
    late_completions: int
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["ok"] = self.ok
        return out

    def summary(self) -> str:
        lines = [
            f"batch {self.batch_id} [{self.batch_state}] expected={self.expected_tasks} "
            f"tasks={self.task_rows} results={self.published_results} attempts={self.attempts} "
            f"lease_expirations={self.lease_expirations} late_completions={self.late_completions}",
            f"task states: {self.counts}   attempts per task: {self.retry_distribution}",
        ]
        for c in self.checks:
            mark = "PASS" if c.ok else ("FAIL" if c.severity == "error" else "WARN")
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
            lines.extend(f"         - {e}" for e in c.examples[:5])
        lines.append("VERIFICATION " + ("PASSED" if self.ok else "FAILED"))
        return "\n".join(lines)


def _rows(engine: Engine, sql: str, **params: Any) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]


def verify_batch(
    engine: Engine,
    store: Any,
    batch_id: uuid.UUID | str,
    *,
    require_complete: bool = True,
    deep: bool = True,
    check_orphans: bool = True,
    recompute_sample: int = 0,
    data_dir: Path | None = None,
) -> VerificationReport:
    bid = str(batch_id)
    batch_rows = _rows(engine, "SELECT * FROM batches WHERE id = :id", id=bid)
    if not batch_rows:
        raise LookupError(f"batch {bid} not found")
    batch = batch_rows[0]
    tasks = _rows(
        engine,
        "SELECT id, ordinal, spec, spec_hash, state, attempts, checksum_mismatch "
        "FROM tasks WHERE batch_id = :id ORDER BY ordinal",
        id=bid,
    )
    results = _rows(
        engine,
        "SELECT task_id, attempt, spec_hash, output_uri, output_sha, size_bytes, row_count, "
        "metrics FROM result_artifacts WHERE batch_id = :id",
        id=bid,
    )
    attempts = _rows(
        engine,
        "SELECT task_id, attempt, status, output_sha, late_completion_at FROM task_attempts "
        "WHERE batch_id = :id ORDER BY task_id, attempt",
        id=bid,
    )
    transitions = _rows(
        engine,
        "SELECT task_id, from_state, to_state, attempt FROM task_transitions "
        "WHERE batch_id = :id ORDER BY id",
        id=bid,
    )
    checks: list[Check] = []
    counts = Counter(t["state"] for t in tasks)
    by_task_results: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_task_results[r["task_id"]].append(r)
    by_task_attempts: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for a in attempts:
        by_task_attempts[a["task_id"]].append(a)

    # 1 + 2: spec integrity and expected task set
    spec = ExperimentBatchSpec.model_validate(batch["spec"])
    checks.append(
        Check(
            "batch_spec_integrity",
            spec.spec_sha() == batch["spec_sha"],
            "stored batch spec re-hashes to spec_sha"
            if spec.spec_sha() == batch["spec_sha"]
            else f"spec re-hashes to {spec.spec_sha()} != stored {batch['spec_sha']}",
        )
    )
    expected = {(e.ordinal, e.spec_hash) for e in expand_grid(spec, max_tasks=batch["task_count"])}
    actual = {(t["ordinal"], t["spec_hash"]) for t in tasks}
    bad_hash = [
        str(t["id"])
        for t in tasks
        if TaskSpec.model_validate(t["spec"]).spec_hash() != t["spec_hash"]
    ]
    missing, extra = expected - actual, actual - expected
    checks.append(
        Check(
            "expected_task_set",
            not missing and not extra and not bad_hash and len(tasks) == batch["task_count"],
            f"{len(expected)} expected, {len(tasks)} stored, {len(missing)} missing, "
            f"{len(extra)} unexpected, {len(bad_hash)} specs not matching their hash",
            examples=[f"missing ordinal {o}" for o, _ in sorted(missing)[:3]]
            + [f"unexpected ordinal {o}" for o, _ in sorted(extra)[:3]]
            + [f"task {t} spec/hash mismatch" for t in bad_hash[:3]],
        )
    )

    # 3: completion
    active = [t for t in tasks if t["state"] in ("PENDING", "RUNNING", "RETRY_PENDING")]
    checks.append(
        Check(
            "all_terminal",
            not active or not require_complete,
            f"{len(active)} task(s) not terminal"
            if active
            else "every task is SUCCEEDED, FAILED or CANCELED",
            severity="error" if require_complete else "warning",
            examples=[f"task {t['id']} {t['state']}" for t in active[:5]],
        )
    )

    # 4: exactly one logical result per succeeded task
    problems: list[str] = []
    for t in tasks:
        n = len(by_task_results.get(t["id"], []))
        if t["state"] == TaskState.SUCCEEDED and n != 1:
            problems.append(f"SUCCEEDED task {t['id']} has {n} result rows")
        if t["state"] != TaskState.SUCCEEDED and n != 0:
            problems.append(f"{t['state']} task {t['id']} has {n} result rows")
    task_ids = {t["id"] for t in tasks}
    problems += [
        f"result row for unknown task {r['task_id']}"
        for r in results
        if r["task_id"] not in task_ids
    ]
    dup_specs = [h for h, n in Counter(r["spec_hash"] for r in results).items() if n > 1]
    problems += [
        f"spec_hash {h[:12]} published {n}x"
        for h, n in Counter(r["spec_hash"] for r in results).items()
        if n > 1
    ]
    checks.append(
        Check(
            "one_result_per_success",
            not problems,
            f"{len(results)} result rows for {counts.get('SUCCEEDED', 0)} succeeded tasks; "
            f"{len(dup_specs)} duplicated spec hashes",
            examples=problems[:10],
        )
    )

    # 5: artifacts verified
    art_problems: list[str] = []
    if deep and results:
        from quantgrid.results.codec import validate_artifact

        for r in results:
            try:
                if not store.verify(
                    r["output_uri"], r["output_sha"], expected_size=r["size_bytes"]
                ):
                    art_problems.append(f"{r['output_uri']} missing or checksum/size mismatch")
                    continue
                data = store.read_bytes(r["output_uri"])  # re-verifies the checksum
                issues = validate_artifact(
                    data, expected_spec_hash=r["spec_hash"], expected_metrics=r["metrics"]
                )
                art_problems += [f"{r['output_uri']}: {i}" for i in issues]
            except Exception as exc:  # report, never crash the verifier
                art_problems.append(f"{r['output_uri']}: {type(exc).__name__}: {exc}")
    checks.append(
        Check(
            "artifacts_verified",
            not art_problems,
            f"{len(results)} artifacts checked (sha256, size, decode, spec_hash, metrics)"
            if deep
            else "skipped (deep=False)",
            examples=art_problems[:10],
        )
    )

    # 6: DuckDB cross-check -----------------------------------------------------------
    if deep and results:
        from quantgrid.results.export import scan_artifacts

        paths = [store.path_for_uri(r["output_uri"]) for r in results]
        scans = {str(s["path"]): s for s in scan_artifacts(paths)}
        dk_problems = []
        for r, p in zip(results, paths, strict=True):
            s = scans.get(str(p))
            if s is None:
                dk_problems.append(f"{p} not scanned")
            elif s["row_count"] != r["row_count"] or s["spec_hash"] != r["spec_hash"]:
                dk_problems.append(
                    f"{p.name}: duckdb rows={s['row_count']} spec={str(s['spec_hash'])[:12]} vs db "
                    f"rows={r['row_count']} spec={r['spec_hash'][:12]}"
                )
        checks.append(
            Check(
                "duckdb_cross_check",
                not dk_problems,
                f"DuckDB scanned {len(scans)} distinct artifact files",
                examples=dk_problems[:10],
            )
        )

    # 7: attempt log
    att_problems: list[str] = []
    result_attempt = {r["task_id"]: r["attempt"] for r in results}
    for t in tasks:
        rows = by_task_attempts.get(t["id"], [])
        numbers = [a["attempt"] for a in rows]
        if numbers != list(range(1, t["attempts"] + 1)):
            att_problems.append(f"task {t['id']}: attempts {numbers} vs counter {t['attempts']}")
        terminal = t["state"] in ("SUCCEEDED", "FAILED", "CANCELED")
        if terminal and any(a["status"] == "RUNNING" for a in rows):
            att_problems.append(f"terminal task {t['id']} has a RUNNING attempt")
        winners = [a for a in rows if a["status"] in ("SUCCEEDED", "CACHED")]
        if t["state"] == TaskState.SUCCEEDED:
            if len(winners) != 1:
                att_problems.append(f"task {t['id']} has {len(winners)} winning attempts")
            elif result_attempt.get(t["id"]) != winners[0]["attempt"]:
                att_problems.append(f"task {t['id']}: published attempt != winning attempt")
        elif winners:
            att_problems.append(f"{t['state']} task {t['id']} has a winning attempt")
    checks.append(
        Check(
            "attempt_log_consistent",
            not att_problems,
            f"{len(attempts)} attempts across {len(tasks)} tasks",
            examples=att_problems[:10],
        )
    )

    # 8: transitions
    tr_problems: list[str] = []
    paths_by_task: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for tr in transitions:
        paths_by_task[tr["task_id"]].append(tr)
    for t in tasks:
        path = paths_by_task.get(t["id"], [])
        if not path or path[0]["from_state"] is not None or path[0]["to_state"] != "PENDING":
            tr_problems.append(f"task {t['id']}: path does not start with creation -> PENDING")
            continue
        for prev, cur in pairwise(path):
            if cur["from_state"] != prev["to_state"] or not can_transition(
                TaskState(cur["from_state"]), TaskState(cur["to_state"])
            ):
                tr_problems.append(
                    f"task {t['id']}: illegal/broken step {cur['from_state']} -> {cur['to_state']}"
                )
        if path[-1]["to_state"] != t["state"]:
            tr_problems.append(
                f"task {t['id']}: path ends in {path[-1]['to_state']} != {t['state']}"
            )
        runs = sum(1 for tr in path if tr["to_state"] == "RUNNING")
        if runs != t["attempts"]:
            tr_problems.append(
                f"task {t['id']}: {runs} ->RUNNING transitions vs {t['attempts']} attempts"
            )
    checks.append(
        Check(
            "transitions_legal",
            not tr_problems,
            f"{len(transitions)} audited transitions",
            examples=tr_problems[:10],
        )
    )

    # 9: retries never changed a result silently
    published_sha = {r["task_id"]: r["output_sha"] for r in results}
    mismatches = [
        f"task {a['task_id']} attempt {a['attempt']} ({a['status']}) produced "
        f"{a['output_sha'][:12]} != published {published_sha[a['task_id']][:12]}"
        for a in attempts
        if a["output_sha"]
        and a["task_id"] in published_sha
        and a["output_sha"] != published_sha[a["task_id"]]
    ]
    flagged = [str(t["id"]) for t in tasks if t["checksum_mismatch"]]
    late = sum(1 for a in attempts if a["late_completion_at"] is not None)
    checks.append(
        Check(
            "retries_consistent",
            not mismatches and not flagged,
            f"{sum(1 for a in attempts if a['output_sha'])} attempts with output, "
            f"{late} late (zombie) completions, {len(mismatches)} checksum mismatches",
            examples=mismatches[:5] + [f"task {f} flagged checksum_mismatch" for f in flagged[:5]],
        )
    )

    # 10: orphans (global, informational)
    if check_orphans and hasattr(store, "iter_artifacts"):
        referenced = {
            r["output_uri"]
            for r in _rows(engine, "SELECT DISTINCT output_uri FROM result_artifacts")
        }
        orphans = [uri for uri, _ in store.iter_artifacts() if uri not in referenced]
        checks.append(
            Check(
                "orphan_artifacts",
                not orphans,
                f"{len(orphans)} stored artifact(s) not referenced by any published result "
                "(e.g. discarded after cancel); safe to garbage-collect",
                severity="warning",
                examples=orphans[:5],
            )
        )

    # 11: recompute sample
    if recompute_sample > 0 and results:
        checks.append(_recompute(results, tasks, recompute_sample, data_dir))

    lease_expirations = sum(1 for a in attempts if a["status"] == "LEASE_EXPIRED")
    dist = Counter(t["attempts"] for t in tasks)
    return VerificationReport(
        batch_id=bid,
        batch_state=batch["state"],
        expected_tasks=len(expected),
        task_rows=len(tasks),
        counts={s.value: counts.get(s.value, 0) for s in TaskState},
        published_results=len(results),
        attempts=len(attempts),
        retry_distribution={str(k): v for k, v in sorted(dist.items())},
        lease_expirations=lease_expirations,
        late_completions=late,
        checks=checks,
    )


def _recompute(
    results: list[dict[str, Any]], tasks: list[dict[str, Any]], n: int, data_dir: Path | None
) -> Check:
    from quantgrid.config import get_settings
    from quantgrid.domain.datasets import load_bars
    from quantgrid.domain.engine import ENGINE_VERSION, run_task_spec
    from quantgrid.domain.hashing import sha256_hex
    from quantgrid.domain.types import NullCancelToken
    from quantgrid.results.codec import encode_result

    settings = get_settings()
    spec_by_task = {t["id"]: t["spec"] for t in tasks}
    chosen = sorted(results, key=lambda r: str(r["task_id"]))[:n]
    problems = []
    for r in chosen:
        spec = TaskSpec.model_validate(spec_by_task[r["task_id"]])
        bars = load_bars(
            spec.dataset,
            data_dir or Path(settings.data_dir),
            auto_materialize=settings.auto_materialize_datasets,
            verify=True,
        )
        out = run_task_spec(spec, bars, cancel=NullCancelToken(), attempt=1)
        data = encode_result(r["spec_hash"], out, engine_version=ENGINE_VERSION).data
        if sha256_hex(data) != r["output_sha"]:
            problems.append(
                f"task {r['task_id']}: recomputed {sha256_hex(data)[:12]} != {r['output_sha'][:12]}"
            )
    return Check(
        "recompute_sample",
        not problems,
        f"re-ran {len(chosen)} task(s) in-process; byte-identical: {len(chosen) - len(problems)}",
        examples=problems,
    )
