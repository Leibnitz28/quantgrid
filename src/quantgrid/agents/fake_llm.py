"""Deterministic offline providers: :class:`FakeLLM` (default, used by CI) and
:class:`ScriptedLLM` (replays fixed turns to simulate a malicious/compromised model).

``FakeLLM`` is a pure rule-based function of (tools offered, user message, tool results):
identical inputs always produce identical outputs. It is *not* trusted by the system -
its output goes through the same strict parsing and policy checks as a real model's.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from quantgrid.agents.llm import (
    LLMSession,
    ProviderError,
    ProviderTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from quantgrid.agents.prompts import GOAL_CLOSE, GOAL_OPEN

_NUM = r"(\d+(?:\.\d+)?)"
_RANGE = rf"{_NUM}\s*(?:-|to|\.\.|\u2013)\s*{_NUM}"
MAX_SEED_COUNT = 8


# --------------------------------------------------------------------------------------
# Goal parsing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedGoal:
    dataset_kw: str | None
    lookbacks: list[int]
    entries: list[float]
    exit_z: float
    seeds: list[int]
    bootstrap: int
    max_tasks: int | None
    pinned_dataset: str | None


def _spread(lo: float, hi: float, n: int) -> list[float]:
    if lo > hi:
        lo, hi = hi, lo
    if n <= 1 or lo == hi:
        return [lo]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def parse_goal(user_text: str) -> ParsedGoal:
    """Parse constraint lines (outside the delimiters) and the goal text (inside)."""
    open_i = user_text.find(GOAL_OPEN)
    close_i = user_text.rfind(GOAL_CLOSE)
    if open_i >= 0 and close_i > open_i:
        goal = user_text[open_i + len(GOAL_OPEN) : close_i]
        header = user_text[:open_i]
    else:
        goal, header = user_text, ""
    g = goal.lower()
    m = re.search(r"^MAX_TASKS:\s*(\d+)\s*$", header, re.MULTILINE)
    cap = int(m.group(1)) if m else None
    m = re.search(r"^DATASET:\s*(\S+)\s*$", header, re.MULTILINE)
    pinned = m.group(1) if m else None
    m2 = re.search(r"(?:at most|no more than|max(?:imum)?(?: of)?)\s+(\d+)\s+tasks?", g)
    if m2:
        n = int(m2.group(1))
        cap = n if cap is None else min(cap, n)

    thorough = "thorough" in g
    quick = "quick" in g and not thorough
    n_vals = 2 if quick else 6 if thorough else 3

    dataset_kw = next((k for k in ("gappy", "large", "medium", "small") if k in g), None)

    lookbacks: list[int]
    m = re.search(rf"(?:lookback|window)s?\D{{0,12}}?{_RANGE}", g)
    if m:
        lookbacks = sorted({round(v) for v in _spread(float(m[1]), float(m[2]), n_vals)})
    else:
        m = re.search(rf"(?:lookback|window)s?\D{{0,12}}?{_NUM}", g)
        if m:
            lookbacks = [round(float(m[1]))]
        else:
            lookbacks = (
                [20, 40] if quick else [10, 20, 30, 40, 60, 80] if thorough else [20, 40, 60]
            )
    lookbacks = [min(max(v, 5), 5000) for v in lookbacks]
    lookbacks = sorted(set(lookbacks))

    entries: list[float]
    m = re.search(rf"entry(?:[_ ]?z|[ _]?threshold)?s?\D{{0,12}}?{_RANGE}", g)
    if m:
        entries = sorted({round(v, 2) for v in _spread(float(m[1]), float(m[2]), n_vals)})
    else:
        m = re.search(rf"entry(?:[_ ]?z|[ _]?threshold)?s?\D{{0,12}}?{_NUM}", g)
        entries = (
            [round(float(m[1]), 2)]
            if m
            else [1.5, 2.0]
            if quick
            else [1.0, 1.5, 2.0, 2.5, 3.0]
            if thorough
            else [1.5, 2.0, 2.5]
        )
    entries = [min(max(v, 0.1), 10.0) for v in entries]

    m = re.search(rf"exit(?:[_ ]?z|[ _]?threshold)?\D{{0,12}}?{_NUM}", g)
    exit_z = round(float(m[1]), 2) if m else 0.5
    exit_z = min(exit_z, round(min(entries) / 2, 2))

    m = re.search(r"(\d+)\s+seeds?", g)
    n_seeds = min(int(m.group(1)), MAX_SEED_COUNT) if m else 1
    seeds = list(range(max(n_seeds, 1)))

    bootstrap = 0
    if "bootstrap" in g:
        m = re.search(r"(\d+)\s*(?:bootstrap|resamples?)", g) or re.search(
            r"bootstrap\D{0,12}?(\d+)", g
        )
        bootstrap = min(int(m.group(1)), 50_000) if m else 1000
    return ParsedGoal(dataset_kw, lookbacks, entries, exit_z, seeds, bootstrap, cap, pinned)


def _pick_dataset(rows: Sequence[dict[str, Any]], goal: ParsedGoal) -> str | None:
    names = sorted(str(r["name"]) for r in rows)
    if goal.pinned_dataset in names:
        return goal.pinned_dataset
    if goal.dataset_kw:
        for n in names:
            if goal.dataset_kw in n:
                return n
    for n in names:
        if "small" in n:
            return n
    bars = {str(r["name"]): int(r.get("n_bars", 0)) for r in rows}
    return min(names, key=lambda n: (bars.get(n, 0), n)) if names else None


def _trim_grid(
    grid: dict[str, list[Any]], seeds: list[int], target: int
) -> tuple[dict[str, list[Any]], list[int]]:
    grid = {k: list(v) for k, v in grid.items()}
    seeds = list(seeds)

    def count() -> int:
        n = len(seeds)
        for v in grid.values():
            n *= len(v)
        return n

    target = max(target, 1)
    while count() > target:
        name = max(sorted(grid), key=lambda k: len(grid[k]))
        if len(grid[name]) > 1:
            grid[name] = grid[name][:-1]
        elif len(seeds) > 1:
            seeds = seeds[:-1]
        else:
            break
    return grid, seeds


# --------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------


@dataclass
class _PlannerSession:
    goal: ParsedGoal | None = None
    stage: int = 0
    dataset_name: str | None = None
    dataset_sha: str | None = None
    n_bars: int = 0
    quota: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    adjusted: bool = False

    def send_user(self, text: str) -> ProviderTurn:
        self.goal = parse_goal(text)
        self.stage = 1
        return _calls(("list_datasets", {}))

    def send_tool_results(self, results: list[ToolResult]) -> ProviderTurn:
        res = results[0] if results else None
        if res is None or not res.ok:
            return _final("Unable to plan: a tool call failed.")
        if self.goal is None:
            raise ProviderError("internal", "no user message received")
        if self.stage == 1:
            rows = [r for r in (res.content or []) if isinstance(r, dict)]
            name = _pick_dataset(rows, self.goal)
            if name is None:
                return _final("No dataset is available to plan on.")
            self.dataset_name = name
            self.stage = 2
            return _calls(("get_dataset_info", {"name": name}))
        if self.stage == 2:
            info = res.content or {}
            self.dataset_sha = str(info.get("sha256", ""))
            self.n_bars = int(info.get("n_bars", 0))
            self.stage = 3
            return _calls(("get_strategy_schema", {"name": "pairs_zscore", "version": "1"}))
        if self.stage == 3:
            self.stage = 4
            return _calls(("get_quota", {}))
        if self.stage == 4:
            self.quota = dict(res.content or {})
            self.spec = self._draft()
            self.stage = 5
            return _calls(("validate_spec", {"spec": self.spec}))
        # stage 5: validate_spec result
        content = res.content or {}
        if not content.get("allowed", False) and not self.adjusted:
            self.adjusted = True
            self.spec = self._adjust(self.spec, content)
            return _calls(("validate_spec", {"spec": self.spec}))
        return _final(
            "Proposed experiment batch (deterministic FakeLLM):\n```json\n"
            + json.dumps(self.spec, sort_keys=True, indent=2)
            + "\n```"
        )

    def _draft(self) -> dict[str, Any]:
        g = self.goal
        assert g is not None  # noqa: S101 - set in send_user
        grid: dict[str, list[Any]] = {"lookback": list(g.lookbacks), "entry_z": list(g.entries)}
        cap = int(self.quota.get("max_tasks_per_batch", 5000))
        if g.max_tasks is not None:
            cap = min(cap, g.max_tasks)
        grid, seeds = _trim_grid(grid, g.seeds, cap)
        return {
            "name": f"Plan {self.dataset_name} pairs_zscore sweep",
            "description": "Proposed by the deterministic FakeLLM planner; review first.",
            "dataset": {"name": self.dataset_name, "sha256": self.dataset_sha},
            "strategy": {"name": "pairs_zscore", "version": "1"},
            "grid": grid,
            "fixed_params": {"exit_z": g.exit_z},
            "seeds": seeds,
            "workload": {"bootstrap_samples": g.bootstrap, "bootstrap_block": 20},
            "budget": {"max_runtime_s": 120, "max_memory_mb": 512},
            "max_attempts": 3,
        }

    def _adjust(self, spec: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(json.dumps(spec))
        codes = {v.get("code") for v in decision.get("violations", []) if isinstance(v, dict)}
        q = self.quota
        if "RUNTIME_BUDGET_EXCEEDS_POLICY" in codes:
            out["budget"]["max_runtime_s"] = int(q.get("max_runtime_s_per_task", 120))
        if "MEMORY_BUDGET_EXCEEDS_POLICY" in codes:
            out["budget"]["max_memory_mb"] = int(q.get("max_memory_mb_per_task", 512))
        if "WORKLOAD_EXCEEDS_POLICY" in codes:
            out["workload"]["bootstrap_samples"] = int(q.get("max_bootstrap_samples", 0))
        if "MAX_ATTEMPTS_EXCEEDS_POLICY" in codes:
            out["max_attempts"] = int(q.get("max_attempts", 1))
        if codes & {"GRID_TOO_LARGE", "WORKER_TIME_QUOTA_EXCEEDED"}:
            runtime = max(int(out["budget"]["max_runtime_s"]), 1)
            target = min(
                int(q.get("max_tasks_per_batch", 5000)),
                int(q.get("max_worker_seconds_per_batch", 10**9)) // runtime,
            )
            out["grid"], out["seeds"] = _trim_grid(out["grid"], out["seeds"], target)
        return out


@dataclass
class _ExplainerSession:
    batch_id: str = ""
    stage: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
    failed: list[dict[str, Any]] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=dict)

    def send_user(self, text: str) -> ProviderTurn:
        m = re.search(r"^BATCH_ID:\s*(\S+)\s*$", text, re.MULTILINE)
        self.batch_id = m.group(1) if m else ""
        self.stage = 1
        return _calls(("get_batch_progress", {"batch_id": self.batch_id}))

    def send_tool_results(self, results: list[ToolResult]) -> ProviderTurn:
        if any(not r.ok for r in results):
            return _final("Unable to explain this batch: a read-only lookup failed.")
        if self.stage == 1:
            self.progress = dict(results[0].content or {})
            self.stage = 2
            return _calls(("list_failed_tasks", {"batch_id": self.batch_id, "limit": 50}))
        if self.stage == 2:
            self.failed = [r for r in (results[0].content or []) if isinstance(r, dict)]
            ids = [str(r["task_id"]) for r in self.failed[:3] if r.get("task_id")]
            self.stage = 3
            if ids:
                return _calls(*[("get_task_attempts", {"task_id": i}) for i in ids])
        else:
            pass
        for r in results:
            if isinstance(r.content, list):
                self.attempts[r.call_id] = len(r.content)
        return _final(self._compose())

    def _compose(self) -> str:
        counts = {str(k): int(v) for k, v in dict(self.progress.get("counts", {})).items()}
        total = int(self.progress.get("total", sum(counts.values())))
        state = _safe_token(self.progress.get("state", "UNKNOWN"))
        done = counts.get("SUCCEEDED", 0)
        summary = f"Batch is {state}: {done} of {total} tasks succeeded, {len(self.failed)} failed."
        hist: dict[str, int] = {}
        for row in self.failed:
            code = _safe_token(row.get("error_code", "UNKNOWN"))
            hist[code] = hist.get(code, 0) + 1
        lines = [summary]
        for k in sorted(counts):
            lines.append(f"- {_safe_token(k)}: {counts[k]} task(s)")
        for code in sorted(hist):
            lines.append(f"- error {code}: {hist[code]} failed task(s)")
        if self.attempts:
            lines.append(f"- inspected attempt history of {len(self.attempts)} failed task(s)")
        return "\n".join(lines)


def _safe_token(value: Any) -> str:
    """Only short upper-case identifiers survive; anything else (e.g. injected text) is
    replaced, so tool-result prose can never flow into the explanation verbatim."""
    s = str(value)
    return s if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,39}", s) else "OTHER"


def _calls(*calls: tuple[str, dict[str, Any]]) -> ProviderTurn:
    return ProviderTurn(
        tool_calls=tuple(
            ToolCall(id=f"fake_{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)
        ),
        stop_reason="tool_use",
    )


def _final(text: str) -> ProviderTurn:
    return ProviderTurn(text=text, stop_reason="end_turn")


class _FakeSession:
    def __init__(self, inner: _PlannerSession | _ExplainerSession) -> None:
        self._inner = inner

    def send_user(self, text: str) -> ProviderTurn:
        return self._inner.send_user(text)

    def send_tool_results(self, results: list[ToolResult]) -> ProviderTurn:
        return self._inner.send_tool_results(results)


class FakeLLM:
    """Deterministic rule-based provider; role is inferred from the offered tools."""

    name = "fake"

    def open_session(self, system: str, tools: list[ToolSpec]) -> LLMSession:
        names = {t.name for t in tools}
        if "validate_spec" in names:
            return _FakeSession(_PlannerSession())
        if "get_batch_progress" in names:
            return _FakeSession(_ExplainerSession())
        raise ProviderError("bad_request", "FakeLLM does not recognise this toolset")


class ScriptedLLM:
    """Replays a fixed list of turns (a 'compromised model'); records what it was sent."""

    name = "scripted"

    def __init__(self, turns: Sequence[ProviderTurn], *, delay_s: float = 0.0) -> None:
        self.turns = list(turns)
        self.delay_s = delay_s
        self.sent: list[Any] = []
        self.systems: list[str] = []
        self.tool_specs: list[list[ToolSpec]] = []
        self._i = 0

    def open_session(self, system: str, tools: list[ToolSpec]) -> LLMSession:
        self.systems.append(system)
        self.tool_specs.append(list(tools))
        return self

    def _next(self) -> ProviderTurn:
        import time

        if self.delay_s:
            time.sleep(self.delay_s)
        if self._i >= len(self.turns):
            raise ProviderError("internal", "script exhausted")
        turn = self.turns[self._i]
        self._i += 1
        return turn

    def send_user(self, text: str) -> ProviderTurn:
        self.sent.append(text)
        return self._next()

    def send_tool_results(self, results: list[ToolResult]) -> ProviderTurn:
        self.sent.append(list(results))
        return self._next()
