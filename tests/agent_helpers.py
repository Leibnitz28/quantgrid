"""Shared fakes for the agent tests (no database, no network, no real catalog)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from quantgrid.agents.llm import ProviderTurn, ToolCall
from quantgrid.config import Settings
from quantgrid.domain.policy import PolicyService, QuotaPolicy
from quantgrid.domain.specs import DatasetInfo

START = datetime(2015, 1, 1, tzinfo=UTC)


def make_info(
    name: str, n_bars: int, interval_s: int, description: str = "fake data"
) -> DatasetInfo:
    return DatasetInfo(
        name=name,
        sha256=("%02x" % (n_bars % 251)) * 32,
        n_bars=n_bars,
        start=START,
        end=START + timedelta(seconds=interval_s * (n_bars - 1)),
        bar_interval_s=interval_s,
        columns=("ts", "close_x", "close_y"),
        synthetic=True,
        license="synthetic",
        description=description,
        generator={"seed": 1},
    )


class FakeCatalog:
    def __init__(self, infos: Sequence[DatasetInfo]) -> None:
        self._by_name = {i.name: i for i in infos}
        self.calls = 0

    def get(self, name: str) -> DatasetInfo | None:
        self.calls += 1
        return self._by_name.get(name)

    def list(self) -> Sequence[DatasetInfo]:
        return list(self._by_name.values())


def default_catalog(description: str = "Synthetic cointegrated pair") -> FakeCatalog:
    return FakeCatalog(
        [
            make_info("pairs-ou-small-v1", 2000, 86400, description),
            make_info("pairs-ou-medium-v1", 20000, 3600),
            make_info("pairs-ou-gappy-v1", 5000, 3600),
        ]
    )


def make_policy(catalog: FakeCatalog | None = None) -> tuple[PolicyService, FakeCatalog]:
    cat = catalog or default_catalog()
    return PolicyService(cat), cat


def fast_settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "agent_max_steps": 8,
        "agent_max_tool_calls": 16,
        "agent_timeout_s": 5.0,
    }
    base.update(kw)
    return Settings(**base)


def quota(**kw: Any) -> QuotaPolicy:
    return QuotaPolicy(**kw)


def call(name: str, args: dict[str, Any], cid: str = "c1") -> ProviderTurn:
    return ProviderTurn(tool_calls=(ToolCall(cid, name, args),), stop_reason="tool_use")


def final(text: str) -> ProviderTurn:
    return ProviderTurn(text=text, stop_reason="end_turn")


def good_spec_dict(
    cat: FakeCatalog, name: str = "pairs-ou-small-v1", **over: Any
) -> dict[str, Any]:
    info = cat.get(name)
    assert info is not None
    spec: dict[str, Any] = {
        "name": "t",
        "dataset": {"name": name, "sha256": info.sha256},
        "strategy": {"name": "pairs_zscore", "version": "1"},
        "grid": {"lookback": [20, 40], "entry_z": [1.5, 2.0]},
        "fixed_params": {"exit_z": 0.5},
        "seeds": [0],
    }
    spec.update(over)
    return spec
