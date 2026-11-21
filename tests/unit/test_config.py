"""Unit tests for Settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quantgrid.config import Settings, get_settings, reset_settings_cache


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("QG_"):
            monkeypatch.delenv(key)
    reset_settings_cache()
    yield  # type: ignore[misc]
    reset_settings_cache()


def test_defaults() -> None:
    s = Settings()
    assert s.env == "dev"
    assert s.api_key_map() == {"dev-key": "dev-user"}
    assert s.allowed_datasets() is None
    assert s.effective_broker_url == s.redis_url
    assert s.allow_fault_injection is False
    assert s.log_level == "INFO"
    assert isinstance(s.data_dir, Path)
    assert s.heartbeat_interval_s < s.lease_ttl_s


def test_broker_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QG_BROKER_URL", "redis://other:1/2")
    assert Settings().effective_broker_url == "redis://other:1/2"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("k1:u1", {"k1": "u1"}),
        ("k1:u1,k2:u2", {"k1": "u1", "k2": "u2"}),
        (" k1:u1 , k2:u2 ,", {"k1": "u1", "k2": "u2"}),
        ("", {}),
        (" , ,", {}),
        ("k:u:with:colons", {"k": "u:with:colons"}),
        ("k:a,k:b", {"k": "b"}),
    ],
)
def test_api_key_map_parsing(raw: str, expected: dict[str, str]) -> None:
    assert Settings(api_keys=raw).api_key_map() == expected


@pytest.mark.parametrize("raw", ["nocolon", ":user", "key:", ":", "ok:u,bad", "ok:u,:x"])
def test_api_key_map_invalid(raw: str) -> None:
    with pytest.raises(ValueError, match="QG_API_KEYS"):
        Settings(api_keys=raw).api_key_map()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("  ", None),
        (",,", None),
        ("a", ("a",)),
        ("a,b", ("a", "b")),
        (" a , b ,, c ", ("a", "b", "c")),
    ],
)
def test_allowed_datasets(raw: str, expected: tuple[str, ...] | None) -> None:
    assert Settings(quota_allowed_datasets=raw).allowed_datasets() == expected


def test_env_overrides_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first  # cached
    monkeypatch.setenv("QG_LEASE_TTL_S", "42.5")
    monkeypatch.setenv("QG_QUEUE_NAME", "q2")
    monkeypatch.setenv("QG_WORKER_CONCURRENCY", "4")
    monkeypatch.setenv("QG_ALLOW_FAULT_INJECTION", "1")
    monkeypatch.setenv("QG_LOG_LEVEL", "debug")
    monkeypatch.setenv("QG_ARTIFACT_ROOT", "/tmp/x")
    assert get_settings() is first  # still cached until reset
    reset_settings_cache()
    s = get_settings()
    assert s is not first
    assert s.lease_ttl_s == 42.5 and s.queue_name == "q2" and s.worker_concurrency == 4
    assert s.allow_fault_injection is True
    assert s.log_level == "DEBUG"
    assert s.artifact_root == Path("/tmp/x")


def test_unknown_env_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QG_TOTALLY_UNKNOWN", "1")
    Settings()


def test_settings_frozen() -> None:
    with pytest.raises(ValidationError):
        Settings().queue_name = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("QG_ENV", "staging"),
        ("QG_LEASE_TTL_S", "0"),
        ("QG_LEASE_TTL_S", "-1"),
        ("QG_LEASE_TTL_S", "abc"),
        ("QG_WORKER_CONCURRENCY", "0"),
        ("QG_WORKER_CONCURRENCY", "65"),
        ("QG_WORKER_METRICS_PORT", "70000"),
        ("QG_DB_POOL_SIZE", "0"),
        ("QG_LLM_PROVIDER", "openai"),
        ("QG_LOG_FORMAT", "xml"),
        ("QG_QUOTA_MAX_TASKS_PER_BATCH", "0"),
        ("QG_QUOTA_MAX_MEMORY_MB_PER_TASK", "63"),
        ("QG_MAX_REQUEST_BYTES", "10"),
        ("QG_AGENT_MAX_STEPS", "51"),
        ("QG_DISPATCH_WINDOW_FACTOR", "0.5"),
    ],
)
def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch, var: str, value: str) -> None:
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings()


def test_small_scheduler_values_allowed_for_tests() -> None:
    s = Settings(
        lease_ttl_s=0.06,
        worker_dead_after_s=0.06,
        dispatch_lost_after_s=0.06,
        heartbeat_interval_s=0.06,
    )
    assert s.lease_ttl_s == 0.06
    with pytest.raises(ValidationError):
        Settings(lease_ttl_s=0.05)
    with pytest.raises(ValidationError):
        Settings(worker_dead_after_s=0.05)
    with pytest.raises(ValidationError):
        Settings(dispatch_lost_after_s=0.05)


def test_bool_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, expected in [("true", True), ("0", False), ("yes", True), ("off", False)]:
        monkeypatch.setenv("QG_STRICT_DATASET_CHECK", raw)
        assert Settings().strict_dataset_check is expected
    monkeypatch.setenv("QG_STRICT_DATASET_CHECK", "maybe")
    with pytest.raises(ValidationError):
        Settings()


def test_api_keys_env_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QG_API_KEYS", "alpha:ann,beta:bob")
    assert Settings().api_key_map() == {"alpha": "ann", "beta": "bob"}
