"""Runtime configuration, read from ``QG_*`` environment variables (see ``.env.example``).

Secrets (e.g. ``ANTHROPIC_API_KEY`` for the optional real LLM provider, API keys for
clients) only ever come from the environment; nothing secret is committed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QG_", extra="ignore", frozen=True)

    env: Literal["dev", "test", "prod"] = "dev"

    # --- infrastructure ---------------------------------------------------------------
    database_url: str = "postgresql+psycopg://quantgrid:quantgrid@127.0.0.1:5433/quantgrid"
    redis_url: str = "redis://127.0.0.1:6380/0"
    broker_url: str | None = None  # defaults to redis_url
    queue_name: str = "quantgrid.tasks"
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_statement_timeout_ms: int = Field(default=30_000, ge=100)

    # --- storage ----------------------------------------------------------------------
    data_dir: Path = Path("var/data")  # materialised datasets
    artifact_root: Path = Path("var/artifacts")  # content-addressed result artifacts
    auto_materialize_datasets: bool = True  # generate registered synthetic data if missing

    # --- scheduler --------------------------------------------------------------------
    lease_ttl_s: float = Field(default=15.0, gt=0.05)
    heartbeat_interval_s: float = Field(default=3.0, gt=0.05)
    worker_dead_after_s: float = Field(default=15.0, gt=0.05)
    coordinator_tick_s: float = Field(default=0.5, gt=0.01)
    dispatch_window_factor: float = Field(default=2.0, ge=1.0)
    dispatch_min_window: int = Field(default=4, ge=1)
    dispatch_lost_after_s: float = Field(default=120.0, gt=0.05)
    retry_backoff_base_s: float = Field(default=1.0, ge=0.0)
    retry_backoff_max_s: float = Field(default=30.0, ge=0.0)
    reconcile_batch_limit: int = Field(default=500, ge=1)

    # --- worker -----------------------------------------------------------------------
    worker_id: str | None = None  # default: <hostname>-<pid>
    worker_concurrency: int = Field(default=1, ge=1, le=64)
    worker_metrics_port: int = Field(default=0, ge=0, le=65535)  # 0 = disabled
    strict_dataset_check: bool = True  # verify dataset checksum on every load

    # --- coordinator ------------------------------------------------------------------
    coordinator_metrics_port: int = Field(default=0, ge=0, le=65535)  # 0 = disabled

    # --- API --------------------------------------------------------------------------
    api_keys: str = "dev-key:dev-user"  # comma separated "<key>:<user_id>" pairs
    max_request_bytes: int = Field(default=256 * 1024, ge=1024)
    rate_limit_per_minute: int = Field(default=600, ge=1)
    planner_rate_limit_per_minute: int = Field(default=30, ge=1)

    # --- quotas (defaults applied per user; see domain/policy.py) ---------------------
    quota_max_tasks_per_batch: int = Field(default=5_000, ge=1)
    quota_max_runtime_s_per_task: int = Field(default=900, ge=1)
    quota_max_memory_mb_per_task: int = Field(default=2_048, ge=64)
    quota_max_worker_seconds_per_batch: int = Field(default=12 * 3600, ge=1)
    quota_max_active_batches: int = Field(default=20, ge=1)
    quota_max_attempts: int = Field(default=10, ge=1)
    quota_max_bootstrap_samples: int = Field(default=20_000, ge=0)
    quota_allowed_datasets: str = ""  # comma separated; empty = every registered dataset
    allow_fault_injection: bool = False

    # --- observability ----------------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    otel_exporter: Literal["none", "console", "otlp"] = "none"
    otel_endpoint: str | None = None
    service_name: str = "quantgrid"

    # --- AI planner -------------------------------------------------------------------
    llm_provider: Literal["fake", "anthropic"] = "fake"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_timeout_s: float = Field(default=60.0, gt=0)
    agent_max_steps: int = Field(default=8, ge=1, le=50)
    agent_max_tool_calls: int = Field(default=16, ge=1, le=200)
    agent_timeout_s: float = Field(default=120.0, gt=0)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def effective_broker_url(self) -> str:
        return self.broker_url or self.redis_url

    def api_key_map(self) -> dict[str, str]:
        """Parse ``api_keys`` into ``{key: user_id}``."""
        out: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, sep, user = pair.partition(":")
            if not sep or not key or not user:
                raise ValueError("QG_API_KEYS entries must look like '<key>:<user_id>'")
            out[key] = user
        return out

    def allowed_datasets(self) -> tuple[str, ...] | None:
        names = tuple(n.strip() for n in self.quota_allowed_datasets.split(",") if n.strip())
        return names or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """For tests that change QG_* environment variables."""
    get_settings.cache_clear()
