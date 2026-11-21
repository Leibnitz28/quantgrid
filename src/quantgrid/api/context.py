"""Application context (dependency container) and FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header, Request
from sqlalchemy import Engine

from quantgrid.api.security import ApiKeyAuthenticator, Principal, RateLimiter
from quantgrid.config import Settings
from quantgrid.domain.policy import DatasetCatalog, PolicyService, QuotaPolicy


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    catalog: DatasetCatalog
    policy: PolicyService
    store: Any  # quantgrid.storage.artifacts.ArtifactStore
    limiter: RateLimiter
    auth: ApiKeyAuthenticator
    provider_factory: Callable[[], Any]  # -> quantgrid.agents LLM provider
    redis_url: str | None = None

    def quota_for(self, user_id: str) -> QuotaPolicy:
        """One deployment-wide per-user quota from settings."""
        return QuotaPolicy.from_settings(self.settings)


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def get_principal(
    ctx: AppContext = Depends(get_ctx),  # noqa: B008 - FastAPI dependency
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    principal = ctx.auth.authenticate(x_api_key)
    ctx.limiter.hit("api", principal.user_id, ctx.settings.rate_limit_per_minute)
    return principal


def get_planner_principal(
    ctx: AppContext = Depends(get_ctx),  # noqa: B008
    principal: Principal = Depends(get_principal),  # noqa: B008
) -> Principal:
    ctx.limiter.hit("planner", principal.user_id, ctx.settings.planner_rate_limit_per_minute)
    return principal
