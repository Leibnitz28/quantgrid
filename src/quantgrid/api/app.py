"""FastAPI application factory.

Run with ``quantgrid api`` or ``uvicorn quantgrid.api.app:create_app --factory``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sqlalchemy import Engine

import quantgrid
from quantgrid.api import routes_agent, routes_batches, routes_meta, routes_tasks
from quantgrid.api.context import AppContext
from quantgrid.api.errors import install_error_handlers
from quantgrid.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from quantgrid.api.security import ApiKeyAuthenticator, RateLimiter, RedisRateLimiter
from quantgrid.config import Settings, get_settings
from quantgrid.domain.policy import DatasetCatalog, PolicyService


def default_provider_factory(settings: Settings) -> Callable[[], Any]:
    """LLM provider chosen by environment only (``QG_LLM_PROVIDER``); FakeLLM by default."""

    def make() -> Any:
        if settings.llm_provider == "anthropic":
            from quantgrid.agents.anthropic_provider import AnthropicProvider

            return AnthropicProvider.from_settings(settings)
        from quantgrid.agents.fake_llm import FakeLLM

        return FakeLLM()

    return make


def create_app(
    settings: Settings | None = None,
    *,
    engine: Engine | None = None,
    catalog: DatasetCatalog | None = None,
    store: Any | None = None,
    limiter: RateLimiter | None = None,
    provider_factory: Callable[[], Any] | None = None,
    configure_observability: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    if configure_observability:
        from quantgrid.observability.logging import configure_logging
        from quantgrid.observability.tracing import init_tracing

        configure_logging(settings.log_level, settings.log_format, service="quantgrid-api")
        init_tracing("quantgrid-api", settings.otel_exporter, settings.otel_endpoint)
    if engine is None:
        from quantgrid.storage.db import make_engine

        engine = make_engine(application_name="qg-api")
    if catalog is None:
        from quantgrid.domain.datasets import get_catalog

        catalog = get_catalog()
    if store is None:
        from quantgrid.storage.artifacts import LocalArtifactStore

        store = LocalArtifactStore.from_settings(settings)
    ctx = AppContext(
        settings=settings,
        engine=engine,
        catalog=catalog,
        policy=PolicyService(catalog),
        store=store,
        limiter=limiter or RedisRateLimiter(settings.redis_url),
        auth=ApiKeyAuthenticator(settings.api_key_map()),
        provider_factory=provider_factory or default_provider_factory(settings),
        redis_url=settings.redis_url,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        ctx.engine.dispose()

    app = FastAPI(
        title="QuantGrid API",
        version=quantgrid.__version__,
        description=(
            "Fault-tolerant scheduler for reproducible quant-research backtests. Offline "
            "research on synthetic or licensed data only - no live trading."
        ),
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    install_error_handlers(app)
    # Starlette: the last middleware added is the outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(routes_meta.router)
    app.include_router(routes_batches.router)
    app.include_router(routes_tasks.router)
    app.include_router(routes_agent.router)
    return app
