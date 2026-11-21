"""Health, readiness, metrics and dataset catalogue endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from sqlalchemy import text

from quantgrid.api.context import AppContext, get_ctx, get_principal
from quantgrid.api.security import Principal

router = APIRouter(tags=["meta"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is serving requests."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(ctx: Annotated[AppContext, Depends(get_ctx)]) -> JSONResponse:
    """Readiness: PostgreSQL (and Redis, when configured) answer."""
    checks: dict[str, Any] = {}
    try:
        with ctx.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {type(exc).__name__}"
    if ctx.redis_url:
        try:
            import redis

            redis.Redis.from_url(ctx.redis_url, socket_timeout=1, socket_connect_timeout=1).ping()
            checks["redis"] = "ok"
        except Exception as exc:
            checks["redis"] = f"error: {type(exc).__name__}"
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503
    )


@router.get("/metrics")
def metrics() -> Response:
    from quantgrid.observability.metrics import render_latest

    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)


@router.get("/v1/datasets")
def list_datasets(
    ctx: Annotated[AppContext, Depends(get_ctx)],
    _principal: Annotated[Principal, Depends(get_principal)],
) -> dict[str, Any]:
    """Registered datasets with their recorded content signatures."""
    return {"items": [d.model_dump(mode="json") for d in ctx.catalog.list()]}
