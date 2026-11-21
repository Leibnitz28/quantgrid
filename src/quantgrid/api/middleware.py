"""Pure-ASGI middleware: correlation IDs, request logging/metrics, payload size limit."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from quantgrid.api.errors import error_body
from quantgrid.domain.errors import ErrorCode

log = logging.getLogger("quantgrid.api.access")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestContextMiddleware:
    """Sets ``request.state.request_id`` (client ``X-Request-ID`` if well-formed, else a UUID),
    echoes and logs it, and records HTTP metrics by route template (bounded cardinality)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status_holder = {"status": 500}
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                raw = list(message.get("headers", []))
                if not any(k.lower() == b"x-request-id" for k, _ in raw):
                    raw.append((b"x-request-id", request_id.encode()))
                message["headers"] = raw
            await send(message)

        cm = _bind_context(request_id=request_id)
        cm.__enter__()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = scope.get("route")
            template = getattr(route, "path", None) or "unmatched"
            method = scope.get("method", "?")
            status = status_holder["status"]
            _record_metrics(method, template, status, elapsed)
            log.info(
                "request",
                extra={
                    "method": method,
                    "route": template,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            cm.__exit__(None, None, None)


class BodySizeLimitMiddleware:
    """413 for bodies over ``max_bytes``; chunked bodies are counted while streaming."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for key, value in scope["headers"]:
            if key.lower() == b"content-length":
                try:
                    too_big = int(value) > self.max_bytes
                except ValueError:
                    too_big = True
                if too_big:
                    await self._reject(scope, send)
                    return
        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    raise _PayloadTooLarge
            return message

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _PayloadTooLarge:
            if not response_started:
                await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        request_id = scope.get("state", {}).get("request_id")
        body = json.dumps(
            error_body(
                ErrorCode.PAYLOAD_TOO_LARGE,
                f"request body exceeds {self.max_bytes} bytes",
                request_id,
            )
        ).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if request_id:
            headers.append((b"x-request-id", request_id.encode()))
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})


class _PayloadTooLarge(Exception):
    pass


def _record_metrics(method: str, route: str, status: int, seconds: float) -> None:
    try:
        from quantgrid.observability import metrics
    except ImportError:  # pragma: no cover
        return
    counter = getattr(metrics, "http_requests_total", None)
    hist = getattr(metrics, "http_request_duration_seconds", None)
    if counter is not None:
        counter.labels(method=method, route=route, status=str(status)).inc()
    if hist is not None:
        hist.labels(method=method, route=route).observe(seconds)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _bind_context(**kv: Any) -> Any:
    try:
        from quantgrid.observability.logging import bind_context
    except ImportError:  # pragma: no cover
        return _NullContext()
    return bind_context(**kv)
