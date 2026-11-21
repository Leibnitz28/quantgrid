"""Uniform error envelope and exception handlers.

Every error response has the shape::

    {"error": {"type": "VALIDATION_ERROR", "message": "...", "details": ..., "request_id": "..."}}

``type`` is one of :class:`quantgrid.domain.errors.ErrorCode`. Validation details never
echo the offending input back (inputs can be large or hostile).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from quantgrid.domain.errors import ErrorCode, QuantGridError, RateLimited

log = logging.getLogger("quantgrid.api")

_HTTP_TO_CODE = {
    400: ErrorCode.VALIDATION_ERROR,
    401: ErrorCode.UNAUTHORIZED,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.VALIDATION_ERROR,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    415: ErrorCode.VALIDATION_ERROR,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
}


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_body(
    code: ErrorCode | str, message: str, request_id: str | None, details: Any = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"type": str(code), "message": message, "request_id": request_id}
    if details is not None:
        body["details"] = details
    return {"error": body}


def error_response(
    status: int,
    code: ErrorCode | str,
    message: str,
    request_id: str | None,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    hdrs = dict(headers or {})
    if request_id:
        hdrs["X-Request-ID"] = request_id
    return JSONResponse(
        error_body(code, message, request_id, details), status_code=status, headers=hdrs
    )


def _clean_validation_errors(errors: list[Any]) -> list[dict[str, Any]]:
    cleaned = []
    for err in errors[:50]:
        cleaned.append(
            {
                "loc": [str(p) for p in err.get("loc", ())],
                "msg": str(err.get("msg", ""))[:300],
                "type": str(err.get("type", "")),
            }
        )
    return cleaned


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(QuantGridError)
    async def _qg_error(request: Request, exc: QuantGridError) -> JSONResponse:
        headers = {}
        if isinstance(exc, RateLimited):
            headers["Retry-After"] = str(max(1, exc.retry_after_s))
        if exc.http_status >= 500:
            log.error("request failed", extra={"error_type": str(exc.code), "error": exc.message})
        return error_response(
            exc.http_status, exc.code, exc.message, request_id_of(request), exc.details, headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(
            422,
            ErrorCode.VALIDATION_ERROR,
            "request validation failed",
            request_id_of(request),
            _clean_validation_errors(list(exc.errors())),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        return error_response(
            exc.status_code,
            code,
            str(exc.detail),
            request_id_of(request),
            headers=dict(exc.headers or {}),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path})
        return error_response(
            500, ErrorCode.INTERNAL_ERROR, "internal server error", request_id_of(request)
        )
