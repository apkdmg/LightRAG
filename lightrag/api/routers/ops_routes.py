"""Admin-only API behind the Operations tab. Every route needs the admin role.

The router is always mounted, next to the Insights router. Routes report what
is available rather than failing when a feature cannot run (for example
without Postgres storage), and never instantiate a workspace or loop over the
shared pipeline state.

Conventions for every route here:

- ``Cache-Control: no-store`` on every response, errors included.
- Successful payloads carry ``generated_at`` (UTC, ISO 8601).
- Errors are ``{"detail": <code>, "request_id": <id>}``. An unexpected
  exception is logged with its traceback (redacted by the log handlers) and
  answered with ``internal_error``; its text never reaches the client.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from lightrag.api.dependencies import UserInfo, require_admin
from lightrag.api.ops.context import (
    REQUEST_ID_HEADER,
    current_request_id,
    new_request_id,
)

logger = logging.getLogger("lightrag.api.ops")

POSTGRES_KV_STORAGE = "PGKVStorage"
_WORKERS_LIVE_TTL_S = 10.0

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "invalid_request",
    429: "rate_limited",
    500: "internal_error",
    503: "unavailable",
}


@dataclass(frozen=True)
class OpsServerInfo:
    """What the ops routes need from the server that mounted them."""

    rag: Any
    workers: Optional[int]
    version: str


def configure_ops(app: Any, *, rag: Any, workers: Optional[int], version: Any) -> None:
    """Record the server's own LightRAG instance and process settings."""
    app.state.ops_server = OpsServerInfo(
        rag=rag, workers=workers, version=str(version or "")
    )


def _no_store(headers: Optional[dict[str, str]] = None) -> dict[str, str]:
    merged = dict(headers or {})
    merged["Cache-Control"] = "no-store"
    return merged


def _err(
    status_code: int, code: str, headers: Optional[dict[str, str]] = None
) -> JSONResponse:
    """The one error shape: ``{"detail": code, "request_id": id}``."""
    request_id = current_request_id()
    extra = dict(headers or {})
    if request_id is None:
        # Mounted without RequestIdMiddleware (tests, embedding apps): still
        # hand the caller a reference and echo it like the middleware would.
        request_id = new_request_id()
        extra[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content={"detail": code, "request_id": request_id},
        headers=_no_store(extra),
    )


def _http_error_code(exc: StarletteHTTPException) -> str:
    # Details raised by our own dependencies are fixed strings; anything else
    # (lists, dicts) is replaced by a code for its status.
    if isinstance(exc.detail, str) and exc.detail:
        return exc.detail
    return _STATUS_CODES.get(exc.status_code, "error")


class OpsRoute(APIRoute):
    """Applies the error shape and ``no-store`` to a route, dependencies included."""

    def get_route_handler(self) -> Callable[[Request], Any]:
        original = super().get_route_handler()
        path = self.path

        async def handler(request: Request) -> Response:
            try:
                response = await original(request)
            except StarletteHTTPException as exc:
                return _err(
                    exc.status_code,
                    _http_error_code(exc),
                    headers=getattr(exc, "headers", None),
                )
            except RequestValidationError:
                return _err(422, "invalid_request")
            except Exception:
                logger.error(
                    "Operations console request failed on %s", path, exc_info=True
                )
                return _err(500, "internal_error")
            response.headers["Cache-Control"] = "no-store"
            return response

        return handler


router = APIRouter(
    prefix="/admin/ops",
    tags=["ops"],
    dependencies=[Depends(require_admin)],
    route_class=OpsRoute,
)


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _server_info(request: Request) -> Optional[OpsServerInfo]:
    return getattr(request.app.state, "ops_server", None)


def _data_rag(request: Request) -> Any:
    """The instance whose storages hold every workspace's rows.

    Multi-tenant servers keep it in ``app.state.default_rag``; a single-tenant
    server has only its own instance.
    """
    rag = getattr(request.app.state, "default_rag", None)
    if rag is not None:
        return rag
    info = _server_info(request)
    return info.rag if info is not None else None


def _kv_storage_name(rag: Any) -> Optional[str]:
    name = getattr(rag, "kv_storage", None) if rag is not None else None
    return name if isinstance(name, str) else None


def process_mode() -> str:
    """``gunicorn`` under the Gunicorn arbiter (any entry point), else ``uvicorn``."""
    if "LIGHTRAG_GUNICORN_MODE" in os.environ or "gunicorn.arbiter" in sys.modules:
        return "gunicorn"
    return "uvicorn"


def _expected_workers(mode: str, configured: Optional[int]) -> int:
    if mode != "gunicorn":
        return 1
    if isinstance(configured, int) and configured > 0:
        return configured
    try:
        return max(1, int(os.getenv("WORKERS", "1")))
    except ValueError:
        return 1


_workers_live_cache: dict[str, Any] = {"at": 0.0, "value": None}
_workers_live_lock = threading.Lock()


def _count_gunicorn_workers() -> Optional[int]:
    """Live children of this worker's Gunicorn master, counted by the OS.

    Host-local and cached briefly; the heartbeat table replaces it once
    workers report themselves.
    """
    now = time.monotonic()
    with _workers_live_lock:
        if now - _workers_live_cache["at"] < _WORKERS_LIVE_TTL_S:
            return _workers_live_cache["value"]
    value: Optional[int]
    try:
        import psutil

        master = psutil.Process(os.getppid())
        value = sum(
            1
            for child in master.children(recursive=False)
            if child.status() != psutil.STATUS_ZOMBIE
        )
    except Exception:
        value = None
    with _workers_live_lock:
        _workers_live_cache["at"] = now
        _workers_live_cache["value"] = value
    return value


def operator_accounts() -> set[str]:
    """``OPS_OPERATOR_ACCOUNTS`` (comma-separated, case-insensitive)."""
    raw = os.getenv("OPS_OPERATOR_ACCOUNTS", "")
    return {a.strip().lower() for a in raw.split(",") if a.strip()}


def is_operator(user: UserInfo) -> bool:
    """When no operator list is set, every admin is an operator (D8)."""
    accounts = operator_accounts()
    if not accounts:
        return True
    candidates = {user.username}
    metadata = user.metadata or {}
    for key in ("email", "preferred_username"):
        value = metadata.get(key)
        if isinstance(value, str):
            candidates.add(value)
    return any(c and c.strip().lower() in accounts for c in candidates)


@router.get("/status")
async def status(
    request: Request, user: UserInfo = Depends(require_admin)
) -> dict[str, Any]:
    """Availability, features and process mode; also the UI's admin probe."""
    info = _server_info(request)
    kv_storage = _kv_storage_name(_data_rag(request))
    available = kv_storage == POSTGRES_KV_STORAGE
    mode = process_mode()
    expected = _expected_workers(mode, info.workers if info else None)
    workers_live = 1 if mode == "uvicorn" else _count_gunicorn_workers()
    payload: dict[str, Any] = {
        "generated_at": _generated_at(),
        "available": available,
        "features": {
            "durable": False,
            "runs": False,
            "logs": False,
            "probes": False,
            "alerts": False,
            "insights": getattr(request.app.state, "insights", None) is not None,
            "release_busy": False,
            "gpu": False,
        },
        "process": {
            "mode": mode,
            "expected_workers": expected,
            "workers_live": workers_live,
        },
        "operator": is_operator(user),
        "version": info.version if info else "",
    }
    if not available:
        payload["unavailable_reason"] = "postgres_required"
    return payload
