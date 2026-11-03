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
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from lightrag.api.dependencies import UserInfo, require_admin
from lightrag.api.ops.auth import require_live_admin, require_ops_client
from lightrag.api.ops.health import dep_card
from lightrag.api.ops.context import (
    REQUEST_ID_HEADER,
    current_request_id,
    new_request_id,
)
from lightrag.api.ops.service import (
    AUDIT_EVENT_KIND,
    DEFAULT_THRESHOLDS,
    EVENT_KINDS,
    OpsBadRequest,
    OpsDataUnavailable,
    OpsNotFound,
    OpsService,
    SEARCH_MAX_LENGTH,
)
from lightrag.api.ops.probes import SMTP_TEST_INTERVAL_S, recheck_limit
from lightrag.api.ops.redact import sanitize_config
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import BACKLOG_STATUSES, Page
from lightrag.kg import shared_storage

logger = logging.getLogger("lightrag.api.ops")

POSTGRES_KV_STORAGE = "PGKVStorage"
_WORKERS_LIVE_TTL_S = 10.0
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
_DOC_STATUSES = (*BACKLOG_STATUSES, "processed", "preprocessed")

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
    # Returns the server's status block (``/health`` configuration shape).
    status_provider: Optional[Callable[[], Any]] = None


def configure_ops(
    app: Any,
    *,
    rag: Any,
    workers: Optional[int],
    version: Any,
    status_provider: Optional[Callable[[], Any]] = None,
) -> None:
    """Record the server's own LightRAG instance and process settings."""
    app.state.ops_server = OpsServerInfo(
        rag=rag,
        workers=workers,
        version=str(version or ""),
        status_provider=status_provider,
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

    The master also owns the shared-state Manager process, which is not a
    worker; when its PID is not known the count is unknown rather than one
    too high (a dead worker would otherwise read as healthy). Host-local and
    cached briefly; the heartbeat table replaces it once workers report
    themselves.
    """
    now = time.monotonic()
    with _workers_live_lock:
        if now - _workers_live_cache["at"] < _WORKERS_LIVE_TTL_S:
            return _workers_live_cache["value"]
    value: Optional[int]
    try:
        import psutil

        manager = shared_storage.manager_pid()
        if manager is None:
            value = None
        else:
            master = psutil.Process(os.getppid())
            value = sum(
                1
                for child in master.children(recursive=False)
                if child.pid != manager and child.status() != psutil.STATUS_ZOMBIE
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


def _service(request: Request) -> OpsService:
    """This worker's ops service, or an unavailable one when ops did not start.

    The unavailable service still answers every route (with empty data and
    ``available: false``), so the console degrades instead of failing.
    """
    runtime = getattr(request.app.state, "ops", None)
    service = getattr(runtime, "service", None)
    if isinstance(service, OpsService):
        return service
    fallback = getattr(request.app.state, "ops_fallback_service", None)
    if not isinstance(fallback, OpsService):
        on_postgres = _kv_storage_name(_data_rag(request)) == POSTGRES_KV_STORAGE
        fallback = OpsService(
            store=None,
            reader=NamespaceReader(),
            app_state=request.app.state,
            unavailable_reason="not_started" if on_postgres else "postgres_required",
        )
        request.app.state.ops_fallback_service = fallback
    return fallback


def _workers(request: Request) -> dict[str, Any]:
    info = _server_info(request)
    mode = process_mode()
    return {
        "mode": mode,
        "expected": _expected_workers(mode, info.workers if info else None),
        "live": 1 if mode == "uvicorn" else _count_gunicorn_workers(),
    }


def _page(page: int, page_size: int) -> Page:
    return Page(limit=page_size, offset=(page - 1) * page_size)


def _payload(
    service: OpsService, body: dict[str, Any], *, needs_docs: bool = True
) -> dict[str, Any]:
    return {
        "generated_at": _generated_at(),
        **service.meta(needs_docs=needs_docs),
        **body,
    }


async def _answer(
    service: OpsService, call, empty: dict[str, Any], *, needs_docs: bool = True
) -> Any:
    """Run a service call and map its failures to the ops error shape.

    Without a working store the route answers ``empty`` with
    ``available: false`` rather than an error. ``needs_docs`` is False for
    routes that read only the ops tables (KV, audit, usage events), which
    work while doc status lives in another storage.
    """
    ready = service.docs_available if needs_docs else service.available
    if not ready and empty is not None:
        return _payload(service, empty, needs_docs=needs_docs)
    try:
        return _payload(service, await call(), needs_docs=needs_docs)
    except OpsBadRequest as e:
        return _err(400, e.code)
    except OpsNotFound:
        return _err(404, "not_found")
    except OpsDataUnavailable:
        return _err(503, "data_unavailable")


def _empty_page(page: int, page_size: int, **extra: Any) -> dict[str, Any]:
    return {"items": [], "page": page, "page_size": page_size, "total": 0, **extra}


def _csv_set(value: Optional[str]) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(p.strip() for p in value.split(",") if p.strip())


def _parse_time(value: Optional[str], default: datetime) -> datetime:
    if not value:
        return default
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise OpsBadRequest("invalid_time") from e
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


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
    service = _service(request)
    heartbeat_live = await service.live_workers()
    if heartbeat_live is not None:
        workers_live = heartbeat_live
    payload: dict[str, Any] = {
        "generated_at": _generated_at(),
        "available": available,
        "features": {
            "durable": service.available,
            "processing": service.docs_available,
            "runs": False,
            "logs": False,
            "probes": service.probes is not None,
            "heartbeats": service.heartbeat is not None,
            "alerts": False,
            "insights": getattr(request.app.state, "insights", None) is not None,
            "release_busy": False,
            "gpu": False,
        },
        "process": {
            "mode": mode,
            "expected_workers": expected,
            "workers_live": workers_live,
            "manager": service.reader.state,
        },
        "operator": is_operator(user),
        "version": info.version if info else "",
    }
    if not available:
        payload["unavailable_reason"] = "postgres_required"
    return payload


# -- Overview and processing (read-only) ----------------------------------------


@router.get("/overview")
async def overview(request: Request) -> dict[str, Any]:
    """State, KPI tiles, attention list, top active jobs and recent markers."""
    service = _service(request)
    try:
        body = await service.overview(_workers(request))
    except OpsDataUnavailable:
        return _err(503, "data_unavailable")
    return {"generated_at": _generated_at(), **body}


@router.get("/processing/summary")
async def processing_summary(request: Request) -> dict[str, Any]:
    service = _service(request)
    return await _answer(
        service,
        service.summary,
        {
            "backlog": None,
            "active": [],
            "active_count": 0,
            "stuck": {"runs": 0, "docs": 0, "reservations": 0},
            "email_queue": None,
            "pauses": {"count": 0, "by_slot_kind": {}, "capped": False},
            "thresholds": dict(DEFAULT_THRESHOLDS),
            "approximate": True,
        },
    )


@router.get("/processing/active")
async def processing_active(request: Request) -> dict[str, Any]:
    """Live jobs; approximate until run tracking: in-flight docs + shared state."""
    service = _service(request)
    return await _answer(
        service, service.active, {"runs": [], "approximate": True, "checked": 0}
    )


@router.get("/processing/stuck")
async def processing_stuck(
    request: Request,
    doc_older_than_min: Optional[int] = Query(None, ge=1, le=10080),
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> dict[str, Any]:
    service = _service(request)
    return await _answer(
        service,
        lambda: service.stuck(doc_older_than_min, _page(page, page_size)),
        {
            "orphaned": [],
            "hung": [],
            "busy_without_run": [],
            "reservations": [],
            "stale_docs": _empty_page(page, page_size),
            "thresholds": {
                "doc_minutes": doc_older_than_min
                or DEFAULT_THRESHOLDS["stale_doc_minutes"],
                "hung_minutes": DEFAULT_THRESHOLDS["hung_minutes"],
                "orphan_seconds": DEFAULT_THRESHOLDS["orphan_seconds"],
            },
            "checked_classes": [],
        },
    )


@router.get("/processing/workspaces")
async def processing_workspaces(
    request: Request,
    q: Optional[str] = Query(None, max_length=SEARCH_MAX_LENGTH),
    has: Optional[str] = Query(None, max_length=64),
    sort: str = Query("oldest", max_length=32),
    dir: str = Query("asc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> dict[str, Any]:
    """Workspaces with backlog: server-side search, filters, sort and paging."""
    service = _service(request)
    return await _answer(
        service,
        lambda: service.workspaces(
            q=q,
            has=_csv_set(has),
            sort=sort,
            descending=dir == "desc",
            page=_page(page, page_size),
        ),
        _empty_page(page, page_size, sort=sort, dir=dir),
    )


@router.get("/processing/failures")
async def processing_failures(
    request: Request,
    window: str = Query("24h", max_length=8),
    group: str = Query("error_kind", max_length=32),
    include_cancelled: bool = False,
) -> dict[str, Any]:
    service = _service(request)
    return await _answer(
        service,
        lambda: service.failures(window, group, include_cancelled),
        {
            "window": window,
            "group": group,
            "groups": [],
            "durable": False,
            "legacy": [],
            "excluded": {"dup": 0, "cancelled": 0},
        },
    )


@router.get("/processing/failures/docs")
async def processing_failure_docs(
    request: Request,
    key: Optional[str] = Query(None, max_length=255),
    group: str = Query("signature", max_length=32),
    window: str = Query("24h", max_length=8),
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> dict[str, Any]:
    service = _service(request)
    return await _answer(
        service,
        lambda: service.failure_docs(key or "", group, window, _page(page, page_size)),
        _empty_page(page, page_size),
    )


@router.get("/email-queue")
async def email_queue(
    request: Request,
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> dict[str, Any]:
    """Waiting email re-syncs per workspace (counts only, never content)."""
    service = _service(request)
    return await _answer(
        service,
        lambda: service.email_queue(_page(page, page_size)),
        {
            "total": 0,
            "workspaces": 0,
            "bad": 0,
            "longest_pending_s": None,
            "scanned_at": None,
            "items": [],
            "page": page,
            "page_size": page_size,
            "total_items": 0,
        },
        needs_docs=False,
    )


@router.get("/pauses")
async def pauses(
    request: Request,
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> dict[str, Any]:
    """Provider pauses by slot and kind (capped candidate set, approximate)."""
    service = _service(request)
    return await _answer(
        service,
        lambda: service.pauses(_page(page, page_size)),
        {
            "by_slot_kind": {},
            "items": [],
            "total": 0,
            "checked": 0,
            "capped": False,
            "approximate": True,
            "page": page,
            "page_size": page_size,
        },
    )


@router.get("/workspaces/{ws}")
async def workspace_detail(request: Request, ws: str) -> dict[str, Any]:
    """Workspace drawer. Never loads the workspace; idle ones show SQL counts."""
    service = _service(request)
    return await _answer(service, lambda: service.workspace(ws), None)


@router.get("/workspaces/{ws}/feed")
async def workspace_feed(
    request: Request,
    ws: str,
    offset: Optional[int] = Query(None, ge=0),
    limit: int = Query(200, ge=1, le=200),
) -> dict[str, Any]:
    """Redacted slice of the workspace's live pipeline messages."""
    service = _service(request)
    return await _answer(service, lambda: service.feed(ws, offset, limit), None)


@router.get("/workspaces/{ws}/documents")
async def workspace_documents(
    request: Request,
    ws: str,
    status: Optional[str] = Query(None, max_length=32),
    page: int = Query(1, ge=1, le=10000),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> dict[str, Any]:
    service = _service(request)
    if status is not None and status not in _DOC_STATUSES:
        return _err(400, "invalid_status")
    return await _answer(
        service,
        lambda: service.documents(ws, status, _page(page, page_size)),
        _empty_page(page, page_size),
    )


@router.get("/workspaces/{ws}/documents/{doc_id}")
async def workspace_document(request: Request, ws: str, doc_id: str) -> dict[str, Any]:
    """Document drawer: masked file name, stage timeline, its LLM calls."""
    service = _service(request)
    return await _answer(
        service,
        lambda: service.document(ws, doc_id),
        {"doc": None, "failures": [], "calls": [], "calls_available": False},
    )


@router.get("/events")
async def events(
    request: Request,
    from_: Optional[str] = Query(None, alias="from", max_length=40),
    to: Optional[str] = Query(None, max_length=40),
    kinds: Optional[str] = Query(None, max_length=256),
) -> dict[str, Any]:
    """Timeline markers: system events merged with ops actions, newest first."""
    service = _service(request)
    now = datetime.now(timezone.utc)

    async def call() -> dict[str, Any]:
        end = _parse_time(to, now)
        start = _parse_time(from_, end - timedelta(hours=6))
        wanted = sorted(_csv_set(kinds)) or None
        return await service.events(start, end, wanted)

    return await _answer(
        service,
        call,
        {"items": [], "kinds": [*EVENT_KINDS, AUDIT_EVENT_KIND]},
        needs_docs=False,
    )


# -- Health and workers ----------------------------------------------------------


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Dependency grid: latest probe, 48-slot availability strip, detail."""
    service = _service(request)
    return _payload(service, await service.health(process_mode()), needs_docs=False)


@router.get("/health/{dep}")
async def health_dependency(
    request: Request, dep: str, hours: int = Query(24, ge=1, le=168)
) -> dict[str, Any]:
    """Dependency drawer: latency series, recent results, error classes and,
    for Postgres, blocked and blocking backends (never query text)."""
    service = _service(request)
    try:
        body = await service.dependency(dep, hours, process_mode())
    except OpsBadRequest as e:
        return _err(400, e.code)
    except OpsNotFound:
        return _err(404, "not_found")
    except OpsDataUnavailable:
        return _err(503, "data_unavailable")
    return _payload(service, body, needs_docs=False)


@router.get("/workers")
async def workers(request: Request) -> dict[str, Any]:
    """Every worker that reported in the last day, live ones first."""
    service = _service(request)
    try:
        body = await service.workers_view(_workers(request))
    except OpsDataUnavailable:
        return _err(503, "data_unavailable")
    return _payload(service, body, needs_docs=False)


@router.get("/workers/{worker_id}/history")
async def worker_history(
    request: Request, worker_id: str, hours: int = Query(24, ge=1, le=168)
) -> dict[str, Any]:
    """Per-minute RSS, CPU, loop lag and pool wait of one worker."""
    service = _service(request)
    return await _answer(
        service,
        lambda: service.worker_history(worker_id, hours),
        {"worker_id": worker_id, "hours": hours, "items": []},
        needs_docs=False,
    )


@router.get("/config")
async def config(request: Request) -> dict[str, Any]:
    """The server's configuration block: secrets redacted, URLs without userinfo."""
    info = _server_info(request)
    provider = info.status_provider if info is not None else None
    if provider is None:
        return {
            "generated_at": _generated_at(),
            "configuration": {},
            "unavailable_reason": "not_configured",
        }
    status_block = provider()
    if hasattr(status_block, "__await__"):
        status_block = await status_block
    status_block = sanitize_config(dict(status_block or {}))
    return {
        "generated_at": _generated_at(),
        "configuration": status_block.get("configuration") or {},
        "working_directory": status_block.get("working_directory"),
        "input_directory": status_block.get("input_directory"),
        "core_version": status_block.get("core_version"),
        "api_version": status_block.get("api_version"),
    }


_POST_GUARDS = [Depends(require_live_admin), Depends(require_ops_client)]


def _rate_limited(retry_after: float) -> JSONResponse:
    return _err(429, "rate_limited", headers={"Retry-After": str(int(retry_after))})


@router.post("/health/{dep}/probe", dependencies=_POST_GUARDS)
async def recheck_dependency(request: Request, dep: str) -> dict[str, Any]:
    """Check one dependency now.

    Host-wide limit: once per 10 s per dependency; once a minute for the
    slow-cadence probes, and SMTP shares the SMTP test's once-a-minute claim.
    """
    service = _service(request)
    runner = service.probes
    if runner is None:
        return _err(503, "probes_disabled")
    try:
        spec = service.known_dep(dep)
    except OpsBadRequest as e:
        return _err(400, e.code)
    if spec is None:
        return _err(404, "not_found")
    if not spec.checkable or spec.passive:
        return _err(400, "not_checkable")
    claim, interval_s = recheck_limit(spec)
    if not await runner.claim(claim, interval_s):
        return _rate_limited(interval_s)
    record = await runner.probe_now(dep)
    service.cache.forget(("probe_records",))
    service.cache.forget(("dependency", dep, 24))
    if record is None:
        return _err(503, "probe_failed")
    card = dep_card(spec, record, None, service.clock())
    card.pop("strip", None)
    return {"generated_at": _generated_at(), **card}


@router.post("/smtp/test", dependencies=_POST_GUARDS)
async def smtp_test(request: Request) -> dict[str, Any]:
    """Connect, EHLO, STARTTLS and NOOP against the configured SMTP server.

    Never logs in and never sends mail. Host-wide limit: once a minute.
    """
    service = _service(request)
    runner = service.probes
    if runner is None:
        return _err(503, "probes_disabled")
    spec = runner.specs.get("smtp")
    if spec is None:
        return {
            "generated_at": _generated_at(),
            "ok": False,
            "error_class": "not_configured",
            "latency_ms": None,
            "detail": {"configured": False},
        }
    if not await runner.claim("smtp_test", SMTP_TEST_INTERVAL_S):
        return _rate_limited(SMTP_TEST_INTERVAL_S)
    record = await runner.probe_now("smtp")
    service.cache.forget(("probe_records",))
    if record is None:
        return _err(503, "probe_failed")
    return {
        "generated_at": _generated_at(),
        "ok": record["status"] in ("ok", "warn"),
        "status": record["status"],
        "error_class": record.get("error_class"),
        "latency_ms": record.get("latency_ms"),
        "detail": record.get("detail") or {},
        "checked_at": record.get("checked_at"),
    }
