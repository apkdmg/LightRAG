"""Request correlation: request IDs, log context fields and the 500 boundary.

- :class:`RequestIdMiddleware` (outermost) gives every HTTP request an ID,
  keeps it in a ContextVar, echoes it in ``X-Request-ID`` and turns an
  exception that escapes every other layer into a JSON 500 that still carries
  the ID. An inbound ``X-Request-ID`` is trusted only from a peer listed in
  ``OPS_TRUSTED_PROXIES``; any other caller's value is kept as
  ``client_request_id`` and never used as the ID.
- :class:`ServerErrorBoundary` (innermost) turns an unhandled route exception
  into the same JSON 500 *inside* the stack, so CORS and security headers are
  still applied. Starlette's own ``ServerErrorMiddleware`` sits outside every
  user middleware and would otherwise answer with a bare 500.
- :class:`ContextFilter` copies the request ID, worker ID, run ID and the
  usage scope (workspace, track, document, operation, source) onto each log
  record. It is attached to handlers, so records propagated from child
  loggers are covered too.

Pure ASGI and stdlib only: no Starlette import, so core code can use the
ContextVars without pulling in the web stack.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Iterable, Iterator, Optional

from lightrag.usage import current_usage_scope

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_BYTES = b"x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_LEADING_WORKSPACE = re.compile(r"^\[([A-Za-z0-9_-]{1,128})\]")
# generate_track_id(): "<prefix>_<YYYYmmdd>_<HHMMSS>_<8 hex>"
_TRACK_ID_SHAPE = re.compile(r"^[a-z]+_\d{8}_\d{6}_[0-9a-f]{8}$")
# Literal "[tag]" prefixes that name a function or subsystem, not a workspace
# (compared lower-cased). tests/test_ops_context.py scans the code base and
# fails when a new literal tag is not listed here.
FUNCTION_LOG_TAGS = frozenset(
    {
        "_build_context_str",
        "_build_query_context",
        "analyze_multimodal",
        "apipeline_enqueue",
        "aquery_data",
        "aquery_llm",
        "bedrock",
        "convert_to_user_format",
        "docling",
        "docling_ir_builder",
        "external_parser",
        "extract_keywords",
        "kg_query",
        "mineru",
        "mineru_ir_builder",
        "mineru_raw",
        "multimodal_context",
        "naive_query",
        "paragraph_semantic_chunking",
        "parse",
        "parse_docling",
        "parse_mineru",
        "parse_native",
        "parser_routing",
        "purge",
        "recursive_character",
        "semantic_vector",
        "sidecar",
        "skip_kg",
        "vector_similarity",
    }
)

# Fields ContextFilter guarantees on every record ("-" when unknown).
CONTEXT_FIELDS = (
    "request_id",
    "client_request_id",
    "worker_id",
    "run_id",
    "workspace",
    "track_id",
    "doc_id",
    "operation",
    "source",
)

_request_id: ContextVar[Optional[str]] = ContextVar("lightrag_request_id", default=None)
_client_request_id: ContextVar[Optional[str]] = ContextVar(
    "lightrag_client_request_id", default=None
)
_run_id: ContextVar[Optional[str]] = ContextVar("lightrag_run_id", default=None)
# Workspace a request resolved to. Kept apart from the usage scope so that
# labelling log lines can never change how LLM usage is attributed.
_log_workspace: ContextVar[Optional[str]] = ContextVar(
    "lightrag_log_workspace", default=None
)
# Process-wide: set once per worker by the ops runtime, never at import time
# (Gunicorn preloads the app before forking).
_worker_id: Optional[str] = None

logger = logging.getLogger("lightrag.api.ops")

# Called once per HTTP request after the response finished (or failed) with
# (scope, status_code, duration_seconds). The performance metrics attach here.
RequestObserver = Callable[[dict, int, float], None]


def current_request_id() -> Optional[str]:
    return _request_id.get()


def current_client_request_id() -> Optional[str]:
    return _client_request_id.get()


def current_run_id() -> Optional[str]:
    return _run_id.get()


def set_run_id(run_id: Optional[str]) -> Any:
    """Bind a run ID to the current context; returns the reset token."""
    return _run_id.set(run_id)


def reset_run_id(token: Any) -> None:
    _run_id.reset(token)


def bind_log_workspace(workspace: Optional[str]) -> None:
    """Label the rest of this request's log lines with ``workspace``.

    Each request runs in its own context, so the value cannot leak into other
    requests; background tasks started by the request inherit it, which is
    the intent.
    """
    try:
        _log_workspace.set(workspace or None)
    except Exception:  # pragma: no cover - ContextVar.set does not fail
        pass


def current_log_workspace() -> Optional[str]:
    return _log_workspace.get()


@contextmanager
def background_log_context(workspace: Optional[str] = None) -> Iterator[None]:
    """Detach long-running background work from the request that started it.

    A task created during a request (``asyncio.create_task``, Starlette
    background tasks) inherits the request's ContextVars. Pipeline runs and
    queue drainers outlive that request and do work for other requests too,
    so inside this block their log lines carry no request ID, only the
    workspace they work for. The previous values come back on exit, so a
    caller that awaited the work inline keeps its own request ID afterwards.
    """
    tokens = []
    try:
        tokens = [
            (_request_id, _request_id.set(None)),
            (_client_request_id, _client_request_id.set(None)),
            (_log_workspace, _log_workspace.set(workspace or None)),
        ]
    except Exception:  # pragma: no cover - ContextVar.set does not fail
        pass
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except ValueError:
                # Exited in another context (should not happen); keep going.
                pass


def set_worker_id(worker_id: Optional[str]) -> None:
    global _worker_id
    _worker_id = worker_id


def current_worker_id() -> Optional[str]:
    return _worker_id


def new_request_id() -> str:
    return uuid.uuid4().hex


def parse_trusted_proxies(value: Optional[str]) -> tuple[Any, ...]:
    """Parse ``OPS_TRUSTED_PROXIES``: comma-separated IPs or CIDR networks.

    Invalid entries are skipped with a warning; an empty value trusts nobody.
    """
    networks: list[Any] = []
    for raw in (value or "").split(","):
        entry = raw.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning(
                "OPS_TRUSTED_PROXIES: ignoring an entry that is not an IP "
                "address or CIDR network (%d characters)",
                len(entry),
            )
    return tuple(networks)


def _peer_is_trusted(scope: dict, networks: Iterable[Any]) -> bool:
    client = scope.get("client")
    if not client or not networks:
        return False
    try:
        address = ipaddress.ip_address(client[0])
    except (ValueError, TypeError, IndexError):
        return False
    return any(address in network for network in networks)


def _header(scope: dict, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            try:
                return value.decode("latin-1")
            except Exception:
                return None
    return None


def error_body(detail: str, request_id: Optional[str]) -> bytes:
    return json.dumps({"detail": detail, "request_id": request_id}).encode("utf-8")


async def send_internal_error(
    send: Callable[[dict], Awaitable[None]],
    request_id: Optional[str],
    extra_headers: Iterable[tuple[bytes, bytes]] = (),
) -> None:
    """Send the generic JSON 500: ``{"detail": "internal_error", "request_id"}``."""
    body = error_body("internal_error", request_id)
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("latin-1")),
        (b"cache-control", b"no-store"),
    ]
    if request_id:
        headers.append((_REQUEST_ID_HEADER_BYTES, request_id.encode("latin-1")))
    headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": 500, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def _route_label(scope: dict) -> str:
    """Route template when the router matched one, so no IDs reach the log."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return str(path)
    return "<unmatched>"


def _log_unhandled(scope: dict, request_id: Optional[str]) -> None:
    try:
        logger.error(
            "Unhandled error on %s %s (request %s)",
            scope.get("method", "?"),
            _route_label(scope),
            request_id or "-",
            exc_info=True,
        )
    except Exception:  # pragma: no cover - logging must not mask the 500
        pass


class RequestIdMiddleware:
    """Outermost middleware: request ID, response header and last-resort 500."""

    def __init__(
        self,
        app: Any,
        trusted_proxies: Optional[str] = None,
        observers: Iterable[RequestObserver] = (),
        error_headers: Optional[dict[str, str]] = None,
    ):
        self.app = app
        if trusted_proxies is None:
            trusted_proxies = os.getenv("OPS_TRUSTED_PROXIES", "")
        self.trusted_networks = parse_trusted_proxies(trusted_proxies)
        self.observers = list(observers)
        # Headers the inner layers would have added (security headers); the
        # last-resort 500 is sent from outside them.
        self._error_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (error_headers or {}).items()
        ]

    def add_observer(self, observer: RequestObserver) -> None:
        self.observers.append(observer)

    def _resolve_ids(self, scope: dict) -> tuple[str, Optional[str]]:
        inbound = _header(scope, _REQUEST_ID_HEADER_BYTES)
        valid = bool(inbound) and bool(_VALID_REQUEST_ID.match(inbound or ""))
        if valid and _peer_is_trusted(scope, self.trusted_networks):
            return inbound, None
        # Never adopt an untrusted ID; keep a well-formed one for reference.
        return new_request_id(), inbound if valid else None

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id, client_request_id = self._resolve_ids(scope)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        if client_request_id:
            state["client_request_id"] = client_request_id
        rid_token = _request_id.set(request_id)
        cid_token = _client_request_id.set(client_request_id)
        header_value = request_id.encode("latin-1")
        started = time.perf_counter()
        status_holder = {"status": 500, "started": False}

        async def send_with_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder["started"] = True
                status_holder["status"] = int(message.get("status", 500))
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != _REQUEST_ID_HEADER_BYTES
                ]
                headers.append((_REQUEST_ID_HEADER_BYTES, header_value))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            if status_holder["started"]:
                # Too late for a clean response; let the server abort it.
                raise
            _log_unhandled(scope, request_id)
            status_holder["status"] = 500
            await send_internal_error(send, request_id, self._error_headers)
        finally:
            duration = time.perf_counter() - started
            for observer in self.observers:
                try:
                    observer(scope, status_holder["status"], duration)
                except Exception:
                    pass
            _client_request_id.reset(cid_token)
            _request_id.reset(rid_token)


class ServerErrorBoundary:
    """Innermost middleware: JSON 500 for unhandled route exceptions.

    Sending the 500 from here (instead of Starlette's outer
    ``ServerErrorMiddleware``) keeps CORS, security headers and the request
    ID on the response. The traceback is logged once, with the request ID.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        started = {"value": False}

        async def tracking_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                started["value"] = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            if started["value"]:
                raise
            request_id = current_request_id()
            _log_unhandled(scope, request_id)
            await send_internal_error(send, request_id)


class ContextFilter(logging.Filter):
    """Handler filter that puts correlation fields on every record.

    Never drops a record. Fields already set (for example by the structured
    sink) are kept. ``workspace`` comes from the usage scope, then from the
    workspace the request resolved to, then from a leading ``[workspace]``
    prefix in the message.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_lightrag_context", False):
            return True
        try:
            self._apply(record)
        except Exception:
            for field in CONTEXT_FIELDS:
                if not hasattr(record, field):
                    setattr(record, field, "-")
        record._lightrag_context = True
        return True

    @staticmethod
    def _apply(record: logging.LogRecord) -> None:
        scope = current_usage_scope()
        values = {
            "request_id": _request_id.get(),
            "client_request_id": _client_request_id.get(),
            "worker_id": _worker_id,
            "run_id": _run_id.get(),
            "workspace": scope.workspace or _log_workspace.get(),
            "track_id": scope.track_id,
            "doc_id": scope.doc_id,
            "operation": scope.operation,
            "source": scope.source,
        }
        if not values["workspace"] and isinstance(record.msg, str):
            match = _LEADING_WORKSPACE.match(record.msg)
            if match:
                label = match.group(1)
                # Some call sites lead with a track ID or a function name
                # rather than a workspace.
                if _TRACK_ID_SHAPE.match(label):
                    values["track_id"] = values["track_id"] or label
                elif label.lower() not in FUNCTION_LOG_TAGS:
                    values["workspace"] = label
        for field, value in values.items():
            if getattr(record, field, None) in (None, ""):
                setattr(record, field, value or "-")


class ContextFormatter(logging.Formatter):
    """Formatter that tolerates records which never met a ContextFilter."""

    def format(self, record: logging.LogRecord) -> str:
        for field in CONTEXT_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, "-")
        return super().format(record)


# Log formats shared by the Uvicorn and Gunicorn logging setups.
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(process)d] "
    "[%(request_id)s] [%(workspace)s] - %(message)s"
)
CONSOLE_LOG_FORMAT = (
    "%(levelname)s: [%(process)d] [%(request_id)s] [%(workspace)s] %(message)s"
)
