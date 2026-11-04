"""The one audit writer of the Operations console (``LIGHTRAG_OPS_AUDIT``).

Every export, reveal, action and log search goes through :func:`write_audit`.
Rows name the actor, how they signed in, who impersonated them (refused for
mutations, but recorded), the action, an intent or result phase, and a hash
of the client address; never filter values or other free text a caller
typed, which are stored as a hash plus the field names (FR-75).

:class:`SearchAudit` records log searches at most once per actor and filter
hash every five minutes, so a following Logs tab (one poll every 5 s) does
not write an audit row per poll.

The existing admin mutations (impersonation, workspace create and delete,
owner bindings, server-default provider changes, admin cross-workspace
``/documents`` mutations) are audited through :class:`AdminAudit` and
:func:`note_document_mutation` with the same writer, best effort (see below).
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from fastapi import Request, Response
from fastapi.routing import APIRoute

from lightrag.api.ops.context import current_request_id

logger = logging.getLogger("lightrag.api.ops")

SEARCH_DEDUP_S = 300.0
SEARCH_DEDUP_MAX = 4096
_AUTH_KIND_MAX = 16


def actor_of(user: Any) -> dict[str, Optional[str]]:
    """``actor``, ``auth_kind`` and ``impersonated_by`` of a signed-in user."""
    metadata = getattr(user, "metadata", None) or {}
    auth_kind = metadata.get("auth_mode") or "token"
    impersonated = metadata.get("impersonated_by")
    return {
        "actor": str(getattr(user, "username", "") or "-")[:255],
        "auth_kind": str(auth_kind)[:_AUTH_KIND_MAX],
        "impersonated_by": str(impersonated)[:255] if impersonated else None,
    }


def client_ip_hash(request: Any) -> Optional[str]:
    client = getattr(request, "client", None) if request is not None else None
    host = getattr(client, "host", None)
    if not host:
        return None
    return hashlib.sha256(str(host).encode("utf-8")).hexdigest()[:16]


def filters_digest(filters: dict[str, Any]) -> dict[str, Any]:
    """``{filter_hash, fields}`` for a set of filters: values never stored."""
    present = {
        str(k): v for k, v in (filters or {}).items() if v is not None and v != ""
    }
    canonical = json.dumps(present, sort_keys=True, default=str)
    return {
        "filter_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "fields": sorted(present),
    }


async def write_audit(
    store: Any,
    *,
    user: Any,
    action: str,
    phase: str = "result",
    request: Any = None,
    workspace: Optional[str] = None,
    target: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    reason: Optional[str] = None,
    outcome: Optional[str] = None,
) -> int:
    """Write one audit row and return its id. Raises when it cannot be written:
    the caller decides whether the action may go on without it."""
    row = {
        **actor_of(user),
        "action": action,
        "phase": phase,
        "workspace": workspace,
        "target": target,
        "params": params or {},
        "reason": reason,
        "outcome": outcome,
        "request_id": current_request_id(),
        "client_ip_hash": client_ip_hash(request),
    }
    return await store.audit_write(row)


class SearchAudit:
    """Audits searches once per actor and filter hash per five minutes."""

    def __init__(
        self,
        window_s: float = SEARCH_DEDUP_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.window_s = window_s
        self.clock = clock
        self._seen: OrderedDict[tuple[str, str, str], float] = OrderedDict()
        self._lock = threading.Lock()
        self.failures = 0

    def _due(self, key: tuple[str, str, str]) -> bool:
        now = self.clock()
        with self._lock:
            last = self._seen.get(key)
            if last is not None and now - last < self.window_s:
                return False
            self._seen[key] = now
            self._seen.move_to_end(key)
            while len(self._seen) > SEARCH_DEDUP_MAX:
                self._seen.popitem(last=False)
            return True

    def _undo(self, key: tuple[str, str, str]) -> None:
        with self._lock:
            self._seen.pop(key, None)

    async def record(
        self,
        store: Any,
        *,
        user: Any,
        action: str,
        filters: dict[str, Any],
        request: Any = None,
        outcome: Optional[str] = "ok",
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Best effort: a search is never refused because its audit failed."""
        digest = filters_digest(filters)
        key = (actor_of(user)["actor"] or "-", action, digest["filter_hash"])
        if not self._due(key):
            return None
        try:
            return await write_audit(
                store,
                user=user,
                action=action,
                request=request,
                params={**digest, **(extra or {})},
                outcome=outcome,
            )
        except Exception as e:
            # Not deduplicated, so the next search tries again.
            self._undo(key)
            self.failures += 1
            logger.warning(
                "Ops search audit could not be written (%s)", type(e).__name__
            )
            return None


# -- existing admin mutations (FR-75) --------------------------------------------
#
# Impersonation, workspace create/delete, owner bindings, server-default
# provider changes and admin cross-workspace /documents/* mutations write an
# intent row before they act and a result row after, through the same writer.
# Unlike the console's own actions (which refuse to act without their intent
# row, §3.9), these endpoints predate the console and also run on servers
# without the ops database: an audit write that fails or is slow (bounded by
# ADMIN_AUDIT_TIMEOUT_S) is logged and counted, and never changes the
# endpoint's response. OPS_ADMIN_AUDIT=false switches these hooks off (the
# rollback switch for them; OPS_ACTIONS_ENABLED covers the console's own
# actions, which are always audited).

ADMIN_AUDIT_FLAG = "OPS_ADMIN_AUDIT"
ADMIN_AUDIT_TIMEOUT_S = 3.0
_OUTCOME_MAX = 24
_PENDING_STATE_KEY = "ops_admin_audit"
_failure_log_at = 0.0
_FAILURE_LOG_INTERVAL_S = 60.0
admin_audit_failures = 0

# Mutating /documents routes that are reads in disguise (POST for a body).
READ_ONLY_DOCUMENT_POSTS = frozenset({"/documents/paginated"})
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_SERVICE_AUTH_MODES = frozenset({"api_key", "client_credentials"})


def admin_audit_enabled() -> bool:
    """``OPS_ADMIN_AUDIT`` (default on)."""
    value = os.getenv(ADMIN_AUDIT_FLAG)
    if value is None or not value.strip():
        return True
    return value.strip().lower() in ("1", "true", "yes", "on")


def audit_store(app: Any) -> Any:
    """The ops store of this worker when it can take writes and the admin
    audit hooks are on, else None (the hooks then write nothing)."""
    if not admin_audit_enabled():
        return None
    state = getattr(app, "state", None)
    runtime = getattr(state, "ops", None)
    service = getattr(runtime, "service", None)
    if service is None or not getattr(service, "available", False):
        return None
    return getattr(service, "store", None)


def _note_failure(phase: str, action: str, error: BaseException) -> None:
    global admin_audit_failures, _failure_log_at
    admin_audit_failures += 1
    now = time.monotonic()
    if now - _failure_log_at >= _FAILURE_LOG_INTERVAL_S:
        _failure_log_at = now
        logger.warning(
            "Admin audit %s row for %s could not be written (%s); the action "
            "went ahead (%d failures so far)",
            phase,
            action,
            type(error).__name__,
            admin_audit_failures,
        )


async def _best_effort_write(store: Any, phase: str, **fields: Any) -> Optional[int]:
    if store is None:
        return None
    try:
        return await asyncio.wait_for(
            write_audit(store, phase=phase, **fields), timeout=ADMIN_AUDIT_TIMEOUT_S
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:  # includes the timeout
        _note_failure(phase, str(fields.get("action")), e)
        return None


def outcome_of(
    error: Optional[BaseException], status_code: Optional[int] = None
) -> str:
    """``ok``, ``http_<status>``, ``cancelled`` or ``error``."""
    if error is None:
        if status_code is None or status_code < 400:
            return "ok"
        return f"http_{int(status_code)}"[:_OUTCOME_MAX]
    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return f"http_{code}"[:_OUTCOME_MAX]
    return "error"


class AdminAudit:
    """``async with`` around an existing admin mutation: intent row on entry,
    result row on exit (outcome from how the block ended). Never raises
    because of the audit and never swallows the block's exception.

    ``outcome`` may be set inside the block to record a more specific
    success; ``result_params`` is merged into the result row's params.
    """

    def __init__(
        self,
        request: Any,
        user: Any,
        action: str,
        *,
        workspace: Optional[str] = None,
        target: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ):
        self.request = request
        self.user = user
        self.action = action
        self.workspace = workspace
        self.target = target
        self.params = dict(params or {})
        self.outcome: Optional[str] = None
        self.result_params: dict[str, Any] = {}
        self.intent_id: Optional[int] = None
        self._store = audit_store(getattr(request, "app", None))

    def forget_workspace(self) -> None:
        """The workspace was just deleted: the result row names it by its
        hash, like every other audit row of a deleted workspace (FR-78)."""
        from lightrag.api.ops.store import workspace_hash

        if self.workspace:
            hashed = workspace_hash(self.workspace)
            if self.target == self.workspace:
                self.target = hashed
            self.workspace = hashed

    def _fields(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "action": self.action,
            "request": self.request,
            "workspace": self.workspace,
            "target": self.target,
        }

    async def __aenter__(self) -> "AdminAudit":
        self.intent_id = await _best_effort_write(
            self._store, "intent", params=self.params, **self._fields()
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        outcome = outcome_of(exc) if exc is not None else (self.outcome or "ok")
        params = {**self.params, **self.result_params}
        if self.intent_id is not None:
            params["intent_id"] = self.intent_id
        await _best_effort_write(
            self._store, "result", params=params, outcome=outcome, **self._fields()
        )


def admin_audit(request: Any, user: Any, action: str, **kwargs: Any) -> AdminAudit:
    return AdminAudit(request, user, action, **kwargs)


def _route_template(request: Any) -> str:
    """The matched route's template (``/documents/{doc_id}``), else the
    request path without the ASGI ``root_path`` (LIGHTRAG_API_PREFIX), so a
    prefixed request is classified like an unprefixed one."""
    scope = getattr(request, "scope", None) or {}
    path = getattr(scope.get("route"), "path", None)
    if isinstance(path, str) and path:
        return path[:200]
    path = str(getattr(getattr(request, "url", None), "path", "") or "")
    root = str(scope.get("root_path") or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root) :] or "/"
    return path[:200]


def document_mutation_mode(request: Any, user: Any, workspace: str) -> Optional[str]:
    """``on_behalf`` or ``impersonated`` when this /documents request is an
    admin acting on someone else's workspace; None otherwise.

    Service accounts always name a workspace (that is how they work), so
    their writes are not cross-workspace admin actions and are not audited
    here (they would add two rows per ingested document for 400 days).
    """
    if getattr(request, "method", "").upper() not in _MUTATING_METHODS:
        return None
    path = _route_template(request)
    if (
        not (path == "/documents" or path.startswith("/documents/"))
        or path in READ_ONLY_DOCUMENT_POSTS
    ):
        return None
    metadata = getattr(user, "metadata", None) or {}
    if metadata.get("impersonated_by") or metadata.get("impersonation"):
        return "impersonated"
    if metadata.get("auth_mode") in _SERVICE_AUTH_MODES:
        return None
    from lightrag.api.dependencies import TARGET_WORKSPACE_HEADER

    if (
        request.headers.get(TARGET_WORKSPACE_HEADER)
        and getattr(user, "role", None) == "admin"
        and workspace != getattr(user, "workspace_id", None)
    ):
        return "on_behalf"
    return None


async def note_document_mutation(request: Any, user: Any, workspace: str) -> None:
    """Intent row for an admin /documents mutation on another workspace.

    Called where the request's workspace is resolved; at most once per
    request. The result row is written by :class:`AuditedDocumentsRoute`
    when the route returns. Never raises.
    """
    try:
        if not admin_audit_enabled():
            return
        mode = document_mutation_mode(request, user, workspace)
        if mode is None:
            return
        state = request.state
        if getattr(state, _PENDING_STATE_KEY, None) is not None:
            return
        params = {
            "method": request.method.upper(),
            "route": _route_template(request),
            "mode": mode,
        }
        pending = {
            "user": user,
            "workspace": workspace,
            "params": params,
            "store": audit_store(request.app),
            "intent_id": None,
        }
        setattr(state, _PENDING_STATE_KEY, pending)
        pending["intent_id"] = await _best_effort_write(
            pending["store"],
            "intent",
            user=user,
            action="documents_mutation",
            request=request,
            workspace=workspace,
            target=workspace,
            params=params,
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _note_failure("intent", "documents_mutation", e)


async def finish_document_mutation(
    request: Any, error: Optional[BaseException], status_code: Optional[int]
) -> None:
    """Result row for a pending :func:`note_document_mutation`. Never raises."""
    try:
        state = getattr(request, "state", None)
        pending = getattr(state, _PENDING_STATE_KEY, None) if state else None
        if not isinstance(pending, dict) or pending.get("done"):
            return
        pending["done"] = True
        params = dict(pending["params"])
        if pending.get("intent_id") is not None:
            params["intent_id"] = pending["intent_id"]
        await _best_effort_write(
            pending["store"],
            "result",
            user=pending["user"],
            action="documents_mutation",
            request=request,
            workspace=pending["workspace"],
            target=pending["workspace"],
            params=params,
            outcome=outcome_of(error, status_code),
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _note_failure("result", "documents_mutation", e)


class AuditedDocumentsRoute(APIRoute):
    """Route class of the /documents router: writes the result row of an
    audited admin mutation once the endpoint has answered or raised."""

    def get_route_handler(self) -> Callable[[Request], Any]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                response = await original(request)
            except BaseException as e:
                await finish_document_mutation(request, e, None)
                raise
            await finish_document_mutation(
                request, None, getattr(response, "status_code", None)
            )
            return response

        return handler


# -- the Audit section of the Activity tab ---------------------------------------

AUDIT_EXPORT_COLUMNS = (
    "id",
    "ts",
    "actor",
    "auth_kind",
    "impersonated_by",
    "action",
    "workspace",
    "target",
    "phase",
    "outcome",
    "reason",
    "request_id",
    "params",
)


def audit_view(row: dict[str, Any]) -> dict[str, Any]:
    """One audit row for the console. Reasons were redacted when written."""
    from lightrag.api.ops.store import utc_iso

    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    return {
        "id": int(row["id"]),
        "ts": utc_iso(row.get("ts")),
        "actor": row.get("actor"),
        "auth_kind": row.get("auth_kind"),
        "impersonated_by": row.get("impersonated_by"),
        "action": row.get("action"),
        "workspace": row.get("workspace"),
        "target": row.get("target"),
        "phase": row.get("phase"),
        "outcome": row.get("outcome"),
        "reason": row.get("reason"),
        "request_id": row.get("request_id"),
        "intent_id": params.get("intent_id"),
        "params": params,
    }


def audit_csv(items: list[dict[str, Any]]) -> bytes:
    """Audit rows as CSV; no cell can start a spreadsheet formula."""
    from lightrag.api.ops.logs import csv_cell

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(AUDIT_EXPORT_COLUMNS)
    for item in items:
        cells = []
        for column in AUDIT_EXPORT_COLUMNS:
            value = item.get(column)
            if column == "params":
                value = json.dumps(value or {}, sort_keys=True, default=str)
            cells.append(csv_cell(value))
        writer.writerow(cells)
    return buffer.getvalue().encode("utf-8")
