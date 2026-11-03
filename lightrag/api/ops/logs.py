"""Read side of the Operations console log viewer and the correlate box.

- :func:`parse_filter` turns query parameters (or an export's ``filters``)
  into a :class:`~lightrag.api.ops.store.LogFilter`: at most 7 days per
  query and at most 24 h with a text search (``q``, an ``ILIKE`` until
  ``pg_trgm`` is available), every ID validated, the level a minimum.
- :class:`LogsReader` pages rows newest first on the ``(ts, id)`` keyset
  (``before`` = ``"<ts>,<id>"``); follow mode (``after_ts``) returns the
  newest rows at or after a time, so a client polling with its newest
  timestamp minus 10 s and de-duplicating by ``id`` skips nothing. Groups by
  call-site fingerprint, one entry with its traceback, the drawer counts and
  the export rows come from here too.
- :func:`correlate` resolves a pasted request ID, run ID, track ID, document
  ID (or ``workspace/doc``), log ID, workspace or owner email into at most
  ten matches with drill-down links, on indexes only.

Stored text was redacted when it was written; it is run through the same
redaction again on read (idempotent), so rows from an older release can
never show more than today's rules allow. Nothing here reveals text.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode
from uuid import UUID

from lightrag.api.ops.redact import mask_path, redact_pipeline, redact_traceback
from lightrag.api.ops.service import (
    OpsBadRequest,
    OpsNotFound,
    OpsService,
    _hash_link,
    validate_workspace,
)
from lightrag.api.ops.store import LogFilter, as_utc

logger = logging.getLogger("lightrag.api.ops")

DEFAULT_WINDOW_S = 3600
MAX_WINDOW_S = 7 * 86400
TEXT_SEARCH_WINDOW_S = 86400
# Clients round ``from`` down to the minute; a window that long is accepted.
WINDOW_SLACK_S = 300
PAGE_DEFAULT = 100
PAGE_MAX = 500
EXPORT_MAX = 10000
# Tracebacks are up to 8 KB a row: an export with them is capped lower, so
# one stays a few tens of MB at most.
EXPORT_MAX_WITH_EXC = 2000
EXPORT_PAGE = 1000
GROUPS_LIMIT = 100
GROUPS_MAX_HOURS = 168
DRAWER_ROWS = 50
COUNT_CAP = 10000
# Drawer counts and rows for a run or document ID cover the same 7 days a
# /logs query (and the drawer's "Open in Logs") can, so a count never names
# lines the list cannot show.
ID_LOOKBACK_DAYS = 7
Q_MAX = 200
CORRELATE_MAX = 10
CORRELATE_INPUT_MAX = 320
CACHE_TTL = {"logs": 2.0, "groups": 30.0, "counts": 10.0, "correlate": 10.0}
# Every filter a query or an export may carry (anything else is refused).
FILTER_FIELDS = (
    "level",
    "from",
    "to",
    "workspace",
    "logger",
    "module",
    "request_id",
    "run_id",
    "track_id",
    "doc_id",
    "fingerprint",
    "worker_id",
    "q",
)
EXPORT_COLUMNS = (
    "id",
    "ts",
    "level",
    "level_name",
    "logger",
    "module",
    "lineno",
    "pid",
    "worker_id",
    "workspace",
    "request_id",
    "client_request_id",
    "run_id",
    "track_id",
    "doc_id",
    "operation",
    "source",
    "fingerprint",
    "repeat_count",
    "exc_type",
    "msg",
)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_TOKEN = re.compile(r"^[^\s\x00-\x1f\x7f]{1,255}$")
_NAME = re.compile(r"^[A-Za-z0-9_.<>-]{1,128}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")
_UUIDISH = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F-]{36}$")
_TRACK_ID = re.compile(r"^[a-z]+_\d{8}_\d{6}_[0-9a-f]{8}$")
_DOC_ID = re.compile(r"^(?:doc|dup)-[A-Za-z0-9_-]{1,251}$")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}$")
_LOG_ID = re.compile(r"^\d{1,18}$")
_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
_WORKSPACE_LIKE = re.compile(r"^[A-Za-z0-9_.\-]{1,255}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# Spreadsheet formula triggers (CSV injection).
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


# -- parsing ----------------------------------------------------------------------


def parse_level(value: Any) -> int:
    if value is None or value == "":
        return logging.WARNING
    text = str(value).strip().upper()
    if text.isdigit():
        number = int(text)
        if 0 < number <= 50:
            return number
        raise OpsBadRequest("invalid_level")
    if text in _LEVELS:
        return _LEVELS[text]
    raise OpsBadRequest("invalid_level")


def parse_time(value: Any, code: str = "invalid_time") -> Optional[datetime]:
    if value is None or value == "":
        return None
    try:
        moment = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as e:
        raise OpsBadRequest(code) from e
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def parse_hex_id(value: Any, code: str) -> Optional[str]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not _UUIDISH.match(text):
        raise OpsBadRequest(code)
    try:
        return UUID(text).hex
    except ValueError as e:
        raise OpsBadRequest(code) from e


def _token(value: Any, pattern: re.Pattern[str], code: str) -> Optional[str]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not pattern.match(text):
        raise OpsBadRequest(code)
    return text


def _open_end(now: datetime) -> datetime:
    """The end of a window with no ``to``: the next whole minute plus one, so
    the same query keeps the same cache key for a minute and still includes
    rows written a moment ago by another worker."""
    floor = now.replace(second=0, microsecond=0)
    return floor + timedelta(minutes=2)


def parse_filter(params: dict[str, Any], now: Optional[datetime] = None) -> LogFilter:
    """A validated log filter. Raises :class:`OpsBadRequest` with a code."""
    unknown = set(params) - set(FILTER_FIELDS)
    if unknown:
        raise OpsBadRequest("invalid_filter")
    now = now or datetime.now(timezone.utc)
    end_given = parse_time(params.get("to"))
    end = end_given or _open_end(now)
    start = parse_time(params.get("from")) or (
        (end_given or now) - timedelta(seconds=DEFAULT_WINDOW_S)
    )
    if end <= start:
        raise OpsBadRequest("invalid_window")
    # The window the caller asked for, measured to now when it is open.
    span = ((end_given or now) - start).total_seconds()
    if span > MAX_WINDOW_S + WINDOW_SLACK_S:
        raise OpsBadRequest("window_too_long")
    q = params.get("q")
    q = str(q).strip() if q is not None else ""
    if len(q) > Q_MAX:
        raise OpsBadRequest("invalid_q")
    if q and span > TEXT_SEARCH_WINDOW_S + WINDOW_SLACK_S:
        raise OpsBadRequest("search_window_too_long")
    workspace = params.get("workspace")
    if workspace not in (None, ""):
        workspace = validate_workspace(str(workspace).strip())
    else:
        workspace = None
    return LogFilter(
        start=start,
        end=end,
        level=parse_level(params.get("level")),
        workspace=workspace,
        logger=_token(params.get("logger"), _NAME, "invalid_logger"),
        module=_token(params.get("module"), _NAME, "invalid_module"),
        request_id=_token(params.get("request_id"), _REQUEST_ID, "invalid_request_id"),
        run_id=parse_hex_id(params.get("run_id"), "invalid_run_id"),
        track_id=_token(params.get("track_id"), _TOKEN, "invalid_track_id"),
        doc_id=_token(params.get("doc_id"), _TOKEN, "invalid_doc_id"),
        fingerprint=_token(
            params.get("fingerprint"), _FINGERPRINT, "invalid_fingerprint"
        ),
        worker_id=parse_hex_id(params.get("worker_id"), "invalid_worker_id"),
        q=q or None,
    )


def filter_params(f: LogFilter) -> dict[str, Any]:
    """The filter as plain values (audit digests, cache keys)."""
    out: dict[str, Any] = {
        "level": f.level,
        "from": f.start.isoformat(),
        "to": f.end.isoformat(),
    }
    for name in (
        "workspace",
        "logger",
        "module",
        "request_id",
        "run_id",
        "track_id",
        "doc_id",
        "fingerprint",
        "worker_id",
        "q",
    ):
        value = getattr(f, name)
        if value is not None:
            out[name] = value
    return out


def search_fields(f: LogFilter) -> dict[str, Any]:
    """What a search audit hashes: the filters, without the moving window."""
    out = filter_params(f)
    out.pop("from", None)
    out.pop("to", None)
    return out


def parse_before(value: Any) -> Optional[tuple[datetime, int]]:
    """``"<iso ts>,<id>"`` -> ``(ts, id)``."""
    if value is None or value == "":
        return None
    ts, sep, raw_id = str(value).strip().rpartition(",")
    if not sep or not raw_id.isdigit() or len(raw_id) > 19:
        raise OpsBadRequest("invalid_cursor")
    moment = parse_time(ts, "invalid_cursor")
    if moment is None:
        raise OpsBadRequest("invalid_cursor")
    return moment, int(raw_id)


# -- views ----------------------------------------------------------------------------


def _iso(value: Any) -> Optional[str]:
    moment = as_utc(value)
    return moment.isoformat() if moment else None


def entry_view(row: dict[str, Any], *, with_exc: bool = False) -> dict[str, Any]:
    level = int(row.get("level") or 0)
    out = {
        "id": int(row["id"]),
        "ts": _iso(row["ts"]),
        "level": level,
        "level_name": logging.getLevelName(level),
        "logger": row.get("logger") or "",
        "module": row.get("module"),
        "lineno": row.get("lineno"),
        "pid": row.get("pid"),
        "worker_id": row.get("worker_id"),
        "workspace": row.get("workspace"),
        "request_id": row.get("request_id"),
        "client_request_id": row.get("client_request_id"),
        "run_id": row.get("run_id"),
        "track_id": row.get("track_id"),
        "doc_id": row.get("doc_id"),
        "operation": row.get("operation"),
        "source": row.get("source"),
        "msg": redact_pipeline(row.get("msg") or ""),
        "exc_type": row.get("exc_type"),
        "has_exc": bool(row.get("has_exc") or row.get("exc")),
        "fingerprint": (row.get("fingerprint") or "").strip() or None,
        "repeat_count": max(1, int(row.get("repeat_count") or 1)),
    }
    if with_exc:
        out["exc"] = redact_traceback(row.get("exc")) or None
    return out


def cursor_of(item: dict[str, Any]) -> dict[str, Any]:
    return {"ts": item["ts"], "id": item["id"]}


def mask_email(email: str) -> str:
    """``a.bin@unimas.my`` -> ``a.•••@unimas.my``."""
    local, _, domain = email.partition("@")
    return f"{local[:2]}•••@{domain}" if domain else "•••"


def csv_cell(value: Any) -> str:
    """A CSV cell a spreadsheet never evaluates as a formula."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if text.startswith(_FORMULA_START):
        return "'" + text
    return text


def export_body(rows: list[dict[str, Any]], fmt: str, include_exc: bool) -> bytes:
    """NDJSON (one object per line) or CSV with formula-safe cells."""
    columns = list(EXPORT_COLUMNS) + (["exc"] if include_exc else [])
    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([csv_cell(row.get(c)) for c in columns])
        return buffer.getvalue().encode("utf-8")
    lines = [
        json.dumps({c: row.get(c) for c in columns}, ensure_ascii=False) for row in rows
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


# -- reading ---------------------------------------------------------------------------


class LogsReader:
    """Log queries over one worker's :class:`OpsService` (store and cache)."""

    def __init__(self, service: OpsService):
        self.service = service
        self.store = service.store

    async def _cached(self, key: tuple, kind: str, compute) -> Any:
        return await self.service.cache.get(
            key, CACHE_TTL[kind], lambda: self.service._q(compute)
        )

    async def page(
        self,
        f: LogFilter,
        *,
        before: Optional[tuple[datetime, int]] = None,
        after: Optional[datetime] = None,
        limit: int = PAGE_DEFAULT,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), PAGE_MAX))
        rows = await self._cached(
            ("logs", f, before, after, limit),
            "logs",
            lambda: self.store.logs_page(
                f, before=before, after=after, limit=limit + 1
            ),
        )
        more = len(rows) > limit
        items = [entry_view(r) for r in rows[:limit]]
        return {
            "items": items,
            "next_before": cursor_of(items[-1]) if more and items else None,
            "newest_ts": items[0]["ts"] if items else None,
            "limit": limit,
            "window": {"from": f.start.isoformat(), "to": f.end.isoformat()},
        }

    async def groups(
        self, hours: int, workspace: Optional[str], level: int
    ) -> dict[str, Any]:
        hours = max(1, min(int(hours), GROUPS_MAX_HOURS))
        now = datetime.now(timezone.utc)
        end = _open_end(now)
        start = end.replace(second=0) - timedelta(hours=hours, minutes=2)
        f = LogFilter(start=start, end=end, level=level, workspace=workspace)
        rows = await self._cached(
            ("log_groups", f),
            "groups",
            lambda: self.store.logs_groups(f, GROUPS_LIMIT),
        )
        groups = []
        for row in rows:
            groups.append(
                {
                    "fingerprint": str(row["fingerprint"]).strip(),
                    "logger": row.get("logger") or "",
                    "module": row.get("module"),
                    "lineno": row.get("lineno"),
                    "level": int(row.get("level") or 0),
                    "exc_type": row.get("exc_type"),
                    "sample": redact_pipeline(row.get("sample") or ""),
                    "sample_id": row.get("sample_id"),
                    "count": int(row.get("count") or 0),
                    "rows": int(row.get("rows") or 0),
                    "first_seen": _iso(row.get("first_seen")),
                    "last_seen": _iso(row.get("last_seen")),
                    "workspaces": int(row.get("workspaces") or 0),
                }
            )
        return {"hours": hours, "groups": groups, "capped": len(rows) >= GROUPS_LIMIT}

    async def entry(self, log_id: int) -> dict[str, Any]:
        row = await self.service._q(lambda: self.store.log_get(int(log_id)))
        if row is None:
            raise OpsNotFound(str(log_id))
        entry = entry_view(row, with_exc=True)
        return {"entry": entry, "exc_withheld": False, "links": entry_links(entry)}

    async def export_rows(
        self, f: LogFilter, *, include_exc: bool, limit: int = EXPORT_MAX
    ) -> list[dict[str, Any]]:
        """Up to ``limit`` rows (:data:`EXPORT_MAX_WITH_EXC` with tracebacks),
        newest first, paged on the keyset.

        Each page is re-redacted in a worker thread, so a large export never
        holds the event loop (heartbeats, probes and requests go on).
        """
        cap = EXPORT_MAX_WITH_EXC if include_exc else EXPORT_MAX
        limit = max(1, min(int(limit), cap))
        out: list[dict[str, Any]] = []
        before: Optional[tuple[datetime, int]] = None
        while len(out) < limit:
            size = min(EXPORT_PAGE, limit - len(out))
            rows = await self.service._q(
                lambda: self.store.logs_page(
                    f, before=before, after=None, limit=size, with_exc=include_exc
                )
            )
            out.extend(
                await asyncio.to_thread(
                    lambda: [entry_view(r, with_exc=include_exc) for r in rows]
                )
            )
            if len(rows) < size:
                break
            last = rows[-1]
            before = (as_utc(last["ts"]), int(last["id"]))
        return out

    async def count(self, f: LogFilter) -> int:
        return await self._cached(
            ("log_count", f),
            "counts",
            lambda: self.store.logs_count(f, COUNT_CAP),
        )


def entry_links(entry: dict[str, Any]) -> dict[str, Optional[str]]:
    """Drill-down links of one entry (the drawer's pivot buttons)."""
    ws = entry.get("workspace")
    doc = entry.get("doc_id")
    return {
        "workspace": _hash_link(otab="processing", ows=ws) if ws else None,
        "run": _hash_link(otab="processing", orun=entry["run_id"])
        if entry.get("run_id")
        else None,
        "doc": _hash_link(otab="processing", odoc=f"{ws}/{doc}")
        if ws and doc
        else None,
        "request": _logs_link(request_id=entry["request_id"])
        if entry.get("request_id")
        else None,
        "fingerprint": _logs_link(fingerprint=entry["fingerprint"])
        if entry.get("fingerprint")
        else None,
    }


# The WebUI's ``olq`` keys (lightrag_webui/src/features/Operations/logs.ts,
# URL_KEYS): server-built links must decode to the same filter there.
OLQ_KEYS = {
    "level": "lv",
    "window": "w",
    "workspace": "ws",
    "module": "mod",
    "q": "q",
    "request_id": "req",
    "run_id": "run",
    "track_id": "trk",
    "doc_id": "doc",
    "fingerprint": "fp",
    "worker_id": "wk",
}
# An ID pivot covers the longest window a query may (FR-20).
PIVOT_WINDOW = "7d"


def _logs_link(**filters: str) -> str:
    """The Logs tab filtered on ``filters`` (filter names, as in /logs)."""
    olq = {OLQ_KEYS[name]: value for name, value in filters.items() if value}
    olq.setdefault(OLQ_KEYS["window"], PIVOT_WINDOW)
    return _hash_link(otab="logs", olq=urlencode(olq))


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def lookback_filter(**fields: Any) -> LogFilter:
    """A filter over :data:`ID_LOOKBACK_DAYS`, for ID lookups (drawers)."""
    now = datetime.now(timezone.utc)
    return LogFilter(
        start=now - timedelta(days=ID_LOOKBACK_DAYS),
        end=_open_end(now),
        **fields,
    )


async def drawer_logs(
    service: OpsService, *, rows: int = DRAWER_ROWS, **fields: Any
) -> dict[str, Any]:
    """``{available, count, first[]}`` for a drawer's log section; never
    fails the drawer (``available: false`` instead)."""
    if not service.available:
        return {"available": False, "count": None, "first": []}
    reader = LogsReader(service)
    f = lookback_filter(level=logging.WARNING, **fields)
    try:
        page = await reader.page(f, limit=rows)
        count = await reader.count(f)
    except Exception as e:
        logger.warning("Ops drawer log lookup failed (%s)", type(e).__name__)
        return {"available": False, "count": None, "first": []}
    return {
        "available": True,
        "count": count,
        "count_capped": count > COUNT_CAP,
        "first": page["items"],
    }


async def drawer_count(
    service: OpsService, *, hours: Optional[float] = None, **fields: Any
) -> Optional[int]:
    """A capped count for a drawer; None when it cannot be read."""
    if not service.available:
        return None
    if hours is not None:
        now = datetime.now(timezone.utc)
        f = LogFilter(start=now - timedelta(hours=hours), end=_open_end(now), **fields)
    else:
        f = lookback_filter(**fields)
    try:
        return min(await LogsReader(service).count(f), COUNT_CAP)
    except Exception as e:
        logger.warning("Ops drawer log count failed (%s)", type(e).__name__)
        return None


# -- correlate ---------------------------------------------------------------------------


def _match(
    type_: str,
    id_: str,
    *,
    workspace: Optional[str] = None,
    ts: Any = None,
    label: Optional[str] = None,
    count: Optional[int] = None,
    link: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "type": type_,
        "id": id_,
        "workspace": workspace,
        "ts": _iso(ts),
        "label": label,
        "count": count,
        "link": link,
    }


def classify_input(value: str) -> list[str]:
    """The ID kinds a pasted value may be, most specific first."""
    kinds: list[str] = []
    if _EMAIL.match(value):
        return ["owner"]
    if "/" in value:
        return ["ws_doc"]
    if _TRACK_ID.match(value):
        return ["track"]
    if _DOC_ID.match(value):
        return ["doc"]
    if _LOG_ID.match(value):
        kinds.append("log")
    if _UUIDISH.match(value) and _is_uuid(value):
        kinds += ["run", "request"]
    elif _REQUEST_ID.match(value) and len(value) >= 8:
        kinds.append("request")
    if _WORKSPACE_LIKE.match(value):
        kinds.append("workspace")
    return kinds


async def correlate(service: OpsService, raw: str) -> dict[str, Any]:
    """Resolve one pasted ID or owner email (FR-50); at most ten matches."""
    value = (raw or "").strip()
    if not value:
        return {"query": None, "kinds": [], "matches": []}
    if len(value) > CORRELATE_INPUT_MAX or _CONTROL.search(value):
        raise OpsBadRequest("invalid_id")
    kinds = classify_input(value)
    query = mask_email(value) if kinds == ["owner"] else value
    if not kinds:
        return {"query": query, "kinds": [], "matches": []}

    async def compute() -> list[dict[str, Any]]:
        return await _resolve(service, value, kinds)

    matches = await service.cache.get(
        ("correlate", value), CACHE_TTL["correlate"], lambda: service._q(compute)
    )
    return {"query": query, "kinds": kinds, "matches": matches[:CORRELATE_MAX]}


async def _resolve(
    service: OpsService, value: str, kinds: list[str]
) -> list[dict[str, Any]]:
    store = service.store
    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, Optional[str]]] = set()

    def add(match: dict[str, Any]) -> None:
        key = (match["type"], match["id"], match["workspace"])
        if key not in seen and len(matches) < CORRELATE_MAX:
            seen.add(key)
            matches.append(match)

    for kind in kinds:
        if len(matches) >= CORRELATE_MAX:
            break
        if kind == "owner":
            await _owner_matches(service, value, add)
        elif kind == "ws_doc":
            workspace, _, doc_id = value.partition("/")
            try:
                validate_workspace(workspace)
            except OpsBadRequest:
                continue
            if not doc_id or len(doc_id) > 255:
                continue
            row = None
            if service.docs_available:
                row = await store.document(workspace, doc_id)
            if row is not None:
                add(_doc_match(workspace, doc_id, row.get("file_path"), row))
            else:
                for item in await store.correlate_docs(doc_id, CORRELATE_MAX):
                    if item["workspace"] == workspace:
                        add(_doc_match(workspace, doc_id, item.get("file_path"), item))
        elif kind == "doc":
            for item in await store.correlate_docs(value, CORRELATE_MAX):
                if item["workspace"]:
                    add(
                        _doc_match(
                            item["workspace"], value, item.get("file_path"), item
                        )
                    )
        elif kind == "track":
            for item in await store.correlate_tracks(value, CORRELATE_MAX):
                add(
                    _match(
                        "track",
                        value,
                        workspace=item["workspace"],
                        ts=item.get("ts"),
                        count=int(item.get("docs") or 0)
                        or int(item.get("log_lines") or 0),
                        label=f"{int(item.get('docs') or 0)} docs",
                        link=_logs_link(track_id=value),
                    )
                )
        elif kind == "log":
            row = await store.log_get(int(value))
            if row is not None:
                entry = entry_view(row)
                add(
                    _match(
                        "log",
                        value,
                        workspace=entry["workspace"],
                        ts=row["ts"],
                        label=f"{entry['level_name']} {entry['msg'][:80]}",
                        link=_hash_link(otab="logs", olog=value),
                    )
                )
        elif kind == "run":
            run_id = UUID(value).hex
            row = await store.run_get(run_id)
            if row is not None:
                add(
                    _match(
                        "run",
                        run_id,
                        workspace=row.get("workspace"),
                        ts=row.get("started_at"),
                        label=f"{row.get('kind')} · {row.get('status')}",
                        link=_hash_link(otab="processing", orun=run_id),
                    )
                )
        elif kind == "request":
            for item in await store.correlate_requests(value, CORRELATE_MAX):
                add(
                    _match(
                        "request",
                        value,
                        workspace=item.get("workspace"),
                        ts=item.get("last_ts"),
                        count=int(item.get("n") or 0),
                        link=_logs_link(request_id=value),
                    )
                )
        elif kind == "workspace":
            await _workspace_match(service, value, add)
    return matches


def _doc_match(
    workspace: str, doc_id: str, file_path: Optional[str], item: dict[str, Any]
) -> dict[str, Any]:
    status = item.get("status")
    label = mask_path(file_path) if file_path else None
    if status:
        label = f"{status} · {label}" if label else status
    return _match(
        "doc",
        doc_id,
        workspace=workspace,
        ts=item.get("ts") or item.get("updated_at"),
        label=label,
        count=int(item.get("failures") or 0) or None,
        link=_hash_link(otab="processing", odoc=f"{workspace}/{doc_id}"),
    )


async def _owner_matches(service: OpsService, email: str, add) -> None:
    workspaces: dict[str, Optional[str]] = {}
    registry = getattr(service.app_state, "workspace_registry", None)
    if registry is not None:
        for identity in dict.fromkeys((email, email.lower())):
            try:
                owner = await registry.get_by_identity(identity)
            except Exception as e:
                logger.warning("Ops owner lookup failed (%s)", type(e).__name__)
                owner = None
            if owner is not None:
                workspaces[owner.workspace_id] = owner.email or owner.identity
    if service.available:
        found = await service.store.owners_by_email(email, CORRELATE_MAX)
        for ws, owner in found.items():
            workspaces.setdefault(ws, owner)
    if not workspaces:
        # Workspace IDs are derived from the owner's email (auth.py).
        from lightrag.api.auth import sanitize_workspace_id

        derived = sanitize_workspace_id(email)
        presence = await service.store.workspace_presence(derived)
        if any(presence.values()):
            workspaces[derived] = email
    for ws in sorted(workspaces):
        add(
            _match(
                "owner",
                ws,
                workspace=ws,
                label=mask_email(email),
                link=_hash_link(otab="processing", ows=ws),
            )
        )


async def _workspace_match(service: OpsService, value: str, add) -> None:
    try:
        validate_workspace(value)
    except OpsBadRequest:
        return
    owner = await service._owner(value)
    presence = await service.store.workspace_presence(value)
    if owner is None and not any(presence.values()):
        return
    add(
        _match(
            "workspace",
            value,
            workspace=value,
            label=mask_email(owner) if owner and "@" in owner else None,
            link=_hash_link(otab="processing", ows=value),
        )
    )
