"""Read models behind the Operations console routes.

Every number here is host-wide: it comes from SQL over tables every process
writes, or from ``LIGHTRAG_OPS_KV`` values written by a leased singleton. The
only process-shared state read directly is ``pipeline_status``, and only
through :class:`~lightrag.api.ops.snapshot.NamespaceReader` for at most 50
SQL-selected workspaces (or the one workspace a drawer shows). Nothing here
instantiates a workspace or loops over ``get_namespace_data``.

Each worker keeps a bounded LRU of recent results with single-flight
computation (256 entries, 5-60 s TTLs), so database load does not grow with
the number of open dashboards.

Until run tracking exists, live jobs are derived from in-flight documents plus
shared state and labelled approximate.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlencode

from lightrag.api.insights.analytics import document_label
from lightrag.api.ops.pauses import OwnProviderCache, read_pauses
from lightrag.api.ops.queue_scan import SCAN_INTERVAL_S, SUMMARY_KEY
from lightrag.api.ops.redact import mask_path, redact, redact_pipeline
from lightrag.api.ops.snapshot import LIVE_KEYS, NamespaceReader
from lightrag.api.ops.store import (
    ACTIVE_STATUSES,
    BACKLOG_FILTERS,
    BACKLOG_SORTS,
    CANCELLED_GROUP_KEY,
    STATUS_FAILED,
    OpsStore,
    Page,
    as_utc,
    is_cancelled,
    utc_iso,
)

logger = logging.getLogger("lightrag.api.ops")

CACHE_MAX_ENTRIES = 256
ACTIVE_CANDIDATES = 50
FEED_MAX_LINES = 200
# Backlog search: as long as the longest workspace ID (_WORKSPACE_ID).
SEARCH_MAX_LENGTH = 255
DOC_CALLS_LIMIT = 50
EVENTS_LIMIT = 500
EVENTS_MAX_DAYS = 7
OVERVIEW_TIMELINE_HOURS = 6

TTL = {
    "totals": 10.0,
    "dup": 60.0,
    "cancelled": 60.0,
    "backlog": 10.0,
    "stale": 15.0,
    "active": 15.0,
    "failures": 15.0,
    "failure_docs": 10.0,
    "queue": 5.0,
    "pauses": 60.0,
    "thresholds": 60.0,
    "workspace": 5.0,
    "documents": 5.0,
    "document": 5.0,
    "events": 15.0,
    "llm": 30.0,
    "overview": 5.0,
}

DEFAULT_THRESHOLDS = {
    "stale_doc_minutes": 60,
    "stale_doc_serious_minutes": 180,
    "hung_minutes": 15,
    "orphan_seconds": 60,
    "oldest_pending_minutes": 120,
    "oldest_pending_serious_minutes": 360,
    "email_waiting_minutes": 30,
    "email_waiting_serious_minutes": 120,
    "llm_error_rate": 0.05,
    "llm_error_rate_serious": 0.20,
    "llm_min_calls": 20,
}

FAILURE_WINDOWS = {"1h": 1, "24h": 24, "7d": 168}
# Groups from the failure taxonomy (durable failure rows, later); until then
# only legacy groupings are filled.
DURABLE_FAILURE_GROUPS = ("error_kind", "stage", "model")
LEGACY_FAILURE_GROUPS = ("signature", "workspace")
EVENT_KINDS = ("deployment", "model_change", "provider_rejection", "ingestion_resumed")
AUDIT_EVENT_KIND = "ops_action"
_EVENT_DETAIL_KEYS = ("old", "new", "status", "byo", "slot", "version", "paused")
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,255}$")
_JOB_FILES = re.compile(r"^(?P<prefix>.*?)(?:\.\.\.)?\[(?P<count>\d+) files\]$", re.S)
_SEVERITY_RANK = {"warning": 1, "serious": 2, "critical": 3}

RUNBOOK = {
    "stale_docs": "Open the workspace; if its pipeline is not busy, run the pipeline "
    "again, otherwise check its live feed for the stage it is stuck in.",
    "oldest_pending": "Check whether the workspace's pipeline is busy or paused; a "
    "pending document waits for the current run to finish.",
    "provider_paused": "Fix the provider key or spend limit in Providers; ingestion "
    "resumes by itself when the key works again.",
    "email_queue_bad": "A queued email re-sync could not be read and was set aside; "
    "the email connector will send the email again on its next sync.",
    "email_queue_stalled": "The workspace's pipeline has stayed busy; check what it "
    "is processing, or whether it is paused.",
    "llm_error_rate_15m": "Open Performance > LLM to see which model fails, then "
    "Failures grouped by model.",
    "manager_unresponsive": "Shared pipeline state is not answering; restart the "
    "container if it persists.",
    "pg_probe": "Postgres is not answering ops queries; check the database container.",
}


class OpsDataUnavailable(Exception):
    """A query failed (database unreachable or timed out)."""


class OpsNotFound(Exception):
    pass


class OpsBadRequest(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class OpsCache:
    """Bounded LRU with per-key TTL and single-flight computation."""

    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._max = max_entries
        self._clock = clock
        self._entries: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()
        self._inflight: dict[tuple, asyncio.Future] = {}
        self.computations = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    async def get(
        self, key: tuple, ttl: float, compute: Callable[[], Awaitable[Any]]
    ) -> Any:
        now = self._clock()
        hit = self._entries.get(key)
        if hit is not None and hit[0] > now:
            self._entries.move_to_end(key)
            return hit[1]
        loop = asyncio.get_running_loop()
        pending = self._inflight.get(key)
        while pending is not None and pending.get_loop() is loop and not pending.done():
            # asyncio.wait neither raises the leader's outcome nor cancels the
            # shared future; only this task's own cancellation propagates.
            await asyncio.wait({pending})
            if not pending.cancelled():
                return pending.result()
            # The leader was cancelled: its cancellation is not ours, so the
            # next caller in line computes the value instead.
            hit = self._entries.get(key)
            if hit is not None and hit[0] > self._clock():
                self._entries.move_to_end(key)
                return hit[1]
            pending = self._inflight.get(key)
        future: asyncio.Future = loop.create_future()
        self._inflight[key] = future
        try:
            self.computations += 1
            value = await compute()
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise
        except BaseException as e:
            if not future.done():
                future.set_exception(e)
                # Retrieved here so a future nobody awaited does not warn.
                future.exception()
            raise
        else:
            self._entries[key] = (self._clock() + ttl, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
            if not future.done():
                future.set_result(value)
            return value
        finally:
            if self._inflight.get(key) is future:
                del self._inflight[key]


def _hash_link(**params: Any) -> str:
    query = {"ops": "1", **{k: v for k, v in params.items() if v is not None}}
    return "#/?" + urlencode(query)


def _age_s(moment: Optional[datetime], now: float) -> Optional[float]:
    if moment is None:
        return None
    return max(0.0, now - moment.timestamp())


def _epoch_iso(value: Any) -> Optional[str]:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return None


def _job_start_iso(value: Any) -> Optional[str]:
    """``job_start`` is an ISO string (UTC since PR-1; older ones naive)."""
    if isinstance(value, datetime):
        return utc_iso(value)
    if isinstance(value, str) and value:
        try:
            return utc_iso(datetime.fromisoformat(value))
        except ValueError:
            return None
    return None


def _file_fields(doc_id: str, path: Optional[str]) -> dict[str, Any]:
    return {"file": document_label(doc_id, path), "file_path_masked": mask_path(path)}


def _job_label(job_name: Any) -> Optional[str]:
    """A job name safe to show: the file name a batch name embeds is masked.

    Processing batches are named ``<first 20 chars of the first path>...[N
    files]``; the cut-off name may hold spaces and no extension, so it is
    masked as a whole rather than left to the token redactor.
    """
    if not isinstance(job_name, str) or not job_name or job_name == "-":
        return None
    match = _JOB_FILES.match(job_name)
    if match:
        prefix = match.group("prefix").strip()
        label = (
            "unknown source" if prefix in ("", "unknown_source") else mask_path(prefix)
        )
        return f"{label} [{match.group('count')} files]"
    return redact(job_name, 128)


def _pause_view(pause: Any, now: float) -> Optional[dict[str, Any]]:
    if not isinstance(pause, dict) or not pause:
        return None
    retry_at = pause.get("retry_at")
    return {
        "role": pause.get("role"),
        "kind": pause.get("kind"),
        "status_code": pause.get("status_code"),
        "since": _epoch_iso(pause.get("since")),
        "retry_at": _epoch_iso(retry_at),
        "active": isinstance(retry_at, (int, float)) and retry_at > now,
        "message": redact(pause.get("message"), 300) or None,
    }


def validate_workspace(workspace: str) -> str:
    if not _WORKSPACE_ID.match(workspace or ""):
        raise OpsBadRequest("invalid_workspace")
    return workspace


class OpsService:
    def __init__(
        self,
        *,
        store: Optional[OpsStore],
        reader: NamespaceReader,
        app_state: Any,
        unavailable_reason: Optional[str] = None,
        own_providers: Optional[OwnProviderCache] = None,
        clock: Callable[[], float] = time.time,
        cache: Optional[OpsCache] = None,
    ):
        self.store = store
        self.reader = reader
        self.app_state = app_state
        self._reason = unavailable_reason
        self.own_providers = own_providers
        self.clock = clock
        self.cache = cache or OpsCache()

    # -- availability --------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.store is not None and bool(getattr(self.store, "ready", False))

    @property
    def docs_available(self) -> bool:
        return self.available and bool(getattr(self.store, "doc_status_ready", False))

    @property
    def unavailable_reason(self) -> Optional[str]:
        if self.available:
            return None if self.docs_available else "doc_status_not_postgres"
        return self._reason or "postgres_required"

    def meta(self, *, needs_docs: bool = True) -> dict[str, Any]:
        """``available`` for a payload; ``needs_docs`` False for ops-table-only views."""
        ready = self.docs_available if needs_docs else self.available
        out: dict[str, Any] = {"available": ready, "manager": self.reader.state}
        if not ready:
            out["unavailable_reason"] = self.unavailable_reason
        return out

    # -- plumbing ----------------------------------------------------------------
    def _data_rag(self) -> Any:
        rag = getattr(self.app_state, "default_rag", None)
        if rag is not None:
            return rag
        server = getattr(self.app_state, "ops_server", None)
        return getattr(server, "rag", None)

    def namespace_of(self, workspace: str) -> str:
        """The ``pipeline_status`` workspace for a doc-status workspace.

        Storage may resolve an unnamed default workspace to its own name
        (``default``), while the pipeline keeps the instance's name.
        """
        rag = self._data_rag()
        storage = getattr(rag, "doc_status", None)
        storage_default = getattr(storage, "workspace", None)
        if storage_default is not None and workspace == storage_default:
            return getattr(rag, "workspace", "") or ""
        return workspace

    async def _q(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """Run one store call; any database failure becomes OpsDataUnavailable."""
        try:
            return await call()
        except (ValueError, OpsBadRequest, OpsNotFound):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Ops query failed (%s)", type(e).__name__)
            raise OpsDataUnavailable(type(e).__name__) from e

    async def _cached(self, key: tuple, kind: str, compute) -> Any:
        return await self.cache.get(key, TTL[kind], lambda: self._q(compute))

    def _require_docs(self) -> None:
        if not self.docs_available:
            raise OpsDataUnavailable(self.unavailable_reason or "unavailable")

    # -- shared pieces ------------------------------------------------------------
    async def thresholds(self) -> dict[str, Any]:
        if not self.available:
            return dict(DEFAULT_THRESHOLDS)

        async def load():
            stored = await self.store.kv_get("thresholds") or {}
            merged = dict(DEFAULT_THRESHOLDS)
            if isinstance(stored, dict):
                for key, default in DEFAULT_THRESHOLDS.items():
                    value = stored.get(key)
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and value > 0
                    ):
                        merged[key] = type(default)(value)
            return merged

        try:
            return await self._cached(("thresholds",), "thresholds", load)
        except OpsDataUnavailable:
            return dict(DEFAULT_THRESHOLDS)

    async def backlog_totals(self) -> dict[str, Any]:
        self._require_docs()
        totals = await self._cached(("totals",), "totals", self.store.backlog_totals)
        # The two counts that read many heap rows change slowly: cached longer.
        # They are extras: when one times out the totals still stand and the
        # count is unknown (null), rather than the whole backlog failing.
        try:
            dup = await self._cached(("dup",), "dup", self.store.dup_count)
        except OpsDataUnavailable:
            dup = None
        try:
            cancelled = await self._cached(
                ("cancelled",), "cancelled", self.store.cancelled_count
            )
        except OpsDataUnavailable:
            cancelled = None
        if cancelled is not None:
            cancelled = min(cancelled, totals["failed"])
        return {
            **totals,
            # Unknown cancellations stay counted as failed (an overcount, flagged).
            "failed": totals["failed"] - (cancelled or 0),
            "cancelled": cancelled,
            "dup": dup,
            "partial": dup is None or cancelled is None,
        }

    async def queue_summary(self) -> dict[str, Any]:
        if not self.available:
            return {}

        async def load():
            return await self.store.kv_get(SUMMARY_KEY) or {}

        return await self._cached(("queue",), "queue", load)

    async def pause_data(self) -> dict[str, Any]:
        self._require_docs()

        async def load():
            return await read_pauses(
                self.store,
                self.reader,
                namespace_of=self.namespace_of,
                own_providers=self.own_providers,
                now=self.clock(),
            )

        return await self._cached(("pauses",), "pauses", load)

    async def stale_page(self, minutes: int, page: Page) -> tuple[list, int]:
        self._require_docs()
        return await self._cached(
            ("stale", minutes, page.limit, page.offset),
            "stale",
            lambda: self.store.stale_docs(minutes, page),
        )

    async def llm_health(self) -> dict[str, Any]:
        insights = getattr(self.app_state, "insights", None)
        if insights is None:
            return {"available": False}

        async def load():
            end = datetime.now(timezone.utc)
            data = await insights.store.llm_health(end - timedelta(minutes=15), end)
            return {"available": True, "window": "15m", **data}

        try:
            return await self.cache.get(("llm",), TTL["llm"], load)
        except Exception as e:
            logger.warning("Ops LLM health read failed (%s)", type(e).__name__)
            return {"available": False}

    # -- active jobs (interim) ----------------------------------------------------------
    async def active(self) -> dict[str, Any]:
        self._require_docs()
        return await self.cache.get(("active",), TTL["active"], self._compute_active)

    async def _compute_active(self) -> dict[str, Any]:
        thresholds = await self.thresholds()
        candidates = await self._q(
            lambda: self.store.live_candidates(ACTIVE_CANDIDATES)
        )
        names = {c["workspace"]: self.namespace_of(c["workspace"]) for c in candidates}
        batch = await self.reader.snapshot_many(list(names.values()), LIVE_KEYS)
        now = self.clock()
        stale_s = thresholds["stale_doc_minutes"] * 60
        runs = []
        for c in candidates:
            snap = batch.data.get(names[c["workspace"]]) if batch.responsive else None
            last_at = as_utc(c.get("last_at"))
            oldest_at = as_utc(c.get("oldest_at"))
            progress_age = _age_s(last_at, now)
            counts = {s: int(c.get(s) or 0) for s in ACTIVE_STATUSES}
            stage = max(
                ACTIVE_STATUSES, key=lambda s: (counts[s], ACTIVE_STATUSES.index(s))
            )
            live = None
            pause = None
            if snap is not None:
                pause = _pause_view(snap.get("provider_rejection"), now)
                live = {
                    "busy": bool(snap.get("busy")),
                    "destructive_busy": bool(snap.get("destructive_busy")),
                    "scanning": bool(snap.get("scanning")),
                    "pending_enqueues": int(snap.get("pending_enqueues") or 0),
                    "cancellation_requested": bool(snap.get("cancellation_requested")),
                    "request_pending": bool(snap.get("request_pending")),
                    "paused": bool(pause and pause["active"]),
                }
            if not batch.responsive:
                state = "unresponsive"
            elif live and live["paused"]:
                state = "paused"
            elif (live and live["busy"]) or (
                progress_age is not None and progress_age < stale_s
            ):
                state = "running"
            else:
                state = "stale_doc"
            if live and live["destructive_busy"]:
                kind = "delete"
            elif live and live["scanning"]:
                kind = "scan"
            else:
                kind = "process"
            runs.append(
                {
                    "run_id": None,
                    "workspace": c["workspace"],
                    "kind": kind,
                    "trigger": None,
                    "label": _job_label(snap.get("job_name")) if snap else None,
                    "worker_id": None,
                    "pid": None,
                    "worker_alive": None,
                    "started_at": _job_start_iso(snap.get("job_start"))
                    if snap
                    else None,
                    "heartbeat_age_s": None,
                    "progress_age_s": progress_age,
                    "last_progress_at": utc_iso(last_at),
                    "oldest_doc_at": utc_iso(oldest_at),
                    "docs_done": int(snap.get("cur_batch") or 0) if snap else None,
                    "docs_total": int(snap.get("docs") or 0) if snap else None,
                    "docs_inflight": counts,
                    "stage": stage,
                    "state": state,
                    "live": live,
                    "pause": pause,
                    "latest_message": redact_pipeline(snap.get("latest_message"), 300)
                    or None
                    if snap
                    else None,
                }
            )
        order = {"running": 0, "paused": 1, "stale_doc": 2, "unresponsive": 3}
        runs.sort(key=lambda r: (order.get(r["state"], 9), r["progress_age_s"] or 0))
        return {
            "runs": runs,
            "approximate": True,
            "source": "doc_status",
            "checked": len(candidates),
            "capped": len(candidates) >= ACTIVE_CANDIDATES,
            "manager": "unresponsive" if not batch.responsive else self.reader.state,
        }

    # -- processing summary ------------------------------------------------------------
    async def summary(self) -> dict[str, Any]:
        self._require_docs()
        thresholds = await self.thresholds()
        backlog = await self.backlog_totals()
        active = await self.active()
        _, stale_total = await self.stale_page(
            thresholds["stale_doc_minutes"], Page(limit=1, offset=0)
        )
        queue = await self.queue_summary()
        try:
            pauses = await self.pause_data()
        except OpsDataUnavailable:
            pauses = {}
        running = [r for r in active["runs"] if r["state"] in ("running", "paused")]
        return {
            "backlog": self._backlog_view(backlog),
            "active": active["runs"][:5],
            "active_count": len(running),
            "stuck": {"runs": 0, "docs": stale_total, "reservations": 0},
            "email_queue": self._queue_counts(queue),
            "pauses": {
                "count": pauses.get("total", 0),
                "by_slot_kind": pauses.get("by_slot_kind", {}),
                "capped": pauses.get("capped", False),
            },
            "thresholds": thresholds,
            "approximate": True,
            "manager": active.get("manager"),
        }

    @staticmethod
    def _backlog_view(totals: dict[str, Any]) -> dict[str, Any]:
        return {
            "pending": totals["pending"],
            "parsing": totals["parsing"],
            "analyzing": totals["analyzing"],
            "processing": totals["processing"],
            "failed": totals["failed"],
            "cancelled": totals["cancelled"],
            "dup": totals["dup"],
            "partial": bool(totals.get("partial")),
            "inflight": totals["inflight"],
            "failed_24h": totals["failed_24h"],
            "cancelled_24h": totals["cancelled_24h"],
            "oldest_inflight_at": utc_iso(totals["oldest_inflight_at"]),
            "oldest_pending_at": utc_iso(totals["oldest_pending_at"]),
        }

    def _queue_counts(self, queue: dict[str, Any]) -> dict[str, Any]:
        scanned_at = queue.get("scanned_at")
        age = (
            self.clock() - scanned_at if isinstance(scanned_at, (int, float)) else None
        )
        longest = queue.get("longest_pending_s")
        if isinstance(longest, (int, float)) and age is not None:
            longest = longest + max(0.0, age)
        stalled = queue.get("longest_stalled_s")
        if isinstance(stalled, (int, float)) and age is not None:
            stalled = stalled + max(0.0, age)
        return {
            "total": int(queue.get("total") or 0),
            "workspaces": int(queue.get("workspaces") or 0),
            "bad": int(queue.get("bad") or 0),
            "longest_pending_s": longest,
            # Time since the slowest workspace's queue last shrank (§1.7: no
            # drain for that long); older summaries lack it.
            "longest_stalled_s": stalled,
            "scanned_at": _epoch_iso(scanned_at),
            "scan_age_s": age,
            "stale": age is None or age > 2 * SCAN_INTERVAL_S + 30,
        }

    # -- stuck ---------------------------------------------------------------------------
    async def stuck(
        self, doc_older_than_min: Optional[int], page: Page
    ) -> dict[str, Any]:
        self._require_docs()
        thresholds = await self.thresholds()
        minutes = doc_older_than_min or thresholds["stale_doc_minutes"]
        rows, total = await self.stale_page(minutes, page)
        now = self.clock()
        items = []
        for r in rows:
            updated = as_utc(r["updated_at"])
            items.append(
                {
                    "class": "stale_doc",
                    "workspace": r["workspace"],
                    "doc_id": r["id"],
                    "status": r["status"],
                    "track_id": r.get("track_id"),
                    "since": utc_iso(updated),
                    "age_s": _age_s(updated, now),
                    **_file_fields(r["id"], r.get("file_path")),
                }
            )
        return {
            "orphaned": [],
            "hung": [],
            "busy_without_run": [],
            "reservations": [],
            "stale_docs": {
                "items": items,
                "total": total,
                "page": page.offset // page.limit + 1,
                "page_size": page.limit,
            },
            "thresholds": {
                "doc_minutes": minutes,
                "hung_minutes": thresholds["hung_minutes"],
                "orphan_seconds": thresholds["orphan_seconds"],
            },
            # Orphaned, hung, busy-without-run and reservation detection need
            # run tracking; until then only stale documents are checked.
            "checked_classes": ["stale_docs"],
        }

    # -- backlog table -------------------------------------------------------------------
    async def workspaces(
        self,
        *,
        q: Optional[str],
        has: frozenset[str],
        sort: str,
        descending: bool,
        page: Page,
    ) -> dict[str, Any]:
        self._require_docs()
        if sort not in BACKLOG_SORTS:
            raise OpsBadRequest("invalid_sort")
        if set(has) - BACKLOG_FILTERS:
            raise OpsBadRequest("invalid_filter")
        thresholds = await self.thresholds()
        queue = await self.queue_summary()
        queue_items = {
            i["workspace"]: i for i in queue.get("items") or [] if isinstance(i, dict)
        }
        # Pauses also drive the state column, so they are read for every page
        # (cached for a minute, at most 50 shared-state reads).
        try:
            pauses = await self.pause_data()
            paused = {i["workspace"]: i for i in pauses.get("items", [])}
        except OpsDataUnavailable:
            paused = {}
        email_set = sorted(w for w, i in queue_items.items() if i.get("pending"))
        needle = (q or "").strip()[:SEARCH_MAX_LENGTH] or None
        key = (
            "backlog",
            needle,
            tuple(sorted(has)),
            sort,
            descending,
            page.limit,
            page.offset,
            thresholds["stale_doc_minutes"],
            tuple(email_set) if "email" in has else (),
            tuple(sorted(paused)) if "paused" in has else (),
        )
        rows, total = await self._cached(
            key,
            "backlog",
            lambda: self.store.backlog_page(
                q=needle,
                has=frozenset(has),
                sort=sort,
                descending=descending,
                page=page,
                stale_minutes=thresholds["stale_doc_minutes"],
                email_workspaces=email_set,
                paused_workspaces=sorted(paused),
            ),
        )
        items = []
        for r in rows:
            ws = r["workspace"]
            queued = queue_items.get(ws) or {}
            pause = paused.get(ws)
            active = r["parsing"] + r["analyzing"] + r["processing"]
            if pause:
                state = "paused"
            elif r.get("stuck"):
                state = "stale_doc"
            elif active:
                state = "running"
            elif r["pending"]:
                state = "pending"
            elif r["failed"]:
                state = "failed"
            else:
                state = "idle"
            items.append(
                {
                    "workspace": ws,
                    "owner": r.get("owner"),
                    "pending": r["pending"],
                    "parsing": r["parsing"],
                    "analyzing": r["analyzing"],
                    "processing": r["processing"],
                    "failed": r["failed"],
                    "cancelled": r["cancelled"],
                    "inflight": r["inflight"],
                    "oldest_inflight_at": utc_iso(r.get("oldest_inflight_at")),
                    "last_activity_at": utc_iso(r.get("last_activity_at")),
                    "queue_pending": int(queued.get("pending") or 0),
                    "paused": bool(pause),
                    "pause_kind": pause.get("kind") if pause else None,
                    "stuck": bool(r.get("stuck")),
                    "last_run": None,
                    "state": state,
                }
            )
        return {
            "items": items,
            "page": page.offset // page.limit + 1,
            "page_size": page.limit,
            "total": total,
            "sort": sort,
            "dir": "desc" if descending else "asc",
            # The paused filter only knows the capped pause candidates.
            "approximate_filters": ["paused"],
        }

    # -- failures ---------------------------------------------------------------------------
    async def failures(
        self, window: str, group: str, include_cancelled: bool
    ) -> dict[str, Any]:
        self._require_docs()
        if window not in FAILURE_WINDOWS:
            raise OpsBadRequest("invalid_window")
        if group not in DURABLE_FAILURE_GROUPS + LEGACY_FAILURE_GROUPS:
            raise OpsBadRequest("invalid_group")
        hours = FAILURE_WINDOWS[window]
        legacy_group = group if group in LEGACY_FAILURE_GROUPS else "signature"
        data = await self._cached(
            ("failures", hours, legacy_group, include_cancelled),
            "failures",
            lambda: self.store.legacy_failures(hours, legacy_group, include_cancelled),
        )
        dup = await self._cached(("dup",), "dup", self.store.dup_count)
        legacy = [
            {
                "key": g["key"],
                "kind": legacy_group,
                "count": int(g["count"]),
                "workspaces": int(g["workspaces"]),
                "first_seen": utc_iso(g.get("first_seen")),
                "last_seen": utc_iso(g.get("last_seen")),
                "example": redact_pipeline(
                    g.get("example"), 300, (g.get("example_path"),)
                )
                or None,
            }
            for g in data["groups"]
        ]
        cancelled = data["cancelled"]
        if include_cancelled and int(cancelled.get("count") or 0):
            legacy.append(
                {
                    "key": CANCELLED_GROUP_KEY,
                    "kind": "cancelled",
                    "count": int(cancelled["count"]),
                    "workspaces": int(cancelled.get("workspaces") or 0),
                    "first_seen": utc_iso(cancelled.get("first_seen")),
                    "last_seen": utc_iso(cancelled.get("last_seen")),
                    "example": redact_pipeline(
                        cancelled.get("example"),
                        300,
                        (cancelled.get("example_path"),),
                    )
                    or None,
                }
            )
        return {
            "window": window,
            "group": group,
            "legacy_group": legacy_group,
            # Failure kinds, stages and models are recorded once failures are
            # classified at the source; older rows are grouped by message.
            "groups": [],
            "durable": False,
            "legacy": legacy,
            "excluded": {"dup": dup, "cancelled": int(cancelled.get("count") or 0)},
        }

    async def failure_docs(
        self, key: str, group: str, window: str, page: Page
    ) -> dict[str, Any]:
        self._require_docs()
        if window not in FAILURE_WINDOWS:
            raise OpsBadRequest("invalid_window")
        if group not in LEGACY_FAILURE_GROUPS + (CANCELLED_GROUP_KEY,):
            raise OpsBadRequest("invalid_group")
        if not key or len(key) > 255:
            raise OpsBadRequest("invalid_key")
        hours = FAILURE_WINDOWS[window]
        rows, total = await self._cached(
            ("failure_docs", hours, group, key, page.limit, page.offset),
            "failure_docs",
            lambda: self.store.failure_docs(hours, group, key, page),
        )
        return {
            "items": [
                {
                    "workspace": r["workspace"],
                    "doc_id": r["id"],
                    "track_id": r.get("track_id"),
                    "updated_at": utc_iso(r.get("updated_at")),
                    "error": redact_pipeline(r.get("error"), 300, (r.get("file_path"),))
                    or None,
                    **_file_fields(r["id"], r.get("file_path")),
                }
                for r in rows
            ],
            "page": page.offset // page.limit + 1,
            "page_size": page.limit,
            "total": total,
        }

    # -- email queue -------------------------------------------------------------------------
    async def email_queue(self, page: Page) -> dict[str, Any]:
        if not self.available:
            raise OpsDataUnavailable(self.unavailable_reason or "unavailable")
        queue = await self.queue_summary()
        all_items = [i for i in queue.get("items") or [] if isinstance(i, dict)]
        picked = all_items[page.offset : page.offset + page.limit]
        resync = await self._q(
            lambda: self.store.kv_get_many([f"resync:{i['workspace']}" for i in picked])
        )
        now = self.clock()
        items = []
        for i in picked:
            state = resync.get(f"resync:{i['workspace']}") or {}
            first = i.get("first_pending_at")
            items.append(
                {
                    "workspace": i["workspace"],
                    "pending": int(i.get("pending") or 0),
                    "bad": int(i.get("bad") or 0),
                    "pending_since": _epoch_iso(first),
                    "pending_age_s": now - first
                    if isinstance(first, (int, float))
                    else None,
                    # The last scan that saw this queue shrink.
                    "drain_seen_at": _epoch_iso(i.get("last_progress_at")),
                    "attempts": state.get("attempts"),
                    "last_result_class": state.get("last_result_class"),
                    "reason": redact(state.get("reason"), 200) or None,
                    "last_attempt_at": _epoch_iso(state.get("last_attempt_at")),
                    "last_ok_at": _epoch_iso(state.get("last_ok_at")),
                    "drainer": None,
                }
            )
        return {
            **self._queue_counts(queue),
            "items": items,
            "page": page.offset // page.limit + 1,
            "page_size": page.limit,
            "total_items": len(all_items),
            "truncated": bool(queue.get("truncated")),
        }

    # -- pauses ----------------------------------------------------------------------------------
    async def pauses(self, page: Page) -> dict[str, Any]:
        data = await self.pause_data()
        items = data.get("items", [])
        return {
            "by_slot_kind": data.get("by_slot_kind", {}),
            "items": items[page.offset : page.offset + page.limit],
            "total": data.get("total", 0),
            "checked": data.get("checked", 0),
            "capped": data.get("capped", False),
            "approximate": True,
            "manager": "unresponsive"
            if not data.get("responsive", True)
            else self.reader.state,
            "page": page.offset // page.limit + 1,
            "page_size": page.limit,
        }

    # -- workspace drawer ------------------------------------------------------------------------
    async def _owner(self, workspace: str) -> Optional[str]:
        registry = getattr(self.app_state, "workspace_registry", None)
        if registry is not None:
            try:
                owner = await registry.get_by_workspace(workspace)
            except Exception as e:
                logger.warning("Ops owner lookup failed (%s)", type(e).__name__)
                owner = None
            if owner is not None:
                return owner.email or owner.identity
            return None
        if self.available:
            owners = await self._q(lambda: self.store.owners([workspace]))
            return owners.get(workspace)
        return None

    def _loaded_here(self, workspace: str) -> bool:
        manager = getattr(self.app_state, "workspace_manager", None)
        if manager is not None:
            try:
                return any(
                    getattr(r, "workspace", None) == workspace
                    for r in manager.loaded_instances()
                )
            except Exception:
                return False
        rag = self._data_rag()
        return self.namespace_of(workspace) == (getattr(rag, "workspace", None) or "")

    async def workspace(self, workspace: str) -> dict[str, Any]:
        validate_workspace(workspace)
        return await self.cache.get(
            ("workspace", workspace),
            TTL["workspace"],
            lambda: self._workspace(workspace),
        )

    async def _workspace(self, workspace: str) -> dict[str, Any]:
        now = self.clock()
        snap, responsive = await self.reader.snapshot(
            self.namespace_of(workspace), LIVE_KEYS
        )
        live = None
        if snap is not None:
            live = {
                "busy": bool(snap.get("busy")),
                "destructive_busy": bool(snap.get("destructive_busy")),
                "scanning": bool(snap.get("scanning")),
                "scanning_exclusive": bool(snap.get("scanning_exclusive")),
                "pending_enqueues": int(snap.get("pending_enqueues") or 0),
                "request_pending": bool(snap.get("request_pending")),
                "cancellation_requested": bool(snap.get("cancellation_requested")),
                "job": _job_label(snap.get("job_name")),
                "job_start": _job_start_iso(snap.get("job_start")),
                "docs": int(snap.get("docs") or 0),
                "cur_batch": int(snap.get("cur_batch") or 0),
                "latest_message_redacted": redact_pipeline(
                    snap.get("latest_message"), 500
                )
                or None,
                "provider_rejection": _pause_view(snap.get("provider_rejection"), now),
                "run_id": snap.get("run_id"),
                "owner_worker_id": snap.get("owner_worker_id"),
                "owner_pid": snap.get("owner_pid"),
            }
        owner = await self._owner(workspace)
        counts: Optional[dict[str, Any]] = None
        queue_view: Optional[dict[str, Any]] = None
        action = None
        if self.docs_available:
            raw = await self._q(lambda: self.store.workspace_counts(workspace))
            counts = {
                "statuses": raw["statuses"],
                "cancelled": raw["cancelled"],
                "dup": raw["dup"],
                "oldest_inflight_at": utc_iso(raw["oldest_inflight_at"]),
                "last_activity_at": utc_iso(raw["last_activity_at"]),
            }
        if self.available:
            queue = await self.queue_summary()
            queued = next(
                (
                    i
                    for i in queue.get("items") or []
                    if i.get("workspace") == workspace
                ),
                None,
            )
            kv = await self._q(
                lambda: self.store.kv_get_many(
                    [f"resync:{workspace}", f"action:{workspace}"]
                )
            )
            state = kv.get(f"resync:{workspace}") or {}
            first = (queued or {}).get("first_pending_at")
            queue_view = {
                "pending": int((queued or {}).get("pending") or 0),
                "bad": int((queued or {}).get("bad") or 0),
                "pending_since": _epoch_iso(first),
                "attempts": state.get("attempts"),
                "last_result_class": state.get("last_result_class"),
                "last_ok_at": _epoch_iso(state.get("last_ok_at")),
                "scanned_at": _epoch_iso(queue.get("scanned_at")),
            }
            action_value = kv.get(f"action:{workspace}")
            if isinstance(action_value, dict) and (action_value.get("exp") or 0) > now:
                action = {
                    "actor": action_value.get("actor"),
                    "action": action_value.get("action"),
                    "since": _epoch_iso(action_value.get("since")),
                }
        has_docs = bool(
            counts
            and (
                sum(counts["statuses"].values()) or counts["dup"] or counts["cancelled"]
            )
        )
        if (
            snap is None
            and owner is None
            and not has_docs
            and not (queue_view and (queue_view["pending"] or queue_view["bad"]))
            and responsive
            and self.docs_available
        ):
            raise OpsNotFound(workspace)
        return {
            "workspace": workspace,
            "owner": owner,
            # Workers report what they have loaded once heartbeats exist; until
            # then only this process can answer.
            "loaded_in": [],
            "loaded_here": self._loaded_here(workspace),
            "idle": snap is None,
            "live": live,
            "manager": "unresponsive" if not responsive else self.reader.state,
            "action_in_progress": action,
            "counts": counts,
            "email_queue": queue_view,
            "recent_runs": [],
            "failures_by_kind": [],
            "errors_1h": None,
            "available": self.docs_available,
        }

    async def feed(
        self, workspace: str, offset: Optional[int], limit: int
    ) -> dict[str, Any]:
        validate_workspace(workspace)
        limit = max(1, min(limit, FEED_MAX_LINES))
        name = self.namespace_of(workspace)
        snap, responsive = await self.reader.snapshot(
            name,
            ("history_messages",),
            list_slices={"history_messages": (offset, limit)},
        )
        if snap is None:
            return {
                "lines": [],
                "start": 0,
                "total": 0,
                "next_offset": 0,
                "reset": bool(offset),
                "loaded": False,
                "manager": "unresponsive" if not responsive else self.reader.state,
            }
        history = snap["history_messages"]
        total = history["total"]
        reset = offset is not None and offset > total
        if reset:
            # The list was cleared since the client's last read: send the tail.
            snap, responsive = await self.reader.snapshot(
                name,
                ("history_messages",),
                list_slices={"history_messages": (None, limit)},
            )
            history = (snap or {}).get("history_messages") or {
                "total": 0,
                "start": 0,
                "items": [],
            }
            total = history["total"]
        lines = [redact_pipeline(line, 2048) for line in history["items"]]
        return {
            "lines": lines,
            "start": history["start"],
            "total": total,
            "next_offset": history["start"] + len(lines),
            "reset": reset,
            "loaded": True,
            "manager": "unresponsive" if not responsive else self.reader.state,
        }

    async def documents(
        self, workspace: str, status: Optional[str], page: Page
    ) -> dict[str, Any]:
        validate_workspace(workspace)
        self._require_docs()
        rows, total = await self._cached(
            ("documents", workspace, status, page.limit, page.offset),
            "documents",
            lambda: self.store.workspace_documents(workspace, status, page),
        )
        return {
            "items": [
                {
                    "doc_id": r["id"],
                    "status": r["status"],
                    "cancelled": r["status"] == STATUS_FAILED
                    and is_cancelled(r.get("error")),
                    "track_id": r.get("track_id"),
                    "chunks_count": r.get("chunks_count"),
                    "created_at": utc_iso(r.get("created_at")),
                    "updated_at": utc_iso(r.get("updated_at")),
                    "error": redact_pipeline(r.get("error"), 300, (r.get("file_path"),))
                    or None,
                    **_file_fields(r["id"], r.get("file_path")),
                }
                for r in rows
            ],
            "page": page.offset // page.limit + 1,
            "page_size": page.limit,
            "total": total,
        }

    async def document(self, workspace: str, doc_id: str) -> dict[str, Any]:
        validate_workspace(workspace)
        if not doc_id or len(doc_id) > 255:
            raise OpsBadRequest("invalid_doc_id")
        self._require_docs()
        return await self.cache.get(
            ("document", workspace, doc_id),
            TTL["document"],
            lambda: self._document(workspace, doc_id),
        )

    async def _document(self, workspace: str, doc_id: str) -> dict[str, Any]:
        row = await self._q(lambda: self.store.document(workspace, doc_id))
        if row is None:
            raise OpsNotFound(doc_id)
        created = as_utc(row.get("created_at"))
        updated = as_utc(row.get("updated_at"))
        calls, calls_available = await self._document_calls(workspace, doc_id, created)
        return {
            "doc": {
                "doc_id": row["id"],
                "workspace": workspace,
                "status": row["status"],
                "cancelled": row["status"] == STATUS_FAILED
                and is_cancelled(row.get("error")),
                "track_id": row.get("track_id"),
                "chunks_count": row.get("chunks_count"),
                "content_length": row.get("content_length"),
                "created_at": utc_iso(created),
                "updated_at": utc_iso(updated),
                "stages": document_stages(
                    row["status"], row.get("metadata") or {}, created, updated
                ),
                "error_redacted": redact_pipeline(
                    row.get("error"), 2048, (row.get("file_path"),)
                )
                or None,
                # Classified failure history arrives with failure tracking.
                "failure": None,
                **_file_fields(row["id"], row.get("file_path")),
            },
            "failures": [],
            "calls": calls,
            "calls_available": calls_available,
            "logs": {"available": False, "count": None, "first": []},
        }

    async def _document_calls(
        self, workspace: str, doc_id: str, created: Optional[datetime]
    ) -> tuple[list[dict[str, Any]], bool]:
        insights = getattr(self.app_state, "insights", None)
        if insights is None:
            return [], False
        end = datetime.now(timezone.utc) + timedelta(minutes=1)
        start = (created - timedelta(hours=1)) if created else end - timedelta(days=90)
        start = max(start, end - timedelta(days=90))
        try:
            rows = await insights.store.calls(
                start,
                end,
                filters={"workspace": workspace},
                doc_id=doc_id,
                limit=DOC_CALLS_LIMIT,
            )
        except Exception as e:
            logger.warning("Ops document calls read failed (%s)", type(e).__name__)
            return [], False
        out = []
        for r in rows[:DOC_CALLS_LIMIT]:
            ts = r.get("ts")
            out.append(
                {
                    "ts": _epoch_iso(ts)
                    if isinstance(ts, (int, float))
                    else utc_iso(ts),
                    **{
                        k: r.get(k)
                        for k in (
                            "purpose",
                            "operation",
                            "model",
                            "prompt_tokens",
                            "completion_tokens",
                            "latency_ms",
                            "status",
                            "cache_hit",
                        )
                    },
                }
            )
        return out, True

    # -- events ------------------------------------------------------------------------------
    async def events(
        self, start: datetime, end: datetime, kinds: Optional[list[str]]
    ) -> dict[str, Any]:
        if not self.available:
            raise OpsDataUnavailable(self.unavailable_reason or "unavailable")
        wanted = [k for k in (kinds or [*EVENT_KINDS, AUDIT_EVENT_KIND])]
        unknown = set(wanted) - set(EVENT_KINDS) - {AUDIT_EVENT_KIND}
        if unknown:
            raise OpsBadRequest("invalid_kind")
        if end <= start or end - start > timedelta(days=EVENTS_MAX_DAYS):
            raise OpsBadRequest("invalid_range")
        # Minute granularity keeps the cache key stable across polls.
        key_start = start.replace(second=0, microsecond=0)
        key_end = end.replace(second=0, microsecond=0)
        return await self._cached(
            ("events", key_start, key_end, tuple(sorted(wanted))),
            "events",
            lambda: self._events(start, end, wanted),
        )

    async def _events(self, start, end, wanted) -> dict[str, Any]:
        usage_kinds = [k for k in wanted if k in EVENT_KINDS]
        items = []
        for row in await self.store.usage_events(start, end, usage_kinds, EVENTS_LIMIT):
            detail = {
                k: v
                for k, v in (row.get("detail") or {}).items()
                if k in _EVENT_DETAIL_KEYS
            }
            items.append(
                {
                    "ts": utc_iso(row["ts"]),
                    "kind": row["kind"],
                    "workspace": row.get("workspace"),
                    "purpose": row.get("purpose"),
                    "detail": detail,
                    "source": "usage",
                }
            )
        if AUDIT_EVENT_KIND in wanted:
            for row in await self.store.audit_events(start, end, EVENTS_LIMIT):
                items.append(
                    {
                        "ts": utc_iso(row["ts"]),
                        "kind": AUDIT_EVENT_KIND,
                        "workspace": row.get("workspace"),
                        "purpose": None,
                        "detail": {
                            "action": row.get("action"),
                            "outcome": row.get("outcome"),
                            "actor": row.get("actor"),
                        },
                        "source": "audit",
                    }
                )
        items.sort(key=lambda e: e["ts"] or "", reverse=True)
        return {
            "items": items[:EVENTS_LIMIT],
            "from": start.isoformat(),
            "to": end.isoformat(),
            "kinds": wanted,
        }

    # -- overview ------------------------------------------------------------------------------
    async def overview(self, workers: dict[str, Any]) -> dict[str, Any]:
        return await self.cache.get(
            (
                "overview",
                workers.get("mode"),
                workers.get("live"),
                workers.get("expected"),
            ),
            TTL["overview"],
            lambda: self._overview(workers),
        )

    async def _overview(self, workers: dict[str, Any]) -> dict[str, Any]:
        now = self.clock()
        now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()
        thresholds = await self.thresholds()
        failed_sections: list[str] = []
        backlog = active = queue = pauses = None
        stale_items: list = []
        stale_total = 0
        events: list = []
        if self.docs_available:
            try:
                backlog = await self.backlog_totals()
                if backlog.get("partial"):
                    failed_sections.append("backlog_counts")
            except OpsDataUnavailable:
                failed_sections.append("backlog")
            try:
                active = await self.active()
            except OpsDataUnavailable:
                failed_sections.append("active")
            try:
                stale_items, stale_total = await self.stale_page(
                    thresholds["stale_doc_minutes"], Page(limit=1, offset=0)
                )
            except OpsDataUnavailable:
                failed_sections.append("stuck")
            try:
                pauses = await self.pause_data()
            except OpsDataUnavailable:
                failed_sections.append("pauses")
        if self.available:
            try:
                queue = await self.queue_summary()
            except OpsDataUnavailable:
                failed_sections.append("email_queue")
            try:
                end = datetime.fromtimestamp(now, timezone.utc)
                events = (
                    await self.events(
                        end - timedelta(hours=OVERVIEW_TIMELINE_HOURS), end, None
                    )
                )["items"]
            except (OpsDataUnavailable, OpsBadRequest):
                failed_sections.append("events")
        llm = await self.llm_health()
        postgres_ok = self.available and not failed_sections
        # A slow query (a statement timeout on a heavy count) is a degraded
        # section; only a trivial probe that fails too means the database is
        # down.
        postgres_down = (
            self.available and bool(failed_sections) and not await self._probe()
        )
        attention = self._attention(
            thresholds=thresholds,
            backlog=backlog,
            stale_items=stale_items,
            stale_total=stale_total,
            queue=queue,
            pauses=pauses,
            llm=llm,
            workers=workers,
            postgres_failed=postgres_down,
            now=now,
        )
        worst = max(
            (a["severity"] for a in attention), key=_SEVERITY_RANK.get, default=None
        )
        severity_counts = {
            s: sum(1 for a in attention if a["severity"] == s)
            for s in ("critical", "serious", "warning")
        }
        if not self.available:
            state = "unknown"
        elif postgres_down or workers.get("live") == 0:
            state = "down"
        elif worst:
            state = "degraded"
        else:
            state = "healthy"
        active_runs = (active or {}).get("runs", [])
        running = [r for r in active_runs if r["state"] in ("running", "paused")]
        longest = max(
            (
                now - datetime.fromisoformat(r["started_at"]).timestamp()
                for r in running
                if r.get("started_at")
            ),
            default=None,
        )
        stuck_known = "stuck" not in failed_sections and self.docs_available
        # No summary yet (the first scan runs a minute after startup): unknown.
        queue_counts = self._queue_counts(queue) if queue else None
        kpis = {
            "active": {
                "value": len(running) if active is not None else None,
                "longest_s": longest,
                "approximate": True,
                "scope": "host",
            },
            "backlog": {
                "value": backlog["inflight"] if backlog else None,
                "pending": backlog["pending"] if backlog else None,
                "oldest_inflight_at": utc_iso(backlog["oldest_inflight_at"])
                if backlog
                else None,
                # A trend needs stored history; not recorded yet.
                "delta_1h": None,
                "scope": "host",
            },
            "failed_24h": {
                "value": backlog["failed_24h"] if backlog else None,
                "cancelled": backlog["cancelled_24h"] if backlog else None,
                "dup": backlog["dup"] if backlog else None,
                "scope": "host",
            },
            # Unknown parts are null, never 0: a failed read is not "nothing stuck".
            "stuck": {
                "value": stale_total if stuck_known else None,
                "runs": 0 if stuck_known else None,
                "docs": stale_total if stuck_known else None,
                "reservations": 0 if stuck_known else None,
                "scope": "host",
            },
            "email_queue": {
                "value": queue_counts["total"] if queue_counts else None,
                "workspaces": queue_counts["workspaces"] if queue_counts else None,
                "longest_pending_s": queue_counts["longest_pending_s"]
                if queue_counts
                else None,
                "bad": queue_counts["bad"] if queue_counts else None,
                "stale": queue_counts["stale"] if queue_counts else True,
                "scope": "host",
            },
            "llm": {
                "available": bool(llm.get("available")),
                "error_rate": llm.get("error_rate") if llm.get("available") else None,
                "p95_ms": llm.get("p95_ms") if llm.get("available") else None,
                "calls": llm.get("calls") if llm.get("available") else None,
                "window": "15m",
                "scope": "host",
            },
            # Dependency probes arrive with the Health tab.
            "deps": {"available": False, "ok": None, "total": None, "scope": "host"},
            "workers": {
                "live": workers.get("live"),
                "expected": workers.get("expected"),
                "mode": workers.get("mode"),
                "scope": "host",
            },
        }
        for name, tile in kpis.items():
            tile["updated_at"] = now_iso
            tile["state"] = _tile_state(name, tile, attention)
        return {
            "state": state,
            "worst": worst or ("ok" if self.available else "unknown"),
            "severity_counts": severity_counts,
            "maintenance": None,
            "kpis": kpis,
            "attention": attention,
            "checked": {
                "workspaces_with_backlog": None,
                "active_candidates": (active or {}).get("checked"),
                "email_queue": queue is not None,
                "at": now_iso,
            },
            "workers": [],
            "timeline": {
                "buckets": [],
                "events": events[:100],
                "hours": OVERVIEW_TIMELINE_HOURS,
            },
            "active_top": active_runs[:5],
            "deps": [],
            "alert_channel": None,
            "approximate": True,
            "postgres_ok": postgres_ok,
            "failed_sections": failed_sections,
            "manager": self.reader.state,
            **(
                {"unavailable_reason": self.unavailable_reason}
                if not self.docs_available
                else {}
            ),
        }

    async def _probe(self) -> bool:
        """True when Postgres answers a trivial query."""
        try:
            await self._q(self.store.ping)
        except OpsDataUnavailable:
            return False
        return True

    def _attention(
        self,
        *,
        thresholds: dict[str, Any],
        backlog: Optional[dict[str, Any]],
        stale_items: list,
        stale_total: int,
        queue: Optional[dict[str, Any]],
        pauses: Optional[dict[str, Any]],
        llm: dict[str, Any],
        workers: dict[str, Any],
        postgres_failed: bool,
        now: float,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        def add(rule, severity, title, link, *, params=None, since=None, count=None):
            items.append(
                {
                    "id": rule,
                    "rule": rule,
                    "severity": severity,
                    "title": title,
                    "params": params or {},
                    "since": since,
                    "count": count,
                    "link": link,
                    "runbook": RUNBOOK.get(rule),
                }
            )

        if postgres_failed:
            add(
                "pg_probe",
                "critical",
                "Postgres is not answering ops queries",
                _hash_link(otab="health"),
            )
        if self.reader.state == "unresponsive" and workers.get("mode") == "gunicorn":
            add(
                "manager_unresponsive",
                "critical",
                "Shared pipeline state is not answering",
                _hash_link(otab="health", osub="workers"),
            )
        if pauses and pauses.get("total"):
            add(
                "provider_paused",
                "critical",
                f"{pauses['total']} workspace(s) paused by the LLM provider",
                _hash_link(otab="processing", osub="pauses"),
                params={
                    "count": pauses["total"],
                    "by_slot_kind": pauses.get("by_slot_kind", {}),
                },
                count=pauses["total"],
            )
        if stale_total and stale_items:
            oldest = as_utc(stale_items[0]["updated_at"])
            age_min = (_age_s(oldest, now) or 0) / 60
            severity = (
                "serious"
                if age_min >= thresholds["stale_doc_serious_minutes"]
                else "warning"
            )
            add(
                "stale_docs",
                severity,
                f"{stale_total} document(s) in progress for over {thresholds['stale_doc_minutes']} min",
                _hash_link(otab="processing", osub="stuck"),
                params={
                    "count": stale_total,
                    "threshold_min": thresholds["stale_doc_minutes"],
                    "oldest_min": round(age_min),
                },
                since=utc_iso(oldest),
                count=stale_total,
            )
        if backlog and backlog.get("oldest_pending_at") is not None:
            age_min = (_age_s(backlog["oldest_pending_at"], now) or 0) / 60
            if age_min >= thresholds["oldest_pending_minutes"]:
                severity = (
                    "serious"
                    if age_min >= thresholds["oldest_pending_serious_minutes"]
                    else "warning"
                )
                add(
                    "oldest_pending",
                    severity,
                    f"A document has been pending for {round(age_min)} min",
                    _hash_link(otab="processing", osub="backlog"),
                    params={
                        "oldest_min": round(age_min),
                        "pending": backlog.get("pending"),
                    },
                    since=utc_iso(backlog["oldest_pending_at"]),
                )
        if queue:
            counts = self._queue_counts(queue)
            if counts["bad"]:
                add(
                    "email_queue_bad",
                    "warning",
                    f"{counts['bad']} queued email re-sync(s) could not be read",
                    _hash_link(otab="processing", osub="email"),
                    params={"bad": counts["bad"]},
                    count=counts["bad"],
                )
            # Re-syncs keep arriving while the queue drains, so the rule reads
            # the time since a scan last saw a queue shrink, not how long it
            # has been non-empty (that is only the fallback for old summaries).
            stalled = counts["longest_stalled_s"]
            basis = "no_drain_seen"
            if not isinstance(stalled, (int, float)):
                stalled, basis = counts["longest_pending_s"], "pending_since"
            if (
                isinstance(stalled, (int, float))
                and stalled >= thresholds["email_waiting_minutes"] * 60
            ):
                minutes = stalled / 60
                severity = (
                    "serious"
                    if minutes >= thresholds["email_waiting_serious_minutes"]
                    else "warning"
                )
                add(
                    "email_queue_stalled",
                    severity,
                    f"Email re-syncs have not drained for {round(minutes)} min",
                    _hash_link(otab="processing", osub="email"),
                    params={
                        "waiting_min": round(minutes),
                        "records": counts["total"],
                        "basis": basis,
                        # Seen by a once-a-minute scan; drain results come later.
                        "approximate": True,
                    },
                )
        if (
            llm.get("available")
            and (llm.get("calls") or 0) >= thresholds["llm_min_calls"]
        ):
            rate = llm.get("error_rate") or 0.0
            if rate > thresholds["llm_error_rate"]:
                severity = (
                    "serious"
                    if rate > thresholds["llm_error_rate_serious"]
                    else "warning"
                )
                add(
                    "llm_error_rate_15m",
                    severity,
                    f"LLM error rate {rate:.0%} over 15 min",
                    _hash_link(otab="performance", osub="llm"),
                    params={"error_rate": rate, "calls": llm.get("calls")},
                )
        items.sort(key=lambda a: -_SEVERITY_RANK[a["severity"]])
        return items


_TILE_RULES = {
    "stuck": ("stale_docs",),
    "backlog": ("oldest_pending",),
    "email_queue": ("email_queue_bad", "email_queue_stalled"),
    "llm": ("llm_error_rate_15m",),
    "active": ("provider_paused",),
}


def _tile_state(
    name: str, tile: dict[str, Any], attention: list[dict[str, Any]]
) -> str:
    """A tile's state: the worst attention rule that feeds it, else ok.

    A tile without data is unknown, never ok.
    """
    if name == "workers":
        live, expected = tile.get("live"), tile.get("expected")
        if not isinstance(live, int) or not isinstance(expected, int) or expected <= 0:
            return "unknown"
        if live <= 0:
            return "critical"
        return "warning" if live < expected else "ok"
    if name in ("llm", "deps"):
        if not tile.get("available"):
            return "unknown"
    elif tile.get("value") is None:
        return "unknown"
    rules = _TILE_RULES.get(name, ())
    severities = [a["severity"] for a in attention if a["rule"] in rules]
    return max(severities, key=_SEVERITY_RANK.get, default="ok")


def document_stages(
    status: str,
    metadata: dict[str, Any],
    created: Optional[datetime],
    updated: Optional[datetime],
) -> list[dict[str, Any]]:
    """Stage timeline from the ``*_start_time`` epoch fields of doc metadata."""

    def epoch(key: str) -> Optional[float]:
        value = metadata.get(key)
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    parse = epoch("parsing_start_time")
    analyze = epoch("analyzing_start_time")
    process = epoch("processing_start_time")
    end = epoch("processing_end_time")
    failed = status == STATUS_FAILED
    final_at = end or (
        updated.timestamp()
        if updated and status in ("processed", STATUS_FAILED)
        else None
    )

    stages = [
        {
            "id": "enqueued",
            "started_at": utc_iso(created),
            "ended_at": utc_iso(created),
            "duration_s": 0.0 if created else None,
            "state": "done" if created else "unknown",
        }
    ]
    plan = [
        ("parse", parse, analyze or process, "parsing"),
        ("analyze", analyze, process, "analyzing"),
        ("extract", process, end, "processing"),
    ]
    started_any = False
    for index, (stage_id, start, stop, active_status) in enumerate(plan):
        later_started = any(p[1] for p in plan[index + 1 :])
        if start is None:
            skipped = later_started or (
                stage_id == "parse" and metadata.get("parse_stage_skipped")
            )
            stages.append(
                {
                    "id": stage_id,
                    "started_at": None,
                    "ended_at": None,
                    "duration_s": None,
                    "state": "skipped" if skipped else "pending",
                }
            )
            continue
        started_any = True
        if stop is None and not later_started:
            if failed:
                stop_at = final_at
                state = "failed"
            elif status == active_status:
                stop_at = None
                state = "running"
            elif status == "processed":
                stop_at = final_at
                state = "done"
            else:
                stop_at = None
                state = "interrupted"
        else:
            stop_at = stop
            state = "done"
        stages.append(
            {
                "id": stage_id,
                "started_at": _epoch_iso(start),
                "ended_at": _epoch_iso(stop_at),
                "duration_s": (stop_at - start) if stop_at is not None else None,
                "state": state,
            }
        )
    if failed and not started_any:
        stages[-1]["state"] = "failed"
    return stages
