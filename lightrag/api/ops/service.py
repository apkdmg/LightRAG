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

With run tracking on (``OPS_RUN_TRACKING``, the default) live jobs, stuck
runs, failure kinds and provider pauses come from ``LIGHTRAG_OPS_RUNS``,
``LIGHTRAG_OPS_DOC_FAILURES`` and persisted KV values. With it off, or for
work the durable tables do not cover yet (a run started before the upgrade),
the interim sources answer: in-flight documents plus shared state, labelled
approximate.
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
from uuid import UUID

from lightrag.api.insights.analytics import document_label
from lightrag.api.ops.audit import SearchAudit
from lightrag.api.ops.health import (
    DEPLOY_SUPPRESS_S,
    build_strips,
    dep_card,
    deps_summary,
    derived_deps,
    health_attention,
    host_info,
    order_workers,
    strip_cell_status,
    worker_item,
)
from lightrag.api.ops.heartbeat import ALIVE_WITHIN_S
from lightrag.api.ops.pauses import OwnProviderCache, read_pauses
from lightrag.api.ops.probes import STRIP_SLOT_S, ProbeSpec
from lightrag.api.ops.queue_scan import SCAN_INTERVAL_S, SUMMARY_KEY
from lightrag.api.ops.redact import mask_path, redact, redact_pipeline
from lightrag.api.ops.runs import job_label
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
OVERVIEW_WORKERS = 20
WORKER_ROWS_LIMIT = 200
RECENT_PROBES = 50
STRIP_HOURS = 24
ATTENTION_KEY = "attention"
# The leased snapshot is used while it is younger than three reconcile cycles.
ATTENTION_FRESH_S = 180.0
# Health cards read from heartbeats rather than probes.
DERIVED_DEPS = ("manager", "recorder", "log_sink")

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
    "workers": 5.0,
    "probes": 5.0,
    "strips": 30.0,
    "dependency": 15.0,
    "worker_history": 30.0,
    "attention": 5.0,
    "stuck_runs": 15.0,
    "busy_without_run": 15.0,
    "reservations": 15.0,
    "runs": 5.0,
    "run": 5.0,
    "recent_runs": 10.0,
    "failures_by_kind": 30.0,
    "doc_failures": 10.0,
    "pauses_durable": 15.0,
}

DEFAULT_THRESHOLDS = {
    "stale_doc_minutes": 60,
    "stale_doc_serious_minutes": 180,
    "hung_minutes": 15,
    "hung_serious_minutes": 45,
    "orphan_seconds": 60,
    "reservation_minutes": 5,
    "busy_without_run_minutes": 2,
    "oldest_pending_minutes": 120,
    "oldest_pending_serious_minutes": 360,
    "email_waiting_minutes": 30,
    "email_waiting_serious_minutes": 120,
    "llm_error_rate": 0.05,
    "llm_error_rate_serious": 0.20,
    "llm_min_calls": 20,
}

FAILURE_WINDOWS = {"1h": 1, "24h": 24, "7d": 168}
# Groups from the failure taxonomy (LIGHTRAG_OPS_DOC_FAILURES); documents that
# failed before classification are grouped by message (legacy).
DURABLE_FAILURE_GROUPS = ("error_kind", "stage", "model")
LEGACY_FAILURE_GROUPS = ("signature", "workspace")
EVENT_KINDS = ("deployment", "model_change", "provider_rejection", "ingestion_resumed")
AUDIT_EVENT_KIND = "ops_action"
RUN_EVENT_KIND = "run_orphaned"
BUSY_WITHOUT_RUN_KEY = "busy_without_run"
RESERVATIONS_KEY = "reservations"
RUNNING_LIMIT = 200
ORPHANED_HOURS = 24
STUCK_LIMIT = 100
RUN_FAILURES_LIMIT = 200
DOC_FAILURES_LIMIT = 50
RECENT_RUNS = 5
WORKSPACE_FAILURE_HOURS = 24
OPEN_PAUSES_LIMIT = 5000
RUNS_PAGE_MAX = 200
# KV results of the reconcile cycle older than this are flagged stale.
RECONCILE_FRESH_S = 180.0
# Run states that count as "active" on the Overview.
ACTIVE_RUN_STATES = ("running", "paused", "hung", "cancelling", "unresponsive")
# Runs that never report per-unit progress are not judged by it: a clear is
# one storage drop, and a scan waiting for processing is covered by the
# process run it started.
_HUNG_EXEMPT_KINDS = ("clear",)
_HUNG_EXEMPT_STAGES = {"scan": ("process",)}
_RUN_ID = re.compile(r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F-]{36}$")
_CURSOR = re.compile(r"^(?P<us>\d{1,19}):(?P<run>[0-9a-f]{32})$")
_EVENT_DETAIL_KEYS = ("old", "new", "status", "byo", "slot", "version", "paused")
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,255}$")
_SEVERITY_RANK = {"warning": 1, "serious": 2, "critical": 3}
_DEP_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

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
    "manager_rtt": "Shared pipeline state is slow or not answering; check the "
    "workers' CPU and restart the container if it persists.",
    "pg_probe": "Postgres is slow or not answering; check the database container "
    "and its connections.",
    "pg_connections": "Find the clients holding connections (Health > Postgres) "
    "before raising max_connections.",
    "pg_blocked": "Open Health > Postgres for the blocking backend's statement "
    "type and fingerprint.",
    "pg_pool_wait": "Look for long queries holding pool connections, then the "
    "pool size.",
    "dependency_down": "Open the dependency in Health and check its container; "
    "processing that needs it waits or fails until it is back.",
    "vllm_waiting": "Compare vLLM's queue with LightRAG's own call rate before "
    "blaming LightRAG load: outside callers reach vLLM through the proxy.",
    "smtp_probe": "Check the SMTP server and SMTP_* settings; alert emails "
    "cannot be delivered meanwhile.",
    "tls_cert_expiry": "Renew the certificate before it expires.",
    "disk_free": "Free space on the volume holding the working directory or the logs.",
    "workers_below_expected": "Open Health > Workers and check the container; a "
    "worker that stopped reporting may have crashed.",
    "log_sink_or_recorder_drops_5m": "Usage records or log lines are being "
    "lost; check Postgres health and write volume (Health > Workers shows "
    "each worker's buffers and drops).",
    "run_orphaned": "The worker that owned this run died and its workspace is "
    "still marked busy; restart the container to clear it (the documents stay "
    "and are processed again).",
    "run_hung": "Open the run's live feed: a long LLM call or a stuck "
    "dependency keeps it from progressing. Cancel it if it does not move.",
    "busy_without_run": "The workspace is marked busy but no run owns it; if "
    "it stays that way, restart the container.",
    "worker_unresponsive": "A worker holding a workspace stopped sending "
    "heartbeats but its process still exists; check its CPU in Health > "
    "Workers and restart the container if it does not recover.",
    "reservation_without_owner": "A scan or upload reservation outlived its "
    "owner, which blocks scans and deletes; restart the container to clear it.",
}

# What the console suggests for each stuck class (copy lives in the WebUI).
STUCK_GUIDANCE = {
    "orphaned": "restart_container",
    "hung": "check_feed_or_cancel",
    "unresponsive": "restart_container",
    "busy_without_run": "restart_container",
    "reservations": "restart_container",
    "stale_docs": "run_pipeline",
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

    def forget(self, key: tuple) -> None:
        """Drop one cached value (after a write that changes it)."""
        self._entries.pop(key, None)

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


def _epoch_of(value: Any) -> Optional[float]:
    """Epoch seconds of an ISO timestamp (naive means UTC)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _probe_row_view(row: dict[str, Any], ts: Optional[str]) -> dict[str, Any]:
    return {
        "ts": ts,
        "status": row.get("status"),
        "latency_ms": row.get("latency_ms"),
        "error_class": row.get("error_class"),
        "detail": row.get("detail") or {},
    }


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
    """A job name safe to show (see :func:`lightrag.api.ops.runs.job_label`)."""
    return job_label(job_name)


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


def _drains_view(state: dict[str, Any], now: float) -> Optional[dict[str, int]]:
    """Drain outcomes of the last hour (ok / failed / slot busy), or None when
    unknown or the counts are older than the hour they cover."""
    drains = state.get("drains_1h")
    last = state.get("last_attempt_at")
    if not isinstance(drains, dict):
        return None
    if not isinstance(last, (int, float)) or now - last > float(
        drains.get("window_s") or 3600
    ):
        return {"ok": 0, "failed": 0, "slot_busy": 0}

    def count(name: str) -> int:
        value = drains.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    return {
        "ok": count("ok"),
        "failed": count("error"),
        "slot_busy": count("slot_busy"),
    }


def failure_meta_view(metadata: Any) -> Optional[dict[str, Any]]:
    """``metadata.failure`` of a doc-status row (current failure state)."""
    failure = metadata.get("failure") if isinstance(metadata, dict) else None
    if not isinstance(failure, dict):
        return None
    attempt = failure.get("attempt")
    return {
        "stage": failure.get("stage"),
        "error_kind": failure.get("error_kind"),
        "http_status": failure.get("http_status"),
        "exc_type": failure.get("exc_type"),
        "attempt": attempt if isinstance(attempt, int) else None,
        "first_failed_at": failure.get("first_failed_at"),
        "failed_at": failure.get("failed_at"),
        "last_run_id": failure.get("last_run_id"),
        "model": failure.get("model"),
    }


def _failure_row_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": utc_iso(row.get("ts")),
        "run_id": row.get("run_id"),
        "track_id": row.get("track_id"),
        "stage": row.get("stage"),
        "error_kind": row.get("error_kind"),
        "http_status": row.get("http_status"),
        "exc_type": row.get("exc_type"),
        "attempt": row.get("attempt"),
        "model": row.get("model"),
        # Stored redacted and path-masked; redacted again on the way out.
        "message": redact(row.get("message"), 1024) or None,
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
        run_tracking: bool = False,
    ):
        self.store = store
        self.reader = reader
        self.app_state = app_state
        self._reason = unavailable_reason
        self.own_providers = own_providers
        self.clock = clock
        self.cache = cache or OpsCache()
        self.started_at = clock()
        # Durable run and failure sources (OPS_RUN_TRACKING and a store).
        self.run_tracking = run_tracking
        # Set by the ops runtime when the features run in this process.
        self.probes: Any = None
        self.heartbeat: Any = None
        self.worker_id: Optional[str] = None
        # Log searches are audited once per actor and filters every 5 min.
        self.search_audit = SearchAudit()

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
        if self.durable_runs:
            return await self._cached(
                ("pauses",), "pauses_durable", self._durable_pauses
            )
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

    # -- runs: shared helpers ---------------------------------------------------------
    @property
    def durable_runs(self) -> bool:
        """Runs, failures and pauses come from the durable tables."""
        return self.run_tracking and self.available

    def _run_view(self, row: dict[str, Any], now: float) -> dict[str, Any]:
        """A ``LIGHTRAG_OPS_RUNS`` row as the console shows it."""
        started = as_utc(row.get("started_at"))
        finished = as_utc(row.get("finished_at"))
        worker_age = row.get("worker_heartbeat_age_s")
        if worker_age is None and row.get("worker_host") is None:
            worker_alive = None
        else:
            worker_alive = (
                isinstance(worker_age, (int, float))
                and worker_age < ALIVE_WITHIN_S
                and not row.get("worker_superseded")
                and row.get("worker_exited_at") is None
            )
        end = finished.timestamp() if finished else now
        return {
            "run_id": row["run_id"],
            "workspace": row["workspace"],
            "kind": row["kind"],
            "trigger": row.get("trigger"),
            "status": row["status"],
            "state": row["status"],
            "was_orphaned": bool(row.get("was_orphaned")),
            # Stored masked; masked again so a raw batch name never shows.
            "label": job_label(row.get("label")),
            "worker_id": row.get("worker_id"),
            "pid": row.get("pid"),
            "worker_alive": worker_alive,
            "started_at": utc_iso(started),
            "finished_at": utc_iso(finished),
            "heartbeat_at": utc_iso(row.get("heartbeat_at")),
            "last_progress_at": utc_iso(row.get("progress_at")),
            "heartbeat_age_s": row.get("heartbeat_age_s"),
            "progress_age_s": row.get("progress_age_s"),
            "worker_heartbeat_age_s": worker_age,
            "duration_s": max(0.0, end - started.timestamp()) if started else None,
            "docs_total": int(row.get("docs_total") or 0),
            "docs_done": int(row.get("docs_done") or 0),
            "docs_failed": int(row.get("docs_failed") or 0),
            "stage": row.get("stage"),
            "error_kind": row.get("error_kind"),
            "error_summary": redact(row.get("error_summary"), 512) or None,
            "request_id": row.get("request_id"),
            # How the owner's death was confirmed (orphaned runs).
            "confirmed_by": row.get("orphaned_by"),
            "source": "runs",
        }

    @staticmethod
    def _judged_by_progress(view: dict[str, Any]) -> bool:
        """Whether a running run reports per-unit progress (can be hung)."""
        if view.get("kind") in _HUNG_EXEMPT_KINDS:
            return False
        return view.get("stage") not in _HUNG_EXEMPT_STAGES.get(view.get("kind"), ())

    @staticmethod
    def _live_view(snap: Optional[dict[str, Any]], now: float) -> tuple[Any, Any]:
        if snap is None:
            return None, None
        pause = _pause_view(snap.get("provider_rejection"), now)
        live = {
            "busy": bool(snap.get("busy")),
            "destructive_busy": bool(snap.get("destructive_busy")),
            "scanning": bool(snap.get("scanning")),
            "pending_enqueues": int(snap.get("pending_enqueues") or 0),
            "cancellation_requested": bool(snap.get("cancellation_requested")),
            "request_pending": bool(snap.get("request_pending")),
            "paused": bool(pause and pause["active"]),
            "run_id": snap.get("run_id"),
            "owner_worker_id": snap.get("owner_worker_id"),
        }
        return live, pause

    def _running_state(
        self, view: dict[str, Any], live: Optional[dict[str, Any]], hung_s: float
    ) -> str:
        """``running``, ``paused``, ``cancelling``, ``hung`` or
        ``unresponsive`` (the owner worker stopped beating; its death is not
        confirmed, else the run would be orphaned)."""
        if view["worker_alive"] is False:
            return "unresponsive"
        if live and live["paused"]:
            return "paused"
        if (
            live
            and live["cancellation_requested"]
            and live.get("run_id") == view["run_id"]
        ):
            return "cancelling"
        age = view.get("progress_age_s")
        if (
            isinstance(age, (int, float))
            and age >= hung_s
            and self._judged_by_progress(view)
        ):
            return "hung"
        return "running"

    # -- active jobs ----------------------------------------------------------------------
    async def active(self) -> dict[str, Any]:
        self._require_docs()
        if self.durable_runs:
            return await self.cache.get(
                ("active",), TTL["active"], self._compute_durable_active
            )
        return await self.cache.get(("active",), TTL["active"], self._compute_active)

    def _legacy_active_row(
        self,
        c: dict[str, Any],
        snap: Optional[dict[str, Any]],
        responsive: bool,
        now: float,
        stale_s: float,
    ) -> dict[str, Any]:
        last_at = as_utc(c.get("last_at"))
        oldest_at = as_utc(c.get("oldest_at"))
        progress_age = _age_s(last_at, now)
        counts = {s: int(c.get(s) or 0) for s in ACTIVE_STATUSES}
        stage = max(
            ACTIVE_STATUSES, key=lambda s: (counts[s], ACTIVE_STATUSES.index(s))
        )
        live, pause = self._live_view(snap, now)
        if live is not None:
            live.pop("run_id", None)
            live.pop("owner_worker_id", None)
        if not responsive:
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
        return {
            "run_id": None,
            "workspace": c["workspace"],
            "kind": kind,
            "trigger": None,
            "label": _job_label(snap.get("job_name")) if snap else None,
            "worker_id": None,
            "pid": None,
            "worker_alive": None,
            "started_at": _job_start_iso(snap.get("job_start")) if snap else None,
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
            "latest_message": redact_pipeline(snap.get("latest_message"), 300) or None
            if snap
            else None,
            "source": "doc_status",
        }

    async def _compute_active(self) -> dict[str, Any]:
        """Interim: in-flight documents plus shared state (tracking off)."""
        thresholds = await self.thresholds()
        candidates = await self._q(
            lambda: self.store.live_candidates(ACTIVE_CANDIDATES)
        )
        names = {c["workspace"]: self.namespace_of(c["workspace"]) for c in candidates}
        batch = await self.reader.snapshot_many(list(names.values()), LIVE_KEYS)
        now = self.clock()
        stale_s = thresholds["stale_doc_minutes"] * 60
        runs = [
            self._legacy_active_row(
                c,
                batch.data.get(names[c["workspace"]]) if batch.responsive else None,
                batch.responsive,
                now,
                stale_s,
            )
            for c in candidates
        ]
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

    async def _compute_durable_active(self) -> dict[str, Any]:
        """Running runs from ``LIGHTRAG_OPS_RUNS``, enriched with live flags.

        In-flight documents of workspaces without a running run (work started
        before run tracking, or a run whose start was not recorded) are added
        from the interim source while they move or their pipeline is busy;
        such rows are labelled ``source: doc_status``.
        """
        thresholds = await self.thresholds()
        rows = await self._q(lambda: self.store.running_runs(RUNNING_LIMIT))
        candidates = await self._q(
            lambda: self.store.live_candidates(ACTIVE_CANDIDATES)
        )
        now = self.clock()
        stale_s = thresholds["stale_doc_minutes"] * 60
        hung_s = thresholds["hung_minutes"] * 60
        covered = {r["workspace"] for r in rows}
        extra = [c for c in candidates if c["workspace"] not in covered]
        wanted: list[str] = []
        for workspace in [r["workspace"] for r in rows] + [
            c["workspace"] for c in extra
        ]:
            if workspace not in wanted:
                wanted.append(workspace)
        names = {ws: self.namespace_of(ws) for ws in wanted[:ACTIVE_CANDIDATES]}
        batch = await self.reader.snapshot_many(list(names.values()), LIVE_KEYS)
        runs = []
        for row in rows:
            view = self._run_view(row, now)
            name = names.get(row["workspace"])
            snap = batch.data.get(name) if (name and batch.responsive) else None
            live, pause = self._live_view(snap, now)
            view["live"] = live
            view["pause"] = pause
            view["state"] = self._running_state(view, live, hung_s)
            view["latest_message"] = (
                redact_pipeline(snap.get("latest_message"), 300) or None
                if snap and live and live.get("run_id") == row["run_id"]
                else None
            )
            runs.append(view)
        approximate = False
        for c in extra:
            name = names.get(c["workspace"])
            snap = batch.data.get(name) if (name and batch.responsive) else None
            legacy = self._legacy_active_row(c, snap, batch.responsive, now, stale_s)
            if legacy["state"] in ("running", "paused"):
                runs.append(legacy)
                approximate = True
        order = {
            "running": 0,
            "cancelling": 1,
            "paused": 2,
            "hung": 3,
            "unresponsive": 4,
        }
        runs.sort(key=lambda r: (order.get(r["state"], 9), r.get("started_at") or ""))
        return {
            "runs": runs,
            "approximate": approximate,
            "source": "runs",
            "checked": len(rows) + len(candidates),
            "capped": len(rows) >= RUNNING_LIMIT,
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
        stuck = {"runs": 0, "docs": stale_total, "reservations": 0}
        if self.durable_runs:
            try:
                runs_stuck = await self.stuck_runs()
                stuck["runs"] = runs_stuck["counts"]["runs"]
                stuck["reservations"] = runs_stuck["counts"]["reservations"]
            except OpsDataUnavailable:
                stuck["runs"] = stuck["reservations"] = None
        running = [r for r in active["runs"] if r["state"] in ACTIVE_RUN_STATES]
        return {
            "backlog": self._backlog_view(backlog),
            "active": active["runs"][:5],
            "active_count": len(running),
            "stuck": stuck,
            "email_queue": self._queue_counts(queue),
            "pauses": {
                "count": pauses.get("total", 0),
                "by_slot_kind": pauses.get("by_slot_kind", {}),
                "capped": pauses.get("capped", False),
            },
            "thresholds": thresholds,
            "approximate": bool(active.get("approximate")),
            "durable": self.durable_runs,
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
    async def _reconcile_kv(self, key: str, now: float) -> dict[str, Any]:
        """A reconcile result from KV (``busy_without_run``/``reservations``)."""

        async def load():
            return await self.store.kv_get(key) or {}

        value = await self._cached((key,), key, load)
        if not isinstance(value, dict):
            value = {}
        at = value.get("at_epoch")
        items = [i for i in value.get("items") or [] if isinstance(i, dict)]
        return {
            "items": items,
            "at": value.get("at"),
            "stale": not isinstance(at, (int, float)) or now - at > RECONCILE_FRESH_S,
            "responsive": value.get("responsive", True),
            "checked": value.get("checked"),
            "capped": bool(value.get("capped")),
        }

    async def stuck_runs(self) -> dict[str, Any]:
        """Orphaned, hung, busy-without-run and reservation classes (durable)."""
        if not self.durable_runs:
            raise OpsDataUnavailable("run_tracking_off")
        return await self.cache.get(
            ("stuck_runs",), TTL["stuck_runs"], self._compute_stuck_runs
        )

    async def _compute_stuck_runs(self) -> dict[str, Any]:
        thresholds = await self.thresholds()
        now = self.clock()
        hung_s = thresholds["hung_minutes"] * 60
        serious_s = thresholds["hung_serious_minutes"] * 60
        running = await self._q(lambda: self.store.running_runs(RUNNING_LIMIT))
        orphaned_rows = await self._q(
            lambda: self.store.orphaned_runs(ORPHANED_HOURS, STUCK_LIMIT)
        )
        orphan_s = float(thresholds["orphan_seconds"])
        hung_rows = []
        unresponsive_rows = []
        for row in running:
            view = self._run_view(row, now)
            age = view.get("progress_age_s")
            worker_age = view.get("worker_heartbeat_age_s")
            if view["worker_alive"] is False:
                # The owner stopped beating and its death is not confirmed
                # (else reconcile orphans the run): a wedged worker.
                if isinstance(worker_age, (int, float)) and worker_age >= orphan_s:
                    unresponsive_rows.append(view)
            elif (
                isinstance(age, (int, float))
                and age >= hung_s
                and self._judged_by_progress(view)
            ):
                hung_rows.append(view)
        orphan_views = [self._run_view(r, now) for r in orphaned_rows]
        wanted = []
        for view in orphan_views + hung_rows + unresponsive_rows:
            if view["workspace"] not in wanted:
                wanted.append(view["workspace"])
        names = {ws: self.namespace_of(ws) for ws in wanted[:ACTIVE_CANDIDATES]}
        batch = await self.reader.snapshot_many(list(names.values()), LIVE_KEYS)
        orphaned = []
        for view in orphan_views:
            name = names.get(view["workspace"])
            snap = batch.data.get(name) if (name and batch.responsive) else None
            known = batch.responsive and name is not None
            stranded = (
                bool(snap and snap.get("busy") and snap.get("run_id") == view["run_id"])
                if known
                else None
            )
            # A run whose workspace no longer holds busy for it needs nothing.
            if stranded is False:
                continue
            view["stranded"] = stranded
            view["guidance"] = STUCK_GUIDANCE["orphaned"]
            # Last sign of life of the owner.
            view["age_s"] = view.get("heartbeat_age_s")
            view["since"] = view.get("heartbeat_at")
            orphaned.append(view)
        hung = []
        for view in hung_rows:
            name = names.get(view["workspace"])
            snap = batch.data.get(name) if (name and batch.responsive) else None
            live, pause = self._live_view(snap, now)
            if live and live["paused"]:
                continue  # waiting for the provider, not hung
            age = view["progress_age_s"] or 0.0
            view["live"] = live
            view["state"] = "hung"
            view["severity"] = "serious" if age >= serious_s else "warning"
            view["guidance"] = STUCK_GUIDANCE["hung"]
            view["age_s"] = age
            view["since"] = view.get("last_progress_at")
            hung.append(view)
        # Listed with the hung runs (the workspace is held and nothing
        # moves), marked so the console can say why and what to do.
        for view in unresponsive_rows:
            name = names.get(view["workspace"])
            snap = batch.data.get(name) if (name and batch.responsive) else None
            live, _ = self._live_view(snap, now)
            view["live"] = live
            view["state"] = "unresponsive"
            view["owner_unresponsive"] = True
            view["severity"] = "warning"
            view["guidance"] = STUCK_GUIDANCE["unresponsive"]
            view["age_s"] = view.get("worker_heartbeat_age_s")
            view["since"] = None
            hung.append(view)
        busy = await self._reconcile_kv(BUSY_WITHOUT_RUN_KEY, now)
        reservations = await self._reconcile_kv(RESERVATIONS_KEY, now)
        for item in busy["items"]:
            item["guidance"] = STUCK_GUIDANCE["busy_without_run"]
        for item in reservations["items"]:
            item["guidance"] = STUCK_GUIDANCE["reservations"]
            # Older reconcile results name the flag by kind and count only.
            item.setdefault("scanning", item.get("kind") == "scanning")
            item.setdefault(
                "pending_enqueues",
                item.get("count") if item.get("kind") == "pending_enqueues" else None,
            )
        for item in busy["items"] + reservations["items"]:
            if "reason_code" not in item and "reason" in item:
                item["reason_code"] = item.pop("reason")
        return {
            "orphaned": orphaned,
            "hung": hung,
            "busy_without_run": busy,
            "reservations": reservations,
            "counts": {
                "runs": len(orphaned) + len(hung) + len(busy["items"]),
                "orphaned": len(orphaned),
                "hung": len(hung) - len(unresponsive_rows),
                "unresponsive": len(unresponsive_rows),
                "busy_without_run": len(busy["items"]),
                "reservations": len(reservations["items"]),
            },
            "manager": "unresponsive" if not batch.responsive else self.reader.state,
        }

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
        out: dict[str, Any] = {
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
                "hung_serious_minutes": thresholds["hung_serious_minutes"],
                "orphan_seconds": thresholds["orphan_seconds"],
                "reservation_minutes": thresholds["reservation_minutes"],
                "busy_without_run_minutes": thresholds["busy_without_run_minutes"],
            },
            "guidance": dict(STUCK_GUIDANCE),
            # Without run tracking only stale documents can be checked.
            "checked_classes": ["stale_docs"],
            "durable": self.durable_runs,
        }
        if self.durable_runs:
            runs = await self.stuck_runs()
            out["orphaned"] = runs["orphaned"]
            out["hung"] = runs["hung"]
            out["busy_without_run"] = runs["busy_without_run"]["items"]
            out["reservations"] = runs["reservations"]["items"]
            out["reconciled"] = {
                "busy_without_run": {
                    k: runs["busy_without_run"][k]
                    for k in ("at", "stale", "responsive", "checked", "capped")
                },
                "reservations": {
                    k: runs["reservations"][k]
                    for k in ("at", "stale", "responsive", "checked", "capped")
                },
            }
            out["checked_classes"] = [
                "orphaned",
                "hung",
                "busy_without_run",
                "reservations",
                "stale_docs",
            ]
            out["manager"] = runs["manager"]
        return out

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
        last_runs: dict[str, Any] = {}
        if self.durable_runs and rows:
            # Joined for the returned page only.
            try:
                last_runs = await self._q(
                    lambda: self.store.last_runs([r["workspace"] for r in rows])
                )
            except OpsDataUnavailable:
                last_runs = {}
        now = self.clock()
        items = []
        for r in rows:
            ws = r["workspace"]
            queued = queue_items.get(ws) or {}
            last = last_runs.get(ws)
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
                    "last_run": self._run_brief(last, now) if last else None,
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
            # Without run tracking the paused filter only knows the capped
            # pause candidates.
            "approximate_filters": [] if self.durable_runs else ["paused"],
        }

    def _run_brief(self, row: dict[str, Any], now: float) -> dict[str, Any]:
        view = self._run_view(row, now)
        return {
            k: view[k]
            for k in (
                "run_id",
                "kind",
                "trigger",
                "status",
                "was_orphaned",
                "started_at",
                "finished_at",
                "duration_s",
                "docs_done",
                "docs_failed",
            )
        }

    # -- failures ---------------------------------------------------------------------------
    @staticmethod
    def _failure_group_view(group: str, g: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": g["key"],
            "kind": group,
            "source": "durable",
            "count": int(g["count"]),
            "docs": int(g.get("docs") or 0),
            "workspaces": int(g["workspaces"]),
            "first_seen": utc_iso(g.get("first_seen")),
            "last_seen": utc_iso(g.get("last_seen")),
            # Stored redacted and path-masked; redacted again on the way out.
            "example": redact(g.get("example"), 300) or None,
            "example_kind": g.get("example_kind"),
            "example_stage": g.get("example_stage"),
        }

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
                "source": "legacy",
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
                    "source": "legacy",
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
        out: dict[str, Any] = {
            "window": window,
            "group": group,
            "legacy_group": legacy_group,
            "groups": [],
            "durable": False,
            "legacy": legacy,
            "excluded": {"dup": dup, "cancelled": int(cancelled.get("count") or 0)},
        }
        if not self.durable_runs or group == "signature":
            # Failure kinds, stages and models are recorded from run tracking
            # on; older failures are grouped by message.
            return out
        durable = await self._cached(
            ("failure_groups", hours, group, include_cancelled),
            "failures",
            lambda: self.store.failure_groups(hours, group, include_cancelled),
        )
        groups = [self._failure_group_view(group, g) for g in durable["groups"]]
        durable_cancelled = durable["cancelled"]
        if include_cancelled and int(durable_cancelled.get("count") or 0):
            groups.append(
                {
                    **self._failure_group_view(
                        "error_kind",
                        {
                            **durable_cancelled,
                            "key": CANCELLED_GROUP_KEY,
                            "example_kind": CANCELLED_GROUP_KEY,
                            "example_stage": None,
                        },
                    ),
                    "cancelled": True,
                }
            )
        out["groups"] = groups
        out["durable"] = True
        out["excluded"]["cancelled_failures"] = int(durable_cancelled.get("count") or 0)
        return out

    async def failure_docs(
        self,
        key: str,
        group: str,
        window: str,
        page: Page,
        source: Optional[str] = None,
    ) -> dict[str, Any]:
        """Documents of one failure group.

        ``source`` picks the group's origin when the name exists in both
        (``workspace``): ``legacy`` (message-grouped rows) or ``durable``
        (classified failures, the default with run tracking on).
        """
        self._require_docs()
        if window not in FAILURE_WINDOWS:
            raise OpsBadRequest("invalid_window")
        if source not in (None, "", "legacy", "durable"):
            raise OpsBadRequest("invalid_source")
        durable_groups = DURABLE_FAILURE_GROUPS + ("workspace",)
        durable = self.durable_runs and group in durable_groups and source != "legacy"
        if not durable and group not in LEGACY_FAILURE_GROUPS + (CANCELLED_GROUP_KEY,):
            raise OpsBadRequest("invalid_group")
        if key is None or len(key) > 255 or (not key and group != "model"):
            raise OpsBadRequest("invalid_key")
        hours = FAILURE_WINDOWS[window]
        if durable:
            return await self._durable_failure_docs(hours, group, key, page)
        if not key:
            raise OpsBadRequest("invalid_key")
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
            "source": "legacy",
        }

    async def _durable_failure_docs(
        self, hours: int, group: str, key: str, page: Page
    ) -> dict[str, Any]:
        rows, total = await self._cached(
            ("failure_group_docs", hours, group, key, page.limit, page.offset),
            "failure_docs",
            lambda: self.store.failure_group_docs(hours, group, key, page),
        )
        return {
            "items": [
                {
                    "workspace": r["workspace"],
                    "doc_id": r["doc_id"],
                    "track_id": r.get("track_id"),
                    "run_id": r.get("run_id"),
                    "updated_at": utc_iso(r.get("ts")),
                    "failed_at": utc_iso(r.get("ts")),
                    "error": redact(r.get("message"), 300) or None,
                    "error_kind": r.get("error_kind"),
                    "stage": r.get("stage"),
                    "http_status": r.get("http_status"),
                    "model": r.get("model"),
                    "attempt": r.get("attempt"),
                    "current_status": r.get("current_status"),
                    **_file_fields(r["doc_id"], r.get("file_path")),
                }
                for r in rows
            ],
            "page": page.offset // page.limit + 1,
            "page_size": page.limit,
            "total": total,
            "source": "durable",
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
        drainers = await self._drainers()
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
                    "backoff_until": _epoch_iso(state.get("backoff_until")),
                    "drains_1h": _drains_view(state, now),
                    "drainer": drainers.get(i["workspace"]),
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

    async def _drainers(self) -> dict[str, dict[str, Any]]:
        """Live drainers by workspace, from the workers' heartbeats."""
        try:
            items = await self._worker_items()
        except OpsDataUnavailable:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for item in items or []:
            if not item.get("alive"):
                continue
            for d in (item.get("snapshot") or {}).get("drainers") or []:
                if isinstance(d, dict) and isinstance(d.get("workspace"), str):
                    out[d["workspace"]] = {
                        "worker_id": item.get("worker_id"),
                        "pid": item.get("pid"),
                        "state": d.get("state"),
                        "backoff_until": d.get("backoff_until"),
                    }
        return out

    # -- pauses ----------------------------------------------------------------------------------
    def _pause_item(self, row: dict[str, Any], now: float) -> Optional[dict[str, Any]]:
        workspace = row.get("workspace")
        if not isinstance(workspace, str):
            return None
        retry_at = row.get("retry_at")
        return {
            "workspace": workspace,
            "slot": row.get("slot") or "unknown",
            "role": row.get("role"),
            "kind": row.get("kind") or "unknown",
            "status_code": row.get("status_code"),
            "byo": bool(row.get("byo")),
            "byo_known": row.get("byo") is not None,
            "since": _epoch_iso(row.get("since")),
            "retry_at": _epoch_iso(retry_at),
            "retry_due": isinstance(retry_at, (int, float)) and retry_at <= now,
            "message": redact(row.get("message"), 300) or None,
        }

    async def _durable_pauses(self) -> dict[str, Any]:
        """Pauses persisted at the pause write sites (plain SQL over KV).

        Counts are SQL aggregates and never capped; the item list (which the
        backlog also uses to mark paused workspaces) is capped.
        """
        counts = await self.store.pause_counts()
        rows = await self.store.open_pauses(OPEN_PAUSES_LIMIT)
        now = self.clock()
        items = [item for item in (self._pause_item(row, now) for row in rows) if item]
        items.sort(key=lambda i: (i["since"] or "", i["workspace"]))
        return {
            "items": items,
            "by_slot_kind": counts["by_slot_kind"],
            "total": counts["total"],
            "workspaces": counts["workspaces"],
            "checked": counts["total"],
            # Only the item list is capped (counts are exact).
            "capped": len(rows) >= OPEN_PAUSES_LIMIT,
            "responsive": True,
            "source": "events",
        }

    async def pauses(self, page: Page) -> dict[str, Any]:
        data = await self.pause_data()
        items = data.get("items", [])
        if data.get("source") == "events":
            # Paged in SQL: the cached list behind the counts is capped.
            rows = await self._cached(
                ("pauses_page", page.limit, page.offset),
                "pauses_durable",
                lambda: self.store.pauses_page(page.limit, page.offset),
            )
            now = self.clock()
            page_items = [i for i in (self._pause_item(r, now) for r in rows) if i]
        else:
            page_items = items[page.offset : page.offset + page.limit]
        return {
            "by_slot_kind": data.get("by_slot_kind", {}),
            "items": page_items,
            "total": data.get("total", 0),
            "checked": data.get("checked", 0),
            "capped": data.get("capped", False),
            "approximate": data.get("source") != "events",
            "source": data.get("source", "shared_state"),
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
                "heartbeat_at": snap.get("heartbeat_at"),
                "scan_owner_worker_id": snap.get("scan_owner_worker_id"),
                "scan_started_at": snap.get("scan_started_at"),
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
                "backoff_until": _epoch_iso(state.get("backoff_until")),
                "reason": redact(state.get("reason"), 200) or None,
                "scanned_at": _epoch_iso(queue.get("scanned_at")),
            }
            action_value = kv.get(f"action:{workspace}")
            if isinstance(action_value, dict) and (action_value.get("exp") or 0) > now:
                action = {
                    "actor": action_value.get("actor"),
                    "action": action_value.get("action"),
                    "since": _epoch_iso(action_value.get("since")),
                }
        recent_runs: list[dict[str, Any]] = []
        failures_by_kind: list[dict[str, Any]] = []
        if self.durable_runs:
            runs = await self._q(
                lambda: self.store.runs_page(
                    workspace=workspace,
                    kind=None,
                    status=None,
                    before=None,
                    limit=RECENT_RUNS,
                )
            )
            recent_runs = [self._run_view(r, now) for r in runs]
            kinds = await self._q(
                lambda: self.store.failures_by_kind(workspace, WORKSPACE_FAILURE_HOURS)
            )
            failures_by_kind = [
                {
                    "key": k["key"],
                    "count": int(k["count"]),
                    "docs": int(k.get("docs") or 0),
                    "last_seen": utc_iso(k.get("last_seen")),
                }
                for k in kinds
            ]
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
            and not recent_runs
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
            "recent_runs": recent_runs,
            "failures_by_kind": failures_by_kind,
            "failures_hours": WORKSPACE_FAILURE_HOURS,
            "runs_available": self.durable_runs,
            "errors_1h": await self._errors_1h(workspace),
            "available": self.docs_available,
        }

    async def _errors_1h(self, workspace: str) -> Optional[int]:
        """ERROR+ log lines of the workspace in the last hour (capped)."""
        from lightrag.api.ops.logs import drawer_count

        return await drawer_count(
            self, hours=1, workspace=workspace, level=logging.ERROR
        )

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
        from lightrag.api.ops.logs import drawer_logs

        row = await self._q(lambda: self.store.document(workspace, doc_id))
        if row is None:
            raise OpsNotFound(doc_id)
        created = as_utc(row.get("created_at"))
        updated = as_utc(row.get("updated_at"))
        calls, calls_available = await self._document_calls(workspace, doc_id, created)
        failures: list[dict[str, Any]] = []
        if self.durable_runs:
            rows = await self._q(
                lambda: self.store.doc_failures(workspace, doc_id, DOC_FAILURES_LIMIT)
            )
            failures = [_failure_row_view(r) for r in rows]
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
                "failure": failure_meta_view(row.get("metadata")),
                **_file_fields(row["id"], row.get("file_path")),
            },
            "failures": failures,
            "failures_available": self.durable_runs,
            "calls": calls,
            "calls_available": calls_available,
            "logs": await drawer_logs(self, workspace=workspace, doc_id=doc_id),
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
        wanted = [
            k for k in (kinds or [*EVENT_KINDS, AUDIT_EVENT_KIND, RUN_EVENT_KIND])
        ]
        unknown = set(wanted) - set(EVENT_KINDS) - {AUDIT_EVENT_KIND, RUN_EVENT_KIND}
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
        if RUN_EVENT_KIND in wanted and self.durable_runs:
            for row in await self.store.run_events(start, end, EVENTS_LIMIT):
                items.append(
                    {
                        "ts": utc_iso(row["ts"]),
                        "kind": RUN_EVENT_KIND,
                        "workspace": row.get("workspace"),
                        "purpose": None,
                        "detail": {
                            "run_id": row.get("run_id"),
                            "run_kind": row.get("kind"),
                            "status": row.get("status"),
                            "recovered": bool(row.get("was_orphaned"))
                            and row.get("status") != "orphaned",
                        },
                        "source": "runs",
                    }
                )
        items.sort(key=lambda e: e["ts"] or "", reverse=True)
        return {
            "items": items[:EVENTS_LIMIT],
            "from": start.isoformat(),
            "to": end.isoformat(),
            "kinds": wanted,
        }

    # -- run history and the run drawer -------------------------------------------------------
    @staticmethod
    def parse_run_cursor(value: Optional[str]) -> Optional[tuple[datetime, str]]:
        """``<epoch microseconds>:<run_id>`` -> keyset tuple."""
        if not value:
            return None
        match = _CURSOR.match(value)
        if not match:
            raise OpsBadRequest("invalid_cursor")
        moment = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            microseconds=int(match.group("us"))
        )
        return moment, match.group("run")

    @staticmethod
    def run_cursor(row: dict[str, Any]) -> Optional[str]:
        started = as_utc(row.get("started_at"))
        if started is None:
            return None
        delta = started - datetime(1970, 1, 1, tzinfo=timezone.utc)
        micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
        return f"{micros}:{row['run_id']}"

    def _require_runs(self) -> None:
        if not self.available:
            raise OpsDataUnavailable(self.unavailable_reason or "unavailable")

    async def runs_page(
        self,
        *,
        workspace: Optional[str],
        kind: Optional[str],
        status: Optional[str],
        before: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """Run history (30 days), newest first, keyset paging."""
        self._require_runs()
        from lightrag.api.ops.store import (
            RUNS_PAGE_FILTER_KINDS,
            RUNS_PAGE_FILTER_STATUSES,
        )

        if workspace:
            validate_workspace(workspace)
        if kind and kind not in RUNS_PAGE_FILTER_KINDS:
            raise OpsBadRequest("invalid_kind")
        if status and status not in RUNS_PAGE_FILTER_STATUSES:
            raise OpsBadRequest("invalid_status")
        limit = max(1, min(int(limit), RUNS_PAGE_MAX))
        cursor = self.parse_run_cursor(before)
        rows = await self._cached(
            ("runs", workspace, kind, status, before, limit),
            "runs",
            lambda: self.store.runs_page(
                workspace=workspace or None,
                kind=kind or None,
                status=status or None,
                before=cursor,
                limit=limit + 1,
            ),
        )
        now = self.clock()
        page = rows[:limit]
        return {
            "items": [self._run_view(r, now) for r in page],
            "next_before": self.run_cursor(page[-1]) if len(rows) > limit else None,
            "limit": limit,
            "tracking": self.run_tracking,
        }

    async def run_detail(self, run_id: str) -> dict[str, Any]:
        """Run drawer: the run, its failed documents and its live flags."""
        self._require_runs()
        if not _RUN_ID.match(run_id or ""):
            raise OpsBadRequest("invalid_run_id")
        try:
            normalized = UUID(run_id).hex
        except (ValueError, AttributeError, TypeError) as e:
            raise OpsBadRequest("invalid_run_id") from e
        return await self.cache.get(
            ("run", normalized), TTL["run"], lambda: self._run_detail(normalized)
        )

    async def _run_detail(self, run_id: str) -> dict[str, Any]:
        from lightrag.api.ops.logs import drawer_count

        row = await self._q(lambda: self.store.run_get(run_id))
        if row is None:
            raise OpsNotFound(run_id)
        now = self.clock()
        thresholds = await self.thresholds()
        view = self._run_view(row, now)
        failures = await self._q(
            lambda: self.store.run_failures(run_id, RUN_FAILURES_LIMIT)
        )
        live = pause = None
        owns_busy = None
        if row["status"] in ("running", "orphaned"):
            snap, responsive = await self.reader.snapshot(
                self.namespace_of(row["workspace"]), LIVE_KEYS
            )
            live, pause = self._live_view(snap, now)
            if responsive:
                owns_busy = bool(
                    snap and snap.get("busy") and snap.get("run_id") == run_id
                )
        if row["status"] == "running":
            view["state"] = self._running_state(
                view, live, thresholds["hung_minutes"] * 60
            )
        view["live"] = live
        view["pause"] = pause
        view["owns_busy"] = owns_busy
        return {
            "run": view,
            "failures": [
                {
                    **_failure_row_view(f),
                    "workspace": f.get("workspace"),
                    "doc_id": f.get("doc_id"),
                }
                for f in failures
            ],
            "failures_capped": len(failures) >= RUN_FAILURES_LIMIT,
            "log_count": await drawer_count(self, run_id=run_id, level=logging.WARNING),
        }

    # -- health and workers ------------------------------------------------------------------
    async def _worker_items(self) -> Optional[list[dict[str, Any]]]:
        """Worker rows: host-wide from heartbeats, else this process only."""
        if self.available:
            rows = await self._cached(
                ("workers",),
                "workers",
                lambda: self.store.workers(WORKER_ROWS_LIMIT),
            )
            return [worker_item(r, self.worker_id) for r in rows]
        heartbeat = self.heartbeat
        if heartbeat is not None and heartbeat.last_beat_at is not None:
            return [worker_item(heartbeat.local_row(), self.worker_id)]
        return None

    async def live_workers(self) -> Optional[int]:
        """Workers with a fresh heartbeat (at least this one), when stored."""
        if not self.available:
            return None
        try:
            items = await self._worker_items()
        except OpsDataUnavailable:
            return None
        return max(1, sum(1 for item in items or [] if item["alive"]))

    async def _probe_records(self) -> tuple[dict[str, Any], Optional[dict], bool]:
        """Latest probe records and cycle; ``partial`` when only this
        process's own results could be read."""
        if self.probes is None:
            return {}, None, False
        if not self.available:
            records, cycle = await self.probes.read_latest()
            return records, cycle, False
        try:
            records, cycle = await self._cached(
                ("probe_records",), "probes", self.probes.read_latest
            )
            return records, cycle, False
        except OpsDataUnavailable:
            return dict(self.probes.latest), self.probes.cycle, True

    async def _strips(self, now: float) -> dict[str, list[dict[str, Any]]]:
        if self.probes is None or not self.available:
            return {}
        try:
            rows = await self._cached(
                ("probe_strips",),
                "strips",
                lambda: self.store.probe_strips(STRIP_HOURS, STRIP_SLOT_S),
            )
        except OpsDataUnavailable:
            return {}
        return build_strips(rows, now)

    async def _dep_cards(
        self,
        *,
        mode: Optional[str],
        worker_items: Optional[list[dict[str, Any]]],
        strips: bool,
        now: float,
    ) -> tuple[list[dict[str, Any]], Optional[dict], bool]:
        records, cycle, partial = await self._probe_records()
        strip_map = await self._strips(now) if strips else {}
        cards: list[dict[str, Any]] = []
        if self.probes is not None:
            for spec in self.probes.specs.values():
                card = dep_card(spec, records.get(spec.id), strip_map.get(spec.id), now)
                if not strips:
                    card["strip"] = []
                cards.append(card)
        cards.extend(derived_deps(worker_items or [], mode or "uvicorn", now))
        return cards, cycle, partial

    def _cycle_view(self, cycle: Optional[dict], now: float) -> Optional[dict]:
        if not isinstance(cycle, dict):
            return None
        at = cycle.get("at_epoch")
        return {
            "at": cycle.get("at"),
            "duration_ms": cycle.get("duration_ms"),
            "holder": cycle.get("holder"),
            "holder_pid": cycle.get("holder_pid"),
            "holder_self": bool(self.worker_id)
            and cycle.get("holder") == self.worker_id,
            "interval_s": cycle.get("interval_s"),
            "age_s": round(now - at, 1) if isinstance(at, (int, float)) else None,
        }

    async def health(self, mode: Optional[str]) -> dict[str, Any]:
        """Dependency grid: latest result, 48-slot strip and detail per dependency."""
        now = self.clock()
        try:
            worker_items = await self._worker_items()
        except OpsDataUnavailable:
            worker_items = None
        cards, cycle, partial = await self._dep_cards(
            mode=mode, worker_items=worker_items, strips=True, now=now
        )
        return {
            "enabled": self.probes is not None,
            "durable": self.available,
            "partial": partial,
            "cycle": self._cycle_view(cycle, now),
            "deps": cards,
            "summary": deps_summary(cards),
        }

    def known_dep(self, dep: str) -> Optional[ProbeSpec]:
        if not _DEP_ID.match(dep or ""):
            raise OpsBadRequest("invalid_dependency")
        if self.probes is None:
            return None
        return self.probes.specs.get(dep)

    async def dependency(
        self, dep: str, hours: int, mode: Optional[str]
    ) -> dict[str, Any]:
        """Dependency drawer: latency series, recent results, error classes."""
        now = self.clock()
        spec = self.known_dep(dep)
        empty: dict[str, Any] = {
            "id": dep,
            "hours": hours,
            "dep": None,
            "latency_series": [],
            "recent": [],
            "detail": {},
            "blocking": [],
            "error_classes": [],
            "enabled": self.probes is not None,
        }
        if spec is None and dep in DERIVED_DEPS:
            try:
                worker_items = await self._worker_items()
            except OpsDataUnavailable:
                worker_items = None
            for card in derived_deps(worker_items or [], mode or "uvicorn", now):
                if card["id"] == dep:
                    return {**empty, "dep": card, "detail": card["detail"]}
            raise OpsNotFound(dep)
        if spec is None:
            if self.probes is None:
                return empty
            raise OpsNotFound(dep)
        records, _, _ = await self._probe_records()
        record = records.get(dep)
        strip = (await self._strips(now)).get(dep)
        card = dep_card(spec, record, strip, now)
        detail = card["detail"]
        out = {
            **empty,
            "dep": card,
            "detail": detail,
            "blocking": list(detail.get("blocking") or []) if dep == "postgres" else [],
        }
        if not self.available:
            if isinstance(record, dict) and record.get("checked_at"):
                out["recent"] = [_probe_row_view(record, record.get("checked_at"))]
            return out
        bucket_s = max(300, (hours * 3600 // 144) // 60 * 60)

        async def load() -> tuple[list, list, list]:
            series = await self.store.probe_series(dep, hours, bucket_s)
            recent = await self.store.probe_recent(dep, RECENT_PROBES)
            classes = await self.store.probe_error_classes(dep, hours)
            return series, recent, classes

        series, recent, classes = await self._cached(
            ("dependency", dep, hours), "dependency", load
        )
        out["latency_series"] = [
            {
                "t": int(row["t"]) * 1000,
                "latency_ms": None
                if row.get("avg_ms") is None
                else round(float(row["avg_ms"]), 1),
                "max_ms": row.get("max_ms"),
                "status": strip_cell_status(
                    int(row.get("total") or 0)
                    - int(row.get("down") or 0)
                    - int(row.get("warn") or 0),
                    int(row.get("warn") or 0),
                    int(row.get("down") or 0),
                    int(row.get("total") or 0),
                ),
                "total": row.get("total"),
                "failed": row.get("down"),
            }
            for row in series
        ]
        out["recent"] = [_probe_row_view(row, utc_iso(row.get("ts"))) for row in recent]
        out["error_classes"] = [
            {"key": row["key"], "count": int(row["count"])} for row in classes
        ]
        return out

    async def workers_view(self, process: dict[str, Any]) -> dict[str, Any]:
        """Workers table: every process that reported in the last 24 h."""
        items = await self._worker_items()
        ordered = order_workers(items or [])
        alive = sum(1 for item in ordered if item["alive"])
        return {
            "items": ordered,
            "host": host_info(ordered),
            "mode": process.get("mode"),
            "expected": process.get("expected"),
            "live": max(alive, 1) if items is not None else process.get("live"),
            "alive_within_s": ALIVE_WITHIN_S,
            "heartbeats": self.available,
        }

    async def worker_history(self, worker_id: str, hours: int) -> dict[str, Any]:
        """Per-minute samples of one worker (hourly-ish buckets beyond a day)."""
        try:
            normalized = UUID(worker_id).hex
        except (ValueError, AttributeError, TypeError) as e:
            raise OpsBadRequest("invalid_worker_id") from e
        bucket_s = 60 if hours <= 24 else 600
        rows = await self._cached(
            ("worker_history", normalized, hours),
            "worker_history",
            lambda: self.store.worker_history(normalized, hours, bucket_s),
        )
        return {
            "worker_id": normalized,
            "hours": hours,
            "bucket_s": bucket_s,
            "items": [
                {
                    "t": int(row["t"]) * 1000,
                    "rss": row.get("rss"),
                    "cpu": None
                    if row.get("cpu") is None
                    else round(float(row["cpu"]), 1),
                    "loop_lag": row.get("loop_lag"),
                    "pool_wait": row.get("pool_wait"),
                }
                for row in rows
            ],
        }

    # -- attention snapshot (leased, see reconcile.py) ---------------------------------------
    async def _attention_snapshot(self, now: float) -> Optional[dict[str, Any]]:
        """The latest leased attention snapshot, when it is fresh."""
        if not self.available:
            return None

        async def load():
            return await self.store.kv_get(ATTENTION_KEY)

        try:
            snapshot = await self._cached(("attention",), "attention", load)
        except OpsDataUnavailable:
            return None
        if not isinstance(snapshot, dict) or not isinstance(
            snapshot.get("items"), list
        ):
            return None
        at = snapshot.get("at_epoch")
        if not isinstance(at, (int, float)) or now - at > ATTENTION_FRESH_S:
            return None
        return snapshot

    async def compute_attention_snapshot(
        self, workers: dict[str, Any], holder: str
    ) -> dict[str, Any]:
        """Evaluate every rule once for the host (the leased reconcile cycle)."""
        now = self.clock()
        inputs = await self._gather(workers, now)
        try:
            previous = await self._q(lambda: self.store.kv_get(ATTENTION_KEY))
        except OpsDataUnavailable:
            previous = None
        streaks = previous.get("streaks") if isinstance(previous, dict) else None
        items, new_streaks = self._attention(
            inputs, now, streaks=streaks if isinstance(streaks, dict) else {}
        )
        worst = max(
            (a["severity"] for a in items), key=_SEVERITY_RANK.get, default=None
        )
        return {
            "at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "at_epoch": now,
            "holder": holder,
            "items": items,
            "worst": worst,
            "streaks": new_streaks,
            "failed_sections": inputs["failed_sections"],
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

    async def _gather(self, workers: dict[str, Any], now: float) -> dict[str, Any]:
        """Everything the KPI tiles and the attention rules read."""
        thresholds = await self.thresholds()
        failed_sections: list[str] = []
        backlog = active = queue = pauses = stuck_runs = None
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
            if self.durable_runs:
                try:
                    stuck_runs = await self.stuck_runs()
                except OpsDataUnavailable:
                    failed_sections.append("stuck_runs")
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
        try:
            worker_items = await self._worker_items()
        except OpsDataUnavailable:
            worker_items = None
            failed_sections.append("workers")
        process = dict(workers)
        if worker_items is not None:
            # This process is answering, so at least one worker is alive.
            process["live"] = max(1, sum(1 for w in worker_items if w["alive"]))
        deps, cycle, _ = await self._dep_cards(
            mode=process.get("mode"), worker_items=worker_items, strips=False, now=now
        )
        llm = await self.llm_health()
        postgres_ok = self.available and not failed_sections
        # A slow query (a statement timeout on a heavy count) is a degraded
        # section; only a trivial probe that fails too means the database is
        # down.
        postgres_down = (
            self.available and bool(failed_sections) and not await self._probe()
        )
        deploy_recent = any(
            e.get("kind") == "deployment"
            and (_epoch_of(e.get("ts")) or 0) >= now - DEPLOY_SUPPRESS_S
            for e in events
        )
        return {
            "thresholds": thresholds,
            "failed_sections": failed_sections,
            "backlog": backlog,
            "active": active,
            "stale_items": stale_items,
            "stale_total": stale_total,
            "stuck_runs": stuck_runs,
            "queue": queue,
            "pauses": pauses,
            "events": events,
            "llm": llm,
            "postgres_ok": postgres_ok,
            "postgres_down": postgres_down,
            "process": process,
            "worker_items": worker_items,
            "deps": deps,
            "cycle": cycle,
            "deploy_recent": deploy_recent,
        }

    async def _overview(self, workers: dict[str, Any]) -> dict[str, Any]:
        now = self.clock()
        now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()
        inputs = await self._gather(workers, now)
        failed_sections = inputs["failed_sections"]
        backlog = inputs["backlog"]
        active = inputs["active"]
        stale_total = inputs["stale_total"]
        queue = inputs["queue"]
        llm = inputs["llm"]
        process = inputs["process"]
        postgres_down = inputs["postgres_down"]
        snapshot = None if postgres_down else await self._attention_snapshot(now)
        if snapshot is not None:
            attention = list(snapshot["items"])
            attention_source, attention_at = "snapshot", snapshot.get("at")
        else:
            attention, _ = self._attention(inputs, now, streaks=None)
            attention_source, attention_at = "request", now_iso
        worst = max(
            (a["severity"] for a in attention), key=_SEVERITY_RANK.get, default=None
        )
        severity_counts = {
            s: sum(1 for a in attention if a["severity"] == s)
            for s in ("critical", "serious", "warning")
        }
        if not self.available:
            state = "unknown"
        elif postgres_down or process.get("live") == 0:
            state = "down"
        elif worst:
            state = "degraded"
        else:
            state = "healthy"
        active_runs = (active or {}).get("runs", [])
        running = [r for r in active_runs if r["state"] in ACTIVE_RUN_STATES]
        stuck_runs = inputs.get("stuck_runs")
        durable = self.durable_runs and active is not None
        longest = max(
            (
                now - datetime.fromisoformat(r["started_at"]).timestamp()
                for r in running
                if r.get("started_at")
            ),
            default=None,
        )
        stuck_known = "stuck" not in failed_sections and self.docs_available
        runs_known = stuck_known and (not self.durable_runs or stuck_runs is not None)
        stuck_counts = (stuck_runs or {}).get("counts") or {}
        stuck_run_count = int(stuck_counts.get("runs") or 0)
        reservation_count = int(stuck_counts.get("reservations") or 0)
        # No summary yet (the first scan runs a minute after startup): unknown.
        queue_counts = self._queue_counts(queue) if queue else None
        deps = inputs["deps"]
        summary = deps_summary(deps)
        deps_known = self.probes is not None and summary["total"] > 0
        worker_items = inputs["worker_items"]
        kpis = {
            "active": {
                "value": len(running) if active is not None else None,
                "longest_s": longest,
                "approximate": bool((active or {}).get("approximate", True)),
                "durable": durable,
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
                "value": stale_total + stuck_run_count + reservation_count
                if runs_known
                else None,
                "runs": stuck_run_count if runs_known else None,
                "docs": stale_total if stuck_known else None,
                "reservations": reservation_count if runs_known else None,
                # Without run tracking only stale documents are checked.
                "classes": "all" if self.durable_runs else "docs_only",
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
            "deps": {
                "available": deps_known,
                "ok": summary["ok"] if deps_known else None,
                "total": summary["total"] if deps_known else None,
                "down": summary["down"] if deps_known else None,
                "scope": "host",
            },
            "workers": {
                "live": process.get("live"),
                "expected": process.get("expected"),
                "mode": process.get("mode"),
                "heartbeats": worker_items is not None,
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
            "attention_source": attention_source,
            "attention_at": attention_at,
            "checked": {
                "workspaces_with_backlog": None,
                "active_candidates": (active or {}).get("checked"),
                "email_queue": queue is not None,
                "at": now_iso,
            },
            "workers": order_workers(worker_items or [])[:OVERVIEW_WORKERS],
            "timeline": {
                "buckets": [],
                "events": inputs["events"][:100],
                "hours": OVERVIEW_TIMELINE_HOURS,
            },
            "active_top": active_runs[:5],
            "deps": deps,
            "deps_cycle": self._cycle_view(inputs["cycle"], now),
            "alert_channel": None,
            "approximate": bool((active or {}).get("approximate", True)),
            "durable": self.durable_runs,
            "postgres_ok": inputs["postgres_ok"],
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
        inputs: dict[str, Any],
        now: float,
        *,
        streaks: Optional[dict[str, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Attention items (worst first) and the updated rule streaks."""
        thresholds = inputs["thresholds"]
        backlog = inputs["backlog"]
        stale_items = inputs["stale_items"]
        stale_total = inputs["stale_total"]
        queue = inputs["queue"]
        pauses = inputs["pauses"]
        llm = inputs["llm"]
        process = inputs["process"]
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(
            rule,
            severity,
            title,
            link,
            *,
            params=None,
            since=None,
            count=None,
            item_id=None,
        ):
            key = item_id or rule
            if key in seen:
                return
            seen.add(key)
            items.append(
                {
                    "id": key,
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

        if inputs["postgres_down"]:
            add(
                "pg_probe",
                "critical",
                "Postgres is not answering ops queries",
                _hash_link(otab="health", odep="postgres"),
            )
        if self.reader.state == "unresponsive" and process.get("mode") == "gunicorn":
            add(
                "manager_rtt",
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
        self._run_attention(add, inputs, thresholds, process)
        new_streaks = health_attention(
            add,
            deps=inputs["deps"],
            workers=inputs["worker_items"],
            process=process,
            now=now,
            streaks=streaks,
            deploy_recent=inputs["deploy_recent"],
            uptime_s=now - self.started_at,
        )
        items.sort(key=lambda a: -_SEVERITY_RANK[a["severity"]])
        return items, new_streaks

    @staticmethod
    def _run_attention(
        add: Callable[..., None],
        inputs: dict[str, Any],
        thresholds: dict[str, Any],
        process: dict[str, Any],
    ) -> None:
        """Rules over the stuck run classes (run tracking only)."""
        stuck = inputs.get("stuck_runs")
        if not stuck:
            return
        gunicorn = process.get("mode") == "gunicorn"
        stuck_link = _hash_link(otab="processing", osub="stuck")
        orphaned = stuck.get("orphaned") or []
        # A worker that died with the pipeline busy strands the workspace; a
        # deploy restarts every worker, so the rule waits out the deploy.
        if orphaned and gunicorn and not inputs.get("deploy_recent"):
            first = orphaned[0]
            add(
                "run_orphaned",
                "critical",
                f"{len(orphaned)} run(s) lost their worker; the workspace stays busy",
                _hash_link(otab="processing", osub="stuck", orun=first["run_id"]),
                params={
                    "count": len(orphaned),
                    "workspaces": sorted({r["workspace"] for r in orphaned})[:10],
                },
                count=len(orphaned),
                since=first.get("heartbeat_at"),
            )
        all_hung = stuck.get("hung") or []
        hung = [r for r in all_hung if not r.get("owner_unresponsive")]
        wedged = [r for r in all_hung if r.get("owner_unresponsive")]
        if wedged:
            first = max(wedged, key=lambda r: r.get("age_s") or 0)
            add(
                "worker_unresponsive",
                "warning",
                f"{len(wedged)} run(s) held by a worker that stopped "
                "sending heartbeats",
                _hash_link(otab="processing", osub="stuck", orun=first["run_id"]),
                params={
                    "count": len(wedged),
                    "workspaces": sorted({r["workspace"] for r in wedged})[:10],
                    "worst_s": round(first.get("age_s") or 0),
                },
                count=len(wedged),
            )
        if hung:
            worst = max(hung, key=lambda r: r.get("progress_age_s") or 0)
            minutes = (worst.get("progress_age_s") or 0) / 60
            severity = (
                "serious"
                if minutes >= thresholds["hung_serious_minutes"]
                else "warning"
            )
            add(
                "run_hung",
                severity,
                f"{len(hung)} run(s) made no progress for over "
                f"{thresholds['hung_minutes']} min",
                _hash_link(otab="processing", osub="stuck", orun=worst["run_id"]),
                params={
                    "count": len(hung),
                    "threshold_min": thresholds["hung_minutes"],
                    "worst_min": round(minutes),
                },
                count=len(hung),
                since=worst.get("last_progress_at"),
            )
        busy = (stuck.get("busy_without_run") or {}).get("items") or []
        if busy:
            add(
                "busy_without_run",
                "warning",
                f"{len(busy)} workspace(s) marked busy with no run",
                stuck_link,
                params={
                    "count": len(busy),
                    "workspaces": sorted({i["workspace"] for i in busy})[:10],
                },
                count=len(busy),
            )
        reservations = (stuck.get("reservations") or {}).get("items") or []
        if reservations and gunicorn:
            add(
                "reservation_without_owner",
                "serious",
                f"{len(reservations)} scan or upload reservation(s) without an owner",
                stuck_link,
                params={
                    "count": len(reservations),
                    "threshold_min": thresholds["reservation_minutes"],
                    "kinds": sorted({i.get("kind") for i in reservations}),
                },
                count=len(reservations),
            )


_TILE_RULES = {
    "stuck": (
        "stale_docs",
        "run_orphaned",
        "run_hung",
        "worker_unresponsive",
        "busy_without_run",
        "reservation_without_owner",
    ),
    "backlog": ("oldest_pending",),
    "email_queue": ("email_queue_bad", "email_queue_stalled"),
    "llm": ("llm_error_rate_15m",),
    "active": ("provider_paused",),
    "deps": (
        "pg_probe",
        "pg_connections",
        "pg_blocked",
        "dependency_down",
        "vllm_waiting",
        "smtp_probe",
        "tls_cert_expiry",
        "disk_free",
    ),
    "workers": (
        "workers_below_expected",
        "manager_rtt",
        "pg_pool_wait",
        "log_sink_or_recorder_drops_5m",
    ),
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
        severities = [
            a["severity"] for a in attention if a["rule"] in _TILE_RULES[name]
        ]
        if live < expected:
            severities.append("warning")
        return max(severities, key=_SEVERITY_RANK.get, default="ok")
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
