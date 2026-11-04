"""Postgres storage and read queries for the Operations console.

``PostgresOpsStore`` creates its own tables idempotently on the shared
``ClientManager`` pool, and only when the server runs on Postgres KV storage
(``kv_storage == "PGKVStorage"``): ``LIGHTRAG_OPS_KV`` (small JSON values,
renewable leases), ``LIGHTRAG_OPS_AUDIT``, the worker heartbeat tables
(``LIGHTRAG_OPS_WORKERS``, ``LIGHTRAG_OPS_WORKER_SAMPLES``), the dependency
probe history (``LIGHTRAG_OPS_PROBES``), run tracking
(``LIGHTRAG_OPS_RUNS``, ``LIGHTRAG_OPS_DOC_FAILURES``), the structured
WARNING+ log store (``LIGHTRAG_OPS_LOGS``) and the latency histograms
(``LIGHTRAG_OPS_METRICS_MIN`` per worker-minute, ``LIGHTRAG_OPS_METRICS_HOUR``
rolled up hourly) and the alert history (``LIGHTRAG_OPS_ALERTS``, with a
partial unique index that allows one open row per fingerprint). Indexes on the large existing tables are never built
here: :mod:`lightrag.api.ops.indexes` builds them in the background.

Rules every query follows:

- reads run inside a read-only transaction with ``statement_timeout`` 3 s;
- status lists are inlined as SQL literals (from constants, never from input)
  so the planner can match the partial doc-status index under generic plans;
- sort columns come from a whitelist; every other value is a bound parameter;
- ``LIGHTRAG_DOC_STATUS.updated_at`` is a naive UTC ``TIMESTAMP``, so it is
  always compared with :data:`UTC_NOW_SQL`, whatever the server's TimeZone;
- heartbeats and leases go through connections of their own, never the shared
  pool, so a starved pool cannot make a live worker look dead.

``MemoryOpsStore`` has the same behaviour for offline tests.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Iterable, Optional

logger = logging.getLogger("lightrag.api.ops")

TABLE_KV = "LIGHTRAG_OPS_KV"
TABLE_AUDIT = "LIGHTRAG_OPS_AUDIT"
TABLE_DOC_STATUS = "LIGHTRAG_DOC_STATUS"
TABLE_OWNERS = "LIGHTRAG_WORKSPACE_OWNERS"
TABLE_USAGE_EVENTS = "LIGHTRAG_USAGE_EVENTS"
# Ops tables created by later features; deletion cleans them when they exist.
WORKSPACE_ROW_TABLES = (
    "LIGHTRAG_OPS_LOGS",
    "LIGHTRAG_OPS_DOC_FAILURES",
    "LIGHTRAG_OPS_RUNS",
)
TABLE_ALERTS = "LIGHTRAG_OPS_ALERTS"
TABLE_RUNS = "LIGHTRAG_OPS_RUNS"
TABLE_DOC_FAILURES = "LIGHTRAG_OPS_DOC_FAILURES"
TABLE_WORKERS = "LIGHTRAG_OPS_WORKERS"
TABLE_WORKER_SAMPLES = "LIGHTRAG_OPS_WORKER_SAMPLES"
TABLE_PROBES = "LIGHTRAG_OPS_PROBES"
TABLE_LOGS = "LIGHTRAG_OPS_LOGS"
TABLE_METRICS_MIN = "LIGHTRAG_OPS_METRICS_MIN"
TABLE_METRICS_HOUR = "LIGHTRAG_OPS_METRICS_HOUR"
METRIC_BUCKETS = 20
# Probe rows that carry vLLM /metrics (running, waiting, KV cache).
VLLM_DEPENDENCIES = ("embed", "rerank")
# Failure kinds that are not real failures (excluded from throughput/top).
NOT_REAL_FAILURE_KINDS = ("cancelled", "duplicate")
TOP_WORKSPACE_METRICS = ("processing_minutes", "failures", "runs")
# Built by ops/indexes.py; correlate reads doc status by id only through it.
DOC_STATUS_ID_INDEX = "idx_lightrag_doc_status_id"
# Columns the log sink writes, in COPY order.
LOG_COLUMNS = (
    "ts",
    "level",
    "logger",
    "worker_id",
    "pid",
    "module",
    "lineno",
    "workspace",
    "request_id",
    "client_request_id",
    "run_id",
    "track_id",
    "doc_id",
    "operation",
    "source",
    "msg",
    "exc_type",
    "exc",
    "fingerprint",
    "repeat_count",
)
# Retention deletes at most this many batches per cycle; the rest waits for
# the next cycle, so one cycle never runs long on a busy table.
LOG_RETENTION_MAX_BATCHES = 200
# Weekly audit: what must never be found in stored log text (FR-72).
LOG_AUDIT_PATTERNS = {
    "email": (
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
        False,
    ),
    "bearer": (r"bearer\s+[A-Za-z0-9._~+/=-]{6,}", True),
    "sk_key": (r"sk-[A-Za-z0-9_-]{8,}", False),
    "jwt": (r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.", False),
    "ic_number": (
        r"(^|[^0-9A-Za-z])[0-9]{6}-?[0-9]{2}-?[0-9]{4}([^0-9A-Za-z]|$)",
        False,
    ),
    "phone": (
        r"(^|[^0-9A-Za-z+])\+?6?0[ -]?1[0-9][ -]?[0-9]{3,4}[ -]?[0-9]{4}"
        r"([^0-9A-Za-z]|$)",
        False,
    ),
    # A name whose last three characters before the extension are not part
    # of a mask: a masked name keeps at most two ("Min\u2022\u2022\u2022an.pdf"), so
    # an unmasked name is counted even in a row that holds masked ones.
    "file_name": (
        "[^\\s\u2022]{3}\\.(pdf|docx?|xlsx?|pptx?|eml|msg)([^A-Za-z0-9]|$)",
        True,
    ),
}

READ_TIMEOUT_MS = 3000
DEDICATED_QUERY_TIMEOUT_S = 5.0
WRITE_TIMEOUT_MS = 5000
POOL_ACQUIRE_TIMEOUT_S = 5.0
CONNECT_TIMEOUT_S = 10.0
_TABLES_TTL_S = 300.0

UTC_NOW_SQL = "(now() AT TIME ZONE 'UTC')"

STATUS_PENDING = "pending"
STATUS_PARSING = "parsing"
STATUS_ANALYZING = "analyzing"
STATUS_PROCESSING = "processing"
STATUS_FAILED = "failed"
ACTIVE_STATUSES = (STATUS_PARSING, STATUS_ANALYZING, STATUS_PROCESSING)
INFLIGHT_STATUSES = (STATUS_PENDING, *ACTIVE_STATUSES)
BACKLOG_STATUSES = (*INFLIGHT_STATUSES, STATUS_FAILED)

CANCELLED_PREFIX = "User cancelled"
DUP_PREFIX = "dup-"

# Workspace-scoped KV keys: removed when the workspace is deleted.
# Audit actions that only read (searches, exports, reveals): kept in the
# Audit section, never shown as timeline markers.
AUDIT_READ_ACTIONS = (
    "log_search",
    "log_groups",
    "correlate",
    "log_export",
    "audit_export",
    "activity_export",
    "reveal",
)
WORKSPACE_KV_KEYS = ("resync:{ws}", "kick:{ws}", "action:{ws}")
WORKSPACE_KV_PREFIXES = ("pause:{ws}:",)
# KV summaries holding a per-workspace ``items`` list.
WORKSPACE_KV_SUMMARIES = ("email_queue_summary", "pause_summary")

BACKLOG_SORTS = {
    "inflight": "inflight",
    "pending": "pending",
    "processing": "processing",
    "failed": "failed",
    "cancelled": "cancelled",
    "oldest": "oldest_inflight_at",
    "activity": "last_activity_at",
    "name": "workspace",
}
BACKLOG_FILTERS = frozenset({"failed", "stuck", "email", "paused"})
FAILURE_GROUPS = ("signature", "workspace")
CANCELLED_GROUP_KEY = "cancelled"
# Durable failure groupings (LIGHTRAG_OPS_DOC_FAILURES columns, whitelisted).
DURABLE_GROUP_COLUMNS = {
    "error_kind": "error_kind",
    "stage": "stage",
    "model": "COALESCE(model, '')",
    "workspace": "workspace",
}
RUN_FINAL_STATUSES = ("succeeded", "failed", "cancelled", "paused")
RUNS_PAGE_FILTER_KINDS = (
    "process",
    "email_batch",
    "delete",
    "clear",
    "email_delete",
    "scan",
    "startup_resume",
    "admin",
)
RUNS_PAGE_FILTER_STATUSES = ("running", *RUN_FINAL_STATUSES, "orphaned")
ORPHAN_SUMMARY = "The process that owned this run exited"
# How an owner's death was confirmed (``orphaned_by``).
ORPHAN_REASONS = (
    "pid_gone",
    "create_time",
    "exit_marker",
    "exited",
    "superseded",
    "host_gone",
)
DELETED_KV_PREFIX = "deleted:"
# Tombstones outlive any row a tracker could still hold for the workspace.
DELETED_KEEP_DAYS = 7
RETENTION_BATCH = 10000

_LITERAL = re.compile(r"^[a-z_]+$")
_LIKE_SPECIAL = re.compile(r"([\\%_])")
# Legacy failure signature: hex runs and numbers collapse to N.
_SIGNATURE_PATTERN = r"[0-9a-f]{8,}|[0-9]+"
_SIGNATURE_RE = re.compile(_SIGNATURE_PATTERN)
_SIGNATURE_CHARS = 300


def sql_in(values: Iterable[str]) -> str:
    """``('a','b')`` from trusted constants only."""
    items = list(values)
    for value in items:
        if not _LITERAL.match(value):
            raise ValueError(f"Not a safe SQL literal: {value!r}")
    return "(" + ",".join(f"'{v}'" for v in items) + ")"


def like_escape(value: str) -> str:
    return _LIKE_SPECIAL.sub(r"\\\1", value)


def as_utc(value: Any) -> Optional[datetime]:
    """Naive values from ``LIGHTRAG_DOC_STATUS`` are UTC; aware ones are converted."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def utc_iso(value: Any) -> Optional[str]:
    moment = as_utc(value)
    return moment.isoformat() if moment else None


def workspace_hash(workspace: str) -> str:
    """What a deleted workspace becomes in kept audit rows."""
    return hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:16]


def deleted_key(workspace: str) -> str:
    """KV tombstone of a deleted workspace (``{"at": epoch}``).

    Run trackers check it before writing buffered rows, so rows buffered for
    a workspace before its deletion (by any worker) are never written back.
    """
    return f"{DELETED_KV_PREFIX}{workspace_hash(workspace)}"


def _pause_counts(rows: list[dict[str, Any]], workspaces: int) -> dict[str, Any]:
    by_slot_kind: dict[str, dict[str, int]] = {}
    total = 0
    for row in rows:
        count = int(row["count"])
        kinds = by_slot_kind.setdefault(row["slot"], {})
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + count
        total += count
    return {"total": total, "workspaces": workspaces, "by_slot_kind": by_slot_kind}


def pg_text(value: Any, limit: int) -> Optional[str]:
    """``value`` cut to ``limit`` characters, without the NUL characters
    Postgres text rejects; None when empty."""
    if value is None:
        return None
    text = str(value)
    if "\x00" in text:
        text = text.replace("\x00", "")
    return text[:limit] or None


def failure_signature(message: Optional[str]) -> str:
    """Python twin of the SQL legacy signature (md5 of the normalised head)."""
    head = (message or "")[:_SIGNATURE_CHARS]
    return hashlib.md5(_SIGNATURE_RE.sub("N", head).encode("utf-8")).hexdigest()


def is_cancelled(message: Optional[str]) -> bool:
    return (message or "").startswith(CANCELLED_PREFIX)


_IN_BACKLOG = sql_in(BACKLOG_STATUSES)
_IN_ACTIVE = sql_in(ACTIVE_STATUSES)
_IN_INFLIGHT = sql_in(INFLIGHT_STATUSES)
_NOT_DUP = "id NOT LIKE 'dup-%'"
_IS_DUP = "id LIKE 'dup-%'"
_CANCELLED = "COALESCE(error_msg, '') LIKE 'User cancelled%'"
_SIGNATURE_SQL = (
    "md5(regexp_replace(left(COALESCE(error_msg, ''), "
    f"{_SIGNATURE_CHARS}), '{_SIGNATURE_PATTERN}', 'N', 'g'))"
)
_NO_FAILURE_META = "NOT (COALESCE(metadata, '{}'::jsonb) ? 'failure')"
# Both example columns of a failure group come from the same (newest) row,
# so the example's own path can be masked inside its message.
_EXAMPLE_ORDER = "ORDER BY updated_at DESC, workspace, id"


@dataclass(frozen=True)
class Page:
    limit: int
    offset: int


@dataclass(frozen=True)
class LogFilter:
    """One ``LIGHTRAG_OPS_LOGS`` query: a time window plus optional filters.

    ``level`` is a minimum (logging level number). ``logger`` matches a
    logger-name prefix; ``module`` matches the module *or* the logger name
    exactly (the viewer shows the logger only when it is not ``lightrag``).
    ``q`` is a case-insensitive substring of the message. ``run_id`` and
    ``worker_id`` are 32 hex characters.
    """

    start: datetime
    end: datetime
    level: int = 30
    workspace: Optional[str] = None
    logger: Optional[str] = None
    module: Optional[str] = None
    request_id: Optional[str] = None
    run_id: Optional[str] = None
    track_id: Optional[str] = None
    doc_id: Optional[str] = None
    fingerprint: Optional[str] = None
    worker_id: Optional[str] = None
    q: Optional[str] = None


@dataclass(frozen=True)
class AuditFilter:
    """One ``LIGHTRAG_OPS_AUDIT`` query: a time window plus exact-match filters.

    ``phase`` is ``intent``, ``result`` or None (both).
    """

    start: datetime
    end: datetime
    actor: Optional[str] = None
    action: Optional[str] = None
    workspace: Optional[str] = None
    phase: Optional[str] = None


class OpsStore(ABC):
    """What the ops service needs from storage."""

    ready: bool = False
    doc_status_ready: bool = False

    async def initialize(self) -> None:  # pragma: no cover - default no-op
        pass

    async def finalize(self) -> None:  # pragma: no cover - default no-op
        pass

    @abstractmethod
    async def ping(self) -> None:
        """A trivial read; raises when the database does not answer."""

    # -- KV ---------------------------------------------------------------
    @abstractmethod
    async def kv_get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    async def kv_get_many(self, keys: list[str]) -> dict[str, Any]: ...

    @abstractmethod
    async def kv_put(self, key: str, value: Any) -> None: ...

    @abstractmethod
    async def kv_merge_many(self, patches: dict[str, dict[str, Any]]) -> None:
        """Shallow-merge each patch into its JSON object (created if missing)."""

    @abstractmethod
    async def kv_cas(self, key: str, expected: Any, value: Any) -> bool:
        """Write ``value`` only if the stored value equals ``expected``
        (``None`` means the key must be absent)."""

    @abstractmethod
    async def kv_delete(self, keys: list[str]) -> None: ...

    # -- leases -------------------------------------------------------------
    @abstractmethod
    async def try_lease(self, name: str, holder: str, ttl_s: float) -> bool:
        """Take or renew ``lease:<name>``: free, expired, or already ours."""

    @abstractmethod
    async def release_lease(self, name: str, holder: str) -> None: ...

    # -- doc status reads ----------------------------------------------------
    @abstractmethod
    async def backlog_totals(self) -> dict[str, Any]:
        """Non-dup counts by status (``failed`` includes cancellations), the
        oldest in-flight and pending times, and the last day's real and
        cancelled failures."""

    @abstractmethod
    async def cancelled_count(self) -> int:
        """Non-dup failed rows that were cancellations (any age)."""

    @abstractmethod
    async def dup_count(self) -> int: ...

    @abstractmethod
    async def backlog_page(
        self,
        *,
        q: Optional[str],
        has: frozenset[str],
        sort: str,
        descending: bool,
        page: Page,
        stale_minutes: int,
        email_workspaces: list[str],
        paused_workspaces: list[str],
    ) -> tuple[list[dict[str, Any]], int]: ...

    @abstractmethod
    async def stale_docs(
        self, older_than_minutes: int, page: Page
    ) -> tuple[list[dict[str, Any]], int]: ...

    @abstractmethod
    async def live_candidates(self, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def pending_candidates(self, limit: int) -> list[str]: ...

    @abstractmethod
    async def paused_candidates(self, days: int, limit: int) -> list[str]:
        """Workspaces whose latest pause event is a provider rejection that paused."""

    @abstractmethod
    async def legacy_failures(
        self, hours: int, group: str, include_cancelled: bool, limit: int = 50
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def failure_docs(
        self, hours: int, group: str, key: str, page: Page
    ) -> tuple[list[dict[str, Any]], int]: ...

    @abstractmethod
    async def workspace_counts(self, workspace: str) -> dict[str, Any]: ...

    @abstractmethod
    async def workspace_documents(
        self, workspace: str, status: Optional[str], page: Page
    ) -> tuple[list[dict[str, Any]], int]: ...

    @abstractmethod
    async def document(
        self, workspace: str, doc_id: str
    ) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    async def owners(self, workspaces: list[str]) -> dict[str, str]:
        """Owner email (or identity) by workspace, from the registry table."""

    # -- events ----------------------------------------------------------------
    @abstractmethod
    async def usage_events(
        self, start: datetime, end: datetime, kinds: list[str], limit: int
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def audit_events(
        self, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]: ...

    # -- deletion ---------------------------------------------------------------
    @abstractmethod
    async def forget_workspace(self, workspace: str) -> None:
        """Delete a workspace's ops rows and KV; hash it where rows are kept."""

    # -- workers (heartbeat; writes on a dedicated connection) ------------------
    @abstractmethod
    async def worker_beat(
        self, row: dict[str, Any], sample: Optional[dict[str, Any]] = None
    ) -> None:
        """Upsert this worker's row (``heartbeat_at`` from the database clock)
        and, when given, its sample for the current minute."""

    @abstractmethod
    async def supersede_workers(
        self,
        host: str,
        worker_id: str,
        started_at: datetime,
        stale_s: float,
        alive: Optional[Callable[[Any, Any], bool]] = None,
    ) -> int:
        """Mark rows of earlier workers on ``host`` that stopped beating.

        Only rows started before ``started_at`` qualify. ``alive(pid,
        pid_create_time)``, when given, vetoes rows whose process still runs
        (a hung sibling is not superseded, it is a dead worker to show).
        """

    @abstractmethod
    async def worker_exited(self, worker_id: str) -> None: ...

    @abstractmethod
    async def cleanup_workers(self, row_hours: int, sample_days: int) -> None: ...

    @abstractmethod
    async def workers(self, limit: int) -> list[dict[str, Any]]:
        """Worker rows, newest beat first, with ``heartbeat_age_s`` by the
        database clock."""

    @abstractmethod
    async def worker_history(
        self, worker_id: str, hours: int, bucket_s: int
    ) -> list[dict[str, Any]]: ...

    def pool_stats(self) -> Optional[dict[str, Any]]:
        """Size of the shared connection pool, when there is one."""
        return None

    async def pool_wait_ms(self, timeout_s: float) -> Optional[float]:
        """How long taking a connection from the shared pool takes now."""
        return None

    # -- dependency probes ------------------------------------------------------
    @abstractmethod
    async def postgres_health(
        self, *, extras: bool, timeout_s: float
    ) -> dict[str, Any]:
        """``SELECT 1`` on a connection of its own; with ``extras`` also
        connections, database size and blocked backends (never query text)."""

    @abstractmethod
    async def probe_write(self, rows: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def probe_strips(self, hours: int, slot_s: int) -> list[dict[str, Any]]:
        """Result counts per dependency and time slot."""

    @abstractmethod
    async def probe_recent(
        self, dependency: str, limit: int
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def probe_series(
        self, dependency: str, hours: int, bucket_s: int
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def probe_error_classes(
        self, dependency: str, hours: int
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def probe_cleanup(self, days: int) -> None: ...

    # -- run tracking (writes come from the buffered RunTracker) ----------------
    @abstractmethod
    async def kv_merge_existing(self, patches: dict[str, dict[str, Any]]) -> None:
        """Shallow-merge each patch into an *existing* JSON object; absent keys
        stay absent."""

    @abstractmethod
    async def runs_write(self, rows: list[dict[str, Any]]) -> None:
        """Upsert run rows (epoch-second times). A row that already reached a
        final status is never changed again; a row the reconciler orphaned
        takes the new status and keeps ``was_orphaned``."""

    @abstractmethod
    async def doc_failures_write(self, rows: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def runs_orphan(
        self, run_ids: list[str], confirmed_by: Optional[dict[str, str]] = None
    ) -> list[str]:
        """Mark still-running runs orphaned; returns the IDs that changed.

        ``confirmed_by`` records, per run, how the owner's death was
        confirmed (one of :data:`ORPHAN_REASONS`)."""

    @abstractmethod
    async def runs_cleanup(self, run_days: int, failure_days: int) -> None: ...

    @abstractmethod
    async def running_runs(self, limit: int) -> list[dict[str, Any]]:
        """Running runs, newest first, with ages by the database clock and the
        owner worker's row (``worker_*`` fields; None when it has none)."""

    @abstractmethod
    async def orphaned_runs(self, hours: int, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def runs_page(
        self,
        *,
        workspace: Optional[str],
        kind: Optional[str],
        status: Optional[str],
        before: Optional[tuple[datetime, str]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Run history, newest first, keyset on ``(started_at, run_id)``."""

    @abstractmethod
    async def run_get(self, run_id: str) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    async def run_failures(self, run_id: str, limit: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def last_runs(self, workspaces: list[str]) -> dict[str, dict[str, Any]]:
        """Latest run per workspace, for the given workspaces only."""

    @abstractmethod
    async def failure_groups(
        self, hours: int, group: str, include_cancelled: bool, limit: int = 50
    ) -> dict[str, Any]:
        """Classified failures grouped by kind, stage, model or workspace."""

    @abstractmethod
    async def failure_group_docs(
        self, hours: int, group: str, key: str, page: Page
    ) -> tuple[list[dict[str, Any]], int]:
        """Documents of one durable failure group (latest failure per doc)."""

    @abstractmethod
    async def failures_by_kind(
        self, workspace: str, hours: int
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def doc_failures(
        self, workspace: str, doc_id: str, limit: int
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def open_pauses(self, limit: int) -> list[dict[str, Any]]:
        """Persisted provider pauses (``pause:<ws>:<slot>``) not cleared yet."""

    @abstractmethod
    async def pauses_page(self, limit: int, offset: int) -> list[dict[str, Any]]:
        """One page of open persisted pauses, oldest ``since`` first."""

    @abstractmethod
    async def pause_counts(self) -> dict[str, Any]:
        """Open persisted pauses counted in SQL (never capped): ``total``,
        ``workspaces`` and ``by_slot_kind`` (``{slot: {kind: count}}``)."""

    @abstractmethod
    async def run_events(
        self, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """Orphaned runs in a window, for the timeline."""

    # -- audit -------------------------------------------------------------------
    @abstractmethod
    async def audit_write(self, row: dict[str, Any]) -> int:
        """Insert one ``LIGHTRAG_OPS_AUDIT`` row; returns its id."""

    @abstractmethod
    async def audit_page(
        self, f: "AuditFilter", before: Optional[tuple[datetime, int]], limit: int
    ) -> list[dict[str, Any]]:
        """Audit rows matching ``f``, newest first, on the ``(ts, id)`` keyset
        (strictly older than ``before``)."""

    # -- admin actions -----------------------------------------------------------
    @abstractmethod
    async def claim_action(
        self, key: str, holder: str, ttl_s: float, info: dict[str, Any]
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Take the action lease ``key`` (``{holder, exp, **info}``) when it is
        free, expired or already ``holder``'s. Returns ``(taken, current)``;
        ``current`` is the holder's value when it was not taken."""

    @abstractmethod
    async def release_action(self, key: str, holder: str) -> None:
        """Drop the action lease ``key`` if ``holder`` still holds it."""

    @abstractmethod
    async def rate_hit(
        self, key: str, limit: int, window_s: float, now: float
    ) -> tuple[bool, float]:
        """Count one hit in ``key``'s fixed window. Returns ``(allowed,
        retry_after_s)``; refused hits count too."""

    @abstractmethod
    async def kv_put_if_older(
        self, key: str, value: dict[str, Any], older_than: float
    ) -> bool:
        """Write ``value`` (which carries ``at``) unless the stored value's
        ``at`` is newer than ``older_than``. True when written."""

    # -- alerts (written by the leased evaluator, see alerts.py) -------------------
    @abstractmethod
    async def alerts_open(self, limit: int) -> list[dict[str, Any]]:
        """Open (unresolved) alert rows, newest first."""

    @abstractmethod
    async def alerts_upsert(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert a firing row per fingerprint, or touch its open row
        (severity, title, scope, link, runbook, ``last_seen``). The partial
        unique index on ``fingerprint WHERE resolved_at IS NULL`` makes a
        second open row for a fingerprint impossible, also when two
        evaluators race. Returns ``{id, fingerprint, inserted}`` per item."""

    @abstractmethod
    async def alerts_resolve(self, fingerprints: list[str]) -> list[dict[str, Any]]:
        """Resolve the open rows of ``fingerprints``; returns the resolved rows."""

    @abstractmethod
    async def alerts_page(
        self, state: Optional[str], before: Optional[int], limit: int
    ) -> list[dict[str, Any]]:
        """Alert rows (``open``, ``resolved`` or all), newest first, with
        ``id < before``."""

    @abstractmethod
    async def alert_get(self, alert_id: int) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    async def alerts_history(
        self, fingerprint: str, limit: int
    ) -> list[dict[str, Any]]:
        """Rows of one fingerprint (its earlier incidents), newest first."""

    @abstractmethod
    async def alert_ack(
        self, alert_id: int, actor: str, note: Optional[str]
    ) -> tuple[str, Optional[dict[str, Any]]]:
        """Record an ack on an open row (its state does not change). Returns
        ``(code, row)``: ``acked``, ``already_acked``, ``resolved`` or
        ``not_found``."""

    @abstractmethod
    async def alert_claim_email(
        self, alert_id: int, kind: str, repeat_s: float
    ) -> Optional[datetime]:
        """Claim one email for a row with a conditional UPDATE; the new
        ``last_emailed_at`` when this caller won, else None. ``open``: the
        row is open and was never emailed or not within ``repeat_s``;
        ``resolve``: the row is resolved and its last email went out before
        it resolved."""

    @abstractmethod
    async def alert_unclaim_email(
        self, alert_id: int, claimed: datetime, previous: Optional[datetime]
    ) -> None:
        """Undo a claim whose send failed (only if nobody claimed since)."""

    @abstractmethod
    async def alerts_window(
        self, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """Alert rows active at any time in ``[start, end)``, newest first."""

    @abstractmethod
    async def alerts_cleanup(self, days: int) -> int:
        """Delete rows resolved more than ``days`` ago, in batches."""

    # -- logs (writes come from the buffered log sink) -----------------------------
    @abstractmethod
    async def logs_write(self, rows: list[dict[str, Any]]) -> None:
        """Insert log rows (epoch-second ``ts``, columns of :data:`LOG_COLUMNS`)."""

    @abstractmethod
    async def logs_page(
        self,
        f: LogFilter,
        *,
        before: Optional[tuple[datetime, int]],
        after: Optional[datetime],
        limit: int,
        with_exc: bool = False,
    ) -> list[dict[str, Any]]:
        """Newest first, keyset on ``(ts, id)`` below ``before``; ``after``
        (follow mode) keeps only rows at or after that time. ``with_exc``
        adds the traceback column (exports)."""

    @abstractmethod
    async def logs_groups(self, f: LogFilter, limit: int) -> list[dict[str, Any]]:
        """Rows grouped by call-site fingerprint, largest first, each with the
        newest row's message as its sample."""

    @abstractmethod
    async def log_get(self, log_id: int) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    async def logs_count(self, f: LogFilter, cap: int) -> int:
        """Matching rows, counted up to ``cap + 1``."""

    @abstractmethod
    async def logs_cleanup(self, days: int) -> int:
        """Delete rows older than ``days`` in batches; returns rows deleted."""

    @abstractmethod
    async def logs_redaction_audit(self, days: int) -> dict[str, int]:
        """Rows of the last ``days`` whose text still matches a pattern of
        :data:`LOG_AUDIT_PATTERNS` (counts only, never the text)."""

    # -- correlation (≤ ``limit`` rows each, all on indexes) ---------------------------
    @abstractmethod
    async def correlate_requests(
        self, request_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Log lines of one request ID, per workspace: ``n``, ``first_ts``, ``last_ts``."""

    @abstractmethod
    async def correlate_tracks(self, track_id: str, limit: int) -> list[dict[str, Any]]:
        """Workspaces holding documents or log lines of a track ID."""

    @abstractmethod
    async def correlate_docs(self, doc_id: str, limit: int) -> list[dict[str, Any]]:
        """Workspaces holding a document ID (doc status, failures, log lines)."""

    @abstractmethod
    async def owners_by_email(self, email: str, limit: int) -> dict[str, str]:
        """Workspaces whose registered owner email or identity is ``email``."""

    @abstractmethod
    async def workspace_presence(self, workspace: str) -> dict[str, bool]:
        """Whether a workspace has documents, runs or log lines."""

    # -- performance metrics (PR-6) -----------------------------------------------
    @abstractmethod
    async def metrics_write(self, rows: list[dict[str, Any]]) -> None:
        """Insert per worker-minute histogram rows (epoch-second ``minute``).

        Insert-only: a row already stored (a retried write) is left as is.
        """

    @abstractmethod
    async def metrics_rollup(self, start: datetime, end: datetime) -> int:
        """Recompute the hour rows for ``[start, end)`` from the minute rows."""

    @abstractmethod
    async def metrics_cleanup(
        self, minute_days: int, hour_days: int
    ) -> tuple[int, int]:
        """Delete minute and hour rows past retention (batched)."""

    @abstractmethod
    async def metrics_read(
        self,
        series: list[str],
        start: datetime,
        cut: datetime,
        end: datetime,
        bucket_s: int,
    ) -> list[dict[str, Any]]:
        """Histograms summed per ``(t, series, key)``, ``t`` an epoch bucket.

        Hour rows cover ``[start, cut)`` and minute rows ``[cut, end)``, so a
        window reads the rollup where it exists and minutes for the rest.
        """

    @abstractmethod
    async def vllm_series(
        self, start: datetime, end: datetime, bucket_s: int
    ) -> list[dict[str, Any]]:
        """vLLM ``/metrics`` from the embed and rerank probe rows per bucket:
        ``[{dependency, t, running, waiting, kv_cache_pct}]`` (running
        averaged, waiting and KV cache use at their maximum)."""

    @abstractmethod
    async def llm_queue_series(
        self, start: datetime, end: datetime, bucket_s: int
    ) -> list[dict[str, Any]]:
        """LLM calls running, in flight and queued per bucket, summed over
        the workers (each worker's maximum in the bucket), from the
        once-a-minute heartbeat samples: ``[{t, running, in_flight, queued}]``."""

    @abstractmethod
    async def throughput(
        self, start: datetime, end: datetime, bucket_s: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Documents processed and failed (runs, real failures) and email
        batches per bucket."""

    @abstractmethod
    async def top_workspaces(
        self, metric: str, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """``[{workspace, value}]`` by processing minutes, real failures or
        runs in the window, largest first."""


def _workspace_keys(workspace: str) -> list[str]:
    return [template.format(ws=workspace) for template in WORKSPACE_KV_KEYS]


def _workspace_prefixes(workspace: str) -> list[str]:
    return [template.format(ws=workspace) for template in WORKSPACE_KV_PREFIXES]


class PostgresOpsStore(OpsStore):
    """Ops tables and queries in Postgres, on the shared LightRAG pool."""

    def __init__(self, vector_storage: Optional[str] = None):
        self._vector_storage = vector_storage
        self._db: Any = None
        self._lease_conn: Any = None
        self._lease_lock: Optional[asyncio.Lock] = None
        self._beat_conn: Any = None
        self._beat_lock: Optional[asyncio.Lock] = None
        self._tables: set[str] = set()
        self._tables_at = 0.0
        self._index_checks: dict[str, tuple[bool, float]] = {}
        self.ready = False
        self.doc_status_ready = False

    # -- lifecycle -----------------------------------------------------------
    async def initialize(self) -> None:
        from lightrag.kg.postgres_impl import ClientManager

        self._db = await ClientManager.get_client(self._vector_storage)
        for statement in (
            f"""CREATE TABLE IF NOT EXISTS {TABLE_KV} (
                key VARCHAR(512) PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_AUDIT} (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                actor VARCHAR(255) NOT NULL,
                auth_kind VARCHAR(16),
                impersonated_by VARCHAR(255),
                action VARCHAR(48) NOT NULL,
                workspace VARCHAR(255),
                target VARCHAR(255),
                params JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                reason VARCHAR(512),
                phase VARCHAR(8) NOT NULL,
                outcome VARCHAR(24),
                request_id VARCHAR(64),
                client_ip_hash CHAR(16)
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_audit_ts ON {TABLE_AUDIT} (ts DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_ops_audit_actor ON {TABLE_AUDIT} (actor, ts DESC)",
            # The Audit section's action and workspace filters (a rare action
            # or one of 3000 workspaces over 400 days must not walk the ts
            # index row by row).
            f"CREATE INDEX IF NOT EXISTS idx_ops_audit_action ON {TABLE_AUDIT} (action, ts DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_ops_audit_workspace ON {TABLE_AUDIT} (workspace, ts DESC)",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_WORKERS} (
                worker_id UUID PRIMARY KEY,
                host VARCHAR(255) NOT NULL,
                pid INT NOT NULL,
                pid_create_time DOUBLE PRECISION,
                mode VARCHAR(16) NOT NULL,
                version VARCHAR(64),
                started_at TIMESTAMPTZ NOT NULL,
                heartbeat_at TIMESTAMPTZ NOT NULL,
                exited_at TIMESTAMPTZ,
                superseded BOOLEAN NOT NULL DEFAULT false,
                leases_held TEXT[] NOT NULL DEFAULT '{{}}',
                snapshot JSONB NOT NULL
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_workers_hb ON {TABLE_WORKERS} (heartbeat_at DESC)",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_WORKER_SAMPLES} (
                worker_id UUID NOT NULL,
                minute TIMESTAMPTZ NOT NULL,
                cpu_pct REAL,
                rss BIGINT,
                loop_lag_ms INT,
                pool_wait_ms INT,
                llm_running INT,
                llm_in_flight INT,
                llm_queued INT,
                PRIMARY KEY (worker_id, minute)
            )""",
            # Summed LLM queue counts per sample (PR-6), for tables made before.
            f"ALTER TABLE {TABLE_WORKER_SAMPLES} ADD COLUMN IF NOT EXISTS llm_running INT",
            f"ALTER TABLE {TABLE_WORKER_SAMPLES} ADD COLUMN IF NOT EXISTS llm_in_flight INT",
            f"ALTER TABLE {TABLE_WORKER_SAMPLES} ADD COLUMN IF NOT EXISTS llm_queued INT",
            f"CREATE INDEX IF NOT EXISTS idx_ops_worker_samples_minute ON {TABLE_WORKER_SAMPLES} (minute)",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_PROBES} (
                ts TIMESTAMPTZ NOT NULL,
                dependency VARCHAR(32) NOT NULL,
                status VARCHAR(8) NOT NULL,
                latency_ms INT,
                error_class VARCHAR(64),
                detail JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                PRIMARY KEY (dependency, ts)
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_probes_ts ON {TABLE_PROBES} (ts DESC)",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_RUNS} (
                run_id UUID PRIMARY KEY,
                workspace VARCHAR(255) NOT NULL,
                kind VARCHAR(24) NOT NULL,
                trigger VARCHAR(32) NOT NULL,
                status VARCHAR(16) NOT NULL,
                was_orphaned BOOLEAN NOT NULL DEFAULT false,
                worker_id UUID,
                pid INT,
                started_at TIMESTAMPTZ NOT NULL,
                heartbeat_at TIMESTAMPTZ NOT NULL,
                progress_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                docs_total INT NOT NULL DEFAULT 0,
                docs_done INT NOT NULL DEFAULT 0,
                docs_failed INT NOT NULL DEFAULT 0,
                stage VARCHAR(16),
                label VARCHAR(128),
                error_kind VARCHAR(32),
                error_summary VARCHAR(512),
                request_id VARCHAR(64),
                host VARCHAR(255),
                orphaned_by VARCHAR(16)
            )""",
            # Columns added while run tracking was in development.
            f"ALTER TABLE {TABLE_RUNS} ADD COLUMN IF NOT EXISTS host VARCHAR(255)",
            f"ALTER TABLE {TABLE_RUNS} ADD COLUMN IF NOT EXISTS orphaned_by VARCHAR(16)",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_runs_running ON {TABLE_RUNS}
                (heartbeat_at) WHERE status = 'running'""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_runs_ws ON {TABLE_RUNS} (workspace, started_at DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_ops_runs_started ON {TABLE_RUNS} (started_at DESC)",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_DOC_FAILURES} (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL,
                workspace VARCHAR(255) NOT NULL,
                doc_id VARCHAR(255) NOT NULL,
                track_id VARCHAR(255),
                run_id UUID,
                stage VARCHAR(16) NOT NULL,
                error_kind VARCHAR(32) NOT NULL,
                http_status SMALLINT,
                exc_type VARCHAR(128),
                message VARCHAR(1024),
                attempt INT NOT NULL,
                model VARCHAR(255)
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_fail_ts ON {TABLE_DOC_FAILURES} (ts DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_ops_fail_kind ON {TABLE_DOC_FAILURES} (error_kind, ts DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_ops_fail_doc ON {TABLE_DOC_FAILURES} (workspace, doc_id, ts DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_ops_fail_docid ON {TABLE_DOC_FAILURES} (doc_id)",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_fail_run ON {TABLE_DOC_FAILURES}
                (run_id) WHERE run_id IS NOT NULL""",
            # Plain table (v1); partition only if volume exceeds ~1M rows/day.
            f"""CREATE TABLE IF NOT EXISTS {TABLE_LOGS} (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL,
                level SMALLINT NOT NULL,
                logger VARCHAR(128) NOT NULL,
                worker_id UUID,
                pid INT,
                module VARCHAR(128),
                lineno INT,
                workspace VARCHAR(255),
                request_id VARCHAR(64),
                client_request_id VARCHAR(64),
                run_id UUID,
                track_id VARCHAR(255),
                doc_id VARCHAR(255),
                operation VARCHAR(64),
                source VARCHAR(32),
                msg TEXT NOT NULL,
                exc_type VARCHAR(128),
                exc TEXT,
                fingerprint CHAR(16) NOT NULL,
                repeat_count INT NOT NULL DEFAULT 1
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_logs_ts ON {TABLE_LOGS} (ts DESC, id DESC)",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_logs_ws ON {TABLE_LOGS}
                (workspace, ts DESC, id DESC)""",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_logs_lvl ON {TABLE_LOGS}
                (level, ts DESC, id DESC) WHERE level >= 40""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_logs_fp ON {TABLE_LOGS} (fingerprint, ts DESC)",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_logs_req ON {TABLE_LOGS}
                (request_id) WHERE request_id IS NOT NULL""",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_logs_run ON {TABLE_LOGS}
                (run_id) WHERE run_id IS NOT NULL""",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_logs_doc ON {TABLE_LOGS}
                (doc_id, ts) WHERE doc_id IS NOT NULL""",
            f"""CREATE INDEX IF NOT EXISTS idx_ops_logs_track ON {TABLE_LOGS}
                (track_id) WHERE track_id IS NOT NULL""",
            # Insert-only per worker-minute histograms (14 d) and their hourly
            # rollup (90 d); 20 log buckets from 5 ms to 120 s.
            f"""CREATE TABLE IF NOT EXISTS {TABLE_METRICS_MIN} (
                minute TIMESTAMPTZ NOT NULL,
                worker_id UUID NOT NULL,
                series VARCHAR(16) NOT NULL,
                key VARCHAR(160) NOT NULL,
                count INT NOT NULL,
                errors INT NOT NULL,
                sum_ms BIGINT NOT NULL,
                max_ms INT NOT NULL,
                buckets INT[] NOT NULL,
                PRIMARY KEY (minute, worker_id, series, key)
            )""",
            f"""CREATE TABLE IF NOT EXISTS {TABLE_METRICS_HOUR} (
                hour TIMESTAMPTZ NOT NULL,
                series VARCHAR(16) NOT NULL,
                key VARCHAR(160) NOT NULL,
                count BIGINT NOT NULL,
                errors BIGINT NOT NULL,
                sum_ms BIGINT NOT NULL,
                max_ms INT NOT NULL,
                buckets BIGINT[] NOT NULL,
                PRIMARY KEY (hour, series, key)
            )""",
            # Alerts (180 d): firing -> resolved; at most one open row per
            # fingerprint, enforced by the partial unique index.
            f"""CREATE TABLE IF NOT EXISTS {TABLE_ALERTS} (
                id BIGSERIAL PRIMARY KEY,
                rule VARCHAR(64) NOT NULL,
                fingerprint VARCHAR(128) NOT NULL,
                severity VARCHAR(16) NOT NULL,
                state VARCHAR(16) NOT NULL,
                title VARCHAR(512) NOT NULL,
                scope JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                link VARCHAR(512),
                runbook VARCHAR(512),
                first_seen TIMESTAMPTZ NOT NULL,
                last_seen TIMESTAMPTZ NOT NULL,
                resolved_at TIMESTAMPTZ,
                acked_by VARCHAR(255),
                acked_at TIMESTAMPTZ,
                ack_note VARCHAR(512),
                last_emailed_at TIMESTAMPTZ
            )""",
            f"""CREATE UNIQUE INDEX IF NOT EXISTS idx_ops_alerts_open ON {TABLE_ALERTS}
                (fingerprint) WHERE resolved_at IS NULL""",
            f"CREATE INDEX IF NOT EXISTS idx_ops_alerts_seen ON {TABLE_ALERTS} (last_seen DESC)",
            # The Alert drawer's earlier occurrences of a fingerprint.
            f"CREATE INDEX IF NOT EXISTS idx_ops_alerts_fp ON {TABLE_ALERTS} (fingerprint, id DESC)",
        ):
            await self._db.execute(statement)
        await self._refresh_tables(force=True)
        self.doc_status_ready = TABLE_DOC_STATUS in self._tables
        self.ready = True

    async def finalize(self) -> None:
        self.ready = False
        for attr in ("_lease_conn", "_beat_conn"):
            conn = getattr(self, attr)
            setattr(self, attr, None)
            if conn is not None:
                await _close_quietly(conn)
        if self._db is not None:
            from lightrag.kg.postgres_impl import ClientManager

            await ClientManager.release_client(self._db)
            self._db = None

    # -- plumbing --------------------------------------------------------------
    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[Any]:
        await self._db._ensure_pool()
        async with self._db.pool.acquire(timeout=POOL_ACQUIRE_TIMEOUT_S) as conn:
            yield conn

    async def _read(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        async with self._conn() as conn:
            async with conn.transaction(readonly=True):
                await conn.execute(f"SET LOCAL statement_timeout = {READ_TIMEOUT_MS}")
                rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]

    async def _write(self, sql: str, *args: Any) -> Any:
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                return await conn.fetch(sql, *args)

    def connect_kwargs(self) -> dict[str, Any]:
        """Parameters for a connection outside the pool (same server and TLS)."""
        db = self._db
        params: dict[str, Any] = {
            "user": db.user,
            "password": db.password,
            "database": db.database,
            "host": db.host,
            "port": db.port,
            "timeout": CONNECT_TIMEOUT_S,
        }
        ssl_context = db._create_ssl_context()
        if ssl_context is not None:
            params["ssl"] = ssl_context
        elif db.ssl_mode:
            mode = db.ssl_mode.lower()
            if mode in ("require", "prefer"):
                params["ssl"] = True
            elif mode == "disable":
                params["ssl"] = False
        settings: dict[str, str] = {}
        if db.server_settings:
            for pair in str(db.server_settings).split("&"):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    settings[key] = value
        settings["application_name"] = "lightrag-ops"
        params["server_settings"] = settings
        return params

    @asynccontextmanager
    async def dedicated_connection(self) -> AsyncIterator[Any]:
        """A short-lived connection of its own, never one from the pool."""
        import asyncpg

        conn = await asyncpg.connect(**self.connect_kwargs())
        try:
            yield conn
        finally:
            await _close_quietly(conn)

    async def _refresh_tables(self, force: bool = False) -> set[str]:
        now = time.monotonic()
        if not force and now - self._tables_at < _TABLES_TTL_S:
            return self._tables
        names = [
            TABLE_DOC_STATUS,
            TABLE_OWNERS,
            TABLE_USAGE_EVENTS,
            TABLE_ALERTS,
            *WORKSPACE_ROW_TABLES,
        ]
        rows = await self._read(
            "SELECT n FROM unnest($1::text[]) AS n WHERE to_regclass(lower(n)) IS NOT NULL",
            names,
        )
        self._tables = {r["n"] for r in rows}
        self._tables_at = now
        return self._tables

    async def _has_table(self, name: str) -> bool:
        try:
            return name in await self._refresh_tables()
        except Exception:
            return name in self._tables

    # -- KV ---------------------------------------------------------------------
    @staticmethod
    def _decode(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    async def ping(self) -> None:
        await self._read("SELECT 1")

    async def kv_get(self, key: str, default: Any = None) -> Any:
        rows = await self._read(f"SELECT value FROM {TABLE_KV} WHERE key = $1", key)
        return self._decode(rows[0]["value"]) if rows else default

    async def kv_get_many(self, keys: list[str]) -> dict[str, Any]:
        if not keys:
            return {}
        rows = await self._read(
            f"SELECT key, value FROM {TABLE_KV} WHERE key = ANY($1::varchar[])",
            list(keys),
        )
        return {r["key"]: self._decode(r["value"]) for r in rows}

    async def kv_put(self, key: str, value: Any) -> None:
        await self._write(
            f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()""",
            key,
            json.dumps(value, default=str),
        )

    async def kv_merge_many(self, patches: dict[str, dict[str, Any]]) -> None:
        if not patches:
            return
        rows = [(k, json.dumps(v, default=str)) for k, v in patches.items()]
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                await conn.executemany(
                    f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                        VALUES ($1, $2::jsonb, now())
                        ON CONFLICT (key) DO UPDATE
                        SET value = CASE
                                WHEN jsonb_typeof({TABLE_KV}.value) = 'object'
                                THEN {TABLE_KV}.value || EXCLUDED.value
                                ELSE EXCLUDED.value END,
                            updated_at = now()""",
                    rows,
                )

    async def kv_cas(self, key: str, expected: Any, value: Any) -> bool:
        payload = json.dumps(value, default=str)
        if expected is None:
            rows = await self._write(
                f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO NOTHING RETURNING key""",
                key,
                payload,
            )
        else:
            rows = await self._write(
                f"""UPDATE {TABLE_KV} SET value = $2::jsonb, updated_at = now()
                    WHERE key = $1 AND value = $3::jsonb RETURNING key""",
                key,
                payload,
                json.dumps(expected, default=str),
            )
        return bool(rows)

    async def kv_delete(self, keys: list[str]) -> None:
        if keys:
            await self._write(
                f"DELETE FROM {TABLE_KV} WHERE key = ANY($1::varchar[])", list(keys)
            )

    # -- leases (dedicated connection) ---------------------------------------------
    def _lease_guard(self) -> asyncio.Lock:
        if self._lease_lock is None:
            self._lease_lock = asyncio.Lock()
        return self._lease_lock

    async def _dedicated_query(
        self, attr: str, guard: asyncio.Lock, sql: str, *args: Any
    ) -> list[Any]:
        """One statement on the long-lived connection kept in ``attr``.

        A broken connection is replaced once; a second failure raises.
        """
        import asyncpg

        async with guard:
            for attempt in (1, 2):
                conn = getattr(self, attr)
                if conn is None or conn.is_closed():
                    conn = await asyncpg.connect(**self.connect_kwargs())
                    setattr(self, attr, conn)
                try:
                    return await conn.fetch(
                        sql, *args, timeout=DEDICATED_QUERY_TIMEOUT_S
                    )
                except (
                    asyncpg.InterfaceError,
                    asyncpg.PostgresConnectionError,
                    OSError,
                ):
                    await _close_quietly(conn)
                    setattr(self, attr, None)
                    if attempt == 2:
                        raise
        return []  # pragma: no cover - loop always returns or raises

    async def _lease_query(self, sql: str, *args: Any) -> list[Any]:
        return await self._dedicated_query(
            "_lease_conn", self._lease_guard(), sql, *args
        )

    def _beat_guard(self) -> asyncio.Lock:
        if self._beat_lock is None:
            self._beat_lock = asyncio.Lock()
        return self._beat_lock

    async def _beat_query(self, sql: str, *args: Any) -> list[Any]:
        return await self._dedicated_query("_beat_conn", self._beat_guard(), sql, *args)

    async def try_lease(self, name: str, holder: str, ttl_s: float) -> bool:
        # The database clock decides expiry, so workers never disagree.
        rows = await self._lease_query(
            f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                VALUES ($1, jsonb_build_object(
                    'holder', $2::text,
                    'exp', extract(epoch FROM clock_timestamp()) + $3::float8), now())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()
                WHERE ({TABLE_KV}.value->>'exp')::float8 < extract(epoch FROM clock_timestamp())
                   OR {TABLE_KV}.value->>'holder' = $2::text
                RETURNING key""",
            f"lease:{name}",
            holder,
            float(ttl_s),
        )
        return bool(rows)

    async def release_lease(self, name: str, holder: str) -> None:
        await self._lease_query(
            f"DELETE FROM {TABLE_KV} WHERE key = $1 AND value->>'holder' = $2::text",
            f"lease:{name}",
            holder,
        )

    # -- doc status: Q1 backlog totals ---------------------------------------------
    async def backlog_totals(self) -> dict[str, Any]:
        # Index-only over the partial index: counts and ages per status.
        rows = await self._read(
            f"""SELECT status, count(*)::bigint AS n, min(updated_at) AS oldest
            FROM {TABLE_DOC_STATUS}
            WHERE status IN {_IN_BACKLOG} AND {_NOT_DUP}
            GROUP BY status"""
        )
        # Telling cancellations apart needs the message, so this one reads
        # the heap, but only for the last day's failures.
        recent = await self._read(
            f"""SELECT count(*) FILTER (WHERE NOT {_CANCELLED})::bigint AS failed_24h,
                count(*) FILTER (WHERE {_CANCELLED})::bigint AS cancelled_24h
            FROM {TABLE_DOC_STATUS}
            WHERE status IN ('failed') AND {_NOT_DUP}
              AND updated_at >= {UTC_NOW_SQL} - interval '24 hours'"""
        )
        return _totals_from_rows(rows, recent[0] if recent else {})

    async def cancelled_count(self) -> int:
        rows = await self._read(
            f"""SELECT count(*)::bigint AS n FROM {TABLE_DOC_STATUS}
            WHERE status IN ('failed') AND {_NOT_DUP} AND {_CANCELLED}"""
        )
        return int(rows[0]["n"]) if rows else 0

    async def dup_count(self) -> int:
        rows = await self._read(
            f"SELECT count(*)::bigint AS n FROM {TABLE_DOC_STATUS} WHERE {_IS_DUP}"
        )
        return int(rows[0]["n"]) if rows else 0

    # -- Q2 backlog page ----------------------------------------------------------
    async def backlog_page(
        self,
        *,
        q,
        has,
        sort,
        descending,
        page,
        stale_minutes,
        email_workspaces,
        paused_workspaces,
    ):
        column = BACKLOG_SORTS.get(sort)
        if column is None:
            raise ValueError(f"Unknown sort: {sort}")
        unknown = set(has) - BACKLOG_FILTERS
        if unknown:
            raise ValueError(f"Unknown filter: {sorted(unknown)}")
        with_owners = await self._has_table(TABLE_OWNERS)
        params: list[Any] = [int(stale_minutes)]
        conditions: list[str] = []
        if q:
            params.append(f"%{like_escape(q)}%")
            match = [f"agg.workspace ILIKE ${len(params)} ESCAPE '\\'"]
            if with_owners:
                match += [
                    f"o.email ILIKE ${len(params)} ESCAPE '\\'",
                    f"o.identity ILIKE ${len(params)} ESCAPE '\\'",
                ]
            conditions.append("(" + " OR ".join(match) + ")")
        if "failed" in has:
            conditions.append("agg.failed > 0")
        if "stuck" in has:
            conditions.append(
                f"agg.oldest_active_at < {UTC_NOW_SQL} - make_interval(mins => $1)"
            )
        if "email" in has:
            params.append(list(email_workspaces))
            conditions.append(f"agg.workspace = ANY(${len(params)}::varchar[])")
        if "paused" in has:
            params.append(list(paused_workspaces))
            conditions.append(f"agg.workspace = ANY(${len(params)}::varchar[])")
        params += [page.limit, page.offset]
        owner_select = (
            "COALESCE(o.email, o.identity) AS owner" if with_owners else "NULL AS owner"
        )
        owner_join = (
            f"LEFT JOIN {TABLE_OWNERS} o ON o.workspace_id = agg.workspace"
            if with_owners
            else ""
        )
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        direction = "DESC" if descending else "ASC"
        rows = await self._read(
            f"""WITH agg AS (
                SELECT workspace,
                    count(*) FILTER (WHERE status = 'pending')::bigint AS pending,
                    count(*) FILTER (WHERE status = 'parsing')::bigint AS parsing,
                    count(*) FILTER (WHERE status = 'analyzing')::bigint AS analyzing,
                    count(*) FILTER (WHERE status = 'processing')::bigint AS processing,
                    count(*) FILTER (WHERE status = 'failed' AND NOT {_CANCELLED})::bigint AS failed,
                    count(*) FILTER (WHERE status = 'failed' AND {_CANCELLED})::bigint AS cancelled,
                    count(*) FILTER (WHERE status IN {_IN_INFLIGHT})::bigint AS inflight,
                    min(updated_at) FILTER (WHERE status IN {_IN_INFLIGHT}) AS oldest_inflight_at,
                    min(updated_at) FILTER (WHERE status IN {_IN_ACTIVE}) AS oldest_active_at,
                    max(updated_at) AS last_activity_at
                FROM {TABLE_DOC_STATUS}
                WHERE status IN {_IN_BACKLOG} AND {_NOT_DUP}
                GROUP BY workspace
            )
            SELECT agg.*, {owner_select},
                (agg.oldest_active_at < {UTC_NOW_SQL} - make_interval(mins => $1)) AS stuck,
                count(*) OVER () AS total
            FROM agg {owner_join}
            {where}
            ORDER BY {column} {direction} NULLS LAST, agg.workspace
            LIMIT ${len(params) - 1} OFFSET ${len(params)}""",
            *params,
        )
        total = int(rows[0]["total"]) if rows else 0
        if not rows and page.offset:
            # Past the last page: the window total is unknown, count separately.
            total = await self._backlog_count(where, owner_join, params[:-2])
        for row in rows:
            row.pop("total", None)
            row["stuck"] = bool(row.get("stuck"))
        return rows, total

    async def _backlog_count(self, where: str, owner_join: str, params: list) -> int:
        rows = await self._read(
            f"""WITH agg AS (
                SELECT workspace,
                    count(*) FILTER (WHERE status = 'failed' AND NOT {_CANCELLED})::bigint AS failed,
                    min(updated_at) FILTER (WHERE status IN {_IN_ACTIVE}) AS oldest_active_at
                FROM {TABLE_DOC_STATUS}
                WHERE status IN {_IN_BACKLOG} AND {_NOT_DUP}
                GROUP BY workspace
            )
            SELECT count(*)::bigint AS n, $1::int AS stale_minutes
            FROM agg {owner_join} {where}""",
            *params,
        )
        return int(rows[0]["n"]) if rows else 0

    # -- Q3 stale docs -------------------------------------------------------------
    async def stale_docs(self, older_than_minutes, page):
        condition = (
            f"status IN {_IN_ACTIVE} AND {_NOT_DUP} "
            f"AND updated_at < {UTC_NOW_SQL} - make_interval(mins => $1)"
        )
        rows = await self._read(
            f"""SELECT workspace, id, status, file_path, track_id, updated_at,
                count(*) OVER () AS total
            FROM {TABLE_DOC_STATUS} WHERE {condition}
            ORDER BY updated_at, workspace, id LIMIT $2 OFFSET $3""",
            int(older_than_minutes),
            page.limit,
            page.offset,
        )
        if rows:
            total = int(rows[0]["total"])
        elif page.offset:
            count = await self._read(
                f"SELECT count(*)::bigint AS n FROM {TABLE_DOC_STATUS} WHERE {condition}",
                int(older_than_minutes),
            )
            total = int(count[0]["n"]) if count else 0
        else:
            total = 0
        for row in rows:
            row.pop("total", None)
        return rows, total

    # -- Q6 live candidates ---------------------------------------------------------
    async def live_candidates(self, limit):
        return await self._read(
            f"""SELECT workspace, max(updated_at) AS last_at, min(updated_at) AS oldest_at,
                count(*) FILTER (WHERE status = 'parsing')::bigint AS parsing,
                count(*) FILTER (WHERE status = 'analyzing')::bigint AS analyzing,
                count(*) FILTER (WHERE status = 'processing')::bigint AS processing
            FROM {TABLE_DOC_STATUS}
            WHERE status IN {_IN_ACTIVE} AND {_NOT_DUP}
            GROUP BY workspace ORDER BY last_at DESC, workspace LIMIT $1""",
            int(limit),
        )

    async def pending_candidates(self, limit):
        rows = await self._read(
            f"""SELECT workspace, max(updated_at) AS last_at FROM {TABLE_DOC_STATUS}
            WHERE status IN ('pending') AND {_NOT_DUP}
            GROUP BY workspace ORDER BY last_at DESC, workspace LIMIT $1""",
            int(limit),
        )
        return [r["workspace"] for r in rows]

    async def paused_candidates(self, days, limit):
        if not await self._has_table(TABLE_USAGE_EVENTS):
            return []
        rows = await self._read(
            f"""SELECT workspace FROM (
                SELECT DISTINCT ON (workspace) workspace, kind, detail, ts
                FROM {TABLE_USAGE_EVENTS}
                WHERE kind IN ('provider_rejection', 'ingestion_resumed')
                  AND workspace IS NOT NULL
                  AND ts >= now() - make_interval(days => $1)
                ORDER BY workspace, ts DESC
            ) latest
            WHERE kind = 'provider_rejection' AND COALESCE(detail->>'paused', 'false') = 'true'
            ORDER BY ts DESC LIMIT $2""",
            int(days),
            int(limit),
        )
        return [r["workspace"] for r in rows]

    # -- Q4 legacy failures ------------------------------------------------------------
    async def legacy_failures(self, hours, group, include_cancelled, limit=50):
        if group not in FAILURE_GROUPS:
            raise ValueError(f"Unknown group: {group}")
        key = _SIGNATURE_SQL if group == "signature" else "workspace"
        window = f"updated_at >= {UTC_NOW_SQL} - make_interval(hours => $1)"
        base = f"status = 'failed' AND {_NOT_DUP} AND {window}"
        groups = await self._read(
            f"""SELECT {key} AS key, count(*)::bigint AS count,
                count(DISTINCT workspace)::bigint AS workspaces,
                min(updated_at) AS first_seen, max(updated_at) AS last_seen,
                (array_agg(left(error_msg, {_SIGNATURE_CHARS}) {_EXAMPLE_ORDER}))[1] AS example,
                (array_agg(file_path {_EXAMPLE_ORDER}))[1] AS example_path
            FROM {TABLE_DOC_STATUS}
            WHERE {base} AND NOT {_CANCELLED} AND {_NO_FAILURE_META}
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT $2""",
            int(hours),
            int(limit),
        )
        cancelled = await self._read(
            f"""SELECT count(*)::bigint AS count, count(DISTINCT workspace)::bigint AS workspaces,
                min(updated_at) AS first_seen, max(updated_at) AS last_seen,
                (array_agg(left(error_msg, {_SIGNATURE_CHARS}) {_EXAMPLE_ORDER}))[1] AS example,
                (array_agg(file_path {_EXAMPLE_ORDER}))[1] AS example_path
            FROM {TABLE_DOC_STATUS} WHERE {base} AND {_CANCELLED}""",
            int(hours),
        )
        summary = cancelled[0] if cancelled else {"count": 0}
        return {
            "groups": groups,
            "cancelled": summary if include_cancelled else {"count": summary["count"]},
        }

    async def failure_docs(self, hours, group, key, page):
        window = f"updated_at >= {UTC_NOW_SQL} - make_interval(hours => $1)"
        base = f"status = 'failed' AND {_NOT_DUP} AND {window}"
        if group == CANCELLED_GROUP_KEY:
            condition, args = f"{base} AND {_CANCELLED}", [int(hours)]
        elif group == "signature":
            condition = (
                f"{base} AND NOT {_CANCELLED} AND {_NO_FAILURE_META} "
                f"AND {_SIGNATURE_SQL} = $2"
            )
            args = [int(hours), key]
        elif group == "workspace":
            condition = (
                f"{base} AND NOT {_CANCELLED} AND {_NO_FAILURE_META} AND workspace = $2"
            )
            args = [int(hours), key]
        else:
            raise ValueError(f"Unknown group: {group}")
        n = len(args)
        rows = await self._read(
            f"""SELECT workspace, id, file_path, track_id, updated_at,
                left(error_msg, {_SIGNATURE_CHARS}) AS error, count(*) OVER () AS total
            FROM {TABLE_DOC_STATUS} WHERE {condition}
            ORDER BY updated_at DESC, workspace, id LIMIT ${n + 1} OFFSET ${n + 2}""",
            *args,
            page.limit,
            page.offset,
        )
        total = int(rows[0]["total"]) if rows else 0
        if not rows and page.offset:
            count = await self._read(
                f"SELECT count(*)::bigint AS n FROM {TABLE_DOC_STATUS} WHERE {condition}",
                *args,
            )
            total = int(count[0]["n"]) if count else 0
        for row in rows:
            row.pop("total", None)
        return rows, total

    # -- one workspace ------------------------------------------------------------------
    async def workspace_counts(self, workspace):
        rows = await self._read(
            f"""SELECT status,
                count(*) FILTER (WHERE {_NOT_DUP} AND NOT (status = 'failed' AND {_CANCELLED}))::bigint AS n,
                count(*) FILTER (WHERE {_NOT_DUP} AND status = 'failed' AND {_CANCELLED})::bigint AS cancelled,
                count(*) FILTER (WHERE {_IS_DUP})::bigint AS dup,
                min(updated_at) FILTER (WHERE {_NOT_DUP}) AS oldest,
                max(updated_at) AS newest
            FROM {TABLE_DOC_STATUS} WHERE workspace = $1 GROUP BY status""",
            workspace,
        )
        return _workspace_counts_from_rows(rows)

    async def workspace_documents(self, workspace, status, page):
        args: list[Any] = [workspace]
        condition = f"workspace = $1 AND {_NOT_DUP}"
        if status:
            args.append(status)
            condition += " AND status = $2"
        n = len(args)
        rows = await self._read(
            f"""SELECT id, status, file_path, track_id, chunks_count, created_at, updated_at,
                left(error_msg, {_SIGNATURE_CHARS}) AS error
            FROM {TABLE_DOC_STATUS} WHERE {condition}
            ORDER BY updated_at DESC NULLS LAST, id LIMIT ${n + 1} OFFSET ${n + 2}""",
            *args,
            page.limit,
            page.offset,
        )
        count = await self._read(
            f"SELECT count(*)::bigint AS n FROM {TABLE_DOC_STATUS} WHERE {condition}",
            *args,
        )
        return rows, int(count[0]["n"]) if count else 0

    async def document(self, workspace, doc_id):
        rows = await self._read(
            f"""SELECT id, status, file_path, track_id, chunks_count, content_length,
                created_at, updated_at, metadata, left(error_msg, 2048) AS error
            FROM {TABLE_DOC_STATUS} WHERE workspace = $1 AND id = $2""",
            workspace,
            doc_id,
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = self._decode(row.get("metadata")) or {}
        return row

    async def owners(self, workspaces):
        if not workspaces or not await self._has_table(TABLE_OWNERS):
            return {}
        rows = await self._read(
            f"""SELECT workspace_id, COALESCE(email, identity) AS owner FROM {TABLE_OWNERS}
            WHERE workspace_id = ANY($1::varchar[])""",
            list(workspaces),
        )
        return {r["workspace_id"]: r["owner"] for r in rows}

    # -- events ---------------------------------------------------------------------------
    async def usage_events(self, start, end, kinds, limit):
        if not kinds or not await self._has_table(TABLE_USAGE_EVENTS):
            return []
        rows = await self._read(
            f"""SELECT ts, kind, workspace, purpose, detail FROM {TABLE_USAGE_EVENTS}
            WHERE ts >= $1 AND ts < $2 AND kind = ANY($3::varchar[])
            ORDER BY ts DESC LIMIT $4""",
            start,
            end,
            list(kinds),
            int(limit),
        )
        for row in rows:
            row["detail"] = self._decode(row.get("detail")) or {}
        return rows

    async def audit_events(self, start, end, limit):
        return await self._read(
            f"""SELECT id, ts, actor, action, workspace, outcome FROM {TABLE_AUDIT}
            WHERE ts >= $1 AND ts < $2 AND phase = 'result'
                AND action NOT IN {sql_in(AUDIT_READ_ACTIONS)}
            ORDER BY ts DESC LIMIT $3""",
            start,
            end,
            int(limit),
        )

    # -- deletion -----------------------------------------------------------------------------
    async def forget_workspace(self, workspace):
        hashed = workspace_hash(workspace)
        tables = await self._refresh_tables(force=True)
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS * 6}"
                )
                await conn.execute(
                    f"""DELETE FROM {TABLE_KV}
                    WHERE key = ANY($1::varchar[]) OR key LIKE ANY($2::text[])""",
                    _workspace_keys(workspace),
                    [like_escape(p) + "%" for p in _workspace_prefixes(workspace)],
                )
                await conn.execute(
                    f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = now()""",
                    deleted_key(workspace),
                    json.dumps({"at": time.time()}),
                )
                for summary in WORKSPACE_KV_SUMMARIES:
                    await conn.execute(
                        f"""UPDATE {TABLE_KV} SET value = jsonb_set(value, '{{items}}',
                            COALESCE((SELECT jsonb_agg(item) FROM jsonb_array_elements(value->'items') item
                                      WHERE item->>'workspace' IS DISTINCT FROM $2), '[]'::jsonb)),
                            updated_at = now()
                        WHERE key = $1 AND jsonb_typeof(value->'items') = 'array'""",
                        summary,
                        workspace,
                    )
                await conn.execute(
                    f"UPDATE {TABLE_AUDIT} SET workspace = $2 WHERE workspace = $1",
                    workspace,
                    hashed,
                )
                await conn.execute(
                    f"UPDATE {TABLE_AUDIT} SET target = $2 WHERE target = $1",
                    workspace,
                    hashed,
                )
                for table in WORKSPACE_ROW_TABLES:
                    if table in tables:
                        await conn.execute(
                            f"DELETE FROM {table} WHERE workspace = $1", workspace
                        )
                if TABLE_ALERTS in tables:
                    await conn.execute(
                        f"""UPDATE {TABLE_ALERTS}
                        SET title = replace(title, $1, $2),
                            scope = jsonb_set(scope, '{{workspace}}', to_jsonb($2::text))
                        WHERE scope->>'workspace' = $1""",
                        workspace,
                        hashed,
                    )
                    # Rules over several workspaces list them in the scope.
                    await conn.execute(
                        f"""UPDATE {TABLE_ALERTS}
                        SET scope = jsonb_set(scope, '{{workspaces}}', (
                            SELECT jsonb_agg(CASE WHEN e = to_jsonb($1::text)
                                THEN to_jsonb($2::text) ELSE e END)
                            FROM jsonb_array_elements(scope->'workspaces') e))
                        WHERE jsonb_typeof(scope->'workspaces') = 'array'
                            AND scope->'workspaces' @> jsonb_build_array($1::text)""",
                        workspace,
                        hashed,
                    )

    # -- workers (heartbeat) ------------------------------------------------------------------
    async def worker_beat(self, row, sample=None):
        await self._beat_query(
            f"""INSERT INTO {TABLE_WORKERS} (worker_id, host, pid, pid_create_time, mode,
                version, started_at, heartbeat_at, leases_held, snapshot)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, now(), $8::text[], $9::jsonb)
            ON CONFLICT (worker_id) DO UPDATE SET
                heartbeat_at = now(), pid = EXCLUDED.pid,
                pid_create_time = EXCLUDED.pid_create_time, mode = EXCLUDED.mode,
                version = EXCLUDED.version, leases_held = EXCLUDED.leases_held,
                snapshot = EXCLUDED.snapshot, exited_at = NULL, superseded = false""",
            row["worker_id"],
            row["host"][:255],
            int(row["pid"]),
            row.get("pid_create_time"),
            row["mode"][:16],
            (row.get("version") or "")[:64] or None,
            row["started_at"],
            list(row.get("leases_held") or []),
            json.dumps(row.get("snapshot") or {}, default=str),
        )
        if sample is not None:
            await self._beat_query(
                f"""INSERT INTO {TABLE_WORKER_SAMPLES}
                    (worker_id, minute, cpu_pct, rss, loop_lag_ms, pool_wait_ms,
                    llm_running, llm_in_flight, llm_queued)
                VALUES ($1::uuid, date_trunc('minute', now()), $2, $3, $4, $5, $6, $7,
                    $8)
                ON CONFLICT (worker_id, minute) DO UPDATE SET
                    cpu_pct = EXCLUDED.cpu_pct, rss = EXCLUDED.rss,
                    loop_lag_ms = EXCLUDED.loop_lag_ms,
                    pool_wait_ms = EXCLUDED.pool_wait_ms,
                    llm_running = EXCLUDED.llm_running,
                    llm_in_flight = EXCLUDED.llm_in_flight,
                    llm_queued = EXCLUDED.llm_queued""",
                row["worker_id"],
                sample.get("cpu_pct"),
                sample.get("rss"),
                sample.get("loop_lag_ms"),
                sample.get("pool_wait_ms"),
                _opt_int(sample.get("llm_running")),
                _opt_int(sample.get("llm_in_flight")),
                _opt_int(sample.get("llm_queued")),
            )

    async def supersede_workers(self, host, worker_id, started_at, stale_s, alive=None):
        where = """host = $1 AND worker_id <> $2::uuid AND started_at < $3
              AND NOT superseded AND exited_at IS NULL
              AND heartbeat_at < now() - make_interval(secs => $4::float8)"""
        args = (host[:255], worker_id, started_at, float(stale_s))
        if alive is None:
            rows = await self._beat_query(
                f"""UPDATE {TABLE_WORKERS} SET superseded = true
                WHERE {where} RETURNING worker_id""",
                *args,
            )
            return len(rows)
        candidates = await self._beat_query(
            f"""SELECT worker_id::text AS worker_id, pid, pid_create_time
            FROM {TABLE_WORKERS} WHERE {where} LIMIT 500""",
            *args,
        )
        ids = [
            r["worker_id"]
            for r in candidates
            if not _still_running(alive, r["pid"], r["pid_create_time"])
        ]
        if not ids:
            return 0
        rows = await self._beat_query(
            f"""UPDATE {TABLE_WORKERS} SET superseded = true
            WHERE {where} AND worker_id = ANY($5::uuid[]) RETURNING worker_id""",
            *args,
            ids,
        )
        return len(rows)

    async def worker_exited(self, worker_id):
        await self._beat_query(
            f"UPDATE {TABLE_WORKERS} SET exited_at = now() WHERE worker_id = $1::uuid",
            worker_id,
        )

    async def cleanup_workers(self, row_hours, sample_days):
        await self._beat_query(
            f"""DELETE FROM {TABLE_WORKERS}
            WHERE heartbeat_at < now() - make_interval(hours => $1::int)""",
            int(row_hours),
        )
        await self._beat_query(
            f"""DELETE FROM {TABLE_WORKER_SAMPLES}
            WHERE minute < now() - make_interval(days => $1::int)""",
            int(sample_days),
        )

    async def workers(self, limit):
        rows = await self._read(
            f"""SELECT worker_id, host, pid, pid_create_time, mode, version, started_at,
                heartbeat_at, exited_at, superseded, leases_held, snapshot,
                extract(epoch FROM (now() - heartbeat_at))::float8 AS heartbeat_age_s
            FROM {TABLE_WORKERS} ORDER BY heartbeat_at DESC LIMIT $1""",
            int(limit),
        )
        for row in rows:
            row["worker_id"] = _uuid_hex(row["worker_id"])
            row["snapshot"] = self._decode(row.get("snapshot")) or {}
            row["leases_held"] = list(row.get("leases_held") or [])
        return rows

    async def worker_history(self, worker_id, hours, bucket_s):
        return await self._read(
            f"""SELECT floor(extract(epoch FROM minute) / $3)::bigint * $3 AS t,
                max(rss)::bigint AS rss, avg(cpu_pct)::float8 AS cpu,
                max(loop_lag_ms)::int AS loop_lag, max(pool_wait_ms)::int AS pool_wait
            FROM {TABLE_WORKER_SAMPLES}
            WHERE worker_id = $1::uuid AND minute >= now() - make_interval(hours => $2::int)
            GROUP BY 1 ORDER BY 1""",
            worker_id,
            int(hours),
            int(bucket_s),
        )

    def pool_stats(self):
        pool = getattr(self._db, "pool", None) if self._db is not None else None
        if pool is None:
            return None
        try:
            return {
                "size": pool.get_size(),
                "idle": pool.get_idle_size(),
                "max": pool.get_max_size(),
                "min": pool.get_min_size(),
            }
        except Exception:
            return None

    async def pool_wait_ms(self, timeout_s):
        pool = getattr(self._db, "pool", None) if self._db is not None else None
        if pool is None:
            return None
        started = time.perf_counter()
        async with pool.acquire(timeout=timeout_s):
            return (time.perf_counter() - started) * 1000.0

    # -- dependency probes ----------------------------------------------------------------------
    async def postgres_health(self, *, extras, timeout_s):
        import asyncpg

        kwargs = self.connect_kwargs()
        kwargs["timeout"] = timeout_s
        started = time.perf_counter()
        conn = await asyncpg.connect(**kwargs)
        connect_ms = (time.perf_counter() - started) * 1000.0
        try:
            started = time.perf_counter()
            await conn.fetchval("SELECT 1", timeout=timeout_s)
            out: dict[str, Any] = {
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "connect_ms": connect_ms,
            }
            if not extras:
                return out
            await conn.execute(
                f"SET statement_timeout = {int(min(timeout_s, 10) * 1000)}"
            )
            summary = await conn.fetchrow(
                """SELECT current_setting('max_connections')::int AS max_connections,
                    (SELECT count(*) FROM pg_stat_activity
                     WHERE backend_type = 'client backend')::int AS connections,
                    pg_database_size(current_database())::bigint AS db_size_bytes,
                    current_setting('server_version') AS server_version""",
                timeout=timeout_s,
            )
            out.update(dict(summary))
            states = await conn.fetch(
                """SELECT COALESCE(state, 'unknown') AS state, count(*)::int AS n
                FROM pg_stat_activity WHERE backend_type = 'client backend'
                GROUP BY 1""",
                timeout=timeout_s,
            )
            out["states"] = {r["state"]: r["n"] for r in states}
            # Cumulative counters; the probe turns them into 5-minute deltas.
            db_stats = await conn.fetchrow(
                """SELECT xact_commit::bigint AS xact_commit,
                    xact_rollback::bigint AS xact_rollback,
                    deadlocks::bigint AS deadlocks,
                    temp_bytes::bigint AS temp_bytes,
                    blks_read::bigint AS blks_read,
                    blks_hit::bigint AS blks_hit
                FROM pg_stat_database WHERE datname = current_database()""",
                timeout=timeout_s,
            )
            out["db_stats"] = dict(db_stats) if db_stats is not None else None
            # Blocked backends and the ones blocking them: durations, statement
            # type, a fingerprint and the length of the query. The query text
            # itself never leaves the database.
            blocking = await conn.fetch(
                """WITH blocked AS (
                    SELECT pid, pg_blocking_pids(pid) AS blockers
                    FROM pg_stat_activity
                    WHERE cardinality(pg_blocking_pids(pid)) > 0
                    LIMIT 50
                ), involved AS (
                    SELECT pid FROM blocked
                    UNION SELECT unnest(blockers) FROM blocked
                )
                SELECT a.pid, COALESCE(b.blockers, '{}'::int[]) AS blocked_by,
                    a.state, a.wait_event_type, a.wait_event,
                    extract(epoch FROM (now() - COALESCE(
                        a.query_start, a.xact_start, a.backend_start)))::float8
                        AS duration_s,
                    left(upper(substring(ltrim(a.query) FROM '^[A-Za-z]+')), 16)
                        AS statement_type,
                    left(md5(regexp_replace(COALESCE(a.query, ''),
                        '[0-9]+|''[^'']*''', '?', 'g')), 16) AS fingerprint,
                    length(a.query) AS query_length
                FROM pg_stat_activity a
                JOIN involved i ON i.pid = a.pid
                LEFT JOIN blocked b ON b.pid = a.pid
                ORDER BY duration_s DESC NULLS LAST
                LIMIT 40""",
                timeout=timeout_s,
            )
            out["blocking"] = [dict(r) for r in blocking]
            return out
        finally:
            await _close_quietly(conn)

    async def probe_write(self, rows):
        if not rows:
            return
        values = [
            (
                r["dependency"][:32],
                r["status"][:8],
                int(r["latency_ms"]) if r.get("latency_ms") is not None else None,
                (r.get("error_class") or None) and r["error_class"][:64],
                json.dumps(r.get("detail") or {}, default=str),
            )
            for r in rows
        ]
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                await conn.executemany(
                    f"""INSERT INTO {TABLE_PROBES}
                        (ts, dependency, status, latency_ms, error_class, detail)
                    VALUES (clock_timestamp(), $1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT DO NOTHING""",
                    values,
                )

    async def probe_strips(self, hours, slot_s):
        return await self._read(
            f"""SELECT dependency, floor(extract(epoch FROM ts) / $2)::bigint AS slot,
                count(*) FILTER (WHERE status = 'ok')::int AS ok,
                count(*) FILTER (WHERE status = 'warn')::int AS warn,
                count(*) FILTER (WHERE status = 'down')::int AS down,
                count(*)::int AS total
            FROM {TABLE_PROBES}
            WHERE ts >= now() - make_interval(hours => $1::int)
            GROUP BY 1, 2""",
            int(hours),
            int(slot_s),
        )

    async def probe_recent(self, dependency, limit):
        rows = await self._read(
            f"""SELECT ts, status, latency_ms, error_class, detail FROM {TABLE_PROBES}
            WHERE dependency = $1 ORDER BY ts DESC LIMIT $2""",
            dependency,
            int(limit),
        )
        for row in rows:
            row["detail"] = self._decode(row.get("detail")) or {}
        return rows

    async def probe_series(self, dependency, hours, bucket_s):
        return await self._read(
            f"""SELECT floor(extract(epoch FROM ts) / $3)::bigint * $3 AS t,
                avg(latency_ms)::float8 AS avg_ms, max(latency_ms)::int AS max_ms,
                count(*) FILTER (WHERE status = 'down')::int AS down,
                count(*) FILTER (WHERE status = 'warn')::int AS warn,
                count(*)::int AS total
            FROM {TABLE_PROBES}
            WHERE dependency = $1 AND ts >= now() - make_interval(hours => $2::int)
            GROUP BY 1 ORDER BY 1""",
            dependency,
            int(hours),
            int(bucket_s),
        )

    async def probe_error_classes(self, dependency, hours):
        return await self._read(
            f"""SELECT error_class AS key, count(*)::int AS count FROM {TABLE_PROBES}
            WHERE dependency = $1 AND error_class IS NOT NULL
              AND ts >= now() - make_interval(hours => $2::int)
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20""",
            dependency,
            int(hours),
        )

    async def probe_cleanup(self, days):
        await self._write(
            f"DELETE FROM {TABLE_PROBES} WHERE ts < now() - make_interval(days => $1::int)",
            int(days),
        )

    # -- run tracking -----------------------------------------------------------------------------
    async def kv_merge_existing(self, patches):
        if not patches:
            return
        rows = [(k, json.dumps(v, default=str)) for k, v in patches.items()]
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                await conn.executemany(
                    f"""UPDATE {TABLE_KV} SET value = value || $2::jsonb, updated_at = now()
                    WHERE key = $1 AND jsonb_typeof(value) = 'object'""",
                    rows,
                )

    async def runs_write(self, rows):
        if not rows:
            return
        values = [
            (
                r["run_id"],
                pg_text(r["workspace"], 255) or "",
                r["kind"][:24],
                pg_text(r.get("trigger"), 32) or "other",
                r["status"][:16],
                r.get("worker_id"),
                r.get("pid"),
                float(r["started_at"]),
                float(r["heartbeat_at"]),
                float(r["progress_at"]),
                None if r.get("finished_at") is None else float(r["finished_at"]),
                int(r.get("docs_total") or 0),
                int(r.get("docs_done") or 0),
                int(r.get("docs_failed") or 0),
                pg_text(r.get("stage"), 16),
                pg_text(r.get("label"), 128),
                pg_text(r.get("error_kind"), 32),
                pg_text(r.get("error_summary"), 512),
                pg_text(r.get("request_id"), 64),
                pg_text(r.get("host"), 255),
            )
            for r in rows
        ]
        final = sql_in(RUN_FINAL_STATUSES)
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                await conn.executemany(
                    f"""INSERT INTO {TABLE_RUNS} AS r (run_id, workspace, kind, trigger,
                        status, worker_id, pid, started_at, heartbeat_at, progress_at,
                        finished_at, docs_total, docs_done, docs_failed, stage, label,
                        error_kind, error_summary, request_id, host)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6::uuid, $7,
                        to_timestamp($8::float8), to_timestamp($9::float8),
                        to_timestamp($10::float8), to_timestamp($11::float8),
                        $12, $13, $14, $15, $16, $17, $18, $19, $20)
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        was_orphaned = r.was_orphaned OR r.status = 'orphaned',
                        trigger = EXCLUDED.trigger,
                        heartbeat_at = GREATEST(r.heartbeat_at, EXCLUDED.heartbeat_at),
                        progress_at = GREATEST(r.progress_at, EXCLUDED.progress_at),
                        finished_at = EXCLUDED.finished_at,
                        docs_total = EXCLUDED.docs_total,
                        docs_done = EXCLUDED.docs_done,
                        docs_failed = EXCLUDED.docs_failed,
                        stage = EXCLUDED.stage,
                        label = COALESCE(EXCLUDED.label, r.label),
                        error_kind = EXCLUDED.error_kind,
                        error_summary = EXCLUDED.error_summary,
                        host = COALESCE(r.host, EXCLUDED.host)
                    WHERE r.status NOT IN {final}""",
                    values,
                )

    async def doc_failures_write(self, rows):
        if not rows:
            return
        values = [
            (
                float(r["ts"]),
                pg_text(r["workspace"], 255) or "",
                pg_text(r["doc_id"], 255) or "",
                pg_text(r.get("track_id"), 255),
                r.get("run_id"),
                pg_text(r["stage"], 16) or "other",
                pg_text(r["error_kind"], 32) or "unknown",
                r.get("http_status"),
                pg_text(r.get("exc_type"), 128),
                pg_text(r.get("message"), 1024),
                int(r.get("attempt") or 1),
                pg_text(r.get("model"), 255),
            )
            for r in rows
        ]
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                await conn.executemany(
                    f"""INSERT INTO {TABLE_DOC_FAILURES} (ts, workspace, doc_id, track_id,
                        run_id, stage, error_kind, http_status, exc_type, message,
                        attempt, model)
                    VALUES (to_timestamp($1::float8), $2, $3, $4, $5::uuid, $6, $7,
                        $8, $9, $10, $11, $12)""",
                    values,
                )

    async def runs_orphan(self, run_ids, confirmed_by=None):
        if not run_ids:
            return []
        ids = list(run_ids)
        how = [pg_text((confirmed_by or {}).get(run_id), 16) for run_id in ids]
        rows = await self._write(
            f"""UPDATE {TABLE_RUNS} AS r
            SET status = 'orphaned', error_summary = $3, orphaned_by = u.how
            FROM unnest($1::uuid[], $2::varchar[]) AS u(id, how)
            WHERE r.run_id = u.id AND r.status = 'running'
            RETURNING r.run_id""",
            ids,
            how,
            ORPHAN_SUMMARY,
        )
        return [_uuid_hex(r["run_id"]) for r in rows]

    async def runs_cleanup(self, run_days, failure_days):
        for table, column, days in (
            (TABLE_RUNS, "started_at", run_days),
            (TABLE_DOC_FAILURES, "ts", failure_days),
        ):
            key = "run_id" if table == TABLE_RUNS else "id"
            # Batched so retention never holds long locks on a busy table.
            while True:
                rows = await self._write(
                    f"""DELETE FROM {table} WHERE {key} IN (
                        SELECT {key} FROM {table}
                        WHERE {column} < now() - make_interval(days => $1::int)
                        LIMIT {RETENTION_BATCH})
                    RETURNING 1""",
                    int(days),
                )
                if len(rows) < RETENTION_BATCH:
                    break
        await self._write(
            f"""DELETE FROM {TABLE_KV} WHERE key LIKE '{DELETED_KV_PREFIX}%'
            AND updated_at < now() - make_interval(days => $1::int)""",
            DELETED_KEEP_DAYS,
        )

    _RUN_COLUMNS = """r.run_id, r.workspace, r.kind, r.trigger, r.status, r.was_orphaned,
        r.worker_id, r.pid, r.started_at, r.heartbeat_at, r.progress_at, r.finished_at,
        r.docs_total, r.docs_done, r.docs_failed, r.stage, r.label, r.error_kind,
        r.error_summary, r.request_id, r.host AS run_host, r.orphaned_by,
        extract(epoch FROM (now() - r.heartbeat_at))::float8 AS heartbeat_age_s,
        extract(epoch FROM (now() - r.progress_at))::float8 AS progress_age_s"""
    _WORKER_COLUMNS = """w.host AS worker_host, w.pid_create_time AS worker_pid_create_time,
        w.started_at AS worker_started_at, w.superseded AS worker_superseded,
        w.exited_at AS worker_exited_at,
        extract(epoch FROM (now() - w.heartbeat_at))::float8 AS worker_heartbeat_age_s"""

    @staticmethod
    def _run_row(row: dict[str, Any]) -> dict[str, Any]:
        row["run_id"] = _uuid_hex(row["run_id"])
        if row.get("worker_id") is not None:
            row["worker_id"] = _uuid_hex(row["worker_id"])
        return row

    async def running_runs(self, limit):
        rows = await self._read(
            f"""SELECT {self._RUN_COLUMNS}, {self._WORKER_COLUMNS}
            FROM {TABLE_RUNS} r LEFT JOIN {TABLE_WORKERS} w ON w.worker_id = r.worker_id
            WHERE r.status = 'running'
            ORDER BY r.started_at DESC LIMIT $1""",
            int(limit),
        )
        return [self._run_row(r) for r in rows]

    async def orphaned_runs(self, hours, limit):
        rows = await self._read(
            f"""SELECT {self._RUN_COLUMNS}, {self._WORKER_COLUMNS}
            FROM {TABLE_RUNS} r LEFT JOIN {TABLE_WORKERS} w ON w.worker_id = r.worker_id
            WHERE r.started_at >= now() - make_interval(hours => $1::int)
              AND r.status = 'orphaned'
            ORDER BY r.started_at DESC LIMIT $2""",
            int(hours),
            int(limit),
        )
        return [self._run_row(r) for r in rows]

    async def runs_page(self, *, workspace, kind, status, before, limit):
        conditions: list[str] = []
        params: list[Any] = []
        if workspace:
            params.append(workspace)
            conditions.append(f"r.workspace = ${len(params)}")
        if kind:
            params.append(kind)
            conditions.append(f"r.kind = ${len(params)}")
        if status:
            params.append(status)
            conditions.append(f"r.status = ${len(params)}")
        if before is not None:
            params += [before[0], before[1]]
            conditions.append(
                f"(r.started_at, r.run_id) < (${len(params) - 1}, ${len(params)}::uuid)"
            )
        params.append(int(limit))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._read(
            f"""SELECT {self._RUN_COLUMNS}, {self._WORKER_COLUMNS}
            FROM {TABLE_RUNS} r LEFT JOIN {TABLE_WORKERS} w ON w.worker_id = r.worker_id
            {where}
            ORDER BY r.started_at DESC, r.run_id DESC LIMIT ${len(params)}""",
            *params,
        )
        return [self._run_row(r) for r in rows]

    async def run_get(self, run_id):
        rows = await self._read(
            f"""SELECT {self._RUN_COLUMNS}, {self._WORKER_COLUMNS}
            FROM {TABLE_RUNS} r LEFT JOIN {TABLE_WORKERS} w ON w.worker_id = r.worker_id
            WHERE r.run_id = $1::uuid""",
            run_id,
        )
        return self._run_row(rows[0]) if rows else None

    _FAILURE_COLUMNS = """id, ts, workspace, doc_id, track_id, run_id, stage, error_kind,
        http_status, exc_type, message, attempt, model"""

    @staticmethod
    def _failure_row(row: dict[str, Any]) -> dict[str, Any]:
        if row.get("run_id") is not None:
            row["run_id"] = _uuid_hex(row["run_id"])
        return row

    async def run_failures(self, run_id, limit):
        rows = await self._read(
            f"""SELECT {self._FAILURE_COLUMNS} FROM {TABLE_DOC_FAILURES}
            WHERE run_id = $1::uuid ORDER BY ts DESC, id DESC LIMIT $2""",
            run_id,
            int(limit),
        )
        return [self._failure_row(r) for r in rows]

    async def last_runs(self, workspaces):
        if not workspaces:
            return {}
        rows = await self._read(
            f"""SELECT DISTINCT ON (r.workspace) {self._RUN_COLUMNS}
            FROM {TABLE_RUNS} r WHERE r.workspace = ANY($1::varchar[])
            ORDER BY r.workspace, r.started_at DESC""",
            list(workspaces),
        )
        return {r["workspace"]: self._run_row(r) for r in rows}

    async def failure_groups(self, hours, group, include_cancelled, limit=50):
        column = DURABLE_GROUP_COLUMNS.get(group)
        if column is None:
            raise ValueError(f"Unknown group: {group}")
        window = "ts >= now() - make_interval(hours => $1::int)"
        groups = await self._read(
            f"""SELECT {column} AS key, count(*)::bigint AS count,
                count(DISTINCT (workspace, doc_id))::bigint AS docs,
                count(DISTINCT workspace)::bigint AS workspaces,
                min(ts) AS first_seen, max(ts) AS last_seen,
                (array_agg(message ORDER BY ts DESC, id DESC))[1] AS example,
                (array_agg(error_kind ORDER BY ts DESC, id DESC))[1] AS example_kind,
                (array_agg(stage ORDER BY ts DESC, id DESC))[1] AS example_stage
            FROM {TABLE_DOC_FAILURES}
            WHERE {window} AND error_kind <> 'cancelled'
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT $2""",
            int(hours),
            int(limit),
        )
        cancelled = await self._read(
            f"""SELECT count(*)::bigint AS count,
                count(DISTINCT (workspace, doc_id))::bigint AS docs,
                count(DISTINCT workspace)::bigint AS workspaces,
                min(ts) AS first_seen, max(ts) AS last_seen,
                (array_agg(message ORDER BY ts DESC, id DESC))[1] AS example
            FROM {TABLE_DOC_FAILURES}
            WHERE {window} AND error_kind = 'cancelled'""",
            int(hours),
        )
        summary = cancelled[0] if cancelled else {"count": 0}
        return {
            "groups": groups,
            "cancelled": summary if include_cancelled else {"count": summary["count"]},
        }

    async def failure_group_docs(self, hours, group, key, page):
        column = DURABLE_GROUP_COLUMNS.get(group)
        if column is None:
            raise ValueError(f"Unknown group: {group}")
        condition = f"ts >= now() - make_interval(hours => $1::int) AND {column} = $2"
        if not (group == "error_kind" and key == CANCELLED_GROUP_KEY):
            condition += " AND error_kind <> 'cancelled'"
        with_docs = self.doc_status_ready and await self._has_table(TABLE_DOC_STATUS)
        doc_select = (
            "ds.file_path, ds.status AS current_status"
            if with_docs
            else "NULL AS file_path, NULL AS current_status"
        )
        doc_join = (
            f"LEFT JOIN {TABLE_DOC_STATUS} ds ON ds.workspace = p.workspace AND ds.id = p.doc_id"
            if with_docs
            else ""
        )
        rows = await self._read(
            f"""WITH latest AS (
                SELECT DISTINCT ON (workspace, doc_id) {self._FAILURE_COLUMNS}
                FROM {TABLE_DOC_FAILURES} WHERE {condition}
                ORDER BY workspace, doc_id, ts DESC, id DESC
            ), page AS (
                SELECT latest.*, count(*) OVER () AS total FROM latest
                ORDER BY ts DESC, workspace, doc_id LIMIT $3 OFFSET $4
            )
            SELECT p.*, {doc_select} FROM page p {doc_join}
            ORDER BY p.ts DESC, p.workspace, p.doc_id""",
            int(hours),
            key,
            page.limit,
            page.offset,
        )
        total = int(rows[0]["total"]) if rows else 0
        if not rows and page.offset:
            count = await self._read(
                f"""SELECT count(*)::bigint AS n FROM (SELECT DISTINCT workspace, doc_id
                FROM {TABLE_DOC_FAILURES} WHERE {condition}) d""",
                int(hours),
                key,
            )
            total = int(count[0]["n"]) if count else 0
        for row in rows:
            row.pop("total", None)
            self._failure_row(row)
        return rows, total

    async def failures_by_kind(self, workspace, hours):
        return await self._read(
            f"""SELECT error_kind AS key, count(*)::bigint AS count,
                count(DISTINCT doc_id)::bigint AS docs, max(ts) AS last_seen
            FROM {TABLE_DOC_FAILURES}
            WHERE workspace = $1 AND ts >= now() - make_interval(hours => $2::int)
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20""",
            workspace,
            int(hours),
        )

    async def doc_failures(self, workspace, doc_id, limit):
        rows = await self._read(
            f"""SELECT {self._FAILURE_COLUMNS} FROM {TABLE_DOC_FAILURES}
            WHERE workspace = $1 AND doc_id = $2 ORDER BY ts DESC, id DESC LIMIT $3""",
            workspace,
            doc_id,
            int(limit),
        )
        return [self._failure_row(r) for r in rows]

    async def open_pauses(self, limit):
        rows = await self._read(
            f"""SELECT key, value FROM {TABLE_KV}
            WHERE key LIKE 'pause:%' AND jsonb_typeof(value) = 'object'
              AND COALESCE(value->>'cleared_at', '') = ''
            ORDER BY key LIMIT $1""",
            int(limit),
        )
        out = []
        for row in rows:
            value = self._decode(row["value"])
            if isinstance(value, dict):
                out.append({**value, "key": row["key"]})
        return out

    async def pauses_page(self, limit, offset):
        rows = await self._read(
            f"""SELECT key, value FROM {TABLE_KV}
            WHERE key LIKE 'pause:%' AND jsonb_typeof(value) = 'object'
              AND COALESCE(value->>'cleared_at', '') = ''
            ORDER BY CASE WHEN jsonb_typeof(value->'since') = 'number'
                     THEN (value->>'since')::float8 ELSE 0 END,
                value->>'workspace', key
            LIMIT $1 OFFSET $2""",
            int(limit),
            int(offset),
        )
        out = []
        for row in rows:
            value = self._decode(row["value"])
            if isinstance(value, dict):
                out.append({**value, "key": row["key"]})
        return out

    async def pause_counts(self):
        rows = await self._read(
            f"""SELECT COALESCE(value->>'slot', 'unknown') AS slot,
                COALESCE(value->>'kind', 'unknown') AS kind,
                count(*)::bigint AS count
            FROM {TABLE_KV}
            WHERE key LIKE 'pause:%' AND jsonb_typeof(value) = 'object'
              AND COALESCE(value->>'cleared_at', '') = ''
            GROUP BY 1, 2"""
        )
        workspaces = await self._read(
            f"""SELECT count(DISTINCT value->>'workspace')::bigint AS n
            FROM {TABLE_KV}
            WHERE key LIKE 'pause:%' AND jsonb_typeof(value) = 'object'
              AND COALESCE(value->>'cleared_at', '') = ''"""
        )
        return _pause_counts(rows, int(workspaces[0]["n"]) if workspaces else 0)

    async def run_events(self, start, end, limit):
        # Orphaned runs are marked at their last sign of life.
        rows = await self._read(
            f"""SELECT run_id, workspace, kind, status, was_orphaned, heartbeat_at AS ts
            FROM {TABLE_RUNS}
            WHERE started_at >= $1::timestamptz - interval '1 day'
              AND started_at < $2 AND heartbeat_at >= $1 AND heartbeat_at < $2
              AND (status = 'orphaned' OR was_orphaned)
            ORDER BY heartbeat_at DESC LIMIT $3""",
            start,
            end,
            int(limit),
        )
        for row in rows:
            row["run_id"] = _uuid_hex(row["run_id"])
        return rows

    # -- audit ---------------------------------------------------------------------------
    async def audit_write(self, row):
        rows = await self._write(
            f"""INSERT INTO {TABLE_AUDIT} (actor, auth_kind, impersonated_by, action,
                workspace, target, params, reason, phase, outcome, request_id,
                client_ip_hash)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
            RETURNING id""",
            *_audit_values(row),
        )
        return int(rows[0]["id"])

    async def audit_page(self, f, before, limit):
        # Served by idx_ops_audit_ts (ts DESC), or by idx_ops_audit_actor,
        # idx_ops_audit_action or idx_ops_audit_workspace ((col, ts DESC))
        # when that column is filtered; ties on ts by id.
        args: list[Any] = [f.start, f.end]
        conditions = ["ts >= $1", "ts < $2"]
        for column in ("actor", "action", "workspace", "phase"):
            value = getattr(f, column)
            if value is not None:
                args.append(value)
                conditions.append(f"{column} = ${len(args)}")
        if before is not None:
            args.extend(before)
            conditions.append(f"(ts, id) < (${len(args) - 1}, ${len(args)})")
        args.append(int(limit))
        rows = await self._read(
            f"""SELECT id, ts, actor, auth_kind, impersonated_by, action, workspace,
                target, params, reason, phase, outcome, request_id
            FROM {TABLE_AUDIT} WHERE {" AND ".join(conditions)}
            ORDER BY ts DESC, id DESC LIMIT ${len(args)}""",
            *args,
        )
        for row in rows:
            row["params"] = self._decode(row.get("params")) or {}
        return rows

    # -- admin actions --------------------------------------------------------------
    async def claim_action(self, key, holder, ttl_s, info):
        value = json.dumps({**info, "holder": holder}, default=str)
        # The database clock decides expiry, like the loop leases.
        rows = await self._write(
            f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                VALUES ($1, $2::jsonb || jsonb_build_object(
                    'exp', extract(epoch FROM clock_timestamp()) + $3::float8), now())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()
                WHERE COALESCE(({TABLE_KV}.value->>'exp')::float8, 0)
                        < extract(epoch FROM clock_timestamp())
                   OR {TABLE_KV}.value->>'holder' = $4::text
                RETURNING key""",
            key,
            value,
            float(ttl_s),
            holder,
        )
        if rows:
            return True, None
        current = await self.kv_get(key, None)
        return False, current if isinstance(current, dict) else None

    async def release_action(self, key, holder):
        await self._write(
            f"DELETE FROM {TABLE_KV} WHERE key = $1 AND value->>'holder' = $2::text",
            key,
            holder,
        )

    async def rate_hit(self, key, limit, window_s, now):
        rows = await self._write(
            f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                VALUES ($1, jsonb_build_object('start', $2::float8, 'n', 1), now())
                ON CONFLICT (key) DO UPDATE
                SET value = CASE
                        WHEN COALESCE(({TABLE_KV}.value->>'start')::float8, 0)
                             <= $2::float8 - $3::float8
                        THEN jsonb_build_object('start', $2::float8, 'n', 1)
                        ELSE jsonb_build_object(
                            'start', ({TABLE_KV}.value->>'start')::float8,
                            'n', COALESCE(({TABLE_KV}.value->>'n')::int, 0) + 1)
                    END,
                    updated_at = now()
                RETURNING (value->>'n')::int AS n, (value->>'start')::float8 AS start""",
            key,
            float(now),
            float(window_s),
        )
        n, start = int(rows[0]["n"]), float(rows[0]["start"])
        return n <= limit, max(0.0, start + window_s - now)

    async def kv_put_if_older(self, key, value, older_than):
        rows = await self._write(
            f"""INSERT INTO {TABLE_KV} (key, value, updated_at)
                VALUES ($1, $2::jsonb, now())
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()
                WHERE jsonb_typeof({TABLE_KV}.value) <> 'object'
                   OR COALESCE(({TABLE_KV}.value->>'at')::float8, 0) <= $3::float8
                RETURNING key""",
            key,
            json.dumps(value, default=str),
            float(older_than),
        )
        return bool(rows)

    # -- alerts ------------------------------------------------------------------------------
    _ALERT_COLUMNS = """id, rule, fingerprint, severity, state, title, scope, link, runbook,
        first_seen, last_seen, resolved_at, acked_by, acked_at, ack_note, last_emailed_at"""

    def _alert_row(self, row: dict[str, Any]) -> dict[str, Any]:
        row["id"] = int(row["id"])
        row["scope"] = self._decode(row.get("scope")) or {}
        return row

    async def alerts_open(self, limit):
        rows = await self._read(
            f"""SELECT {self._ALERT_COLUMNS} FROM {TABLE_ALERTS}
            WHERE resolved_at IS NULL ORDER BY id DESC LIMIT $1""",
            int(limit),
        )
        return [self._alert_row(r) for r in rows]

    async def alerts_upsert(self, items):
        if not items:
            return []
        columns: list[list[Any]] = [[] for _ in range(7)]
        for item in items:
            for index, value in enumerate(
                (
                    item["rule"],
                    item["fingerprint"],
                    item["severity"],
                    item["title"],
                    json.dumps(item.get("scope") or {}, default=str),
                    item.get("link"),
                    item.get("runbook"),
                )
            ):
                columns[index].append(value)
        # ON CONFLICT names the partial unique index's predicate, so a
        # concurrent insert of the same fingerprint waits and then touches
        # the row the other one created instead of adding a second open row.
        rows = await self._write(
            f"""INSERT INTO {TABLE_ALERTS} (rule, fingerprint, severity, state, title,
                scope, link, runbook, first_seen, last_seen)
            SELECT u.rule, u.fingerprint, u.severity, 'firing', u.title, u.scope::jsonb,
                u.link, u.runbook, now(), now()
            FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::text[],
                        $6::text[], $7::text[])
                AS u(rule, fingerprint, severity, title, scope, link, runbook)
            ON CONFLICT (fingerprint) WHERE resolved_at IS NULL DO UPDATE
            SET severity = EXCLUDED.severity, title = EXCLUDED.title,
                scope = EXCLUDED.scope, link = EXCLUDED.link,
                runbook = EXCLUDED.runbook, last_seen = EXCLUDED.last_seen
            RETURNING id, fingerprint, (xmax = 0) AS inserted""",
            *columns,
        )
        return [
            {
                "id": int(r["id"]),
                "fingerprint": r["fingerprint"],
                "inserted": bool(r["inserted"]),
            }
            for r in rows
        ]

    async def alerts_resolve(self, fingerprints):
        if not fingerprints:
            return []
        rows = await self._write(
            f"""UPDATE {TABLE_ALERTS} SET state = 'resolved', resolved_at = now()
            WHERE resolved_at IS NULL AND fingerprint = ANY($1::text[])
            RETURNING {self._ALERT_COLUMNS}""",
            list(fingerprints),
        )
        return [self._alert_row(dict(r)) for r in rows]

    async def alerts_page(self, state, before, limit):
        conditions: list[str] = []
        params: list[Any] = []
        if state == "open":
            conditions.append("resolved_at IS NULL")
        elif state == "resolved":
            conditions.append("resolved_at IS NOT NULL")
        if before is not None:
            params.append(int(before))
            conditions.append(f"id < ${len(params)}")
        params.append(int(limit))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._read(
            f"""SELECT {self._ALERT_COLUMNS} FROM {TABLE_ALERTS} {where}
            ORDER BY id DESC LIMIT ${len(params)}""",
            *params,
        )
        return [self._alert_row(r) for r in rows]

    async def alert_get(self, alert_id):
        rows = await self._read(
            f"SELECT {self._ALERT_COLUMNS} FROM {TABLE_ALERTS} WHERE id = $1",
            int(alert_id),
        )
        return self._alert_row(rows[0]) if rows else None

    async def alerts_history(self, fingerprint, limit):
        rows = await self._read(
            f"""SELECT {self._ALERT_COLUMNS} FROM {TABLE_ALERTS}
            WHERE fingerprint = $1 ORDER BY id DESC LIMIT $2""",
            fingerprint,
            int(limit),
        )
        return [self._alert_row(r) for r in rows]

    async def alert_ack(self, alert_id, actor, note):
        rows = await self._write(
            f"""UPDATE {TABLE_ALERTS} SET acked_by = $2, acked_at = now(), ack_note = $3
            WHERE id = $1 AND resolved_at IS NULL AND acked_at IS NULL
            RETURNING {self._ALERT_COLUMNS}""",
            int(alert_id),
            str(actor)[:255],
            note[:512] if note else None,
        )
        if rows:
            return "acked", self._alert_row(dict(rows[0]))
        row = await self.alert_get(alert_id)
        if row is None:
            return "not_found", None
        if row.get("resolved_at") is not None:
            return "resolved", row
        return "already_acked", row

    async def alert_claim_email(self, alert_id, kind, repeat_s):
        if kind == "resolve":
            condition = """resolved_at IS NOT NULL AND last_emailed_at IS NOT NULL
                AND last_emailed_at < resolved_at"""
            args: list[Any] = [int(alert_id)]
        else:
            condition = """resolved_at IS NULL AND (last_emailed_at IS NULL
                OR last_emailed_at < now() - make_interval(secs => $2::float8))"""
            args = [int(alert_id), float(repeat_s)]
        rows = await self._write(
            f"""UPDATE {TABLE_ALERTS} SET last_emailed_at = clock_timestamp()
            WHERE id = $1 AND {condition} RETURNING last_emailed_at""",
            *args,
        )
        return rows[0]["last_emailed_at"] if rows else None

    async def alert_unclaim_email(self, alert_id, claimed, previous):
        await self._write(
            f"""UPDATE {TABLE_ALERTS} SET last_emailed_at = $3
            WHERE id = $1 AND last_emailed_at = $2""",
            int(alert_id),
            claimed,
            previous,
        )

    async def alerts_window(self, start, end, limit):
        rows = await self._read(
            f"""SELECT {self._ALERT_COLUMNS} FROM {TABLE_ALERTS}
            WHERE last_seen >= $1 AND first_seen < $2
            ORDER BY first_seen DESC, id DESC LIMIT $3""",
            start,
            end,
            int(limit),
        )
        return [self._alert_row(r) for r in rows]

    async def alerts_cleanup(self, days):
        deleted = 0
        for _ in range(LOG_RETENTION_MAX_BATCHES):
            rows = await self._write(
                f"""DELETE FROM {TABLE_ALERTS} WHERE id IN (
                    SELECT id FROM {TABLE_ALERTS}
                    WHERE resolved_at IS NOT NULL
                        AND resolved_at < now() - make_interval(days => $1::int)
                    LIMIT {RETENTION_BATCH}) RETURNING 1""",
                int(days),
            )
            deleted += len(rows)
            if len(rows) < RETENTION_BATCH:
                break
        return deleted

    # -- logs ----------------------------------------------------------------------------
    async def logs_write(self, rows):
        if not rows:
            return
        records = [_log_record(r) for r in rows]
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                # Unquoted DDL names fold to lower case; COPY quotes its target.
                await conn.copy_records_to_table(
                    TABLE_LOGS.lower(), records=records, columns=list(LOG_COLUMNS)
                )

    _LOG_SELECT = """id, ts, level, logger, worker_id, pid, module, lineno, workspace,
        request_id, client_request_id, run_id, track_id, doc_id, operation, source,
        msg, exc_type, (exc IS NOT NULL AND exc <> '') AS has_exc, fingerprint,
        repeat_count"""

    @staticmethod
    def _log_view(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("worker_id", "run_id"):
            if row.get(key) is not None:
                row[key] = _uuid_hex(row[key])
        if isinstance(row.get("fingerprint"), str):
            row["fingerprint"] = row["fingerprint"].strip()
        return row

    async def logs_page(self, f, *, before, after, limit, with_exc=False):
        args: list[Any] = []
        where = _log_conditions(f, args)
        if after is not None:
            args.append(after)
            where.append(f"ts >= ${len(args)}")
        if before is not None:
            args.extend(before)
            where.append(
                f"(ts, id) < (${len(args) - 1}::timestamptz, ${len(args)}::bigint)"
            )
        args.append(int(limit))
        rows = await self._read(
            f"""SELECT {self._LOG_SELECT}{", exc" if with_exc else ""} FROM {TABLE_LOGS}
            WHERE {" AND ".join(where)} ORDER BY ts DESC, id DESC LIMIT ${len(args)}""",
            *args,
        )
        return [self._log_view(r) for r in rows]

    async def logs_groups(self, f, limit):
        args: list[Any] = []
        where = " AND ".join(_log_conditions(f, args))
        args.append(int(limit))
        rows = await self._read(
            f"""WITH g AS (
                SELECT fingerprint, count(*)::bigint AS rows,
                    sum(repeat_count)::bigint AS count, min(ts) AS first_seen,
                    max(ts) AS last_seen, count(DISTINCT workspace)::int AS workspaces,
                    max(level)::int AS level
                FROM {TABLE_LOGS} WHERE {where}
                GROUP BY fingerprint
                ORDER BY count DESC, last_seen DESC LIMIT ${len(args)})
            SELECT g.*, s.id AS sample_id, s.logger, s.module, s.lineno, s.exc_type,
                s.msg AS sample
            FROM g LEFT JOIN LATERAL (
                SELECT id, logger, module, lineno, exc_type, msg FROM {TABLE_LOGS} l
                WHERE l.fingerprint = g.fingerprint AND l.ts = g.last_seen
                ORDER BY l.id DESC LIMIT 1) s ON true
            ORDER BY g.count DESC, g.last_seen DESC""",
            *args,
        )
        for row in rows:
            row["fingerprint"] = str(row["fingerprint"]).strip()
        return rows

    async def log_get(self, log_id):
        rows = await self._read(
            f"SELECT {self._LOG_SELECT}, exc FROM {TABLE_LOGS} WHERE id = $1",
            int(log_id),
        )
        return self._log_view(rows[0]) if rows else None

    async def logs_count(self, f, cap):
        args: list[Any] = []
        where = " AND ".join(_log_conditions(f, args))
        rows = await self._read(
            f"""SELECT count(*)::int AS n FROM (
                SELECT 1 FROM {TABLE_LOGS} WHERE {where} LIMIT {int(cap) + 1}) t""",
            *args,
        )
        return int(rows[0]["n"]) if rows else 0

    async def logs_cleanup(self, days):
        deleted = 0
        for _ in range(LOG_RETENTION_MAX_BATCHES):
            # Batched so retention never holds long locks on the busiest table.
            rows = await self._write(
                f"""DELETE FROM {TABLE_LOGS} WHERE id IN (
                    SELECT id FROM {TABLE_LOGS}
                    WHERE ts < now() - make_interval(days => $1::int)
                    LIMIT {RETENTION_BATCH})
                RETURNING 1""",
                int(days),
            )
            deleted += len(rows)
            if len(rows) < RETENTION_BATCH:
                break
        return deleted

    async def logs_redaction_audit(self, days):
        text = "(msg || ' ' || COALESCE(exc, ''))"
        counts = []
        for name, (pattern, insensitive) in LOG_AUDIT_PATTERNS.items():
            op = "~*" if insensitive else "~"
            counts.append(
                f"count(*) FILTER (WHERE {text} {op} '{pattern}')::int AS {name}"
            )
        async with self._conn() as conn:
            async with conn.transaction(readonly=True):
                # A weekly scan over days of rows: a longer budget than reads.
                await conn.execute("SET LOCAL statement_timeout = 120000")
                row = await conn.fetchrow(
                    f"""SELECT count(*)::int AS rows, {", ".join(counts)}
                    FROM {TABLE_LOGS}
                    WHERE ts >= now() - make_interval(days => $1::int)""",
                    int(days),
                )
        return dict(row) if row else {}

    # -- correlation ---------------------------------------------------------------------
    async def correlate_requests(self, request_id, limit):
        return await self._read(
            f"""SELECT workspace, count(*)::int AS n, min(ts) AS first_ts,
                max(ts) AS last_ts
            FROM {TABLE_LOGS} WHERE request_id = $1
            GROUP BY workspace ORDER BY max(ts) DESC LIMIT $2""",
            request_id,
            int(limit),
        )

    async def correlate_tracks(self, track_id, limit):
        found: dict[Any, dict[str, Any]] = {}
        if self.doc_status_ready and await self._has_table(TABLE_DOC_STATUS):
            for row in await self._read(
                f"""SELECT workspace, count(*)::int AS docs, max(updated_at) AS ts
                FROM {TABLE_DOC_STATUS} WHERE track_id = $1
                GROUP BY workspace ORDER BY max(updated_at) DESC LIMIT $2""",
                track_id,
                int(limit),
            ):
                found[row["workspace"]] = {
                    "workspace": row["workspace"],
                    "docs": row["docs"],
                    "log_lines": 0,
                    "ts": as_utc(row["ts"]),
                }
        for row in await self._read(
            f"""SELECT workspace, count(*)::int AS n, max(ts) AS ts
            FROM {TABLE_LOGS} WHERE track_id = $1
            GROUP BY workspace ORDER BY max(ts) DESC LIMIT $2""",
            track_id,
            int(limit),
        ):
            item = found.setdefault(
                row["workspace"],
                {"workspace": row["workspace"], "docs": 0, "log_lines": 0, "ts": None},
            )
            item["log_lines"] = row["n"]
            item["ts"] = _later(item["ts"], as_utc(row["ts"]))
        return _newest(found.values(), limit)

    async def _index_valid(self, name: str) -> bool:
        """Whether an index exists and is valid (cached; the builder makes it)."""
        cached = self._index_checks.get(name)
        now = time.monotonic()
        if cached is not None and now - cached[1] < _TABLES_TTL_S:
            return cached[0]
        rows = await self._read(
            """SELECT i.indisvalid FROM pg_class c
            JOIN pg_index i ON i.indexrelid = c.oid WHERE c.relname = $1""",
            name,
        )
        valid = bool(rows and rows[0]["indisvalid"])
        self._index_checks[name] = (valid, now)
        return valid

    async def correlate_docs(self, doc_id, limit):
        found: dict[Any, dict[str, Any]] = {}

        def item(workspace: Any) -> dict[str, Any]:
            return found.setdefault(
                workspace,
                {
                    "workspace": workspace,
                    "doc_id": doc_id,
                    "status": None,
                    "file_path": None,
                    "failures": 0,
                    "log_lines": 0,
                    "ts": None,
                },
            )

        # Without its (id) index the doc-status lookup would scan the table.
        if (
            self.doc_status_ready
            and await self._has_table(TABLE_DOC_STATUS)
            and await self._index_valid(DOC_STATUS_ID_INDEX)
        ):
            for row in await self._read(
                f"""SELECT workspace, status, file_path, updated_at
                FROM {TABLE_DOC_STATUS} WHERE id = $1 LIMIT $2""",
                doc_id,
                int(limit),
            ):
                entry = item(row["workspace"])
                entry["status"] = row["status"]
                entry["file_path"] = row["file_path"]
                entry["ts"] = as_utc(row["updated_at"])
        for table, column, key in (
            (TABLE_DOC_FAILURES, "ts", "failures"),
            (TABLE_LOGS, "ts", "log_lines"),
        ):
            for row in await self._read(
                f"""SELECT workspace, count(*)::int AS n, max({column}) AS ts
                FROM {table} WHERE doc_id = $1
                GROUP BY workspace ORDER BY max({column}) DESC LIMIT $2""",
                doc_id,
                int(limit),
            ):
                entry = item(row["workspace"])
                entry[key] = row["n"]
                entry["ts"] = _later(entry["ts"], as_utc(row["ts"]))
        return _newest(found.values(), limit)

    async def owners_by_email(self, email, limit):
        if not await self._has_table(TABLE_OWNERS):
            return {}
        # The registry holds a few thousand rows: a scan is cheap.
        rows = await self._read(
            f"""SELECT workspace_id, COALESCE(email, identity) AS owner
            FROM {TABLE_OWNERS}
            WHERE lower(email) = lower($1) OR lower(identity) = lower($1)
            ORDER BY workspace_id LIMIT $2""",
            email,
            int(limit),
        )
        return {r["workspace_id"]: r["owner"] for r in rows}

    async def workspace_presence(self, workspace):
        docs = False
        if self.doc_status_ready and await self._has_table(TABLE_DOC_STATUS):
            rows = await self._read(
                f"SELECT EXISTS (SELECT 1 FROM {TABLE_DOC_STATUS} WHERE workspace = $1) AS e",
                workspace,
            )
            docs = bool(rows and rows[0]["e"])
        rows = await self._read(
            f"""SELECT EXISTS (SELECT 1 FROM {TABLE_RUNS} WHERE workspace = $1) AS runs,
                EXISTS (SELECT 1 FROM {TABLE_LOGS} WHERE workspace = $1) AS logs""",
            workspace,
        )
        row = rows[0] if rows else {}
        return {
            "docs": docs,
            "runs": bool(row.get("runs")),
            "logs": bool(row.get("logs")),
        }

    # -- performance metrics -------------------------------------------------------------
    async def metrics_write(self, rows):
        if not rows:
            return
        values = [_metric_record(r) for r in rows]
        async with self._conn() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL statement_timeout = {WRITE_TIMEOUT_MS}")
                await conn.executemany(
                    f"""INSERT INTO {TABLE_METRICS_MIN} (minute, worker_id, series, key,
                        count, errors, sum_ms, max_ms, buckets)
                    VALUES (to_timestamp($1::float8), $2::uuid, $3, $4, $5, $6, $7, $8,
                        $9::int[])
                    ON CONFLICT (minute, worker_id, series, key) DO NOTHING""",
                    values,
                )

    async def metrics_rollup(self, start, end):
        async with self._conn() as conn:
            async with conn.transaction():
                # Hours of every worker's rows: a longer budget than reads.
                await conn.execute("SET LOCAL statement_timeout = 60000")
                result = await conn.execute(
                    f"""WITH src AS (
                        SELECT date_trunc('hour', minute) AS hour, series, key, count,
                            errors, sum_ms, max_ms, buckets
                        FROM {TABLE_METRICS_MIN}
                        WHERE minute >= $1 AND minute < $2
                    ), totals AS (
                        SELECT hour, series, key, sum(count)::bigint AS count,
                            sum(errors)::bigint AS errors, sum(sum_ms)::bigint AS sum_ms,
                            max(max_ms)::int AS max_ms
                        FROM src GROUP BY hour, series, key
                    ), cells AS (
                        SELECT hour, series, key, u.i, sum(u.v)::bigint AS v
                        FROM src, unnest(src.buckets) WITH ORDINALITY AS u(v, i)
                        GROUP BY hour, series, key, u.i
                    ), arrays AS (
                        SELECT hour, series, key, array_agg(v ORDER BY i) AS buckets
                        FROM cells GROUP BY hour, series, key
                    )
                    INSERT INTO {TABLE_METRICS_HOUR} AS h (hour, series, key, count,
                        errors, sum_ms, max_ms, buckets)
                    SELECT t.hour, t.series, t.key, t.count, t.errors, t.sum_ms, t.max_ms,
                        a.buckets
                    FROM totals t JOIN arrays a USING (hour, series, key)
                    ON CONFLICT (hour, series, key) DO UPDATE SET
                        count = EXCLUDED.count, errors = EXCLUDED.errors,
                        sum_ms = EXCLUDED.sum_ms, max_ms = EXCLUDED.max_ms,
                        buckets = EXCLUDED.buckets""",
                    start,
                    end,
                )
        try:
            return int(str(result).rsplit(" ", 1)[-1])
        except (ValueError, IndexError):
            return 0

    async def metrics_cleanup(self, minute_days, hour_days):
        deleted = []
        for table, column, days in (
            (TABLE_METRICS_MIN, "minute", minute_days),
            (TABLE_METRICS_HOUR, "hour", hour_days),
        ):
            total = 0
            for _ in range(LOG_RETENTION_MAX_BATCHES):
                # No id column: batches by ctid, found through the primary
                # key's leading time column.
                rows = await self._write(
                    f"""DELETE FROM {table} WHERE ctid = ANY(ARRAY(
                        SELECT ctid FROM {table}
                        WHERE {column} < now() - make_interval(days => $1::int)
                        LIMIT {RETENTION_BATCH}))
                    RETURNING 1""",
                    int(days),
                )
                total += len(rows)
                if len(rows) < RETENTION_BATCH:
                    break
            deleted.append(total)
        return deleted[0], deleted[1]

    async def metrics_read(self, series, start, cut, end, bucket_s):
        bucket = max(60, int(bucket_s))
        rows = await self._read(
            f"""WITH src AS (
                SELECT hour AS ts, series, key, count, errors, sum_ms, max_ms, buckets
                FROM {TABLE_METRICS_HOUR}
                WHERE series = ANY($1::text[]) AND hour >= $2 AND hour < $3
                UNION ALL
                SELECT minute, series, key, count, errors, sum_ms, max_ms,
                    buckets::bigint[]
                FROM {TABLE_METRICS_MIN}
                WHERE minute >= $3 AND minute < $4 AND series = ANY($1::text[])
            ), keyed AS (
                SELECT (floor(extract(epoch FROM ts) / {bucket}) * {bucket})::bigint AS t,
                    series, key, count, errors, sum_ms, max_ms, buckets
                FROM src
            ), totals AS (
                SELECT t, series, key, sum(count)::bigint AS count,
                    sum(errors)::bigint AS errors, sum(sum_ms)::bigint AS sum_ms,
                    max(max_ms)::bigint AS max_ms
                FROM keyed GROUP BY t, series, key
            ), cells AS (
                SELECT t, series, key, u.i, sum(u.v)::bigint AS v
                FROM keyed, unnest(keyed.buckets) WITH ORDINALITY AS u(v, i)
                GROUP BY t, series, key, u.i
            ), arrays AS (
                SELECT t, series, key, array_agg(v ORDER BY i) AS buckets
                FROM cells GROUP BY t, series, key
            )
            SELECT t.t, t.series, t.key, t.count, t.errors, t.sum_ms, t.max_ms,
                a.buckets
            FROM totals t JOIN arrays a USING (t, series, key)
            ORDER BY t.t, t.series, t.key""",
            list(series),
            start,
            max(start, min(cut, end)),
            end,
        )
        for row in rows:
            row["buckets"] = [int(v) for v in (row.get("buckets") or [])]
        return rows

    async def vllm_series(self, start, end, bucket_s):
        bucket = max(60, int(bucket_s))
        # The primary key (dependency, ts) bounds the scan; detail values a
        # probe could not read as numbers are skipped, never cast.
        return await self._read(
            f"""WITH src AS (
                SELECT dependency,
                    (floor(extract(epoch FROM ts) / {bucket}) * {bucket})::bigint AS t,
                    CASE WHEN jsonb_typeof(detail->'running') = 'number'
                        THEN (detail->>'running')::float8 END AS running,
                    CASE WHEN jsonb_typeof(detail->'waiting') = 'number'
                        THEN (detail->>'waiting')::float8 END AS waiting,
                    CASE WHEN jsonb_typeof(detail->'kv_cache_pct') = 'number'
                        THEN (detail->>'kv_cache_pct')::float8 END AS kv_cache_pct
                FROM {TABLE_PROBES}
                WHERE dependency = ANY($1::text[]) AND ts >= $2 AND ts < $3
            )
            SELECT dependency, t, avg(running)::float8 AS running,
                max(waiting)::float8 AS waiting, max(kv_cache_pct)::float8 AS kv_cache_pct
            FROM src
            WHERE running IS NOT NULL OR waiting IS NOT NULL OR kv_cache_pct IS NOT NULL
            GROUP BY dependency, t ORDER BY dependency, t""",
            list(VLLM_DEPENDENCIES),
            start,
            end,
        )

    async def llm_queue_series(self, start, end, bucket_s):
        bucket = max(60, int(bucket_s))
        return await self._read(
            f"""WITH per_worker AS (
                SELECT (floor(extract(epoch FROM minute) / {bucket}) * {bucket})::bigint
                        AS t,
                    worker_id, max(llm_running) AS running,
                    max(llm_in_flight) AS in_flight, max(llm_queued) AS queued
                FROM {TABLE_WORKER_SAMPLES}
                WHERE minute >= $1 AND minute < $2
                    AND (llm_running IS NOT NULL OR llm_queued IS NOT NULL)
                GROUP BY 1, worker_id
            )
            SELECT t, sum(running)::bigint AS running,
                sum(in_flight)::bigint AS in_flight, sum(queued)::bigint AS queued
            FROM per_worker GROUP BY t ORDER BY t""",
            start,
            end,
        )

    async def throughput(self, start, end, bucket_s):
        bucket = max(60, int(bucket_s))
        not_real = sql_in(NOT_REAL_FAILURE_KINDS)
        docs = await self._read(
            f"""SELECT (floor(extract(epoch FROM finished_at) / {bucket}) * {bucket})::bigint
                    AS t,
                sum(docs_done)::bigint AS processed
            FROM {TABLE_RUNS}
            WHERE kind = 'process' AND started_at >= $1::timestamptz - interval '1 day'
                AND finished_at >= $1 AND finished_at < $2
            GROUP BY 1 ORDER BY 1""",
            start,
            end,
        )
        failed = await self._read(
            f"""SELECT (floor(extract(epoch FROM ts) / {bucket}) * {bucket})::bigint AS t,
                count(*)::bigint AS failed
            FROM {TABLE_DOC_FAILURES}
            WHERE ts >= $1 AND ts < $2 AND error_kind NOT IN {not_real}
            GROUP BY 1 ORDER BY 1""",
            start,
            end,
        )
        batches = await self._read(
            f"""SELECT (floor(extract(epoch FROM started_at) / {bucket}) * {bucket})::bigint
                    AS t,
                count(*)::bigint AS batches,
                count(*) FILTER (WHERE status IN ('failed', 'orphaned'))::bigint AS failed
            FROM {TABLE_RUNS}
            WHERE kind = 'email_batch' AND started_at >= $1 AND started_at < $2
            GROUP BY 1 ORDER BY 1""",
            start,
            end,
        )
        return _throughput_rows(docs, failed, batches)

    async def top_workspaces(self, metric, start, end, limit):
        if metric not in TOP_WORKSPACE_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if metric == "failures":
            not_real = sql_in(NOT_REAL_FAILURE_KINDS)
            sql = f"""SELECT workspace, count(*)::float8 AS value
                FROM {TABLE_DOC_FAILURES}
                WHERE ts >= $1 AND ts < $2 AND error_kind NOT IN {not_real}
                GROUP BY workspace ORDER BY value DESC, workspace LIMIT $3"""
        elif metric == "runs":
            sql = f"""SELECT workspace, count(*)::float8 AS value
                FROM {TABLE_RUNS}
                WHERE started_at >= $1 AND started_at < $2
                GROUP BY workspace ORDER BY value DESC, workspace LIMIT $3"""
        else:
            # Minutes of processing inside the window (running runs up to now).
            sql = f"""SELECT workspace, (sum(extract(epoch FROM (
                    LEAST(COALESCE(finished_at, now()), $2::timestamptz)
                    - GREATEST(started_at, $1::timestamptz)
                ))) / 60.0)::float8 AS value
                FROM {TABLE_RUNS}
                WHERE kind = 'process' AND started_at < $2
                    AND (finished_at IS NULL OR finished_at >= $1)
                    AND started_at >= $1::timestamptz - interval '7 days'
                GROUP BY workspace ORDER BY value DESC, workspace LIMIT $3"""
        rows = await self._read(sql, start, end, int(limit))
        return [
            {"workspace": r["workspace"], "value": round(float(r["value"] or 0), 2)}
            for r in rows
        ]


def _metric_record(r: dict[str, Any]) -> tuple:
    buckets = [int(v or 0) for v in (r.get("buckets") or [])][:METRIC_BUCKETS]
    buckets += [0] * (METRIC_BUCKETS - len(buckets))
    return (
        float(r["minute"]),
        str(r["worker_id"]),
        str(r["series"])[:16],
        str(r["key"])[:160],
        int(r.get("count") or 0),
        int(r.get("errors") or 0),
        int(r.get("sum_ms") or 0),
        min(int(r.get("max_ms") or 0), 2_147_483_647),
        buckets,
    )


def _throughput_rows(
    docs: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    batches: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in docs:
        merged.setdefault(int(row["t"]), {"processed": 0, "failed": 0})["processed"] = (
            int(row["processed"] or 0)
        )
    for row in failed:
        merged.setdefault(int(row["t"]), {"processed": 0, "failed": 0})["failed"] = int(
            row["failed"] or 0
        )
    return {
        "docs": [{"t": t, **merged[t]} for t in sorted(merged)],
        "email_batches": [
            {
                "t": int(r["t"]),
                "batches": int(r["batches"] or 0),
                "failed": int(r["failed"] or 0),
            }
            for r in batches
        ],
    }


def _still_running(
    alive: Optional[Callable[[Any, Any], bool]], pid, create_time
) -> bool:
    """The supersede veto; a failing check never blocks superseding."""
    if alive is None:
        return False
    try:
        return bool(alive(pid, create_time))
    except Exception:
        return False


def _opt_int(value: Any) -> Optional[int]:
    """A count for a nullable INT column; anything else is NULL."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _uuid_hex(value: Any) -> str:
    """Worker IDs travel as 32 hex characters; asyncpg returns ``uuid.UUID``."""
    hex_value = getattr(value, "hex", None)
    return hex_value if isinstance(hex_value, str) else str(value).replace("-", "")


def _epoch_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(float(value), timezone.utc)


def _uuid_value(value: Any) -> Any:
    """A UUID column value from 32 hex characters (None when not a UUID)."""
    if value is None or value == "":
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _log_record(r: dict[str, Any]) -> tuple:
    """One COPY record in :data:`LOG_COLUMNS` order."""
    return (
        _epoch_dt(r["ts"]),
        int(r.get("level") or 0),
        pg_text(r.get("logger"), 128) or "-",
        _uuid_value(r.get("worker_id")),
        r.get("pid"),
        pg_text(r.get("module"), 128),
        r.get("lineno"),
        pg_text(r.get("workspace"), 255),
        pg_text(r.get("request_id"), 64),
        pg_text(r.get("client_request_id"), 64),
        _uuid_value(r.get("run_id")),
        pg_text(r.get("track_id"), 255),
        pg_text(r.get("doc_id"), 255),
        pg_text(r.get("operation"), 64),
        pg_text(r.get("source"), 32),
        pg_text(r.get("msg"), 1 << 20) or "",
        pg_text(r.get("exc_type"), 128),
        pg_text(r.get("exc"), 1 << 20),
        (r.get("fingerprint") or "0" * 16)[:16],
        max(1, int(r.get("repeat_count") or 1)),
    )


def _log_conditions(f: LogFilter, args: list[Any]) -> list[str]:
    """WHERE clauses of a log query; every value is a bound parameter except
    the level, an integer inlined so the partial level index can match."""

    def bind(value: Any, cast: str = "") -> str:
        args.append(value)
        return f"${len(args)}{cast}"

    clauses = [f"ts >= {bind(f.start)}", f"ts < {bind(f.end)}"]
    if f.level:
        clauses.append(f"level >= {int(f.level)}")
    if f.workspace is not None:
        clauses.append(f"workspace = {bind(f.workspace)}")
    for column in ("request_id", "track_id", "doc_id", "fingerprint"):
        value = getattr(f, column)
        if value is not None:
            clauses.append(f"{column} = {bind(value)}")
    for column in ("run_id", "worker_id"):
        value = getattr(f, column)
        if value is not None:
            clauses.append(f"{column} = {bind(value, '::uuid')}")
    if f.logger:
        clauses.append(f"logger LIKE {bind(like_escape(f.logger) + '%')}")
    if f.module:
        name = bind(f.module)
        clauses.append(f"(module = {name} OR logger = {name})")
    if f.q:
        clauses.append(f"msg ILIKE {bind('%' + like_escape(f.q) + '%')}")
    return clauses


def _later(a: Optional[datetime], b: Optional[datetime]) -> Optional[datetime]:
    """The later of two optional moments."""
    if a is None:
        return b
    return a if b is None or a >= b else b


def _newest(items: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    epoch = datetime.fromtimestamp(0, timezone.utc)
    return sorted(items, key=lambda i: i.get("ts") or epoch, reverse=True)[:limit]


def _audit_values(row: dict[str, Any]) -> tuple:
    return (
        pg_text(row.get("actor"), 255) or "-",
        pg_text(row.get("auth_kind"), 16),
        pg_text(row.get("impersonated_by"), 255),
        pg_text(row.get("action"), 48) or "-",
        pg_text(row.get("workspace"), 255),
        pg_text(row.get("target"), 255),
        json.dumps(row.get("params") or {}, default=str),
        pg_text(row.get("reason"), 512),
        pg_text(row.get("phase"), 8) or "result",
        pg_text(row.get("outcome"), 24),
        pg_text(row.get("request_id"), 64),
        pg_text(row.get("client_ip_hash"), 16),
    )


async def _close_quietly(conn: Any) -> None:
    try:
        await asyncio.wait_for(conn.close(), timeout=5)
    except Exception:
        try:
            conn.terminate()
        except Exception:
            pass


# -- shared result shaping (Postgres rows and memory rows alike) -----------------------


def _totals_from_rows(
    rows: list[dict[str, Any]], recent: dict[str, Any]
) -> dict[str, Any]:
    by_status = {r["status"]: r for r in rows}
    out: dict[str, Any] = {}
    for status in BACKLOG_STATUSES:
        row = by_status.get(status) or {}
        out[status] = int(row.get("n") or 0)
    out["failed_24h"] = int(recent.get("failed_24h") or 0)
    out["cancelled_24h"] = int(recent.get("cancelled_24h") or 0)
    oldest = [
        as_utc(by_status[s]["oldest"])
        for s in INFLIGHT_STATUSES
        if s in by_status and by_status[s].get("oldest") is not None
    ]
    out["oldest_inflight_at"] = min(oldest) if oldest else None
    pending = by_status.get(STATUS_PENDING) or {}
    out["oldest_pending_at"] = as_utc(pending.get("oldest"))
    out["inflight"] = sum(out[s] for s in INFLIGHT_STATUSES)
    return out


def _workspace_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    cancelled = dup = 0
    oldest_inflight: Optional[datetime] = None
    newest: Optional[datetime] = None
    for row in rows:
        status = row["status"] or "unknown"
        counts[status] = counts.get(status, 0) + int(row.get("n") or 0)
        cancelled += int(row.get("cancelled") or 0)
        dup += int(row.get("dup") or 0)
        if status in INFLIGHT_STATUSES and row.get("oldest") is not None:
            moment = as_utc(row["oldest"])
            oldest_inflight = (
                moment if oldest_inflight is None else min(oldest_inflight, moment)
            )
        if row.get("newest") is not None:
            moment = as_utc(row["newest"])
            newest = moment if newest is None else max(newest, moment)
    return {
        "statuses": counts,
        "cancelled": cancelled,
        "dup": dup,
        "oldest_inflight_at": oldest_inflight,
        "last_activity_at": newest,
    }


class MemoryOpsStore(OpsStore):
    """The same behaviour as Postgres over in-memory rows, for offline tests.

    ``doc_rows`` hold ``LIGHTRAG_DOC_STATUS`` rows with naive UTC timestamps.
    ``clock`` returns epoch seconds and drives lease expiry.
    """

    def __init__(self, clock=time.time):
        self.clock = clock
        self.kv: dict[str, Any] = {}
        self.doc_rows: list[dict[str, Any]] = []
        self.owner_rows: dict[str, str] = {}
        self.usage_rows: list[dict[str, Any]] = []
        self.audit_rows: list[dict[str, Any]] = []
        self.workspace_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.alert_rows: list[dict[str, Any]] = []
        self.worker_rows: dict[str, dict[str, Any]] = {}
        self.sample_rows: dict[tuple[str, int], dict[str, Any]] = {}
        self.probe_rows: list[dict[str, Any]] = []
        # What postgres_health answers (or raises), for probe tests.
        self.pg_health: dict[str, Any] = {"latency_ms": 1.0, "connect_ms": 2.0}
        self.pg_health_error: Optional[BaseException] = None
        self.fail_reads: Optional[BaseException] = None
        # Run tracking tables, and switches for the writer tests.
        self.run_rows: dict[str, dict[str, Any]] = {}
        self.failure_rows: list[dict[str, Any]] = []
        self.log_rows: list[dict[str, Any]] = []
        self._log_seq = 0
        # Performance histograms: minute rows, and hour rows by (hour, series, key).
        self.metric_rows: list[dict[str, Any]] = []
        self.metric_hours: dict[tuple, dict[str, Any]] = {}
        self.fail_writes: Optional[BaseException] = None
        self.write_delay_s: float = 0.0
        self.ready = True
        self.doc_status_ready = True

    # -- helpers ---------------------------------------------------------------------
    def _check(self) -> None:
        if self.fail_reads is not None:
            raise self.fail_reads

    def _utc_now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc).replace(tzinfo=None)

    def _aware_now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    @staticmethod
    def _not_dup(row: dict[str, Any]) -> bool:
        return not str(row["id"]).startswith(DUP_PREFIX)

    def add_doc(self, workspace: str, doc_id: str, status: str, **fields: Any) -> dict:
        row = {
            "workspace": workspace,
            "id": doc_id,
            "status": status,
            "file_path": fields.pop("file_path", None),
            "track_id": fields.pop("track_id", None),
            "chunks_count": fields.pop("chunks_count", None),
            "content_length": fields.pop("content_length", None),
            "error_msg": fields.pop("error_msg", None),
            "metadata": fields.pop("metadata", {}),
            "created_at": fields.pop("created_at", self._utc_now()),
            "updated_at": fields.pop("updated_at", self._utc_now()),
        }
        row.update(fields)
        self.doc_rows.append(row)
        return row

    async def ping(self):
        self._check()

    # -- KV ------------------------------------------------------------------------------
    async def kv_get(self, key, default=None):
        self._check()
        return copy.deepcopy(self.kv.get(key, default))

    async def kv_get_many(self, keys):
        self._check()
        return {k: copy.deepcopy(self.kv[k]) for k in keys if k in self.kv}

    async def kv_put(self, key, value):
        self.kv[key] = json.loads(json.dumps(value, default=str))

    async def kv_merge_many(self, patches):
        for key, patch in patches.items():
            current = self.kv.get(key)
            value = json.loads(json.dumps(patch, default=str))
            self.kv[key] = {**current, **value} if isinstance(current, dict) else value

    async def kv_cas(self, key, expected, value):
        if expected is None:
            if key in self.kv:
                return False
        elif self.kv.get(key) != json.loads(json.dumps(expected, default=str)):
            return False
        self.kv[key] = json.loads(json.dumps(value, default=str))
        return True

    async def kv_delete(self, keys):
        for key in keys:
            self.kv.pop(key, None)

    # -- leases ---------------------------------------------------------------------------
    async def try_lease(self, name, holder, ttl_s):
        self._check()
        key = f"lease:{name}"
        now = self.clock()
        current = self.kv.get(key)
        if current and current.get("exp", 0) >= now and current.get("holder") != holder:
            return False
        self.kv[key] = {"holder": holder, "exp": now + float(ttl_s)}
        return True

    async def release_lease(self, name, holder):
        key = f"lease:{name}"
        if (self.kv.get(key) or {}).get("holder") == holder:
            self.kv.pop(key, None)

    # -- doc status ----------------------------------------------------------------------
    def _backlog_rows(self) -> list[dict[str, Any]]:
        return [
            r
            for r in self.doc_rows
            if r["status"] in BACKLOG_STATUSES and self._not_dup(r)
        ]

    async def backlog_totals(self):
        self._check()
        cutoff = self._utc_now() - timedelta(hours=24)
        grouped: dict[str, dict[str, Any]] = {}
        recent = {"failed_24h": 0, "cancelled_24h": 0}
        for row in self._backlog_rows():
            g = grouped.setdefault(
                row["status"], {"status": row["status"], "n": 0, "oldest": None}
            )
            g["n"] += 1
            if g["oldest"] is None or row["updated_at"] < g["oldest"]:
                g["oldest"] = row["updated_at"]
            if row["status"] == STATUS_FAILED and row["updated_at"] >= cutoff:
                key = (
                    "cancelled_24h" if is_cancelled(row["error_msg"]) else "failed_24h"
                )
                recent[key] += 1
        return _totals_from_rows(list(grouped.values()), recent)

    async def cancelled_count(self):
        self._check()
        return sum(
            1
            for r in self._backlog_rows()
            if r["status"] == STATUS_FAILED and is_cancelled(r["error_msg"])
        )

    async def dup_count(self):
        self._check()
        return sum(1 for r in self.doc_rows if not self._not_dup(r))

    def _aggregate(self, stale_minutes: int) -> list[dict[str, Any]]:
        stale_cutoff = self._utc_now() - timedelta(minutes=stale_minutes)
        agg: dict[str, dict[str, Any]] = {}
        for row in self._backlog_rows():
            a = agg.setdefault(
                row["workspace"],
                {
                    "workspace": row["workspace"],
                    "pending": 0,
                    "parsing": 0,
                    "analyzing": 0,
                    "processing": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "inflight": 0,
                    "oldest_inflight_at": None,
                    "oldest_active_at": None,
                    "last_activity_at": None,
                },
            )
            status = row["status"]
            ts = row["updated_at"]
            if status == STATUS_FAILED:
                a["cancelled" if is_cancelled(row["error_msg"]) else "failed"] += 1
            else:
                a[status] += 1
                a["inflight"] += 1
                if a["oldest_inflight_at"] is None or ts < a["oldest_inflight_at"]:
                    a["oldest_inflight_at"] = ts
                if status in ACTIVE_STATUSES and (
                    a["oldest_active_at"] is None or ts < a["oldest_active_at"]
                ):
                    a["oldest_active_at"] = ts
            if a["last_activity_at"] is None or ts > a["last_activity_at"]:
                a["last_activity_at"] = ts
        for a in agg.values():
            a["owner"] = self.owner_rows.get(a["workspace"])
            a["stuck"] = bool(
                a["oldest_active_at"] is not None
                and a["oldest_active_at"] < stale_cutoff
            )
        return list(agg.values())

    async def backlog_page(
        self,
        *,
        q,
        has,
        sort,
        descending,
        page,
        stale_minutes,
        email_workspaces,
        paused_workspaces,
    ):
        self._check()
        column = BACKLOG_SORTS.get(sort)
        if column is None:
            raise ValueError(f"Unknown sort: {sort}")
        if set(has) - BACKLOG_FILTERS:
            raise ValueError("Unknown filter")
        rows = self._aggregate(stale_minutes)
        if q:
            needle = q.lower()
            rows = [
                r
                for r in rows
                if needle in r["workspace"].lower()
                or needle in (r["owner"] or "").lower()
            ]
        if "failed" in has:
            rows = [r for r in rows if r["failed"] > 0]
        if "stuck" in has:
            rows = [r for r in rows if r["stuck"]]
        if "email" in has:
            rows = [r for r in rows if r["workspace"] in set(email_workspaces)]
        if "paused" in has:
            rows = [r for r in rows if r["workspace"] in set(paused_workspaces)]
        rows.sort(key=lambda r: r["workspace"])
        present = [r for r in rows if r[column] is not None]
        missing = [r for r in rows if r[column] is None]
        present.sort(key=lambda r: r[column], reverse=descending)
        ordered = present + missing
        return ordered[page.offset : page.offset + page.limit], len(ordered)

    async def stale_docs(self, older_than_minutes, page):
        self._check()
        cutoff = self._utc_now() - timedelta(minutes=older_than_minutes)
        rows = sorted(
            (
                r
                for r in self.doc_rows
                if r["status"] in ACTIVE_STATUSES
                and self._not_dup(r)
                and r["updated_at"] < cutoff
            ),
            key=lambda r: (r["updated_at"], r["workspace"], r["id"]),
        )
        picked = rows[page.offset : page.offset + page.limit]
        return [
            {
                k: r[k]
                for k in (
                    "workspace",
                    "id",
                    "status",
                    "file_path",
                    "track_id",
                    "updated_at",
                )
            }
            for r in picked
        ], len(rows)

    def _candidates(self, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
        agg: dict[str, dict[str, Any]] = {}
        for r in self.doc_rows:
            if r["status"] not in statuses or not self._not_dup(r):
                continue
            a = agg.setdefault(
                r["workspace"],
                {
                    "workspace": r["workspace"],
                    "last_at": r["updated_at"],
                    "oldest_at": r["updated_at"],
                    "parsing": 0,
                    "analyzing": 0,
                    "processing": 0,
                },
            )
            a["last_at"] = max(a["last_at"], r["updated_at"])
            a["oldest_at"] = min(a["oldest_at"], r["updated_at"])
            if r["status"] in a:
                a[r["status"]] += 1
        rows = sorted(agg.values(), key=lambda a: a["workspace"])
        rows.sort(key=lambda a: a["last_at"], reverse=True)
        return rows

    async def live_candidates(self, limit):
        self._check()
        return self._candidates(ACTIVE_STATUSES)[:limit]

    async def pending_candidates(self, limit):
        self._check()
        return [r["workspace"] for r in self._candidates((STATUS_PENDING,))[:limit]]

    async def paused_candidates(self, days, limit):
        self._check()
        cutoff = datetime.fromtimestamp(self.clock() - days * 86400, timezone.utc)
        latest: dict[str, dict[str, Any]] = {}
        for e in sorted(self.usage_rows, key=lambda e: e["ts"]):
            if (
                e["kind"] in ("provider_rejection", "ingestion_resumed")
                and e.get("workspace")
                and e["ts"] >= cutoff
            ):
                latest[e["workspace"]] = e
        picked = [
            e
            for e in latest.values()
            if e["kind"] == "provider_rejection"
            and (e.get("detail") or {}).get("paused")
        ]
        picked.sort(key=lambda e: e["ts"], reverse=True)
        return [e["workspace"] for e in picked[:limit]]

    def _failed_in_window(self, hours: int) -> list[dict[str, Any]]:
        cutoff = self._utc_now() - timedelta(hours=hours)
        return [
            r
            for r in self.doc_rows
            if r["status"] == STATUS_FAILED
            and self._not_dup(r)
            and r["updated_at"] >= cutoff
        ]

    @staticmethod
    def _group_summary(key: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
        latest = max(rows, key=lambda r: r["updated_at"])
        return {
            "key": key,
            "count": len(rows),
            "workspaces": len({r["workspace"] for r in rows}),
            "first_seen": min(r["updated_at"] for r in rows),
            "last_seen": latest["updated_at"],
            "example": (latest["error_msg"] or "")[:_SIGNATURE_CHARS]
            if latest["error_msg"] is not None
            else None,
            "example_path": latest.get("file_path"),
        }

    async def legacy_failures(self, hours, group, include_cancelled, limit=50):
        self._check()
        if group not in FAILURE_GROUPS:
            raise ValueError(f"Unknown group: {group}")
        rows = self._failed_in_window(hours)
        real = [
            r
            for r in rows
            if not is_cancelled(r["error_msg"])
            and "failure" not in (r["metadata"] or {})
        ]
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in real:
            key = (
                failure_signature(r["error_msg"])
                if group == "signature"
                else r["workspace"]
            )
            buckets[key].append(r)
        groups = [self._group_summary(k, v) for k, v in buckets.items()]
        groups.sort(key=lambda g: g["key"])
        groups.sort(key=lambda g: g["count"], reverse=True)
        cancelled_rows = [r for r in rows if is_cancelled(r["error_msg"])]
        if include_cancelled and cancelled_rows:
            cancelled = self._group_summary(None, cancelled_rows)
            cancelled.pop("key")
        else:
            cancelled = {"count": len(cancelled_rows)}
        return {"groups": groups[:limit], "cancelled": cancelled}

    async def failure_docs(self, hours, group, key, page):
        self._check()
        rows = self._failed_in_window(hours)
        if group == CANCELLED_GROUP_KEY:
            rows = [r for r in rows if is_cancelled(r["error_msg"])]
        elif group == "signature":
            rows = [
                r
                for r in rows
                if not is_cancelled(r["error_msg"])
                and "failure" not in (r["metadata"] or {})
                and failure_signature(r["error_msg"]) == key
            ]
        elif group == "workspace":
            rows = [
                r
                for r in rows
                if not is_cancelled(r["error_msg"])
                and "failure" not in (r["metadata"] or {})
                and r["workspace"] == key
            ]
        else:
            raise ValueError(f"Unknown group: {group}")
        rows.sort(key=lambda r: (r["workspace"], r["id"]))
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        picked = rows[page.offset : page.offset + page.limit]
        return [
            {
                "workspace": r["workspace"],
                "id": r["id"],
                "file_path": r["file_path"],
                "track_id": r["track_id"],
                "updated_at": r["updated_at"],
                "error": (r["error_msg"] or "")[:_SIGNATURE_CHARS]
                if r["error_msg"] is not None
                else None,
            }
            for r in picked
        ], len(rows)

    async def workspace_counts(self, workspace):
        self._check()
        grouped: dict[str, dict[str, Any]] = {}
        for r in self.doc_rows:
            if r["workspace"] != workspace:
                continue
            g = grouped.setdefault(
                r["status"],
                {
                    "status": r["status"],
                    "n": 0,
                    "cancelled": 0,
                    "dup": 0,
                    "oldest": None,
                    "newest": None,
                },
            )
            if not self._not_dup(r):
                g["dup"] += 1
            elif r["status"] == STATUS_FAILED and is_cancelled(r["error_msg"]):
                g["cancelled"] += 1
            else:
                g["n"] += 1
            if self._not_dup(r) and (
                g["oldest"] is None or r["updated_at"] < g["oldest"]
            ):
                g["oldest"] = r["updated_at"]
            if g["newest"] is None or r["updated_at"] > g["newest"]:
                g["newest"] = r["updated_at"]
        return _workspace_counts_from_rows(list(grouped.values()))

    async def workspace_documents(self, workspace, status, page):
        self._check()
        rows = [
            r
            for r in self.doc_rows
            if r["workspace"] == workspace
            and self._not_dup(r)
            and (not status or r["status"] == status)
        ]
        rows.sort(key=lambda r: r["id"])
        rows.sort(key=lambda r: r["updated_at"], reverse=True)
        picked = rows[page.offset : page.offset + page.limit]
        return [
            {
                "id": r["id"],
                "status": r["status"],
                "file_path": r["file_path"],
                "track_id": r["track_id"],
                "chunks_count": r["chunks_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "error": (r["error_msg"] or "")[:_SIGNATURE_CHARS]
                if r["error_msg"] is not None
                else None,
            }
            for r in picked
        ], len(rows)

    async def document(self, workspace, doc_id):
        self._check()
        for r in self.doc_rows:
            if r["workspace"] == workspace and r["id"] == doc_id:
                return {
                    "id": r["id"],
                    "status": r["status"],
                    "file_path": r["file_path"],
                    "track_id": r["track_id"],
                    "chunks_count": r["chunks_count"],
                    "content_length": r["content_length"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "metadata": copy.deepcopy(r["metadata"] or {}),
                    "error": (r["error_msg"] or "")[:2048]
                    if r["error_msg"] is not None
                    else None,
                }
        return None

    async def owners(self, workspaces):
        self._check()
        return {w: self.owner_rows[w] for w in workspaces if w in self.owner_rows}

    async def usage_events(self, start, end, kinds, limit):
        self._check()
        rows = [
            dict(e)
            for e in self.usage_rows
            if start <= e["ts"] < end and e["kind"] in kinds
        ]
        rows.sort(key=lambda e: e["ts"], reverse=True)
        return rows[:limit]

    async def audit_events(self, start, end, limit):
        self._check()
        rows = [
            dict(a)
            for a in self.audit_rows
            if start <= a["ts"] < end
            and a.get("phase") == "result"
            and a.get("action") not in AUDIT_READ_ACTIONS
        ]
        rows.sort(key=lambda a: a["ts"], reverse=True)
        return rows[:limit]

    async def forget_workspace(self, workspace):
        hashed = workspace_hash(workspace)
        for key in _workspace_keys(workspace):
            self.kv.pop(key, None)
        self.kv[deleted_key(workspace)] = {"at": time.time()}
        for prefix in _workspace_prefixes(workspace):
            for key in [k for k in self.kv if k.startswith(prefix)]:
                self.kv.pop(key, None)
        for summary in WORKSPACE_KV_SUMMARIES:
            value = self.kv.get(summary)
            if isinstance(value, dict) and isinstance(value.get("items"), list):
                value["items"] = [
                    i for i in value["items"] if i.get("workspace") != workspace
                ]
        for row in self.audit_rows:
            if row.get("workspace") == workspace:
                row["workspace"] = hashed
            if row.get("target") == workspace:
                row["target"] = hashed
        self.workspace_rows.pop(workspace, None)
        self.run_rows = {
            k: r for k, r in self.run_rows.items() if r["workspace"] != workspace
        }
        self.failure_rows = [
            r for r in self.failure_rows if r["workspace"] != workspace
        ]
        self.log_rows = [r for r in self.log_rows if r.get("workspace") != workspace]
        for alert in self.alert_rows:
            scope = alert.get("scope") or {}
            if scope.get("workspace") == workspace:
                alert["title"] = alert["title"].replace(workspace, hashed)
                alert["scope"]["workspace"] = hashed
            listed = scope.get("workspaces")
            if isinstance(listed, list) and workspace in listed:
                scope["workspaces"] = [hashed if w == workspace else w for w in listed]

    # -- workers ---------------------------------------------------------------------------
    async def worker_beat(self, row, sample=None):
        self._check()
        now = self._aware_now()
        current = self.worker_rows.get(row["worker_id"]) or {}
        self.worker_rows[row["worker_id"]] = {
            "worker_id": row["worker_id"],
            "host": row["host"],
            "pid": row["pid"],
            "pid_create_time": row.get("pid_create_time"),
            "mode": row["mode"],
            "version": row.get("version"),
            "started_at": current.get("started_at", row["started_at"]),
            "heartbeat_at": now,
            "exited_at": None,
            "superseded": False,
            "leases_held": list(row.get("leases_held") or []),
            "snapshot": json.loads(json.dumps(row.get("snapshot") or {}, default=str)),
        }
        if sample is not None:
            minute = int(self.clock() // 60 * 60)
            self.sample_rows[(row["worker_id"], minute)] = {
                "worker_id": row["worker_id"],
                "minute": minute,
                **{
                    k: sample.get(k)
                    for k in ("cpu_pct", "rss", "loop_lag_ms", "pool_wait_ms")
                },
                **{
                    k: _opt_int(sample.get(k))
                    for k in ("llm_running", "llm_in_flight", "llm_queued")
                },
            }

    async def supersede_workers(self, host, worker_id, started_at, stale_s, alive=None):
        self._check()
        now = self._aware_now()
        count = 0
        for row in self.worker_rows.values():
            if (
                row["host"] == host
                and row["worker_id"] != worker_id
                and row["started_at"] < started_at
                and not row["superseded"]
                and row["exited_at"] is None
                and (now - row["heartbeat_at"]).total_seconds() > stale_s
                and not _still_running(alive, row["pid"], row["pid_create_time"])
            ):
                row["superseded"] = True
                count += 1
        return count

    async def worker_exited(self, worker_id):
        row = self.worker_rows.get(worker_id)
        if row is not None:
            row["exited_at"] = self._aware_now()

    async def cleanup_workers(self, row_hours, sample_days):
        self._check()
        now = self._aware_now()
        for key in [
            k
            for k, r in self.worker_rows.items()
            if (now - r["heartbeat_at"]).total_seconds() > row_hours * 3600
        ]:
            del self.worker_rows[key]
        cutoff = self.clock() - sample_days * 86400
        for key in [k for k, r in self.sample_rows.items() if r["minute"] < cutoff]:
            del self.sample_rows[key]

    async def workers(self, limit):
        self._check()
        now = self._aware_now()
        rows = sorted(
            (copy.deepcopy(r) for r in self.worker_rows.values()),
            key=lambda r: r["heartbeat_at"],
            reverse=True,
        )[:limit]
        for row in rows:
            row["heartbeat_age_s"] = (now - row["heartbeat_at"]).total_seconds()
        return rows

    async def worker_history(self, worker_id, hours, bucket_s):
        self._check()
        cutoff = self.clock() - hours * 3600
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for (wid, minute), row in self.sample_rows.items():
            if wid == worker_id and minute >= cutoff:
                buckets[minute // bucket_s * bucket_s].append(row)
        out = []
        for t in sorted(buckets):
            rows = buckets[t]

            def values(key: str) -> list:
                return [r[key] for r in rows if r.get(key) is not None]

            cpu = values("cpu_pct")
            out.append(
                {
                    "t": t,
                    "rss": max(values("rss"), default=None),
                    "cpu": sum(cpu) / len(cpu) if cpu else None,
                    "loop_lag": max(values("loop_lag_ms"), default=None),
                    "pool_wait": max(values("pool_wait_ms"), default=None),
                }
            )
        return out

    # -- probes ----------------------------------------------------------------------------
    async def postgres_health(self, *, extras, timeout_s):
        if self.pg_health_error is not None:
            raise self.pg_health_error
        out = dict(self.pg_health)
        if not extras:
            for key in (
                "max_connections",
                "connections",
                "db_size_bytes",
                "states",
                "blocking",
                "db_stats",
            ):
                out.pop(key, None)
        return out

    async def probe_write(self, rows):
        self._check()
        now = self._aware_now()
        for row in rows:
            self.probe_rows.append(
                {
                    "ts": now,
                    "dependency": row["dependency"],
                    "status": row["status"],
                    "latency_ms": None
                    if row.get("latency_ms") is None
                    else int(row["latency_ms"]),
                    "error_class": row.get("error_class"),
                    "detail": json.loads(
                        json.dumps(row.get("detail") or {}, default=str)
                    ),
                }
            )

    def _probe_window(self, hours: int, dependency: Optional[str] = None) -> list:
        cutoff = self._aware_now() - timedelta(hours=hours)
        return [
            r
            for r in self.probe_rows
            if r["ts"] >= cutoff
            and (dependency is None or r["dependency"] == dependency)
        ]

    async def probe_strips(self, hours, slot_s):
        self._check()
        cells: dict[tuple[str, int], dict[str, Any]] = {}
        for row in self._probe_window(hours):
            slot = int(row["ts"].timestamp() // slot_s)
            cell = cells.setdefault(
                (row["dependency"], slot),
                {
                    "dependency": row["dependency"],
                    "slot": slot,
                    "ok": 0,
                    "warn": 0,
                    "down": 0,
                    "total": 0,
                },
            )
            if row["status"] in ("ok", "warn", "down"):
                cell[row["status"]] += 1
            cell["total"] += 1
        return list(cells.values())

    async def probe_recent(self, dependency, limit):
        self._check()
        rows = [dict(r) for r in self.probe_rows if r["dependency"] == dependency]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]

    async def probe_series(self, dependency, hours, bucket_s):
        self._check()
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self._probe_window(hours, dependency):
            buckets[int(row["ts"].timestamp()) // bucket_s * bucket_s].append(row)
        out = []
        for t in sorted(buckets):
            rows = buckets[t]
            latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
            out.append(
                {
                    "t": t,
                    "avg_ms": sum(latencies) / len(latencies) if latencies else None,
                    "max_ms": max(latencies, default=None),
                    "down": sum(1 for r in rows if r["status"] == "down"),
                    "warn": sum(1 for r in rows if r["status"] == "warn"),
                    "total": len(rows),
                }
            )
        return out

    async def probe_error_classes(self, dependency, hours):
        self._check()
        counts: dict[str, int] = defaultdict(int)
        for row in self._probe_window(hours, dependency):
            if row.get("error_class"):
                counts[row["error_class"]] += 1
        return [
            {"key": k, "count": n}
            for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ][:20]

    async def probe_cleanup(self, days):
        cutoff = self._aware_now() - timedelta(days=days)
        self.probe_rows = [r for r in self.probe_rows if r["ts"] >= cutoff]

    # -- run tracking ----------------------------------------------------------------------
    async def _write_gate(self) -> None:
        if self.write_delay_s:
            await asyncio.sleep(self.write_delay_s)
        if self.fail_writes is not None:
            raise self.fail_writes

    async def kv_merge_existing(self, patches):
        await self._write_gate()
        for key, patch in patches.items():
            current = self.kv.get(key)
            if isinstance(current, dict):
                value = json.loads(json.dumps(patch, default=str))
                self.kv[key] = {**current, **value}

    def _moment(self, epoch: Any) -> Optional[datetime]:
        if epoch is None:
            return None
        return datetime.fromtimestamp(float(epoch), timezone.utc)

    async def runs_write(self, rows):
        await self._write_gate()
        for r in rows:
            current = self.run_rows.get(r["run_id"])
            if current is not None and current["status"] in RUN_FINAL_STATUSES:
                continue
            row = {
                "run_id": r["run_id"],
                "workspace": r["workspace"],
                "kind": r["kind"],
                "trigger": r.get("trigger") or "other",
                "status": r["status"],
                "was_orphaned": bool(current and current["was_orphaned"])
                or bool(current and current["status"] == "orphaned"),
                "worker_id": r.get("worker_id"),
                "pid": r.get("pid"),
                "started_at": current["started_at"]
                if current
                else self._moment(r["started_at"]),
                "heartbeat_at": max(
                    self._moment(r["heartbeat_at"]),
                    current["heartbeat_at"] if current else self._moment(0),
                ),
                "progress_at": max(
                    self._moment(r["progress_at"]),
                    current["progress_at"] if current else self._moment(0),
                ),
                "finished_at": self._moment(r.get("finished_at")),
                "docs_total": int(r.get("docs_total") or 0),
                "docs_done": int(r.get("docs_done") or 0),
                "docs_failed": int(r.get("docs_failed") or 0),
                "stage": pg_text(r.get("stage"), 16),
                "label": pg_text(r.get("label"), 128) or (current or {}).get("label"),
                "error_kind": pg_text(r.get("error_kind"), 32),
                "error_summary": pg_text(r.get("error_summary"), 512),
                "request_id": (current or {}).get("request_id", r.get("request_id")),
                "host": (current or {}).get("host") or r.get("host"),
                "orphaned_by": (current or {}).get("orphaned_by"),
            }
            self.run_rows[r["run_id"]] = row

    async def doc_failures_write(self, rows):
        await self._write_gate()
        for r in rows:
            row = {
                key: r.get(key)
                for key in (
                    "workspace",
                    "doc_id",
                    "track_id",
                    "run_id",
                    "stage",
                    "error_kind",
                    "http_status",
                    "exc_type",
                    "message",
                    "attempt",
                    "model",
                )
            }
            row["message"] = pg_text(row["message"], 1024)
            row["id"] = len(self.failure_rows) + 1
            row["ts"] = self._moment(r["ts"])
            self.failure_rows.append(row)

    async def runs_orphan(self, run_ids, confirmed_by=None):
        changed = []
        for run_id in run_ids:
            row = self.run_rows.get(run_id)
            if row is not None and row["status"] == "running":
                row["status"] = "orphaned"
                row["error_summary"] = ORPHAN_SUMMARY
                row["orphaned_by"] = (confirmed_by or {}).get(run_id)
                changed.append(run_id)
        return changed

    async def runs_cleanup(self, run_days, failure_days):
        now = self._aware_now()
        self.run_rows = {
            k: r
            for k, r in self.run_rows.items()
            if (now - r["started_at"]).total_seconds() <= run_days * 86400
        }
        self.failure_rows = [
            r
            for r in self.failure_rows
            if (now - r["ts"]).total_seconds() <= failure_days * 86400
        ]
        cutoff = time.time() - DELETED_KEEP_DAYS * 86400
        for key in [k for k in self.kv if k.startswith(DELETED_KV_PREFIX)]:
            value = self.kv[key]
            if isinstance(value, dict) and float(value.get("at") or 0) < cutoff:
                self.kv.pop(key, None)

    def _run_view(self, row: dict[str, Any]) -> dict[str, Any]:
        now = self._aware_now()
        out = copy.deepcopy(row)
        out["run_host"] = out.pop("host", None)
        out.setdefault("orphaned_by", None)
        out["heartbeat_age_s"] = (now - row["heartbeat_at"]).total_seconds()
        out["progress_age_s"] = (now - row["progress_at"]).total_seconds()
        worker = self.worker_rows.get(row.get("worker_id") or "")
        out["worker_host"] = worker["host"] if worker else None
        out["worker_pid_create_time"] = worker["pid_create_time"] if worker else None
        out["worker_started_at"] = worker["started_at"] if worker else None
        out["worker_superseded"] = worker["superseded"] if worker else None
        out["worker_exited_at"] = worker["exited_at"] if worker else None
        out["worker_heartbeat_age_s"] = (
            (now - worker["heartbeat_at"]).total_seconds() if worker else None
        )
        return out

    def _runs_sorted(self) -> list[dict[str, Any]]:
        rows = sorted(self.run_rows.values(), key=lambda r: r["run_id"], reverse=True)
        rows.sort(key=lambda r: r["started_at"], reverse=True)
        return rows

    async def running_runs(self, limit):
        self._check()
        return [
            self._run_view(r) for r in self._runs_sorted() if r["status"] == "running"
        ][:limit]

    async def orphaned_runs(self, hours, limit):
        self._check()
        cutoff = self._aware_now() - timedelta(hours=hours)
        return [
            self._run_view(r)
            for r in self._runs_sorted()
            if r["status"] == "orphaned" and r["started_at"] >= cutoff
        ][:limit]

    async def runs_page(self, *, workspace, kind, status, before, limit):
        self._check()
        out = []
        for r in self._runs_sorted():
            if workspace and r["workspace"] != workspace:
                continue
            if kind and r["kind"] != kind:
                continue
            if status and r["status"] != status:
                continue
            if before is not None and (r["started_at"], r["run_id"]) >= (
                before[0],
                before[1],
            ):
                continue
            out.append(self._run_view(r))
            if len(out) >= limit:
                break
        return out

    async def run_get(self, run_id):
        self._check()
        row = self.run_rows.get(run_id)
        return self._run_view(row) if row else None

    def _failures_sorted(self) -> list[dict[str, Any]]:
        rows = sorted(self.failure_rows, key=lambda r: r["id"], reverse=True)
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows

    async def run_failures(self, run_id, limit):
        self._check()
        return [dict(r) for r in self._failures_sorted() if r.get("run_id") == run_id][
            :limit
        ]

    async def last_runs(self, workspaces):
        self._check()
        out: dict[str, dict[str, Any]] = {}
        wanted = set(workspaces)
        for r in self._runs_sorted():
            if r["workspace"] in wanted and r["workspace"] not in out:
                out[r["workspace"]] = self._run_view(r)
        return out

    def _failures_in_window(self, hours: int) -> list[dict[str, Any]]:
        cutoff = self._aware_now() - timedelta(hours=hours)
        return [r for r in self._failures_sorted() if r["ts"] >= cutoff]

    @staticmethod
    def _group_key(row: dict[str, Any], group: str) -> str:
        if group == "model":
            return row.get("model") or ""
        return row[group]

    @staticmethod
    def _failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        latest = rows[0]
        return {
            "count": len(rows),
            "docs": len({(r["workspace"], r["doc_id"]) for r in rows}),
            "workspaces": len({r["workspace"] for r in rows}),
            "first_seen": min(r["ts"] for r in rows),
            "last_seen": latest["ts"],
            "example": latest.get("message"),
            "example_kind": latest["error_kind"],
            "example_stage": latest["stage"],
        }

    async def failure_groups(self, hours, group, include_cancelled, limit=50):
        self._check()
        if group not in DURABLE_GROUP_COLUMNS:
            raise ValueError(f"Unknown group: {group}")
        rows = self._failures_in_window(hours)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r["error_kind"] != "cancelled":
                buckets[self._group_key(r, group)].append(r)
        groups = [{"key": k, **self._failure_summary(v)} for k, v in buckets.items()]
        groups.sort(key=lambda g: g["key"])
        groups.sort(key=lambda g: g["count"], reverse=True)
        cancelled_rows = [r for r in rows if r["error_kind"] == "cancelled"]
        if include_cancelled and cancelled_rows:
            cancelled = self._failure_summary(cancelled_rows)
            for key in ("example_kind", "example_stage"):
                cancelled.pop(key)
        else:
            cancelled = {"count": len(cancelled_rows)}
        return {"groups": groups[:limit], "cancelled": cancelled}

    async def failure_group_docs(self, hours, group, key, page):
        self._check()
        if group not in DURABLE_GROUP_COLUMNS:
            raise ValueError(f"Unknown group: {group}")
        include_cancelled = group == "error_kind" and key == CANCELLED_GROUP_KEY
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for r in self._failures_in_window(hours):
            if self._group_key(r, group) != key:
                continue
            if r["error_kind"] == "cancelled" and not include_cancelled:
                continue
            latest.setdefault((r["workspace"], r["doc_id"]), r)
        rows = sorted(latest.values(), key=lambda r: (r["workspace"], r["doc_id"]))
        rows.sort(key=lambda r: r["ts"], reverse=True)
        picked = rows[page.offset : page.offset + page.limit]
        out = []
        for r in picked:
            doc = next(
                (
                    d
                    for d in self.doc_rows
                    if d["workspace"] == r["workspace"] and d["id"] == r["doc_id"]
                ),
                None,
            )
            out.append(
                {
                    **r,
                    "file_path": doc["file_path"] if doc else None,
                    "current_status": doc["status"] if doc else None,
                }
            )
        return out, len(rows)

    async def failures_by_kind(self, workspace, hours):
        self._check()
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self._failures_in_window(hours):
            if r["workspace"] == workspace:
                buckets[r["error_kind"]].append(r)
        out = [
            {
                "key": k,
                "count": len(v),
                "docs": len({r["doc_id"] for r in v}),
                "last_seen": v[0]["ts"],
            }
            for k, v in buckets.items()
        ]
        out.sort(key=lambda g: g["key"])
        out.sort(key=lambda g: g["count"], reverse=True)
        return out[:20]

    async def doc_failures(self, workspace, doc_id, limit):
        self._check()
        return [
            dict(r)
            for r in self._failures_sorted()
            if r["workspace"] == workspace and r["doc_id"] == doc_id
        ][:limit]

    async def open_pauses(self, limit):
        self._check()
        out = []
        for key in sorted(self.kv):
            value = self.kv[key]
            if (
                key.startswith("pause:")
                and isinstance(value, dict)
                and not value.get("cleared_at")
            ):
                out.append({**copy.deepcopy(value), "key": key})
        return out[:limit]

    async def pauses_page(self, limit, offset):
        rows = await self.open_pauses(10**9)

        def since(row: dict[str, Any]) -> float:
            value = row.get("since")
            return float(value) if isinstance(value, (int, float)) else 0.0

        rows.sort(key=lambda r: (since(r), str(r.get("workspace")), r["key"]))
        return rows[offset : offset + limit]

    async def pause_counts(self):
        self._check()
        rows: dict[tuple[str, str], int] = {}
        workspaces = set()
        for key, value in self.kv.items():
            if (
                key.startswith("pause:")
                and isinstance(value, dict)
                and not value.get("cleared_at")
            ):
                slot = value.get("slot") or "unknown"
                kind = value.get("kind") or "unknown"
                rows[(slot, kind)] = rows.get((slot, kind), 0) + 1
                workspaces.add(value.get("workspace"))
        return _pause_counts(
            [{"slot": s, "kind": k, "count": n} for (s, k), n in rows.items()],
            len(workspaces),
        )

    async def run_events(self, start, end, limit):
        self._check()
        rows = [
            {
                "run_id": r["run_id"],
                "workspace": r["workspace"],
                "kind": r["kind"],
                "status": r["status"],
                "was_orphaned": r["was_orphaned"],
                "ts": r["heartbeat_at"],
            }
            for r in self.run_rows.values()
            if (r["status"] == "orphaned" or r["was_orphaned"])
            and start <= r["heartbeat_at"] < end
        ]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]

    # -- audit ---------------------------------------------------------------------------
    async def audit_write(self, row):
        await self._write_gate()
        values = _audit_values(row)
        audit_id = max((int(a.get("id") or 0) for a in self.audit_rows), default=0) + 1
        self.audit_rows.append(
            {
                "id": audit_id,
                "ts": self._aware_now(),
                **dict(
                    zip(
                        (
                            "actor",
                            "auth_kind",
                            "impersonated_by",
                            "action",
                            "workspace",
                            "target",
                            "params",
                            "reason",
                            "phase",
                            "outcome",
                            "request_id",
                            "client_ip_hash",
                        ),
                        values,
                    )
                ),
            }
        )
        self.audit_rows[-1]["params"] = json.loads(self.audit_rows[-1]["params"])
        return audit_id

    async def audit_page(self, f, before, limit):
        self._check()
        rows = []
        for row in self.audit_rows:
            ts = row["ts"]
            if not (f.start <= ts < f.end):
                continue
            if any(
                getattr(f, c) is not None and row.get(c) != getattr(f, c)
                for c in ("actor", "action", "workspace", "phase")
            ):
                continue
            if before is not None and (ts, int(row["id"])) >= before:
                continue
            rows.append(copy.deepcopy(row))
        rows.sort(key=lambda r: (r["ts"], int(r["id"])), reverse=True)
        return rows[:limit]

    # -- admin actions ------------------------------------------------------------
    async def claim_action(self, key, holder, ttl_s, info):
        await self._write_gate()
        now = self.clock()
        current = self.kv.get(key)
        if (
            isinstance(current, dict)
            and float(current.get("exp") or 0) >= now
            and current.get("holder") != holder
        ):
            return False, copy.deepcopy(current)
        self.kv[key] = json.loads(
            json.dumps(
                {**info, "holder": holder, "exp": now + float(ttl_s)}, default=str
            )
        )
        return True, None

    async def release_action(self, key, holder):
        if (self.kv.get(key) or {}).get("holder") == holder:
            self.kv.pop(key, None)

    async def rate_hit(self, key, limit, window_s, now):
        await self._write_gate()
        current = self.kv.get(key)
        if not isinstance(current, dict) or float(current.get("start") or 0) <= (
            now - window_s
        ):
            current = {"start": float(now), "n": 0}
        current = {"start": current["start"], "n": int(current.get("n") or 0) + 1}
        self.kv[key] = current
        return current["n"] <= limit, max(0.0, current["start"] + window_s - now)

    async def kv_put_if_older(self, key, value, older_than):
        await self._write_gate()
        current = self.kv.get(key)
        if isinstance(current, dict) and float(current.get("at") or 0) > older_than:
            return False
        self.kv[key] = json.loads(json.dumps(value, default=str))
        return True

    # -- alerts ------------------------------------------------------------------------
    def _alert_copy(self, row: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(row)

    async def alerts_open(self, limit):
        self._check()
        rows = [r for r in self.alert_rows if r.get("resolved_at") is None]
        rows.sort(key=lambda r: -int(r["id"]))
        return [self._alert_copy(r) for r in rows[:limit]]

    async def alerts_upsert(self, items):
        await self._write_gate()
        # No await below: like the partial unique index, the check and the
        # insert are one step, so racing evaluators never add a second open
        # row for a fingerprint.
        now = self._aware_now()
        out = []
        for item in items:
            current = next(
                (
                    r
                    for r in self.alert_rows
                    if r["fingerprint"] == item["fingerprint"]
                    and r.get("resolved_at") is None
                ),
                None,
            )
            scope = json.loads(json.dumps(item.get("scope") or {}, default=str))
            if current is None:
                alert_id = max((int(r["id"]) for r in self.alert_rows), default=0) + 1
                self.alert_rows.append(
                    {
                        "id": alert_id,
                        "rule": item["rule"],
                        "fingerprint": item["fingerprint"],
                        "severity": item["severity"],
                        "state": "firing",
                        "title": item["title"],
                        "scope": scope,
                        "link": item.get("link"),
                        "runbook": item.get("runbook"),
                        "first_seen": now,
                        "last_seen": now,
                        "resolved_at": None,
                        "acked_by": None,
                        "acked_at": None,
                        "ack_note": None,
                        "last_emailed_at": None,
                    }
                )
                out.append(
                    {
                        "id": alert_id,
                        "fingerprint": item["fingerprint"],
                        "inserted": True,
                    }
                )
                continue
            current.update(
                {
                    "severity": item["severity"],
                    "title": item["title"],
                    "scope": scope,
                    "link": item.get("link"),
                    "runbook": item.get("runbook"),
                    "last_seen": now,
                }
            )
            out.append(
                {
                    "id": int(current["id"]),
                    "fingerprint": item["fingerprint"],
                    "inserted": False,
                }
            )
        return out

    async def alerts_resolve(self, fingerprints):
        await self._write_gate()
        wanted = set(fingerprints)
        now = self._aware_now()
        out = []
        for row in self.alert_rows:
            if row.get("resolved_at") is None and row["fingerprint"] in wanted:
                row["state"] = "resolved"
                row["resolved_at"] = now
                out.append(self._alert_copy(row))
        return out

    async def alerts_page(self, state, before, limit):
        self._check()
        rows = []
        for row in self.alert_rows:
            resolved = row.get("resolved_at") is not None
            if (state == "open" and resolved) or (state == "resolved" and not resolved):
                continue
            if before is not None and int(row["id"]) >= int(before):
                continue
            rows.append(row)
        rows.sort(key=lambda r: -int(r["id"]))
        return [self._alert_copy(r) for r in rows[:limit]]

    async def alert_get(self, alert_id):
        self._check()
        row = next((r for r in self.alert_rows if int(r["id"]) == int(alert_id)), None)
        return self._alert_copy(row) if row else None

    async def alerts_history(self, fingerprint, limit):
        self._check()
        rows = [r for r in self.alert_rows if r["fingerprint"] == fingerprint]
        rows.sort(key=lambda r: -int(r["id"]))
        return [self._alert_copy(r) for r in rows[:limit]]

    async def alert_ack(self, alert_id, actor, note):
        await self._write_gate()
        row = next((r for r in self.alert_rows if int(r["id"]) == int(alert_id)), None)
        if row is None:
            return "not_found", None
        if row.get("resolved_at") is not None:
            return "resolved", self._alert_copy(row)
        if row.get("acked_at") is not None:
            return "already_acked", self._alert_copy(row)
        row["acked_by"] = str(actor)[:255]
        row["acked_at"] = self._aware_now()
        row["ack_note"] = note[:512] if note else None
        return "acked", self._alert_copy(row)

    async def alert_claim_email(self, alert_id, kind, repeat_s):
        await self._write_gate()
        row = next((r for r in self.alert_rows if int(r["id"]) == int(alert_id)), None)
        if row is None:
            return None
        now = self._aware_now()
        last = row.get("last_emailed_at")
        if kind == "resolve":
            resolved = row.get("resolved_at")
            ok = resolved is not None and last is not None and last < resolved
        else:
            ok = row.get("resolved_at") is None and (
                last is None or last < now - timedelta(seconds=float(repeat_s))
            )
        if not ok:
            return None
        # Strictly later than the last claim, as clock_timestamp() would be.
        if last is not None and now <= last:
            now = last + timedelta(microseconds=1)
        row["last_emailed_at"] = now
        return now

    async def alert_unclaim_email(self, alert_id, claimed, previous):
        await self._write_gate()
        for row in self.alert_rows:
            if (
                int(row["id"]) == int(alert_id)
                and row.get("last_emailed_at") == claimed
            ):
                row["last_emailed_at"] = previous

    async def alerts_window(self, start, end, limit):
        self._check()
        rows = [
            r
            for r in self.alert_rows
            if r["last_seen"] >= start and r["first_seen"] < end
        ]
        rows.sort(key=lambda r: (r["first_seen"], int(r["id"])), reverse=True)
        return [self._alert_copy(r) for r in rows[:limit]]

    async def alerts_cleanup(self, days):
        await self._write_gate()
        cutoff = self._aware_now() - timedelta(days=int(days))
        kept = [
            r
            for r in self.alert_rows
            if r.get("resolved_at") is None or r["resolved_at"] >= cutoff
        ]
        deleted = len(self.alert_rows) - len(kept)
        self.alert_rows = kept
        return deleted

    # -- logs ----------------------------------------------------------------------------
    async def logs_write(self, rows):
        await self._write_gate()
        records = [_log_record(r) for r in rows]
        for record in records:
            self._log_seq += 1
            row = dict(zip(LOG_COLUMNS, record))
            row["id"] = self._log_seq
            for key in ("worker_id", "run_id"):
                if row[key] is not None:
                    row[key] = row[key].hex
            self.log_rows.append(row)

    @staticmethod
    def _log_matches(row: dict[str, Any], f: LogFilter) -> bool:
        if not (f.start <= row["ts"] < f.end) or row["level"] < (f.level or 0):
            return False
        for column in (
            "workspace",
            "request_id",
            "track_id",
            "doc_id",
            "fingerprint",
            "run_id",
            "worker_id",
        ):
            wanted = getattr(f, column)
            if wanted is not None and row.get(column) != wanted:
                return False
        if f.logger and not (row["logger"] or "").startswith(f.logger):
            return False
        if f.module and f.module not in (row.get("module"), row["logger"]):
            return False
        if f.q and f.q.lower() not in (row["msg"] or "").lower():
            return False
        return True

    @staticmethod
    def _log_out(row: dict[str, Any], with_exc: bool = False) -> dict[str, Any]:
        out = {k: v for k, v in row.items() if k != "exc"}
        out["has_exc"] = bool(row.get("exc"))
        if with_exc:
            out["exc"] = row.get("exc")
        return out

    async def logs_page(self, f, *, before, after, limit, with_exc=False):
        self._check()
        rows = [r for r in self.log_rows if self._log_matches(r, f)]
        if after is not None:
            rows = [r for r in rows if r["ts"] >= after]
        rows.sort(key=lambda r: (r["ts"], r["id"]), reverse=True)
        if before is not None:
            rows = [r for r in rows if (r["ts"], r["id"]) < tuple(before)]
        return [self._log_out(r, with_exc=with_exc) for r in rows[:limit]]

    async def logs_groups(self, f, limit):
        self._check()
        groups: dict[str, dict[str, Any]] = {}
        for row in sorted(self.log_rows, key=lambda r: (r["ts"], r["id"])):
            if not self._log_matches(row, f):
                continue
            g = groups.setdefault(
                row["fingerprint"],
                {
                    "fingerprint": row["fingerprint"],
                    "rows": 0,
                    "count": 0,
                    "first_seen": row["ts"],
                    "_workspaces": set(),
                    "level": row["level"],
                },
            )
            g["rows"] += 1
            g["count"] += int(row["repeat_count"])
            g["last_seen"] = row["ts"]
            g["level"] = max(g["level"], row["level"])
            g["_workspaces"].add(row.get("workspace"))
            g.update(
                sample_id=row["id"],
                logger=row["logger"],
                module=row.get("module"),
                lineno=row.get("lineno"),
                exc_type=row.get("exc_type"),
                sample=row["msg"],
            )
        out = []
        for g in groups.values():
            workspaces = g.pop("_workspaces")
            g["workspaces"] = len({w for w in workspaces if w is not None})
            out.append(g)
        out.sort(key=lambda g: (g["count"], g["last_seen"]), reverse=True)
        return out[:limit]

    async def log_get(self, log_id):
        self._check()
        for row in self.log_rows:
            if row["id"] == int(log_id):
                return self._log_out(row, with_exc=True)
        return None

    async def logs_count(self, f, cap):
        self._check()
        return min(cap + 1, sum(1 for r in self.log_rows if self._log_matches(r, f)))

    async def logs_cleanup(self, days):
        cutoff = self._aware_now() - timedelta(days=days)
        before = len(self.log_rows)
        self.log_rows = [r for r in self.log_rows if r["ts"] >= cutoff]
        return before - len(self.log_rows)

    async def logs_redaction_audit(self, days):
        self._check()
        cutoff = self._aware_now() - timedelta(days=days)
        compiled = {
            name: re.compile(pattern, re.IGNORECASE if insensitive else 0)
            for name, (pattern, insensitive) in LOG_AUDIT_PATTERNS.items()
        }
        counts = {"rows": 0, **{name: 0 for name in compiled}}
        for row in self.log_rows:
            if row["ts"] < cutoff:
                continue
            counts["rows"] += 1
            text = f"{row['msg']} {row.get('exc') or ''}"
            for name, pattern in compiled.items():
                if pattern.search(text):
                    counts[name] += 1
        return counts

    # -- correlation ---------------------------------------------------------------------
    async def correlate_requests(self, request_id, limit):
        self._check()
        found: dict[Any, dict[str, Any]] = {}
        for row in self.log_rows:
            if row.get("request_id") != request_id:
                continue
            item = found.setdefault(
                row.get("workspace"),
                {
                    "workspace": row.get("workspace"),
                    "n": 0,
                    "first_ts": row["ts"],
                    "last_ts": row["ts"],
                },
            )
            item["n"] += 1
            item["first_ts"] = min(item["first_ts"], row["ts"])
            item["last_ts"] = max(item["last_ts"], row["ts"])
        return sorted(found.values(), key=lambda i: i["last_ts"], reverse=True)[:limit]

    async def correlate_tracks(self, track_id, limit):
        self._check()
        found: dict[Any, dict[str, Any]] = {}

        def item(workspace: Any) -> dict[str, Any]:
            return found.setdefault(
                workspace,
                {"workspace": workspace, "docs": 0, "log_lines": 0, "ts": None},
            )

        for row in self.doc_rows:
            if row.get("track_id") == track_id:
                entry = item(row["workspace"])
                entry["docs"] += 1
                entry["ts"] = _later(entry["ts"], as_utc(row.get("updated_at")))
        for row in self.log_rows:
            if row.get("track_id") == track_id:
                entry = item(row.get("workspace"))
                entry["log_lines"] += 1
                entry["ts"] = _later(entry["ts"], row["ts"])
        return _newest(found.values(), limit)

    async def correlate_docs(self, doc_id, limit):
        self._check()
        found: dict[Any, dict[str, Any]] = {}

        def item(workspace: Any) -> dict[str, Any]:
            return found.setdefault(
                workspace,
                {
                    "workspace": workspace,
                    "doc_id": doc_id,
                    "status": None,
                    "file_path": None,
                    "failures": 0,
                    "log_lines": 0,
                    "ts": None,
                },
            )

        for row in self.doc_rows:
            if row["id"] == doc_id:
                entry = item(row["workspace"])
                entry["status"] = row["status"]
                entry["file_path"] = row.get("file_path")
                entry["ts"] = as_utc(row.get("updated_at"))
        for row in self.failure_rows:
            if row["doc_id"] == doc_id:
                entry = item(row["workspace"])
                entry["failures"] += 1
                entry["ts"] = _later(entry["ts"], row["ts"])
        for row in self.log_rows:
            if row.get("doc_id") == doc_id:
                entry = item(row.get("workspace"))
                entry["log_lines"] += 1
                entry["ts"] = _later(entry["ts"], row["ts"])
        return _newest(found.values(), limit)

    async def owners_by_email(self, email, limit):
        self._check()
        wanted = email.lower()
        return (
            {
                ws: owner
                for ws, owner in sorted(self.owner_rows.items())
                if (owner or "").lower() == wanted
            }
            if wanted
            else {}
        )

    async def workspace_presence(self, workspace):
        self._check()
        return {
            "docs": any(r["workspace"] == workspace for r in self.doc_rows),
            "runs": any(r["workspace"] == workspace for r in self.run_rows.values()),
            "logs": any(r.get("workspace") == workspace for r in self.log_rows),
        }

    # -- performance metrics -------------------------------------------------------------
    async def metrics_write(self, rows):
        await self._write_gate()
        stored = {
            (r["minute"], r["worker_id"], r["series"], r["key"])
            for r in self.metric_rows
        }
        for r in rows:
            record = _metric_record(r)
            minute = datetime.fromtimestamp(record[0], timezone.utc)
            identity = (minute, uuid.UUID(record[1]).hex, record[2], record[3])
            if identity in stored:
                continue
            stored.add(identity)
            self.metric_rows.append(
                {
                    "minute": minute,
                    "worker_id": identity[1],
                    "series": record[2],
                    "key": record[3],
                    "count": record[4],
                    "errors": record[5],
                    "sum_ms": record[6],
                    "max_ms": record[7],
                    "buckets": list(record[8]),
                }
            )

    async def metrics_rollup(self, start, end):
        await self._write_gate()
        hours: dict[tuple, dict[str, Any]] = {}
        for r in self.metric_rows:
            if not start <= r["minute"] < end:
                continue
            hour = r["minute"].replace(minute=0, second=0, microsecond=0)
            row = hours.setdefault(
                (hour, r["series"], r["key"]),
                {
                    "count": 0,
                    "errors": 0,
                    "sum_ms": 0,
                    "max_ms": 0,
                    "buckets": [0] * METRIC_BUCKETS,
                },
            )
            row["count"] += r["count"]
            row["errors"] += r["errors"]
            row["sum_ms"] += r["sum_ms"]
            row["max_ms"] = max(row["max_ms"], r["max_ms"])
            for i, v in enumerate(r["buckets"][:METRIC_BUCKETS]):
                row["buckets"][i] += v
        self.metric_hours.update(hours)
        return len(hours)

    async def metrics_cleanup(self, minute_days, hour_days):
        now = self._aware_now()
        minute_cut = now - timedelta(days=minute_days)
        hour_cut = now - timedelta(days=hour_days)
        before = len(self.metric_rows)
        self.metric_rows = [r for r in self.metric_rows if r["minute"] >= minute_cut]
        old_hours = [k for k in self.metric_hours if k[0] < hour_cut]
        for key in old_hours:
            del self.metric_hours[key]
        return before - len(self.metric_rows), len(old_hours)

    async def metrics_read(self, series, start, cut, end, bucket_s):
        self._check()
        bucket = max(60, int(bucket_s))
        cut = max(start, min(cut, end))
        wanted = set(series)
        sources: list[tuple[datetime, str, str, dict[str, Any]]] = [
            (hour, s, k, row)
            for (hour, s, k), row in self.metric_hours.items()
            if s in wanted and start <= hour < cut
        ]
        sources += [
            (r["minute"], r["series"], r["key"], r)
            for r in self.metric_rows
            if r["series"] in wanted and cut <= r["minute"] < end
        ]
        out: dict[tuple, dict[str, Any]] = {}
        for ts, s, k, row in sources:
            t = int(ts.timestamp()) // bucket * bucket
            agg = out.setdefault(
                (t, s, k),
                {
                    "t": t,
                    "series": s,
                    "key": k,
                    "count": 0,
                    "errors": 0,
                    "sum_ms": 0,
                    "max_ms": 0,
                    "buckets": [0] * METRIC_BUCKETS,
                },
            )
            agg["count"] += int(row["count"])
            agg["errors"] += int(row["errors"])
            agg["sum_ms"] += int(row["sum_ms"])
            agg["max_ms"] = max(agg["max_ms"], int(row["max_ms"]))
            for i, v in enumerate(row["buckets"][:METRIC_BUCKETS]):
                agg["buckets"][i] += int(v)
        return [out[key] for key in sorted(out)]

    async def vllm_series(self, start, end, bucket_s):
        self._check()
        bucket = max(60, int(bucket_s))
        cells: dict[tuple[str, int], dict[str, list[float]]] = {}
        for row in self.probe_rows:
            if row["dependency"] not in VLLM_DEPENDENCIES or not (
                start <= row["ts"] < end
            ):
                continue
            detail = row.get("detail") or {}
            values = {
                k: float(detail[k])
                for k in ("running", "waiting", "kv_cache_pct")
                if isinstance(detail.get(k), (int, float))
                and not isinstance(detail.get(k), bool)
            }
            if not values:
                continue
            t = int(row["ts"].timestamp()) // bucket * bucket
            cell = cells.setdefault(
                (row["dependency"], t),
                {"running": [], "waiting": [], "kv_cache_pct": []},
            )
            for k, v in values.items():
                cell[k].append(v)
        return [
            {
                "dependency": dep,
                "t": t,
                "running": sum(c["running"]) / len(c["running"])
                if c["running"]
                else None,
                "waiting": max(c["waiting"], default=None),
                "kv_cache_pct": max(c["kv_cache_pct"], default=None),
            }
            for (dep, t), c in sorted(cells.items())
        ]

    async def llm_queue_series(self, start, end, bucket_s):
        self._check()
        bucket = max(60, int(bucket_s))
        lo, hi = start.timestamp(), end.timestamp()
        per_worker: dict[tuple[int, str], dict[str, Optional[int]]] = {}
        for (wid, minute), row in self.sample_rows.items():
            if not lo <= minute < hi:
                continue
            if row.get("llm_running") is None and row.get("llm_queued") is None:
                continue
            cell = per_worker.setdefault(
                (minute // bucket * bucket, wid),
                {"running": None, "in_flight": None, "queued": None},
            )
            for name in ("running", "in_flight", "queued"):
                value = row.get(f"llm_{name}")
                if value is not None:
                    cell[name] = max(cell[name] or 0, int(value))
        totals: dict[int, dict[str, Optional[int]]] = {}
        for (t, _), cell in per_worker.items():
            target = totals.setdefault(
                t, {"running": None, "in_flight": None, "queued": None}
            )
            for name, value in cell.items():
                if value is not None:
                    target[name] = (target[name] or 0) + value
        return [{"t": t, **totals[t]} for t in sorted(totals)]

    async def throughput(self, start, end, bucket_s):
        self._check()
        bucket = max(60, int(bucket_s))

        def slot(moment: datetime) -> int:
            return int(moment.timestamp()) // bucket * bucket

        processed: dict[int, int] = defaultdict(int)
        batches: dict[int, list[int]] = {}
        for r in self.run_rows.values():
            finished = r.get("finished_at")
            if r["kind"] == "process" and finished and start <= finished < end:
                processed[slot(finished)] += int(r["docs_done"])
            if r["kind"] == "email_batch" and start <= r["started_at"] < end:
                entry = batches.setdefault(slot(r["started_at"]), [0, 0])
                entry[0] += 1
                entry[1] += r["status"] in ("failed", "orphaned")
        failed: dict[int, int] = defaultdict(int)
        for r in self.failure_rows:
            if start <= r["ts"] < end and r["error_kind"] not in NOT_REAL_FAILURE_KINDS:
                failed[slot(r["ts"])] += 1
        return _throughput_rows(
            [{"t": t, "processed": v} for t, v in sorted(processed.items())],
            [{"t": t, "failed": v} for t, v in sorted(failed.items())],
            [
                {"t": t, "batches": v[0], "failed": v[1]}
                for t, v in sorted(batches.items())
            ],
        )

    async def top_workspaces(self, metric, start, end, limit):
        self._check()
        if metric not in TOP_WORKSPACE_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        values: dict[str, float] = defaultdict(float)
        if metric == "failures":
            for r in self.failure_rows:
                if (
                    start <= r["ts"] < end
                    and r["error_kind"] not in NOT_REAL_FAILURE_KINDS
                ):
                    values[r["workspace"]] += 1
        else:
            now = self._aware_now()
            for r in self.run_rows.values():
                if metric == "runs":
                    if start <= r["started_at"] < end:
                        values[r["workspace"]] += 1
                    continue
                finished = r.get("finished_at")
                if r["kind"] != "process" or r["started_at"] >= end:
                    continue
                if finished is not None and finished < start:
                    continue
                stop = min(finished or now, end)
                begin = max(r["started_at"], start)
                values[r["workspace"]] += max(0.0, (stop - begin).total_seconds()) / 60
        ranked = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[: int(limit)]
        return [{"workspace": ws, "value": round(v, 2)} for ws, v in ranked]
