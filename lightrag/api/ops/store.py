"""Postgres storage and read queries for the Operations console.

``PostgresOpsStore`` creates its own tables idempotently on the shared
``ClientManager`` pool, and only when the server runs on Postgres KV storage
(``kv_storage == "PGKVStorage"``). This PR creates ``LIGHTRAG_OPS_KV`` (small
JSON values, renewable leases) and ``LIGHTRAG_OPS_AUDIT``; the other ops
tables arrive with the features that write them. Indexes on the large existing
tables are never built here: :mod:`lightrag.api.ops.indexes` builds them in
the background.

Rules every query follows:

- reads run inside a read-only transaction with ``statement_timeout`` 3 s;
- status lists are inlined as SQL literals (from constants, never from input)
  so the planner can match the partial doc-status index under generic plans;
- sort columns come from a whitelist; every other value is a bound parameter;
- ``LIGHTRAG_DOC_STATUS.updated_at`` is a naive UTC ``TIMESTAMP``, so it is
  always compared with :data:`UTC_NOW_SQL`, whatever the server's TimeZone.

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
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Iterable, Optional

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

READ_TIMEOUT_MS = 3000
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
        self._tables: set[str] = set()
        self._tables_at = 0.0
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
        ):
            await self._db.execute(statement)
        await self._refresh_tables(force=True)
        self.doc_status_ready = TABLE_DOC_STATUS in self._tables
        self.ready = True

    async def finalize(self) -> None:
        self.ready = False
        conn, self._lease_conn = self._lease_conn, None
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

    async def _lease_query(self, sql: str, *args: Any) -> list[Any]:
        import asyncpg

        async with self._lease_guard():
            for attempt in (1, 2):
                if self._lease_conn is None or self._lease_conn.is_closed():
                    self._lease_conn = await asyncpg.connect(**self.connect_kwargs())
                try:
                    return await self._lease_conn.fetch(sql, *args, timeout=5)
                except (
                    asyncpg.InterfaceError,
                    asyncpg.PostgresConnectionError,
                    OSError,
                ):
                    await _close_quietly(self._lease_conn)
                    self._lease_conn = None
                    if attempt == 2:
                        raise
        return []  # pragma: no cover - loop always returns or raises

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
            condition = f"{base} AND NOT {_CANCELLED} AND workspace = $2"
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
        self.fail_reads: Optional[BaseException] = None
        self.ready = True
        self.doc_status_ready = True

    # -- helpers ---------------------------------------------------------------------
    def _check(self) -> None:
        if self.fail_reads is not None:
            raise self.fail_reads

    def _utc_now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc).replace(tzinfo=None)

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
                if not is_cancelled(r["error_msg"]) and r["workspace"] == key
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
            if start <= a["ts"] < end and a.get("phase") == "result"
        ]
        rows.sort(key=lambda a: a["ts"], reverse=True)
        return rows[:limit]

    async def forget_workspace(self, workspace):
        hashed = workspace_hash(workspace)
        for key in _workspace_keys(workspace):
            self.kv.pop(key, None)
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
        for alert in self.alert_rows:
            if (alert.get("scope") or {}).get("workspace") == workspace:
                alert["title"] = alert["title"].replace(workspace, hashed)
                alert["scope"]["workspace"] = hashed
