"""Storage for token usage: individual calls, rollups, events and settings.

Each batch of records is written to the calls table and added to the hourly and
daily rollups in one transaction, so the rollups never need rebuilding. Retention
follows the admin decision: calls for 90 days, hourly for 2 years, daily always.

``PostgresUsageStore`` is what runs in production and shares the LightRAG pool.
``MemoryUsageStore`` has the same behaviour for offline tests.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from lightrag.usage import SystemEvent, UsageRecord
from lightrag.utils import logger

DIMENSIONS = ("workspace", "purpose", "operation", "source", "model", "billing")
SUM_METRICS = (
    "calls",
    "cache_hits",
    "errors",
    "truncated",
    "abandoned",
    "estimated_calls",
    "not_measured",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "wasted_tokens",
    "latency_ms_sum",
    "reported_cost",
    # Tokens on calls with no provider-reported cost, which the price table prices.
    "priced_prompt_tokens",
    "priced_completion_tokens",
)
BACKFILL_SOURCE = "backfill"
# Largest page of individual calls one read returns; older pages use before_ts.
CALLS_LIMIT_MAX = 200
CALL_RETENTION_DAYS = 90
HOURLY_RETENTION_DAYS = 730
UNKNOWN_MODEL = "unknown"

_TABLE_CALLS = "LIGHTRAG_USAGE_CALLS"
_TABLE_HOURLY = "LIGHTRAG_USAGE_HOURLY"
_TABLE_DAILY = "LIGHTRAG_USAGE_DAILY"
_TABLE_EVENTS = "LIGHTRAG_USAGE_EVENTS"
_TABLE_KV = "LIGHTRAG_USAGE_KV"
_TABLE_DOC_STATUS = "LIGHTRAG_DOC_STATUS"


def insights_timezone() -> tzinfo:
    """Day boundaries for the dashboard. Malaysia time unless configured."""
    name = os.getenv("INSIGHTS_TIMEZONE", "Asia/Kuala_Lumpur")
    try:
        return ZoneInfo(name)
    except Exception:
        logger.warning(f"Unknown INSIGHTS_TIMEZONE {name!r}; using UTC+8")
        return timezone(timedelta(hours=8))


def track_source_sql(column: str = "track_id") -> str:
    return (
        f"CASE WHEN {column} LIKE 'upload\\_%' THEN 'file_upload' "
        f"WHEN {column} LIKE 'email\\_%' THEN 'email' "
        f"WHEN {column} LIKE 'scan\\_%' THEN 'folder_scan' "
        f"WHEN {column} LIKE 'insert\\_%' OR {column} LIKE 'enqueue\\_%' THEN 'text_api' "
        "ELSE 'other' END"
    )


def record_metrics(record: UsageRecord) -> dict[str, float]:
    """What one record adds to its rollup row."""
    tokens = record.prompt_tokens + record.completion_tokens
    reported = record.cost is not None
    failed = record.status in ("truncated", "abandoned", "error")
    return {
        "calls": 0 if record.cache_hit else 1,
        "cache_hits": 1 if record.cache_hit else 0,
        "errors": 1 if record.status == "error" else 0,
        "truncated": 1 if record.status == "truncated" else 0,
        "abandoned": 1 if record.status == "abandoned" else 0,
        "estimated_calls": 1 if record.estimated else 0,
        "not_measured": 0 if record.measured else 1,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "cached_tokens": record.cached_tokens,
        "wasted_tokens": tokens if failed else 0,
        "latency_ms_sum": record.latency_ms,
        "reported_cost": record.cost or 0.0,
        "priced_prompt_tokens": 0 if reported else record.prompt_tokens,
        "priced_completion_tokens": 0 if reported else record.completion_tokens,
    }


def llm_rates(row: dict[str, Any]) -> dict[str, float]:
    """Error, truncation and abandon rates over attempted (non-cache) calls.

    The one definition shared by Token Insights and the Operations console, so
    both pages show the same number for the same window.
    """
    attempts = row.get("calls") or 0
    if not attempts:
        return {"error_rate": 0.0, "truncated_rate": 0.0, "abandoned_rate": 0.0}
    return {
        "error_rate": (row.get("errors") or 0) / attempts,
        "truncated_rate": (row.get("truncated") or 0) / attempts,
        "abandoned_rate": (row.get("abandoned") or 0) / attempts,
    }


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), CALLS_LIMIT_MAX))


def _epoch(value: Any) -> float:
    return value.timestamp() if isinstance(value, datetime) else float(value)


def _percentile(sorted_values: list[float], fraction: float) -> Optional[float]:
    """Linear interpolation, like Postgres ``percentile_cont``."""
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def rollup_increments(
    records: Iterable[UsageRecord], tz: tzinfo
) -> tuple[dict[tuple, dict[str, float]], dict[tuple, dict[str, float]]]:
    """Group records into (hourly, daily) rollup rows keyed on bucket + dimensions."""
    hourly: dict[tuple, dict[str, float]] = {}
    daily: dict[tuple, dict[str, float]] = {}
    for record in records:
        local = datetime.fromtimestamp(record.ts, tz)
        hour = local.replace(minute=0, second=0, microsecond=0)
        dims = tuple(getattr(record, d) or "" for d in DIMENSIONS)
        metrics = record_metrics(record)
        for table, bucket in ((hourly, hour), (daily, local.date())):
            row = table.setdefault((bucket, *dims), dict.fromkeys(SUM_METRICS, 0))
            for name, value in metrics.items():
                row[name] += value
            row["latency_ms_max"] = max(row.get("latency_ms_max", 0), record.latency_ms)
    return hourly, daily


class UsageStore(ABC):
    def __init__(self, tz: Optional[tzinfo] = None):
        self.tz = tz or insights_timezone()

    async def initialize(self) -> None:  # pragma: no cover - default no-op
        pass

    async def finalize(self) -> None:  # pragma: no cover - default no-op
        pass

    @abstractmethod
    async def write(
        self, records: list[UsageRecord], events: list[SystemEvent]
    ) -> None: ...

    @abstractmethod
    async def rollup(
        self,
        grain: str,
        start: Any,
        end: Any,
        *,
        filters: Optional[dict[str, str]] = None,
        group_by: Iterable[str] = (),
        exclude_backfill: bool = False,
    ) -> list[dict[str, Any]]:
        """Sum metrics by ``group_by`` (dimensions, plus ``bucket``).

        ``grain`` is ``day`` (inclusive dates) or ``hour`` (aware datetimes,
        ``end`` exclusive). ``bucket`` comes back as a date or an aware datetime.
        """

    @abstractmethod
    async def top_items(
        self,
        kind: str,
        start: datetime,
        end: datetime,
        *,
        filters: Optional[dict[str, str]] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Tokens by document or client, per model, from the calls table."""

    @abstractmethod
    async def calls(
        self,
        start: datetime,
        end: datetime,
        *,
        filters: Optional[dict[str, str]] = None,
        doc_id: Optional[str] = None,
        client: Optional[str] = None,
        limit: int = 100,
        before_ts: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Newest first, at most ``CALLS_LIMIT_MAX``; ``before_ts`` (epoch
        seconds or datetime) pages further back."""

    @abstractmethod
    async def llm_health(
        self,
        start: datetime,
        end: datetime,
        *,
        filters: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Attempts, failures and latency percentiles (cache hits excluded),
        plus :func:`llm_rates`. Shared by Insights and the Operations console."""

    @abstractmethod
    async def latency_p95(
        self,
        start: datetime,
        end: datetime,
        *,
        filters: Optional[dict[str, str]] = None,
    ) -> dict[str, float]:
        """95th percentile latency in ms by purpose, or by model if filtered to one."""

    @abstractmethod
    async def peak_tokens_per_minute(
        self, start: datetime, end: datetime
    ) -> list[dict[str, Any]]:
        """Busiest minute on the system key, with each purpose's share of it."""

    @abstractmethod
    async def events(
        self,
        start: datetime,
        end: datetime,
        *,
        workspace: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Events for a workspace, plus events that apply to every workspace."""

    @abstractmethod
    async def kv_get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    async def kv_set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    async def try_lease(self, key: str, seconds: int) -> bool:
        """True for one holder across all workers until the lease runs out."""

    @abstractmethod
    async def anonymize_workspace(self, workspace: str, label: str) -> None:
        """Keep a deleted workspace's numbers under ``label``, without its name."""

    @abstractmethod
    async def purge(self) -> None: ...

    @abstractmethod
    async def first_recorded_day(self) -> Optional[date]:
        """First day of real recording; backfilled estimates don't count."""

    @abstractmethod
    async def known_workspaces(self) -> list[str]: ...

    @abstractmethod
    async def llm_cache_batch(
        self, after: Optional[tuple[str, str]], limit: int
    ) -> list[dict[str, Any]]:
        """Saved LLM cache entries in (workspace, id) order, for the backfill."""

    @abstractmethod
    async def last_activity(self) -> dict[str, datetime]:
        """Latest hour with recorded usage, by workspace."""

    @abstractmethod
    async def document_days(
        self, start: date, end: date, workspace: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Processed documents per day, workspace and source, without duplicates."""

    @abstractmethod
    async def document_status_counts(self) -> dict[str, dict[str, int]]: ...

    @abstractmethod
    async def document_paths(
        self, workspace: str, ids: list[str]
    ) -> dict[str, str]: ...


class MemoryUsageStore(UsageStore):
    """In-memory store with the same semantics as Postgres, for offline tests."""

    def __init__(self, tz: Optional[tzinfo] = None):
        super().__init__(tz)
        self.call_rows: list[UsageRecord] = []
        self.hourly: dict[tuple, dict[str, float]] = {}
        self.daily: dict[tuple, dict[str, float]] = {}
        self.event_rows: list[SystemEvent] = []
        self.kv: dict[str, Any] = {}
        self.leases: dict[str, float] = {}
        self.document_rows: list[dict[str, Any]] = []
        self.status_counts: dict[str, dict[str, int]] = {}
        self.paths: dict[tuple[str, str], str] = {}
        self.cache_rows: list[dict[str, Any]] = []

    async def write(self, records, events) -> None:
        self.call_rows.extend(r for r in records if r.source != BACKFILL_SOURCE)
        hourly, daily = rollup_increments(records, self.tz)
        for target, increments in ((self.hourly, hourly), (self.daily, daily)):
            for key, row in increments.items():
                current = target.setdefault(key, dict.fromkeys(SUM_METRICS, 0))
                for name in SUM_METRICS:
                    current[name] += row[name]
                current["latency_ms_max"] = max(
                    current.get("latency_ms_max", 0), row["latency_ms_max"]
                )
        self.event_rows.extend(events)

    async def rollup(
        self, grain, start, end, *, filters=None, group_by=(), exclude_backfill=False
    ):
        table = self.daily if grain == "day" else self.hourly
        group_by = tuple(group_by)
        out: dict[tuple, dict[str, Any]] = {}
        for key, row in table.items():
            bucket = key[0]
            if grain == "day" and not start <= bucket <= end:
                continue
            if grain == "hour" and not start <= bucket < end:
                continue
            dims = dict(zip(DIMENSIONS, key[1:]))
            if filters and any(dims[k] != v for k, v in filters.items()):
                continue
            if exclude_backfill and dims["source"] == BACKFILL_SOURCE:
                continue
            values = {"bucket": bucket, **dims}
            group = tuple(values[g] for g in group_by)
            merged = out.setdefault(
                group,
                {
                    **{g: values[g] for g in group_by},
                    **dict.fromkeys(SUM_METRICS, 0),
                    "latency_ms_max": 0,
                },
            )
            for name in SUM_METRICS:
                merged[name] += row[name]
            merged["latency_ms_max"] = max(
                merged["latency_ms_max"], row["latency_ms_max"]
            )
        return list(out.values())

    def _matching_calls(self, start, end, filters):
        for r in self.call_rows:
            if not start.timestamp() <= r.ts < end.timestamp():
                continue
            if filters and any(getattr(r, k) != v for k, v in filters.items()):
                continue
            yield r

    async def top_items(self, kind, start, end, *, filters=None, limit=50):
        column = "doc_id" if kind == "document" else "client"
        out: dict[tuple, dict[str, Any]] = {}
        for r in self._matching_calls(start, end, filters):
            item = getattr(r, column)
            if not item:
                continue
            row = out.setdefault(
                (item, r.workspace, r.purpose, r.model),
                {
                    "item": item,
                    "workspace": r.workspace,
                    "purpose": r.purpose,
                    "model": r.model,
                    **dict.fromkeys(SUM_METRICS, 0),
                },
            )
            for name, value in record_metrics(r).items():
                row[name] += value
        rows = sorted(
            out.values(),
            key=lambda x: -(x["prompt_tokens"] + x["completion_tokens"]),
        )
        return rows[:limit]

    async def calls(
        self,
        start,
        end,
        *,
        filters=None,
        doc_id=None,
        client=None,
        limit=100,
        before_ts=None,
    ):
        before = _epoch(before_ts) if before_ts is not None else None
        rows = [
            asdict(r)
            for r in self._matching_calls(start, end, filters)
            if (doc_id is None or r.doc_id == doc_id)
            and (client is None or r.client == client)
            and (before is None or r.ts < before)
        ]
        rows.sort(key=lambda x: -x["ts"])
        return rows[: _clamp_limit(limit)]

    async def llm_health(self, start, end, *, filters=None):
        row = dict.fromkeys(
            ("calls", "cache_hits", "errors", "truncated", "abandoned"), 0
        )
        latencies: list[float] = []
        for r in self._matching_calls(start, end, filters):
            if r.cache_hit:
                row["cache_hits"] += 1
                continue
            row["calls"] += 1
            latencies.append(r.latency_ms)
            if r.status in ("error", "truncated", "abandoned"):
                row["errors" if r.status == "error" else r.status] += 1
        latencies.sort()
        return {
            **row,
            "p50_ms": _percentile(latencies, 0.5),
            "p95_ms": _percentile(latencies, 0.95),
            "p99_ms": _percentile(latencies, 0.99),
            **llm_rates(row),
        }

    async def latency_p95(self, start, end, *, filters=None):
        values: dict[str, list[int]] = defaultdict(list)
        for r in self._matching_calls(start, end, filters):
            if not r.cache_hit:
                values[r.purpose].append(r.latency_ms)
        return {k: sorted(v)[max(0, int(len(v) * 0.95) - 1)] for k, v in values.items()}

    async def peak_tokens_per_minute(self, start, end):
        minutes: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in self._matching_calls(start, end, {"billing": "system"}):
            minutes[int(r.ts // 60)][r.purpose] += r.prompt_tokens + r.completion_tokens
        if not minutes:
            return []
        peak = max(minutes, key=lambda m: sum(minutes[m].values()))
        ts = datetime.fromtimestamp(peak * 60, timezone.utc)
        return [
            {"minute": ts, "purpose": k, "tokens": v} for k, v in minutes[peak].items()
        ]

    async def events(self, start, end, *, workspace=None, limit=500):
        rows = [
            e
            for e in self.event_rows
            if start.timestamp() <= e.ts < end.timestamp()
            and (workspace is None or e.workspace in (None, workspace))
        ]
        rows.sort(key=lambda e: -e.ts)
        return [
            {
                "ts": datetime.fromtimestamp(e.ts, timezone.utc),
                "kind": e.kind,
                "workspace": e.workspace,
                "purpose": e.purpose,
                "detail": e.detail,
            }
            for e in rows[:limit]
        ]

    async def kv_get(self, key, default=None):
        return self.kv.get(key, default)

    async def kv_set(self, key, value):
        self.kv[key] = value

    async def try_lease(self, key, seconds):
        now = time.time()
        if self.leases.get(key, 0) > now:
            return False
        self.leases[key] = now + seconds
        return True

    async def anonymize_workspace(self, workspace, label):
        for table in (self.hourly, self.daily):
            for key in [k for k in table if k[1] == workspace]:
                table[(key[0], label, *key[2:])] = table.pop(key)
        self.call_rows = [
            UsageRecord(
                **{
                    **asdict(r),
                    "workspace": label,
                    "doc_id": None,
                    "track_id": None,
                    "client": None,
                }
            )
            if r.workspace == workspace
            else r
            for r in self.call_rows
        ]
        self.event_rows = [
            SystemEvent(kind=e.kind, workspace=label, purpose=e.purpose, ts=e.ts)
            if e.workspace == workspace
            else e
            for e in self.event_rows
        ]

    async def purge(self):
        now = time.time()
        self.call_rows = [
            r for r in self.call_rows if r.ts >= now - CALL_RETENTION_DAYS * 86400
        ]
        cutoff = datetime.now(self.tz) - timedelta(days=HOURLY_RETENTION_DAYS)
        for key in [k for k in self.hourly if k[0] < cutoff]:
            del self.hourly[key]

    async def first_recorded_day(self):
        days = [
            k[0]
            for k in self.daily
            if dict(zip(DIMENSIONS, k[1:]))["source"] != BACKFILL_SOURCE
        ]
        return min(days) if days else None

    async def known_workspaces(self):
        return sorted({k[1] for k in self.daily} | set(self.status_counts))

    async def llm_cache_batch(self, after, limit):
        rows = sorted(self.cache_rows, key=lambda r: (r["workspace"], r["id"]))
        if after:
            rows = [r for r in rows if (r["workspace"], r["id"]) > after]
        return rows[:limit]

    async def last_activity(self):
        out: dict[str, datetime] = {}
        for key in self.hourly:
            ws = key[1]
            if ws not in out or key[0] > out[ws]:
                out[ws] = key[0]
        return out

    async def document_days(self, start, end, workspace=None):
        return [
            r
            for r in self.document_rows
            if start <= r["day"] <= end
            and (workspace is None or r["workspace"] == workspace)
        ]

    async def document_status_counts(self):
        return self.status_counts

    async def document_paths(self, workspace, ids):
        return {
            i: self.paths[(workspace, i)] for i in ids if (workspace, i) in self.paths
        }


class PostgresUsageStore(UsageStore):
    """Usage tables in Postgres, created on first start with the shared pool."""

    def __init__(
        self, vector_storage: Optional[str] = None, tz: Optional[tzinfo] = None
    ):
        super().__init__(tz)
        self._vector_storage = vector_storage
        self._db: Any = None

    async def initialize(self) -> None:
        from lightrag.kg.postgres_impl import ClientManager

        self._db = await ClientManager.get_client(self._vector_storage)
        metric_columns = ", ".join(
            f"{m} DOUBLE PRECISION NOT NULL DEFAULT 0"
            if m == "reported_cost"
            else f"{m} BIGINT NOT NULL DEFAULT 0"
            for m in SUM_METRICS
        )
        dims = ", ".join(f"{d} VARCHAR(255) NOT NULL DEFAULT ''" for d in DIMENSIONS)
        dim_names = ", ".join(DIMENSIONS)
        statements = (
            f"""CREATE TABLE IF NOT EXISTS {_TABLE_CALLS} (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL,
                {dims},
                client VARCHAR(255),
                host VARCHAR(512),
                prompt_tokens BIGINT NOT NULL DEFAULT 0,
                completion_tokens BIGINT NOT NULL DEFAULT 0,
                reasoning_tokens BIGINT NOT NULL DEFAULT 0,
                cached_tokens BIGINT NOT NULL DEFAULT 0,
                cost DOUBLE PRECISION,
                latency_ms BIGINT NOT NULL DEFAULT 0,
                status VARCHAR(16) NOT NULL DEFAULT 'ok',
                cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
                estimated BOOLEAN NOT NULL DEFAULT FALSE,
                measured BOOLEAN NOT NULL DEFAULT TRUE,
                doc_id VARCHAR(255),
                track_id VARCHAR(255)
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_lightrag_usage_calls_ts ON {_TABLE_CALLS} (ts)",
            f"CREATE INDEX IF NOT EXISTS idx_lightrag_usage_calls_ws_ts ON {_TABLE_CALLS} (workspace, ts)",
            f"""CREATE TABLE IF NOT EXISTS {_TABLE_HOURLY} (
                bucket TIMESTAMPTZ NOT NULL, {dims}, {metric_columns},
                latency_ms_max BIGINT NOT NULL DEFAULT 0,
                PRIMARY KEY (bucket, {dim_names})
            )""",
            f"""CREATE TABLE IF NOT EXISTS {_TABLE_DAILY} (
                bucket DATE NOT NULL, {dims}, {metric_columns},
                latency_ms_max BIGINT NOT NULL DEFAULT 0,
                PRIMARY KEY (bucket, {dim_names})
            )""",
            f"""CREATE TABLE IF NOT EXISTS {_TABLE_EVENTS} (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL,
                kind VARCHAR(64) NOT NULL,
                workspace VARCHAR(255),
                purpose VARCHAR(32),
                detail JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )""",
            f"CREATE INDEX IF NOT EXISTS idx_lightrag_usage_events_ts ON {_TABLE_EVENTS} (ts)",
            f"""CREATE TABLE IF NOT EXISTS {_TABLE_KV} (
                key VARCHAR(128) PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )""",
        )
        # Workers starting together against an empty database would race on
        # CREATE ... IF NOT EXISTS; take turns.
        async with self._db.ddl_lock():
            for statement in statements:
                await self._db.execute(statement)

    async def finalize(self) -> None:
        if self._db is not None:
            from lightrag.kg.postgres_impl import ClientManager

            await ClientManager.release_client(self._db)
            self._db = None

    async def _connection(self):
        await self._db._ensure_pool()
        return self._db.pool.acquire()

    async def write(self, records, events) -> None:
        calls = [
            (
                datetime.fromtimestamp(r.ts, timezone.utc),
                *(getattr(r, d) or "" for d in DIMENSIONS),
                r.client,
                r.host,
                r.prompt_tokens,
                r.completion_tokens,
                r.reasoning_tokens,
                r.cached_tokens,
                r.cost,
                r.latency_ms,
                r.status,
                r.cache_hit,
                r.estimated,
                r.measured,
                r.doc_id,
                r.track_id,
            )
            for r in records
            if r.source != BACKFILL_SOURCE
        ]
        hourly, daily = rollup_increments(records, self.tz)
        dim_names = ", ".join(DIMENSIONS)
        columns = [*DIMENSIONS, *SUM_METRICS, "latency_ms_max"]
        marks = ", ".join(f"${i}" for i in range(1, len(columns) + 2))
        updates = ", ".join(
            [f"{m} = {{t}}.{m} + EXCLUDED.{m}" for m in SUM_METRICS]
            + ["latency_ms_max = GREATEST({t}.latency_ms_max, EXCLUDED.latency_ms_max)"]
        )

        def upsert(table: str) -> str:
            return (
                f"INSERT INTO {table} (bucket, {', '.join(columns)}) VALUES ({marks}) "
                f"ON CONFLICT (bucket, {dim_names}) DO UPDATE SET "
                + updates.format(t=table)
            )

        def rows(increments: dict[tuple, dict[str, float]]) -> list[tuple]:
            return [
                (*key, *(row[m] for m in SUM_METRICS), row["latency_ms_max"])
                for key, row in sorted(increments.items(), key=lambda kv: str(kv[0]))
            ]

        call_marks = (
            ", ".join(f"${i}" for i in range(1, len(calls[0]) + 1)) if calls else ""
        )
        async with await self._connection() as conn:
            async with conn.transaction():
                if calls:
                    await conn.executemany(
                        f"""INSERT INTO {_TABLE_CALLS} (ts, {dim_names}, client, host,
                            prompt_tokens, completion_tokens, reasoning_tokens,
                            cached_tokens, cost, latency_ms, status, cache_hit,
                            estimated, measured, doc_id, track_id)
                            VALUES ({call_marks})""",
                        calls,
                    )
                if hourly:
                    await conn.executemany(upsert(_TABLE_HOURLY), rows(hourly))
                if daily:
                    await conn.executemany(upsert(_TABLE_DAILY), rows(daily))
                if events:
                    await conn.executemany(
                        f"""INSERT INTO {_TABLE_EVENTS} (ts, kind, workspace, purpose, detail)
                            VALUES ($1, $2, $3, $4, $5::jsonb)""",
                        [
                            (
                                datetime.fromtimestamp(e.ts, timezone.utc),
                                e.kind,
                                e.workspace,
                                e.purpose,
                                json.dumps(e.detail, default=str),
                            )
                            for e in events
                        ],
                    )

    @staticmethod
    def _where(
        filters: Optional[dict[str, str]], params: list[Any], conditions: list[str]
    ) -> None:
        for name, value in (filters or {}).items():
            if name not in DIMENSIONS:
                raise ValueError(f"Unknown filter: {name}")
            params.append(value)
            conditions.append(f"{name} = ${len(params)}")

    async def _rows(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        return await self._db.query(sql, params, multirows=True) or []

    async def rollup(
        self, grain, start, end, *, filters=None, group_by=(), exclude_backfill=False
    ):
        table = _TABLE_DAILY if grain == "day" else _TABLE_HOURLY
        groups = list(group_by)
        for g in groups:
            if g != "bucket" and g not in DIMENSIONS:
                raise ValueError(f"Unknown group: {g}")
        params: list[Any] = [start, end]
        conditions = (
            ["bucket >= $1", "bucket <= $2"]
            if grain == "day"
            else ["bucket >= $1", "bucket < $2"]
        )
        self._where(filters, params, conditions)
        if exclude_backfill:
            params.append(BACKFILL_SOURCE)
            conditions.append(f"source <> ${len(params)}")
        sums = ", ".join(
            f"SUM({m})::float8 AS {m}"
            if m == "reported_cost"
            else f"SUM({m})::bigint AS {m}"
            for m in SUM_METRICS
        )
        select = ", ".join(
            groups + [sums, "MAX(latency_ms_max)::bigint AS latency_ms_max"]
        )
        group_sql = f" GROUP BY {', '.join(groups)}" if groups else ""
        return await self._rows(
            f"SELECT {select} FROM {table} WHERE {' AND '.join(conditions)}{group_sql}",
            params,
        )

    async def top_items(self, kind, start, end, *, filters=None, limit=50):
        column = "doc_id" if kind == "document" else "client"
        params: list[Any] = [start, end]
        conditions = ["ts >= $1", "ts < $2", f"{column} IS NOT NULL", f"{column} <> ''"]
        self._where(filters, params, conditions)
        params.append(limit)
        return await self._rows(
            f"""SELECT {column} AS item, workspace, purpose, model,
                COUNT(*) FILTER (WHERE NOT cache_hit)::bigint AS calls,
                COUNT(*) FILTER (WHERE cache_hit)::bigint AS cache_hits,
                SUM(prompt_tokens)::bigint AS prompt_tokens,
                SUM(completion_tokens)::bigint AS completion_tokens,
                COALESCE(SUM(cost), 0)::float8 AS reported_cost,
                COALESCE(SUM(prompt_tokens) FILTER (WHERE cost IS NULL), 0)::bigint AS priced_prompt_tokens,
                COALESCE(SUM(completion_tokens) FILTER (WHERE cost IS NULL), 0)::bigint AS priced_completion_tokens
            FROM {_TABLE_CALLS} WHERE {" AND ".join(conditions)}
            GROUP BY {column}, workspace, purpose, model
            ORDER BY SUM(prompt_tokens + completion_tokens) DESC LIMIT ${len(params)}""",
            params,
        )

    async def calls(
        self,
        start,
        end,
        *,
        filters=None,
        doc_id=None,
        client=None,
        limit=100,
        before_ts=None,
    ):
        params: list[Any] = [start, end]
        conditions = ["ts >= $1", "ts < $2"]
        self._where(filters, params, conditions)
        if doc_id is not None:
            params.append(doc_id)
            conditions.append(f"doc_id = ${len(params)}")
        if client is not None:
            params.append(client)
            conditions.append(f"client = ${len(params)}")
        if before_ts is not None:
            params.append(datetime.fromtimestamp(_epoch(before_ts), timezone.utc))
            conditions.append(f"ts < ${len(params)}")
        params.append(_clamp_limit(limit))
        rows = await self._rows(
            f"""SELECT EXTRACT(EPOCH FROM ts)::float8 AS ts, workspace, purpose, operation,
                source, model, billing, prompt_tokens, completion_tokens,
                reasoning_tokens, cached_tokens, latency_ms, status, cache_hit,
                estimated, measured, doc_id, client
            FROM {_TABLE_CALLS} WHERE {" AND ".join(conditions)}
            ORDER BY ts DESC LIMIT ${len(params)}""",
            params,
        )
        return rows

    async def llm_health(self, start, end, *, filters=None):
        params: list[Any] = [start, end]
        conditions = ["ts >= $1", "ts < $2"]
        self._where(filters, params, conditions)
        rows = await self._rows(
            f"""SELECT
                COUNT(*) FILTER (WHERE NOT cache_hit)::bigint AS calls,
                COUNT(*) FILTER (WHERE cache_hit)::bigint AS cache_hits,
                COUNT(*) FILTER (WHERE NOT cache_hit AND status = 'error')::bigint AS errors,
                COUNT(*) FILTER (WHERE NOT cache_hit AND status = 'truncated')::bigint AS truncated,
                COUNT(*) FILTER (WHERE NOT cache_hit AND status = 'abandoned')::bigint AS abandoned,
                percentile_cont(ARRAY[0.5, 0.95, 0.99]) WITHIN GROUP (ORDER BY latency_ms)
                    FILTER (WHERE NOT cache_hit) AS pct
            FROM {_TABLE_CALLS} WHERE {" AND ".join(conditions)}""",
            params,
        )
        row = dict(rows[0]) if rows else {}
        pct = row.pop("pct", None) or [None, None, None]
        out = {
            k: int(row.get(k) or 0)
            for k in ("calls", "cache_hits", "errors", "truncated", "abandoned")
        }
        return {
            **out,
            "p50_ms": float(pct[0]) if pct[0] is not None else None,
            "p95_ms": float(pct[1]) if pct[1] is not None else None,
            "p99_ms": float(pct[2]) if pct[2] is not None else None,
            **llm_rates(out),
        }

    async def latency_p95(self, start, end, *, filters=None):
        params: list[Any] = [start, end]
        conditions = ["ts >= $1", "ts < $2", "NOT cache_hit"]
        self._where(filters, params, conditions)
        rows = await self._rows(
            f"""SELECT purpose, percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::float8 AS p95
            FROM {_TABLE_CALLS} WHERE {" AND ".join(conditions)} GROUP BY purpose""",
            params,
        )
        return {r["purpose"]: r["p95"] for r in rows}

    async def peak_tokens_per_minute(self, start, end):
        return await self._rows(
            f"""WITH per_minute AS (
                SELECT date_trunc('minute', ts) AS minute, purpose,
                       SUM(prompt_tokens + completion_tokens)::bigint AS tokens
                FROM {_TABLE_CALLS}
                WHERE ts >= $1 AND ts < $2 AND billing = 'system' AND NOT cache_hit
                GROUP BY 1, 2
            ), peak AS (
                SELECT minute FROM per_minute GROUP BY minute
                ORDER BY SUM(tokens) DESC LIMIT 1
            )
            SELECT p.minute, p.purpose, p.tokens FROM per_minute p
            JOIN peak ON p.minute = peak.minute""",
            [start, end],
        )

    async def events(self, start, end, *, workspace=None, limit=500):
        params: list[Any] = [start, end]
        conditions = ["ts >= $1", "ts < $2"]
        if workspace is not None:
            params.append(workspace)
            conditions.append(f"(workspace IS NULL OR workspace = ${len(params)})")
        params.append(limit)
        rows = await self._rows(
            f"""SELECT ts, kind, workspace, purpose, detail FROM {_TABLE_EVENTS}
            WHERE {" AND ".join(conditions)} ORDER BY ts DESC LIMIT ${len(params)}""",
            params,
        )
        for row in rows:
            if isinstance(row.get("detail"), str):
                row["detail"] = json.loads(row["detail"])
        return rows

    async def kv_get(self, key, default=None):
        row = await self._db.query(
            f"SELECT value FROM {_TABLE_KV} WHERE key = $1", [key]
        )
        if not row:
            return default
        value = row["value"]
        return json.loads(value) if isinstance(value, str) else value

    async def kv_set(self, key, value):
        async with await self._connection() as conn:
            await conn.execute(
                f"""INSERT INTO {_TABLE_KV} (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = now()""",
                key,
                json.dumps(value, default=str),
            )

    async def try_lease(self, key, seconds):
        expires = time.time() + seconds
        async with await self._connection() as conn:
            row = await conn.fetchrow(
                f"""INSERT INTO {_TABLE_KV} (key, value, updated_at)
                    VALUES ($1, $2::jsonb, now())
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = now()
                    WHERE ({_TABLE_KV}.value)::text::float8 < $3
                    RETURNING key""",
                f"lease:{key}",
                json.dumps(expires),
                time.time(),
            )
        return row is not None

    async def anonymize_workspace(self, workspace, label):
        async with await self._connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""UPDATE {_TABLE_CALLS}
                        SET workspace = $2, doc_id = NULL, track_id = NULL, client = NULL
                        WHERE workspace = $1""",
                    workspace,
                    label,
                )
                for table in (_TABLE_HOURLY, _TABLE_DAILY):
                    await conn.execute(
                        f"UPDATE {table} SET workspace = $2 WHERE workspace = $1",
                        workspace,
                        label,
                    )
                await conn.execute(
                    f"""UPDATE {_TABLE_EVENTS}
                        SET workspace = $2, detail = '{{}}'::jsonb WHERE workspace = $1""",
                    workspace,
                    label,
                )

    async def purge(self):
        async with await self._connection() as conn:
            await conn.execute(
                f"DELETE FROM {_TABLE_CALLS} WHERE ts < now() - make_interval(days => $1)",
                CALL_RETENTION_DAYS,
            )
            await conn.execute(
                f"DELETE FROM {_TABLE_HOURLY} WHERE bucket < now() - make_interval(days => $1)",
                HOURLY_RETENTION_DAYS,
            )

    async def first_recorded_day(self):
        row = await self._db.query(
            f"SELECT MIN(bucket) AS first FROM {_TABLE_DAILY} WHERE source <> $1",
            [BACKFILL_SOURCE],
        )
        return row["first"] if row else None

    async def known_workspaces(self):
        rows = await self._rows(
            f"""SELECT DISTINCT workspace FROM {_TABLE_DAILY}
                UNION SELECT DISTINCT workspace FROM {_TABLE_DOC_STATUS}""",
            [],
        )
        return sorted(r["workspace"] for r in rows)

    async def llm_cache_batch(self, after, limit):
        params: list[Any] = []
        where = ""
        if after:
            params = list(after)
            where = "WHERE (workspace, id) > ($1, $2)"
        params.append(limit)
        return await self._rows(
            f"""SELECT workspace, id, cache_type, original_prompt, return_value, create_time
                FROM LIGHTRAG_LLM_CACHE {where}
                ORDER BY workspace, id LIMIT ${len(params)}""",
            params,
        )

    async def last_activity(self):
        rows = await self._rows(
            f"SELECT workspace, MAX(bucket) AS last FROM {_TABLE_HOURLY} GROUP BY workspace",
            [],
        )
        return {r["workspace"]: r["last"] for r in rows}

    def _local_day_sql(self, column: str) -> str:
        name = getattr(self.tz, "key", "Asia/Kuala_Lumpur")
        return f"(({column} AT TIME ZONE 'UTC') AT TIME ZONE '{name}')::date"

    async def document_days(self, start, end, workspace=None):
        day = self._local_day_sql("updated_at")
        params: list[Any] = [start, end]
        conditions = [
            "status = 'processed'",
            "id NOT LIKE 'dup-%'",
            f"{day} >= $1",
            f"{day} <= $2",
        ]
        if workspace is not None:
            params.append(workspace)
            conditions.append(f"workspace = ${len(params)}")
        return await self._rows(
            f"""SELECT {day} AS day, workspace, {track_source_sql()} AS source,
                COUNT(*)::bigint AS documents,
                COALESCE(SUM(chunks_count), 0)::bigint AS chunks
            FROM {_TABLE_DOC_STATUS} WHERE {" AND ".join(conditions)}
            GROUP BY 1, 2, 3""",
            params,
        )

    async def document_status_counts(self):
        rows = await self._rows(
            f"""SELECT workspace, status, COUNT(*)::bigint AS n FROM {_TABLE_DOC_STATUS}
                WHERE id NOT LIKE 'dup-%' GROUP BY workspace, status""",
            [],
        )
        out: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:
            out[row["workspace"]][row["status"]] = row["n"]
        return dict(out)

    async def document_paths(self, workspace, ids):
        if not ids:
            return {}
        rows = await self._rows(
            f"""SELECT id, file_path FROM {_TABLE_DOC_STATUS}
                WHERE workspace = $1 AND id = ANY($2::varchar[])""",
            [workspace, ids],
        )
        return {r["id"]: r["file_path"] for r in rows}


def create_usage_store(args: Any) -> Optional[UsageStore]:
    """The usage store for this deployment; version 1 needs Postgres."""
    if getattr(args, "kv_storage", "") == "PGKVStorage":
        return PostgresUsageStore(getattr(args, "vector_storage", None))
    return None
