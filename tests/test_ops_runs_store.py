"""Run-tracking SQL: Postgres statements are parameterised and whitelisted,
and the memory store follows the same rules (final status sticks, orphaning,
keyset paging, durable failure groups, open pauses, retention, deletion)."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lightrag.api.ops.store import (
    DURABLE_GROUP_COLUMNS,
    MemoryOpsStore,
    Page,
    PostgresOpsStore,
)

pytestmark = pytest.mark.offline


class _Conn:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed: list[str] = []
        self.fetched: list[tuple[str, tuple]] = []
        self.many: list[tuple[str, list]] = []
        self.readonly = None

    def transaction(self, readonly=False):
        self.readonly = readonly

        @asynccontextmanager
        async def tx():
            yield

        return tx()

    async def execute(self, sql, *args):
        self.executed.append(sql)

    async def executemany(self, sql, rows):
        self.many.append((sql, list(rows)))

    async def fetch(self, sql, *args):
        self.fetched.append((sql, args))
        return self.rows


def _store(conn: _Conn, tables=()) -> PostgresOpsStore:
    store = PostgresOpsStore()
    store._db = SimpleNamespace()

    @asynccontextmanager
    async def fake():
        yield conn

    store._conn = fake
    store._tables = set(tables)
    store._tables_at = float("inf")
    store.doc_status_ready = True
    return store


def _run(run_id: str, status: str = "running", **extra) -> dict:
    now = time.time()
    return {
        "run_id": run_id,
        "workspace": "fsktm",
        "kind": "process",
        "trigger": "upload",
        "status": status,
        "worker_id": "0123456789abcdef0123456789abcdef",
        "pid": 7,
        "started_at": now - 60,
        "heartbeat_at": now,
        "progress_at": now,
        "finished_at": None,
        "docs_total": 2,
        "docs_done": 1,
        "docs_failed": 0,
        "stage": "extract",
        "label": "x [2 files]",
        "error_kind": None,
        "error_summary": None,
        "request_id": None,
        **extra,
    }


async def test_run_upsert_keeps_final_statuses_and_binds_values():
    conn = _Conn()
    await _store(conn).runs_write([_run("a" * 32)])
    [(sql, rows)] = conn.many
    assert "WHERE r.status NOT IN ('succeeded','failed','cancelled','paused')" in sql
    assert "was_orphaned = r.was_orphaned OR r.status = 'orphaned'" in sql
    assert "$1::uuid" in sql and "to_timestamp($8::float8)" in sql
    assert "fsktm" not in sql
    assert rows[0][0] == "a" * 32 and rows[0][1] == "fsktm"
    assert conn.executed == ["SET LOCAL statement_timeout = 5000"]


async def test_failure_rows_are_bound_and_capped():
    conn = _Conn()
    await _store(conn).doc_failures_write(
        [
            {
                "ts": time.time(),
                "workspace": "w",
                "doc_id": "d" * 400,
                "run_id": None,
                "stage": "extract",
                "error_kind": "timeout",
                "message": "m" * 5000,
                "attempt": 3,
            }
        ]
    )
    [(sql, rows)] = conn.many
    assert "$5::uuid" in sql
    assert len(rows[0][2]) == 255 and len(rows[0][9]) == 1024


async def test_group_columns_are_whitelisted():
    conn = _Conn()
    store = _store(conn)
    with pytest.raises(ValueError):
        await store.failure_groups(24, "message; DROP TABLE x", False)
    with pytest.raises(ValueError):
        await store.failure_group_docs(24, "nope", "k", Page(10, 0))
    for group in DURABLE_GROUP_COLUMNS:
        await store.failure_groups(24, group, True)
    assert all("$1::int" in sql for sql, _ in conn.fetched)
    assert all(conn.readonly is True for _ in [0])


async def test_drill_joins_doc_status_after_paging_and_binds_the_key():
    conn = _Conn()
    store = _store(conn, tables={"LIGHTRAG_DOC_STATUS"})
    await store.failure_group_docs(24, "error_kind", "o'neil", Page(50, 100))
    # The page query, then (an empty page past the end) the count.
    (sql, args), (count_sql, count_args) = conn.fetched[-2:]
    assert args == (24, "o'neil", 50, 100) and count_args == (24, "o'neil")
    assert sql.index("LIMIT $3 OFFSET $4") < sql.index("LEFT JOIN LIGHTRAG_DOC_STATUS")
    assert "error_kind <> 'cancelled'" in sql
    await store.failure_group_docs(24, "error_kind", "cancelled", Page(50, 0))
    assert "error_kind <> 'cancelled'" not in conn.fetched[-1][0]


async def test_runs_page_builds_only_the_filters_it_needs():
    conn = _Conn()
    store = _store(conn)
    moment = datetime.now(timezone.utc)
    await store.runs_page(
        workspace="fsktm",
        kind=None,
        status="failed",
        before=(moment, "b" * 32),
        limit=51,
    )
    sql, args = conn.fetched[-1]
    assert "r.kind" not in sql.split("WHERE", 1)[1].split("ORDER")[0]
    assert "(r.started_at, r.run_id) < ($3, $4::uuid)" in sql
    assert args == ("fsktm", "failed", moment, "b" * 32, 51)
    await store.runs_page(workspace=None, kind=None, status=None, before=None, limit=5)
    assert "WHERE" not in conn.fetched[-1][0].split("ON w.worker_id")[1]


async def test_open_pauses_skip_cleared_and_non_objects():
    conn = _Conn(
        rows=[
            {
                "key": "pause:a:extraction",
                "value": '{"workspace": "a", "cleared_at": null}',
            }
        ]
    )
    [pause] = await _store(conn).open_pauses(10)
    assert pause["key"] == "pause:a:extraction" and pause["workspace"] == "a"
    sql = conn.fetched[-1][0]
    assert "key LIKE 'pause:%'" in sql and "cleared_at" in sql


# -- memory store semantics ------------------------------------------------------


async def test_memory_store_runs_lifecycle_and_paging():
    store = MemoryOpsStore()
    await store.runs_write(
        [_run("a" * 32), _run("b" * 32, started_at=time.time() - 30)]
    )
    await store.runs_write(
        [_run("a" * 32, status="succeeded", finished_at=time.time())]
    )
    await store.runs_write([_run("a" * 32)])  # late write after the final one
    assert store.run_rows["a" * 32]["status"] == "succeeded"
    assert await store.runs_orphan(["a" * 32, "b" * 32]) == ["b" * 32]
    running = await store.running_runs(10)
    assert running == []
    page = await store.runs_page(
        workspace=None, kind=None, status=None, before=None, limit=1
    )
    assert [r["run_id"] for r in page] == ["b" * 32]
    cursor = (page[0]["started_at"], page[0]["run_id"])
    rest = await store.runs_page(
        workspace=None, kind=None, status=None, before=cursor, limit=5
    )
    assert [r["run_id"] for r in rest] == ["a" * 32]
    last = await store.last_runs(["fsktm", "other"])
    assert set(last) == {"fsktm"}


async def test_memory_store_retention_and_deletion():
    now = time.time()
    store = MemoryOpsStore()
    await store.runs_write(
        [
            _run("a" * 32, started_at=now - 40 * 86400),
            _run("b" * 32),
            {**_run("c" * 32), "workspace": "keep"},
        ]
    )
    await store.doc_failures_write(
        [
            {
                "ts": now - 100 * 86400,
                "workspace": "fsktm",
                "doc_id": "d1",
                "stage": "parse",
                "error_kind": "timeout",
                "attempt": 1,
            },
            {
                "ts": now,
                "workspace": "fsktm",
                "doc_id": "d2",
                "stage": "parse",
                "error_kind": "timeout",
                "attempt": 1,
            },
        ]
    )
    await store.runs_cleanup(30, 90)
    assert set(store.run_rows) == {"b" * 32, "c" * 32}
    assert [r["doc_id"] for r in store.failure_rows] == ["d2"]
    store.kv["pause:fsktm:extraction"] = {"workspace": "fsktm"}
    await store.forget_workspace("fsktm")
    assert set(store.run_rows) == {"c" * 32}
    assert store.failure_rows == []
    assert "pause:fsktm:extraction" not in store.kv


async def test_merge_existing_never_creates_keys():
    store = MemoryOpsStore()
    store.kv["pause:a:x"] = {"workspace": "a", "cleared_at": None}
    await store.kv_merge_existing(
        {"pause:a:x": {"cleared_at": 1.0}, "pause:b:x": {"cleared_at": 1.0}}
    )
    assert store.kv["pause:a:x"]["cleared_at"] == 1.0
    assert "pause:b:x" not in store.kv


async def test_run_events_mark_orphaned_runs():
    store = MemoryOpsStore()
    await store.runs_write([_run("a" * 32)])
    await store.runs_orphan(["a" * 32])
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    [event] = await store.run_events(end - timedelta(hours=1), end, 10)
    assert event["run_id"] == "a" * 32 and event["status"] == "orphaned"
