"""PostgresOpsStore SQL hygiene: bound parameters, inlined literals, timeouts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from lightrag.api.ops.store import (
    MemoryOpsStore,
    Page,
    PostgresOpsStore,
    as_utc,
    failure_signature,
    like_escape,
    sql_in,
)

pytestmark = pytest.mark.offline


class _Conn:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed: list[str] = []
        self.fetched: list[tuple[str, tuple]] = []
        self.readonly = None

    def transaction(self, readonly=False):
        self.readonly = readonly

        @asynccontextmanager
        async def tx():
            yield

        return tx()

    async def execute(self, sql, *args):
        self.executed.append(sql)

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
    return store


async def test_reads_run_read_only_with_a_statement_timeout():
    conn = _Conn()
    await _store(conn).backlog_totals()
    assert conn.readonly is True
    assert conn.executed == ["SET LOCAL statement_timeout = 3000"] * 2


async def test_status_lists_are_inlined_and_search_text_is_bound():
    conn = _Conn()
    store = _store(conn, tables={"LIGHTRAG_WORKSPACE_OWNERS"})
    await store.backlog_page(
        q="o'neil_100%",
        has=frozenset({"failed", "email"}),
        sort="failed",
        descending=True,
        page=Page(50, 100),
        stale_minutes=60,
        email_workspaces=["space1"],
        paused_workspaces=[],
    )
    sql, args = conn.fetched[0]
    assert "status IN ('pending','parsing','analyzing','processing','failed')" in sql
    assert "id NOT LIKE 'dup-%'" in sql
    assert "o'neil" not in sql
    assert args == (60, "%o'neil\\_100\\%%", ["space1"], 50, 100)
    assert "ORDER BY failed DESC NULLS LAST, agg.workspace" in sql
    assert "LEFT JOIN LIGHTRAG_WORKSPACE_OWNERS" in sql
    assert "(now() AT TIME ZONE 'UTC')" in sql


async def test_owner_join_only_when_the_registry_table_exists():
    conn = _Conn()
    await _store(conn).backlog_page(
        q="x",
        has=frozenset(),
        sort="name",
        descending=False,
        page=Page(10, 0),
        stale_minutes=60,
        email_workspaces=[],
        paused_workspaces=[],
    )
    sql, _ = conn.fetched[0]
    assert "LIGHTRAG_WORKSPACE_OWNERS" not in sql and "NULL AS owner" in sql


async def test_sort_and_filters_are_whitelisted():
    store = _store(_Conn())
    base = dict(
        q=None,
        descending=False,
        page=Page(10, 0),
        stale_minutes=60,
        email_workspaces=[],
        paused_workspaces=[],
    )
    with pytest.raises(ValueError):
        await store.backlog_page(has=frozenset(), sort="id; DROP TABLE x", **base)
    with pytest.raises(ValueError):
        await store.backlog_page(has=frozenset({"1=1"}), sort="name", **base)
    with pytest.raises(ValueError):
        await store.legacy_failures(24, "error_msg", False)
    with pytest.raises(ValueError):
        await store.failure_docs(24, "status", "x", Page(10, 0))


async def test_stale_docs_bind_the_threshold_and_compare_naive_utc():
    conn = _Conn()
    await _store(conn).stale_docs(45, Page(20, 40))
    sql, args = conn.fetched[0]
    assert args == (45, 20, 40)
    assert "status IN ('parsing','analyzing','processing')" in sql
    assert "updated_at < (now() AT TIME ZONE 'UTC') - make_interval(mins => $1)" in sql


def test_helpers():
    assert sql_in(["pending", "failed"]) == "('pending','failed')"
    with pytest.raises(ValueError):
        sql_in(["x'); DROP TABLE y; --"])
    assert like_escape("a_b%c\\") == "a\\_b\\%c\\\\"
    naive = as_utc(__import__("datetime").datetime(2026, 11, 3, 1, 0))
    assert naive.utcoffset().total_seconds() == 0 and naive.hour == 1
    assert failure_signature("Timeout after 120s req 7f3a9c21") == failure_signature(
        "Timeout after 300s req 0000aaaa"
    )
    assert failure_signature("a") != failure_signature("b")


async def test_memory_store_kv_cas_and_merge():
    store = MemoryOpsStore()
    assert await store.kv_cas("action:ws", None, {"actor": "a"})
    assert not await store.kv_cas("action:ws", None, {"actor": "b"})
    assert not await store.kv_cas("action:ws", {"actor": "x"}, {"actor": "b"})
    assert await store.kv_cas("action:ws", {"actor": "a"}, {"actor": "b"})
    await store.kv_merge_many({"resync:ws": {"a": 1}})
    await store.kv_merge_many({"resync:ws": {"b": 2}})
    assert await store.kv_get("resync:ws") == {"a": 1, "b": 2}
    await store.kv_delete(["resync:ws"])
    assert await store.kv_get("resync:ws", "gone") == "gone"


class _DedicatedConn:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def is_closed(self):
        return False

    async def fetch(self, sql, *args, timeout=None):
        self.calls.append((sql, args))
        return []

    async def close(self):
        pass


async def test_heartbeats_use_their_own_connection_and_the_database_clock():
    from datetime import datetime, timezone

    store = PostgresOpsStore()
    store._db = SimpleNamespace(pool=None)
    beat = _DedicatedConn()
    store._beat_conn = beat

    @asynccontextmanager
    async def no_pool():
        raise AssertionError("heartbeats must not use the shared pool")
        yield

    store._conn = no_pool
    row = {
        "worker_id": "a" * 32,
        "host": "lightrag-1",
        "pid": 812,
        "pid_create_time": 1.5,
        "mode": "uvicorn",
        "version": "0295",
        "started_at": datetime.now(timezone.utc),
        "leases_held": ["ops:probes"],
        "snapshot": {"rss": 1},
    }
    await store.worker_beat(row, {"cpu_pct": 1.0, "rss": 1})
    await store.supersede_workers("lightrag-1", "a" * 32, row["started_at"], 45)
    await store.worker_exited("a" * 32)
    await store.cleanup_workers(24, 7)
    sqls = [sql for sql, _ in beat.calls]
    assert "heartbeat_at = now()" in sqls[0] and "superseded = false" in sqls[0]
    assert "date_trunc('minute', now())" in sqls[1]
    assert "heartbeat_at < now() - make_interval(secs => $4::float8)" in sqls[2]
    assert "exited_at = now()" in sqls[3]
    assert "make_interval(hours => $1::int)" in sqls[4]
    assert "make_interval(days => $1::int)" in sqls[5]
    assert beat.calls[0][1][0] == "a" * 32 and beat.calls[0][1][7] == ["ops:probes"]


async def test_probe_reads_are_bounded_and_parameterised():
    conn = _Conn()
    store = _store(conn)
    await store.probe_strips(24, 1800)
    await store.probe_series("neo4j'; DROP TABLE x;--", 24, 600)
    await store.probe_recent("neo4j", 50)
    await store.probe_error_classes("neo4j", 24)
    await store.workers(200)
    await store.worker_history("a" * 32, 24, 60)
    assert conn.readonly is True
    for sql, args in conn.fetched:
        assert "DROP TABLE" not in sql
    assert conn.fetched[1][1][0] == "neo4j'; DROP TABLE x;--"
    assert all(s == "SET LOCAL statement_timeout = 3000" for s in conn.executed)


async def test_supersede_with_a_liveness_veto_updates_only_dead_candidates():
    from datetime import datetime, timezone

    class Candidates(_DedicatedConn):
        async def fetch(self, sql, *args, timeout=None):
            self.calls.append((sql, args))
            if sql.lstrip().startswith("SELECT"):
                return [
                    {"worker_id": "a-live", "pid": 10, "pid_create_time": 1.0},
                    {"worker_id": "b-dead", "pid": 11, "pid_create_time": 2.0},
                ]
            return [{"worker_id": "b-dead"}]

    store = PostgresOpsStore()
    store._db = SimpleNamespace(pool=None)
    beat = Candidates()
    store._beat_conn = beat
    count = await store.supersede_workers(
        "lightrag-1",
        "c" * 32,
        datetime.now(timezone.utc),
        45,
        alive=lambda pid, create_time: pid == 10,
    )
    assert count == 1
    select, update = beat.calls
    assert "heartbeat_at < now() - make_interval(secs => $4::float8)" in select[0]
    assert "worker_id = ANY($5::uuid[])" in update[0]
    assert update[1][4] == ["b-dead"]
    # Nothing dead: no UPDATE at all.
    beat.calls.clear()
    assert (
        await store.supersede_workers(
            "lightrag-1",
            "c" * 32,
            datetime.now(timezone.utc),
            45,
            alive=lambda *a: True,
        )
        == 0
    )
    assert len(beat.calls) == 1
