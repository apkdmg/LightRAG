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
