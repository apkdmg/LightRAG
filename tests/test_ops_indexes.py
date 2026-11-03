"""Background builder for the ops indexes on large existing tables."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from lightrag.api.ops import indexes

pytestmark = pytest.mark.offline


class FakeConn:
    """Answers the builder's catalog queries from a small model."""

    def __init__(self, *, tables=None, states=None, locked=False, fail_create=()):
        self.tables = set(
            tables
            or ("lightrag_doc_status", "lightrag_usage_calls", "lightrag_usage_events")
        )
        self.states = dict(states or {})
        self.locked = locked
        self.fail_create = set(fail_create)
        self.statements: list[str] = []

    async def fetchval(self, sql, *args):
        if "pg_try_advisory_lock" in sql:
            assert args == (indexes.ADVISORY_LOCK_KEY,)
            return not self.locked
        if "to_regclass" in sql:
            return 1 if args[0] in self.tables else None
        raise AssertionError(sql)

    async def fetchrow(self, sql, name):
        return self.states.get(name)

    async def execute(self, sql, *args):
        self.statements.append(sql)
        if sql.startswith("CREATE INDEX"):
            name = sql.split()[6]
            if name in self.fail_create:
                raise TimeoutError("canceling statement due to lock timeout")
            self.states[name] = {
                "oid": 1,
                "indisvalid": True,
                "indisready": True,
                "building": False,
            }
        if sql.startswith("DROP INDEX"):
            self.states.pop(sql.split()[-1], None)


def test_spec_indexes_and_never_the_pagination_ones():
    names = [spec.name for spec in indexes.OPS_INDEXES]
    assert names == [
        "idx_lightrag_doc_status_inflight",
        "idx_lightrag_doc_status_dup",
        "idx_lightrag_doc_status_id",
        "idx_lightrag_usage_calls_doc",
        "idx_lightrag_usage_calls_err",
        "idx_usage_events_kind_ts",
    ]
    assert not any("workspace_status" in n for n in names)
    for spec in indexes.OPS_INDEXES:
        assert spec.create_sql().startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS ")
    inflight = indexes.OPS_INDEXES[0].definition
    assert "('pending','parsing','analyzing','processing','failed')" in inflight
    assert "id NOT LIKE 'dup-%'" in inflight


async def test_builds_missing_indexes_concurrently_with_a_lock_timeout():
    conn = FakeConn()
    outcome = await indexes.build_indexes(conn)
    assert set(outcome.values()) == {"created"}
    assert conn.statements[0] == f"SET lock_timeout = {indexes.LOCK_TIMEOUT_MS}"
    creates = [s for s in conn.statements if s.startswith("CREATE")]
    assert len(creates) == 6 and all("CONCURRENTLY" in s for s in creates)
    assert conn.statements[-1] == "SELECT pg_advisory_unlock($1)"


async def test_valid_indexes_are_left_alone():
    valid = {"oid": 1, "indisvalid": True, "indisready": True, "building": False}
    conn = FakeConn(states={s.name: valid for s in indexes.OPS_INDEXES})
    assert set((await indexes.build_indexes(conn)).values()) == {"valid"}
    assert not [s for s in conn.statements if s.startswith(("CREATE", "DROP"))]


async def test_invalid_index_is_dropped_and_rebuilt_unless_still_building():
    invalid = {"oid": 1, "indisvalid": False, "indisready": False, "building": False}
    building = {**invalid, "building": True}
    conn = FakeConn(
        states={
            "idx_lightrag_doc_status_id": invalid,
            "idx_lightrag_doc_status_dup": building,
        }
    )
    outcome = await indexes.build_indexes(conn)
    assert outcome["idx_lightrag_doc_status_id"] == "rebuilt"
    assert outcome["idx_lightrag_doc_status_dup"] == "building"
    assert (
        "DROP INDEX CONCURRENTLY IF EXISTS idx_lightrag_doc_status_id"
        in conn.statements
    )
    assert not any(
        "DROP INDEX CONCURRENTLY IF EXISTS idx_lightrag_doc_status_dup" == s
        for s in conn.statements
    )


async def test_another_builder_holding_the_lock_skips():
    conn = FakeConn(locked=True)
    assert set((await indexes.build_indexes(conn)).values()) == {"locked"}
    assert conn.statements == []


async def test_missing_tables_skip_and_failures_continue():
    conn = FakeConn(
        tables={"lightrag_doc_status"}, fail_create={"idx_lightrag_doc_status_dup"}
    )
    outcome = await indexes.build_indexes(conn)
    assert outcome["idx_lightrag_doc_status_dup"] == "failed"
    assert outcome["idx_lightrag_doc_status_id"] == "created"
    assert outcome["idx_lightrag_usage_calls_doc"] == "skipped"
    assert conn.statements[-1] == "SELECT pg_advisory_unlock($1)"


def test_window_parsing_and_wait():
    assert indexes.parse_window("") is None
    assert indexes.parse_window("nonsense") is None
    assert indexes.parse_window("25:00-01:00") is None
    window = indexes.parse_window("01:00-05:00")
    assert window == (time(1, 0), time(5, 0))
    tz = ZoneInfo("Asia/Kuala_Lumpur")
    assert (
        indexes.seconds_until_window(window, datetime(2026, 11, 3, 2, 0, tzinfo=tz))
        == 0
    )
    assert (
        indexes.seconds_until_window(window, datetime(2026, 11, 3, 0, 30, tzinfo=tz))
        == 1800
    )
    assert (
        indexes.seconds_until_window(window, datetime(2026, 11, 3, 6, 0, tzinfo=tz))
        == 19 * 3600
    )
    wrap = indexes.parse_window("23:00-02:00")
    assert (
        indexes.seconds_until_window(wrap, datetime(2026, 11, 3, 1, 0, tzinfo=tz)) == 0
    )
    assert (
        indexes.seconds_until_window(wrap, datetime(2026, 11, 3, 23, 30, tzinfo=tz))
        == 0
    )
    assert (
        indexes.seconds_until_window(None, datetime(2026, 11, 3, 12, 0, tzinfo=tz)) == 0
    )


class _Store:
    def __init__(self, conn=None, fail=False):
        self.conn = conn
        self.fail = fail
        self.opened = 0

    @asynccontextmanager
    async def dedicated_connection(self):
        self.opened += 1
        if self.fail:
            raise OSError("no route to host")
        yield self.conn


async def test_runner_never_raises_and_skips_the_window_when_nothing_is_missing():
    assert (
        await indexes.run_index_builder(
            _Store(fail=True), delay_s=0, window="", attempts=1
        )
        is None
    )
    valid = {"oid": 1, "indisvalid": True, "indisready": True, "building": False}
    conn = FakeConn(states={s.name: valid for s in indexes.OPS_INDEXES})
    # A window hours away must not be waited for when every index is valid.
    outcome = await indexes.run_index_builder(
        _Store(conn), delay_s=0, window="03:00-03:01"
    )
    assert set(outcome.values()) == {"valid"}


async def test_runner_builds_missing_indexes():
    store = _Store(FakeConn())
    outcome = await indexes.run_index_builder(store, delay_s=0, window="")
    assert set(outcome.values()) == {"created"}
    assert store.opened == 2


class FailingBuildConn(FakeConn):
    """CREATE fails after the catalog entry exists (CIC phase 2 or 3)."""

    def __init__(self, *, fail_times=1, **kwargs):
        super().__init__(**kwargs)
        self.fail_times = fail_times

    async def execute(self, sql, *args):
        if sql.startswith("CREATE INDEX") and self.fail_times > 0:
            self.statements.append(sql)
            self.fail_times -= 1
            name = sql.split()[6]
            self.states[name] = {
                "oid": 1,
                "indisvalid": False,
                "indisready": True,
                "building": False,
            }
            raise TimeoutError("canceling statement due to lock timeout")
        await super().execute(sql, *args)


async def test_a_failed_build_drops_its_invalid_index_at_once():
    spec = indexes.OPS_INDEXES[0]
    conn = FailingBuildConn(fail_times=1)
    outcome = await indexes.build_indexes(conn, (spec,))
    assert outcome == {spec.name: "failed"}
    # Writers must not keep maintaining an index the planner never uses.
    assert f"DROP INDEX CONCURRENTLY IF EXISTS {spec.name}" in conn.statements
    assert spec.name not in conn.states
    assert conn.statements[-1] == "SELECT pg_advisory_unlock($1)"


async def test_a_failed_build_leaves_an_index_another_session_is_building():
    spec = indexes.OPS_INDEXES[0]

    class Racing(FailingBuildConn):
        async def fetchrow(self, sql, name):
            state = self.states.get(name)
            return {**state, "building": True} if state else None

    conn = Racing(fail_times=1)
    assert await indexes.build_indexes(conn, (spec,)) == {spec.name: "failed"}
    assert not [s for s in conn.statements if s.startswith("DROP")]


async def test_runner_retries_failed_builds_with_backoff(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(indexes.asyncio, "sleep", fake_sleep)
    conn = FailingBuildConn(fail_times=2)
    outcome = await indexes.run_index_builder(
        _Store(conn), delay_s=5, window="", retry_delay_s=60, attempts=4
    )
    # The first attempt fails two builds (and drops them); the retry builds them.
    assert outcome["idx_lightrag_doc_status_inflight"] == "created"
    assert outcome["idx_lightrag_doc_status_dup"] == "created"
    assert set(outcome.values()) == {"created", "valid"}
    assert sleeps == [5, 60]


async def test_runner_gives_up_after_the_last_attempt(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(indexes.asyncio, "sleep", fake_sleep)
    store = _Store(fail=True)
    assert (
        await indexes.run_index_builder(
            store, delay_s=0, window="", retry_delay_s=10, attempts=3
        )
        is None
    )
    assert store.opened == 3 and sleeps == [0, 10, 20]


async def test_runner_does_not_retry_when_another_process_holds_the_lock():
    store = _Store(FakeConn(locked=True))
    outcome = await indexes.run_index_builder(
        store, delay_s=0, window="", retry_delay_s=3600, attempts=4
    )
    assert set(outcome.values()) == {"locked"}
    assert store.opened == 2
