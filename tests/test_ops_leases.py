"""Renewable {holder, exp} leases and the leased singleton loop."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.store import MemoryOpsStore, PostgresOpsStore

pytestmark = pytest.mark.offline


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_holder_is_stored_and_renewed_by_the_holder():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    assert await store.try_lease("ops:scan", "worker-a", 90)
    assert store.kv["lease:ops:scan"] == {"holder": "worker-a", "exp": 1090.0}
    clock.now += 60
    assert await store.try_lease("ops:scan", "worker-a", 90)
    assert store.kv["lease:ops:scan"]["exp"] == 1150.0


async def test_takeover_only_after_expiry():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    assert await store.try_lease("ops:scan", "worker-a", 90)
    clock.now += 89
    assert not await store.try_lease("ops:scan", "worker-b", 90)
    clock.now += 2
    assert await store.try_lease("ops:scan", "worker-b", 90)
    assert store.kv["lease:ops:scan"]["holder"] == "worker-b"
    assert not await store.try_lease("ops:scan", "worker-a", 90)


async def test_release_only_by_the_holder():
    store = MemoryOpsStore(clock=Clock())
    await store.try_lease("ops:scan", "worker-a", 90)
    await store.release_lease("ops:scan", "worker-b")
    assert "lease:ops:scan" in store.kv
    await store.release_lease("ops:scan", "worker-a")
    assert "lease:ops:scan" not in store.kv


async def test_cycle_runs_only_while_the_lease_is_ours():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    ran: list[str] = []

    async def cycle(loop):
        ran.append(loop.holder)

    a = LeasedLoop(
        store, "ops:test", "a", cycle, ttl_s=90, interval_s=60, cycle_timeout_s=30
    )
    b = LeasedLoop(
        store, "ops:test", "b", cycle, ttl_s=90, interval_s=60, cycle_timeout_s=30
    )
    assert await a.run_once()
    assert not await b.run_once()
    clock.now += 60
    assert await a.run_once()  # renewed at the start of its cycle
    clock.now += 80
    assert not await b.run_once()  # a renewed 80 s ago, still valid
    clock.now += 20
    assert await b.run_once()  # expired: taken over
    assert ran == ["a", "a", "b"]
    assert a.held is True and b.held is True  # a only learns on its next renewal
    assert not await a.run_once()
    assert a.held is False


async def test_cycle_aborts_when_renewal_fails():
    store = MemoryOpsStore(clock=Clock())
    store.fail_reads = ConnectionError("database down")
    ran = []

    async def cycle(loop):
        ran.append(1)

    loop = LeasedLoop(
        store, "ops:test", "a", cycle, ttl_s=90, interval_s=60, cycle_timeout_s=30
    )
    assert not await loop.run_once()
    assert ran == [] and loop.held is False


async def test_long_cycle_stops_before_writing_when_the_lease_is_lost():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    wrote = []

    async def cycle(loop):
        # Another worker took over while this cycle was busy.
        store.kv["lease:ops:test"] = {"holder": "other", "exp": clock.now + 90}
        if await loop.still_held():
            wrote.append(1)

    loop = LeasedLoop(
        store, "ops:test", "a", cycle, ttl_s=90, interval_s=60, cycle_timeout_s=30
    )
    assert await loop.run_once()
    assert wrote == []


async def test_cycle_failures_and_timeouts_never_raise():
    store = MemoryOpsStore(clock=Clock())

    async def broken(loop):
        raise RuntimeError("boom")

    async def slow(loop):
        await asyncio.sleep(1)

    loop = LeasedLoop(
        store, "ops:a", "a", broken, ttl_s=2, interval_s=1, cycle_timeout_s=1
    )
    assert await loop.run_once()
    assert loop.failures == 1
    loop = LeasedLoop(
        store, "ops:b", "a", slow, ttl_s=0.5, interval_s=1, cycle_timeout_s=0.05
    )
    assert await loop.run_once()
    assert loop.failures == 1


@pytest.mark.skipif(
    not hasattr(asyncio.Task, "cancelling"), reason="needs Task.cancelling()"
)
async def test_a_stray_cancelled_error_inside_a_cycle_does_not_end_the_loop():
    store = MemoryOpsStore(clock=Clock())
    ran = []

    async def cycle(loop):
        ran.append(1)
        if len(ran) == 1:
            fut = asyncio.get_running_loop().create_future()
            fut.cancel()  # someone else's future, cancelled under us
            await fut

    loop = LeasedLoop(
        store, "ops:c", "a", cycle, ttl_s=2, interval_s=0.01, cycle_timeout_s=1
    )
    task = loop.start()
    for _ in range(100):
        if len(ran) >= 2:
            break
        await asyncio.sleep(0.01)
    try:
        assert not task.done()
        assert len(ran) >= 2 and loop.failures == 1
    finally:
        await loop.stop()
    # A real stop still cancels the task.
    assert task.cancelled()


def test_lease_must_outlast_the_cycle():
    with pytest.raises(ValueError):
        LeasedLoop(
            MemoryOpsStore(),
            "x",
            "a",
            None,
            ttl_s=30,
            interval_s=60,
            cycle_timeout_s=30,
        )


async def test_start_and_stop_release_the_lease():
    store = MemoryOpsStore()
    ran = asyncio.Event()

    async def cycle(loop):
        ran.set()

    loop = LeasedLoop(
        store, "ops:test", "a", cycle, ttl_s=90, interval_s=60, cycle_timeout_s=30
    )
    loop.start()
    await asyncio.wait_for(ran.wait(), 2)
    assert store.kv["lease:ops:test"]["holder"] == "a"
    await loop.stop()
    assert "lease:ops:test" not in store.kv


class _LeaseConn:
    def __init__(self):
        self.calls = []

    def is_closed(self):
        return False

    async def fetch(self, sql, *args, timeout=None):
        self.calls.append((sql, args))
        return [{"key": args[0]}]

    async def close(self):
        pass


async def test_postgres_lease_sql_uses_the_database_clock_and_a_dedicated_connection(
    monkeypatch,
):
    store = PostgresOpsStore()
    conn = _LeaseConn()
    store._lease_conn = conn
    store._db = SimpleNamespace(pool=None)

    @asynccontextmanager
    async def no_pool():
        raise AssertionError("leases must not use the shared pool")
        yield

    monkeypatch.setattr(store, "_conn", no_pool)
    assert await store.try_lease("ops:scan", "worker-a", 90)
    sql, args = conn.calls[0]
    assert args == ("lease:ops:scan", "worker-a", 90.0)
    assert "clock_timestamp()" in sql
    assert "value->>'holder' = $2::text" in sql
    assert "value->>'exp')::float8 < extract(epoch FROM clock_timestamp())" in sql
    await store.release_lease("ops:scan", "worker-a")
    assert conn.calls[1][1] == ("lease:ops:scan", "worker-a")
