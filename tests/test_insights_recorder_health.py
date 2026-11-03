"""Read-only health accessors of the Insights usage recorder (heartbeat input)."""

from __future__ import annotations

import time

import pytest

from lightrag.api.insights.recorder import UsageRecorder
from lightrag.api.insights.store import MemoryUsageStore
from lightrag.usage import SystemEvent
from tests.test_insights import TODAY, TZ, _record

pytestmark = pytest.mark.offline


async def test_buffer_and_last_flush():
    recorder = UsageRecorder(MemoryUsageStore(TZ))
    assert recorder.buffer_len == 0 and recorder.last_flush_ok_at is None
    recorder.record(_record(TODAY))
    recorder.event(SystemEvent(kind="cache_cleared", workspace="alpha"))
    assert recorder.buffer_len == 2
    before = time.time()
    await recorder.flush()
    assert recorder.buffer_len == 0
    assert recorder.last_flush_ok_at is not None and recorder.last_flush_ok_at >= before
    health = recorder.health()
    assert health["buffer"] == 0 and health["flush_failures"] == 0
    assert health["running"] is False  # not started


async def test_flush_failures_are_counted_and_last_ok_is_kept():
    store = MemoryUsageStore(TZ)
    recorder = UsageRecorder(store)
    await recorder.flush()
    ok_at = recorder.last_flush_ok_at

    async def broken(records, events):
        raise RuntimeError("db down")

    store.write = broken
    recorder.record(_record(TODAY))
    await recorder.flush()
    await recorder.flush()
    assert recorder.flush_failures == 2
    assert recorder.last_flush_ok_at == ok_at
    assert recorder.buffer_len == 1


async def test_dropped_events_and_lost_rows_are_counted_cumulatively():
    recorder = UsageRecorder(MemoryUsageStore(TZ), max_buffer=1)
    recorder.event(SystemEvent(kind="cache_cleared", workspace="alpha"))
    recorder.event(SystemEvent(kind="cache_cleared", workspace="beta"))
    recorder.event(SystemEvent(kind="cache_cleared", workspace="gamma"))
    assert recorder.dropped_events == 2
    recorder.record(_record(TODAY))
    recorder.record(_record(TODAY))
    assert recorder.lost_rows == 1
    await recorder.flush()
    # The pending lost count is written and reset; the health total stays.
    assert recorder.lost_rows == 1 and recorder.health()["lost_rows"] == 1
    assert recorder.health()["dropped_events"] == 2


async def test_running_while_started():
    recorder = UsageRecorder(MemoryUsageStore(TZ), flush_seconds=0.01)
    recorder.start()
    try:
        assert recorder.health()["running"] is True
    finally:
        await recorder.stop()
    assert recorder.health()["running"] is False
