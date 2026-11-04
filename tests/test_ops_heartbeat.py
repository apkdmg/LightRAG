"""Worker heartbeat: rows, samples, superseded rows, liveness, cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lightrag.api.ops import heartbeat as hb_module
from lightrag.api.ops.health import order_workers, worker_item
from lightrag.api.ops.heartbeat import (
    ALIVE_WITHIN_S,
    Heartbeat,
    LoopLagMonitor,
    ProcessSampler,
)
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import MemoryOpsStore

pytestmark = pytest.mark.offline


class Clock:
    def __init__(self, now: float | None = None):
        self.now = now if now is not None else time.time()

    def __call__(self) -> float:
        return self.now


class FakeRag:
    def __init__(self, workspace: str, running: int = 0, queued: int = 0):
        self.workspace = workspace
        self.running = running
        self.queued = queued

    def _stats(self, name: str) -> dict:
        return {
            "available": True,
            "queue_name": name,
            "max_async": 4,
            "running": self.running,
            "in_flight": self.running + self.queued,
            "queued": self.queued,
            "secret_field": "never copied",
        }

    async def get_llm_queue_status(self, include_base: bool = True):
        return {"extract": self._stats("extract"), "query": {"available": False}}

    async def get_embedding_queue_status(self):
        return self._stats("embed")

    async def get_rerank_queue_status(self):
        return {"available": False}


class FakeRecorder:
    def __init__(self):
        self.lost = 0
        self.dropped = 0

    def health(self):
        return {
            "buffer": 3,
            "last_flush_ok_at": time.time() - 2,
            "flush_failures": 1,
            "lost_rows": self.lost,
            "dropped_events": self.dropped,
            "running": True,
        }


def _heartbeat(store, *, app_state=None, worker_id="a" * 32, leases=None, rag=None):
    state = app_state or SimpleNamespace()
    return Heartbeat(
        worker_id=worker_id,
        mode="uvicorn",
        version="0295",
        store=store,
        reader=NamespaceReader(),
        app_state=state,
        leases=leases or (lambda: ["ops:probes"]),
        data_rag=lambda: rag,
    )


async def test_beat_writes_the_worker_row_and_a_sample():
    store = MemoryOpsStore()
    manager = SimpleNamespace(
        loaded_instances=lambda: [FakeRag("fsktm", 2, 5), FakeRag("space1", 1, 0)]
    )
    queue = SimpleNamespace(active_drainers=lambda: ["space1"])
    state = SimpleNamespace(
        workspace_manager=manager,
        insights_recorder=FakeRecorder(),
        email_resync_queue=queue,
    )
    beat = _heartbeat(store, app_state=state, rag=FakeRag("", 0, 1))
    assert await beat.beat_once() is True
    row = store.worker_rows["a" * 32]
    assert row["mode"] == "uvicorn" and row["version"] == "0295"
    assert row["pid"] == beat.pid and row["leases_held"] == ["ops:probes"]
    snapshot = row["snapshot"]
    for key in ("rss", "cpu_pct", "threads", "loop_lag_ms", "manager_state"):
        assert key in snapshot
    assert snapshot["manager_state"] == "local"
    assert snapshot["instances"] == 3
    assert snapshot["queues"]["llm"] == {"running": 3, "in_flight": 9, "queued": 6}
    assert snapshot["queues"]["rerank"] is None
    assert snapshot["top_instances"][0] == {
        "workspace": "fsktm",
        "running": 4,
        "queued": 10,
    }
    assert snapshot["recorder"]["buffer"] == 3
    assert snapshot["recorder"]["last_flush_ok_at"].endswith("+00:00")
    assert snapshot["drainers"] == [
        {"workspace": "space1", "state": "draining", "backoff_until": None}
    ]
    # StatusCard-shaped queues of the server's own instance, whitelisted.
    assert snapshot["embedding_queue_status"]["queue_name"] == "embed"
    assert "secret_field" not in json.dumps(snapshot)
    # Never the command line or the environment.
    assert "cmdline" not in json.dumps(snapshot)
    assert "environ" not in json.dumps(snapshot)
    assert len(store.sample_rows) == 1
    await beat.beat_once()
    assert len(store.sample_rows) == 1  # one sample a minute


async def test_superseded_rows_from_an_earlier_boot():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    old = _heartbeat(store, worker_id="b" * 32)
    old.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    other_host = _heartbeat(store, worker_id="c" * 32)
    other_host.host = "another-host"
    other_host.started_at = old.started_at
    await old.beat_once()
    await other_host.beat_once()
    # The old worker stops beating; a new one starts a minute later.
    clock.now += ALIVE_WITHIN_S + 15
    new = _heartbeat(store, worker_id="d" * 32)
    await new.beat_once()
    assert store.worker_rows["b" * 32]["superseded"] is True
    assert store.worker_rows["c" * 32]["superseded"] is False  # other host
    assert store.worker_rows["d" * 32]["superseded"] is False
    # A beat from a worker that was only stalled clears the mark.
    await old.beat_once()
    assert store.worker_rows["b" * 32]["superseded"] is False


async def test_a_live_sibling_is_never_superseded():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    sibling = _heartbeat(store, worker_id="e" * 32)
    sibling.started_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    await sibling.beat_once()
    clock.now += 10
    await _heartbeat(store, worker_id="f" * 32).beat_once()
    assert store.worker_rows["e" * 32]["superseded"] is False


async def test_a_hung_sibling_of_the_same_boot_is_never_superseded():
    # Gunicorn forks its workers a few ms apart; one of them hangs.
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    boot = datetime.now(timezone.utc) - timedelta(minutes=10)
    hung = _heartbeat(store, worker_id="1" * 32)
    hung.started_at = boot
    hung.pid = os.getppid()  # a process that is still running
    hung.process.create_time = None
    busy = _heartbeat(store, worker_id="2" * 32)
    busy.started_at = boot + timedelta(milliseconds=50)
    await hung.beat_once()
    await busy.beat_once()
    for _ in range(5):  # the sibling keeps beating long past 45 s
        clock.now += 15
        await busy.beat_once()
    assert store.worker_rows["1" * 32]["superseded"] is False
    item = worker_item(store.worker_rows["1" * 32] | {"heartbeat_age_s": 75}, None)
    assert item["alive"] is False and item["superseded"] is False


async def test_an_earlier_boot_whose_process_still_runs_is_not_superseded():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    old = _heartbeat(store, worker_id="3" * 32)
    old.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    old.pid = os.getppid()
    old.process.create_time = None
    await old.beat_once()
    clock.now += ALIVE_WITHIN_S + 15
    await _heartbeat(store, worker_id="4" * 32).beat_once()
    assert store.worker_rows["3" * 32]["superseded"] is False


def test_process_alive():
    assert hb_module.process_alive(os.getpid(), None) is False  # a reused pid
    assert hb_module.process_alive(os.getppid(), None) is True
    assert hb_module.process_alive(os.getppid(), 1.0) is False  # other start time
    assert hb_module.process_alive(None, None) is False
    assert hb_module.process_alive(2**22 + 12345, None) is False


async def test_liveness_is_45_seconds():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    await _heartbeat(store).beat_once()
    rows = await store.workers(10)
    assert worker_item(rows[0], "a" * 32)["alive"] is True
    assert worker_item(rows[0], "a" * 32)["self"] is True
    clock.now += 44
    assert worker_item((await store.workers(10))[0], None)["alive"] is True
    clock.now += 2
    item = worker_item((await store.workers(10))[0], None)
    assert item["alive"] is False and item["heartbeat_age_s"] == pytest.approx(
        46, abs=1
    )


async def test_rows_are_deleted_a_day_after_the_last_beat():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    await _heartbeat(store).beat_once()
    clock.now += 23 * 3600
    await store.cleanup_workers(24, 7)
    assert store.worker_rows
    clock.now += 2 * 3600
    await store.cleanup_workers(24, 7)
    assert not store.worker_rows
    assert store.sample_rows  # samples are kept 7 days
    clock.now += 7 * 86400
    await store.cleanup_workers(24, 7)
    assert not store.sample_rows


async def test_cleanup_runs_from_the_heartbeat_hourly(monkeypatch):
    store = MemoryOpsStore()
    calls = []

    async def cleanup(row_hours, sample_days):
        calls.append((row_hours, sample_days))

    monkeypatch.setattr(store, "cleanup_workers", cleanup)
    beat = _heartbeat(store)
    await beat.beat_once()
    await beat.beat_once()
    assert calls == [(24, 7)]


async def test_a_write_failure_never_raises_and_keeps_the_local_snapshot():
    store = MemoryOpsStore()
    store.fail_reads = ConnectionError("database down")
    beat = _heartbeat(store)
    assert await beat.beat_once() is False
    assert beat.write_failures == 1 and beat.snapshot
    store.fail_reads = None
    assert await beat.beat_once() is True
    assert store.worker_rows


async def test_without_a_store_the_heartbeat_samples_locally():
    beat = _heartbeat(None)
    assert await beat.beat_once() is True
    row = beat.local_row()
    item = worker_item(row, "a" * 32)
    assert item["alive"] is True and item["self"] is True
    assert item["snapshot"]["pool"] is None


async def test_stop_records_the_exit():
    store = MemoryOpsStore()
    beat = _heartbeat(store)
    beat.start()
    for _ in range(50):
        if beat.last_write_ok_at:
            break
        await asyncio.sleep(0.01)
    await beat.stop()
    item = worker_item((await store.workers(10))[0], None)
    assert item["exited_at"] is not None and item["alive"] is False


async def test_loop_lag_is_measured():
    monitor = LoopLagMonitor(tick_s=0.05)
    monitor.start()
    await asyncio.sleep(0.12)
    time.sleep(0.25)  # block the event loop
    await asyncio.sleep(0.1)
    last, worst = monitor.take()
    await monitor.stop()
    assert worst is not None and worst >= 150
    assert monitor.take()[1] is None or monitor.take()[1] < worst


async def test_recorder_drops_are_counted_over_five_minutes(monkeypatch):
    recorder = FakeRecorder()
    beat = _heartbeat(None, app_state=SimpleNamespace(insights_recorder=recorder))
    now = [1000.0]
    monkeypatch.setattr(hb_module.time, "time", lambda: now[0])
    assert beat._recorder(now[0])["lost_5m"] == 0
    recorder.lost, recorder.dropped = 7, 2
    now[0] += 60
    view = beat._recorder(now[0])
    assert view["lost_5m"] == 7 and view["dropped_5m"] == 2
    now[0] += 400
    assert beat._recorder(now[0])["lost_5m"] == 0


def test_process_sampler_works_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    sampler = ProcessSampler()
    first = sampler.sample()
    second = sampler.sample()
    assert sampler.create_time is None
    assert isinstance(first["threads"], int)
    assert second["cpu_pct"] is None or second["cpu_pct"] >= 0


def test_workers_are_ordered_live_first():
    def row(worker_id, age, superseded=False):
        return {
            "worker_id": worker_id,
            "heartbeat_age_s": age,
            "superseded": superseded,
            "exited_at": None,
            "snapshot": {},
        }

    items = [
        worker_item(row("old", 4000, superseded=True), None),
        worker_item(row("dead", 600), None),
        worker_item(row("live", 3), None),
    ]
    assert [i["worker_id"] for i in order_workers(items)] == ["live", "dead", "old"]
