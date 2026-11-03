"""Email re-sync queue scan: counts only, never opens files, first_pending_at."""

from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest

from lightrag.api.email_resync_queue import DeferredResyncQueue, workspace_for_dir
from lightrag.api.ops import queue_scan
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.store import MemoryOpsStore

pytestmark = pytest.mark.offline


def _touch(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def queue_dir(tmp_path) -> Path:
    base = DeferredResyncQueue(str(tmp_path), get_rag=None).base_dir
    _touch(base / "space1" / ("a" * 40 + ".json"), '{"components": "secret mail body"}')
    _touch(base / "space1" / ("b" * 40 + ".json"))
    _touch(base / "space1" / ("c" * 40 + ".bad"))
    _touch(base / "space1" / ".lock", "")
    _touch(base / "space1" / ".abc.123.tmp")
    _touch(base / "fsktm" / ("d" * 40 + ".bad"))
    _touch(base / "_default" / ("e" * 40 + ".json"))
    (base / "empty_ws").mkdir(parents=True)
    _touch(base / "bad name!" / ("f" * 40 + ".json"))
    _touch(base / "stray-file.json")
    return base


def test_workspace_for_dir():
    assert workspace_for_dir("_default") == ""
    assert workspace_for_dir("space1") == "space1"
    assert workspace_for_dir("..") is None
    assert workspace_for_dir("bad name!") is None


def test_counts_json_and_bad_per_workspace(queue_dir):
    counts = queue_scan.scan_queue_dir(queue_dir)
    assert counts == {
        "space1": queue_scan.QueueCounts(pending=2, bad=1),
        "fsktm": queue_scan.QueueCounts(pending=0, bad=1),
        "": queue_scan.QueueCounts(pending=1, bad=0),
    }


def test_missing_queue_directory_is_empty(tmp_path):
    assert queue_scan.scan_queue_dir(tmp_path / "nope") == {}


def test_scan_never_opens_or_locks_a_file(queue_dir, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the scan opened a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    import fcntl

    monkeypatch.setattr(fcntl, "flock", refuse)
    assert queue_scan.scan_queue_dir(queue_dir)["space1"].pending == 2


class _Loop:
    holder = "worker-a"

    def __init__(self, held: bool = True):
        self.held = held

    async def still_held(self) -> bool:
        return self.held


async def test_first_pending_at_set_kept_and_cleared(queue_dir):
    store = MemoryOpsStore()
    clock = iter([1000.0, 1060.0, 1120.0])
    summary = await queue_scan.scan_cycle(
        _Loop(), store=store, base_dir=queue_dir, clock=lambda: next(clock)
    )
    assert summary["total"] == 3 and summary["workspaces"] == 2 and summary["bad"] == 2
    items = {i["workspace"]: i for i in summary["items"]}
    assert items["space1"] == {
        "workspace": "space1",
        "pending": 2,
        "bad": 1,
        "first_pending_at": 1000.0,
        "last_progress_at": 1000.0,
    }
    assert items["fsktm"]["first_pending_at"] is None
    assert store.kv["resync:space1"] == {
        "first_pending_at": 1000.0,
        "last_progress_at": 1000.0,
    }
    assert store.kv["email_queue_summary"]["holder"] == "worker-a"
    # Nothing a file contains reaches the summary.
    assert "secret" not in str(store.kv)

    # Still pending: the original time stays.
    second = await queue_scan.scan_cycle(
        _Loop(), store=store, base_dir=queue_dir, clock=lambda: next(clock)
    )
    assert {i["workspace"]: i for i in second["items"]}["space1"][
        "first_pending_at"
    ] == 1000.0
    assert second["longest_pending_s"] == 60.0

    # Drained: cleared in the summary and in resync:<ws>, other fields kept.
    store.kv["resync:space1"]["attempts"] = 3
    for path in (queue_dir / "space1").glob("*.json"):
        path.unlink()
    third = await queue_scan.scan_cycle(
        _Loop(), store=store, base_dir=queue_dir, clock=lambda: next(clock)
    )
    assert {i["workspace"]: i for i in third["items"]}["space1"][
        "first_pending_at"
    ] is None
    assert store.kv["resync:space1"] == {
        "first_pending_at": None,
        "last_progress_at": None,
        "attempts": 3,
    }


async def test_first_pending_at_survives_a_lost_summary(queue_dir):
    store = MemoryOpsStore()
    store.kv["resync:space1"] = {"first_pending_at": 500.0}
    summary = await queue_scan.scan_cycle(
        _Loop(), store=store, base_dir=queue_dir, clock=lambda: 1000.0
    )
    assert {i["workspace"]: i for i in summary["items"]}["space1"][
        "first_pending_at"
    ] == 500.0


async def test_nothing_is_written_after_losing_the_lease(queue_dir):
    store = MemoryOpsStore()
    assert (
        await queue_scan.scan_cycle(_Loop(held=False), store=store, base_dir=queue_dir)
        is None
    )
    assert store.kv == {}


async def test_runs_as_a_leased_singleton(queue_dir):
    store = MemoryOpsStore()

    async def cycle(loop):
        await queue_scan.scan_cycle(loop, store=store, base_dir=queue_dir)

    make = lambda holder: LeasedLoop(  # noqa: E731
        store,
        queue_scan.SCAN_LEASE,
        holder,
        cycle,
        ttl_s=queue_scan.SCAN_LEASE_TTL_S,
        interval_s=queue_scan.SCAN_INTERVAL_S,
        cycle_timeout_s=queue_scan.SCAN_CYCLE_TIMEOUT_S,
    )
    assert await make("worker-a").run_once()
    assert not await make("worker-b").run_once()
    assert store.kv["email_queue_summary"]["holder"] == "worker-a"
    assert queue_scan.SCAN_LEASE_TTL_S > queue_scan.SCAN_CYCLE_TIMEOUT_S


async def test_a_shrinking_queue_is_progress_not_a_stall(queue_dir):
    """Re-syncs keep arriving while drains succeed (§1.7: no drain > 30 min)."""
    store = MemoryOpsStore()
    clock = iter([1000.0, 1060.0, 1120.0, 1180.0])
    space1 = queue_dir / "space1"
    for path in (queue_dir / "_default").glob("*.json"):
        path.unlink()

    async def scan():
        return await queue_scan.scan_cycle(
            _Loop(), store=store, base_dir=queue_dir, clock=lambda: next(clock)
        )

    await scan()  # 2 pending, first seen at 1000
    _touch(space1 / ("g" * 40 + ".json"))
    grown = await scan()  # 3 pending: no drain seen yet
    item = {i["workspace"]: i for i in grown["items"]}["space1"]
    assert item["last_progress_at"] == 1000.0
    assert grown["longest_stalled_s"] == 60.0
    (space1 / ("a" * 40 + ".json")).unlink()
    drained = await scan()  # 2 pending: one went through
    item = {i["workspace"]: i for i in drained["items"]}["space1"]
    assert item["first_pending_at"] == 1000.0
    assert item["last_progress_at"] == 1120.0
    assert store.kv["resync:space1"]["last_progress_at"] == 1120.0
    steady = await scan()
    assert steady["longest_pending_s"] == 180.0
    assert steady["longest_stalled_s"] == 60.0
