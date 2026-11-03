"""Read-only snapshots of shared pipeline state for the Operations console."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from lightrag.api.ops.snapshot import LIVE_KEYS, NamespaceReader
from lightrag.kg import shared_storage
from lightrag.kg.shared_storage import snapshot_namespace

pytestmark = pytest.mark.offline


@pytest.fixture
def shared(monkeypatch):
    data = {
        "ws1:pipeline_status": {
            "busy": True,
            "job_name": "Indexing report.pdf",
            "provider_rejection": {
                "role": "extract",
                "kind": "budget",
                "retry_at": 5.0,
            },
            "history_messages": [f"line {i}" for i in range(10)],
            "secret_internal": "never read",
        },
        "pipeline_status": {"busy": False, "history_messages": []},
    }
    monkeypatch.setattr(shared_storage, "_shared_dicts", data)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    return data


def test_reads_whitelisted_keys_as_detached_copies(shared):
    snap = snapshot_namespace(
        "pipeline_status",
        "ws1",
        ["busy", "job_name", "provider_rejection", "cur_batch"],
    )
    assert snap == {
        "busy": True,
        "job_name": "Indexing report.pdf",
        "provider_rejection": {"role": "extract", "kind": "budget", "retry_at": 5.0},
        "cur_batch": None,
    }
    snap["provider_rejection"]["kind"] = "changed"
    assert shared["ws1:pipeline_status"]["provider_rejection"]["kind"] == "budget"


def test_never_creates_a_missing_namespace(shared):
    before = set(shared)
    assert snapshot_namespace("pipeline_status", "never_loaded", ["busy"]) is None
    assert set(shared) == before


def test_returns_none_before_shared_data_exists(monkeypatch):
    monkeypatch.setattr(shared_storage, "_shared_dicts", None)
    assert snapshot_namespace("pipeline_status", "ws1", ["busy"]) is None


def test_empty_workspace_is_the_global_namespace(shared):
    assert snapshot_namespace("pipeline_status", "", ["busy"]) == {"busy": False}


def test_rejects_keys_and_namespaces_outside_the_whitelist(shared):
    with pytest.raises(ValueError):
        snapshot_namespace("pipeline_status", "ws1", ["secret_internal"])
    with pytest.raises(ValueError):
        snapshot_namespace("full_docs", "ws1", ["busy"])


def test_does_not_take_the_internal_lock(shared, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("internal lock taken")

    monkeypatch.setattr(shared_storage, "get_internal_lock", boom)
    assert snapshot_namespace("pipeline_status", "ws1", ["busy"]) == {"busy": True}


def test_list_keys_come_back_as_bounded_slices(shared):
    tail = snapshot_namespace(
        "pipeline_status",
        "ws1",
        ["history_messages"],
        list_slices={"history_messages": (None, 3)},
    )["history_messages"]
    assert tail == {"total": 10, "start": 7, "items": ["line 7", "line 8", "line 9"]}
    window = snapshot_namespace(
        "pipeline_status",
        "ws1",
        ["history_messages"],
        list_slices={"history_messages": (2, 2)},
    )["history_messages"]
    assert window == {"total": 10, "start": 2, "items": ["line 2", "line 3"]}
    # Without a slice only the length is read.
    assert snapshot_namespace("pipeline_status", "ws1", ["history_messages"])[
        "history_messages"
    ] == {"total": 10, "start": 10, "items": []}
    past = snapshot_namespace(
        "pipeline_status",
        "ws1",
        ["history_messages"],
        list_slices={"history_messages": (50, 5)},
    )["history_messages"]
    assert past == {"total": 10, "start": 10, "items": []}


def test_slice_is_capped(shared):
    shared["ws1:pipeline_status"]["history_messages"] = [str(i) for i in range(2000)]
    snap = snapshot_namespace(
        "pipeline_status",
        "ws1",
        ["history_messages"],
        list_slices={"history_messages": (0, 10_000)},
    )
    assert len(snap["history_messages"]["items"]) == shared_storage.SNAPSHOT_MAX_SLICE


async def test_reader_reads_directly_in_single_process_mode(shared, monkeypatch):
    def no_threads(*args, **kwargs):
        raise AssertionError("executor used in single-process mode")

    reader = NamespaceReader()
    monkeypatch.setattr(reader, "_pool", no_threads)
    batch = await reader.snapshot_many(["ws1", "missing"], ("busy",))
    assert batch.responsive
    assert batch.data == {"ws1": {"busy": True}, "missing": None}
    assert reader.state == "local"


async def test_reader_rejects_unknown_keys(shared):
    with pytest.raises(ValueError):
        await NamespaceReader().snapshot_many(["ws1"], ("secret_internal",))


class _SlowManager:
    """A Manager dict whose reads block until released."""

    def __init__(self, data):
        self.data = data
        self.release = threading.Event()
        self.calls = 0

    def get(self, key):
        self.calls += 1
        self.release.wait(5)
        return self.data.get(key)


async def test_reader_times_out_and_keeps_one_call_in_flight(shared, monkeypatch):
    slow = _SlowManager(dict(shared))
    monkeypatch.setattr(shared_storage, "_shared_dicts", slow)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", True)
    reader = NamespaceReader(timeout_s=0.2)
    try:
        started = time.perf_counter()
        batch = await reader.snapshot_many(["ws1"], LIVE_KEYS)
        assert time.perf_counter() - started < 1.0
        assert batch.responsive is False and batch.data == {}
        assert reader.state == "unresponsive"
        # While the first read is stuck, a second one does not queue behind it.
        started = time.perf_counter()
        again = await reader.snapshot_many(["ws1"], ("busy",))
        assert again.responsive is False
        assert time.perf_counter() - started < 0.1
        assert slow.calls == 1
        slow.release.set()
        for _ in range(50):
            if not reader._inflight.is_set():
                break
            await asyncio.sleep(0.02)
        recovered = await reader.snapshot_many(["ws1"], ("busy",))
        assert recovered.responsive and recovered.data == {"ws1": {"busy": True}}
        assert reader.state == "ok"
        assert reader.last_rtt_ms is not None
    finally:
        slow.release.set()
        reader.close()


async def test_reader_reports_a_dead_manager(shared, monkeypatch):
    class Dead:
        def get(self, key):
            raise ConnectionRefusedError("manager gone")

    monkeypatch.setattr(shared_storage, "_shared_dicts", Dead())
    monkeypatch.setattr(shared_storage, "_is_multiprocess", True)
    reader = NamespaceReader(timeout_s=1.0)
    try:
        batch = await reader.snapshot_many(["ws1", "ws2"], ("busy",))
        assert batch.responsive is False
        assert reader.state == "unresponsive"
    finally:
        reader.close()


async def test_a_read_in_flight_within_the_deadline_is_not_unresponsive(
    shared, monkeypatch
):
    """Only a timed-out (or overdue) call marks the Manager unresponsive."""
    slow = _SlowManager(dict(shared))
    monkeypatch.setattr(shared_storage, "_shared_dicts", slow)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", True)
    reader = NamespaceReader(timeout_s=1.0)
    try:
        task = asyncio.create_task(reader.snapshot_many(["ws1"], ("busy",)))
        for _ in range(50):
            if reader._inflight.is_set():
                break
            await asyncio.sleep(0.01)
        assert reader._inflight.is_set()
        # A concurrent /overview looks while the healthy read is in flight.
        assert reader.state == "ok"
        # Past the deadline the in-flight call counts, even before its caller
        # has recorded the timeout.
        reader._inflight_since = time.monotonic() - 5.0
        assert reader.state == "unresponsive"
        reader._inflight_since = time.monotonic()
        slow.release.set()
        batch = await task
        assert batch.responsive and reader.state == "ok"
    finally:
        slow.release.set()
        reader.close()


def test_manager_pid_is_read_only_and_safe(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    assert shared_storage.manager_pid() is None
    monkeypatch.setattr(shared_storage, "_is_multiprocess", True)
    monkeypatch.setattr(
        shared_storage, "_manager", SimpleNamespace(_process=SimpleNamespace(pid=4321))
    )
    assert shared_storage.manager_pid() == 4321
    monkeypatch.setattr(shared_storage, "_manager", SimpleNamespace())
    assert shared_storage.manager_pid() is None
