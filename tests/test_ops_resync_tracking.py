"""Email re-sync drains as the Operations console sees them: KV
``resync:<ws>`` on every attempt (slot busy, errors, success), the backoff
deadline in heartbeats, and a kick from any process cutting the backoff."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from lightrag.api.email_resync_queue import DeferredResyncQueue
import lightrag.api.email_resync_queue as resync_module
from lightrag.api.ops import runs
from lightrag.api.ops.store import MemoryOpsStore

pytestmark = pytest.mark.offline

WORKER = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def store():
    return MemoryOpsStore()


@pytest.fixture
def tracker(store):
    tracker = runs.RunTracker(store, WORKER, interval_s=3600)
    runs.install(tracker)
    yield tracker
    runs.install(None)


def _seed(tmp_path, workspace="space1") -> None:
    directory = tmp_path / ".email_resync_queue" / workspace
    directory.mkdir(parents=True, mode=0o700)
    (directory / "a.json").write_text(json.dumps({"version": 2}))


def _queue(tmp_path, **kwargs) -> DeferredResyncQueue:
    async def get_rag(_workspace):
        return object()

    return DeferredResyncQueue(str(tmp_path), get_rag, **kwargs)


async def test_slot_busy_errors_and_success_are_recorded(tmp_path, tracker, store):
    _seed(tmp_path)
    queue = _queue(tmp_path, min_backoff=0.01, max_backoff=0.02)
    outcomes = ["busy", "error", "ok"]

    async def apply(rag, workspace, paths):
        outcome = outcomes.pop(0)
        if outcome == "busy":
            queue._busy_reasons[workspace] = "Pipeline is busy with another operation."
            return False
        if outcome == "error":
            raise TimeoutError("docling timed out for someone@unimas.my")
        for path in paths:
            path.unlink()
        return True

    queue._apply_batch = apply
    seen: list[dict] = []
    real = runs.resync_result

    def spy(workspace, patch):
        seen.append(dict(patch))
        return real(workspace, patch)

    resync_module.ops_runs.resync_result = spy
    try:
        queue.ensure_draining("space1")
        await asyncio.wait_for(queue._drainers["space1"], timeout=5)
    finally:
        resync_module.ops_runs.resync_result = real
    assert [p["last_result_class"] for p in seen] == ["slot_busy", "error", "ok"]
    assert [p["attempts"] for p in seen] == [1, 2, 0]
    assert seen[0]["reason"] == "Pipeline is busy with another operation."
    assert seen[0]["backoff_until"] > time.time() - 5
    assert seen[1]["error_kind"] == "timeout"
    assert "unimas.my" not in seen[1]["reason"]
    assert seen[2]["last_ok_at"] and seen[2]["backoff_until"] is None
    # Drains of the last hour, counted per outcome (FR-15).
    assert [p["drains_1h"] for p in seen][-1] == {
        "ok": 1,
        "slot_busy": 1,
        "error": 1,
        "window_s": 3600,
    }
    await tracker.flush_once()
    state = store.kv["resync:space1"]
    assert state["attempts"] == 0 and state["last_result_class"] == "ok"
    assert state["drains_1h"]["slot_busy"] == 1


async def test_a_kick_from_another_process_cuts_the_backoff(
    tmp_path, tracker, store, monkeypatch
):
    monkeypatch.setattr(resync_module, "_KICK_SLICE_SECONDS", 0.02)
    _seed(tmp_path)
    queue = _queue(tmp_path, min_backoff=60.0, max_backoff=60.0)
    calls = []

    async def apply(rag, workspace, paths):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return False  # slot busy: back off for a minute
        for path in paths:
            path.unlink()
        return True

    queue._apply_batch = apply
    queue.ensure_draining("space1")
    await asyncio.sleep(0.1)
    [state] = queue.drainer_states()
    assert state["state"] == "backoff" and state["backoff_until"] > time.time() + 50
    # Another process writes the kick straight into the shared store.
    store.kv["kick:space1"] = {"at": time.time(), "actor": "admin"}
    await asyncio.wait_for(queue._drainers["space1"], timeout=5)
    assert len(calls) == 2 and calls[1] - calls[0] < 5
    assert queue.drainer_states() == []


async def test_an_old_kick_does_not_cut_a_new_backoff(
    tmp_path, tracker, store, monkeypatch
):
    monkeypatch.setattr(resync_module, "_KICK_SLICE_SECONDS", 0.02)
    store.kv["kick:space1"] = {"at": time.time() - 10}
    _seed(tmp_path)
    queue = _queue(tmp_path, min_backoff=0.3, max_backoff=0.3)
    calls = []

    async def apply(rag, workspace, paths):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return False
        for path in paths:
            path.unlink()
        return True

    queue._apply_batch = apply
    queue.ensure_draining("space1")
    await asyncio.wait_for(queue._drainers["space1"], timeout=5)
    assert calls[1] - calls[0] >= 0.25


async def test_without_tracking_the_drainer_behaves_as_before(tmp_path):
    runs.install(None)
    _seed(tmp_path)
    queue = _queue(tmp_path, min_backoff=0.01, max_backoff=0.02)
    results = [False, True]

    async def apply(rag, workspace, paths):
        if results.pop(0):
            for path in paths:
                path.unlink()
            return True
        return False

    queue._apply_batch = apply
    queue.ensure_draining("space1")
    await asyncio.wait_for(queue._drainers["space1"], timeout=5)
    assert results == []


async def test_heartbeat_reports_drainer_backoff(tmp_path):
    from types import SimpleNamespace

    from lightrag.api.ops.heartbeat import Heartbeat
    from lightrag.api.ops.snapshot import NamespaceReader

    queue = _queue(tmp_path)
    task = asyncio.create_task(asyncio.sleep(10))
    queue._drainers["space1"] = task
    queue._states["space1"] = {"state": "backoff", "backoff_until": time.time() + 30}
    try:
        heartbeat = Heartbeat(
            worker_id=WORKER,
            mode="uvicorn",
            version="",
            store=None,
            reader=NamespaceReader(),
            app_state=SimpleNamespace(email_resync_queue=queue),
            leases=lambda: [],
            data_rag=lambda: None,
        )
        [drainer] = heartbeat._drainers()
        assert drainer["state"] == "backoff"
        assert drainer["backoff_until"].endswith("+00:00")
    finally:
        task.cancel()
