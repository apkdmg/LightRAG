"""Reconcile: a run is orphaned only after its owner's death is confirmed by
the operating system; wedged owners, other hosts and this worker never are.
Shared-state checks find busy-without-run and reservations without owner,
and persisted pauses that shared state no longer holds are closed."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import psutil
import pytest

from lightrag.api.ops import reconcile, runs
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.service import (
    ATTENTION_KEY,
    BUSY_WITHOUT_RUN_KEY,
    RESERVATIONS_KEY,
    OpsService,
)
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import MemoryOpsStore
from lightrag.kg import shared_storage

pytestmark = pytest.mark.offline

ME = "0123456789abcdef0123456789abcdef"
OWNER = "fedcba9876543210fedcba9876543210"
HOST = socket.gethostname()[:255]


def _spawn() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


async def _seed_owner(
    store: MemoryOpsStore, pid: int, *, beat_age_s: float, host: str = HOST
) -> str:
    """A worker row for ``pid`` whose last beat is ``beat_age_s`` old, owning
    one running run."""
    create_time = psutil.Process(pid).create_time() if psutil.pid_exists(pid) else None
    await store.worker_beat(
        {
            "worker_id": OWNER,
            "host": host,
            "pid": pid,
            "pid_create_time": create_time,
            "mode": "gunicorn",
            "started_at": datetime.now(timezone.utc) - timedelta(minutes=10),
            "snapshot": {},
        }
    )
    store.worker_rows[OWNER]["heartbeat_at"] = datetime.now(timezone.utc) - timedelta(
        seconds=beat_age_s
    )
    now = time.time()
    run_id = runs.uuid4().hex
    await store.runs_write(
        [
            {
                "run_id": run_id,
                "workspace": "fsktm",
                "kind": "process",
                "trigger": "upload",
                "status": "running",
                "worker_id": OWNER,
                "pid": pid,
                "started_at": now - 300,
                "heartbeat_at": now - beat_age_s,
                "progress_at": now - beat_age_s,
                "finished_at": None,
            }
        ]
    )
    return run_id


async def _reconcile(store: MemoryOpsStore, markers=None, host: str = HOST) -> dict:
    return await reconcile.reconcile_runs(
        store,
        my_worker_id=ME,
        host=host,
        stale_s=60.0,
        markers=markers or {},
    )


async def test_kill_9_orphans_only_after_the_os_confirms_death():
    store = MemoryOpsStore()
    proc = _spawn()
    try:
        run_id = await _seed_owner(store, proc.pid, beat_age_s=120)
        # Heartbeat stale but the process runs: a wedged worker, not dead.
        result = await _reconcile(store)
        assert result["orphaned"] == []
        assert result["unresponsive"] == [run_id]
        assert store.run_rows[run_id]["status"] == "running"

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        result = await _reconcile(store)
        assert result["orphaned"] == [run_id]
        assert store.run_rows[run_id]["status"] == "orphaned"
    finally:
        _kill(proc)


async def test_a_stopped_but_alive_owner_is_never_orphaned():
    store = MemoryOpsStore()
    proc = _spawn()
    try:
        os.kill(proc.pid, signal.SIGSTOP)  # wedged: alive, does nothing
        run_id = await _seed_owner(store, proc.pid, beat_age_s=600)
        for _ in range(3):
            result = await _reconcile(store)
            assert result["orphaned"] == []
        assert store.run_rows[run_id]["status"] == "running"
    finally:
        try:
            os.kill(proc.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        _kill(proc)


async def test_a_fresh_heartbeat_is_alive_even_if_the_pid_is_gone():
    store = MemoryOpsStore()
    proc = _spawn()
    run_id = await _seed_owner(store, proc.pid, beat_age_s=5)
    _kill(proc)
    result = await _reconcile(store)
    assert result["orphaned"] == [] and result["unresponsive"] == []
    assert store.run_rows[run_id]["status"] == "running"


async def test_a_reused_pid_is_dead():
    store = MemoryOpsStore()
    # Our own pid recorded for another worker: a previous boot (PID 1 in a
    # container), so the start time differs and the owner is gone.
    run_id = await _seed_owner(store, os.getpid(), beat_age_s=120)
    store.worker_rows[OWNER]["pid_create_time"] = 1.0
    result = await _reconcile(store)
    assert result["orphaned"] == [run_id]


async def test_child_exit_marker_confirms_death_and_other_hosts_are_unknown(tmp_path):
    store = MemoryOpsStore()
    proc = _spawn()
    try:
        run_id = await _seed_owner(store, proc.pid, beat_age_s=120, host="other-host")
        result = await _reconcile(store)
        assert result["orphaned"] == []  # cannot be checked from here
        # A marker from the Gunicorn master for that pid, after its start.
        store.worker_rows[OWNER]["host"] = HOST
        result = await _reconcile(store, markers={proc.pid: time.time()})
        assert result["orphaned"] == [run_id]
    finally:
        _kill(proc)


async def test_this_worker_is_never_orphaned():
    store = MemoryOpsStore()
    run_id = await _seed_owner(store, os.getpid(), beat_age_s=600)
    result = await reconcile.reconcile_runs(
        store, my_worker_id=OWNER, host=HOST, stale_s=60.0, markers={}
    )
    assert result["orphaned"] == []
    assert store.run_rows[run_id]["status"] == "running"


def test_owner_state_without_a_worker_row():
    alive = lambda pid, ct: pid == 10  # noqa: E731
    common = dict(
        my_worker_id=ME,
        host=HOST,
        markers={},
        stale_s=60.0,
        alive=alive,
        since=None,
        owner_host=HOST,
    )
    assert reconcile.owner_state(worker=None, pid=10, worker_id=OWNER, **common) == (
        "unresponsive"
    )
    assert (
        reconcile.owner_state(worker=None, pid=11, worker_id=OWNER, **common) == "dead"
    )
    assert reconcile.owner_state(worker=None, pid=None, worker_id=OWNER, **common) == (
        "unknown"
    )
    # A PID recorded on another host (or on an unknown one) is never checked
    # here: a missing local PID proves nothing.
    for owner_host in ("other-host", None):
        assert (
            reconcile.owner_state(
                worker=None,
                pid=11,
                worker_id=OWNER,
                **{**common, "owner_host": owner_host},
            )
            == "unknown"
        )


def test_an_owner_host_is_gone_only_when_none_of_its_workers_beats():
    common = dict(
        worker=None,
        pid=11,
        worker_id=OWNER,
        my_worker_id=ME,
        host=HOST,
        markers={},
        stale_s=60.0,
        alive=lambda pid, ct: False,
        since=None,
        owner_host="old-container",
    )
    gone = reconcile.HOST_GONE_S + 1
    # Without the host table nothing on another host is ever dead.
    assert reconcile.owner_check(**common, owner_age_s=gone) == ("unknown", None)
    # A sibling on that host still beats: the host is there.
    assert reconcile.owner_check(
        **common, owner_age_s=gone, host_ages={"old-container": 5.0}
    ) == ("unknown", None)
    # Stale, but not for long enough.
    assert reconcile.owner_check(
        **common, owner_age_s=120.0, host_ages={"old-container": 120.0}
    ) == ("unknown", None)
    assert reconcile.owner_check(
        **common, owner_age_s=gone, host_ages={"old-container": gone}
    ) == ("dead", "host_gone")
    assert reconcile.owner_check(**common, owner_age_s=gone, host_ages={}) == (
        "dead",
        "host_gone",
    )


async def test_runs_of_a_recreated_container_are_orphaned_with_the_reason():
    """A deploy that SIGKILLed the container: the new one has another host
    name, the old worker never recorded its exit."""
    store = MemoryOpsStore()
    run_id = await _seed_owner(store, 4242, beat_age_s=60, host="old-container")
    stale = datetime.now(timezone.utc) - timedelta(seconds=reconcile.HOST_GONE_S + 60)
    store.worker_rows[OWNER]["heartbeat_at"] = stale
    store.run_rows[run_id]["heartbeat_at"] = stale
    result = await reconcile.reconcile_runs(
        store,
        my_worker_id=ME,
        host=HOST,
        stale_s=60.0,
        markers={},
        workers=await store.workers(500),
    )
    assert result["orphaned"] == [run_id]
    assert store.run_rows[run_id]["orphaned_by"] == "host_gone"


async def test_orphaned_runs_record_how_death_was_confirmed():
    store = MemoryOpsStore()
    proc = _spawn()
    try:
        run_id = await _seed_owner(store, proc.pid, beat_age_s=120)
        _kill(proc)
        assert (await _reconcile(store))["orphaned"] == [run_id]
        assert store.run_rows[run_id]["orphaned_by"] == "pid_gone"
    finally:
        _kill(proc)


# -- shared-state checks -----------------------------------------------------------


class _Loop:
    holder = ME

    async def still_held(self) -> bool:
        return True


@pytest.fixture
def shared(monkeypatch):
    data: dict = {}
    monkeypatch.setattr(shared_storage, "_shared_dicts", data)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    return data


def _service(store: MemoryOpsStore) -> OpsService:
    from types import SimpleNamespace

    service = OpsService(
        store=store,
        reader=NamespaceReader(),
        app_state=SimpleNamespace(),
        run_tracking=True,
    )
    service.worker_id = ME
    return service


async def test_busy_without_run_reservations_and_stale_pauses(shared, tmp_path):
    store = MemoryOpsStore()
    service = _service(store)
    now = time.time()
    old = datetime.fromtimestamp(now - 900, timezone.utc).isoformat()
    # space1: busy for 15 min, no run anywhere (started before tracking).
    store.add_doc("space1", "doc-1", "processing")
    shared["space1:pipeline_status"] = {"busy": True, "job_start": old}
    # fsktm: busy, its run is recorded and running: fine.
    run_id = await _seed_owner(store, os.getpid(), beat_age_s=5)
    shared["fsktm:pipeline_status"] = {
        "busy": True,
        "job_start": old,
        "run_id": run_id,
    }
    # hr: a scan whose owner worker is gone (no row, dead after a marker).
    store.add_doc("hr", "doc-2", "processing")
    shared["hr:pipeline_status"] = {
        "busy": False,
        "scanning": True,
        "scan_owner_worker_id": OWNER,
        "scan_started_at": old,
        "pending_enqueues": 2,
    }
    # A pause persisted for law, which shared state no longer holds.
    store.add_doc("law", "doc-3", "pending")
    store.kv["pause:law:extraction"] = {
        "workspace": "law",
        "slot": "extraction",
        "kind": "invalid_key",
        "cleared_at": None,
    }
    shared["law:pipeline_status"] = {"busy": False}

    loop = _Loop()
    out = await reconcile.reconcile_cycle(
        loop, service=service, process=lambda: {"mode": "gunicorn"}
    )
    # First seen now: reported once it stays that way for the grace period
    # (job_start is not trusted: a destructive slot never refreshes it).
    assert out["busy_without_run"] == 0
    service.clock = lambda: now + 180
    out = await reconcile.reconcile_cycle(
        loop, service=service, process=lambda: {"mode": "gunicorn"}
    )
    assert out["busy_without_run"] == 1
    busy = store.kv[BUSY_WITHOUT_RUN_KEY]
    assert [(i["workspace"], i["reason_code"]) for i in busy["items"]] == [
        ("space1", "no_run")
    ]
    assert busy["items"][0]["age_s"] == pytest.approx(180, abs=5)
    # The scan owner (worker OWNER) has a row: alive -> no reservation yet;
    # pending_enqueues is only first seen now.
    assert store.kv[RESERVATIONS_KEY]["items"] == []
    assert store.kv["pause:law:extraction"]["cleared_by"] == "reconcile"

    # Ten minutes later the counter is still held, and the scan owner's
    # process is confirmed gone.
    service.clock = lambda: now + 600
    store.worker_rows[OWNER]["heartbeat_at"] = datetime.now(timezone.utc) - timedelta(
        minutes=10
    )
    store.worker_rows[OWNER]["pid"] = 999999
    store.worker_rows[OWNER]["pid_create_time"] = 1.0
    await reconcile.reconcile_cycle(
        loop, service=service, process=lambda: {"mode": "gunicorn"}
    )
    items = store.kv[RESERVATIONS_KEY]["items"]
    kinds = sorted((i["workspace"], i["kind"], i["reason_code"]) for i in items)
    assert ("hr", "pending_enqueues", "held") in kinds
    # The fields the console renders: which flag, and how many uploads.
    held = next(i for i in items if i["kind"] == "pending_enqueues")
    assert held["scanning"] is False and held["pending_enqueues"] == 2
    scan = next(i for i in items if i["kind"] == "scanning")
    assert scan["scanning"] is True
    # The fsktm run was orphaned meanwhile (same owner, now dead).
    assert ("hr", "scanning", "owner_dead") in kinds
    # Reconcile never clears anything in shared state.
    assert shared["hr:pipeline_status"]["scanning"] is True
    assert shared["space1:pipeline_status"]["busy"] is True


async def test_reconcile_cycle_is_a_noop_without_tracking(shared):
    store = MemoryOpsStore()
    service = _service(store)
    service.run_tracking = False
    out = await reconcile.reconcile_cycle(
        _Loop(), service=service, process=lambda: {"mode": "uvicorn"}
    )
    assert out == {}
    assert BUSY_WITHOUT_RUN_KEY not in store.kv


async def test_attention_cycle_reconciles_then_publishes(shared):
    store = MemoryOpsStore()
    service = _service(store)
    proc = _spawn()
    try:
        run_id = await _seed_owner(store, proc.pid, beat_age_s=120)
        shared["fsktm:pipeline_status"] = {"busy": True, "run_id": run_id}
        _kill(proc)
        loop = LeasedLoop(
            store,
            reconcile.RECONCILE_LEASE,
            ME,
            lambda loop: reconcile.attention_cycle(
                loop, service=service, process=lambda: {"mode": "gunicorn"}
            ),
            ttl_s=120,
            interval_s=60,
            cycle_timeout_s=90,
        )
        assert await loop.run_once()
        assert store.run_rows[run_id]["status"] == "orphaned"
        rules = {a["rule"] for a in store.kv[ATTENTION_KEY]["items"]}
        assert "run_orphaned" in rules
    finally:
        _kill(proc)


async def test_retention_runs_hourly(monkeypatch):
    store = MemoryOpsStore()
    calls = []

    async def cleanup(run_days, failure_days):
        calls.append((run_days, failure_days))

    store.runs_cleanup = cleanup
    monkeypatch.setitem(reconcile._state, "last_cleanup", 0.0)
    await reconcile._maybe_cleanup(store, 10_000.0)
    await reconcile._maybe_cleanup(store, 10_100.0)
    assert calls == [(30, 90)]
    await asyncio.sleep(0)


async def test_an_owner_that_recorded_its_exit_is_dead_on_any_host():
    store = MemoryOpsStore()
    proc = _spawn()
    try:
        # A deploy recreated the container: new host name, old worker
        # shut down cleanly and wrote exited_at.
        run_id = await _seed_owner(store, proc.pid, beat_age_s=120, host="old-box")
        assert (await _reconcile(store))["orphaned"] == []
        await store.worker_exited(OWNER)
        assert (await _reconcile(store))["orphaned"] == [run_id]
    finally:
        _kill(proc)


async def _cycle(service, mode: str = "gunicorn") -> dict:
    return await reconcile.reconcile_cycle(
        _Loop(), service=service, process=lambda: {"mode": mode}
    )


async def test_a_live_run_missing_from_the_running_read_is_not_busy_without_run(
    shared,
):
    """A run that started after the running-runs read (or beyond its limit)
    is still running in the table: it owns the slot."""
    store = MemoryOpsStore()
    service = _service(store)
    now = time.time()
    run_id = await _seed_owner(store, os.getpid(), beat_age_s=5)
    shared["fsktm:pipeline_status"] = {
        "busy": True,
        "destructive_busy": True,
        "job_start": "2020-01-01T00:00:00+00:00",  # never refreshed
        "run_id": run_id,
    }
    checks = []
    for offset in (0, 300, 600):
        checks.append(
            await reconcile.check_shared_state(
                store,
                service,
                running=[],  # the read missed it
                workers={},
                my_worker_id=ME,
                host=HOST,
                markers={},
                thresholds={},
                now=now + offset,
            )
        )
    assert all(c["busy_without_run"]["items"] == [] for c in checks)


async def test_busy_check_is_skipped_when_runs_could_not_be_read(shared):
    store = MemoryOpsStore()
    service = _service(store)
    store.add_doc("space1", "doc-1", "processing")
    shared["space1:pipeline_status"] = {"busy": True, "run_id": "a" * 32}
    previous = {"at": "x", "at_epoch": 1.0, "items": [], "responsive": True}
    store.kv[BUSY_WITHOUT_RUN_KEY] = previous

    async def broken(limit):
        raise TimeoutError("running_runs timed out")

    store.running_runs = broken
    service.clock = lambda: time.time() + 3600
    out = await _cycle(service)
    assert "busy_without_run" in out and out["busy_without_run"] == 0
    # The previous result stays as it was (nothing reported as finished).
    assert store.kv[BUSY_WITHOUT_RUN_KEY]["items"] == []
    assert store.kv[BUSY_WITHOUT_RUN_KEY]["at_epoch"] == 1.0


async def test_a_purge_slot_with_a_live_owner_is_not_busy_without_run(shared):
    store = MemoryOpsStore()
    service = _service(store)
    store.add_doc("purged", "doc-1", "processing")
    # A workspace purge holds the destructive slot without a run.
    shared["purged:pipeline_status"] = {
        "busy": True,
        "destructive_busy": True,
        "run_id": None,
        "owner_worker_id": ME,  # this worker: alive
    }
    now = time.time()
    for offset in (0, 600):
        service.clock = lambda offset=offset: now + offset
        await _cycle(service)
    assert store.kv[BUSY_WITHOUT_RUN_KEY]["items"] == []


async def test_an_upload_reservation_held_by_a_live_run_is_not_stranded(shared):
    """/upload keeps pending_enqueues for its whole processing run."""
    store = MemoryOpsStore()
    service = _service(store)
    run_id = await _seed_owner(store, os.getpid(), beat_age_s=5)
    shared["fsktm:pipeline_status"] = {
        "busy": True,
        "run_id": run_id,
        "pending_enqueues": 1,
    }
    now = time.time()
    for offset in (0, 400, 900, 1800):
        service.clock = lambda offset=offset: now + offset
        store.run_rows[run_id]["heartbeat_at"] = datetime.now(timezone.utc)
        store.worker_rows[OWNER]["heartbeat_at"] = datetime.now(timezone.utc)
        await _cycle(service)
        assert store.kv[RESERVATIONS_KEY]["items"] == []
    # The run ends but the counter stays: the hold is counted from now.
    shared["fsktm:pipeline_status"] = {"busy": False, "pending_enqueues": 1}
    service.clock = lambda: now + 1900
    await _cycle(service)
    assert store.kv[RESERVATIONS_KEY]["items"] == []
    service.clock = lambda: now + 1900 + 301
    await _cycle(service)
    [item] = store.kv[RESERVATIONS_KEY]["items"]
    assert item["kind"] == "pending_enqueues" and item["reason_code"] == "held"


async def test_reservations_are_not_checked_in_single_process_mode(shared):
    store = MemoryOpsStore()
    service = _service(store)
    store.add_doc("hr", "doc-2", "processing")
    shared["hr:pipeline_status"] = {"busy": False, "pending_enqueues": 2}
    now = time.time()
    for offset in (0, 900):
        service.clock = lambda offset=offset: now + offset
        await _cycle(service, mode="uvicorn")
    reservations = store.kv[RESERVATIONS_KEY]
    assert reservations["items"] == []
    assert reservations["checked_mode"] == "single_process"
