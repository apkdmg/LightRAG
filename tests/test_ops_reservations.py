"""Per-worker shares of ``pending_enqueues`` and the guarded release of a dead
worker's reservations (PR-9, OQ-12).

The shares are written next to the counter, in the same locked block, only
with run tracking on; the counter stays what every reader uses and the two
stay consistent under concurrent enqueues. The operator-only release
follows the §3.9 sequence and touches only what a confirmed-dead worker
holds: its scan and its share, nothing of a live worker, never ``busy``.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.api.ops import actions as ops_actions
from lightrag.api.ops import runs
from lightrag.kg import shared_storage
from tests.ops_fixtures import seeded_store
from tests import test_ops_actions as actions_tests
from tests.test_ops_actions import REASON, _app, _audit, _post

# The action tests' shared-state fixtures (single process, Gunicorn stand-in)
# and the OPS_ACTIONS_ENABLED switch.
shared = actions_tests.shared
multiprocess = actions_tests.multiprocess
enabled = actions_tests.enabled

pytestmark = pytest.mark.offline

DEAD = "deaddeaddeaddeaddeaddeaddeaddead"
LIVE = "11112222333344445555666677778888"
SCAN_START = "2026-11-04T09:00:00+00:00"
OWNERS = runs.PENDING_OWNERS_KEY


@pytest.fixture
async def pipeline():
    """Real single-process shared storage with fsktm's pipeline_status, as
    the document routes read it (``get_namespace_data``)."""
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace="fsktm")
    state = await shared_storage.get_namespace_data(
        "pipeline_status", workspace="fsktm"
    )
    yield state
    shared_storage.finalize_share_data()


@pytest.fixture
def tracker():
    current = runs.RunTracker(None, LIVE)
    runs.install(current)
    yield current
    runs.install(None)


# -- the shares stay consistent with the counter ----------------------------------------------


async def test_reserve_and_release_maintain_this_workers_share(pipeline, tracker):
    from lightrag.api.routers.document_routes import (
        _release_enqueue_slot,
        _reserve_enqueue_slot,
    )

    rag = SimpleNamespace(workspace="fsktm")
    state = pipeline
    assert await _reserve_enqueue_slot(rag) is True
    assert await _reserve_enqueue_slot(rag) is True
    assert state["pending_enqueues"] == 2 and state[OWNERS] == {LIVE: 2}
    await _release_enqueue_slot(rag)
    assert state["pending_enqueues"] == 1 and state[OWNERS] == {LIVE: 1}
    await _release_enqueue_slot(rag)
    assert state["pending_enqueues"] == 0 and state[OWNERS] == {}
    # A stray release is harmless for both.
    await _release_enqueue_slot(rag)
    assert state["pending_enqueues"] == 0 and state[OWNERS] == {}


async def test_another_workers_share_is_left_alone(pipeline, tracker):
    from lightrag.api.routers.document_routes import (
        _release_enqueue_slot,
        _reserve_enqueue_slot,
    )

    rag = SimpleNamespace(workspace="fsktm")
    state = pipeline
    state.update({"pending_enqueues": 3, OWNERS: {DEAD: 3}})
    await _reserve_enqueue_slot(rag)
    assert state["pending_enqueues"] == 4 and state[OWNERS] == {DEAD: 3, LIVE: 1}
    await _release_enqueue_slot(rag)
    await _release_enqueue_slot(rag)
    # This worker had one share: the dead worker's three stay.
    assert state["pending_enqueues"] == 2 and state[OWNERS] == {DEAD: 3}


async def test_tracking_off_writes_no_shares(pipeline):
    from lightrag.api.routers.document_routes import (
        _release_enqueue_slot,
        _reserve_enqueue_slot,
    )

    runs.install(None)
    rag = SimpleNamespace(workspace="fsktm")
    state = pipeline
    await _reserve_enqueue_slot(rag)
    assert state["pending_enqueues"] == 1 and OWNERS not in state
    await _release_enqueue_slot(rag)
    assert state["pending_enqueues"] == 0 and OWNERS not in state


async def test_shares_track_the_counter_under_concurrent_enqueues(pipeline, tracker):
    """Many uploads reserve and release concurrently (each holds its slot for
    a random time, as a background enqueue does); at every quiescent point
    the share equals the counter, and both end at zero."""
    from lightrag.api.routers.document_routes import (
        _release_enqueue_slot,
        _reserve_enqueue_slot,
    )

    rag = SimpleNamespace(workspace="fsktm")
    state = pipeline
    rng = random.Random(7)
    seen: list[tuple[int, int]] = []

    async def upload():
        await asyncio.sleep(rng.random() * 0.01)
        assert await _reserve_enqueue_slot(rag)
        seen.append((state["pending_enqueues"], sum(state[OWNERS].values())))
        await asyncio.sleep(rng.random() * 0.02)
        await _release_enqueue_slot(rag)
        seen.append(
            (state["pending_enqueues"], sum((state.get(OWNERS) or {}).values()))
        )

    for _ in range(3):
        await asyncio.gather(*(upload() for _ in range(150)))
        assert state["pending_enqueues"] == 0 and state[OWNERS] == {}
    # The pair is read after the locked block: always equal.
    assert seen and all(counter == share for counter, share in seen)
    assert max(counter for counter, _ in seen) > 10


def test_pending_owner_helpers_replace_the_value():
    current = {LIVE: 1, "junk": "x", "neg": -1, "bool": True}
    runs.install(runs.RunTracker(None, LIVE))
    try:
        fields = runs.pending_reserve_fields(current)
        assert fields == {OWNERS: {LIVE: 2}}
        assert current[LIVE] == 1  # never mutated in place
        assert runs.pending_release_fields({LIVE: 1}) == {OWNERS: {}}
        assert runs.pending_release_fields({DEAD: 1}) == {}
    finally:
        runs.install(None)
    assert runs.pending_reserve_fields({}) == {}


# -- the guarded release ------------------------------------------------------------------------


def _worker_row(worker_id: str, *, age_s: float, pid: int) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "host": "host-a",
        "pid": pid,
        "pid_create_time": 1.0,
        "mode": "gunicorn",
        "started_at": datetime(2026, 11, 4, 8, 0, tzinfo=timezone.utc),
        "heartbeat_at": datetime.fromtimestamp(time.time() - age_s, timezone.utc),
        "exited_at": None,
        "superseded": False,
        "leases_held": [],
        "snapshot": {},
    }


def _store():
    store = seeded_store()
    store.worker_rows[DEAD] = _worker_row(DEAD, age_s=600, pid=811)
    store.worker_rows[LIVE] = _worker_row(LIVE, age_s=2, pid=812)
    return store


def _held(state, *, scan_owner=DEAD, dead_share=2, live_share=1):
    state.update(
        {
            "busy": True,
            "request_pending": True,
            "scanning": scan_owner is not None,
            "scanning_exclusive": scan_owner is not None,
            "scan_owner_worker_id": scan_owner,
            "scan_started_at": SCAN_START if scan_owner else None,
            "pending_enqueues": dead_share + live_share,
            OWNERS: {k: v for k, v in ((DEAD, dead_share), (LIVE, live_share)) if v},
        }
    )


def _release_app(store, *, alive=lambda pid, create_time: False, lock_timeout=0.5):
    app = _app(store)
    service = app.state.ops.service
    service.actions = ops_actions.OpsActions(
        service,
        lock_timeout_s=lock_timeout,
        host="host-a",
        alive=alive,
        markers=lambda: {},
    )
    return app


PATH = "/workspaces/fsktm/actions/release-reservations"
BODY = {
    **REASON,
    "confirm": "fsktm",
    "worker_id": DEAD,
    "expected_scan_started_at": SCAN_START,
    "expected_pending": 2,
}


async def test_release_only_touches_the_dead_workers_share(multiprocess):
    state = multiprocess["fsktm:pipeline_status"]
    _held(state)
    store = _store()
    response = await _post(_release_app(store), PATH, BODY)
    body = response.json()
    assert response.status_code == 200, body
    assert body["result"] == "released" and body["released"] is True
    assert body["released_scanning"] is True and body["released_pending"] == 2
    assert body["how"]
    assert state["scanning"] is False and state["scanning_exclusive"] is False
    assert state["scan_owner_worker_id"] is None and state["scan_started_at"] is None
    # Only the dead worker's share went; the live worker's stays counted.
    assert state["pending_enqueues"] == 1 and state[OWNERS] == {LIVE: 1}
    # Never touched: busy and the "process again" request.
    assert state["busy"] is True and state["request_pending"] is True
    rows = _audit(store, "release_reservations")
    assert [r["phase"] for r in rows] == ["intent", "result"]
    assert rows[-1]["outcome"] == "released"
    assert rows[0]["params"]["worker_id"] == DEAD


async def test_a_scan_of_a_live_worker_is_never_released(multiprocess):
    state = multiprocess["fsktm:pipeline_status"]
    _held(state, scan_owner=LIVE)
    store = _store()
    body = {**BODY, "expected_scan_started_at": None}
    response = await _post(_release_app(store), PATH, body)
    result = response.json()
    assert result["result"] == "released" and result["released_scanning"] is False
    assert state["scanning"] is True and state["scan_owner_worker_id"] == LIVE
    assert state["pending_enqueues"] == 1 and state[OWNERS] == {LIVE: 1}


async def test_release_refused_in_single_process(shared):
    _held(shared["fsktm:pipeline_status"])
    store = _store()
    response = await _post(_release_app(store), PATH, BODY)
    assert response.json()["result"] == "single_process"
    assert shared["fsktm:pipeline_status"]["pending_enqueues"] == 3
    assert _audit(store, "release_reservations")[-1]["outcome"] == "single_process"


async def test_release_refuses_a_live_owner(multiprocess):
    state = multiprocess["fsktm:pipeline_status"]
    _held(state)
    store = _store()
    # Fresh heartbeat: alive.
    store.worker_rows[DEAD] = _worker_row(DEAD, age_s=3, pid=811)
    result = (await _post(_release_app(store), PATH, BODY)).json()
    assert result["result"] == "owner_alive" and result["owner_state"] == "alive"
    # Stale heartbeat but the OS says the process runs: unresponsive.
    store.worker_rows[DEAD] = _worker_row(DEAD, age_s=600, pid=811)
    app = _release_app(store, alive=lambda pid, create_time: True)
    result = (await _post(app, PATH, BODY)).json()
    assert result["result"] == "owner_alive"
    assert result["owner_state"] == "unresponsive"
    assert state["pending_enqueues"] == 3 and state["scanning"] is True


async def test_release_refuses_an_unknown_owner(multiprocess):
    _held(multiprocess["fsktm:pipeline_status"])
    store = _store()
    del store.worker_rows[DEAD]
    result = (await _post(_release_app(store), PATH, BODY)).json()
    assert result["result"] == "owner_unknown" and result["released"] is False
    assert multiprocess["fsktm:pipeline_status"]["pending_enqueues"] == 3


async def test_release_cas_mismatch_and_not_held(multiprocess):
    state = multiprocess["fsktm:pipeline_status"]
    _held(state)
    store = _store()
    app = _release_app(store)
    for override in (
        {"expected_pending": 1},
        {"expected_scan_started_at": "2026-11-04T09:30:00+00:00"},
        {"expected_scan_started_at": None},
    ):
        result = (await _post(app, PATH, {**BODY, **override})).json()
        assert result["result"] == "cas_mismatch", override
    result = (await _post(app, PATH, {**BODY, "worker_id": "a" * 32})).json()
    assert result["result"] == "not_held"
    assert state["pending_enqueues"] == 3 and state["scanning"] is True


async def test_release_cas_under_the_lock(multiprocess, monkeypatch):
    """The share changes between the snapshot and the lock (the dead
    worker's bookkeeping cannot, but a test can): refused, nothing changed."""
    state = multiprocess["fsktm:pipeline_status"]
    _held(state)
    store = _store()
    app = _release_app(store)
    actions = app.state.ops.service.actions
    original = actions._worker_state

    async def racing(worker_id, since):
        out = await original(worker_id, since)
        state[OWNERS] = {DEAD: 5, LIVE: 1}
        return out

    monkeypatch.setattr(actions, "_worker_state", racing)
    result = (await _post(app, PATH, BODY)).json()
    assert result["result"] == "cas_mismatch"
    assert state["scanning"] is True and state[OWNERS] == {DEAD: 5, LIVE: 1}


async def test_release_lock_timeout(multiprocess):
    _held(multiprocess["fsktm:pipeline_status"])
    store = _store()
    app = _release_app(store, lock_timeout=0.3)
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=1
    ):
        pass
    raw = shared_storage._lock_registry["fsktm:pipeline_status:default_key"]
    assert raw.acquire(timeout=1)
    try:
        started = time.monotonic()
        result = (await _post(app, PATH, BODY)).json()
        assert time.monotonic() - started < 2.0
    finally:
        raw.release()
    assert result["result"] == "lock_unavailable" and result["released"] is False
    assert multiprocess["fsktm:pipeline_status"]["pending_enqueues"] == 3
    assert _audit(store, "release_reservations")[-1]["outcome"] == "lock_unavailable"


async def test_release_guards(multiprocess, monkeypatch):
    _held(multiprocess["fsktm:pipeline_status"])
    store = _store()
    app = _release_app(store)
    response = await _post(app, PATH, {**BODY, "confirm": "other"})
    assert response.status_code == 400
    assert response.json()["detail"] == "confirm_mismatch"
    response = await _post(app, PATH, {**BODY, "worker_id": "z" * 32})
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_worker_id"
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    response = await _post(app, PATH, BODY)
    assert response.status_code == 403 and response.json()["detail"] == "not_operator"
    monkeypatch.delenv("OPS_OPERATOR_ACCOUNTS")
    monkeypatch.setenv("OPS_ACTIONS_ENABLED", "false")
    response = await _post(app, PATH, BODY)
    assert response.status_code == 404


async def test_status_offers_release_reservations_like_release_busy(
    monkeypatch, shared
):
    import httpx

    from tests.ops_fixtures import ops_routes

    monkeypatch.setattr(ops_routes, "process_mode", lambda: "gunicorn")
    app = _app()
    app.state.ops.service.run_tracking = True
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        features = (await client.get("/admin/ops/status")).json()["features"]
    assert features["release_reservations"] is True
    app.state.ops.service.run_tracking = False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        features = (await client.get("/admin/ops/status")).json()["features"]
    assert features["release_reservations"] is False


async def test_reconcile_reports_a_dead_workers_share_at_once(monkeypatch):
    """A share whose worker the OS confirms dead is reported on the first
    cycle (``owner_dead``, naming the worker), while a live worker's share
    held by a live run is not; nothing is cleared."""
    from lightrag.api.ops import reconcile
    from lightrag.api.ops.service import RESERVATIONS_KEY, OpsService
    from lightrag.api.ops.snapshot import NamespaceReader
    from lightrag.api.ops.store import MemoryOpsStore

    data = {
        "fsktm:pipeline_status": {
            "busy": False,
            "pending_enqueues": 3,
            OWNERS: {DEAD: 2, LIVE: 1},
        },
    }
    monkeypatch.setattr(shared_storage, "_shared_dicts", data)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    store = MemoryOpsStore()
    store.add_doc("fsktm", "doc-1", "processing")
    store.worker_rows[DEAD] = {
        **_worker_row(DEAD, age_s=900, pid=999999),
        "host": reconcile.socket.gethostname()[:255],
    }
    store.worker_rows[LIVE] = {
        **_worker_row(LIVE, age_s=2, pid=812),
        "host": reconcile.socket.gethostname()[:255],
    }
    service = OpsService(
        store=store,
        reader=NamespaceReader(),
        app_state=SimpleNamespace(),
        run_tracking=True,
    )
    service.worker_id = "0" * 32

    class Loop:
        holder = "0" * 32

        async def still_held(self):
            return True

    await reconcile.reconcile_cycle(
        Loop(), service=service, process=lambda: {"mode": "gunicorn"}
    )
    [item] = store.kv[RESERVATIONS_KEY]["items"]
    assert item["kind"] == "pending_enqueues" and item["reason_code"] == "owner_dead"
    assert item["owner_worker_id"] == DEAD
    assert item["dead_owners"] == [{"worker_id": DEAD, "count": 2}]
    assert item["count"] == 2 and item["pending_enqueues"] == 3
    assert data["fsktm:pipeline_status"]["pending_enqueues"] == 3
