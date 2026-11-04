"""Operations console admin actions (PR-7): cancel, run now, drain the email
queue, release a stale busy slot, the audited reveal, and the bounded
namespace lock they all use.

Every action follows §3.9: live admin + CSRF at the route, the per-workspace
action lease (``in_progress_by``), the intent audit row (503 when it cannot
be written, and nothing changes), the preconditions, the namespace lock with
a 2 s timeout (``lock_unavailable``, audited), the result row, and 10
actions per minute per actor.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from lightrag.api.ops import actions as ops_actions
from lightrag.api.ops import auth as ops_auth
from lightrag.api.ops.auth import require_live_admin
from lightrag.api.ops.store import MemoryOpsStore
from lightrag.kg import shared_storage
from lightrag.usage import current_usage_scope
from tests.ops_fixtures import (
    SECRET_EMAIL,
    SECRET_FILE,
    SECRET_KEY,
    admin,
    build_app,
    ops_routes,
    seeded_store,
)

pytestmark = pytest.mark.offline

CLIENT = {"X-LightRAG-Client": "webui"}
RUN = "5d1e0c3f9a2b4c6d8e0f1a2b3c4d5e6f"
OWNER = "fedcba9876543210fedcba9876543210"
JOB_START = "2026-11-04T09:00:00+00:00"
BASE = "/admin/ops"


# -- fixtures -----------------------------------------------------------------------


def _pipeline(**fields: Any) -> dict[str, Any]:
    state = {
        "busy": False,
        "destructive_busy": False,
        "scanning": False,
        "scanning_exclusive": False,
        "pending_enqueues": 0,
        "request_pending": False,
        "cancellation_requested": False,
        "job_name": "-",
        "job_start": None,
        "latest_message": "",
        "history_messages": [],
    }
    state.update(fields)
    return state


@pytest.fixture
def shared(monkeypatch):
    """Fresh single-process shared state with real keyed locks."""
    data: dict[str, Any] = {
        "fsktm:pipeline_status": _pipeline(),
        "space1:pipeline_status": _pipeline(),
        "hr_office:pipeline_status": _pipeline(),
    }
    monkeypatch.setattr(shared_storage, "_shared_dicts", data)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    monkeypatch.setattr(
        shared_storage, "_storage_keyed_lock", shared_storage.KeyedUnifiedLock()
    )
    return data


@pytest.fixture
def multiprocess(monkeypatch, shared):
    """Gunicorn mode with an in-process stand-in for the Manager: the raw
    locks are ``threading.Lock``s, so a "dead worker" can hold one."""
    monkeypatch.setattr(shared_storage, "_is_multiprocess", True)
    monkeypatch.setattr(
        shared_storage, "_manager", SimpleNamespace(Lock=threading.Lock)
    )
    monkeypatch.setattr(shared_storage, "_lock_registry", {})
    monkeypatch.setattr(shared_storage, "_lock_registry_count", {})
    monkeypatch.setattr(shared_storage, "_lock_cleanup_data", {})
    monkeypatch.setattr(shared_storage, "_registry_guard", threading.RLock())
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "gunicorn")
    return shared


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("OPS_ACTIONS_ENABLED", "true")
    monkeypatch.delenv("OPS_OPERATOR_ACCOUNTS", raising=False)


class FakeRag:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.runs: list[str] = []
        self.done = asyncio.Event()

    async def apipeline_process_enqueue_documents(self):
        self.runs.append(current_usage_scope().source)
        self.done.set()


class FakeManager:
    def __init__(self):
        self.instances: dict[str, FakeRag] = {}
        self.deleting: set[str] = set()

    async def get_instance(self, workspace: str):
        from lightrag.api.workspace_manager import WorkspaceDeletionInProgress

        if workspace in self.deleting:
            raise WorkspaceDeletionInProgress(workspace)
        return self.instances.setdefault(workspace, FakeRag(workspace))

    def loaded_instances(self):
        return list(self.instances.values())


def _app(store=None, *, manager=None, live_admin=True, lock_timeout=0.3):
    store = store if store is not None else seeded_store()
    app = build_app(store, manager=manager or FakeManager())
    if live_admin:
        app.dependency_overrides[require_live_admin] = admin
    service = app.state.ops.service
    service.worker_id = "0123456789abcdef0123456789abcdef"
    service.actions = ops_actions.OpsActions(
        service, lock_timeout_s=lock_timeout, host="host-a"
    )
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def _post(app, path: str, body: dict, headers=CLIENT) -> httpx.Response:
    async with _client(app) as client:
        return await client.post(BASE + path, json=body, headers=headers)


def _audit(store: MemoryOpsStore, action: str) -> list[dict[str, Any]]:
    """This test's rows (the seeded store holds one older row by another actor)."""
    return [
        r for r in store.audit_rows if r["action"] == action and r["actor"] == "root"
    ]


REASON = {"reason": "ticket 4411: stuck after deploy"}


# -- the bounded namespace lock --------------------------------------------------------


async def test_try_acquire_single_process_acquires_and_times_out(shared):
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=1
    ) as lock:
        assert lock.acquired and bool(lock)
        started = time.monotonic()
        async with shared_storage.try_acquire_namespace_lock(
            "pipeline_status", "fsktm", timeout=0.2
        ) as second:
            assert not second.acquired
        assert 0.15 < time.monotonic() - started < 1.0
    # Nothing leaked: the regular lock and a new attempt both get it.
    async with shared_storage.get_namespace_lock("pipeline_status", "fsktm"):
        pass
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=0.2
    ) as again:
        assert again.acquired


async def test_try_acquire_serialises_with_get_namespace_lock(shared):
    held = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        async with shared_storage.get_namespace_lock("pipeline_status", "fsktm"):
            held.set()
            await release.wait()

    task = asyncio.create_task(holder())
    await held.wait()
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=0.2
    ) as lock:
        assert not lock.acquired
    release.set()
    await task
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=0.2
    ) as lock:
        assert lock.acquired


async def test_try_acquire_cancelled_while_waiting_holds_nothing(shared):
    async with shared_storage.get_namespace_lock("pipeline_status", "fsktm"):

        async def attempt():
            async with shared_storage.try_acquire_namespace_lock(
                "pipeline_status", "fsktm", timeout=5
            ):
                pass

        task = asyncio.create_task(attempt())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=0.2
    ) as lock:
        assert lock.acquired


async def test_try_acquire_multiprocess_does_not_block_the_loop(multiprocess):
    # A dead worker holds the Manager lock (another thread here).
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=1
    ) as lock:
        assert lock.acquired
    raw = shared_storage._lock_registry["fsktm:pipeline_status:default_key"]
    assert raw.acquire(timeout=1)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    tick_task = asyncio.create_task(ticker())
    started = time.monotonic()
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=0.4
    ) as lock:
        assert not lock.acquired
    elapsed = time.monotonic() - started
    tick_task.cancel()
    assert 0.3 < elapsed < 1.5
    # The event loop kept running while the Manager call waited.
    assert ticks >= 10
    raw.release()
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=1
    ) as lock:
        assert lock.acquired
    assert shared_storage._lock_registry_count["fsktm:pipeline_status:default_key"] == 0


def test_existing_namespace_data_never_creates(shared):
    assert shared_storage.existing_namespace_data("pipeline_status", "nobody") is None
    assert "nobody:pipeline_status" not in shared
    assert (
        shared_storage.existing_namespace_data("pipeline_status", "fsktm")
        is (shared["fsktm:pipeline_status"])
    )


# -- flag, guards, rate limit ----------------------------------------------------------

ACTION_POSTS = [
    ("/workspaces/fsktm/actions/cancel", REASON),
    ("/workspaces/fsktm/actions/run", {**REASON, "dry_run": True}),
    ("/workspaces/fsktm/actions/drain-email", REASON),
    (
        "/workspaces/fsktm/actions/release-busy",
        {**REASON, "confirm": "fsktm"},
    ),
    ("/reveal", {**REASON, "kind": "doc_file_path", "workspace": "fsktm", "id": "x"}),
]


@pytest.mark.parametrize("path,body", ACTION_POSTS)
async def test_flag_off_hides_the_action_routes(monkeypatch, shared, path, body):
    monkeypatch.delenv("OPS_ACTIONS_ENABLED")
    response = await _post(_app(), path, body)
    assert response.status_code == 404
    assert response.json()["detail"] == "not_found"


async def test_status_reports_the_action_features(monkeypatch, shared):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "gunicorn")
    app = _app()
    app.state.ops.service.run_tracking = True
    async with _client(app) as client:
        features = (await client.get(BASE + "/status")).json()["features"]
    assert features["actions"] and features["release_busy"] and features["audit"]
    monkeypatch.delenv("OPS_ACTIONS_ENABLED")
    async with _client(_app()) as client:
        features = (await client.get(BASE + "/status")).json()["features"]
    assert not features["actions"] and not features["release_busy"]


@pytest.mark.parametrize("path,body", ACTION_POSTS)
async def test_every_action_needs_the_client_header(shared, path, body):
    response = await _post(_app(), path, body, headers={})
    assert response.status_code == 403
    assert response.json()["detail"] == "client_header_required"


@pytest.mark.parametrize("path,body", ACTION_POSTS)
async def test_every_action_refuses_a_removed_admin(monkeypatch, shared, path, body):
    monkeypatch.setattr(ops_auth, "is_live_admin", lambda user: False)
    response = await _post(_app(live_admin=False), path, body)
    assert response.status_code == 403
    assert response.json()["detail"] == "admin_not_current"


async def test_impersonated_sessions_are_refused(shared):
    app = _app(live_admin=False)
    from lightrag.api.dependencies import UserInfo, require_admin

    app.dependency_overrides[require_admin] = lambda: UserInfo(
        username="owner",
        role="admin",
        workspace_id="w",
        metadata={"impersonated_by": "root", "impersonation": True},
    )
    response = await _post(app, "/workspaces/fsktm/actions/cancel", REASON)
    assert response.status_code == 403
    assert response.json()["detail"] == "impersonation_not_allowed"


def test_every_action_route_has_the_guards():
    names = {}
    for route in ops_routes.router.routes:
        if "POST" not in route.methods:
            continue
        names[route.path] = {
            getattr(d.call, "__name__", "") for d in route.dependant.dependencies
        }
    for path, _ in ACTION_POSTS:
        guards = names[BASE + path.replace("fsktm", "{ws}")]
        assert {
            "require_actions_enabled",
            "require_live_admin",
            "require_ops_client",
        } <= guards, path
    for path in ("/workspaces/{ws}/actions/release-busy", "/reveal", "/audit/export"):
        assert "require_operator" in names[BASE + path], path


async def test_reason_is_required(shared):
    response = await _post(
        _app(), "/workspaces/fsktm/actions/cancel", {"reason": "  x "}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_reason"
    response = await _post(_app(), "/workspaces/fsktm/actions/run", {})
    assert response.status_code == 400


async def test_ten_actions_per_minute_per_actor(shared):
    store = seeded_store()
    app = _app(store)
    for _ in range(10):
        response = await _post(app, "/workspaces/space1/actions/cancel", REASON)
        assert response.json()["result"] == "not_busy"
    response = await _post(app, "/workspaces/space1/actions/cancel", REASON)
    assert response.status_code == 429
    assert response.json()["detail"] == "rate_limited"
    assert 1 <= int(response.headers["retry-after"]) <= 60
    # The refused one did nothing and was not audited.
    assert len(_audit(store, "cancel")) == 20


# -- the seven steps -------------------------------------------------------------------


async def test_cancel_sets_the_flag_like_the_documents_route(shared):
    shared["fsktm:pipeline_status"].update(
        busy=True, request_pending=True, job_name="Default Job"
    )
    store = seeded_store()
    response = await _post(_app(store), "/workspaces/fsktm/actions/cancel", REASON)
    body = response.json()
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert body["result"] == "cancellation_requested"
    state = shared["fsktm:pipeline_status"]
    # The same fields document_routes.cancel_pipeline sets, nothing else.
    assert state["cancellation_requested"] is True
    assert state["busy"] is True and state["request_pending"] is True
    assert state["latest_message"] == ops_actions.CANCEL_MESSAGE
    assert state["history_messages"] == [ops_actions.CANCEL_MESSAGE]
    intent, result = _audit(store, "cancel")
    assert intent["phase"] == "intent" and result["phase"] == "result"
    assert body["audit_id"] == intent["id"]
    assert result["outcome"] == "cancellation_requested"
    assert result["params"]["intent_id"] == intent["id"]
    assert intent["reason"] == REASON["reason"]
    assert intent["workspace"] == "fsktm"
    # The lease is gone once the action finished.
    assert "action:fsktm" not in store.kv


async def test_cancel_is_a_no_op_when_not_busy(shared):
    store = seeded_store()
    response = await _post(_app(store), "/workspaces/space1/actions/cancel", REASON)
    assert response.json()["result"] == "not_busy"
    assert shared["space1:pipeline_status"]["cancellation_requested"] is False
    assert _audit(store, "cancel")[-1]["outcome"] == "not_busy"


async def test_lease_conflict_returns_in_progress_by(shared):
    shared["fsktm:pipeline_status"]["busy"] = True
    store = seeded_store()
    now = time.time()
    store.kv["action:fsktm"] = {
        "holder": "other",
        "exp": now + 60,
        "actor": "khairil",
        "action": "run",
        "since": now - 3,
    }
    response = await _post(_app(store), "/workspaces/fsktm/actions/cancel", REASON)
    body = response.json()
    assert body["result"] == "in_progress_by"
    assert body["in_progress_by"]["actor"] == "khairil"
    assert body["in_progress_by"]["action"] == "run"
    assert body["audit_id"] is None
    assert shared["fsktm:pipeline_status"]["cancellation_requested"] is False
    assert _audit(store, "cancel") == []
    assert store.kv["action:fsktm"]["holder"] == "other"


async def test_concurrent_actions_on_one_workspace(shared):
    shared["fsktm:pipeline_status"]["busy"] = True
    store = seeded_store()
    app = _app(store, lock_timeout=2)
    held = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with shared_storage.get_namespace_lock("pipeline_status", "fsktm"):
            held.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await held.wait()
    first = asyncio.create_task(_post(app, "/workspaces/fsktm/actions/cancel", REASON))
    await asyncio.sleep(0.2)  # the first one holds the lease, waiting for the lock
    second = await _post(app, "/workspaces/fsktm/actions/cancel", REASON)
    assert second.json()["result"] == "in_progress_by"
    release.set()
    await holder
    assert (await first).json()["result"] == "cancellation_requested"


async def test_intent_audit_failure_refuses_with_503_and_changes_nothing(shared):
    shared["fsktm:pipeline_status"]["busy"] = True
    store = seeded_store()

    async def broken(row):
        raise ConnectionError("database gone")

    store.audit_write = broken
    response = await _post(_app(store), "/workspaces/fsktm/actions/cancel", REASON)
    assert response.status_code == 503
    assert response.json()["detail"] == "audit_unavailable"
    assert shared["fsktm:pipeline_status"]["cancellation_requested"] is False
    assert "action:fsktm" not in store.kv


async def test_lock_timeout_is_lock_unavailable_and_audited(shared):
    """The held-lock drill: another task holds the lock; the action answers
    lock_unavailable within ~2 s and nothing deadlocks."""
    shared["fsktm:pipeline_status"]["busy"] = True
    store = seeded_store()
    app = _app(store, lock_timeout=ops_actions.LOCK_TIMEOUT_S)
    held = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with shared_storage.get_namespace_lock("pipeline_status", "fsktm"):
            held.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    await held.wait()
    started = time.monotonic()
    response = await _post(app, "/workspaces/fsktm/actions/cancel", REASON)
    elapsed = time.monotonic() - started
    body = response.json()
    assert body["result"] == "lock_unavailable"
    assert body["guidance"] == "restart_container"
    assert 1.8 <= elapsed < 3.0
    assert shared["fsktm:pipeline_status"]["cancellation_requested"] is False
    assert _audit(store, "cancel")[-1]["outcome"] == "lock_unavailable"
    release.set()
    await asyncio.wait_for(holder, timeout=1)
    # No deadlock: the next action gets the lock at once.
    response = await _post(app, "/workspaces/fsktm/actions/cancel", REASON)
    assert response.json()["result"] == "cancellation_requested"


async def test_action_errors_are_audited_and_generic(shared, monkeypatch):
    store = seeded_store()

    def explode(namespace, workspace=None):
        raise RuntimeError(f"boom {SECRET_KEY}")

    monkeypatch.setattr(shared_storage, "existing_namespace_data", explode)
    response = await _post(_app(store), "/workspaces/fsktm/actions/cancel", REASON)
    assert response.status_code == 500
    assert response.json()["detail"] == "internal_error"
    assert SECRET_KEY not in response.text
    assert _audit(store, "cancel")[-1]["outcome"] == "error"
    assert "action:fsktm" not in store.kv


# -- run pipeline now ---------------------------------------------------------------------


async def test_run_dry_run_counts_cancelled_and_changes_nothing(shared):
    store = seeded_store()
    response = await _post(
        _app(store), "/workspaces/fsktm/actions/run", {"dry_run": True}
    )
    body = response.json()
    assert body["dry_run"] is True
    # 1 pending; failed 2 = 1 real + 1 'User cancelled…'; dup rows excluded.
    assert body["would_process"] == {
        "pending": 1,
        "failed": 2,
        "cancelled": 1,
        "in_flight": 2,
    }
    assert body["total"] == 5
    assert body["live"] == {
        "busy": False,
        "destructive_busy": False,
        "paused": False,
        "pause_retry_at": None,
    }
    assert store.audit_rows[-1]["action"] != "run"  # a preview is not audited


async def test_run_starts_the_pipeline_in_an_admin_scope(shared):
    store = seeded_store()
    manager = FakeManager()
    response = await _post(
        _app(store, manager=manager), "/workspaces/fsktm/actions/run", REASON
    )
    body = response.json()
    assert body["result"] == "started"
    assert body["would_process"]["cancelled"] == 1
    rag = manager.instances["fsktm"]
    await asyncio.wait_for(rag.done.wait(), timeout=2)
    assert rag.runs == ["admin"]
    assert _audit(store, "run")[-1]["outcome"] == "started"


@pytest.mark.parametrize(
    "fields,result,reason",
    [
        ({"busy": True}, "refused_busy", "processing"),
        ({"busy": True, "destructive_busy": True}, "refused_busy", "destructive"),
    ],
)
async def test_run_is_refused_while_busy(shared, fields, result, reason):
    shared["fsktm:pipeline_status"].update(fields)
    manager = FakeManager()
    response = await _post(
        _app(manager=manager), "/workspaces/fsktm/actions/run", REASON
    )
    body = response.json()
    assert body["result"] == result and body["reason"] == reason
    assert manager.instances == {}


async def test_run_is_refused_while_paused_unless_overridden(shared):
    pause = {"role": "extract", "kind": "budget", "retry_at": time.time() + 600}
    shared["hr_office:pipeline_status"]["provider_rejection"] = dict(pause)
    manager = FakeManager()
    app = _app(manager=manager)
    response = await _post(app, "/workspaces/hr_office/actions/run", REASON)
    assert response.json()["result"] == "refused_paused"
    assert manager.instances == {}
    response = await _post(
        app, "/workspaces/hr_office/actions/run", {**REASON, "override_pause": True}
    )
    body = response.json()
    assert body["result"] == "started" and body["pause_overridden"] is True
    rejection = shared["hr_office:pipeline_status"]["provider_rejection"]
    # The provider is retried now: the next document becomes the probe.
    assert rejection["retry_at"] <= time.time() and rejection["probe_doc"] is None
    assert rejection["kind"] == "budget"
    await asyncio.wait_for(manager.instances["hr_office"].done.wait(), timeout=2)


async def test_run_override_leaves_the_pause_when_the_workspace_cannot_load(shared):
    """The override is written only once the instance is loaded: a refusal
    after it must not have lifted the pause (a refusal changes nothing)."""
    pause = {"role": "extract", "kind": "budget", "retry_at": time.time() + 600}
    shared["hr_office:pipeline_status"]["provider_rejection"] = dict(pause)
    manager = FakeManager()
    manager.deleting.add("hr_office")
    response = await _post(
        _app(manager=manager),
        "/workspaces/hr_office/actions/run",
        {**REASON, "override_pause": True},
    )
    body = response.json()
    assert body["result"] == "refused_busy" and body["reason"] == "workspace_deleting"
    assert shared["hr_office:pipeline_status"]["provider_rejection"] == pause


async def test_run_with_nothing_to_process_is_a_no_op(shared):
    manager = FakeManager()
    response = await _post(
        _app(manager=manager), "/workspaces/quiet_ws/actions/run", REASON
    )
    assert response.json()["result"] == "nothing_to_process"
    assert manager.instances == {}


async def test_run_refuses_a_workspace_being_deleted(shared):
    manager = FakeManager()
    manager.deleting.add("fsktm")
    response = await _post(
        _app(manager=manager), "/workspaces/fsktm/actions/run", REASON
    )
    body = response.json()
    assert body["result"] == "refused_busy" and body["reason"] == "workspace_deleting"


# -- drain the email queue now ---------------------------------------------------------------


def _resync_queue(tmp_path, *, min_backoff=60.0):
    from lightrag.api.email_resync_queue import DeferredResyncQueue

    async def get_rag(_workspace):
        return object()

    return DeferredResyncQueue(
        str(tmp_path), get_rag, min_backoff=min_backoff, max_backoff=min_backoff
    )


def _seed_queue(tmp_path, workspace="space1", n=2, bad=1):
    directory = tmp_path / ".email_resync_queue" / workspace
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    for i in range(n):
        (directory / f"r{i}.json").write_text(json.dumps({"version": 2}))
    for i in range(bad):
        (directory / f"b{i}.bad").write_text("not json")
    return directory


async def test_drain_kick_cuts_a_drainers_backoff_and_is_limited(shared, tmp_path):
    directory = _seed_queue(tmp_path)
    queue = _resync_queue(tmp_path)
    attempts: list[float] = []

    async def apply(rag, workspace, paths):
        attempts.append(time.monotonic())
        if len(attempts) == 1:
            return False  # slot busy: a minute of backoff
        for path in paths:
            path.unlink()
        return True

    queue._apply_batch = apply
    store = seeded_store()
    app = _app(store)
    app.state.email_resync_queue = queue
    queue.ensure_draining("space1")
    await asyncio.sleep(0.1)
    assert queue.drainer_states()[0]["state"] == "backoff"
    response = await _post(app, "/workspaces/space1/actions/drain-email", REASON)
    body = response.json()
    assert body["result"] == "kicked" and body["pending"] == 2
    await asyncio.wait_for(queue._drainers["space1"], timeout=3)
    assert len(attempts) == 2 and attempts[1] - attempts[0] < 3
    # .bad files are never touched; the kick is in KV for other processes.
    assert (directory / "b0.bad").exists()
    assert store.kv["kick:space1"]["at"] > time.time() - 10
    # Once a minute per workspace.
    (directory / "r9.json").write_text("{}")
    response = await _post(app, "/workspaces/space1/actions/drain-email", REASON)
    assert response.json()["result"] == "rate_limited"
    assert _audit(store, "drain_email")[-1]["outcome"] == "rate_limited"
    await queue.stop()


async def test_drain_starts_a_local_drainer_when_none_runs(shared, tmp_path):
    _seed_queue(tmp_path, bad=0)
    queue = _resync_queue(tmp_path)
    started = []
    queue.ensure_draining = lambda ws: started.append(ws)
    app = _app()
    app.state.email_resync_queue = queue
    response = await _post(app, "/workspaces/space1/actions/drain-email", REASON)
    assert response.json()["result"] == "started_local"
    assert started == ["space1"]


async def test_drain_with_nothing_queued_is_a_no_op(shared, tmp_path):
    _seed_queue(tmp_path, n=0, bad=2)
    store = seeded_store()
    app = _app(store)
    app.state.email_resync_queue = _resync_queue(tmp_path)
    response = await _post(app, "/workspaces/space1/actions/drain-email", REASON)
    assert response.json()["result"] == "nothing_pending"
    assert "kick:space1" not in store.kv


# -- release a stale busy slot -----------------------------------------------------------------


def _stale_slot(shared, *, heartbeat_age_s=600.0, **fields):
    shared["fsktm:pipeline_status"].update(
        {
            "busy": True,
            "destructive_busy": True,
            "request_pending": True,
            "scanning": True,
            "pending_enqueues": 2,
            "cancellation_requested": True,
            "run_id": RUN,
            "owner_worker_id": OWNER,
            "owner_pid": 811,
            "job_start": JOB_START,
            "heartbeat_at": datetime.fromtimestamp(
                time.time() - heartbeat_age_s, timezone.utc
            ).isoformat(),
            **fields,
        }
    )


def _release_store(*, owner_age_s=600.0, owner_pid=811):
    store = seeded_store()
    store.worker_rows[OWNER] = {
        "worker_id": OWNER,
        "host": "host-a",
        "pid": owner_pid,
        "pid_create_time": 1.0,
        "mode": "gunicorn",
        "started_at": datetime(2026, 11, 4, 8, 0, tzinfo=timezone.utc),
        "heartbeat_at": datetime.fromtimestamp(time.time() - owner_age_s, timezone.utc),
        "exited_at": None,
        "superseded": False,
        "leases_held": [],
        "snapshot": {},
    }
    store.run_rows[RUN] = {"run_id": RUN, "workspace": "fsktm", "status": "running"}
    return store


def _release_app(store, *, alive=False):
    app = _app(store)
    service = app.state.ops.service
    service.actions = ops_actions.OpsActions(
        service,
        lock_timeout_s=0.5,
        host="host-a",
        alive=lambda pid, create_time: alive,
        markers=lambda: {},
    )
    return app


RELEASE = {
    **REASON,
    "confirm": "fsktm",
    "expected_run_id": RUN,
    "expected_job_start": JOB_START,
}


async def test_release_is_refused_in_single_process(shared):
    _stale_slot(shared)
    store = _release_store()
    response = await _post(
        _release_app(store), "/workspaces/fsktm/actions/release-busy", RELEASE
    )
    body = response.json()
    assert body["result"] == "single_process" and body["released"] is False
    assert shared["fsktm:pipeline_status"]["busy"] is True
    assert _audit(store, "release_busy")[-1]["outcome"] == "single_process"


async def test_release_success_keeps_request_pending_and_reservations(multiprocess):
    _stale_slot(multiprocess)
    store = _release_store()
    response = await _post(
        _release_app(store), "/workspaces/fsktm/actions/release-busy", RELEASE
    )
    body = response.json()
    assert body["result"] == "released" and body["released"] is True
    assert body["run_now_offered"] is True
    assert body["run_marked_orphaned"] is True
    state = multiprocess["fsktm:pipeline_status"]
    assert state["busy"] is False and state["destructive_busy"] is False
    assert state["cancellation_requested"] is False
    assert state["run_id"] is None and state["owner_worker_id"] is None
    assert state["owner_pid"] is None
    assert state["latest_message"] == "Released by admin root"
    # Never touched: the "process again" request and the reservations.
    assert state["request_pending"] is True
    assert state["scanning"] is True and state["pending_enqueues"] == 2
    assert store.run_rows[RUN]["status"] == "orphaned"
    assert store.run_rows[RUN]["orphaned_by"] == "admin_release"
    assert _audit(store, "release_busy")[-1]["outcome"] == "released"


async def test_release_is_refused_while_the_owner_is_alive(multiprocess):
    # A fresh slot heartbeat: the run is still progressing.
    _stale_slot(multiprocess, heartbeat_age_s=5)
    store = _release_store()
    response = await _post(
        _release_app(store), "/workspaces/fsktm/actions/release-busy", RELEASE
    )
    assert response.json()["result"] == "owner_alive"
    # A stale heartbeat but a live (wedged) process: restart, not release.
    _stale_slot(multiprocess)
    response = await _post(
        _release_app(store, alive=True),
        "/workspaces/fsktm/actions/release-busy",
        RELEASE,
    )
    body = response.json()
    assert body["result"] == "owner_alive" and body["owner_state"] == "unresponsive"
    assert multiprocess["fsktm:pipeline_status"]["busy"] is True
    assert store.run_rows[RUN]["status"] == "running"


async def test_release_cas_mismatch_changes_nothing(multiprocess):
    _stale_slot(multiprocess)
    store = _release_store()
    app = _release_app(store)
    for override in (
        {"expected_run_id": "0" * 32},
        {"expected_job_start": "2026-11-04T09:30:00+00:00"},
    ):
        response = await _post(
            app, "/workspaces/fsktm/actions/release-busy", {**RELEASE, **override}
        )
        assert response.json()["result"] == "cas_mismatch"
    assert multiprocess["fsktm:pipeline_status"]["busy"] is True
    assert store.run_rows[RUN]["status"] == "running"


async def test_release_needs_the_typed_workspace_name(multiprocess):
    _stale_slot(multiprocess)
    response = await _post(
        _release_app(_release_store()),
        "/workspaces/fsktm/actions/release-busy",
        {**RELEASE, "confirm": "fsktmx"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "confirm_mismatch"


async def test_release_is_operator_only(multiprocess, monkeypatch):
    _stale_slot(multiprocess)
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    response = await _post(
        _release_app(_release_store()),
        "/workspaces/fsktm/actions/release-busy",
        RELEASE,
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "not_operator"


async def test_release_lock_held_by_a_dead_worker(multiprocess):
    """The dead worker died inside the critical section: its Manager lock
    stays held. Release answers lock_unavailable within the timeout."""
    _stale_slot(multiprocess)
    store = _release_store()
    app = _release_app(store)
    async with shared_storage.try_acquire_namespace_lock(
        "pipeline_status", "fsktm", timeout=1
    ):
        pass
    raw = shared_storage._lock_registry["fsktm:pipeline_status:default_key"]
    assert raw.acquire(timeout=1)
    try:
        started = time.monotonic()
        response = await _post(app, "/workspaces/fsktm/actions/release-busy", RELEASE)
        assert time.monotonic() - started < 2.5
    finally:
        raw.release()
    body = response.json()
    assert body["result"] == "lock_unavailable" and body["released"] is False
    assert body["guidance"] == "restart_container"
    assert multiprocess["fsktm:pipeline_status"]["busy"] is True
    assert _audit(store, "release_busy")[-1]["outcome"] == "lock_unavailable"


async def test_release_without_owner_keys_is_never_released(multiprocess):
    """A slot without owner keys (run tracking off, or its hook failed) has
    no owner that can be confirmed dead: heartbeats list only the busiest
    instances, so a live pipeline in Docling or between LLM calls would look
    unowned. Refused, whatever the workers report."""
    _stale_slot(
        multiprocess,
        run_id=None,
        owner_worker_id=None,
        owner_pid=None,
        heartbeat_at=None,
    )
    store = _release_store()
    del store.run_rows[RUN]
    store.worker_rows.clear()
    app = _release_app(store)
    body = {**RELEASE, "expected_run_id": None}
    response = await _post(app, "/workspaces/fsktm/actions/release-busy", body)
    result = response.json()
    assert result["result"] == "owner_unknown" and result["released"] is False
    state = multiprocess["fsktm:pipeline_status"]
    assert state["busy"] is True and state["destructive_busy"] is True
    assert _audit(store, "release_busy")[-1]["outcome"] == "owner_unknown"


async def test_status_offers_release_only_with_run_tracking(monkeypatch, shared):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "gunicorn")
    app = _app()
    app.state.ops.service.run_tracking = False
    async with _client(app) as client:
        features = (await client.get(BASE + "/status")).json()["features"]
    assert features["actions"] and not features["release_busy"]


async def test_release_cas_compares_the_confirmed_snapshot(multiprocess, monkeypatch):
    """The owner was confirmed dead from one snapshot; by the time the lock
    is taken another worker holds a new slot with the same run_id and
    job_start (a destructive slot without a run). Its owner keys differ, so
    the CAS refuses and the live slot is left alone."""
    _stale_slot(multiprocess)
    store = _release_store()
    app = _release_app(store)
    actions = app.state.ops.service.actions
    original = actions._owner_state

    async def confirm_then_retake(workspace, snap):
        state = await original(workspace, snap)
        multiprocess["fsktm:pipeline_status"].update(
            {
                "owner_worker_id": "b" * 32,
                "owner_pid": 912,
                "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return state

    monkeypatch.setattr(actions, "_owner_state", confirm_then_retake)
    response = await _post(app, "/workspaces/fsktm/actions/release-busy", RELEASE)
    assert response.json()["result"] == "cas_mismatch"
    state = multiprocess["fsktm:pipeline_status"]
    assert state["busy"] is True and state["owner_worker_id"] == "b" * 32
    assert store.run_rows[RUN]["status"] == "running"


# -- reveal -------------------------------------------------------------------------------------------


async def test_reveal_unmasks_the_file_name_but_not_secrets(shared):
    store = seeded_store()
    response = await _post(
        _app(store),
        "/reveal",
        {**REASON, "kind": "doc_file_path", "workspace": "fsktm", "id": "doc-aaaa0001"},
    )
    body = response.json()
    assert response.status_code == 200
    # The real name, with the IC number in it still masked.
    assert body["value"] == "Senate minutes <ic>.pdf"
    assert "900101" not in response.text
    intent, result = _audit(store, "reveal")
    assert intent["params"] == {"kind": "doc_file_path", "id": "doc-aaaa0001"}
    assert intent["target"] == "doc-aaaa0001" and result["outcome"] == "ok"
    assert body["audit_id"] == intent["id"]


async def test_reveal_error_keeps_secrets_and_emails_masked(shared):
    response = await _post(
        _app(),
        "/reveal",
        {**REASON, "kind": "doc_error", "workspace": "fsktm", "id": "doc-aaaa0004"},
    )
    value = response.json()["value"]
    assert value.startswith("RateLimitError")
    assert SECRET_EMAIL not in value and SECRET_KEY not in value
    assert "<email>" in value


async def test_reveal_job_name(shared):
    shared["fsktm:pipeline_status"].update(
        job_name=f"{SECRET_FILE[:20]}...[2 files]", run_id=RUN
    )
    app = _app()
    body = {**REASON, "kind": "job_name", "workspace": "fsktm", "id": RUN}
    response = await _post(app, "/reveal", body)
    assert response.json()["value"].startswith("Senate minutes")
    # The job changed since the page was drawn.
    response = await _post(app, "/reveal", {**body, "id": "1" * 32})
    assert response.status_code == 404


async def test_reveal_is_operator_only_and_audited_when_missing(shared, monkeypatch):
    store = seeded_store()
    body = {**REASON, "kind": "doc_file_path", "workspace": "fsktm", "id": "nope"}
    response = await _post(_app(store), "/reveal", body)
    assert response.status_code == 404
    assert _audit(store, "reveal")[-1]["outcome"] == "not_found"
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    response = await _post(_app(store), "/reveal", body)
    assert response.status_code == 403
    assert response.json()["detail"] == "not_operator"


async def test_reveal_is_refused_without_its_audit_row(shared):
    store = seeded_store()

    async def broken(row):
        raise ConnectionError("down")

    store.audit_write = broken
    response = await _post(
        _app(store),
        "/reveal",
        {**REASON, "kind": "doc_file_path", "workspace": "fsktm", "id": "doc-aaaa0001"},
    )
    assert response.status_code == 503
    assert "Senate" not in response.text


async def test_reveal_rejects_unknown_kinds(shared):
    response = await _post(
        _app(),
        "/reveal",
        {**REASON, "kind": "log_text", "workspace": "fsktm", "id": "1"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_kind"


# -- without the ops database ---------------------------------------------------------------------


async def test_actions_without_postgres_answer_503(shared):
    app = build_app(None, kv_storage="JsonKVStorage")
    app.dependency_overrides[require_live_admin] = admin
    response = await _post(app, "/workspaces/fsktm/actions/cancel", REASON)
    assert response.status_code == 503
    assert response.json()["detail"] == "data_unavailable"
    assert shared["fsktm:pipeline_status"]["cancellation_requested"] is False
