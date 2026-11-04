"""No Operations console GET may load a workspace or loop over shared state.

Loading a workspace can evict another one and finalize its running pipeline,
and ``get_namespace_data`` takes the one Manager lock every worker shares. So
with ``WorkspaceManager.get_instance`` (and friends) and ``get_namespace_data``
patched to raise, every GET route must still answer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lightrag.api.workspace_manager import WorkspaceManager
from lightrag.kg import shared_storage
from tests.ops_fixtures import (
    RUN_ID,
    WORKER_ID,
    FakeRegistry,
    build_app,
    ops_routes,
    seed_health,
    seed_logs,
    seed_runs,
    seeded_store,
    shared_state,
)

pytestmark = pytest.mark.offline


def _forbidden(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"Operations GET called {name}")

    return fail


@pytest.fixture
def guarded(monkeypatch):
    for name in (
        "get_instance",
        "get_lightrag_instance",
        "create_workspace",
        "begin_deletion",
    ):
        monkeypatch.setattr(
            WorkspaceManager, name, _forbidden(f"WorkspaceManager.{name}")
        )
    for name in (
        "get_namespace_data",
        "initialize_pipeline_status",
        "get_internal_lock",
        "get_namespace_lock",
        "try_initialize_namespace",
    ):
        monkeypatch.setattr(shared_storage, name, _forbidden(name))
    monkeypatch.setattr(shared_storage, "_shared_dicts", shared_state())
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)


# Routes that need a parameter to name something.
_PARAMS = {
    "/admin/ops/processing/failures/docs": {"key": "fsktm", "group": "workspace"},
    "/admin/ops/correlate": {"id": "fsktm"},
}


def _concrete(path: str) -> str:
    return (
        path.replace("{ws}", "fsktm")
        .replace("{doc_id}", "doc-aaaa0007")
        .replace("{dep}", "postgres")
        .replace("{worker_id}", WORKER_ID)
        .replace("{run_id}", RUN_ID)
        .replace("{log_id}", "1")
        .replace("{alert_id}", "1")
    )


@pytest.mark.parametrize("tracking", [False, True], ids=["legacy", "run_tracking"])
def test_every_get_answers_without_instantiating(guarded, tracking, monkeypatch):
    # The flag-gated Prometheus route is checked too (it answers 404 when off).
    monkeypatch.setenv("OPS_PROMETHEUS", "true")
    manager = WorkspaceManager(instance_factory=_forbidden("instance_factory"))
    store = seeded_store()
    app = build_app(
        store,
        registry=FakeRegistry({"fsktm": "a.bin@unimas.my"}),
        manager=manager,
    )
    # Probe results and heartbeats, so the Health and Workers GETs read data.
    seed_health(app, store)
    # Durable runs and failures, read by the same routes with tracking on.
    seed_runs(store)
    # Stored log lines, read by the Logs routes and the drawers' log sections.
    seed_logs(store)
    app.state.ops.service.run_tracking = tracking
    client = TestClient(app, raise_server_exceptions=True)
    gets = [r for r in ops_routes.router.routes if "GET" in r.methods]
    assert len(gets) >= 20
    for route in gets:
        path = _concrete(route.path)
        response = client.get(path, params=_PARAMS.get(route.path))
        assert response.status_code == 200, (path, response.text)
    # Health and workers views carry data, still without loading anything.
    assert len(client.get("/admin/ops/health").json()["deps"]) >= 4
    assert client.get("/admin/ops/workers").json()["items"]
    assert client.get("/admin/ops/overview").json()["deps"]
    # Workspace-scoped routes on a workspace never loaded anywhere.
    for path in (
        "/admin/ops/workspaces/hr_office",
        "/admin/ops/workspaces/hr_office/feed",
        "/admin/ops/workspaces/hr_office/documents",
    ):
        assert client.get(path).status_code == 200, path
    assert manager.loaded_instances() == []


def test_every_get_answers_without_postgres(guarded, monkeypatch):
    monkeypatch.setenv("OPS_PROMETHEUS", "true")
    app = build_app(None, kv_storage="JsonKVStorage")
    client = TestClient(app, raise_server_exceptions=True)
    for route in ops_routes.router.routes:
        if "GET" in route.methods:
            response = client.get(_concrete(route.path))
            assert response.status_code == 200, (route.path, response.text)
            if response.headers["content-type"].startswith("text/plain"):
                # Prometheus text: only this process's numbers.
                assert "lightrag_ops_store_available 0" in response.text
                continue
            assert (
                response.json().get("available", False) is False
                or route.path == "/admin/ops/status"
            )
