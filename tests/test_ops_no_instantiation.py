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
    FakeRegistry,
    build_app,
    ops_routes,
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
    "/admin/ops/processing/failures/docs": {"key": "fsktm", "group": "workspace"}
}


def _concrete(path: str) -> str:
    return path.replace("{ws}", "fsktm").replace("{doc_id}", "doc-aaaa0007")


def test_every_get_answers_without_instantiating(guarded):
    manager = WorkspaceManager(instance_factory=_forbidden("instance_factory"))
    app = build_app(
        seeded_store(),
        registry=FakeRegistry({"fsktm": "a.bin@unimas.my"}),
        manager=manager,
    )
    client = TestClient(app, raise_server_exceptions=True)
    gets = [r for r in ops_routes.router.routes if "GET" in r.methods]
    assert len(gets) >= 15
    for route in gets:
        path = _concrete(route.path)
        response = client.get(path, params=_PARAMS.get(route.path))
        assert response.status_code == 200, (path, response.text)
    # Workspace-scoped routes on a workspace never loaded anywhere.
    for path in (
        "/admin/ops/workspaces/hr_office",
        "/admin/ops/workspaces/hr_office/feed",
        "/admin/ops/workspaces/hr_office/documents",
    ):
        assert client.get(path).status_code == 200, path
    assert manager.loaded_instances() == []


def test_every_get_answers_without_postgres(guarded):
    app = build_app(None, kv_storage="JsonKVStorage")
    client = TestClient(app, raise_server_exceptions=True)
    for route in ops_routes.router.routes:
        if "GET" in route.methods:
            response = client.get(_concrete(route.path))
            assert response.status_code == 200, (route.path, response.text)
            assert (
                response.json().get("available", False) is False
                or route.path == "/admin/ops/status"
            )
