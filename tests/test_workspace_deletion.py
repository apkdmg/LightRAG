"""Tests for permanent workspace deletion.

Covers registry tombstones, the purge itself (storages, files, provider
credentials, ownership), its failure and concurrency guards, the owner's
``DELETE /workspace`` checks, sessions that outlive their workspace, and the
admin ``DELETE /admin/workspaces/{id}``. Storages are mocks: the per-backend
``drop`` methods have their own tests.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.api.workspace_deletion import WorkspaceBusyError, purge_workspace
from lightrag.api.workspace_manager import (
    WorkspaceDeletionInProgress,
    WorkspaceManager,
)
from lightrag.api.workspace_registry import (
    JsonWorkspaceRegistry,
    local_identity,
    resolve_workspace,
    sso_identity,
)

pytestmark = pytest.mark.offline

STORAGE_NAMES = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "doc_status",
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def shared_storage():
    from lightrag.kg import shared_storage

    shared_storage.initialize_share_data()
    yield
    shared_storage.finalize_share_data()


def fake_rag(workspace: str, **_kwargs) -> MagicMock:
    rag = MagicMock()
    rag.workspace = workspace
    rag.initialize_storages = AsyncMock()
    rag.finalize_storages = AsyncMock()
    for name in STORAGE_NAMES:
        storage = MagicMock()
        storage.workspace = workspace
        storage.namespace = name
        storage.drop = AsyncMock(return_value={"status": "success"})
        storage.drop_workspace_schema = AsyncMock()
        setattr(rag, name, storage)
    return rag


# ---------------------------------------------------------------------------
# Registry tombstones
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    return JsonWorkspaceRegistry(tmp_path)


def test_retire_unbinds_and_keeps_only_hashes(registry, tmp_path):
    ws = run(resolve_workspace(registry, sso_identity("sub-a"), "a@x.my", "a@x.my"))
    run(registry.retire(ws, sso_identity("sub-a")))

    assert run(registry.get_by_workspace(ws)) is None
    tombstone = run(registry.get_tombstone(ws))
    assert tombstone is not None and tombstone.deleted_at <= time.time()
    stored = (tmp_path / ".workspace_tombstones.json").read_text()
    assert ws not in stored and "a@x.my" not in stored and "sub-a" not in stored


def test_former_owner_gets_the_id_back(registry):
    ws = run(resolve_workspace(registry, sso_identity("sub-a"), "a@x.my", "a@x.my"))
    run(registry.retire(ws, sso_identity("sub-a")))
    again = run(resolve_workspace(registry, sso_identity("sub-a"), "a@x.my", "a@x.my"))
    assert again == ws


def test_deleted_id_is_not_handed_to_another_account(registry):
    ws = run(resolve_workspace(registry, sso_identity("sub-a"), "a@x.my", "a@x.my"))
    run(registry.retire(ws, sso_identity("sub-a")))
    # A re-created account with the same email maps to the same ID
    other = run(resolve_workspace(registry, sso_identity("sub-b"), "a@x.my", "a@x.my"))
    assert other != ws and other.startswith(f"{ws}_")


def test_unbound_deleted_workspace_goes_to_nobody(registry):
    run(registry.retire("legacy_ws", None))
    ws = run(
        resolve_workspace(registry, local_identity("legacy_ws"), None, "legacy_ws")
    )
    assert ws != "legacy_ws"


def test_tombstone_cache_is_refreshed_by_retire(registry):
    assert run(registry.cached_tombstone("alice")) is None
    run(registry.retire("alice", local_identity("alice")))
    assert run(registry.cached_tombstone("alice")) is not None


# ---------------------------------------------------------------------------
# purge_workspace
# ---------------------------------------------------------------------------


@pytest.fixture
def purge_env(tmp_path, monkeypatch):
    # The purge imports the API routers, which read the server's arguments
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    for var in ("MILVUS_WORKSPACE", "POSTGRES_WORKSPACE", "NEO4J_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)
    working_dir = tmp_path / "rag_storage"
    input_dir = tmp_path / "inputs"
    for base in (working_dir, input_dir):
        (base / "alice").mkdir(parents=True)
        (base / "bob").mkdir(parents=True)
    (input_dir / "alice" / "report.pdf").write_text("alice's file")
    (input_dir / "alice" / "__parsed__").mkdir()
    (working_dir / "alice" / ".api_keys.json").write_text("{}")

    registry = JsonWorkspaceRegistry(working_dir)
    run(registry.claim("alice", local_identity("alice"), None))
    provider_store = MagicMock()
    rags: dict[str, MagicMock] = {}

    def factory(workspace):
        rags[workspace] = fake_rag(workspace)
        return rags[workspace]

    manager = WorkspaceManager(instance_factory=factory)

    async def purge(workspace_id="alice"):
        return await purge_workspace(
            workspace_id,
            workspace_manager=manager,
            registry=registry,
            working_dir=working_dir,
            input_dir=input_dir,
            provider_store=provider_store,
        )

    env = MagicMock()
    env.working_dir, env.input_dir = working_dir, input_dir
    env.registry, env.manager, env.provider_store = registry, manager, provider_store
    env.rags, env.purge = rags, purge
    return env


def test_purge_removes_everything_of_the_workspace(purge_env):
    result = run(purge_env.purge())

    assert result.ok, result.errors
    rag = purge_env.rags["alice"]
    for name in STORAGE_NAMES:
        getattr(rag, name).drop.assert_awaited_once()
        getattr(rag, name).drop_workspace_schema.assert_awaited_once()
    rag.finalize_storages.assert_awaited_once()
    assert not (purge_env.input_dir / "alice").exists()
    assert not (purge_env.working_dir / "alice").exists()
    purge_env.provider_store.delete.assert_called_once_with("alice")
    assert run(purge_env.registry.get_by_workspace("alice")) is None
    assert run(purge_env.registry.get_tombstone("alice")) is not None
    assert not purge_env.manager.is_known("alice")
    # Other workspaces are untouched
    assert (purge_env.input_dir / "bob").is_dir()
    assert (purge_env.working_dir / "bob").is_dir()


def test_failed_drop_keeps_files_and_ownership(purge_env):
    async def scenario():
        rag = await purge_env.manager.get_instance("alice")
        rag.chunks_vdb.drop.return_value = {"status": "error", "message": "down"}
        return await purge_env.purge()

    result = run(scenario())

    assert not result.ok
    assert any("down" in error for error in result.errors)
    assert (purge_env.input_dir / "alice" / "report.pdf").exists()
    assert run(purge_env.registry.get_by_workspace("alice")) is not None
    assert run(purge_env.registry.get_tombstone("alice")) is None
    purge_env.provider_store.delete.assert_not_called()
    # The workspace keeps working, so the deletion can be retried
    assert purge_env.manager.is_known("alice")
    assert run(purge_env.purge()).ok is False


def test_retry_after_failure_succeeds(purge_env):
    async def scenario():
        rag = await purge_env.manager.get_instance("alice")
        rag.chunk_entity_relation_graph.drop.side_effect = [
            RuntimeError("timeout"),
            {"status": "success"},
        ]
        first = await purge_env.purge()
        second = await purge_env.purge()
        return first, second

    first, second = run(scenario())
    assert not first.ok and second.ok
    assert not (purge_env.input_dir / "alice").exists()


def test_busy_pipeline_refuses_and_releases(purge_env, monkeypatch):
    from lightrag.api.routers import document_routes

    monkeypatch.setattr(
        document_routes,
        "_acquire_destructive_busy",
        AsyncMock(return_value=(False, "Pipeline is busy with another operation.")),
    )
    with pytest.raises(WorkspaceBusyError):
        run(purge_env.purge())

    assert (purge_env.input_dir / "alice" / "report.pdf").exists()
    for name in STORAGE_NAMES:
        getattr(purge_env.rags["alice"], name).drop.assert_not_awaited()
    # Not left marked as deleting
    run(purge_env.manager.get_instance("alice"))


def test_workspace_is_unavailable_while_being_deleted(purge_env):
    async def scenario():
        await purge_env.manager.begin_deletion("alice")
        with pytest.raises(WorkspaceDeletionInProgress):
            await purge_env.manager.get_instance("alice")
        with pytest.raises(WorkspaceDeletionInProgress):
            await purge_env.purge()
        await purge_env.manager.get_instance("bob")
        await purge_env.manager.end_deletion("alice")
        await purge_env.manager.get_instance("alice")

    run(scenario())


def test_shared_storage_override_blocks_the_purge(purge_env, monkeypatch):
    monkeypatch.setenv("MILVUS_WORKSPACE", "shared")
    result = run(purge_env.purge())

    assert not result.ok
    assert "MILVUS_WORKSPACE" in result.errors[0]
    for name in STORAGE_NAMES:
        getattr(purge_env.rags["alice"], name).drop.assert_not_awaited()
    assert (purge_env.input_dir / "alice").exists()


def test_storage_scoped_to_another_workspace_blocks_the_purge(purge_env):
    async def scenario():
        rag = await purge_env.manager.get_instance("alice")
        rag.entities_vdb.workspace = "everyone"
        return await purge_env.purge()

    result = run(scenario())
    assert not result.ok
    purge_env.rags["alice"].full_docs.drop.assert_not_awaited()


@pytest.mark.parametrize("bad_id", ["", "../alice", ".", "Alice", "a/b"])
def test_invalid_ids_are_rejected(purge_env, bad_id):
    with pytest.raises(ValueError):
        run(purge_env.purge(bad_id))


def test_stale_instance_is_rebuilt_after_deletion_elsewhere(tmp_path):
    registry = JsonWorkspaceRegistry(tmp_path)
    built = []

    def factory(workspace):
        built.append(fake_rag(workspace))
        return built[-1]

    async def is_stale(workspace_id, created_at):
        tombstone = await registry.cached_tombstone(workspace_id)
        return tombstone is not None and tombstone.deleted_at > created_at

    manager = WorkspaceManager(instance_factory=factory, instance_is_stale=is_stale)

    async def scenario():
        first = await manager.get_instance("alice")
        assert await manager.get_instance("alice") is first
        await registry.retire("alice", local_identity("alice"))
        second = await manager.get_instance("alice")
        assert second is not first
        first.finalize_storages.assert_awaited_once()
        assert await manager.get_instance("alice") is second

    run(scenario())


# ---------------------------------------------------------------------------
# Server: DELETE /workspace and DELETE /admin/workspaces/{id}
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    for var in ("LLM_BINDING_HOST", "EMBEDDING_BINDING_HOST", "LIGHTRAG_API_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api import config, lightrag_server, utils_api
    from lightrag.api.config import parse_args

    args = parse_args()
    args.working_dir = str(tmp_path / "rag_storage")
    args.input_dir = str(tmp_path / "inputs")
    args.enable_multi_tenancy = True
    args.kv_storage = "JsonKVStorage"
    # The purge reads directories from global_args: never the repo's own
    monkeypatch.setattr(config, "_global_args", args)
    monkeypatch.setattr(config, "_initialized", True)
    # Other test modules reload utils_api; patch the copy each router holds
    from lightrag.api.routers import document_routes, workspace_routes

    monkeypatch.setattr(utils_api, "auth_configured", True)
    for router_module in (document_routes, workspace_routes):
        monkeypatch.setitem(
            router_module.get_combined_auth_dependency.__globals__,
            "auth_configured",
            True,
        )
    monkeypatch.setattr(
        lightrag_server.auth_handler,
        "accounts",
        {"admin": "admin-pw", "alice": "alice-pw"},
    )
    monkeypatch.setattr(
        lightrag_server, "_is_admin_user", lambda *names: "admin" in names
    )
    import lightrag.api.auth as current_auth

    monkeypatch.setattr(current_auth, "auth_handler", lightrag_server.auth_handler)
    rags: dict[str, MagicMock] = {}

    def make_rag(*_args, workspace="", **_kwargs):
        rags[workspace] = fake_rag(workspace)
        return rags[workspace]

    with patch("lightrag.api.lightrag_server.LightRAG", side_effect=make_rag):
        client = TestClient(lightrag_server.create_app(args))
        yield client, lightrag_server, rags, tmp_path


def _login(client, username, password):
    response = client.post("/login", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


CONFIRM = {"confirm": "DELETE MY WORKSPACE"}


def _delete_own(client, token, body=CONFIRM, headers=None):
    return client.request(
        "DELETE", "/workspace", json=body, headers={**_bearer(token), **(headers or {})}
    )


def test_owner_deletes_own_workspace(app_client):
    client, server, rags, tmp_path = app_client
    alice = _login(client, "alice", "alice-pw")
    client.get("/documents/pipeline_status", headers=_bearer(alice))
    inputs = tmp_path / "inputs" / "alice"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "notes.txt").write_text("private")

    response = _delete_own(client, alice)

    assert response.status_code == 200, response.text
    cleared = response.headers.get_list("set-cookie")
    assert any(h.startswith("lightrag_token=") and "Max-Age=0" in h for h in cleared)
    for name in STORAGE_NAMES:
        getattr(rags["alice"], name).drop.assert_awaited_once()
    assert not inputs.exists()

    # The old session no longer works anywhere
    stale = client.get("/documents/pipeline_status", headers=_bearer(alice))
    assert stale.status_code == 401
    assert "deleted" in stale.json()["detail"]

    # Signing in again gives the owner a fresh, empty workspace with the same ID
    time.sleep(1.1)  # session start times have one-second resolution
    fresh = _login(client, "alice", "alice-pw")
    assert server.auth_handler.validate_token(fresh)["workspace_id"] == "alice"
    # Accepted (the mocked instance has no pipeline, hence not a plain 200)
    assert (
        client.get("/documents/pipeline_status", headers=_bearer(fresh)).status_code
        != 401
    )


def test_owner_can_see_whether_deletion_is_allowed(app_client):
    client, server, _rags, _tmp = app_client
    alice = _login(client, "alice", "alice-pw")

    info = client.get("/workspace", headers=_bearer(alice)).json()
    assert info["workspace_id"] == "alice" and info["can_delete"] is True
    assert 0 < info["sign_in_seconds_left"] <= 15 * 60

    old = server.auth_handler.create_token(
        username="alice",
        role="user",
        workspace_id="alice",
        metadata={"auth_mode": "enabled"},
        session_started_at=time.time() - 3600,
    )
    info = client.get("/workspace", headers=_bearer(old)).json()
    assert info["can_delete"] is True and info["sign_in_seconds_left"] == 0


def test_request_body_cannot_choose_the_workspace(app_client):
    client, _server, rags, _tmp = app_client
    _login(client, "admin", "admin-pw")
    alice = _login(client, "alice", "alice-pw")

    response = _delete_own(client, alice, body={**CONFIRM, "workspace_id": "admin"})

    assert response.status_code == 200
    assert "admin" not in rags or not rags["admin"].full_docs.drop.await_count


@pytest.mark.parametrize(
    "body", [{}, {"confirm": "yes"}, {"confirm": "delete my workspace"}]
)
def test_confirmation_phrase_is_required(app_client, body):
    client, _server, rags, _tmp = app_client
    alice = _login(client, "alice", "alice-pw")

    response = _delete_own(client, alice, body=body)

    assert response.status_code in (400, 422)
    assert "alice" not in rags or not rags["alice"].full_docs.drop.await_count


def test_target_workspace_header_is_refused(app_client):
    client, _server, _rags, _tmp = app_client
    alice = _login(client, "alice", "alice-pw")
    response = _delete_own(client, alice, headers={"X-Target-Workspace": "admin"})
    assert response.status_code == 400


def test_old_sign_in_must_sign_in_again(app_client):
    client, server, _rags, _tmp = app_client
    _login(client, "alice", "alice-pw")
    old = server.auth_handler.create_token(
        username="alice",
        role="user",
        workspace_id="alice",
        metadata={"auth_mode": "enabled"},
        session_started_at=time.time() - 3600,
    )
    response = _delete_own(client, old)
    assert response.status_code == 403
    assert "Sign in again" in response.json()["detail"]


def test_impersonation_token_cannot_delete(app_client):
    client, server, _rags, _tmp = app_client
    _login(client, "alice", "alice-pw")
    # As issued by POST /admin/workspaces/{id}/impersonate, plus an auth_mode
    # to show the impersonation flag alone is enough to refuse it
    impersonation = server.auth_handler.create_token(
        username="alice",
        role="user",
        custom_expire_hours=1,
        workspace_id="alice",
        metadata={
            "auth_mode": "enabled",
            "impersonated_by": "admin",
            "impersonation": True,
        },
    )
    response = _delete_own(client, impersonation)
    assert response.status_code == 403


def test_per_user_api_key_cannot_delete(app_client):
    client, _server, _rags, _tmp = app_client
    alice = _login(client, "alice", "alice-pw")
    created = client.post("/api-keys", json={"name": "ci"}, headers=_bearer(alice))
    assert created.status_code in (200, 201), created.text
    key = created.json()["api_key"]

    response = client.request(
        "DELETE", "/workspace", json=CONFIRM, headers={"X-API-Key": key}
    )
    assert response.status_code in (401, 403)


def test_session_for_unowned_workspace_cannot_delete(app_client):
    client, server, _rags, _tmp = app_client
    _login(client, "alice", "alice-pw")
    forged = server.auth_handler.create_token(
        username="alice",
        role="user",
        workspace_id="admin",
        metadata={"auth_mode": "enabled"},
    )
    response = _delete_own(client, forged)
    assert response.status_code == 403


def test_admin_delete_purges_data(app_client):
    client, _server, rags, tmp_path = app_client
    alice = _login(client, "alice", "alice-pw")
    client.get("/documents/pipeline_status", headers=_bearer(alice))
    admin = _login(client, "admin", "admin-pw")

    response = client.delete("/admin/workspaces/alice", headers=_bearer(admin))

    assert response.status_code == 200, response.text
    rags["alice"].doc_status.drop.assert_awaited_once()
    owners = client.get("/admin/workspace-owners", headers=_bearer(admin)).json()
    assert "alice" not in {o["workspace_id"] for o in owners}
    assert (
        client.delete(
            "/admin/workspaces/nobody_here", headers=_bearer(admin)
        ).status_code
        == 404
    )
    assert client.delete(
        "/admin/workspaces/alice", headers=_bearer(alice)
    ).status_code in (401, 403)


def test_tombstone_file_holds_no_workspace_names(app_client):
    client, _server, _rags, tmp_path = app_client
    alice = _login(client, "alice", "alice-pw")
    assert _delete_own(client, alice).status_code == 200
    tombstones = json.loads(
        (Path(tmp_path) / "rag_storage" / ".workspace_tombstones.json").read_text()
    )
    assert len(tombstones) == 1
    assert "alice" not in json.dumps(tombstones)
