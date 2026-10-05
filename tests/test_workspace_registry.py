"""Tests for the workspace ownership registry.

Covers ID derivation, binding on first sign-in, collisions, reassigned emails,
concurrent claims, admin transfer/release, and the /login and /admin wiring.
The JSON backend is used offline; the Postgres backend shares the same
``resolve_workspace`` logic and is exercised by the live test run.
"""

import asyncio
import sys

import pytest

from lightrag.api.workspace_manager import sanitize_workspace_id
from lightrag.api.workspace_registry import (
    JsonWorkspaceRegistry,
    WorkspaceOwnershipError,
    local_identity,
    resolve_workspace,
    sso_identity,
    storage_safe_workspace_id,
)

pytestmark = pytest.mark.offline


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ID derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    ["fadam@unimas.my", "rkhairilzamrie@unimas.my", "Admin", "john.doe@unimas.my"],
)
def test_storage_safe_id_matches_legacy_for_existing_style_names(source):
    assert storage_safe_workspace_id(source) == sanitize_workspace_id(source)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("nur-aina@unimas.my", "nur_aina_unimas_my"),
        ("12345@siswa.unimas.my", "u_12345_siswa_unimas_my"),
        ("", "default"),
        ("@@@", "default"),
    ],
)
def test_storage_safe_id_is_valid_collection_name(source, expected):
    workspace_id = storage_safe_workspace_id(source)
    assert workspace_id == expected
    assert workspace_id[0].isalpha()
    assert set(workspace_id) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")


# ---------------------------------------------------------------------------
# resolve_workspace
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    return JsonWorkspaceRegistry(tmp_path)


def test_first_sign_in_claims_derived_id(registry):
    ws = run(resolve_workspace(registry, sso_identity("sub-a"), "a.b@x.my", "a.b@x.my"))
    assert ws == "a_b_x_my"
    owner = run(registry.get_by_workspace("a_b_x_my"))
    assert owner.identity == "kc:sub-a"
    assert owner.email == "a.b@x.my"


def test_same_identity_gets_same_workspace_even_after_email_change(registry):
    first = run(resolve_workspace(registry, sso_identity("sub-a"), "a@x.my", "a@x.my"))
    renamed = run(
        resolve_workspace(registry, sso_identity("sub-a"), "a.new@x.my", "a.new@x.my")
    )
    assert renamed == first == "a_x_my"


def test_collision_gets_distinct_workspace(registry):
    first = run(
        resolve_workspace(
            registry, sso_identity("sub-a"), "john.doe@x.my", "john.doe@x.my"
        )
    )
    second = run(
        resolve_workspace(
            registry, sso_identity("sub-b"), "john_doe@x.my", "john_doe@x.my"
        )
    )
    assert first == "john_doe_x_my"
    assert second.startswith("john_doe_x_my_") and second != first
    # Stable on the next sign-in
    again = run(
        resolve_workspace(
            registry, sso_identity("sub-b"), "john_doe@x.my", "john_doe@x.my"
        )
    )
    assert again == second


def test_reassigned_email_is_refused(registry):
    run(
        resolve_workspace(registry, sso_identity("old-sub"), "staff@x.my", "staff@x.my")
    )
    with pytest.raises(WorkspaceOwnershipError) as excinfo:
        run(
            resolve_workspace(
                registry, sso_identity("new-sub"), "Staff@x.my", "Staff@x.my"
            )
        )
    assert excinfo.value.workspace_id == "staff_x_my"
    assert excinfo.value.identity == "kc:new-sub"
    # The original owner is untouched
    assert run(registry.get_by_workspace("staff_x_my")).identity == "kc:old-sub"


def test_local_accounts_use_username(registry):
    ws = run(resolve_workspace(registry, local_identity("admin"), None, "admin"))
    assert ws == "admin"
    assert run(registry.get_by_identity("local:admin")).workspace_id == "admin"


def test_concurrent_claims_give_one_owner(registry):
    async def many():
        return await asyncio.gather(
            *[
                resolve_workspace(
                    registry, sso_identity(f"sub-{i}"), f"user{i}.x@y.my", "same@y.my"
                )
                for i in range(8)
            ]
        )

    results = run(many())
    assert results.count("same_y_my") == 1
    assert len(set(results)) == 8


# ---------------------------------------------------------------------------
# Admin operations
# ---------------------------------------------------------------------------


def test_set_owner_resolves_a_refused_sign_in(registry):
    run(
        resolve_workspace(registry, sso_identity("old-sub"), "staff@x.my", "staff@x.my")
    )

    # Same person, re-created account: hand them the existing workspace.
    run(registry.set_owner("staff_x_my", sso_identity("new-sub"), "staff@x.my"))
    assert (
        run(
            resolve_workspace(
                registry, sso_identity("new-sub"), "staff@x.my", "staff@x.my"
            )
        )
        == "staff_x_my"
    )
    assert run(registry.get_by_identity("kc:old-sub")) is None


def test_set_owner_can_bind_a_fresh_workspace(registry):
    run(
        resolve_workspace(registry, sso_identity("old-sub"), "staff@x.my", "staff@x.my")
    )

    # Different person given the old address: give them a new workspace.
    run(registry.set_owner("staff_x_my_new", sso_identity("new-sub"), "staff@x.my"))
    assert (
        run(
            resolve_workspace(
                registry, sso_identity("new-sub"), "staff@x.my", "staff@x.my"
            )
        )
        == "staff_x_my_new"
    )
    assert run(registry.get_by_workspace("staff_x_my")).identity == "kc:old-sub"


def test_release_lets_next_sign_in_claim(registry):
    run(
        resolve_workspace(registry, sso_identity("old-sub"), "staff@x.my", "staff@x.my")
    )
    assert run(registry.release("staff_x_my")) is True
    assert run(registry.release("staff_x_my")) is False
    assert (
        run(
            resolve_workspace(
                registry, sso_identity("new-sub"), "staff@x.my", "staff@x.my"
            )
        )
        == "staff_x_my"
    )


def test_registry_survives_new_instance(tmp_path):
    run(
        resolve_workspace(
            JsonWorkspaceRegistry(tmp_path), local_identity("alice"), None, "alice"
        )
    )
    owners = run(JsonWorkspaceRegistry(tmp_path).list_owners())
    assert [(o.workspace_id, o.identity) for o in owners] == [("alice", "local:alice")]


def test_cached_lookup(registry):
    run(resolve_workspace(registry, local_identity("alice"), None, "alice"))
    assert run(registry.cached_workspace_for("local:alice")) == "alice"
    assert run(registry.cached_workspace_for("local:nobody")) is None


# ---------------------------------------------------------------------------
# Server wiring: /login and admin endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    from unittest.mock import MagicMock, patch

    from fastapi.testclient import TestClient

    for var in ("LLM_BINDING_HOST", "EMBEDDING_BINDING_HOST", "LIGHTRAG_API_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api import lightrag_server
    from lightrag.api.config import parse_args

    args = parse_args()
    args.working_dir = str(tmp_path)
    args.enable_multi_tenancy = True
    args.kv_storage = "JsonKVStorage"
    monkeypatch.setattr(
        lightrag_server.auth_handler,
        "accounts",
        {"admin": "admin-pw", "alice": "alice-pw"},
    )
    # Patch the admin check the server holds directly: other test modules
    # reload lightrag.api.auth, which can leave the server bound to a copy
    # reading a different global_args.
    monkeypatch.setattr(
        lightrag_server, "_is_admin_user", lambda *names: "admin" in names
    )
    # Request-time auth (dependencies._resolve_user) imports the current
    # lightrag.api.auth module; make it validate with the server's handler.
    import lightrag.api.auth as current_auth

    monkeypatch.setattr(current_auth, "auth_handler", lightrag_server.auth_handler)
    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        yield TestClient(lightrag_server.create_app(args)), lightrag_server


def _login(client, username, password):
    response = client.post("/login", data={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def test_login_binds_workspace_and_admin_can_manage_it(app_client):
    client, server = app_client
    alice = _login(client, "alice", "alice-pw")
    assert server.auth_handler.validate_token(alice)["workspace_id"] == "alice"

    admin = _login(client, "admin", "admin-pw")
    headers = {"Authorization": f"Bearer {admin}"}

    response = client.get("/admin/workspace-owners", headers=headers)
    assert response.status_code == 200, response.text
    owners = response.json()
    assert {o["workspace_id"]: o["identity"] for o in owners} == {
        "alice": "local:alice",
        "admin": "local:admin",
    }

    # Non-admins cannot use the owner endpoints
    denied = client.get(
        "/admin/workspace-owners", headers={"Authorization": f"Bearer {alice}"}
    )
    assert denied.status_code == 403

    moved = client.put(
        "/admin/workspaces/alice_archive/owner",
        json={"identity": "local:alice"},
        headers=headers,
    )
    assert moved.status_code == 200
    assert (
        _login(client, "alice", "alice-pw")
        and server.auth_handler.validate_token(_login(client, "alice", "alice-pw"))[
            "workspace_id"
        ]
        == "alice_archive"
    )

    bad = client.put(
        "/admin/workspaces/x/owner", json={"identity": "nobody"}, headers=headers
    )
    assert bad.status_code == 422

    released = client.delete("/admin/workspaces/alice_archive/owner", headers=headers)
    assert released.status_code == 200
    assert (
        client.delete(
            "/admin/workspaces/alice_archive/owner", headers=headers
        ).status_code
        == 404
    )


def test_failed_sso_callback_ends_existing_session(app_client, monkeypatch):
    """A refused or failed SSO sign-in must also clear any earlier session
    cookie, so the browser cannot carry on under a previous login."""
    import lightrag.api.oauth2 as oauth2_mod

    client, _server = app_client
    monkeypatch.setattr(oauth2_mod, "get_keycloak_client", lambda: None)
    client.cookies.set("lightrag_token", "earlier-session")

    response = client.get(
        "/oauth2/callback", params={"code": "c", "state": "s"}, follow_redirects=False
    )

    assert response.status_code in (302, 307)
    assert "error=auth_failed" in response.headers["location"]
    cleared = [h for h in response.headers.get_list("set-cookie")]
    assert any(h.startswith("lightrag_token=") and "Max-Age=0" in h for h in cleared)
    assert any(h.startswith("lightrag_user=") and "Max-Age=0" in h for h in cleared)
