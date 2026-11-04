"""Admin "act on workspace" in Documents (PR-9, OQ-13): the server side.

The WebUI sends ``X-Target-Workspace`` on every Documents call while an
admin acts on another workspace. The server must honour it only for admins
(a user, even one who forges the header, gets 403 and nothing happens),
resolve reads and writes to the target, and audit every mutation with its
intent and result rows; without the header everyone stays in their own
workspace.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from lightrag.api import dependencies
from lightrag.api.dependencies import UserInfo
from lightrag.api.ops.store import MemoryOpsStore
from tests.test_ops_audit import ADMIN, TARGET, _documents_app, _pair, _rows

pytestmark = pytest.mark.offline

USER = UserInfo(username="a.bin", role="user", workspace_id="abin", metadata={})


def _live_admins(monkeypatch, accounts: str) -> None:
    """ADMIN_ACCOUNTS as the server reads it now."""
    from lightrag.api import auth as core_auth

    monkeypatch.setattr(
        core_auth.global_args, "admin_accounts", accounts, raising=False
    )


def test_admin_reads_and_writes_land_in_the_target_workspace(monkeypatch):
    store = MemoryOpsStore()
    client = TestClient(_documents_app(store, ADMIN, monkeypatch))
    write = client.post("/documents/text", headers=TARGET)
    assert write.json()["workspace"] == "fsktm"
    intent, result = _pair(store, "documents_mutation")
    assert intent["phase"] == "intent" and result["phase"] == "result"
    assert intent["params"]["mode"] == "on_behalf" and result["outcome"] == "ok"
    # A read with the header resolves to the target and is not audited.
    assert client.get("/documents/status_counts", headers=TARGET).status_code == 200
    assert len(_rows(store, "documents_mutation")) == 2


def test_a_non_admin_can_never_act_on_another_workspace(monkeypatch):
    store = MemoryOpsStore()
    client = TestClient(
        _documents_app(store, USER, monkeypatch), raise_server_exceptions=False
    )
    for method, path in (
        ("POST", "/documents/text"),
        ("GET", "/documents/status_counts"),
    ):
        response = client.request(method, path, headers=TARGET)
        assert response.status_code == 403, (method, path)
        assert "admins" in response.json()["detail"]
    assert store.audit_rows == []
    # Without the header the user works in their own workspace.
    own = client.post("/documents/text")
    assert own.json()["workspace"] == "abin"
    assert store.audit_rows == []


def test_exit_drops_the_header_and_returns_to_the_own_workspace(monkeypatch):
    store = MemoryOpsStore()
    client = TestClient(_documents_app(store, ADMIN, monkeypatch))
    assert client.post("/documents/text", headers=TARGET).json()["workspace"] == (
        "fsktm"
    )
    assert client.post("/documents/text").json()["workspace"] == "root"
    # Only the on-behalf write was audited.
    assert len(_rows(store, "documents_mutation")) == 2


def test_dependency_injection_path_refuses_non_admins(monkeypatch):
    app = FastAPI()

    @app.get("/documents/which")
    async def which(workspace: str = Depends(dependencies.get_current_workspace)):
        return {"workspace": workspace}

    async def as_user():
        return USER

    async def as_admin():
        return ADMIN

    app.dependency_overrides[dependencies.get_current_user] = as_user
    _live_admins(monkeypatch, "root")
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/documents/which", headers=TARGET).status_code == 403
    assert client.get("/documents/which").json() == {"workspace": "abin"}
    app.dependency_overrides[dependencies.get_current_user] = as_admin
    assert client.get("/documents/which", headers=TARGET).json() == {
        "workspace": "fsktm"
    }
    # The header value is sanitised before use.
    response = client.get("/documents/which", headers={"X-Target-Workspace": "FS KTM"})
    assert response.status_code == 200
    assert response.json()["workspace"] != "FS KTM"


def test_a_removed_admin_with_a_stale_claim_is_refused(monkeypatch):
    # The token still says role=admin (renewal copies the claim forward), but
    # root is no longer in ADMIN_ACCOUNTS: no on-behalf read or write, and
    # nothing reaches the target workspace or the audit trail.
    store = MemoryOpsStore()
    client = TestClient(
        _documents_app(store, ADMIN, monkeypatch), raise_server_exceptions=False
    )
    _live_admins(monkeypatch, "someone.else")
    for method, path in (
        ("POST", "/documents/text"),
        ("DELETE", "/documents/delete_document"),
        ("GET", "/documents/status_counts"),
    ):
        response = client.request(method, path, headers=TARGET)
        assert response.status_code == 403, (method, path)
        assert response.json()["detail"] == "admin_not_current"
    assert _rows(store, "documents_mutation") == []
    # Their own workspace still works (the role claim governs that as before).
    assert client.post("/documents/text").json()["workspace"] == "root"
    # Restored to ADMIN_ACCOUNTS: on-behalf works again.
    _live_admins(monkeypatch, "root")
    assert client.post("/documents/text", headers=TARGET).json()["workspace"] == (
        "fsktm"
    )


def test_dependency_injection_path_refuses_stale_and_impersonated(monkeypatch):
    app = FastAPI()

    @app.get("/documents/which")
    async def which(workspace: str = Depends(dependencies.get_current_workspace)):
        return {"workspace": workspace}

    current = {"user": ADMIN}

    async def as_current():
        return current["user"]

    app.dependency_overrides[dependencies.get_current_user] = as_current
    client = TestClient(app, raise_server_exceptions=False)
    _live_admins(monkeypatch, "")
    response = client.get("/documents/which", headers=TARGET)
    assert response.status_code == 403
    assert response.json()["detail"] == "admin_not_current"
    _live_admins(monkeypatch, "root")
    current["user"] = UserInfo(
        username="root",
        role="admin",
        workspace_id="root",
        metadata={"impersonated_by": "other", "impersonation": True},
    )
    response = client.get("/documents/which", headers=TARGET)
    assert response.status_code == 403
    assert response.json()["detail"] == "impersonation_not_allowed"


def test_service_accounts_keep_the_obo_allowlist_path(monkeypatch):
    # An API-key caller is not in ADMIN_ACCOUNTS; its on-behalf access is
    # governed by the OBO allowlist, not the interactive live-admin check.
    from lightrag.api import obo_allowlist

    app = FastAPI()

    @app.get("/documents/which")
    async def which(workspace: str = Depends(dependencies.get_current_workspace)):
        return {"workspace": workspace}

    async def as_key():
        return UserInfo(
            username="api_key_user",
            role="admin",
            workspace_id="",
            metadata={"auth_mode": "api_key"},
        )

    app.dependency_overrides[dependencies.get_current_user] = as_key
    _live_admins(monkeypatch, "")
    monkeypatch.setattr(obo_allowlist, "check_obo_allowed", lambda c, w: True)
    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/documents/which", headers=TARGET).json() == {
        "workspace": "fsktm"
    }
