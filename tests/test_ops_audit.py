"""Operations console audit (PR-7): the Audit section's reads and CSV export,
and the audit rows the existing admin mutations now write (impersonation,
workspace create/delete, owner bindings, the server-default provider, admin
cross-workspace /documents mutations).

The existing endpoints audit best effort: a missing or failing ops database
never changes their responses.
"""

from __future__ import annotations

import csv
import importlib
import io
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from lightrag.api.ops import audit as ops_audit
from lightrag.api.ops.auth import require_live_admin
from lightrag.api.ops.service import OpsService
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import MemoryOpsStore, workspace_hash
from tests.ops_fixtures import admin, build_app

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
admin_routes = importlib.import_module("lightrag.api.routers.admin_routes")
dependencies = importlib.import_module("lightrag.api.dependencies")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

CLIENT = {"X-LightRAG-Client": "webui"}
UserInfo = dependencies.UserInfo


def _ops_state(app: FastAPI, store: Any) -> None:
    app.state.ops = SimpleNamespace(
        service=OpsService(store=store, reader=NamespaceReader(), app_state=app.state)
    )


class BrokenAuditStore(MemoryOpsStore):
    async def audit_write(self, row):
        raise ConnectionError("database gone")


def _rows(store: MemoryOpsStore, action: str) -> list[dict[str, Any]]:
    return [r for r in store.audit_rows if r["action"] == action]


def _pair(store: MemoryOpsStore, action: str) -> tuple[dict, dict]:
    intent, result = _rows(store, action)
    assert intent["phase"] == "intent" and result["phase"] == "result"
    assert result["params"]["intent_id"] == intent["id"]
    return intent, result


# -- GET /audit -------------------------------------------------------------------------


def _seed_audit(store: MemoryOpsStore, n: int = 7) -> None:
    now = datetime.now(timezone.utc)
    for i in range(n):
        store.audit_rows.append(
            {
                "id": i + 1,
                "ts": now - timedelta(minutes=10 * (n - i)),
                "actor": "khairil" if i % 2 else "root",
                "auth_kind": "sso",
                "impersonated_by": None,
                "action": "cancel" if i % 3 else "reveal",
                "workspace": "fsktm",
                "target": "fsktm",
                "params": {"intent_id": i} if i else {},
                "reason": "ticket",
                "phase": "result" if i else "intent",
                "outcome": "ok",
                "request_id": f"req{i}",
                "client_ip_hash": "0" * 16,
            }
        )


def test_audit_pages_newest_first_with_a_keyset():
    store = MemoryOpsStore()
    _seed_audit(store)
    client = TestClient(build_app(store))
    first = client.get("/admin/ops/audit", params={"limit": 3}).json()
    assert first["available"] is True
    assert [i["id"] for i in first["items"]] == [7, 6, 5]
    assert first["items"][0]["intent_id"] == 6
    second = client.get(
        "/admin/ops/audit", params={"limit": 3, "before": first["next_before"]}
    ).json()
    assert [i["id"] for i in second["items"]] == [4, 3, 2]
    third = client.get(
        "/admin/ops/audit", params={"limit": 3, "before": second["next_before"]}
    ).json()
    assert [i["id"] for i in third["items"]] == [1]
    assert third["next_before"] is None


def test_audit_filters():
    store = MemoryOpsStore()
    _seed_audit(store)
    client = TestClient(build_app(store))
    items = client.get("/admin/ops/audit", params={"actor": "khairil"}).json()["items"]
    assert {i["actor"] for i in items} == {"khairil"} and len(items) == 3
    items = client.get("/admin/ops/audit", params={"action": "reveal"}).json()["items"]
    assert {i["id"] for i in items} == {1, 4, 7}
    items = client.get("/admin/ops/audit", params={"phase": "intent"}).json()["items"]
    assert [i["id"] for i in items] == [1]
    since = (datetime.now(timezone.utc) - timedelta(minutes=25)).isoformat()
    items = client.get("/admin/ops/audit", params={"from": since}).json()["items"]
    assert [i["id"] for i in items] == [7, 6]
    bad = client.get("/admin/ops/audit", params={"phase": "x"})
    assert bad.status_code == 400 and bad.json()["detail"] == "invalid_phase"
    bad = client.get(
        "/admin/ops/audit",
        params={"from": "2020-01-01T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
    )
    assert bad.json()["detail"] == "invalid_range"


def test_audit_without_the_ops_database_is_empty_not_an_error():
    client = TestClient(build_app(None, kv_storage="JsonKVStorage"))
    body = client.get("/admin/ops/audit").json()
    assert body["available"] is False and body["items"] == []


# -- POST /audit/export --------------------------------------------------------------------


def _export_app(store):
    app = build_app(store)
    app.dependency_overrides[require_live_admin] = admin
    return app


def test_audit_export_is_formula_safe_and_itself_audited(monkeypatch):
    monkeypatch.delenv("OPS_OPERATOR_ACCOUNTS", raising=False)
    store = MemoryOpsStore()
    _seed_audit(store, 2)
    store.audit_rows[0]["reason"] = '=HYPERLINK("http://x")'
    store.audit_rows[1]["actor"] = "@evil"
    store.audit_rows[1]["target"] = "  +cmd|' /C calc'!A0"
    client = TestClient(_export_app(store))
    response = client.post(
        "/admin/ops/audit/export",
        json={"filters": {}, "reason": "quarterly review"},
        headers=CLIENT,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["x-ops-export-rows"] == "2"
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows[0][:3] == ["id", "ts", "actor"]
    cells = [cell for row in rows[1:] for cell in row]
    for cell in cells:
        assert not cell.lstrip(" ").startswith(("=", "+", "-", "@")), cell
    assert "'@evil" in cells and '\'=HYPERLINK("http://x")' in cells
    intent, result = _pair(store, "audit_export")
    assert intent["reason"] == "quarterly review"
    assert response.headers["x-ops-audit-id"] == str(intent["id"])
    assert result["outcome"] == "ok" and result["params"]["rows"] == 2


def test_audit_export_is_operator_only_and_needs_its_audit_row(monkeypatch):
    store = MemoryOpsStore()
    client = TestClient(_export_app(store))
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    response = client.post(
        "/admin/ops/audit/export", json={"reason": "review"}, headers=CLIENT
    )
    assert response.status_code == 403
    monkeypatch.delenv("OPS_OPERATOR_ACCOUNTS")
    response = client.post(
        "/admin/ops/audit/export", json={"reason": "review"}, headers=CLIENT
    )
    assert response.status_code == 200
    broken = TestClient(_export_app(BrokenAuditStore()))
    response = broken.post(
        "/admin/ops/audit/export", json={"reason": "review"}, headers=CLIENT
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "audit_unavailable"
    response = client.post("/admin/ops/audit/export", json={"reason": "review"})
    assert response.status_code == 403  # no client header (CSRF)


# -- admin_routes ---------------------------------------------------------------------------

admin_routes.create_admin_routes()


class FakeRegistry:
    def __init__(self):
        self.owners: dict[str, Any] = {}

    async def set_owner(self, workspace, identity, email):
        self.owners[workspace] = SimpleNamespace(
            workspace_id=workspace,
            identity=identity,
            email=email,
            created_at=1.0,
            last_login_at=2.0,
        )

    async def get_by_workspace(self, workspace):
        return self.owners.get(workspace)

    async def release(self, workspace):
        return self.owners.pop(workspace, None) is not None


class FakeWorkspaceManager:
    def get_workspace_info(self, workspace_id, persisted=None):
        if workspace_id != "fsktm":
            return None
        return SimpleNamespace(owner_username="a.bin@unimas.my")

    async def create_workspace(self, workspace_id, owner_username):
        return SimpleNamespace(
            workspace_id=workspace_id,
            owner_username=owner_username,
            created_at=1.0,
            last_accessed_at=1.0,
            access_count=0,
            is_active=True,
        )


@pytest.fixture
def admin_app(monkeypatch):
    def make(store):
        app = FastAPI()
        app.include_router(admin_routes.router)
        app.dependency_overrides[dependencies.require_admin] = admin
        app.state.workspace_registry = FakeRegistry()
        app.state.workspace_manager = FakeWorkspaceManager()
        if store is not None:
            _ops_state(app, store)
        return app

    async def exists(app, workspace):
        return True

    async def purge(app, workspace):
        return SimpleNamespace(ok=True, errors=[])

    async def persisted(app):
        return {}

    monkeypatch.setattr(admin_routes, "workspace_exists", exists)
    monkeypatch.setattr(admin_routes, "purge_app_workspace", purge)
    monkeypatch.setattr(admin_routes, "persisted_workspaces", persisted)
    return make


def _owner_calls(client: TestClient) -> list[tuple[int, Any]]:
    out = []
    response = client.put(
        "/admin/workspaces/fsktm/owner",
        json={"identity": "kc:1234-abcd", "email": "a.bin@unimas.my"},
    )
    out.append((response.status_code, response.json()))
    response = client.delete("/admin/workspaces/fsktm/owner")
    out.append((response.status_code, response.json()))
    response = client.delete("/admin/workspaces/fsktm/owner")
    out.append((response.status_code, response.json()))
    response = client.post("/admin/workspaces", json={"username": "new.user"})
    out.append((response.status_code, response.json()))
    response = client.delete("/admin/workspaces/space9")
    out.append((response.status_code, response.json()))
    return out


@pytest.mark.parametrize("variant", ["audited", "no_ops_db", "audit_failing"])
def test_admin_mutations_answer_the_same_with_or_without_audit(admin_app, variant):
    baseline = _owner_calls(TestClient(admin_app(None)))
    store = {
        "audited": MemoryOpsStore(),
        "no_ops_db": None,
        "audit_failing": BrokenAuditStore(),
    }[variant]
    assert _owner_calls(TestClient(admin_app(store))) == baseline
    assert [code for code, _ in baseline] == [200, 200, 404, 200, 200]


def test_admin_mutations_write_intent_and_result_rows(admin_app):
    store = MemoryOpsStore()
    _owner_calls(TestClient(admin_app(store)))
    intent, result = _pair(store, "owner_set")
    assert intent["actor"] == "root" and intent["workspace"] == "fsktm"
    # The identity kind only: never the subject or the email.
    assert intent["params"] == {"identity_kind": "kc"}
    assert "1234-abcd" not in str(store.audit_rows)
    assert "a.bin@unimas.my" not in str(store.audit_rows)
    assert result["outcome"] == "ok"
    rows = _rows(store, "owner_release")
    assert [r["outcome"] for r in rows if r["phase"] == "result"] == ["ok", "http_404"]
    _, created = _pair(store, "workspace_create")
    assert created["workspace"] == admin_routes.sanitize_workspace_id("new.user")
    assert created["outcome"] == "ok"
    intent, result = _pair(store, "workspace_delete")
    assert intent["workspace"] == "space9"
    # Written after the purge: the deleted workspace appears hashed (FR-78).
    assert result["workspace"] == workspace_hash("space9")
    assert result["target"] == workspace_hash("space9")


def test_impersonation_is_audited(admin_app):
    store = MemoryOpsStore()
    client = TestClient(admin_app(store))
    response = client.post("/admin/workspaces/fsktm/impersonate")
    assert response.status_code == 200 and response.json()["access_token"]
    intent, result = _pair(store, "impersonate")
    assert intent["target"] == "fsktm" and result["outcome"] == "ok"
    assert intent["params"] == {"expires_in_hours": 1}
    assert response.json()["access_token"] not in str(store.audit_rows)
    response = client.post("/admin/workspaces/nobody/impersonate")
    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found: nobody"
    assert _rows(store, "impersonate")[-1]["outcome"] == "http_404"


# -- admin cross-workspace /documents mutations ------------------------------------------------


def _documents_app(store, user: UserInfo, monkeypatch) -> FastAPI:
    router = APIRouter(prefix="/documents", route_class=ops_audit.AuditedDocumentsRoute)

    @router.post("/text")
    async def insert(request: Request):
        workspace = await dependencies.resolve_workspace_from_request(request)
        # resolve_rag and resolve_doc_manager both resolve: audited once.
        await dependencies.resolve_workspace_from_request(request)
        return {"status": "success", "workspace": workspace}

    @router.delete("/delete_document")
    async def delete(request: Request):
        await dependencies.resolve_workspace_from_request(request)
        raise HTTPException(status_code=409, detail="busy")

    @router.post("/paginated")
    async def paginated(request: Request):
        await dependencies.resolve_workspace_from_request(request)
        return {"documents": []}

    @router.get("/status_counts")
    async def counts(request: Request):
        await dependencies.resolve_workspace_from_request(request)
        return {"counts": {}}

    app = FastAPI()
    app.include_router(router)
    if store is not None:
        _ops_state(app, store)

    async def resolve_user(request):
        return user

    monkeypatch.setattr(dependencies, "resolve_user_from_request", resolve_user)
    return app


ADMIN = UserInfo(username="root", role="admin", workspace_id="root", metadata={})
TARGET = {"X-Target-Workspace": "fsktm"}


def test_on_behalf_document_writes_are_audited(monkeypatch):
    store = MemoryOpsStore()
    client = TestClient(_documents_app(store, ADMIN, monkeypatch))
    response = client.post("/documents/text", headers=TARGET)
    assert response.json() == {"status": "success", "workspace": "fsktm"}
    intent, result = _pair(store, "documents_mutation")
    assert intent["workspace"] == "fsktm" and intent["actor"] == "root"
    assert intent["params"] == {
        "method": "POST",
        "route": "/documents/text",
        "mode": "on_behalf",
    }
    assert result["outcome"] == "ok"
    response = client.delete("/documents/delete_document", headers=TARGET)
    assert response.status_code == 409
    assert _rows(store, "documents_mutation")[-1]["outcome"] == "http_409"


def test_impersonated_document_writes_are_audited(monkeypatch):
    store = MemoryOpsStore()
    user = UserInfo(
        username="a.bin",
        role="user",
        workspace_id="fsktm",
        metadata={"impersonated_by": "root", "impersonation": True},
    )
    client = TestClient(_documents_app(store, user, monkeypatch))
    client.post("/documents/text")
    intent, _ = _pair(store, "documents_mutation")
    assert intent["actor"] == "a.bin" and intent["impersonated_by"] == "root"
    assert intent["params"]["mode"] == "impersonated"


@pytest.mark.parametrize(
    "user,method,path,headers",
    [
        (ADMIN, "POST", "/documents/text", {}),  # own workspace
        (ADMIN, "POST", "/documents/paginated", TARGET),  # a read
        (ADMIN, "GET", "/documents/status_counts", TARGET),
        (
            UserInfo("api", "admin", "api", {"auth_mode": "api_key"}),
            "POST",
            "/documents/text",
            TARGET,
        ),
        (
            UserInfo("svc", "admin", "svc", {"auth_mode": "client_credentials"}),
            "POST",
            "/documents/text",
            TARGET,
        ),
    ],
    ids=["own", "read-post", "get", "api-key", "client-credentials"],
)
def test_other_document_requests_are_not_audited(
    monkeypatch, user, method, path, headers
):
    from lightrag.api import obo_allowlist

    monkeypatch.setattr(obo_allowlist, "check_obo_allowed", lambda *a: True)
    store = MemoryOpsStore()
    client = TestClient(_documents_app(store, user, monkeypatch))
    response = client.request(method, path, headers=headers)
    assert response.status_code == 200, response.text
    assert store.audit_rows == []


def test_document_writes_answer_the_same_when_audit_fails(monkeypatch):
    client = TestClient(_documents_app(BrokenAuditStore(), ADMIN, monkeypatch))
    response = client.post("/documents/text", headers=TARGET)
    assert response.status_code == 200
    assert response.json() == {"status": "success", "workspace": "fsktm"}
    assert client.delete("/documents/delete_document", headers=TARGET).status_code == (
        409
    )


def test_documents_router_uses_the_audited_route_class():
    from lightrag.api.routers.document_routes import create_document_routes

    router = create_document_routes(
        SimpleNamespace(workspace=""), SimpleNamespace(), None
    )
    assert router.route_class is ops_audit.AuditedDocumentsRoute
    assert all(
        isinstance(route, ops_audit.AuditedDocumentsRoute) for route in router.routes
    )


def test_on_behalf_email_ingest_writes_intent_and_result(monkeypatch):
    """POST /documents/email lives on its own router under /documents: it
    must get the result row too, not a permanently pending intent."""
    email_routes = importlib.import_module("lightrag.api.routers.email_routes")
    assert email_routes.router.route_class is ops_audit.AuditedDocumentsRoute

    async def fake_ingest(self, parsed, **_kwargs):
        return {
            "documents_created": 1,
            "attachments_processed": 0,
            "inline_images_processed": 0,
        }

    class Manager:
        async def get_lightrag_instance(self, workspace):
            return SimpleNamespace(workspace=workspace)

    async def resolve_user(request):
        return ADMIN

    monkeypatch.setattr(email_routes.EmailIngestionService, "ingest_email", fake_ingest)
    monkeypatch.setattr(dependencies, "resolve_user_from_request", resolve_user)
    store = MemoryOpsStore()
    app = FastAPI()
    app.include_router(email_routes.create_email_routes(SimpleNamespace()))
    app.state.workspace_manager = Manager()
    _ops_state(app, store)
    response = TestClient(app).post(
        "/documents/email",
        headers=TARGET,
        data={
            "metadata": '{"from": "pat@example.test", "to": ["a@example.test"], '
            '"subject": "Timetable"}',
            "body_text": "See attached.",
        },
    )
    assert response.status_code == 200, response.text
    intent, result = _pair(store, "documents_mutation")
    assert intent["workspace"] == "fsktm"
    assert intent["params"]["route"] == "/documents/email"
    assert result["outcome"] == "ok"


def test_prefixed_document_writes_are_still_audited(monkeypatch):
    """Behind a proxy that keeps LIGHTRAG_API_PREFIX (FastAPI root_path), the
    request path carries the prefix; the matched route decides."""
    store = MemoryOpsStore()
    app = _documents_app(store, ADMIN, monkeypatch)
    app.root_path = "/site01"
    client = TestClient(app, root_path="/site01")
    response = client.post("/site01/documents/text", headers=TARGET)
    assert response.status_code == 200, response.text
    intent, result = _pair(store, "documents_mutation")
    assert intent["params"]["route"] == "/documents/text"
    assert result["outcome"] == "ok"
    # A read in disguise stays unaudited under the prefix too.
    client.post("/site01/documents/paginated", headers=TARGET)
    assert len(_rows(store, "documents_mutation")) == 2


def test_admin_audit_flag_off_writes_nothing(monkeypatch):
    monkeypatch.setenv("OPS_ADMIN_AUDIT", "false")
    store = MemoryOpsStore()
    client = TestClient(_documents_app(store, ADMIN, monkeypatch))
    response = client.post("/documents/text", headers=TARGET)
    assert response.json() == {"status": "success", "workspace": "fsktm"}
    assert store.audit_rows == []


def test_csv_cells_never_start_a_formula():
    from lightrag.api.ops.logs import csv_cell

    for value in ("=1+1", "+1", "-1", "@SUM(A1)", "\t=1", "  =1", " @x"):
        assert csv_cell(value).startswith("'"), value
    assert csv_cell("plain") == "plain" and csv_cell(None) == ""


def test_timeline_markers_come_from_audited_actions_only():
    """State-changing audit rows are timeline markers; searches, exports and
    reveals stay in the Audit section."""
    store = MemoryOpsStore()
    _seed_audit(store, 0)
    now = datetime.now(timezone.utc)
    for i, action in enumerate(
        ("cancel", "log_search", "reveal", "workspace_delete", "audit_export")
    ):
        for phase in ("intent", "result"):
            store.audit_rows.append(
                {
                    "id": len(store.audit_rows) + 1,
                    "ts": now - timedelta(minutes=5 + i),
                    "actor": "root",
                    "action": action,
                    "workspace": "fsktm",
                    "phase": phase,
                    "outcome": "ok",
                    "params": {},
                }
            )
    client = TestClient(build_app(store))
    items = client.get("/admin/ops/events", params={"kinds": "ops_action"}).json()[
        "items"
    ]
    assert [i["detail"]["action"] for i in items] == ["cancel", "workspace_delete"]
