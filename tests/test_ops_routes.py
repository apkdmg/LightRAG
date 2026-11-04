"""Operations console router: admin gate, error shape, no-store, /status."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

# The API config parses sys.argv on first use; keep pytest's flags out of it.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
ops_routes = importlib.import_module("lightrag.api.routers.ops_routes")
sys.argv = _original_argv

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lightrag.api.dependencies import UserInfo, require_admin  # noqa: E402
from lightrag.api.ops.context import RequestIdMiddleware  # noqa: E402

pytestmark = pytest.mark.offline


def _admin(username: str = "root") -> UserInfo:
    return UserInfo(username=username, role="admin", workspace_id="w", metadata={})


def _deny():
    raise HTTPException(status_code=403, detail="Admin access required")


def _app(
    *,
    admin: bool = True,
    kv_storage: str = "PGKVStorage",
    default_rag: bool = False,
    insights: bool = False,
    middleware: bool = True,
) -> FastAPI:
    app = FastAPI()
    if middleware:
        app.add_middleware(RequestIdMiddleware, trusted_proxies="")
    app.include_router(ops_routes.router)
    rag = SimpleNamespace(kv_storage=kv_storage)
    ops_routes.configure_ops(app, rag=rag, workers=None, version="0295")
    if default_rag:
        app.state.default_rag = SimpleNamespace(kv_storage="PGKVStorage")
    if insights:
        app.state.insights = object()
    app.dependency_overrides[require_admin] = _admin if admin else _deny
    return app


def _route_path(route) -> str:
    return (
        route.path.replace("{ws}", "ws")
        .replace("{run_id}", "r")
        .replace("{doc_id}", "d")
        .replace("{dep}", "postgres")
        .replace("{worker_id}", "w")
        .replace("{id}", "1")
        .replace("{log_id}", "1")
        .replace("{alert_id}", "1")
        .replace("{rule}", "x")
    )


def test_router_wide_admin_dependency_is_present():
    assert any(
        dep.dependency is require_admin for dep in ops_routes.router.dependencies
    )


def test_every_route_needs_admin():
    denied = TestClient(_app(admin=False), raise_server_exceptions=False)
    allowed = TestClient(_app(admin=True), raise_server_exceptions=False)
    assert ops_routes.router.routes, "router has no routes"
    for route in ops_routes.router.routes:
        for method in sorted(route.methods):
            path = _route_path(route)
            response = denied.request(method, path, json={})
            assert response.status_code == 403, (method, path)
            body = response.json()
            assert body["detail"] == "Admin access required"
            assert body["request_id"] == response.headers["x-request-id"]
            assert response.headers["cache-control"] == "no-store"
            if method == "GET":
                ok = allowed.request(method, path)
                assert ok.status_code == 200, (method, path, ok.text)
                assert ok.headers["cache-control"] == "no-store"
                assert "generated_at" in ok.json()


def test_every_post_needs_live_admin():
    """Mutations must re-check the admin list, not only the token's role."""
    for route in ops_routes.router.routes:
        if "POST" not in route.methods and "PUT" not in route.methods:
            continue
        names = {
            getattr(dep.call, "__name__", "") for dep in route.dependant.dependencies
        }
        assert "require_live_admin" in names, route.path


def test_status_payload_shape(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    monkeypatch.delenv("OPS_OPERATOR_ACCOUNTS", raising=False)
    client = TestClient(_app(insights=True))
    response = client.get("/admin/ops/status")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["available"] is True
    assert "unavailable_reason" not in body
    assert body["features"] == {
        "durable": False,
        "processing": False,
        "runs": False,
        "logs": False,
        "log_sink": False,
        "probes": False,
        "heartbeats": False,
        "alerts": False,
        "alert_email": False,
        "log_capture": False,
        "insights": True,
        "actions": False,
        "release_busy": False,
        "audit": False,
        "gpu": False,
        "metrics": False,
    }
    assert body["process"] == {
        "mode": "uvicorn",
        "expected_workers": 1,
        "workers_live": 1,
        "manager": "local",
    }
    assert body["operator"] is True
    assert body["version"] == "0295"
    assert body["generated_at"].endswith("+00:00")


def test_single_tenant_without_default_rag_still_serves_status(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    # No app.state.default_rag (single-tenant), non-Postgres storage.
    app = _app(kv_storage="JsonKVStorage")
    assert not hasattr(app.state, "default_rag")
    body = TestClient(app).get("/admin/ops/status").json()
    assert body["available"] is False
    assert body["unavailable_reason"] == "postgres_required"
    assert body["features"]["insights"] is False


def test_status_prefers_default_rag(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    app = _app(kv_storage="JsonKVStorage", default_rag=True)
    assert TestClient(app).get("/admin/ops/status").json()["available"] is True


def test_status_without_server_info_still_answers(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    app = FastAPI()
    app.include_router(ops_routes.router)
    app.dependency_overrides[require_admin] = _admin
    body = TestClient(app).get("/admin/ops/status").json()
    assert body["available"] is False
    assert body["version"] == ""


def test_gunicorn_mode_reports_workers(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "gunicorn")
    monkeypatch.setattr(ops_routes, "_count_gunicorn_workers", lambda: 3)
    app = _app()
    app.state.ops_server = ops_routes.OpsServerInfo(
        rag=SimpleNamespace(kv_storage="PGKVStorage"), workers=4, version="v"
    )
    body = TestClient(app).get("/admin/ops/status").json()
    assert body["process"] == {
        "mode": "gunicorn",
        "expected_workers": 4,
        "workers_live": 3,
        "manager": "local",
    }


def test_operator_accounts(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "Ops@Unimas.my, lead@unimas.my")
    app = _app()
    client = TestClient(app)
    assert client.get("/admin/ops/status").json()["operator"] is False
    app.dependency_overrides[require_admin] = lambda: _admin("ops@unimas.my")
    assert client.get("/admin/ops/status").json()["operator"] is True
    app.dependency_overrides[require_admin] = lambda: UserInfo(
        username="sso-sub-123",
        role="admin",
        workspace_id="w",
        metadata={"email": "LEAD@unimas.my"},
    )
    assert client.get("/admin/ops/status").json()["operator"] is True


def test_error_body_has_request_id_and_no_secret(monkeypatch):
    def explode(_request):
        raise RuntimeError("connect failed with key sk-test-SECRET1234567890")

    monkeypatch.setattr(ops_routes, "_data_rag", explode)
    client = TestClient(_app(), raise_server_exceptions=False)
    response = client.get("/admin/ops/status")
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body == {
        "detail": "internal_error",
        "request_id": response.headers["x-request-id"],
    }
    assert "sk-test" not in response.text
    assert "SECRET" not in response.text


def test_error_without_request_id_middleware_still_has_reference(monkeypatch):
    def explode(_request):
        raise RuntimeError("sk-test-SECRET1234567890")

    monkeypatch.setattr(ops_routes, "_data_rag", explode)
    client = TestClient(_app(middleware=False), raise_server_exceptions=False)
    response = client.get("/admin/ops/status")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "internal_error"
    assert body["request_id"] and body["request_id"] == response.headers["x-request-id"]
    assert "SECRET" not in response.text
