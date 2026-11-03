"""``/health`` after D6: liveness for anonymous callers, detail for any
authenticated caller, the old ``keyed_locks`` shape from a read-only lock
status, URL userinfo stripped, and the ``LIGHTRAG-WORKSPACE`` header ignored
without credentials."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from tests.test_bedrock_llm import (
    _FakeLightRAG,
    _FakeOllamaAPI,
    _make_args,
    _reload_api_modules_if_mocked,
)

pytestmark = pytest.mark.offline

LIVENESS_KEYS = {
    "status",
    "version",
    "auth_mode",
    "core_version",
    "api_version",
    "webui_title",
    "webui_description",
}
LOCKS = {
    "process_id": 4242,
    "total_mp_locks": 2,
    "pending_mp_cleanup": 1,
    "total_async_locks": 3,
    "pending_async_cleanup": 0,
}
SECRET_HOST = (
    "https://alice:hunter2-secret@llm.example.com/v1?api_key=sk-test-abcdef123456"
)


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    """Build the real server app with fake storages; ``auth`` turns logins on."""
    _reload_api_modules_if_mocked()
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    monkeypatch.delenv("HEALTH_DETAIL", raising=False)

    def build(
        *, auth: bool = False, api_key: str | None = None, detail: str | None = None
    ):
        if detail is not None:
            monkeypatch.setenv("HEALTH_DETAIL", detail)
        args = _make_args(tmp_path)
        args.whitelist_paths = "/health"
        args.key = api_key
        args.llm_binding_host = SECRET_HOST
        config = importlib.import_module("lightrag.api.config")
        config.initialize_config(args, force=True)
        server = importlib.import_module("lightrag.api.lightrag_server")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        monkeypatch.setattr(server, "auth_configured", auth)
        monkeypatch.setattr(utils_api, "auth_configured", auth)
        monkeypatch.setattr(server, "LightRAG", _FakeLightRAG)
        monkeypatch.setattr(server, "check_frontend_build", lambda: (True, False))
        for name in (
            "create_document_routes",
            "create_query_routes",
            "create_graph_routes",
        ):
            monkeypatch.setattr(server, name, lambda *_a, **_k: APIRouter())
        monkeypatch.setattr(server, "OllamaAPI", _FakeOllamaAPI)
        namespace = AsyncMock(return_value={"busy": True})
        monkeypatch.setattr(server, "get_namespace_data", namespace)
        monkeypatch.setattr(server, "get_default_workspace", lambda: "default")
        monkeypatch.setattr(server, "get_keyed_lock_status", lambda: dict(LOCKS))
        app = server.create_app(args)
        return app, namespace, server

    return build


def _token(role: str = "user", **metadata) -> str:
    from lightrag.api.auth import auth_handler

    return auth_handler.create_token("alice", role=role, metadata=metadata or None)


def test_open_server_gives_everyone_the_detail(make_app):
    # No logins and no API key: the whole API is open, so /health is too.
    app, namespace, _ = make_app()
    body = TestClient(app).get("/health").json()
    assert body["pipeline_busy"] is True
    assert body["configuration"]["kv_storage"] == "JsonKVStorage"
    namespace.assert_awaited_once()


def test_keyed_locks_keep_the_old_shape_without_cleanup(make_app):
    app, _, server = make_app()
    body = TestClient(app).get("/health").json()
    assert body["keyed_locks"] == {
        "process_id": 4242,
        "cleanup_performed": {"mp_cleaned": 0, "async_cleaned": 0},
        "current_status": {
            "total_mp_locks": 2,
            "pending_mp_cleanup": 1,
            "total_async_locks": 3,
            "pending_async_cleanup": 0,
        },
    }
    # The mutating cleanup is not reachable from /health any more.
    assert not hasattr(server, "cleanup_keyed_lock")


def test_userinfo_and_query_are_stripped_from_hosts(make_app):
    app, _, _ = make_app()
    response = TestClient(app).get("/health")
    assert response.json()["configuration"]["llm_binding_host"] == (
        "https://llm.example.com/v1"
    )
    assert "hunter2" not in response.text
    assert "sk-test-abcdef" not in response.text
    assert "alice:" not in response.text


def test_anonymous_caller_gets_liveness_only(make_app):
    app, namespace, _ = make_app(auth=True)
    response = TestClient(app).get(
        "/health", headers={"LIGHTRAG-WORKSPACE": "other_workspace"}
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == LIVENESS_KEYS
    assert body["status"] == "healthy" and body["auth_mode"] == "enabled"
    # Liveness reads no shared state, so the workspace header is never used.
    namespace.assert_not_awaited()
    assert "llm.example.com" not in response.text


def test_invalid_token_is_anonymous_not_an_error(make_app):
    app, _, _ = make_app(auth=True)
    response = TestClient(app).get(
        "/health", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert response.status_code == 200
    assert set(response.json()) == LIVENESS_KEYS


def test_any_authenticated_user_gets_the_detail(make_app):
    app, namespace, _ = make_app(auth=True)
    response = TestClient(app).get(
        "/health",
        headers={
            "Authorization": f"Bearer {_token()}",
            "LIGHTRAG-WORKSPACE": "other_workspace",
        },
    )
    body = response.json()
    assert body["pipeline_busy"] is True and body["pipeline_active"] is True
    assert body["configuration"]["max_graph_nodes"] == 1000
    assert body["keyed_locks"]["current_status"]["total_mp_locks"] == 2
    for key in ("core_version", "api_version", "webui_title", "auth_mode"):
        assert key in body
    # An authenticated caller may still pick the workspace by header.
    assert namespace.await_args.kwargs["workspace"] == "other_workspace"


def test_sso_cookie_counts_as_authenticated(make_app):
    app, _, _ = make_app(auth=True)
    client = TestClient(app)
    client.cookies.set("lightrag_token", _token())
    assert "configuration" in client.get("/health").json()


def test_guest_token_is_not_enough_when_logins_are_required(make_app):
    app, _, _ = make_app(auth=True)
    response = TestClient(app).get(
        "/health", headers={"Authorization": f"Bearer {_token(role='guest')}"}
    )
    assert set(response.json()) == LIVENESS_KEYS


def test_api_key_holders_get_the_detail(make_app):
    app, _, _ = make_app(api_key="k-123456789")
    client = TestClient(app)
    assert set(client.get("/health").json()) == LIVENESS_KEYS
    assert (
        set(client.get("/health", headers={"X-API-Key": "wrong"}).json())
        == LIVENESS_KEYS
    )
    assert (
        "configuration"
        in client.get("/health", headers={"X-API-Key": "k-123456789"}).json()
    )
    # A guest token never stands in for a configured API key.
    assert (
        set(
            client.get(
                "/health", headers={"Authorization": f"Bearer {_token(role='guest')}"}
            ).json()
        )
        == LIVENESS_KEYS
    )


def test_health_detail_all_restores_the_old_behaviour(make_app):
    app, namespace, _ = make_app(auth=True, detail="all")
    response = TestClient(app).get(
        "/health", headers={"LIGHTRAG-WORKSPACE": "other_workspace"}
    )
    body = response.json()
    assert "configuration" in body and "keyed_locks" in body
    assert namespace.await_args.kwargs["workspace"] == "other_workspace"
    # Userinfo is stripped in every mode.
    assert "hunter2" not in response.text


def test_errors_do_not_echo_exception_text(make_app, monkeypatch):
    app, namespace, _ = make_app()
    namespace.side_effect = RuntimeError("connect to postgres://u:sekret@db failed")
    response = TestClient(app, raise_server_exceptions=False).get("/health")
    assert response.status_code == 500
    assert "sekret" not in response.text


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("http://u:p1@h1:1,http://u:p2@h2:2", "http://h1:1,http://h2:2"),
        ("https://user:p/ss@host:443/v1", "https://host:443/v1"),
        ("https://user:pa?ss@host/v1", "https://host/v1"),
        ("https://user:pa#ss@host/v1", "https://host/v1"),
        (
            "redis://:pw@r:6379/0;bolt://n:pw@neo:7687",
            "redis://r:6379/0;bolt://neo:7687",
        ),
        ("https://bob:pw@router.example/v1?key=sk-1", "https://router.example/v1"),
        ("bolt://neo4j:7687", "bolt://neo4j:7687"),
        ("not a url", "not a url"),
    ],
)
def test_strip_userinfo_handles_every_url_shape(raw, clean):
    from lightrag.api.ops.redact import sanitize_config, strip_userinfo

    assert strip_userinfo(raw) == clean
    assert sanitize_config({"configuration": {"hosts": raw}}) == {
        "configuration": {"hosts": clean}
    }
