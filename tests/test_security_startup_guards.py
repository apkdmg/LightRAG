"""Tests for the startup guards and secure defaults in the API layer.

Covers:

- ``resolve_cors_settings``: a wildcard origin never allows credentials.
- ``validate_tenant_isolation_configuration``: backend ``*_WORKSPACE`` overrides
  are rejected when multi-tenancy is on.
- ``AuthHandler.login_required``: usable OAuth2 counts as configured auth, so an
  SSO-only deployment is not treated as "authentication disabled".
- Whitelist matching on path-segment boundaries and the ``/health``-only default.
- ``combined_dependency``: a guest token does not bypass a configured API key,
  and SSO-only deployments reject anonymous requests.
- ``_resolve_user``: no guest fallback when a shared API key is configured.

All tests run offline (no network, no DB).
"""

import argparse
import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

pytestmark = pytest.mark.offline

API_KEY = "shared-api-key"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["*", "", None, " * ", "https://a.example,*"])
def test_cors_wildcard_never_allows_credentials(value):
    from lightrag.api.config import resolve_cors_settings

    assert resolve_cors_settings(value) == (["*"], False)


def test_cors_explicit_origins_allow_credentials():
    from lightrag.api.config import resolve_cors_settings

    origins, allow_credentials = resolve_cors_settings(
        "https://rag.example.org, http://localhost:3000"
    )
    assert origins == ["https://rag.example.org", "http://localhost:3000"]
    assert allow_credentials is True


# ---------------------------------------------------------------------------
# Storage workspace overrides vs multi-tenancy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_var", ["POSTGRES_WORKSPACE", "MILVUS_WORKSPACE", "NEO4J_WORKSPACE"]
)
def test_workspace_override_rejected_with_multi_tenancy(env_var):
    from lightrag.api.config import validate_tenant_isolation_configuration

    args = argparse.Namespace(enable_multi_tenancy=True)
    with pytest.raises(ValueError, match=env_var):
        validate_tenant_isolation_configuration(args, {env_var: "shared"})


def test_workspace_override_allowed_without_multi_tenancy():
    from lightrag.api.config import validate_tenant_isolation_configuration

    args = argparse.Namespace(enable_multi_tenancy=False)
    validate_tenant_isolation_configuration(args, {"POSTGRES_WORKSPACE": "shared"})


def test_blank_workspace_override_is_ignored():
    from lightrag.api.config import validate_tenant_isolation_configuration

    args = argparse.Namespace(enable_multi_tenancy=True)
    validate_tenant_isolation_configuration(
        args, {"POSTGRES_WORKSPACE": "  ", "WORKSPACE": "space1"}
    )


def test_every_backend_override_read_by_storage_is_guarded():
    """Any ``*_WORKSPACE`` env var a storage backend reads must be in the guard."""
    import re
    from pathlib import Path

    from lightrag.api.config import STORAGE_WORKSPACE_OVERRIDE_ENV_VARS

    kg_dir = Path(__file__).resolve().parents[1] / "lightrag" / "kg"
    found = set()
    for path in kg_dir.glob("*.py"):
        found.update(re.findall(r'"([A-Z0-9]+_WORKSPACE)"', path.read_text()))
    assert found, "expected storage backends to read *_WORKSPACE overrides"
    assert found <= set(STORAGE_WORKSPACE_OVERRIDE_ENV_VARS)


# ---------------------------------------------------------------------------
# Whitelist matching
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    base = dict(
        token_secret="test-jwt-secret",
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="",
        admin_accounts="",
        oauth2_enabled=False,
        oauth2_client_id="",
        oauth2_client_secret="",
        whitelist_paths="/health",
        token_auto_renew=False,
        token_renew_threshold=0.5,
        api_key_role="admin",
        key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def api_modules(monkeypatch):
    """Reload auth / utils_api / dependencies against a synthetic ``global_args``.

    Returns a loader so each test picks its own configuration.
    """
    import lightrag.api.config as config

    reloaded = ("lightrag.api.auth", "lightrag.api.utils_api")

    def load(**overrides):
        args = _make_args(**overrides)
        monkeypatch.setattr(config, "global_args", args)
        for name in reloaded:
            sys.modules.pop(name, None)
        auth = importlib.import_module("lightrag.api.auth")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        dependencies = importlib.import_module("lightrag.api.dependencies")
        return SimpleNamespace(
            args=args, auth=auth, utils_api=utils_api, dependencies=dependencies
        )

    yield load

    for name in reloaded:
        sys.modules.pop(name, None)


def test_whitelist_prefix_matches_on_segment_boundary(api_modules):
    utils_api = api_modules().utils_api
    patterns = utils_api.parse_whitelist_patterns("/health, /api/*")

    assert utils_api.is_whitelisted_path("/health", patterns)
    assert utils_api.is_whitelisted_path("/api", patterns)
    assert utils_api.is_whitelisted_path("/api/chat", patterns)
    assert not utils_api.is_whitelisted_path("/api-keys", patterns)
    assert not utils_api.is_whitelisted_path("/apikeys/x", patterns)
    assert not utils_api.is_whitelisted_path("/healthz", patterns)
    assert not utils_api.is_whitelisted_path("/documents", patterns)


def test_default_whitelist_is_health_only(monkeypatch):
    from lightrag.api import config

    monkeypatch.delenv("WHITELIST_PATHS", raising=False)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("ENABLE_MULTI_TENANCY", "false")
    args = config.parse_args()
    assert args.whitelist_paths == "/health"


# ---------------------------------------------------------------------------
# login_required
# ---------------------------------------------------------------------------


def test_login_required_false_without_any_auth(api_modules):
    assert api_modules().auth.auth_handler.login_required is False


def test_login_required_with_accounts(api_modules):
    mods = api_modules(auth_accounts="admin:pw")
    assert mods.auth.auth_handler.login_required is True


def test_login_required_with_usable_oauth2_only(api_modules):
    mods = api_modules(
        oauth2_enabled=True, oauth2_client_id="cid", oauth2_client_secret="secret"
    )
    assert mods.auth.auth_handler.accounts == {}
    assert mods.auth.auth_handler.login_required is True


def test_login_not_required_when_oauth2_enabled_without_credentials(api_modules):
    # OAUTH2_ENABLED defaults to true; without client credentials nobody can
    # log in through it, so it must not flip the server into "auth required".
    mods = api_modules(oauth2_enabled=True)
    assert mods.auth.auth_handler.login_required is False


# ---------------------------------------------------------------------------
# combined_dependency
# ---------------------------------------------------------------------------


def _client(mods, api_key=None):
    app = FastAPI()
    combined_auth = mods.utils_api.get_combined_auth_dependency(api_key)

    @app.get("/documents", dependencies=[Depends(combined_auth)])
    async def documents():
        return {"ok": True}

    @app.get("/api/chat", dependencies=[Depends(combined_auth)])
    async def ollama_chat():
        return {"ok": True}

    @app.get("/health", dependencies=[Depends(combined_auth)])
    async def health():
        return {"ok": True}

    return TestClient(app)


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_no_protection_configured_allows_anonymous(api_modules):
    client = _client(api_modules())
    assert client.get("/documents").status_code == 200


def test_guest_token_does_not_bypass_api_key(api_modules):
    mods = api_modules(key=API_KEY)
    client = _client(mods, api_key=API_KEY)
    guest = mods.auth.auth_handler.create_token(username="guest", role="guest")

    assert client.get("/documents", headers=_bearer(guest)).status_code == 403
    assert client.get("/documents").status_code == 403

    with_key = {**_bearer(guest), "X-API-Key": API_KEY}
    assert client.get("/documents", headers=with_key).status_code == 200
    assert client.get("/documents", headers={"X-API-Key": API_KEY}).status_code == 200


def test_sso_only_deployment_rejects_anonymous_and_guest(api_modules):
    mods = api_modules(
        oauth2_enabled=True, oauth2_client_id="cid", oauth2_client_secret="secret"
    )
    client = _client(mods)
    handler = mods.auth.auth_handler

    assert client.get("/documents").status_code == 401

    guest = handler.create_token(username="guest", role="guest")
    assert client.get("/documents", headers=_bearer(guest)).status_code == 401

    user = handler.create_token(username="alice@example.com", role="user")
    assert client.get("/documents", headers=_bearer(user)).status_code == 200
    client.cookies.set("lightrag_token", user)
    assert client.get("/documents").status_code == 200


def test_ollama_routes_require_auth_by_default(api_modules):
    mods = api_modules(auth_accounts="admin:pw")
    client = _client(mods)

    assert client.get("/health").status_code == 200
    assert client.get("/api/chat").status_code == 401

    user = mods.auth.auth_handler.create_token(username="admin", role="user")
    assert client.get("/api/chat", headers=_bearer(user)).status_code == 200


# ---------------------------------------------------------------------------
# _resolve_user guest fallback
# ---------------------------------------------------------------------------


def test_resolve_user_guest_when_nothing_configured(api_modules):
    mods = api_modules()
    user = asyncio.run(mods.dependencies._resolve_user(None))
    assert (user.role, user.workspace_id) == ("guest", "guest")


def test_resolve_user_no_guest_fallback_when_api_key_configured(api_modules):
    mods = api_modules(key=API_KEY)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(mods.dependencies._resolve_user(None))
    assert excinfo.value.status_code == 401


def test_resolve_user_requires_token_for_sso_only(api_modules):
    mods = api_modules(
        oauth2_enabled=True, oauth2_client_id="cid", oauth2_client_secret="secret"
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(mods.dependencies._resolve_user(None))
    assert excinfo.value.status_code == 401


# ---------------------------------------------------------------------------
# Full app: CORS headers and removed debug endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def build_app(monkeypatch):
    """Build the real FastAPI app with a mocked LightRAG and a chosen CORS_ORIGINS."""
    from unittest.mock import MagicMock, patch

    for var in ("LLM_BINDING_HOST", "EMBEDDING_BINDING_HOST", "LIGHTRAG_API_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    def build(cors_origins):
        from lightrag.api import lightrag_server
        from lightrag.api.config import parse_args

        args = parse_args()
        monkeypatch.setattr(lightrag_server.global_args, "cors_origins", cors_origins)
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            return TestClient(lightrag_server.create_app(args))

    return build


def test_wildcard_cors_does_not_echo_origin_with_credentials(build_app):
    client = build_app("*")
    client.cookies.set("lightrag_token", "session-cookie")
    response = client.get("/auth-status", headers={"Origin": "https://evil.example"})

    assert response.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in response.headers

    preflight = client.options(
        "/documents",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-credentials" not in preflight.headers


def test_explicit_cors_origin_gets_credentials_and_others_do_not(build_app):
    client = build_app("https://rag.example.org")

    allowed = client.get("/auth-status", headers={"Origin": "https://rag.example.org"})
    assert allowed.headers.get("access-control-allow-origin") == (
        "https://rag.example.org"
    )
    assert allowed.headers.get("access-control-allow-credentials") == "true"

    denied = client.get("/auth-status", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_debug_auth_endpoint_is_removed(build_app):
    client = build_app("*")
    assert client.get("/debug/auth").status_code == 404
