"""Focused tests for the OAuth2 / auth hardening changes.

Covers three security fixes:

FIX #1 -- ``lightrag.api.auth.validate_any_token``: service-account /
client-credentials tokens are granted ``role="admin"`` ONLY when their
``clientId``/``azp`` is in the comma-separated allowlist
``global_args.oauth2_service_account_admin_clients``
(env ``OAUTH2_SERVICE_ACCOUNT_ADMIN_CLIENTS``); otherwise ``role="user"``.

FIX #2 -- ``lightrag.api.config.validate_auth_configuration`` (and the mirrored
check in ``AuthHandler.__init__``): a strong ``TOKEN_SECRET`` is required
(``ValueError``) whenever ``AUTH_ACCOUNTS`` is set OR OAuth2 is "usable"
(enabled AND client_id AND client_secret). The no-auth case only warns.

FIX #3 -- ``lightrag.api.oauth2.KeycloakClient`` PKCE state tokens:
``create_state_token`` / ``verify_state_token`` (HS256), the 3-tuple return of
``get_authorization_url``, and ``exchange_code`` honouring an explicitly passed
``code_verifier`` (skipping the in-memory store).

These tests run fully offline (no network, no DB). ``validate_any_token`` and
the KeycloakClient validation helpers are synchronous in the production code, so
no event loop is needed for FIX #1/#2; only ``exchange_code`` is async.
"""

import argparse
import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.offline


def _run(coro):
    """Run a coroutine on a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# FIX #1 -- validate_any_token role resolution for service accounts
# ---------------------------------------------------------------------------
#
# Importing ``lightrag.api.auth`` constructs the module-level
# ``auth_handler = AuthHandler()``, which reads ``config.global_args``. The real
# ``global_args`` is a lazy proxy that calls ``parse_args()`` (and thus argparse
# against pytest's own argv) on first access, so we must seed a synthetic
# ``global_args`` on the config module and reload auth before touching it —
# mirroring the pattern already used in ``tests/test_auth.py``.


class _FakeKeycloakClient:
    """Minimal stub mirroring the (synchronous) KeycloakClient surface used by
    ``validate_any_token``."""

    def __init__(self, payload, is_service):
        self._payload = payload
        self._is_service = is_service

    def is_service_account_token(self, payload):
        return self._is_service

    def validate_access_token(self, token):
        return self._payload


@pytest.fixture
def auth_env(monkeypatch):
    """Provide a freshly-reloaded ``lightrag.api.auth`` bound to a controlled,
    mutable ``global_args``. Returns the synthetic namespace so each test can
    tweak the allowlist / admin accounts in place."""
    import lightrag.api.config as config

    mock_global_args = SimpleNamespace(
        token_secret="test-jwt-secret",
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="",
        admin_accounts="",
        oauth2_service_account_admin_clients="",
    )
    monkeypatch.setattr(config, "global_args", mock_global_args)

    sys.modules.pop("lightrag.api.auth", None)
    auth = importlib.import_module("lightrag.api.auth")
    auth = importlib.reload(auth)
    # auth.py read ``global_args`` via ``from .config import global_args`` so its
    # module global is now bound to our mock; keep them in lock-step.
    monkeypatch.setattr(auth, "global_args", mock_global_args)

    yield SimpleNamespace(auth=auth, args=mock_global_args)

    sys.modules.pop("lightrag.api.auth", None)


def _patch_keycloak(monkeypatch, payload, is_service):
    """Patch the lazily-imported ``get_keycloak_client`` (resolved from the
    ``lightrag.api.oauth2`` module inside ``validate_any_token``)."""
    import lightrag.api.oauth2 as oauth2_mod

    monkeypatch.setattr(
        oauth2_mod,
        "get_keycloak_client",
        lambda: _FakeKeycloakClient(payload, is_service),
    )


def test_service_account_not_in_allowlist_is_user(auth_env, monkeypatch):
    """A service-account token whose client_id is NOT in the allowlist => user."""
    payload = {"clientId": "some-svc", "azp": "some-svc", "scope": "openid"}
    _patch_keycloak(monkeypatch, payload, is_service=True)
    # Empty allowlist => no admin.
    auth_env.args.oauth2_service_account_admin_clients = ""

    info = auth_env.auth.validate_any_token("not-a-local-jwt")

    assert info["role"] == "user"
    assert info["username"] == "service-account-some-svc"
    assert info["metadata"]["auth_mode"] == "client_credentials"


def test_service_account_in_allowlist_is_admin(auth_env, monkeypatch):
    """Same token, but with its client_id added to the allowlist => admin."""
    payload = {"clientId": "some-svc", "azp": "some-svc", "scope": "openid"}
    _patch_keycloak(monkeypatch, payload, is_service=True)
    auth_env.args.oauth2_service_account_admin_clients = "other-svc, some-svc , third-svc"

    info = auth_env.auth.validate_any_token("not-a-local-jwt")

    assert info["role"] == "admin"
    assert info["username"] == "service-account-some-svc"


def test_service_account_azp_only_in_allowlist_is_admin(auth_env, monkeypatch):
    """When only ``azp`` is present (no clientId), it is used for the allowlist."""
    payload = {"azp": "azp-svc", "scope": "openid"}
    _patch_keycloak(monkeypatch, payload, is_service=True)
    auth_env.args.oauth2_service_account_admin_clients = "azp-svc"

    info = auth_env.auth.validate_any_token("not-a-local-jwt")

    assert info["role"] == "admin"
    assert info["username"] == "service-account-azp-svc"


def test_regular_user_token_resolves_via_admin_accounts(auth_env, monkeypatch):
    """Sanity: a non-service-account user token still resolves role via
    ADMIN_ACCOUNTS (existing behaviour preserved)."""
    payload = {
        "preferred_username": "alice",
        "email": "alice@example.com",
        "sub": "u1",
    }
    _patch_keycloak(monkeypatch, payload, is_service=False)
    auth_env.args.oauth2_service_account_admin_clients = ""

    # alice IS an admin account (auth.py reads global_args.admin_accounts).
    auth_env.args.admin_accounts = "alice,bob"
    info_admin = auth_env.auth.validate_any_token("not-a-local-jwt")
    assert info_admin["username"] == "alice"
    assert info_admin["role"] == "admin"

    # alice is NOT an admin account.
    auth_env.args.admin_accounts = "carol"
    info_user = auth_env.auth.validate_any_token("not-a-local-jwt")
    assert info_user["role"] == "user"


# ---------------------------------------------------------------------------
# FIX #2 -- validate_auth_configuration TOKEN_SECRET enforcement
# ---------------------------------------------------------------------------


def _auth_args(**overrides):
    base = dict(
        token_secret="",
        auth_accounts="",
        oauth2_enabled=False,
        oauth2_client_id="",
        oauth2_client_secret="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _default_secret():
    from lightrag.api.config import DEFAULT_TOKEN_SECRET

    return DEFAULT_TOKEN_SECRET


def test_oauth2_usable_with_empty_secret_raises():
    from lightrag.api import config

    args = _auth_args(
        oauth2_enabled=True,
        oauth2_client_id="cid",
        oauth2_client_secret="csecret",
        token_secret="",
    )
    with pytest.raises(ValueError) as exc:
        config.validate_auth_configuration(args)
    assert "OAuth2" in str(exc.value)


def test_oauth2_usable_with_default_secret_raises():
    from lightrag.api import config

    args = _auth_args(
        oauth2_enabled=True,
        oauth2_client_id="cid",
        oauth2_client_secret="csecret",
        token_secret=_default_secret(),
    )
    with pytest.raises(ValueError) as exc:
        config.validate_auth_configuration(args)
    assert "OAuth2" in str(exc.value)


def test_oauth2_usable_with_strong_secret_ok():
    from lightrag.api import config

    args = _auth_args(
        oauth2_enabled=True,
        oauth2_client_id="cid",
        oauth2_client_secret="csecret",
        token_secret="a-strong-unique-secret-value-123",
    )
    # Must not raise.
    config.validate_auth_configuration(args)


def test_auth_accounts_with_default_secret_raises():
    from lightrag.api import config

    args = _auth_args(
        auth_accounts="admin:password",
        token_secret=_default_secret(),
    )
    with pytest.raises(ValueError) as exc:
        config.validate_auth_configuration(args)
    assert "AUTH_ACCOUNTS" in str(exc.value)


def test_auth_accounts_with_empty_secret_raises():
    from lightrag.api import config

    args = _auth_args(auth_accounts="admin:password", token_secret="")
    with pytest.raises(ValueError):
        config.validate_auth_configuration(args)


def test_no_auth_empty_secret_ok():
    from lightrag.api import config

    args = _auth_args()  # neither auth_accounts nor oauth2 usable, empty secret
    # Must only warn, not raise.
    config.validate_auth_configuration(args)


def test_oauth2_enabled_but_not_usable_empty_secret_ok():
    from lightrag.api import config

    # enabled but missing client_id/secret => not "usable" => no enforcement.
    args = _auth_args(
        oauth2_enabled=True,
        oauth2_client_id="",
        oauth2_client_secret="",
        token_secret="",
    )
    config.validate_auth_configuration(args)


# ---------------------------------------------------------------------------
# FIX #3 -- KeycloakClient PKCE state tokens & exchange_code verifier selection
# ---------------------------------------------------------------------------


def _make_client():
    from lightrag.api.oauth2 import KeycloakClient, OAuth2Config

    cfg = OAuth2Config(
        enabled=True,
        client_id="my-client",
        client_secret="my-secret",
        authorization_endpoint="https://kc.example.com/auth",
        token_endpoint="https://kc.example.com/token",
        userinfo_endpoint="https://kc.example.com/userinfo",
        jwks_uri="https://kc.example.com/certs",
        issuer="https://kc.example.com/realms/myrealm",
        redirect_uri="https://app.example.com/callback",
    )
    return KeycloakClient(cfg)


def test_state_token_roundtrip_returns_verifier():
    client = _make_client()
    secret = "state-signing-secret"
    token = client.create_state_token("state-abc", "verifier-xyz", secret)
    assert isinstance(token, str)

    cv = client.verify_state_token(token, "state-abc", secret)
    assert cv == "verifier-xyz"


def test_state_token_wrong_expected_state_returns_none():
    client = _make_client()
    secret = "state-signing-secret"
    token = client.create_state_token("state-abc", "verifier-xyz", secret)

    assert client.verify_state_token(token, "DIFFERENT-state", secret) is None


def test_state_token_wrong_secret_returns_none():
    client = _make_client()
    token = client.create_state_token("state-abc", "verifier-xyz", "secret-A")

    assert client.verify_state_token(token, "state-abc", "secret-B") is None


def test_state_token_garbage_returns_none_no_exception():
    client = _make_client()
    secret = "state-signing-secret"
    # Tampered / non-JWT inputs must not raise.
    assert client.verify_state_token("not-a-jwt", "state-abc", secret) is None
    assert client.verify_state_token("aaa.bbb.ccc", "state-abc", secret) is None
    assert client.verify_state_token("", "state-abc", secret) is None


def test_state_token_expired_returns_none():
    client = _make_client()
    secret = "state-signing-secret"
    # Negative expiry => already expired the moment it is created.
    token = client.create_state_token(
        "state-abc", "verifier-xyz", secret, expires_in=-1
    )
    assert client.verify_state_token(token, "state-abc", secret) is None


def test_state_token_wrong_typ_returns_none():
    """A validly-signed JWT with the wrong ``typ`` must be rejected."""
    import jwt
    from datetime import datetime, timedelta

    client = _make_client()
    secret = "state-signing-secret"
    now = datetime.utcnow()
    bogus = jwt.encode(
        {
            "state": "state-abc",
            "cv": "verifier-xyz",
            "typ": "not-pkce",
            "iat": now,
            "exp": now + timedelta(seconds=600),
        },
        secret,
        algorithm="HS256",
    )
    assert client.verify_state_token(bogus, "state-abc", secret) is None


def test_get_authorization_url_returns_3_tuple_and_roundtrips():
    client = _make_client()

    result = client.get_authorization_url()
    assert isinstance(result, tuple)
    assert len(result) == 3
    url, state, code_verifier = result
    assert url.startswith("https://kc.example.com/auth?")
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert state and code_verifier

    # The cookie round-trips through the signed state token.
    secret = "cookie-secret"
    token = client.create_state_token(state, code_verifier, secret)
    assert client.verify_state_token(token, state, secret) == code_verifier


def test_exchange_code_uses_provided_verifier_without_store(monkeypatch):
    """When a code_verifier is passed, exchange_code must NOT consult the
    in-memory store, and must forward that verifier to the token endpoint."""
    import lightrag.api.oauth2 as oauth2_mod

    client = _make_client()
    # Guarantee the store is empty so any fallback would fail loudly.
    client._state_store.clear()

    captured = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"access_token": "AT", "id_token": "IDT"}

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, endpoint, data=None, **kwargs):
            captured["endpoint"] = endpoint
            captured["data"] = data
            return _FakeResp()

    monkeypatch.setattr(oauth2_mod.httpx, "AsyncClient", _FakeAsyncClient)

    out = _run(
        client.exchange_code("the-code", "the-state", code_verifier="explicit-cv")
    )

    assert out == {"access_token": "AT", "id_token": "IDT"}
    assert captured["endpoint"] == "https://kc.example.com/token"
    assert captured["data"]["code_verifier"] == "explicit-cv"
    assert captured["data"]["code"] == "the-code"
    assert captured["data"]["client_id"] == "my-client"


def test_exchange_code_falls_back_to_store_when_no_verifier(monkeypatch):
    """When code_verifier is None, exchange_code must consume the store entry."""
    import lightrag.api.oauth2 as oauth2_mod

    client = _make_client()
    # Seed the in-memory store via the real store_auth_state API.
    client.store_auth_state("the-state", "stored-cv")

    captured = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"access_token": "AT"}

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, endpoint, data=None, **kwargs):
            captured["data"] = data
            return _FakeResp()

    monkeypatch.setattr(oauth2_mod.httpx, "AsyncClient", _FakeAsyncClient)

    out = _run(client.exchange_code("the-code", "the-state"))

    assert out == {"access_token": "AT"}
    assert captured["data"]["code_verifier"] == "stored-cv"
    # Store entry must have been consumed (get_auth_state pops it).
    assert "the-state" not in client._state_store


def test_exchange_code_no_verifier_empty_store_raises_400():
    """No code_verifier + empty store => HTTP 400 (invalid/expired state)."""
    from fastapi import HTTPException

    client = _make_client()
    client._state_store.clear()

    with pytest.raises(HTTPException) as exc:
        _run(client.exchange_code("the-code", "missing-state"))
    assert exc.value.status_code == 400
