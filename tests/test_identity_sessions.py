"""Tests for identity and session hardening.

Covers:

- Keycloak access tokens must be issued for this server (aud / azp check).
- SSO identities based on email require ``email_verified``.
- Tokens record the session start; auto-renewal keeps the workspace and
  session start, never extends impersonation tokens, and stops after
  TOKEN_MAX_SESSION_HOURS.
- Password-login tokens stop working when the account is removed.
- Password login throttling (unit and through the real /login endpoint).

All tests run offline.
"""

import importlib
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.offline


def _args(**overrides):
    base = dict(
        token_secret="test-jwt-secret",
        jwt_algorithm="HS256",
        token_expire_hours=48,
        guest_token_expire_hours=24,
        auth_accounts="alice:pw-alice,bob:pw-bob",
        admin_accounts="",
        oauth2_enabled=False,
        oauth2_client_id="lightrag-server",
        oauth2_client_secret="",
        oauth2_allowed_clients="",
        oauth2_allowed_audiences="",
        oauth2_require_verified_email=True,
        oauth2_service_account_admin_clients="",
        whitelist_paths="/health",
        token_auto_renew=True,
        token_renew_threshold=0.5,
        token_max_session_hours=168,
        api_key_role="admin",
        key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def mods(monkeypatch, tmp_path):
    """Reload auth / utils_api against a synthetic ``global_args``."""
    import lightrag.api.config as config
    import lightrag.api.obo_allowlist as obo

    # Isolate the OBO allowlist: empty file, no admin env knobs.
    allowlist = tmp_path / ".obo_allowlist"
    allowlist.write_text("")
    monkeypatch.setenv("OBO_ALLOWLIST_PATH", str(allowlist))
    monkeypatch.delenv("OBO_ADMIN_CLIENTS", raising=False)
    monkeypatch.delenv("OAUTH2_SERVICE_ACCOUNT_ADMIN_CLIENTS", raising=False)
    obo._manager = None

    reloaded = ("lightrag.api.auth", "lightrag.api.utils_api")

    def load(**overrides):
        args = _args(**overrides)
        monkeypatch.setattr(config, "global_args", args)
        for name in reloaded:
            sys.modules.pop(name, None)
        auth = importlib.import_module("lightrag.api.auth")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        utils_api._token_renewal_cache.clear()
        return SimpleNamespace(args=args, auth=auth, utils_api=utils_api, obo=obo)

    yield load

    for name in reloaded:
        sys.modules.pop(name, None)
    obo._manager = None


# ---------------------------------------------------------------------------
# Keycloak access token audience
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"azp": "lightrag-server"},
        {"aud": "lightrag-server", "azp": "other-app"},
        {"aud": ["account", "lightrag-server"], "azp": "other-app"},
    ],
)
def test_token_for_lightrag_is_accepted(mods, payload):
    assert mods().auth.is_token_issued_for_lightrag(payload) is True


@pytest.mark.parametrize(
    "payload",
    [
        {"azp": "other-app", "aud": "account"},
        {"clientId": "other-app"},
        {},
    ],
)
def test_token_for_another_client_is_rejected(mods, payload):
    assert mods().auth.is_token_issued_for_lightrag(payload) is False


def test_extra_clients_and_audiences_can_be_trusted(mods):
    m = mods(oauth2_allowed_clients="partner-app", oauth2_allowed_audiences="rag-api")
    assert m.auth.is_token_issued_for_lightrag({"azp": "partner-app"}) is True
    assert m.auth.is_token_issued_for_lightrag({"aud": "rag-api", "azp": "x"}) is True


def test_obo_allowlisted_client_is_trusted(mods, tmp_path, monkeypatch):
    m = mods()
    path = tmp_path / ".obo_allowlist_clients"
    path.write_text("OBO_ALLOWED_CLIENTS=[backend-service:tenant_a]\n")
    monkeypatch.setenv("OBO_ALLOWLIST_PATH", str(path))
    m.obo._manager = None
    assert m.auth.is_token_issued_for_lightrag({"azp": "backend-service"}) is True


class _FakeKeycloak:
    def __init__(self, payload, is_service=False):
        self.payload = payload
        self.is_service = is_service

    def validate_access_token(self, token):
        return self.payload

    def is_service_account_token(self, payload):
        return self.is_service


def _patch_keycloak(monkeypatch, payload, is_service=False):
    import lightrag.api.oauth2 as oauth2_mod

    monkeypatch.setattr(
        oauth2_mod, "get_keycloak_client", lambda: _FakeKeycloak(payload, is_service)
    )


def test_validate_any_token_rejects_token_for_other_client(mods, monkeypatch):
    m = mods()
    _patch_keycloak(
        monkeypatch,
        {"azp": "other-app", "email": "a@x.my", "email_verified": True},
    )
    with pytest.raises(HTTPException) as excinfo:
        m.auth.validate_any_token("keycloak-token")
    assert excinfo.value.status_code == 401


def test_validate_any_token_rejects_unverified_email(mods, monkeypatch):
    m = mods()
    _patch_keycloak(
        monkeypatch,
        {"azp": "lightrag-server", "email": "victim@x.my", "email_verified": False},
    )
    with pytest.raises(HTTPException):
        m.auth.validate_any_token("keycloak-token")


def test_validate_any_token_accepts_verified_email(mods, monkeypatch):
    m = mods()
    _patch_keycloak(
        monkeypatch,
        {
            "azp": "lightrag-server",
            "email": "alice@x.my",
            "email_verified": True,
            "preferred_username": "alice",
        },
    )
    info = m.auth.validate_any_token("keycloak-token")
    assert info["workspace_id"] == "alice_x_my"


def test_unverified_email_allowed_when_check_disabled(mods, monkeypatch):
    m = mods(oauth2_require_verified_email=False)
    _patch_keycloak(
        monkeypatch,
        {"azp": "lightrag-server", "email": "alice@x.my", "email_verified": False},
    )
    assert m.auth.validate_any_token("keycloak-token")["workspace_id"] == "alice_x_my"


# ---------------------------------------------------------------------------
# Token session start and removed accounts
# ---------------------------------------------------------------------------


def test_token_records_session_start(mods):
    handler = mods().auth.auth_handler
    before = int(time.time())
    info = handler.validate_token(handler.create_token("alice"))
    assert before <= info["session_started_at"] <= int(time.time())

    started = before - 3600
    info = handler.validate_token(
        handler.create_token("alice", session_started_at=started)
    )
    assert info["session_started_at"] == started


def test_password_token_rejected_after_account_removed(mods):
    m = mods()
    token = m.auth.auth_handler.create_token("bob", metadata={"auth_mode": "enabled"})
    assert m.auth.auth_handler.validate_token(token)["username"] == "bob"

    m2 = mods(auth_accounts="alice:pw-alice")  # bob removed
    with pytest.raises(HTTPException) as excinfo:
        m2.auth.auth_handler.validate_token(token)
    assert excinfo.value.status_code == 401


def test_sso_token_not_affected_by_auth_accounts(mods):
    m = mods(auth_accounts="alice:pw-alice")
    token = m.auth.auth_handler.create_token(
        "carol@x.my", metadata={"auth_mode": "sso"}
    )
    assert m.auth.auth_handler.validate_token(token)["username"] == "carol@x.my"


def test_unknown_user_password_check_is_false(mods):
    m = mods(
        auth_accounts="alice:{bcrypt}$2b$04$abcdefghijklmnopqrstuuJ6tdyYdyQ4gq7l0Q2o5mJ6cXk5r3c7e"
    )
    assert m.auth.auth_handler.verify_password("nobody", "x") is False


# ---------------------------------------------------------------------------
# Auto-renewal
# ---------------------------------------------------------------------------


def _token_info(
    m,
    *,
    hours_left,
    session_age_hours=1,
    workspace_id="team_ws",
    metadata=None,
    role="user",
):
    now = datetime.now(timezone.utc)
    return {
        "username": "alice",
        "role": role,
        "workspace_id": workspace_id,
        "metadata": metadata if metadata is not None else {"auth_mode": "sso"},
        "exp": now + timedelta(hours=hours_left),
        "session_started_at": int(now.timestamp() - session_age_hours * 3600),
    }


def test_renewal_not_due(mods):
    m = mods()
    assert m.utils_api._renew_token_if_due(_token_info(m, hours_left=40)) is None


def test_renewal_keeps_workspace_and_session_start(mods):
    m = mods()
    info = _token_info(m, hours_left=2, session_age_hours=10)
    new_token = m.utils_api._renew_token_if_due(info)
    assert new_token
    renewed = m.auth.auth_handler.validate_token(new_token)
    assert renewed["workspace_id"] == "team_ws"
    assert renewed["session_started_at"] == info["session_started_at"]
    assert renewed["exp"] > info["exp"]


def test_impersonation_token_is_never_renewed(mods):
    m = mods()
    info = _token_info(
        m,
        hours_left=0.5,
        metadata={"impersonation": True, "impersonated_by": "admin"},
    )
    assert m.utils_api._renew_token_if_due(info) is None


def test_renewal_stops_after_max_session_length(mods):
    m = mods(token_max_session_hours=24)
    assert (
        m.utils_api._renew_token_if_due(
            _token_info(m, hours_left=2, session_age_hours=25)
        )
        is None
    )
    assert m.utils_api._renew_token_if_due(
        _token_info(m, hours_left=2, session_age_hours=23)
    )


def test_session_cap_disabled_with_zero(mods):
    m = mods(token_max_session_hours=0)
    assert m.utils_api._renew_token_if_due(
        _token_info(m, hours_left=2, session_age_hours=10_000)
    )


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_login_throttle_locks_after_max_failures():
    from lightrag.api.login_throttle import LoginThrottle

    clock = _Clock()
    throttle = LoginThrottle(max_failures=3, window_seconds=60, clock=clock)
    for _ in range(3):
        assert throttle.retry_after("Alice") == 0
        throttle.record_failure("Alice")
    assert throttle.retry_after("alice") == 60  # case-insensitive key
    assert throttle.retry_after("bob") == 0

    clock.now += 30
    assert throttle.retry_after("alice") == 30
    clock.now += 31
    assert throttle.retry_after("alice") == 0


def test_login_throttle_reset_on_success():
    from lightrag.api.login_throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=2, window_seconds=60, clock=_Clock())
    throttle.record_failure("alice")
    throttle.reset("alice")
    throttle.record_failure("alice")
    assert throttle.retry_after("alice") == 0


def test_login_throttle_disabled_with_zero():
    from lightrag.api.login_throttle import LoginThrottle

    throttle = LoginThrottle(max_failures=0, window_seconds=60)
    for _ in range(10):
        throttle.record_failure("alice")
    assert throttle.retry_after("alice") == 0


def test_login_endpoint_throttles_failed_attempts(monkeypatch):
    """Through the real /login route: lockout, Retry-After, and no bypass with
    the right password while locked out."""
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
    args.login_max_failed_attempts = 3
    args.login_lockout_minutes = 1
    monkeypatch.setattr(
        lightrag_server.auth_handler, "accounts", {"alice": "right-password"}
    )
    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        client = TestClient(lightrag_server.create_app(args))

    def login(password):
        return client.post("/login", data={"username": "alice", "password": password})

    assert [login("wrong").status_code for _ in range(3)] == [401, 401, 401]
    locked = login("right-password")
    assert locked.status_code == 429
    assert 0 < int(locked.headers["Retry-After"]) <= 60
