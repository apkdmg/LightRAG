"""Offline tests for the server default LLM provider set by admins in the WebUI.

Covers the order of precedence (workspace provider, server default, ``.env``),
reusing a key without sending it to another host, the admin routes, and the
provider test.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from lightrag import LightRAG, RoleLLMConfig
from lightrag.api.provider_alerts import build_system_provider_change_message
from lightrag.api.workspace_manager import sanitize_workspace_id
from lightrag.api.workspace_providers import (
    SYSTEM_RECORD_ID,
    ProviderSlot,
    WorkspaceProviderConfig,
    WorkspaceProviderStore,
    apply_workspace_provider_overrides,
    reconcile_loaded_instances,
    reconcile_system_provider,
    resolve_server_slot_key,
    watch_system_provider,
)
from lightrag.exceptions import PipelineNotInitializedError
from lightrag.usage import add_usage_sink, remove_usage_sink
from lightrag.utils import EmbeddingFunc, Tokenizer

# Importing the routers package triggers import-time arg parsing (auth handler);
# neutralize pytest's argv during the import, mirroring the other API tests.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
routes = importlib.import_module("lightrag.api.routers.system_provider_routes")
_dependencies = importlib.import_module("lightrag.api.dependencies")
sys.argv = _original_argv

UserInfo = _dependencies.UserInfo
get_current_user = _dependencies.get_current_user

pytestmark = pytest.mark.offline

ENV_KEY = "sk-env-secret-1234"
ENV_HOST = "https://env.example/v1"
ALL_ROLES = {"extract", "keyword", "query", "vlm"}


def _env(role: str) -> dict:
    return {
        "binding": "openai",
        "model": f"env-{role}",
        "host": ENV_HOST,
        "api_key": ENV_KEY,
        "provider_options": {"temperature": 0.0},
    }


def _slot(host="https://new.example/v1", key="sk-new-5678", model="new-model", **kw):
    return ProviderSlot(base_url=host, api_key=SecretStr(key), model=model, **kw)


class _FakeRag:
    """Records role updates; compares settings the way the real one does."""

    def __init__(self, workspace="space1"):
        self.workspace = workspace
        self.roles = {role: _env(role) for role in ALL_ROLES}
        self.updates: list[str] = []
        self.billing: dict[str, str] = {}
        self.doc_status = SimpleNamespace(
            workspace=workspace, list_workspaces_with_statuses=self._list_workspaces
        )
        self.listed_workspaces = [workspace]

    async def _list_workspaces(self, statuses):
        return self.listed_workspaces

    def role_llm_config_matches(self, role, **settings):
        current = self.roles[role]
        return all((current.get(k) or None) == (v or None) for k, v in settings.items())

    def update_llm_role_config(self, role, **settings):
        self.roles[role] = settings
        self.updates.append(role)

    def set_role_llm_metadata(self, role, **metadata):
        self.billing[role] = metadata["billing"]

    def get_llm_role_config(self, role=None):
        current = self.roles[role]
        return {
            "binding": current["binding"],
            "model": current["model"],
            "host": current["host"],
            "metadata": {"provider_options": current["provider_options"]},
        }


@pytest.fixture
def store(tmp_path):
    return WorkspaceProviderStore(tmp_path, get_secret=lambda: "unit-test-secret")


def _set_system(store, **slots):
    store.set(SYSTEM_RECORD_ID, WorkspaceProviderConfig(**slots))


# ----------------------------- precedence -----------------------------
def test_server_default_reaches_its_roles_and_is_billed_as_system(store):
    _set_system(store, extraction=_slot())
    rag = _FakeRag()

    assert set(reconcile_system_provider(rag, "space1", store, _env)) == {
        "extract",
        "keyword",
    }
    assert rag.roles["extract"]["model"] == "new-model"
    assert rag.roles["extract"]["api_key"] == "sk-new-5678"
    assert rag.roles["query"]["model"] == "env-query"  # still on .env
    assert rag.billing == {"extract": "system", "keyword": "system"}


def test_workspace_provider_wins_over_the_server_default(store):
    _set_system(store, query=_slot())
    store.set("ws1", WorkspaceProviderConfig(query=_slot(model="own-model")))
    rag = _FakeRag("ws1")

    assert reconcile_system_provider(rag, "ws1", store, _env) == []
    assert rag.roles["query"]["model"] == "env-query"


def test_workspace_provider_only_shields_its_own_slot(store):
    _set_system(store, query=_slot(), vision=_slot(model="new-vision"))
    store.set("ws1", WorkspaceProviderConfig(query=_slot(model="own-model")))
    rag = _FakeRag("ws1")

    assert reconcile_system_provider(rag, "ws1", store, _env) == ["vlm"]
    assert rag.roles["vlm"]["model"] == "new-vision"


def test_default_instance_ignores_a_workspace_provider(store):
    """Email images use this instance, so they stay on the system key."""
    store.set("space1", WorkspaceProviderConfig(vision=_slot(key="sk-owner")))
    rag = _FakeRag()

    assert reconcile_system_provider(rag, None, store, _env) == []
    assert rag.roles["vlm"]["api_key"] == ENV_KEY


def test_clearing_returns_to_env_values(store):
    _set_system(store, query=_slot(reasoning_effort="high"))
    rag = _FakeRag()
    reconcile_system_provider(rag, "space1", store, _env)
    assert rag.roles["query"]["provider_options"] == {
        "temperature": 0.0,
        "reasoning_effort": "high",
    }

    store.delete(SYSTEM_RECORD_ID)
    assert reconcile_system_provider(rag, "space1", store, _env) == ["query"]
    assert rag.roles["query"] == _env("query")


def test_roles_already_on_target_are_not_rebuilt(store):
    _set_system(store, query=_slot())
    rag = _FakeRag()
    reconcile_system_provider(rag, "space1", store, _env)
    rag.updates.clear()

    assert reconcile_system_provider(rag, "space1", store, _env) == []
    assert rag.updates == []


def test_restoring_a_keyless_binding_clears_the_stale_key(store):
    """The Bedrock builder raises on a key, so None must not leave the old one."""

    def bedrock_env(role):
        return {
            "binding": "bedrock",
            "model": "anthropic.claude",
            "host": "https://bedrock.example",
            "api_key": None,
            "provider_options": {"max_tokens": 10},
        }

    _set_system(store, query=_slot())
    rag = _FakeRag()
    reconcile_system_provider(rag, "space1", store, bedrock_env)
    # Options for another binding don't apply to an OpenAI-compatible provider.
    assert rag.roles["query"]["provider_options"] == {}

    store.delete(SYSTEM_RECORD_ID)
    reconcile_system_provider(rag, "space1", store, bedrock_env)
    assert rag.roles["query"]["api_key"] == ""
    assert rag.roles["query"]["binding"] == "bedrock"


def test_one_role_failing_does_not_stop_the_others(store):
    _set_system(store, extraction=_slot(), query=_slot())
    rag = _FakeRag()
    real_update = rag.update_llm_role_config

    def fail_on_extract(role, **settings):
        if role == "extract":
            raise ValueError("bad builder")
        real_update(role, **settings)

    rag.update_llm_role_config = fail_on_extract
    assert set(reconcile_system_provider(rag, "space1", store, _env)) == {
        "keyword",
        "query",
    }


# ----------------------------- store -----------------------------
def test_no_workspace_id_can_reach_the_system_record():
    assert sanitize_workspace_id(SYSTEM_RECORD_ID) != SYSTEM_RECORD_ID


def test_own_provider_listing_skips_the_system_record(store):
    _set_system(store, query=_slot())
    store.set("ws1", WorkspaceProviderConfig(query=_slot()))
    assert store.workspaces_with_own_provider() == {
        "extraction": [],
        "query": ["ws1"],
        "vision": [],
    }


def test_unreadable_record_falls_back_to_env(tmp_path, store):
    _set_system(store, query=_slot())
    changed_secret = WorkspaceProviderStore(tmp_path, get_secret=lambda: "other")
    assert changed_secret.is_unreadable(SYSTEM_RECORD_ID)
    rag = _FakeRag()
    assert reconcile_system_provider(rag, "space1", changed_secret, _env) == []


# ----------------------------- reusing a key -----------------------------
def test_blank_key_reuses_the_env_key_for_the_env_host():
    key = resolve_server_slot_key(
        "query", ENV_HOST + "/", WorkspaceProviderConfig(), _env
    )
    assert key.get_secret_value() == ENV_KEY


def test_blank_key_is_not_sent_to_a_new_host():
    assert (
        resolve_server_slot_key(
            "query", "https://evil.example/v1", WorkspaceProviderConfig(), _env
        )
        is None
    )


def test_a_saved_key_for_this_slot_wins_over_other_slots():
    config = WorkspaceProviderConfig(
        extraction=_slot(host=ENV_HOST, key="sk-extraction"),
        query=_slot(host=ENV_HOST, key="sk-query"),
    )
    key = resolve_server_slot_key("query", ENV_HOST, config, _env)
    assert key.get_secret_value() == "sk-query"


def test_a_key_saved_in_another_slot_is_reused_for_the_same_host():
    config = WorkspaceProviderConfig(extraction=_slot(key="sk-extraction"))
    key = resolve_server_slot_key("vision", "https://new.example/v1", config, _env)
    assert key.get_secret_value() == "sk-extraction"


# ----------------------------- routes -----------------------------
def _admin(**metadata) -> UserInfo:
    return UserInfo(
        username="alice", role="admin", workspace_id="alice", metadata=metadata
    )


@pytest.fixture
def app(store, monkeypatch):
    default_rag = _FakeRag()
    loaded = _FakeRag("ws1")
    app = FastAPI()
    app.include_router(routes.create_system_provider_routes())
    app.dependency_overrides[get_current_user] = lambda: _admin()
    app.state.workspace_provider_store = store
    app.state.env_role_llm_settings = _env
    app.state.default_rag = default_rag
    app.state.workspace_registry = None
    app.state.workspace_manager = SimpleNamespace(
        loaded_instances=lambda: [loaded],
        list_workspaces=lambda page, page_size, persisted: ([], 42),
        max_instances=10000,
    )
    app.state.loaded_rag = loaded

    async def not_initialized(namespace, workspace=None):
        raise PipelineNotInitializedError(workspace)

    monkeypatch.setattr(routes, "get_namespace_data", not_initialized)
    app.state.emails = []

    async def record_email(actor, changes):
        app.state.emails.append((actor, changes))

    monkeypatch.setattr(routes, "email_system_provider_change", record_email)
    return app


@pytest.mark.parametrize(
    "user",
    [
        UserInfo("bob", "user", "bob", {}),
        _admin(auth_mode="api_key"),
        _admin(auth_mode="client_credentials"),
    ],
    ids=["not-admin", "api-key", "client-credentials"],
)
@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", ""),
        ("PUT", ""),
        ("DELETE", ""),
        ("POST", "/test"),
    ],
)
def test_only_signed_in_admins_can_use_the_routes(app, user, method, path):
    app.dependency_overrides[get_current_user] = lambda: user
    response = TestClient(app).request(
        method, f"/admin/system-provider{path}", json={"slot": "query"}
    )
    assert response.status_code == 403


def test_view_reports_env_as_the_source_and_never_a_full_key(app, store):
    store.set("ws1", WorkspaceProviderConfig(query=_slot()))
    response = TestClient(app).get("/admin/system-provider")

    assert response.status_code == 200
    assert ENV_KEY not in response.text
    assert "sk-new-5678" not in response.text
    data = response.json()
    assert data["status"] == "ok"
    assert data["query"]["effective"]["source"] == "env"
    assert data["query"]["env"]["host"] == ENV_HOST
    assert data["query"]["env"]["api_key_preview"] == "...1234"
    assert data["impact"] == {
        "total_workspaces": 42,
        "own_provider_by_slot": {"extraction": 0, "query": 1, "vision": 0},
        "own_provider_total": 1,
    }


def test_changing_only_the_model_keeps_the_env_key_and_applies_now(app, store):
    response = TestClient(app).put(
        "/admin/system-provider",
        json={"query": {"base_url": ENV_HOST, "model": "new-model"}},
    )

    assert response.status_code == 200
    assert ENV_KEY not in response.text
    data = response.json()
    assert data["query"]["effective"]["source"] == "webui"
    assert data["query"]["effective"]["model"] == "new-model"
    assert data["updated_by"] == "alice"
    # Saved encrypted, and live on both the default and the loaded instance.
    assert store.get(SYSTEM_RECORD_ID).query.api_key.get_secret_value() == ENV_KEY
    assert app.state.default_rag.roles["query"]["model"] == "new-model"
    assert app.state.loaded_rag.roles["query"]["model"] == "new-model"


def test_a_new_host_with_a_blank_key_saves_nothing(app, store):
    response = TestClient(app).put(
        "/admin/system-provider",
        json={"query": {"base_url": "https://evil.example/v1", "model": "m"}},
    )

    assert response.status_code == 400
    assert "own API key" in response.json()["detail"]
    assert store.get(SYSTEM_RECORD_ID) is None
    assert app.state.default_rag.updates == []


def test_saving_without_a_secret_is_refused(app, tmp_path):
    app.state.workspace_provider_store = WorkspaceProviderStore(
        tmp_path, get_secret=lambda: None
    )
    client = TestClient(app)
    assert client.get("/admin/system-provider").json()["status"] == "no_secret"
    response = client.put(
        "/admin/system-provider",
        json={"query": {"base_url": ENV_HOST, "model": "m"}},
    )
    assert response.status_code == 503


def test_returning_to_env_restores_the_env_values(app):
    client = TestClient(app)
    client.put(
        "/admin/system-provider",
        json={
            "query": {
                "base_url": "https://new.example/v1",
                "api_key": "k",
                "model": "m",
            }
        },
    )
    response = client.delete("/admin/system-provider", params={"slot": "query"})

    assert response.status_code == 200
    assert response.json()["query"]["effective"]["source"] == "env"
    assert app.state.default_rag.roles["query"] == _env("query")


def test_invalid_slot_is_rejected(app):
    response = TestClient(app).delete("/admin/system-provider", params={"slot": "x"})
    assert response.status_code == 400


def test_a_change_is_emailed_without_the_key(app):
    client = TestClient(app)
    client.put(
        "/admin/system-provider",
        json={"query": {"base_url": ENV_HOST, "model": "new-model"}},
    )

    actor, changes = app.state.emails[0]
    assert actor == "alice"
    assert changes == {
        "query": {
            "old": {"host": ENV_HOST, "model": "env-query"},
            "new": {"host": ENV_HOST, "model": "new-model"},
        }
    }
    assert ENV_KEY not in json.dumps(changes)


def test_saving_the_same_values_sends_no_email(app):
    TestClient(app).put(
        "/admin/system-provider",
        json={"query": {"base_url": ENV_HOST, "model": "env-query"}},
    )
    assert app.state.emails == []


def test_change_email_names_the_admin_and_models():
    msg = build_system_provider_change_message(
        "rag@example.com",
        ["admin@example.com"],
        "alice",
        {
            "query": {
                "old": {"host": ENV_HOST, "model": "a"},
                "new": {"host": ENV_HOST, "model": "b"},
            }
        },
    )
    assert msg["Subject"].startswith("[LightRAG]")
    assert "alice" in msg["Subject"]
    assert "Before: https://env.example/v1 / a" in msg.get_content()


# ----------------------------- paused ingestion -----------------------------
def _pause_status(role="extract", kind="budget_exhausted", code=412):
    return {
        "provider_rejection": {
            "role": role,
            "kind": kind,
            "status_code": code,
            "since": 1000.0,
        }
    }


@pytest.fixture
def paused_app(app, store, monkeypatch):
    """ws1 and ws2 paused on extraction, ws3 on vision; ws2 has its own key."""
    app.state.default_rag.listed_workspaces = ["ws1", "ws2", "ws3", "idle"]
    statuses = {
        "ws1": _pause_status(),
        "ws2": _pause_status(),
        "ws3": _pause_status(role="vlm"),
        "idle": {},
    }

    async def namespace_data(namespace, workspace=None):
        return statuses[workspace]

    monkeypatch.setattr(routes, "get_namespace_data", namespace_data)
    store.set("ws2", WorkspaceProviderConfig(extraction=_slot()))
    app.state.lifted = []

    async def record_lift(workspace):
        app.state.lifted.append(workspace)
        return True

    app.state.ran = []

    async def record_runs(app_state, workspaces):
        app.state.ran.append(workspaces)

    monkeypatch.setattr(routes, "_lift_provider_pause", record_lift)
    monkeypatch.setattr(routes, "_run_resumed_pipelines", record_runs)
    return app


def test_view_sums_up_only_pauses_the_server_default_controls(paused_app):
    data = TestClient(paused_app).get("/admin/system-provider").json()
    assert data["paused"] == {
        "total": 2,
        "by_slot": {"extraction": 1, "query": 0, "vision": 1},
        "reasons": [{"kind": "budget_exhausted", "status_code": 412, "count": 2}],
        "since": 1000.0,
    }


def test_saving_resumes_only_ingestion_paused_on_that_slot(paused_app):
    with TestClient(paused_app) as client:
        response = client.put(
            "/admin/system-provider",
            json={"extraction": {"base_url": ENV_HOST, "model": "m"}},
        )
    assert response.json()["resumed"] == 1
    assert paused_app.state.lifted == ["ws1"]
    assert paused_app.state.ran == [["ws1"]]


def test_returning_to_env_also_resumes(paused_app):
    with TestClient(paused_app) as client:
        response = client.delete("/admin/system-provider")
    assert response.json()["resumed"] == 2
    assert paused_app.state.lifted == ["ws1", "ws3"]


async def test_an_outage_ending_restarts_pipelines_a_few_at_a_time():
    """Thousands of workspaces can be paused when a spend limit is raised."""
    running = 0
    peak = 0
    done = []

    class _Rag:
        def __init__(self, workspace):
            self.workspace = workspace

        async def apipeline_process_enqueue_documents(self):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0)
            running -= 1
            done.append(self.workspace)

    async def get_instance(workspace):
        return _Rag(workspace)

    app_state = SimpleNamespace(
        default_rag=_Rag("default"),
        workspace_manager=SimpleNamespace(
            get_lightrag_instance=get_instance, max_instances=10000
        ),
    )
    workspaces = [f"ws{i}" for i in range(3000)] + [""]
    await routes._run_resumed_pipelines(app_state, workspaces)

    assert peak == routes.MAX_CONCURRENT_RESUMES
    assert len(done) == 3001
    assert "default" in done  # the unnamed workspace runs on the default instance


async def test_one_workspace_failing_to_resume_does_not_stop_the_others():
    done = []

    async def get_instance(workspace):
        if workspace == "bad":
            raise ValueError("no instance")

        async def run():
            done.append(workspace)

        return SimpleNamespace(apipeline_process_enqueue_documents=run)

    app_state = SimpleNamespace(
        default_rag=None,
        workspace_manager=SimpleNamespace(
            get_lightrag_instance=get_instance, max_instances=10000
        ),
    )
    await routes._run_resumed_pipelines(app_state, ["a", "bad", "b"])
    assert sorted(done) == ["a", "b"]


# ----------------------------- thousands of workspaces -----------------------------
def test_availability_check_does_not_scan_workspaces(app):
    def fail(**kwargs):
        raise AssertionError("must stay cheap")

    app.state.workspace_manager.list_workspaces = fail
    response = TestClient(app).get("/admin/system-provider/status")
    assert response.json() == {"available": True}


def test_view_size_does_not_grow_with_the_number_of_workspaces(app, store):
    for i in range(50):
        store.set(f"ws{i}", WorkspaceProviderConfig(query=_slot()))
    response = TestClient(app).get("/admin/system-provider")
    assert "ws1" not in response.text
    assert response.json()["impact"]["own_provider_total"] == 50


def test_own_provider_list_is_paged_and_searchable(app, store):
    for i in range(45):
        store.set(f"ws{i:02}", WorkspaceProviderConfig(query=_slot()))
    store.set(
        "Special-1",
        WorkspaceProviderConfig(query=_slot(), vision=_slot()),
    )
    client = TestClient(app)

    first = client.get("/admin/system-provider/own-providers").json()
    assert first["total"] == 46
    assert len(first["workspaces"]) == 20

    last = client.get(
        "/admin/system-provider/own-providers", params={"page": 3, "page_size": 20}
    ).json()
    assert len(last["workspaces"]) == 6

    found = client.get(
        "/admin/system-provider/own-providers", params={"search": "special"}
    ).json()
    assert found["total"] == 1
    assert found["workspaces"] == [
        {"workspace_id": "Special-1", "slots": ["query", "vision"]}
    ]


def test_own_provider_page_size_is_capped(app):
    response = TestClient(app).get(
        "/admin/system-provider/own-providers", params={"page_size": 5000}
    )
    assert response.status_code == 422


# ----------------------------- provider test -----------------------------
@pytest.fixture
def provider(monkeypatch):
    """Answer the provider test's HTTP calls from ``provider.handler``."""
    state = SimpleNamespace(handler=None, requests=[])
    real_client = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        state.requests.append(request)
        return state.handler(request)

    monkeypatch.setattr(
        routes.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    return state


def _test_request(app, slot="query", **body):
    return TestClient(app).post(
        "/admin/system-provider/test",
        json={"slot": slot, "base_url": ENV_HOST, "model": "m", **body},
    )


def test_provider_test_passes_on_a_real_completion(app, store, provider):
    provider.handler = lambda r: httpx.Response(200, json={"choices": [{}]})
    data = _test_request(app).json()

    assert data["ok"] is True
    assert data["outcome"] == "ok"
    sent = json.loads(provider.requests[0].content)
    assert provider.requests[0].url == ENV_HOST + "/chat/completions"
    assert provider.requests[0].headers["authorization"] == f"Bearer {ENV_KEY}"
    assert sent["model"] == "m"
    assert sent["messages"][0]["content"] == "Reply with OK."
    assert store.get(SYSTEM_RECORD_ID) is None  # saves nothing


def test_provider_test_for_vision_sends_an_image(app, provider):
    provider.handler = lambda r: httpx.Response(200, json={"choices": [{}]})
    _test_request(app, slot="vision")

    content = json.loads(provider.requests[0].content)["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "status,outcome,can_apply_anyway",
    [
        (401, "invalid_key", False),
        (403, "invalid_key", False),
        (402, "spend_limit", False),
        (412, "spend_limit", False),
        (400, "model_rejected", False),
        (404, "model_rejected", False),
        (429, "provider_error", True),
        (500, "provider_error", True),
        (503, "provider_error", True),
    ],
)
def test_provider_test_outcomes(app, provider, status, outcome, can_apply_anyway):
    provider.handler = lambda r: httpx.Response(
        status, json={"error": {"message": "nope"}}
    )
    data = _test_request(app).json()

    assert data["ok"] is False
    assert data["outcome"] == outcome
    assert data["status_code"] == status
    assert data["message"] == "nope"
    assert data["can_apply_anyway"] is can_apply_anyway


def test_provider_test_unreachable_can_still_be_applied(app, provider):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    provider.handler = refuse
    data = _test_request(app).json()
    assert data["outcome"] == "unreachable"
    assert data["can_apply_anyway"] is True


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>Not an API</html>"),
        httpx.Response(200, json={"error": {"message": "no such model"}}),
    ],
    ids=["web-page", "error-in-200"],
)
def test_provider_test_rejects_a_200_without_a_completion(app, provider, response):
    provider.handler = lambda r: response
    data = _test_request(app).json()
    assert data["outcome"] == "model_rejected"
    assert data["can_apply_anyway"] is False


def test_provider_test_retries_when_max_tokens_is_refused_by_name(app, provider):
    def handler(request):
        if "max_tokens" in json.loads(request.content):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'."
                    }
                },
            )
        return httpx.Response(200, json={"choices": [{}]})

    provider.handler = handler
    assert _test_request(app).json()["ok"] is True
    assert len(provider.requests) == 2


def test_provider_test_scrubs_the_key_from_provider_messages(app, provider):
    provider.handler = lambda r: httpx.Response(
        401, json={"error": {"message": f"Bad key {ENV_KEY}"}}
    )
    response = _test_request(app)
    assert ENV_KEY not in response.text
    assert response.json()["message"] == "Bad key ***"


def test_provider_test_applies_the_key_rules(app, provider):
    response = TestClient(app).post(
        "/admin/system-provider/test",
        json={"slot": "query", "base_url": "https://evil.example/v1", "model": "m"},
    )
    assert response.status_code == 400
    assert provider.requests == []


# ----------------------------- other workers -----------------------------
async def test_watch_applies_a_change_made_by_another_worker(store):
    default_rag = _FakeRag()
    loaded = _FakeRag("ws1")
    manager = SimpleNamespace(loaded_instances=lambda: [loaded])
    task = asyncio.create_task(
        watch_system_provider(default_rag, manager, store, _env, interval_seconds=0.01)
    )
    try:
        _set_system(store, query=_slot())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if loaded.roles["query"]["model"] == "new-model":
                break
        assert default_rag.roles["query"]["model"] == "new-model"
        assert loaded.roles["query"]["model"] == "new-model"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_applying_a_change_lets_other_requests_run_between_instances(store):
    _set_system(store, query=_slot())
    loaded = [_FakeRag(f"ws{i}") for i in range(20)]
    progress = []

    async def other_request():
        while len(progress) < 3:
            progress.append(
                sum(r.roles["query"]["model"] == "new-model" for r in loaded)
            )
            await asyncio.sleep(0)

    other = asyncio.create_task(other_request())
    await reconcile_loaded_instances(
        _FakeRag(), SimpleNamespace(loaded_instances=lambda: loaded), store, _env
    )
    await other
    assert 0 < progress[1] < 20  # it ran while instances were still being updated
    assert all(r.roles["query"]["model"] == "new-model" for r in loaded)


# ----------------------------- a real instance -----------------------------
def _real_rag(tmp_path):
    """A real LightRAG whose role builder records the settings it is built with."""

    class _Tokenizer:
        def encode(self, content):
            return [ord(ch) for ch in content]

        def decode(self, tokens):
            return "".join(chr(t) for t in tokens)

    async def embed(texts):
        return np.zeros((len(texts), 16))

    def built_func(metadata):
        async def call(*args, **kwargs):
            return f"{metadata.get('host')} {metadata.get('model')}"

        return call

    env = _env("query")
    rag = LightRAG(
        working_dir=str(tmp_path / "real"),
        workspace="real",
        llm_model_func=built_func(env),
        embedding_func=EmbeddingFunc(embedding_dim=16, max_token_size=4096, func=embed),
        tokenizer=Tokenizer("mock", _Tokenizer()),
        role_llm_configs={
            "query": RoleLLMConfig(func=built_func(env), metadata=dict(env))
        },
    )
    rag.register_role_llm_builder(lambda role, meta: (built_func(dict(meta)), None))
    return rag


async def test_a_real_instance_switches_records_system_billing_and_returns(
    tmp_path, store
):
    records = []
    add_usage_sink(records.append)
    try:
        rag = _real_rag(tmp_path)
        captured = rag._build_global_config()["role_llm_funcs"]["query"]
        env = lambda role: _env("query")  # noqa: E731

        _set_system(store, query=_slot())
        assert "query" in reconcile_system_provider(rag, "real", store, env)
        assert await captured("q") == "https://new.example/v1 new-model"
        assert records[-1].billing == "system"
        assert records[-1].model == "new-model"
        cache_identity = rag._build_global_config()["llm_cache_identities"]["query"]
        assert cache_identity["model"] == "new-model"

        store.delete(SYSTEM_RECORD_ID)
        assert "query" in reconcile_system_provider(rag, "real", store, env)
        assert await captured("q") == f"{ENV_HOST} env-query"
        assert records[-1].billing == "system"
        await rag.wait_for_retired_llm_queues()
    finally:
        remove_usage_sink(records.append)


async def test_a_real_instance_records_byo_billing_for_a_workspace_provider(
    tmp_path, store
):
    records = []
    add_usage_sink(records.append)
    try:
        rag = _real_rag(tmp_path)
        store.set("real", WorkspaceProviderConfig(query=_slot(model="own-model")))
        await apply_workspace_provider_overrides(rag, "real", store)
        await rag.role_llm_funcs["query"]("q")
        assert records[-1].billing == "byo"
        assert records[-1].model == "own-model"
    finally:
        remove_usage_sink(records.append)
