"""Health tab routes: dependency grid and drawer, workers, config, re-check
and SMTP test (live admin, CSRF, host-wide rate limits), and the Overview's
dependency and worker parts including the leased attention snapshot."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lightrag.api.ops import auth as ops_auth
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.reconcile import attention_cycle
from tests.ops_fixtures import (
    OTHER_WORKER_ID,
    WORKER_ID,
    admin,
    build_app,
    ops_routes,
    seed_health,
    seeded_store,
)

pytestmark = pytest.mark.offline

CLIENT = {"X-LightRAG-Client": "webui"}


@pytest.fixture
def store():
    return seeded_store()


@pytest.fixture
def app(store, monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    application = build_app(store)
    seed_health(application, store)
    application.dependency_overrides[ops_auth.require_live_admin] = admin
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_grid(client):
    response = client.get("/admin/ops/health")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["enabled"] is True and body["available"] is True
    deps = {d["id"]: d for d in body["deps"]}
    # Sign-in is the passive card from the auth counters (no probe of its own).
    assert set(deps) == {"postgres", "neo4j", "smtp", "llm", "etcd", "signin"}
    assert deps["signin"]["passive"] is True and deps["signin"]["status"] == "ok"
    assert deps["postgres"]["status"] == "ok"
    assert deps["neo4j"]["status"] == "down"
    assert deps["neo4j"]["consecutive_failures"] == 4
    assert deps["llm"]["status"] == "unknown" and deps["llm"]["passive"] is True
    assert deps["etcd"]["status"] == "unprobed"
    strip = deps["neo4j"]["strip"]
    assert len(strip) == 48
    assert strip[-1]["status"] in ("down", "unknown")
    assert any(c["status"] == "down" and c["failed"] >= 1 for c in strip)
    assert strip[0]["start"] < strip[-1]["start"]
    assert deps["etcd"]["strip"] == []
    assert body["cycle"]["holder"] == WORKER_ID and body["cycle"]["holder_self"]
    assert body["summary"]["total"] == 5 and body["summary"]["down"] == 1


def test_status_reports_probes_and_heartbeat_workers(client):
    body = client.get("/admin/ops/status").json()
    assert body["features"]["probes"] is True
    assert body["process"]["workers_live"] == 1


def test_dependency_drawer(client, store):
    body = client.get("/admin/ops/health/neo4j", params={"hours": 24}).json()
    assert body["dep"]["id"] == "neo4j"
    assert body["error_classes"] == [{"key": "timeout", "count": 3}]
    assert len(body["recent"]) == 3
    assert body["recent"][0]["status"] == "down"
    assert body["latency_series"] and body["latency_series"][0]["t"] > 1e12
    assert body["blocking"] == []
    store.kv["probe:postgres"]["detail"]["blocking"] = [
        {"pid": 5, "blocked_by": [6], "duration_s": 40.0, "fingerprint": "ab"}
    ]
    client.app.state.ops.service.cache.clear()
    pg = client.get("/admin/ops/health/postgres").json()
    assert pg["blocking"][0]["pid"] == 5


def test_dependency_drawer_errors(client):
    assert client.get("/admin/ops/health/unknown_dep").status_code == 404
    bad = client.get("/admin/ops/health/Bad-Dep!")
    assert bad.status_code == 400 and bad.json()["detail"] == "invalid_dependency"
    # The recorder card exists only when a worker reports a recorder.
    assert client.get("/admin/ops/health/recorder").status_code == 404
    assert (
        client.get("/admin/ops/health/neo4j", params={"hours": 500}).status_code == 422
    )


def test_workers_table(client):
    body = client.get("/admin/ops/workers").json()
    items = body["items"]
    assert [i["worker_id"] for i in items] == [WORKER_ID, OTHER_WORKER_ID]
    assert items[0]["alive"] is True and items[0]["self"] is True
    assert items[1]["alive"] is False and items[1]["heartbeat_age_s"] >= 590
    assert items[0]["leases_held"] == ["ops:probes"]
    assert body["live"] == 1 and body["expected"] == 1 and body["mode"] == "uvicorn"
    assert body["host"]["load"] == [3.2, 2.9, 2.7]


def test_worker_history(client):
    body = client.get(f"/admin/ops/workers/{WORKER_ID}/history").json()
    assert body["worker_id"] == WORKER_ID and body["bucket_s"] == 60
    [sample] = body["items"]
    assert set(sample) == {"t", "rss", "cpu", "loop_lag", "pool_wait"}
    assert sample["t"] % 60000 == 0
    week = client.get(
        f"/admin/ops/workers/{WORKER_ID}/history", params={"hours": 168}
    ).json()
    assert week["bucket_s"] == 600
    bad = client.get("/admin/ops/workers/not-a-uuid/history")
    assert bad.status_code == 400 and bad.json()["detail"] == "invalid_worker_id"


def test_config_is_redacted(app, client):
    app.state.ops_server = ops_routes.OpsServerInfo(
        rag=app.state.ops_server.rag,
        workers=None,
        version="0295",
        status_provider=lambda: {
            "working_directory": "/app/data/rag_storage",
            "input_directory": "/app/data/inputs",
            "configuration": {
                "llm_binding_host": "https://bob:pa55word@router.example/v1?key=sk-abc12345678",
                "embedding_binding_api_key": "sk-live-123456789",
                "max_graph_nodes": 1000,
                "role_llm_config": {"extract": {"host": "http://u:p@vllm:8000/v1"}},
            },
            "core_version": "1.5",
            "api_version": "0295",
        },
    )
    response = client.get("/admin/ops/config")
    body = response.json()
    config = body["configuration"]
    assert config["llm_binding_host"] == "https://router.example/v1"
    assert config["embedding_binding_api_key"] == "<redacted>"
    assert config["max_graph_nodes"] == 1000
    assert config["role_llm_config"]["extract"]["host"] == "http://vllm:8000/v1"
    assert body["working_directory"] == "/app/data/rag_storage"
    for secret in ("pa55word", "sk-abc", "sk-live", "u:p@"):
        assert secret not in response.text


def test_overview_fills_deps_and_workers(client):
    body = client.get("/admin/ops/overview").json()
    assert {d["id"] for d in body["deps"]} >= {"postgres", "neo4j"}
    assert all(d["strip"] == [] for d in body["deps"])
    assert body["workers"][0]["worker_id"] == WORKER_ID
    tile = body["kpis"]["deps"]
    # postgres, smtp and sign-in ok; neo4j down; etcd not probed.
    assert tile["available"] is True and tile["ok"] == 3 and tile["total"] == 5
    assert tile["state"] == "critical"
    down = [a for a in body["attention"] if a["rule"] == "dependency_down"]
    assert down and down[0]["severity"] == "critical"
    assert down[0]["id"] == "dependency_down:neo4j"
    assert body["attention_source"] == "request"
    assert body["kpis"]["workers"]["live"] == 1


def test_overview_reads_the_leased_attention_snapshot(client, store):
    store.kv["attention"] = {
        "at": "2026-11-03T00:00:00+00:00",
        "at_epoch": time.time() - 20,
        "holder": OTHER_WORKER_ID,
        "items": [
            {
                "id": "disk_free",
                "rule": "disk_free",
                "severity": "warning",
                "title": "Disk 12% free",
                "params": {"free_pct": 12},
                "since": None,
                "count": None,
                "link": "#/?ops=1&otab=health",
                "runbook": None,
            }
        ],
        "streaks": {},
    }
    body = client.get("/admin/ops/overview").json()
    assert body["attention_source"] == "snapshot"
    assert [a["rule"] for a in body["attention"]] == ["disk_free"]
    assert body["kpis"]["deps"]["state"] == "warning"


def test_a_stale_snapshot_is_ignored(client, store):
    store.kv["attention"] = {"at_epoch": time.time() - 3600, "items": [], "streaks": {}}
    body = client.get("/admin/ops/overview").json()
    assert body["attention_source"] == "request"


async def test_attention_cycle_publishes_under_the_lease_with_streaks(app, store):
    service = app.state.ops.service
    service.started_at = time.time() - 600
    process = {"mode": "gunicorn", "expected": 2, "live": None}
    loop = LeasedLoop(
        store,
        "ops:reconcile",
        WORKER_ID,
        lambda lp: attention_cycle(lp, service=service, process=lambda: process),
        ttl_s=120,
        interval_s=60,
        cycle_timeout_s=90,
    )
    assert await loop.run_once()
    first = store.kv["attention"]
    assert first["holder"] == WORKER_ID
    assert first["streaks"] == {"workers_below_expected": 1}
    assert "workers_below_expected" not in {a["rule"] for a in first["items"]}
    assert "dependency_down" in {a["rule"] for a in first["items"]}
    service.cache.clear()
    assert await loop.run_once()
    second = store.kv["attention"]
    rules = {a["rule"]: a for a in second["items"]}
    assert rules["workers_below_expected"]["severity"] == "warning"
    assert rules["workers_below_expected"]["params"] == {"live": 1, "expected": 2}
    # Another worker holds the lease: nothing is written.
    store.kv["lease:ops:reconcile"] = {"holder": "other", "exp": time.time() + 120}
    store.kv["attention"] = {"marker": True}
    assert not await loop.run_once()
    assert store.kv["attention"] == {"marker": True}


def test_recheck_runs_one_probe_and_is_rate_limited(client, store):
    response = client.post("/admin/ops/health/neo4j/probe", headers=CLIENT, json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "neo4j" and body["status"] == "ok"
    assert body["consecutive_failures"] == 0
    assert store.kv["probe:neo4j"]["status"] == "ok"
    assert store.probe_rows[-1]["dependency"] == "neo4j"
    limited = client.post("/admin/ops/health/neo4j/probe", headers=CLIENT, json={})
    assert limited.status_code == 429
    assert limited.json()["detail"] == "rate_limited"
    assert limited.headers["retry-after"] == "10"
    # The limit is per dependency.
    other = client.post("/admin/ops/health/postgres/probe", headers=CLIENT, json={})
    assert other.status_code == 200


def test_recheck_refusals(client, app):
    assert (
        client.post("/admin/ops/health/llm/probe", headers=CLIENT).json()["detail"]
        == "not_checkable"
    )
    assert (
        client.post("/admin/ops/health/etcd/probe", headers=CLIENT).status_code == 400
    )
    assert (
        client.post("/admin/ops/health/nope/probe", headers=CLIENT).status_code == 404
    )
    app.state.ops.service.probes = None
    disabled = client.post("/admin/ops/health/neo4j/probe", headers=CLIENT)
    assert (
        disabled.status_code == 503 and disabled.json()["detail"] == "probes_disabled"
    )


def test_smtp_test_never_sends_and_is_limited_to_once_a_minute(client):
    response = client.post("/admin/ops/smtp/test", headers=CLIENT, json={})
    body = response.json()
    assert response.status_code == 200 and body["ok"] is True
    assert body["detail"]["starttls"] is True
    again = client.post("/admin/ops/smtp/test", headers=CLIENT, json={})
    assert again.status_code == 429 and again.headers["retry-after"] == "60"


def test_smtp_recheck_shares_the_smtp_test_limit(client):
    first = client.post("/admin/ops/health/smtp/probe", headers=CLIENT, json={})
    assert first.status_code == 200, first.text
    # Neither route opens another session with the relay within the minute.
    again = client.post("/admin/ops/health/smtp/probe", headers=CLIENT, json={})
    assert again.status_code == 429 and again.headers["retry-after"] == "60"
    test = client.post("/admin/ops/smtp/test", headers=CLIENT, json={})
    assert test.status_code == 429 and test.headers["retry-after"] == "60"


def test_slow_cadence_probes_are_rechecked_at_most_once_a_minute():
    from lightrag.api.ops.probes import ProbeSpec, recheck_limit

    assert recheck_limit(ProbeSpec("neo4j", "Neo4j", "graph")) == (
        "recheck:neo4j",
        10.0,
    )
    assert recheck_limit(ProbeSpec("tls", "TLS", "tls", interval_s=900.0)) == (
        "recheck:tls",
        60.0,
    )
    assert recheck_limit(ProbeSpec("smtp", "SMTP", "mail", interval_s=900.0)) == (
        "smtp_test",
        60.0,
    )


def test_smtp_test_without_smtp_configured(client, app):
    runner = app.state.ops.service.probes
    del runner.specs["smtp"]
    body = client.post("/admin/ops/smtp/test", headers=CLIENT).json()
    assert body["ok"] is False and body["error_class"] == "not_configured"


def test_posts_need_the_webui_client_header_and_a_same_origin(client):
    missing = client.post("/admin/ops/smtp/test", json={})
    assert missing.status_code == 403
    assert missing.json()["detail"] == "client_header_required"
    foreign = client.post(
        "/admin/ops/smtp/test",
        headers={**CLIENT, "Origin": "https://evil.example"},
        json={},
    )
    assert (
        foreign.status_code == 403 and foreign.json()["detail"] == "origin_not_allowed"
    )
    form = client.post(
        "/admin/ops/health/neo4j/probe",
        headers={**CLIENT, "Content-Type": "application/x-www-form-urlencoded"},
        content=b"a=1",
    )
    assert form.status_code == 415
    same = client.post(
        "/admin/ops/health/neo4j/probe",
        headers={**CLIENT, "Origin": "http://testserver"},
        json={},
    )
    assert same.status_code == 200


def test_post_origin_check_behind_a_reverse_proxy(client, monkeypatch):
    args = SimpleNamespace(
        api_key_role="admin",
        cors_origins="*",
        oauth2_enabled=True,
        oauth2_client_id="lightrag",
        oauth2_client_secret="s",
        oauth2_redirect_uri="https://rag.example/oauth2/callback",
    )
    monkeypatch.setattr(ops_auth, "_global_args", lambda: args)
    monkeypatch.setattr(ops_auth, "_refusal_logged", False)
    # nginx `proxy_pass $var` sends Host: <upstream>; the browser still says
    # the request is same-origin.
    proxied = {**CLIENT, "Host": "lightrag:9621", "Origin": "https://10.0.0.5"}
    ok = client.post(
        "/admin/ops/health/neo4j/probe",
        headers={**proxied, "Sec-Fetch-Site": "same-origin"},
        json={},
    )
    assert ok.status_code == 200, ok.text
    # Without fetch metadata: the SSO redirect URI's origin is ours.
    sso = client.post(
        "/admin/ops/health/postgres/probe",
        headers={**proxied, "Origin": "https://rag.example"},
        json={},
    )
    assert sso.status_code == 200, sso.text
    # X-Forwarded-Host / Forwarded, and a Host without its port.
    for headers in (
        {"X-Forwarded-Host": "rag.internal"},
        {"Forwarded": 'for=10.0.0.9;proto=https;host="rag.internal"'},
        {"Host": "rag.internal"},
    ):
        assert ops_auth.origin_allowed(
            SimpleNamespace(
                headers={
                    "host": "lightrag:9621",
                    **{k.lower(): v for k, v in headers.items()},
                }
            ),
            "https://rag.internal:8443",
        ), headers
    # Default ports compare equal; other ports and hosts do not.
    request = SimpleNamespace(headers={"host": "rag.internal:443"})
    assert ops_auth.origin_allowed(request, "https://rag.internal")
    assert not ops_auth.origin_allowed(request, "https://rag.internal:8443")
    assert not ops_auth.origin_allowed(request, "https://evil.example")
    assert not ops_auth.origin_allowed(request, "null")
    # "*" is not an origin; a cross-site page is refused even with the header.
    cross = client.post(
        "/admin/ops/health/neo4j/probe",
        headers={
            **proxied,
            "Origin": "https://evil.example",
            "Sec-Fetch-Site": "cross-site",
        },
        json={},
    )
    assert cross.status_code == 403
    no_origin = client.post(
        "/admin/ops/smtp/test",
        headers={**CLIENT, "Sec-Fetch-Site": "cross-site"},
        json={},
    )
    assert no_origin.status_code == 403


def test_posts_need_a_live_admin(app, monkeypatch):
    del app.dependency_overrides[ops_auth.require_live_admin]
    import lightrag.api.auth as core_auth

    monkeypatch.setattr(
        core_auth, "global_args", SimpleNamespace(admin_accounts="root")
    )
    monkeypatch.setattr(
        ops_auth,
        "_global_args",
        lambda: SimpleNamespace(api_key_role="admin", cors_origins=""),
    )
    client = TestClient(app)
    assert (
        client.post("/admin/ops/health/neo4j/probe", headers=CLIENT).status_code == 200
    )
    # Removed from ADMIN_ACCOUNTS while the token still says admin.
    monkeypatch.setattr(
        core_auth, "global_args", SimpleNamespace(admin_accounts="other")
    )
    refused = client.post("/admin/ops/health/postgres/probe", headers=CLIENT)
    assert (
        refused.status_code == 403 and refused.json()["detail"] == "admin_not_current"
    )
    # Impersonated sessions are refused whatever the admin list says.
    monkeypatch.setattr(
        core_auth, "global_args", SimpleNamespace(admin_accounts="root")
    )
    app.dependency_overrides[ops_routes.require_admin] = lambda: SimpleNamespace(
        username="root",
        role="admin",
        workspace_id="w",
        metadata={"impersonated_by": "root", "impersonation": True},
    )
    impersonated = client.post("/admin/ops/health/postgres/probe", headers=CLIENT)
    assert impersonated.status_code == 403
    assert impersonated.json()["detail"] == "impersonation_not_allowed"


def test_live_admin_for_service_accounts(monkeypatch):
    from lightrag.api.dependencies import UserInfo

    monkeypatch.setattr(
        ops_auth, "_global_args", lambda: SimpleNamespace(api_key_role="user")
    )
    key_user = UserInfo(
        username="api_key_service_account",
        role="admin",
        workspace_id="service_account",
        metadata={"auth_mode": "api_key"},
    )
    assert ops_auth.is_live_admin(key_user) is False
    monkeypatch.setattr(
        ops_auth, "_global_args", lambda: SimpleNamespace(api_key_role="admin")
    )
    assert ops_auth.is_live_admin(key_user) is True
    import lightrag.api.obo_allowlist as obo

    monkeypatch.setattr(obo, "check_admin_client", lambda cid: cid == "ops-bot")
    bot = UserInfo(
        username="service-account-ops-bot",
        role="admin",
        workspace_id="service_account",
        metadata={"auth_mode": "client_credentials", "client_id": "ops-bot"},
    )
    assert ops_auth.is_live_admin(bot) is True
    bot.metadata["client_id"] = "removed-bot"
    assert ops_auth.is_live_admin(bot) is False


def test_health_routes_without_postgres_still_answer(monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    app = build_app(None, kv_storage="JsonKVStorage")
    client = TestClient(app)
    for path in (
        "/admin/ops/health",
        "/admin/ops/health/postgres",
        "/admin/ops/workers",
        f"/admin/ops/workers/{WORKER_ID}/history",
        "/admin/ops/config",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
    body = client.get("/admin/ops/health").json()
    assert body["available"] is False and body["enabled"] is False


async def test_local_probes_and_heartbeat_without_postgres(monkeypatch):
    """NFR-8: without the ops database the probes and this process still show."""
    from lightrag.api.ops.heartbeat import Heartbeat
    from lightrag.api.ops.snapshot import NamespaceReader
    from tests.ops_fixtures import health_runner

    monkeypatch.setattr(ops_routes, "process_mode", lambda: "uvicorn")
    app = build_app(None, kv_storage="JsonKVStorage")
    service = ops_routes._service(SimpleNamespace(app=app))
    runner = health_runner(None)
    service.probes = runner
    beat = Heartbeat(
        worker_id=WORKER_ID,
        mode="uvicorn",
        version="0295",
        store=None,
        reader=NamespaceReader(),
        app_state=app.state,
        leases=lambda: [],
        data_rag=lambda: None,
    )
    service.heartbeat = beat
    service.worker_id = WORKER_ID
    await beat.beat_once()
    await runner.run_cycle()
    runner.close()
    client = TestClient(app)
    deps = {d["id"]: d for d in client.get("/admin/ops/health").json()["deps"]}
    assert deps["neo4j"]["status"] == "ok" and deps["neo4j"]["strip"] == []
    workers = client.get("/admin/ops/workers").json()
    assert workers["items"][0]["self"] is True and workers["live"] == 1
    assert workers["available"] is False
