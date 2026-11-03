"""Operations console routes on the durable run-tracking sources: run
history and drawer, active runs, the five stuck classes, failure kinds,
persisted pauses, and the drawers' runs and failures."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from lightrag.api.ops.service import BUSY_WITHOUT_RUN_KEY, RESERVATIONS_KEY
from lightrag.kg import shared_storage
from tests.ops_fixtures import (
    DONE_RUN_ID,
    ORPHAN_RUN_ID,
    RUN_ID,
    SECRET_EMAIL,
    SECRET_KEY,
    WORKER_ID,
    FakeRegistry,
    build_app,
    ops_routes,
    seed_health,
    seed_runs,
    seeded_store,
    shared_state,
)

pytestmark = pytest.mark.offline


def _clean(text: str) -> None:
    for secret in (SECRET_EMAIL, SECRET_KEY, "Senate minutes", "900101"):
        assert secret not in text, secret


@pytest.fixture
def shared(monkeypatch):
    data = shared_state()
    # fsktm is busy for RUN_ID; space1 still holds busy for the orphaned run.
    data["fsktm:pipeline_status"]["run_id"] = RUN_ID
    data["space1:pipeline_status"]["run_id"] = ORPHAN_RUN_ID
    monkeypatch.setattr(shared_storage, "_shared_dicts", data)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    return data


@pytest.fixture
def store():
    store = seeded_store()
    seed_runs(store)
    now = time.time()
    store.kv[BUSY_WITHOUT_RUN_KEY] = {
        "at_epoch": now - 10,
        "at": datetime.fromtimestamp(now - 10, timezone.utc).isoformat(),
        "items": [{"workspace": "law", "reason": "no_run", "age_s": 600}],
        "responsive": True,
    }
    store.kv[RESERVATIONS_KEY] = {
        "at_epoch": now - 10,
        "items": [
            {
                "workspace": "hr_office",
                "kind": "pending_enqueues",
                "reason": "held",
                "count": 2,
            }
        ],
        "responsive": True,
    }
    store.kv["pause:hr_office:extraction"] = {
        "workspace": "hr_office",
        "slot": "extraction",
        "role": "extract",
        "kind": "budget_exhausted",
        "status_code": 402,
        "byo": True,
        "since": now - 600,
        "retry_at": now + 600,
        "message": f"Payment required for {SECRET_KEY}",
        "cleared_at": None,
    }
    store.kv["pause:law:query"] = {
        "workspace": "law",
        "slot": "query",
        "kind": "invalid_key",
        "since": now - 60,
        "retry_at": now - 1,
        "cleared_at": now - 5,
    }
    return store


@pytest.fixture
def app(store, shared):
    app = build_app(store, registry=FakeRegistry({"fsktm": "a.bin@unimas.my"}))
    seed_health(app, store)
    app.state.ops.service.run_tracking = True
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_status_reports_run_tracking(client, app):
    assert client.get("/admin/ops/status").json()["features"]["runs"] is True
    app.state.ops.service.run_tracking = False
    assert client.get("/admin/ops/status").json()["features"]["runs"] is False


def test_run_history_keyset_paging_and_filters(client):
    first = client.get("/admin/ops/runs", params={"limit": 2}).json()
    assert [r["run_id"] for r in first["items"]] == [RUN_ID, ORPHAN_RUN_ID]
    assert first["next_before"]
    second = client.get(
        "/admin/ops/runs", params={"limit": 2, "before": first["next_before"]}
    ).json()
    assert [r["run_id"] for r in second["items"]] == [DONE_RUN_ID]
    assert second["next_before"] is None
    only = client.get(
        "/admin/ops/runs", params={"workspace": "fsktm", "status": "succeeded"}
    ).json()
    assert [r["run_id"] for r in only["items"]] == [DONE_RUN_ID]
    assert only["items"][0]["duration_s"] == pytest.approx(100, abs=2)
    kinds = client.get("/admin/ops/runs", params={"kind": "email_batch"}).json()
    assert [r["status"] for r in kinds["items"]] == ["orphaned"]
    for bad in (
        {"kind": "nope"},
        {"status": "nope"},
        {"before": "yesterday"},
        {"workspace": "../etc"},
    ):
        response = client.get("/admin/ops/runs", params=bad)
        assert response.status_code == 400, bad


def test_run_drawer(client):
    body = client.get(f"/admin/ops/runs/{RUN_ID}").json()
    run = body["run"]
    assert run["workspace"] == "fsktm" and run["status"] == "running"
    assert run["worker_alive"] is True and run["owns_busy"] is True
    assert run["live"]["busy"] is True
    assert "Senate minutes" not in (run["label"] or "")
    assert [f["error_kind"] for f in body["failures"]] == ["cancelled", "rate_limited"]
    assert body["log_count"] is None
    orphan = client.get(f"/admin/ops/runs/{ORPHAN_RUN_ID}").json()["run"]
    assert orphan["status"] == "orphaned" and orphan["owns_busy"] is True
    assert client.get("/admin/ops/runs/" + "0" * 32).status_code == 404
    assert client.get("/admin/ops/runs/not-a-run").status_code == 400
    _clean(client.get(f"/admin/ops/runs/{RUN_ID}").text)


def test_active_runs_come_from_the_runs_table(client):
    body = client.get("/admin/ops/processing/active").json()
    by_ws = {r["workspace"]: r for r in body["runs"]}
    fsktm = by_ws["fsktm"]
    assert fsktm["run_id"] == RUN_ID and fsktm["source"] == "runs"
    assert fsktm["state"] == "running" and fsktm["worker_alive"] is True
    assert (fsktm["docs_done"], fsktm["docs_total"]) == (1, 4)
    assert fsktm["live"]["busy"] is True and fsktm["live"]["run_id"] == RUN_ID
    # space1 has a parsing document but no running run (its run is
    # orphaned): it is added from doc status, labelled approximate.
    assert by_ws["space1"]["source"] == "doc_status"
    assert body["approximate"] is True and body["source"] == "runs"
    _clean(str(body))


def test_a_hung_run_and_a_lost_owner(client, store):
    row = store.run_rows[RUN_ID]
    row["progress_at"] = datetime.fromtimestamp(time.time() - 50 * 60, timezone.utc)
    body = client.get("/admin/ops/processing/active").json()
    fsktm = next(r for r in body["runs"] if r["workspace"] == "fsktm")
    assert fsktm["state"] == "hung"
    stuck = client.get("/admin/ops/processing/stuck").json()
    [hung] = stuck["hung"]
    assert hung["run_id"] == RUN_ID and hung["severity"] == "serious"
    assert hung["guidance"] == "check_feed_or_cancel"


def test_stuck_lists_the_five_classes(client, shared):
    body = client.get("/admin/ops/processing/stuck").json()
    assert body["checked_classes"] == [
        "orphaned",
        "hung",
        "busy_without_run",
        "reservations",
        "stale_docs",
    ]
    [orphan] = body["orphaned"]
    assert orphan["run_id"] == ORPHAN_RUN_ID and orphan["stranded"] is True
    assert orphan["guidance"] == "restart_container"
    assert body["hung"] == []
    assert body["busy_without_run"][0]["workspace"] == "law"
    assert body["reservations"][0]["kind"] == "pending_enqueues"
    assert body["stale_docs"]["total"] == 1
    assert body["thresholds"]["hung_serious_minutes"] == 45
    assert body["reconciled"]["reservations"]["stale"] is False
    # Once space1 no longer holds busy for it, the orphaned run is history.
    shared["space1:pipeline_status"]["busy"] = False
    app_client = client
    app_client.app.state.ops.service.cache.clear()
    assert app_client.get("/admin/ops/processing/stuck").json()["orphaned"] == []


def test_summary_and_overview_count_stuck_runs(client):
    summary = client.get("/admin/ops/processing/summary").json()
    # orphaned 1 + busy-without-run 1 runs; 1 reservation.
    assert summary["stuck"] == {"runs": 2, "docs": 1, "reservations": 1}
    assert summary["durable"] is True
    overview = client.get("/admin/ops/overview").json()
    stuck = overview["kpis"]["stuck"]
    assert (stuck["runs"], stuck["docs"], stuck["reservations"]) == (2, 1, 1)
    assert stuck["value"] == 4 and stuck["classes"] == "all"
    rules = {a["rule"] for a in overview["attention"]}
    # Orphaned-run and reservation rules are Gunicorn-only.
    assert "run_orphaned" not in rules
    assert "reservation_without_owner" not in rules
    assert "busy_without_run" in rules


def test_gunicorn_overview_raises_run_rules(client, monkeypatch):
    monkeypatch.setattr(ops_routes, "process_mode", lambda: "gunicorn")
    monkeypatch.setattr(ops_routes, "_count_gunicorn_workers", lambda: 2)
    overview = client.get("/admin/ops/overview").json()
    rules = {a["rule"]: a for a in overview["attention"]}
    assert rules["run_orphaned"]["severity"] == "critical"
    assert "orun=" in rules["run_orphaned"]["link"]
    assert rules["reservation_without_owner"]["severity"] == "serious"
    assert overview["kpis"]["stuck"]["state"] == "critical"


def test_failures_group_by_kind_stage_and_model_with_drill(client):
    body = client.get("/admin/ops/processing/failures").json()
    assert body["durable"] is True and body["group"] == "error_kind"
    keys = {g["key"]: g for g in body["groups"]}
    assert set(keys) == {"rate_limited", "timeout"}
    assert keys["rate_limited"]["count"] == 1 and keys["rate_limited"]["docs"] == 1
    assert body["excluded"]["cancelled_failures"] == 1
    # Legacy groups still cover rows without metadata.failure.
    assert body["legacy"]
    with_cancelled = client.get(
        "/admin/ops/processing/failures", params={"include_cancelled": "true"}
    ).json()
    assert any(g["key"] == "cancelled" for g in with_cancelled["groups"])
    stages = client.get(
        "/admin/ops/processing/failures", params={"group": "stage"}
    ).json()
    assert {g["key"] for g in stages["groups"]} == {"extract", "parse"}
    models = client.get(
        "/admin/ops/processing/failures", params={"group": "model"}
    ).json()
    assert {g["key"] for g in models["groups"]} == {"gpt-4o-mini", ""}

    docs = client.get(
        "/admin/ops/processing/failures/docs",
        params={"group": "error_kind", "key": "rate_limited"},
    ).json()
    [item] = docs["items"]
    assert item["doc_id"] == "doc-aaaa0004" and item["attempt"] == 2
    assert item["current_status"] == "failed"
    assert item["file"]["label"] == "Sen•••.pdf"
    assert docs["source"] == "durable"
    unknown_model = client.get(
        "/admin/ops/processing/failures/docs", params={"group": "model", "key": ""}
    ).json()
    assert [i["doc_id"] for i in unknown_model["items"]] == ["doc-bbbb0002"]
    legacy_ws = client.get(
        "/admin/ops/processing/failures/docs",
        params={"group": "workspace", "key": "space1", "source": "legacy"},
    ).json()
    assert legacy_ws["source"] == "legacy"
    assert [i["doc_id"] for i in legacy_ws["items"]] == ["doc-bbbb0002"]
    durable_ws = client.get(
        "/admin/ops/processing/failures/docs",
        params={"group": "workspace", "key": "space1"},
    ).json()
    assert durable_ws["source"] == "durable"
    assert (
        client.get(
            "/admin/ops/processing/failures/docs",
            params={"group": "workspace", "key": "space1", "source": "x"},
        ).status_code
        == 400
    )
    cancelled = client.get(
        "/admin/ops/processing/failures/docs",
        params={"group": "error_kind", "key": "cancelled"},
    ).json()
    assert [i["doc_id"] for i in cancelled["items"]] == ["doc-aaaa0005"]
    _clean(str(body) + str(docs))


def test_pauses_come_from_persisted_events(client):
    body = client.get("/admin/ops/pauses").json()
    assert body["approximate"] is False and body["source"] == "events"
    [item] = body["items"]
    assert item["workspace"] == "hr_office" and item["byo"] is True
    assert body["by_slot_kind"] == {"extraction": {"budget_exhausted": 1}}
    _clean(str(body))
    table = client.get(
        "/admin/ops/processing/workspaces", params={"has": "paused"}
    ).json()
    assert [i["workspace"] for i in table["items"]] == ["hr_office"]
    assert table["approximate_filters"] == []


def test_backlog_rows_carry_their_last_run(client):
    body = client.get("/admin/ops/processing/workspaces").json()
    last = {i["workspace"]: i["last_run"] for i in body["items"]}
    assert last["fsktm"]["run_id"] == RUN_ID
    assert last["space1"]["status"] == "orphaned"
    assert last["hr_office"] is None


def test_workspace_drawer_recent_runs_and_failure_kinds(client):
    body = client.get("/admin/ops/workspaces/fsktm").json()
    assert [r["run_id"] for r in body["recent_runs"]] == [RUN_ID, DONE_RUN_ID]
    kinds = {k["key"]: k["count"] for k in body["failures_by_kind"]}
    assert kinds == {"rate_limited": 1, "cancelled": 1}
    assert body["runs_available"] is True
    assert body["live"]["run_id"] == RUN_ID


def test_document_drawer_failure_and_attempts(client, store):
    doc = next(r for r in store.doc_rows if r["id"] == "doc-aaaa0004")
    doc["metadata"] = {
        "failure": {
            "stage": "extract",
            "error_kind": "rate_limited",
            "http_status": 429,
            "attempt": 2,
            "first_failed_at": "2026-11-03T01:00:00+00:00",
            "last_run_id": RUN_ID,
            "model": "gpt-4o-mini",
        }
    }
    body = client.get("/admin/ops/workspaces/fsktm/documents/doc-aaaa0004").json()
    assert body["doc"]["failure"]["attempt"] == 2
    assert body["doc"]["failure"]["last_run_id"] == RUN_ID
    [failure] = body["failures"]
    assert failure["error_kind"] == "rate_limited" and failure["run_id"] == RUN_ID
    assert body["failures_available"] is True


def test_email_queue_shows_drainers_and_backoff(client, store):
    store.kv["resync:space1"]["backoff_until"] = time.time() + 30
    store.worker_rows[WORKER_ID]["snapshot"]["drainers"] = [
        {
            "workspace": "space1",
            "state": "backoff",
            "backoff_until": "2026-11-03T02:00:00+00:00",
        }
    ]
    body = client.get("/admin/ops/email-queue").json()
    [item] = body["items"]
    assert item["drainer"]["worker_id"] == WORKER_ID
    assert item["drainer"]["state"] == "backoff"
    assert item["backoff_until"].endswith("+00:00")


def test_events_include_orphaned_runs(client):
    body = client.get("/admin/ops/events").json()
    runs = [e for e in body["items"] if e["kind"] == "run_orphaned"]
    assert [e["detail"]["run_id"] for e in runs] == [ORPHAN_RUN_ID]


def test_legacy_fallback_when_tracking_is_off(client, app):
    app.state.ops.service.run_tracking = False
    active = client.get("/admin/ops/processing/active").json()
    assert active["source"] == "doc_status" and active["approximate"] is True
    stuck = client.get("/admin/ops/processing/stuck").json()
    assert stuck["checked_classes"] == ["stale_docs"] and stuck["orphaned"] == []
    failures = client.get("/admin/ops/processing/failures").json()
    assert failures["durable"] is False and failures["groups"] == []
    pauses = client.get("/admin/ops/pauses").json()
    assert pauses["approximate"] is True
    drawer = client.get("/admin/ops/workspaces/fsktm").json()
    assert drawer["recent_runs"] == [] and drawer["runs_available"] is False
    # Run history still answers from what was recorded.
    assert client.get("/admin/ops/runs").json()["items"]


def test_runs_routes_degrade_without_postgres(shared):
    client = TestClient(build_app(None, kv_storage="JsonKVStorage"))
    body = client.get("/admin/ops/runs").json()
    assert body["available"] is False and body["items"] == []
    body = client.get(f"/admin/ops/runs/{RUN_ID}").json()
    assert body["available"] is False and body["run"] is None


def test_runs_routes_are_admin_only(store, shared):
    from lightrag.api.dependencies import require_admin
    from fastapi import HTTPException

    app = build_app(store)

    def deny():
        raise HTTPException(status_code=403, detail="forbidden")

    app.dependency_overrides[require_admin] = deny
    client = TestClient(app)
    for path in ("/admin/ops/runs", f"/admin/ops/runs/{RUN_ID}"):
        response = client.get(path)
        assert response.status_code == 403
        assert response.headers["cache-control"] == "no-store"


# -- review fixes ------------------------------------------------------------------


def test_a_wedged_owner_is_listed_unresponsive_with_restart_guidance(client, store):
    """Owner heartbeat stale, PID alive (never orphaned): the workspace is
    held and nothing moves, so Stuck lists it and a warning is raised."""
    from datetime import timedelta

    store.worker_rows[WORKER_ID]["heartbeat_at"] = datetime.now(
        timezone.utc
    ) - timedelta(minutes=7)
    active = client.get("/admin/ops/processing/active").json()
    fsktm = next(r for r in active["runs"] if r["workspace"] == "fsktm")
    # One vocabulary with the WebUI: never "owner_lost".
    assert fsktm["state"] == "unresponsive" and fsktm["worker_alive"] is False
    stuck = client.get("/admin/ops/processing/stuck").json()
    [wedged] = stuck["hung"]
    assert wedged["run_id"] == RUN_ID and wedged["state"] == "unresponsive"
    assert wedged["owner_unresponsive"] is True
    assert wedged["guidance"] == "restart_container"
    assert wedged["age_s"] == pytest.approx(420, abs=30)
    overview = client.get("/admin/ops/overview").json()
    rules = {a["rule"]: a for a in overview["attention"]}
    assert rules["worker_unresponsive"]["severity"] == "warning"
    assert "run_hung" not in rules


@pytest.mark.parametrize(
    ("kind", "stage", "hung"),
    [
        ("process", "extract", True),
        ("email_batch", "delete", True),
        ("clear", None, False),
        ("scan", "process", False),
        ("scan", None, True),
    ],
)
def test_hung_only_judges_runs_that_report_progress(client, store, kind, stage, hung):
    row = store.run_rows[RUN_ID]
    row["kind"], row["stage"] = kind, stage
    row["progress_at"] = datetime.fromtimestamp(time.time() - 50 * 60, timezone.utc)
    active = client.get("/admin/ops/processing/active").json()
    fsktm = next(r for r in active["runs"] if r["workspace"] == "fsktm")
    assert fsktm["state"] == ("hung" if hung else "running")
    stuck = client.get("/admin/ops/processing/stuck").json()
    assert [r["run_id"] for r in stuck["hung"]] == ([RUN_ID] if hung else [])


def test_stuck_rows_carry_what_the_console_renders(client, store):
    store.run_rows[ORPHAN_RUN_ID]["orphaned_by"] = "exit_marker"
    body = client.get("/admin/ops/processing/stuck").json()
    [orphan] = body["orphaned"]
    assert orphan["confirmed_by"] == "exit_marker"
    assert orphan["age_s"] == pytest.approx(900, abs=30)
    # Reason codes (translated by the WebUI), never prose to print as-is.
    [busy] = body["busy_without_run"]
    assert busy["reason_code"] == "no_run" and "reason" not in busy
    [held] = body["reservations"]
    assert held["reason_code"] == "held"
    assert held["scanning"] is False and held["pending_enqueues"] == 2


def test_pause_counts_are_never_capped(client, store, monkeypatch):
    from lightrag.api.ops import service as service_module

    now = time.time()
    for i in range(3):
        store.kv[f"pause:ws{i}:extraction"] = {
            "workspace": f"ws{i}",
            "slot": "extraction",
            "kind": "budget_exhausted",
            "since": now - 100 + i,
            "cleared_at": None,
        }
    monkeypatch.setattr(service_module, "OPEN_PAUSES_LIMIT", 2)
    body = client.get("/admin/ops/pauses", params={"page_size": 2, "page": 2}).json()
    # 4 open pauses (hr_office and three more), counted in SQL.
    assert body["total"] == 4 and body["capped"] is True
    assert body["by_slot_kind"] == {"extraction": {"budget_exhausted": 4}}
    # Pages come from SQL too, oldest first: page 2 holds the two newest.
    assert [i["workspace"] for i in body["items"]] == ["ws1", "ws2"]
    summary = client.get("/admin/ops/processing/summary").json()
    assert summary["pauses"]["count"] == 4


def test_email_queue_counts_drains_of_the_last_hour(client, store):
    store.kv["resync:space1"].update(
        {
            "last_attempt_at": time.time() - 30,
            "drains_1h": {"ok": 2, "slot_busy": 5, "error": 1, "window_s": 3600},
        }
    )
    [item] = client.get("/admin/ops/email-queue").json()["items"]
    assert item["drains_1h"] == {"ok": 2, "failed": 1, "slot_busy": 5}
    # Counts older than the hour they cover say nothing about this hour.
    store.kv["resync:space1"]["last_attempt_at"] = time.time() - 7200
    client.app.state.ops.service.cache.clear()
    [item] = client.get("/admin/ops/email-queue").json()["items"]
    assert item["drains_1h"] == {"ok": 0, "failed": 0, "slot_busy": 0}


def test_legacy_workspace_drill_matches_the_legacy_count(client, store):
    """A workspace's legacy group counts rows without metadata.failure; its
    drill lists exactly those."""
    doc = next(r for r in store.doc_rows if r["id"] == "doc-bbbb0002")
    doc["metadata"] = {"failure": {"error_kind": "timeout", "stage": "parse"}}
    body = client.get(
        "/admin/ops/processing/failures", params={"group": "workspace"}
    ).json()
    legacy_counts = {g["key"]: g["count"] for g in body["legacy"]}
    docs = client.get(
        "/admin/ops/processing/failures/docs",
        params={"group": "workspace", "key": "space1", "source": "legacy"},
    ).json()
    assert docs["total"] == legacy_counts.get("space1", 0)
    assert "doc-bbbb0002" not in [i["doc_id"] for i in docs["items"]]
