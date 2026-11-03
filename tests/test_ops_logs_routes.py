"""Logs viewer and correlate routes: keyset paging, follow, filters, entry,
groups, audited export (cap, hashed filters, CSV escaping), correlate and the
drawers' log sections."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from lightrag.api.ops import auth as ops_auth
from lightrag.api.ops import logs as ops_logs
from tests.ops_fixtures import (
    LOG_REQUEST_ID,
    LOG_TRACK_ID,
    RUN_ID,
    FakeRegistry,
    admin,
    build_app,
    ops_routes,
    seed_logs,
    seed_runs,
    seeded_store,
)

pytestmark = pytest.mark.offline

CLIENT = {"X-LightRAG-Client": "webui"}


@pytest.fixture(autouse=True)
def _reset_export_limits():
    ops_routes._export_times.clear()
    yield
    ops_routes._export_times.clear()


def _app(store=None, **kwargs):
    store = store or seeded_store()
    app = build_app(store, **kwargs)
    app.dependency_overrides[ops_auth.require_live_admin] = admin
    return app, store


def _rows(store, rows):
    asyncio.run(store.logs_write(rows))


def _row(ts, **fields):
    return {
        "ts": ts,
        "level": fields.pop("level", 30),
        "logger": fields.pop("logger", "lightrag"),
        "module": fields.pop("module", "pipeline"),
        "lineno": fields.pop("lineno", 10),
        "msg": fields.pop("msg", "line"),
        "fingerprint": fields.pop("fingerprint", "f" * 16),
        **fields,
    }


# -- query and paging ------------------------------------------------------------------


def test_keyset_paging_is_exact_with_equal_timestamps():
    app, store = _app()
    now = time.time()
    same = now - 300.0
    rows = [_row(same, msg=f"same {i}") for i in range(120)]
    rows += [_row(now - 600 - i, msg=f"older {i}") for i in range(130)]
    _rows(store, rows)
    client = TestClient(app)
    seen: list[int] = []
    order: list[tuple[str, int]] = []
    params = {"limit": 100}
    pages = 0
    while True:
        body = client.get("/admin/ops/logs", params=params).json()
        pages += 1
        for item in body["items"]:
            seen.append(item["id"])
            order.append((item["ts"], item["id"]))
        if not body["next_before"]:
            break
        cursor = body["next_before"]
        assert set(cursor) == {"ts", "id"}
        params = {"limit": 100, "before": f"{cursor['ts']},{cursor['id']}"}
    assert pages == 3
    assert len(seen) == len(set(seen)) == 250
    assert order == sorted(order, key=lambda p: (p[0], p[1]), reverse=True)


def test_follow_returns_the_newest_rows_since_a_time():
    app, store = _app()
    now = time.time()
    _rows(store, [_row(now - 100 + i, msg=f"line {i}") for i in range(30)])
    client = TestClient(app)
    head = client.get("/admin/ops/logs", params={"limit": 10}).json()
    assert head["newest_ts"] == head["items"][0]["ts"]
    since = (datetime.now(timezone.utc) - timedelta(seconds=75.5)).isoformat()
    follow = client.get(
        "/admin/ops/logs", params={"after_ts": since, "limit": 500}
    ).json()
    msgs = [i["msg"] for i in follow["items"]]
    assert msgs == [f"line {i}" for i in range(29, 24, -1)]
    assert follow["next_before"] is None
    # A full page: the client restarts from it, and can page older from there.
    full = client.get("/admin/ops/logs", params={"after_ts": since, "limit": 2}).json()
    assert [i["msg"] for i in full["items"]] == ["line 29", "line 28"]
    assert full["next_before"]["id"] == full["items"][-1]["id"]


def test_filters():
    app, store = _app()
    seed_logs(store)
    client = TestClient(app)
    get = lambda **p: client.get("/admin/ops/logs", params=p).json()["items"]  # noqa: E731
    assert len(get()) == 3
    assert [i["level"] for i in get(level="ERROR")] == [40, 40]
    assert [i["workspace"] for i in get(workspace="space1")] == ["space1"]
    assert [i["msg"] for i in get(request_id=LOG_REQUEST_ID)][0].startswith("Failed")
    assert [i["run_id"] for i in get(run_id=RUN_ID)] == [RUN_ID]
    assert len(get(doc_id="doc-aaaa0004")) == 1
    assert len(get(track_id=LOG_TRACK_ID)) == 1
    assert len(get(fingerprint="b2b2b2b2b2b2b2b2")) == 1
    assert [i["logger"] for i in get(module="uvicorn.error")] == ["uvicorn.error"]
    assert [i["module"] for i in get(module="operate")] == ["operate"]
    assert len(get(logger="uvicorn")) == 1
    assert len(get(q="ratelimit")) == 2
    item = get(workspace="fsktm")[0]
    assert item["has_exc"] is True and "exc" not in item
    assert item["level_name"] == "ERROR" and item["repeat_count"] == 1


@pytest.mark.parametrize(
    "params,code",
    [
        ({"level": "loud"}, "invalid_level"),
        ({"run_id": "not-a-run"}, "invalid_run_id"),
        ({"worker_id": "zz"}, "invalid_worker_id"),
        ({"fingerprint": "XYZ"}, "invalid_fingerprint"),
        ({"workspace": "bad ws"}, "invalid_workspace"),
        ({"from": "yesterday"}, "invalid_time"),
        ({"before": "nonsense"}, "invalid_cursor"),
        ({"after_ts": "soon"}, "invalid_after_ts"),
        (
            {"from": "2026-01-01T00:00:00+00:00", "to": "2025-01-01T00:00:00+00:00"},
            "invalid_window",
        ),
    ],
)
def test_bad_parameters_are_refused(params, code):
    app, _ = _app()
    response = TestClient(app).get("/admin/ops/logs", params=params)
    assert response.status_code == 400
    assert response.json()["detail"] == code


def test_window_limits():
    app, _ = _app()
    client = TestClient(app)
    now = datetime.now(timezone.utc)
    eight_days = (now - timedelta(days=8)).isoformat()
    two_days = (now - timedelta(days=2)).isoformat()
    # A client rounding "from" down to the minute still gets its 7 days.
    seven_days = (now - timedelta(days=7, seconds=59)).isoformat()
    assert (
        client.get("/admin/ops/logs", params={"from": eight_days}).json()["detail"]
        == "window_too_long"
    )
    assert client.get("/admin/ops/logs", params={"from": seven_days}).status_code == 200
    assert (
        client.get("/admin/ops/logs", params={"from": two_days, "q": "boom"}).json()[
            "detail"
        ]
        == "search_window_too_long"
    )
    assert client.get("/admin/ops/logs", params={"from": two_days}).status_code == 200


def test_payload_reports_the_sink_and_stays_redacted_on_read():
    app, store = _app()
    # A row written before today's rules (older release): redacted on read.
    _rows(store, [_row(time.time() - 5, msg="legacy line for frank@unimas.my")])
    body = TestClient(app).get("/admin/ops/logs").json()
    assert body["enabled"] is False
    assert body["sink"]["enabled"] is False
    assert body["sink"]["retention_days"] == 14
    assert body["items"][0]["msg"] == "legacy line for <email>"
    assert body["generated_at"] and body["available"] is True


def test_without_postgres_the_logs_routes_answer_unavailable():
    app = build_app(None, kv_storage="JsonKVStorage")
    client = TestClient(app)
    for path in ("/admin/ops/logs", "/admin/ops/logs/groups", "/admin/ops/logs/5"):
        body = client.get(path).json()
        assert body["available"] is False, path
    body = client.get("/admin/ops/correlate", params={"id": "fsktm"}).json()
    assert body == {**body, "available": False, "matches": []}


# -- entry and groups ----------------------------------------------------------------------


def test_entry_has_traceback_and_links():
    app, store = _app()
    ids = seed_logs(store)
    client = TestClient(app)
    body = client.get(f"/admin/ops/logs/{ids[0]}").json()
    entry = body["entry"]
    assert entry["exc"].startswith("Traceback") and body["exc_withheld"] is False
    assert body["links"]["run"] == f"#/?ops=1&otab=processing&orun={RUN_ID}"
    assert body["links"]["doc"].endswith("odoc=fsktm%2Fdoc-aaaa0004")
    # Server links use the WebUI's olq keys, so they open the same filter.
    assert body["links"]["request"].endswith(
        f"otab=logs&olq=req%3D{entry['request_id']}%26w%3D7d"
    )
    assert "olq=fp%3D" in body["links"]["fingerprint"]
    assert client.get("/admin/ops/logs/999999").status_code == 404
    assert client.get("/admin/ops/logs/0").status_code == 422


def test_groups_by_call_site():
    app, store = _app()
    now = time.time()
    _rows(
        store,
        [
            _row(now - 50, fingerprint="1" * 16, msg="first", workspace="a"),
            _row(
                now - 40,
                fingerprint="1" * 16,
                msg="second",
                workspace="b",
                repeat_count=5,
            ),
            _row(now - 30, fingerprint="2" * 16, msg="other", level=40),
        ],
    )
    body = TestClient(app).get("/admin/ops/logs/groups", params={"hours": 1}).json()
    first, second = body["groups"]
    assert first["fingerprint"] == "1" * 16
    assert first["count"] == 6 and first["rows"] == 2 and first["workspaces"] == 2
    assert first["sample"] == "second"
    assert second["level"] == 40
    errors = (
        TestClient(app)
        .get("/admin/ops/logs/groups", params={"hours": 1, "level": "error"})
        .json()
    )
    assert [g["fingerprint"] for g in errors["groups"]] == ["2" * 16]


def test_grouped_search_is_audited_with_hashed_filters():
    app, store = _app()
    client = TestClient(app)
    before = len(store.audit_rows)
    params = {"hours": 168, "workspace": "fsktm", "level": "error"}
    assert client.get("/admin/ops/logs/groups", params=params).status_code == 200
    client.get("/admin/ops/logs/groups", params=params)
    rows = store.audit_rows[before:]
    assert len(rows) == 1
    assert rows[0]["action"] == "log_groups"
    assert rows[0]["params"]["fields"] == ["hours", "level", "workspace"]
    assert "fsktm" not in json.dumps(rows[0], default=str)


# -- search audit -----------------------------------------------------------------------------


def test_searches_are_audited_with_hashed_filters_once_per_window():
    app, store = _app()
    client = TestClient(app)
    before = len(store.audit_rows)
    client.get("/admin/ops/logs", params={"workspace": "fsktm", "q": "secret term"})
    client.get("/admin/ops/logs", params={"workspace": "fsktm", "q": "secret term"})
    rows = store.audit_rows[before:]
    assert len(rows) == 1
    row = rows[0]
    assert row["action"] == "log_search" and row["actor"] == "root"
    assert row["params"]["fields"] == ["level", "q", "workspace"]
    assert len(row["params"]["filter_hash"]) == 16
    text = json.dumps(row, default=str)
    assert "secret term" not in text and "fsktm" not in text
    client.get("/admin/ops/logs", params={"workspace": "space1"})
    assert len(store.audit_rows) == before + 2


def test_failed_search_audit_never_fails_the_search():
    app, store = _app()

    async def broken(row):
        raise ConnectionError("down")

    store.audit_write = broken
    response = TestClient(app).get("/admin/ops/logs")
    assert response.status_code == 200


# -- export ---------------------------------------------------------------------------------------


def _export(client, **body):
    payload = {"reason": "incident review", "filters": {}, **body}
    return client.post("/admin/ops/logs/export", json=payload, headers=CLIENT)


def test_export_ndjson_is_redacted_capped_and_audited():
    app, store = _app()
    now = time.time()
    _rows(
        store,
        [
            _row(now - 100 + i * 0.01, msg=f"line {i}", workspace="fsktm")
            for i in range(30)
        ],
    )
    seed_logs(store)
    client = TestClient(app)
    response = _export(client, filters={"workspace": "fsktm", "q": "line"}, limit=25)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["x-ops-export-rows"] == "25"
    assert 'filename="ops-logs-' in response.headers["content-disposition"]
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert len(lines) == 25 and "exc" not in lines[0]
    assert lines[0]["msg"] == "line 29"
    intent, result = store.audit_rows[-2:]
    assert intent["action"] == result["action"] == "log_export"
    assert (intent["phase"], result["phase"]) == ("intent", "result")
    assert str(intent["id"]) == response.headers["x-ops-audit-id"]
    assert result["params"]["rows"] == 25 and result["outcome"] == "ok"
    assert intent["reason"] == "incident review"
    assert intent["params"]["fields"] == ["level", "q", "workspace"]
    assert "fsktm" not in json.dumps(intent["params"])
    assert intent["params"]["include_exc"] is False


def test_export_never_exceeds_ten_thousand_rows():
    app, store = _app()
    now = time.time()
    _rows(store, [_row(now - 3000 + i * 0.1, msg="x") for i in range(10_050)])
    client = TestClient(app)
    response = _export(client, limit=10_000)
    assert response.headers["x-ops-export-rows"] == "10000"
    assert _export(client, limit=10_001).status_code == 422
    # Tracebacks make rows large: such an export stops lower.
    response = _export(client, limit=10_000, include_exc=True)
    assert response.headers["x-ops-export-rows"] == str(ops_logs.EXPORT_MAX_WITH_EXC)
    intent = [
        r
        for r in store.audit_rows
        if r["action"] == "log_export" and r["params"].get("include_exc")
    ]
    assert intent and intent[0]["params"]["limit"] == ops_logs.EXPORT_MAX_WITH_EXC


def test_export_formats_rows_off_the_event_loop(monkeypatch):
    """Re-redaction and serialisation run in worker threads, page by page."""
    import threading

    app, store = _app()
    now = time.time()
    _rows(store, [_row(now - 100 + i * 0.1, msg="x") for i in range(1500)])
    loop_threads: set[int] = set()
    view_threads: set[int] = set()
    body_threads: set[int] = set()
    real_view, real_body = ops_logs.entry_view, ops_logs.export_body

    def view(row, **kwargs):
        view_threads.add(threading.get_ident())
        return real_view(row, **kwargs)

    def body(*args):
        body_threads.add(threading.get_ident())
        return real_body(*args)

    async def page(*args, **kwargs):
        loop_threads.add(threading.get_ident())
        return await real_page(*args, **kwargs)

    real_page = store.logs_page
    monkeypatch.setattr(ops_logs, "entry_view", view)
    monkeypatch.setattr(ops_logs, "export_body", body)
    monkeypatch.setattr(store, "logs_page", page)
    response = _export(TestClient(app))
    assert response.headers["x-ops-export-rows"] == "1500"
    assert view_threads and body_threads
    assert not (view_threads | body_threads) & loop_threads


def test_export_csv_escapes_formulas_and_includes_tracebacks_on_request():
    app, store = _app()
    now = time.time()
    _rows(
        store,
        [
            _row(now - 10, msg='=HYPERLINK("http://evil","x")', lineno=1),
            _row(now - 9, msg="+cmd|' /C calc'!A0", lineno=2),
            _row(now - 8, msg="@SUM(1+1)", lineno=3, exc="Traceback\nValueError: v"),
            _row(now - 7, msg="-2+3", lineno=4),
        ],
    )
    response = _export(TestClient(app), format="csv", include_exc=True)
    assert response.headers["content-type"].startswith("text/csv")
    table = list(csv.DictReader(io.StringIO(response.text)))
    msgs = [r["msg"] for r in table]
    assert all(m.startswith("'") for m in msgs), msgs
    assert "exc" in table[0]
    by_line = {r["lineno"]: r for r in table}
    assert by_line["3"]["exc"].startswith("Traceback")
    assert ops_logs.csv_cell("=1") == "'=1" and ops_logs.csv_cell("plain") == "plain"
    assert ops_logs.csv_cell("\tx") == "'\tx" and ops_logs.csv_cell(None) == ""


def test_export_is_operator_only_and_needs_the_webui_client(monkeypatch):
    app, _ = _app()
    client = TestClient(app)
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "lead@unimas.my")
    response = _export(client)
    assert response.status_code == 403 and response.json()["detail"] == "not_operator"
    monkeypatch.delenv("OPS_OPERATOR_ACCOUNTS")
    response = client.post("/admin/ops/logs/export", json={"reason": "incident review"})
    assert response.status_code == 403
    assert response.json()["detail"] == "client_header_required"
    assert _export(client, reason="x").status_code == 422
    assert _export(client, filters={"nope": "1"}).json()["detail"] == "invalid_filter"


def test_export_without_an_intent_audit_is_refused():
    app, store = _app()

    async def broken(row):
        raise ConnectionError("down")

    store.audit_write = broken
    response = _export(TestClient(app))
    assert response.status_code == 503
    assert response.json()["detail"] == "audit_unavailable"


def test_exports_are_rate_limited_per_actor():
    app, _ = _app()
    client = TestClient(app)
    codes = [_export(client).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]


def test_export_routes_depend_on_operator_and_live_admin():
    route = next(
        r for r in ops_routes.router.routes if r.path == "/admin/ops/logs/export"
    )
    names = {getattr(d.call, "__name__", "") for d in route.dependant.dependencies}
    assert {"require_operator", "require_live_admin", "require_ops_client"} <= names


# -- correlate --------------------------------------------------------------------------------------


@pytest.fixture
def correlate_app():
    store = seeded_store()
    store.doc_rows[3]["track_id"] = LOG_TRACK_ID  # doc-aaaa0004
    seed_runs(store)
    seed_logs(store)
    app, _ = _app(store, registry=FakeRegistry({"fsktm": "a.bin@unimas.my"}))
    return TestClient(app), store


def _matches(client, value):
    body = client.get("/admin/ops/correlate", params={"id": value}).json()
    return body, [(m["type"], m["id"], m["workspace"]) for m in body["matches"]]


def test_correlate_each_id_type(correlate_app):
    client, store = correlate_app
    _, found = _matches(client, LOG_REQUEST_ID)
    assert found == [("request", LOG_REQUEST_ID, "fsktm")]
    body, found = _matches(client, RUN_ID)
    assert found == [("run", RUN_ID, "fsktm")]
    assert body["matches"][0]["link"].endswith(f"orun={RUN_ID}")
    _, found = _matches(client, LOG_TRACK_ID)
    assert found == [("track", LOG_TRACK_ID, "fsktm")]
    body, found = _matches(client, "doc-aaaa0004")
    assert found == [("doc", "doc-aaaa0004", "fsktm")]
    label = body["matches"][0]["label"]
    assert label.startswith("failed") and "Senate" not in label and "•••" in label
    _, found = _matches(client, "fsktm/doc-aaaa0007")
    assert found == [("doc", "doc-aaaa0007", "fsktm")]
    _, found = _matches(client, "space1")
    assert found == [("workspace", "space1", "space1")]
    _, found = _matches(client, "1")
    assert found and found[0][0] == "log"
    _, found = _matches(client, "nothing_here")
    assert found == []


def test_correlate_owner_email_is_masked(correlate_app):
    client, store = correlate_app
    body, found = _matches(client, "a.bin@unimas.my")
    assert found == [("owner", "fsktm", "fsktm")]
    assert body["query"] == "a.•••@unimas.my"
    assert body["matches"][0]["label"] == "a.•••@unimas.my"
    assert "a.bin@unimas.my" not in json.dumps(body)
    # An email that is not in the registry, but whose derived workspace exists.
    store.add_doc("s_two_unimas_my", "doc-zz", "processed")
    _, found = _matches(client, "s.two@unimas.my")
    assert found == [("owner", "s_two_unimas_my", "s_two_unimas_my")]


def test_correlate_is_audited_with_a_hashed_id(correlate_app):
    client, store = correlate_app
    before = len(store.audit_rows)
    _matches(client, "a.bin@unimas.my")
    row = store.audit_rows[before]
    assert row["action"] == "correlate" and row["params"]["fields"] == ["id"]
    assert "a.bin" not in json.dumps(row, default=str)
    assert (
        client.get("/admin/ops/correlate", params={"id": "bad\x01"}).json()["detail"]
        == "invalid_id"
    )
    assert client.get("/admin/ops/correlate").json()["matches"] == []


def test_correlate_malformed_uuid_is_not_a_server_error(correlate_app):
    client, _ = correlate_app
    value = "0123456789abcdef0123456789abcde-----"
    assert len(value) == 36
    assert "run" not in ops_logs.classify_input(value)
    response = client.get("/admin/ops/correlate", params={"id": value})
    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_server_log_links_decode_with_the_webui_keys():
    """OLQ_KEYS must match URL_KEYS in the WebUI's logs.ts."""
    import re
    from pathlib import Path
    from urllib.parse import parse_qs, unquote, urlsplit

    source = (
        Path(__file__).resolve().parents[1]
        / "lightrag_webui/src/features/Operations/logs.ts"
    ).read_text(encoding="utf-8")
    block = re.search(r"const URL_KEYS[^{]*\{(?P<body>[^}]*)\}", source)
    assert block is not None
    webui = dict(re.findall(r"(\w+): '(\w+)'", block.group("body")))
    assert webui == ops_logs.OLQ_KEYS
    link = ops_logs._logs_link(track_id=LOG_TRACK_ID)
    olq = parse_qs(urlsplit(link.replace("#/", "", 1)).query)["olq"][0]
    assert parse_qs(unquote(olq)) == {"trk": [LOG_TRACK_ID], "w": ["7d"]}


def test_correlate_input_kinds():
    assert ops_logs.classify_input("a@b.co") == ["owner"]
    assert ops_logs.classify_input("ws/doc-1") == ["ws_doc"]
    assert ops_logs.classify_input("upload_20261103_101010_aaaa0004") == ["track"]
    assert ops_logs.classify_input("doc-abc") == ["doc"]
    assert ops_logs.classify_input(RUN_ID) == ["run", "request", "workspace"]
    assert ops_logs.classify_input("42") == ["log", "workspace"]
    assert ops_logs.classify_input("fsktm") == ["workspace"]


# -- drawers -------------------------------------------------------------------------------------------


def test_drawers_carry_their_log_sections():
    store = seeded_store()
    seed_runs(store)
    seed_logs(store)
    app, _ = _app(store)
    app.state.ops.service.run_tracking = True
    client = TestClient(app)
    run = client.get(f"/admin/ops/runs/{RUN_ID}").json()
    assert run["log_count"] == 1
    # The count covers the same 7 days as the section's /logs query.
    assert ops_logs.ID_LOOKBACK_DAYS * 86400 == ops_logs.MAX_WINDOW_S
    doc = client.get("/admin/ops/workspaces/fsktm/documents/doc-aaaa0004").json()
    assert doc["logs"]["available"] is True and doc["logs"]["count"] == 1
    assert doc["logs"]["first"][0]["doc_id"] == "doc-aaaa0004"
    ws = client.get("/admin/ops/workspaces/fsktm").json()
    assert ws["errors_1h"] == 1
    status = client.get("/admin/ops/status").json()
    assert status["features"]["logs"] is True
    assert status["features"]["log_sink"] is False


def test_workspace_deletion_removes_its_log_lines():
    store = seeded_store()
    seed_logs(store)
    asyncio.run(store.forget_workspace("fsktm"))
    assert {r["workspace"] for r in store.log_rows} == {"space1", None}
