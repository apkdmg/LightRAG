"""Operations console: Overview and Processing routes (read-only, SQL-first)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lightrag.api.insights.store import MemoryUsageStore
from lightrag.api.ops.service import OpsCache
from lightrag.api.ops.store import failure_signature
from lightrag.kg import shared_storage
from lightrag.usage import UsageRecord
from tests.ops_fixtures import (
    RATE_LIMIT_ERROR,
    SECRET_EMAIL,
    SECRET_KEY,
    FakeRegistry,
    build_app,
    seeded_store,
    shared_state,
)

pytestmark = pytest.mark.offline

_SECRETS = (
    SECRET_EMAIL,
    SECRET_KEY,
    "Senate minutes",
    "900101",
    "other@unimas.my",
    "sk-other",
)


def _assert_clean(text: str) -> None:
    for secret in _SECRETS:
        assert secret not in text, secret


@pytest.fixture
def shared(monkeypatch):
    data = shared_state()
    monkeypatch.setattr(shared_storage, "_shared_dicts", data)
    monkeypatch.setattr(shared_storage, "_is_multiprocess", False)
    return data


@pytest.fixture
def store():
    return seeded_store()


@pytest.fixture
def client(store, shared):
    registry = FakeRegistry({"fsktm": "a.bin@unimas.my"})
    return TestClient(build_app(store, registry=registry))


def test_summary_counts_exclude_dups_and_split_cancellations(client):
    body = client.get("/admin/ops/processing/summary").json()
    assert body["available"] is True and body["approximate"] is True
    assert body["backlog"] == {
        "pending": 2,
        "parsing": 1,
        "analyzing": 0,
        "processing": 2,
        "failed": 2,
        "cancelled": 1,
        "dup": 1,
        "partial": False,
        "inflight": 5,
        "failed_24h": 2,
        "cancelled_24h": 1,
        "oldest_inflight_at": body["backlog"]["oldest_inflight_at"],
        "oldest_pending_at": body["backlog"]["oldest_pending_at"],
    }
    assert body["backlog"]["oldest_inflight_at"].endswith("+00:00")
    assert body["stuck"] == {"runs": 0, "docs": 1, "reservations": 0}
    assert body["email_queue"]["total"] == 3 and body["email_queue"]["bad"] == 1
    assert body["pauses"]["count"] == 1
    assert body["pauses"]["by_slot_kind"] == {"extraction": {"budget": 1}}
    assert body["active_count"] == 2


def test_overview_state_kpis_attention(client):
    response = client.get("/admin/ops/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "degraded"
    assert body["worst"] == "critical"
    rules = {a["rule"]: a for a in body["attention"]}
    assert rules["provider_paused"]["severity"] == "critical"
    assert rules["stale_docs"]["severity"] == "serious"
    assert rules["oldest_pending"]["severity"] == "warning"
    assert rules["email_queue_bad"]["severity"] == "warning"
    assert rules["email_queue_stalled"]["severity"] == "warning"
    for item in body["attention"]:
        assert item["link"].startswith("#/?ops=1&otab=")
        assert item["runbook"]
    assert body["attention"][0]["severity"] == "critical"
    kpis = body["kpis"]
    assert kpis["backlog"]["value"] == 5
    assert kpis["failed_24h"] == {
        **kpis["failed_24h"],
        "value": 2,
        "cancelled": 1,
        "dup": 1,
    }
    assert kpis["stuck"]["value"] == 1 and kpis["stuck"]["state"] == "serious"
    assert (
        kpis["email_queue"]["value"] == 3 and kpis["email_queue"]["state"] == "warning"
    )
    assert kpis["active"]["value"] == 2 and kpis["active"]["approximate"] is True
    assert kpis["llm"]["state"] == "unknown"  # Insights off
    assert kpis["deps"]["state"] == "unknown"
    assert kpis["workers"]["state"] == "ok"
    assert all(t["scope"] == "host" and t["updated_at"] for t in kpis.values())
    assert [r["workspace"] for r in body["active_top"]][:2] == ["space1", "fsktm"]
    kinds = [e["kind"] for e in body["timeline"]["events"]]
    assert kinds == ["ops_action", "deployment"]
    assert body["timeline"]["events"][1]["detail"] == {"version": "0295"}
    assert body["generated_at"].endswith("+00:00")
    _assert_clean(response.text)


def test_overview_single_tenant_without_default_rag(store, shared):
    app = build_app(store, default_rag=False)
    assert not hasattr(app.state, "default_rag")
    response = TestClient(app).get("/admin/ops/overview")
    assert response.status_code == 200
    assert response.json()["state"] == "degraded"


def test_overview_without_postgres_reports_unknown(shared):
    app = build_app(None, kv_storage="JsonKVStorage")
    body = TestClient(app).get("/admin/ops/overview").json()
    assert body["state"] == "unknown"
    assert body["unavailable_reason"] == "postgres_required"
    assert body["attention"] == []
    summary = TestClient(app).get("/admin/ops/processing/summary").json()
    assert summary["available"] is False and summary["backlog"] is None


def test_overview_reports_postgres_down(store, shared):
    store.fail_reads = ConnectionError(
        "connection refused to 10.0.0.1 password=hunter2"
    )
    response = TestClient(build_app(store)).get("/admin/ops/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "down"
    assert body["attention"][0]["rule"] == "pg_probe"
    assert body["postgres_ok"] is False
    assert "hunter2" not in response.text


def test_routes_answer_503_when_queries_fail(store, shared):
    store.fail_reads = TimeoutError("statement timeout")
    client = TestClient(build_app(store))
    for path in (
        "/admin/ops/processing/summary",
        "/admin/ops/processing/active",
        "/admin/ops/processing/stuck",
        "/admin/ops/processing/workspaces",
        "/admin/ops/processing/failures",
        "/admin/ops/email-queue",
        "/admin/ops/events",
    ):
        response = client.get(path)
        assert response.status_code == 503, path
        assert response.json()["detail"] == "data_unavailable"
        assert response.headers["cache-control"] == "no-store"


def test_active_jobs_are_approximate_and_masked(client):
    response = client.get("/admin/ops/processing/active")
    body = response.json()
    assert body["approximate"] is True and body["source"] == "doc_status"
    runs = {r["workspace"]: r for r in body["runs"]}
    assert set(runs) == {"fsktm", "space1"}
    fsktm = runs["fsktm"]
    assert fsktm["state"] == "running"
    assert fsktm["live"]["busy"] is True
    assert fsktm["docs_total"] == 4 and fsktm["docs_done"] == 1
    assert fsktm["docs_inflight"] == {"parsing": 0, "analyzing": 0, "processing": 2}
    assert fsktm["stage"] == "processing"
    assert fsktm["started_at"].endswith("+00:00")
    assert fsktm["label"] and "Senate" not in fsktm["label"]
    _assert_clean(response.text)


def test_active_jobs_when_the_manager_is_unresponsive(client, monkeypatch):
    from lightrag.api.ops.snapshot import NamespaceReader, SnapshotBatch

    async def stuck(self, *args, **kwargs):
        return SnapshotBatch(responsive=False)

    monkeypatch.setattr(NamespaceReader, "snapshot_many", stuck)
    body = client.get("/admin/ops/processing/active").json()
    assert body["manager"] == "unresponsive"
    assert {r["state"] for r in body["runs"]} == {"unresponsive"}
    assert all(r["live"] is None for r in body["runs"])


def test_stuck_lists_stale_docs_with_thresholds(client):
    response = client.get("/admin/ops/processing/stuck")
    body = response.json()
    assert body["orphaned"] == [] and body["hung"] == []
    assert body["busy_without_run"] == [] and body["reservations"] == []
    assert body["checked_classes"] == ["stale_docs"]
    assert body["thresholds"] == {
        "doc_minutes": 60,
        "hung_minutes": 15,
        "hung_serious_minutes": 45,
        "orphan_seconds": 60,
        "reservation_minutes": 5,
        "busy_without_run_minutes": 2,
    }
    [item] = body["stale_docs"]["items"]
    assert item["workspace"] == "fsktm" and item["doc_id"] == "doc-aaaa0002"
    assert item["age_s"] > 180 * 60
    assert item["file"]["label"] == "Sen•••.pdf"
    assert body["stale_docs"]["total"] == 1
    _assert_clean(response.text)
    wide = client.get("/admin/ops/processing/stuck?doc_older_than_min=1").json()
    assert wide["stale_docs"]["total"] == 3
    assert wide["thresholds"]["doc_minutes"] == 1


def test_thresholds_come_from_kv(store, shared):
    store.kv["thresholds"] = {"stale_doc_minutes": 1, "bogus": 5, "hung_minutes": "x"}
    body = TestClient(build_app(store)).get("/admin/ops/processing/stuck").json()
    assert body["thresholds"]["doc_minutes"] == 1
    assert body["thresholds"]["hung_minutes"] == 15


def test_backlog_table_search_sort_filter_page(client):
    body = client.get("/admin/ops/processing/workspaces").json()
    assert body["sort"] == "oldest" and body["dir"] == "asc"
    assert [i["workspace"] for i in body["items"]] == ["fsktm", "hr_office", "space1"]
    fsktm = body["items"][0]
    assert fsktm["owner"] == "a.bin@unimas.my"
    assert (
        fsktm["pending"],
        fsktm["processing"],
        fsktm["failed"],
        fsktm["cancelled"],
    ) == (1, 2, 1, 1)
    assert fsktm["stuck"] is True and fsktm["state"] == "stale_doc"
    paused = body["items"][1]
    assert paused["paused"] is True and paused["state"] == "paused"
    space1 = body["items"][2]
    assert space1["queue_pending"] == 3

    by_owner = client.get("/admin/ops/processing/workspaces?q=S.ONE").json()
    assert [i["workspace"] for i in by_owner["items"]] == ["space1"]
    failed = client.get(
        "/admin/ops/processing/workspaces?has=failed&sort=name&dir=desc"
    ).json()
    assert [i["workspace"] for i in failed["items"]] == ["space1", "fsktm"]
    email = client.get("/admin/ops/processing/workspaces?has=email").json()
    assert [i["workspace"] for i in email["items"]] == ["space1"]
    paused_only = client.get("/admin/ops/processing/workspaces?has=paused").json()
    assert [i["workspace"] for i in paused_only["items"]] == ["hr_office"]
    stuck = client.get("/admin/ops/processing/workspaces?has=stuck").json()
    assert [i["workspace"] for i in stuck["items"]] == ["fsktm"]
    paged = client.get(
        "/admin/ops/processing/workspaces?sort=name&page=2&page_size=2"
    ).json()
    assert paged["total"] == 3 and paged["page"] == 2
    assert [i["workspace"] for i in paged["items"]] == ["space1"]


def test_backlog_table_validation(client):
    assert (
        client.get("/admin/ops/processing/workspaces?page_size=201").status_code == 422
    )
    assert (
        client.get("/admin/ops/processing/workspaces?dir=sideways").status_code == 422
    )
    bad_sort = client.get("/admin/ops/processing/workspaces?sort=error_msg")
    assert bad_sort.status_code == 400 and bad_sort.json()["detail"] == "invalid_sort"
    bad_has = client.get("/admin/ops/processing/workspaces?has=failed,drop_table")
    assert bad_has.status_code == 400 and bad_has.json()["detail"] == "invalid_filter"


def test_failures_group_legacy_messages(client):
    response = client.get("/admin/ops/processing/failures")
    body = response.json()
    assert body["durable"] is False and body["groups"] == []
    assert body["group"] == "error_kind" and body["legacy_group"] == "signature"
    assert body["excluded"] == {"dup": 1, "cancelled": 1}
    [group] = body["legacy"]
    assert group["count"] == 2 and group["workspaces"] == 2
    assert group["key"] == failure_signature(
        RATE_LIMIT_ERROR.format(request="ffff0000")
    )
    assert group["example"].startswith("RateLimitError: Error code: 429")
    _assert_clean(response.text)

    with_cancelled = client.get(
        "/admin/ops/processing/failures?include_cancelled=true"
    ).json()
    assert [g["key"] for g in with_cancelled["legacy"]][-1] == "cancelled"
    by_ws = client.get(
        "/admin/ops/processing/failures?group=workspace&window=1h"
    ).json()
    assert sorted(g["key"] for g in by_ws["legacy"]) == ["fsktm", "space1"]
    assert (
        client.get("/admin/ops/processing/failures?window=2d").json()["detail"]
        == "invalid_window"
    )
    assert client.get("/admin/ops/processing/failures?group=sql").status_code == 400


def test_failure_docs_are_masked(client):
    key = client.get("/admin/ops/processing/failures").json()["legacy"][0]["key"]
    response = client.get(f"/admin/ops/processing/failures/docs?key={key}")
    body = response.json()
    assert body["total"] == 2
    assert {i["doc_id"] for i in body["items"]} == {"doc-aaaa0004", "doc-bbbb0002"}
    _assert_clean(response.text)
    cancelled = client.get(
        "/admin/ops/processing/failures/docs?key=cancelled&group=cancelled"
    ).json()
    assert [i["doc_id"] for i in cancelled["items"]] == ["doc-aaaa0005"]
    assert (
        client.get("/admin/ops/processing/failures/docs").json()["detail"]
        == "invalid_key"
    )


def test_email_queue(client):
    body = client.get("/admin/ops/email-queue").json()
    assert body["total"] == 3 and body["bad"] == 1 and body["workspaces"] == 1
    assert body["stale"] is False and body["scanned_at"].endswith("+00:00")
    [item] = body["items"]
    assert item["workspace"] == "space1"
    assert item["attempts"] == 7 and item["last_result_class"] == "slot_busy"
    assert item["pending_since"].endswith("+00:00")
    assert item["pending_age_s"] >= 2999
    assert body["longest_pending_s"] >= 3000


def test_email_queue_before_the_first_scan(store, shared):
    del store.kv["email_queue_summary"]
    body = TestClient(build_app(store)).get("/admin/ops/email-queue").json()
    assert body["items"] == [] and body["scanned_at"] is None and body["stale"] is True


class FakeProviderStore:
    def __init__(self, own: dict[str, set[str]]):
        self.own = own
        self.decrypts = 0

    def record_mtime(self, workspace):
        return 1.0 if workspace in self.own else None

    def get(self, workspace):
        self.decrypts += 1
        slots = self.own[workspace]
        return SimpleNamespace(
            slot=lambda name: SimpleNamespace(is_active=lambda: name in slots)
        )


def test_pauses_grouped_by_slot_and_byo_flagged(store, shared):
    provider_store = FakeProviderStore({"hr_office": {"extraction"}})
    client = TestClient(build_app(store, provider_store=provider_store))
    response = client.get("/admin/ops/pauses")
    body = response.json()
    assert body["approximate"] is True and body["total"] == 1
    assert body["by_slot_kind"] == {"extraction": {"budget": 1}}
    [item] = body["items"]
    assert item["workspace"] == "hr_office" and item["byo"] is True
    assert item["status_code"] == 402 and item["slot"] == "extraction"
    assert item["since"].endswith("+00:00") and item["retry_due"] is False
    _assert_clean(response.text)
    client.get("/admin/ops/pauses")
    assert provider_store.decrypts == 1  # cached, and the record did not change


async def test_pause_candidates_are_capped(store, shared):
    from lightrag.api.ops import pauses
    from lightrag.api.ops.snapshot import NamespaceReader

    for i in range(80):
        store.add_doc(f"ws{i:03d}", f"doc-{i}", "pending")
    seen = []

    class Reader(NamespaceReader):
        async def snapshot_many(self, workspaces, keys=(), **kwargs):
            seen.extend(workspaces)
            return await super().snapshot_many(workspaces, keys, **kwargs)

    data = await pauses.read_pauses(store, Reader(), namespace_of=lambda w: w)
    assert len(seen) == pauses.MAX_CANDIDATES
    assert data["capped"] is True


def test_workspace_drawer(client):
    response = client.get("/admin/ops/workspaces/fsktm")
    body = response.json()
    assert response.status_code == 200
    assert body["owner"] == "a.bin@unimas.my"
    assert body["idle"] is False and body["live"]["busy"] is True
    assert body["live"]["latest_message_redacted"]
    assert body["counts"]["statuses"]["processing"] == 2
    assert body["counts"]["cancelled"] == 1 and body["counts"]["dup"] == 1
    assert body["loaded_in"] == [] and body["recent_runs"] == []
    assert body["email_queue"]["pending"] == 0
    _assert_clean(response.text)
    space1 = client.get("/admin/ops/workspaces/space1").json()
    assert (
        space1["email_queue"]["pending"] == 3 and space1["email_queue"]["attempts"] == 7
    )


def test_workspace_drawer_idle_and_unknown(client, store):
    store.add_doc("quiet_ws", "doc-q1", "processed")
    body = client.get("/admin/ops/workspaces/quiet_ws").json()
    assert body["idle"] is True and body["live"] is None
    assert body["counts"]["statuses"] == {"processed": 1}
    missing = client.get("/admin/ops/workspaces/nobody_here")
    assert missing.status_code == 404 and missing.json()["detail"] == "not_found"
    assert (
        client.get("/admin/ops/workspaces/bad%20name").json()["detail"]
        == "invalid_workspace"
    )


def test_workspace_drawer_shows_an_action_in_progress(client, store):
    store.kv["action:fsktm"] = {
        "actor": "khairil",
        "action": "run",
        "since": time.time(),
        "exp": time.time() + 60,
    }
    body = client.get("/admin/ops/workspaces/fsktm").json()
    assert body["action_in_progress"]["actor"] == "khairil"


def test_feed_is_redacted_and_detects_resets(client, shared):
    body = client.get("/admin/ops/workspaces/fsktm/feed").json()
    assert body["total"] == 5 and body["start"] == 0 and body["next_offset"] == 5
    assert body["reset"] is False and len(body["lines"]) == 5
    assert SECRET_EMAIL not in str(body["lines"])
    shared["fsktm:pipeline_status"]["history_messages"].append("msg new")
    follow = client.get("/admin/ops/workspaces/fsktm/feed?offset=5").json()
    assert follow["lines"] == ["msg new"] and follow["reset"] is False
    # A new run clears the list: the client's offset is now past the end.
    shared["fsktm:pipeline_status"]["history_messages"][:] = ["fresh start"]
    reset = client.get("/admin/ops/workspaces/fsktm/feed?offset=6").json()
    assert reset["reset"] is True and reset["lines"] == ["fresh start"]
    idle = client.get("/admin/ops/workspaces/never_loaded/feed").json()
    assert idle["loaded"] is False and idle["lines"] == []
    assert client.get("/admin/ops/workspaces/fsktm/feed?limit=201").status_code == 422


def test_documents_list_masks_file_names(client):
    response = client.get("/admin/ops/workspaces/fsktm/documents?page_size=3")
    body = response.json()
    assert body["total"] == 6 and len(body["items"]) == 3
    _assert_clean(response.text)
    failed = client.get("/admin/ops/workspaces/fsktm/documents?status=failed").json()
    assert {(i["doc_id"], i["cancelled"]) for i in failed["items"]} == {
        ("doc-aaaa0004", False),
        ("doc-aaaa0005", True),
    }
    assert (
        client.get("/admin/ops/workspaces/fsktm/documents?status=gone").status_code
        == 400
    )


def test_document_drawer(store, shared):
    insights_store = MemoryUsageStore()
    now = time.time()
    insights_store.call_rows = [
        UsageRecord(
            workspace="fsktm",
            purpose="extract",
            operation="extract_entities",
            model="gpt-x",
            prompt_tokens=1204,
            latency_ms=8102,
            status="error",
            doc_id="doc-aaaa0007",
            ts=now - 18500,
        ),
        UsageRecord(
            workspace="fsktm", purpose="extract", operation="x", doc_id="other", ts=now
        ),
    ]
    app = build_app(store, insights=SimpleNamespace(store=insights_store))
    response = TestClient(app).get("/admin/ops/workspaces/fsktm/documents/doc-aaaa0007")
    assert response.status_code == 200
    body = response.json()
    doc = body["doc"]
    assert doc["status"] == "processed" and doc["file"]["label"] == "Sen•••.pdf"
    assert [s["id"] for s in doc["stages"]] == [
        "enqueued",
        "parse",
        "analyze",
        "extract",
    ]
    stages = {s["id"]: s for s in doc["stages"]}
    assert stages["parse"]["duration_s"] == 100 and stages["parse"]["state"] == "done"
    assert stages["analyze"]["state"] == "skipped"
    assert stages["extract"]["duration_s"] == 900
    assert doc["failure"] is None and body["failures"] == []
    assert body["calls_available"] is True
    [call] = body["calls"]
    assert call["latency_ms"] == 8102 and call["status"] == "error"
    _assert_clean(response.text)
    failed = (
        TestClient(app).get("/admin/ops/workspaces/fsktm/documents/doc-aaaa0004").json()
    )
    assert failed["doc"]["error_redacted"].startswith("RateLimitError")
    assert SECRET_KEY not in str(failed)
    missing = TestClient(app).get("/admin/ops/workspaces/fsktm/documents/doc-none")
    assert missing.status_code == 404


def test_events_merge_usage_and_audit(client):
    body = client.get("/admin/ops/events").json()
    assert [e["kind"] for e in body["items"]] == ["ops_action", "deployment"]
    assert body["items"][0]["detail"] == {
        "action": "cancel",
        "outcome": "cancellation_requested",
        "actor": "khairil",
    }
    only = client.get("/admin/ops/events?kinds=deployment").json()
    assert [e["kind"] for e in only["items"]] == ["deployment"]
    assert (
        client.get("/admin/ops/events?kinds=everything").json()["detail"]
        == "invalid_kind"
    )
    too_long = client.get(
        "/admin/ops/events?from=2026-01-01T00:00:00Z&to=2026-02-01T00:00:00Z"
    )
    assert too_long.json()["detail"] == "invalid_range"
    assert (
        client.get("/admin/ops/events?from=yesterday").json()["detail"]
        == "invalid_time"
    )


def test_responses_are_cached_per_worker(store, shared):
    calls = {"n": 0}
    original = store.backlog_totals

    async def counted():
        calls["n"] += 1
        return await original()

    store.backlog_totals = counted
    client = TestClient(build_app(store))
    for _ in range(5):
        assert client.get("/admin/ops/processing/summary").status_code == 200
    assert calls["n"] == 1


async def test_cache_is_bounded_single_flight_and_skips_errors():
    cache = OpsCache(max_entries=3)
    started = asyncio.Event()
    release = asyncio.Event()
    runs = {"n": 0}

    async def slow():
        runs["n"] += 1
        started.set()
        await release.wait()
        return 42

    first = asyncio.create_task(cache.get(("k",), 10, slow))
    await started.wait()
    second = asyncio.create_task(cache.get(("k",), 10, slow))
    await asyncio.sleep(0)
    release.set()
    assert await first == 42 and await second == 42
    assert runs["n"] == 1

    async def boom():
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError):
        await cache.get(("e",), 10, boom)
    with pytest.raises(RuntimeError):
        await cache.get(("e",), 10, boom)  # not cached

    for i in range(10):
        await cache.get(("n", i), 10, lambda i=i: asyncio.sleep(0, result=i))
    assert len(cache) == 3


async def test_cache_expires():
    now = [0.0]
    cache = OpsCache(clock=lambda: now[0])
    values = iter([1, 2])

    async def compute():
        return next(values)

    assert await cache.get(("k",), 5, compute) == 1
    now[0] = 4.9
    assert await cache.get(("k",), 5, compute) == 1
    now[0] = 5.1
    assert await cache.get(("k",), 5, compute) == 2


def test_unexpected_errors_never_leak_their_text(store, shared, monkeypatch):
    from lightrag.api.ops.service import OpsService

    async def explode(self, *args, **kwargs):
        raise RuntimeError(f"bad thing with {SECRET_KEY} for {SECRET_EMAIL}")

    monkeypatch.setattr(OpsService, "summary", explode)
    monkeypatch.setattr(OpsService, "workspace", explode)
    client = TestClient(build_app(store), raise_server_exceptions=False)
    for path in ("/admin/ops/processing/summary", "/admin/ops/workspaces/fsktm"):
        response = client.get(path)
        assert response.status_code == 500
        assert response.json() == {
            "detail": "internal_error",
            "request_id": response.headers["x-request-id"],
        }
        _assert_clean(response.text)


# -- review fixes ---------------------------------------------------------------------


async def test_cache_leader_cancellation_is_not_passed_to_followers():
    cache = OpsCache()
    started = asyncio.Event()
    release = asyncio.Event()
    runs = {"n": 0}

    async def slow():
        runs["n"] += 1
        started.set()
        await release.wait()
        return runs["n"]

    leader = asyncio.create_task(cache.get(("k",), 10, slow))
    await started.wait()
    follower = asyncio.create_task(cache.get(("k",), 10, slow))
    await asyncio.sleep(0)
    leader.cancel()
    await asyncio.sleep(0)
    release.set()
    # The follower was not cancelled: it computes the value itself.
    assert await follower == 2
    assert leader.cancelled() and not follower.cancelled()
    assert await cache.get(("k",), 10, slow) == 2  # cached from the follower

    # A follower's own cancellation still cancels it, and only it.
    started.clear()
    release.clear()
    lead = asyncio.create_task(cache.get(("j",), 10, slow))
    await started.wait()
    follow = asyncio.create_task(cache.get(("j",), 10, slow))
    await asyncio.sleep(0)
    follow.cancel()
    release.set()
    assert await lead == 3
    with pytest.raises(asyncio.CancelledError):
        await follow


def test_backlog_search_accepts_the_longest_workspace_id(client):
    longest = "w" * 255
    response = client.get(f"/admin/ops/processing/workspaces?q={longest}")
    assert response.status_code == 200 and response.json()["total"] == 0
    assert (
        client.get(f"/admin/ops/processing/workspaces?q={longest}x").status_code == 422
    )


def test_email_queue_and_events_need_only_the_ops_tables(store, shared):
    """KV on Postgres, doc status elsewhere: the queue and timeline still answer."""
    store.doc_status_ready = False
    client = TestClient(build_app(store))
    queue = client.get("/admin/ops/email-queue").json()
    assert queue["available"] is True and queue["total"] == 3
    assert [i["workspace"] for i in queue["items"]] == ["space1"]
    events = client.get("/admin/ops/events").json()
    assert events["available"] is True
    assert [e["kind"] for e in events["items"]] == ["ops_action", "deployment"]
    summary = client.get("/admin/ops/processing/summary").json()
    assert summary["available"] is False


def test_failure_docs_for_a_workspace_named_cancelled(store, shared):
    store.add_doc(
        "cancelled",
        "doc-dddd0001",
        "failed",
        error_msg="KeyError: 'x'",
    )
    client = TestClient(build_app(store))
    by_ws = client.get("/admin/ops/processing/failures?group=workspace").json()
    row = next(g for g in by_ws["legacy"] if g["key"] == "cancelled")
    assert row["count"] == 1
    body = client.get(
        "/admin/ops/processing/failures/docs?group=workspace&key=cancelled"
    ).json()
    # The workspace's own failure, not every cancellation on the host.
    assert [i["doc_id"] for i in body["items"]] == ["doc-dddd0001"]


_SPACED = "Minit Mesyuarat Jabatan Keselamatan.pdf"
_PERSON = "Laporan Kewangan Ali bin Abu 2024.docx"
_DATA = "senarai_pelajar_ahmad_bin_ali.json"
_NAME_PARTS = ("Mesyuarat", "Jabatan", "Kewangan", "Ali bin Abu", "pelajar_ahmad")


def _assert_names_hidden(text: str) -> None:
    for part in _NAME_PARTS:
        assert part not in text, part


def test_pipeline_lines_never_show_file_names(store, shared):
    """The pipeline's real line shapes, with spaces and unlisted extensions."""
    fsktm = shared["fsktm:pipeline_status"]
    fsktm["latest_message"] = f"User cancelled 1/4: inputs/user1/{_SPACED}"
    fsktm["history_messages"][:] = [
        f"Extracting stage 1/3: {_SPACED}",
        f"Failed to extract document 2/3: {_PERSON}",
        f"Merging stage failed in document 3/3: {_DATA}",
        f"Duplicate content skipped after parsing: {_PERSON}; archived to "
        f"/data/archive/{_PERSON}",
        f"Deleting doc-aaaa0001 {_SPACED}(previous status: FAILED)",
        f"Failed to delete 1/2: doc-aaaa0002[{_DATA}] - storage error",
        "Processed entity batch 1/3: 500 records",
    ]
    store.add_doc(
        "fsktm",
        "doc-eeee0001",
        "failed",
        file_path=f"inputs/{_PERSON}",
        error_msg=f"Failed to extract document 1/1: inputs/{_PERSON}",
        updated_at=store._utc_now(),
    )
    store.add_doc(
        "fsktm",
        "doc-eeee0002",
        "failed",
        file_path=_DATA,
        error_msg=f"ValueError: could not parse {_DATA} at row 3",
        updated_at=store._utc_now(),
    )
    client = TestClient(build_app(store))
    feed = client.get("/admin/ops/workspaces/fsktm/feed").json()
    _assert_names_hidden(str(feed))
    assert feed["lines"][0] == "Extracting stage 1/3: Min•••an.pdf"
    assert feed["lines"][-1] == "Processed entity batch 1/3: 500 records"
    assert (
        feed["lines"][5]
        == "Failed to delete 1/2: doc-aaaa0002[sen•••li.json] - storage error"
    )
    for path in (
        "/admin/ops/processing/active",
        "/admin/ops/workspaces/fsktm",
        "/admin/ops/workspaces/fsktm/documents?status=failed",
        "/admin/ops/workspaces/fsktm/documents/doc-eeee0001",
        "/admin/ops/workspaces/fsktm/documents/doc-eeee0002",
        "/admin/ops/processing/failures?group=workspace",
        "/admin/ops/processing/failures",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        _assert_names_hidden(response.text)
    drawer = client.get("/admin/ops/workspaces/fsktm").json()
    assert (
        drawer["live"]["latest_message_redacted"] == "User cancelled 1/4: Min•••an.pdf"
    )
    doc = client.get("/admin/ops/workspaces/fsktm/documents/doc-eeee0002").json()
    assert (
        doc["doc"]["error_redacted"]
        == "ValueError: could not parse sen•••li.json at row 3"
    )
    signature = failure_signature(f"ValueError: could not parse {_DATA} at row 3")
    docs = client.get(
        f"/admin/ops/processing/failures/docs?group=signature&key={signature}"
    )
    assert docs.json()["total"] == 1
    _assert_names_hidden(docs.text)


def test_a_slow_count_degrades_the_overview_without_marking_postgres_down(
    store, shared, monkeypatch
):
    async def timeout():
        raise TimeoutError("canceling statement due to statement timeout")

    monkeypatch.setattr(store, "cancelled_count", timeout)
    body = TestClient(build_app(store)).get("/admin/ops/overview").json()
    assert body["state"] != "down"
    assert "pg_probe" not in {a["rule"] for a in body["attention"]}
    assert body["failed_sections"] == ["backlog_counts"]
    # The backlog tile still has its numbers.
    assert body["kpis"]["backlog"]["value"] == 5
    assert body["kpis"]["failed_24h"]["value"] == 2


def test_a_failed_stuck_read_is_an_unknown_tile_not_zero(store, shared, monkeypatch):
    async def timeout(*args, **kwargs):
        raise TimeoutError("canceling statement due to statement timeout")

    monkeypatch.setattr(store, "stale_docs", timeout)
    body = TestClient(build_app(store)).get("/admin/ops/overview").json()
    assert body["state"] != "down"
    assert body["kpis"]["stuck"] == {
        **body["kpis"]["stuck"],
        "value": None,
        "runs": None,
        "docs": None,
        "reservations": None,
        "state": "unknown",
    }


def test_email_stall_rule_reads_time_since_the_last_drain(store, shared):
    summary = store.kv["email_queue_summary"]
    # Waiting for 50 min, but a drain was seen 2 min ago: not stalled.
    summary["longest_stalled_s"] = 120.0
    client = TestClient(build_app(store))
    rules = {a["rule"] for a in client.get("/admin/ops/overview").json()["attention"]}
    assert "email_queue_stalled" not in rules
    # No drain seen for 3 h: serious, and said to be approximate.
    summary["longest_stalled_s"] = 3 * 3600.0
    client = TestClient(build_app(store))
    attention = client.get("/admin/ops/overview").json()["attention"]
    stalled = next(a for a in attention if a["rule"] == "email_queue_stalled")
    assert stalled["severity"] == "serious"
    assert stalled["params"]["basis"] == "no_drain_seen"
    assert stalled["params"]["approximate"] is True


def test_gunicorn_worker_count_skips_the_manager_process(monkeypatch):
    import psutil

    from tests.ops_fixtures import ops_routes

    class Child:
        def __init__(self, pid, status="running"):
            self.pid = pid
            self._status = status

        def status(self):
            return self._status

    class Master:
        def __init__(self, pid):
            pass

        def children(self, recursive=False):
            return [Child(11), Child(12), Child(13, psutil.STATUS_ZOMBIE), Child(99)]

    monkeypatch.setattr(psutil, "Process", Master)
    monkeypatch.setitem(ops_routes._workers_live_cache, "at", 0.0)
    monkeypatch.setattr(shared_storage, "manager_pid", lambda: 99)
    # Two live workers (one zombie, one Manager): a dead worker shows.
    assert ops_routes._count_gunicorn_workers() == 2
    monkeypatch.setitem(ops_routes._workers_live_cache, "at", 0.0)
    monkeypatch.setattr(shared_storage, "manager_pid", lambda: None)
    # Without the Manager's PID the count is unknown, not one too high.
    assert ops_routes._count_gunicorn_workers() is None
