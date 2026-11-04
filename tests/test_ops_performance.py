"""Performance tab and Sign-in read models (PR-6): Q5 LLM latency, the
metric-backed views, concurrency from heartbeats, throughput, top workspaces,
the hourly LLM p95 baseline and the rules that read these metrics."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lightrag.api.insights.store import MemoryUsageStore
from lightrag.api.ops import perf
from lightrag.api.ops.metrics import BUCKETS, bucket_index
from lightrag.api.ops.probes import PassiveLlm
from lightrag.api.ops.store import MemoryOpsStore
from lightrag.usage import UsageRecord
from tests.ops_fixtures import WORKER_ID, build_app

pytestmark = pytest.mark.offline

DEFAULTS = {**perf.METRIC_THRESHOLDS, "llm_min_calls": 20}

OTHER = "fedcba9876543210fedcba9876543210"


class Held:
    def __init__(self, held: bool = True):
        self.held = held

    async def still_held(self) -> bool:
        return self.held


def _call(ts: float, **kw) -> UsageRecord:
    kw.setdefault("workspace", "fsktm")
    kw.setdefault("purpose", "extract")
    kw.setdefault("model", "gpt-4o-mini")
    return UsageRecord(operation="x", ts=ts, **kw)


def _metric(
    minute: float, series: str, key: str, latencies, errors=0, worker=WORKER_ID
):
    buckets = [0] * BUCKETS
    for ms in latencies:
        buckets[bucket_index(ms)] += 1
    return {
        "minute": int(minute) // 60 * 60,
        "worker_id": worker,
        "series": series,
        "key": key,
        "count": len(latencies),
        "errors": errors,
        "sum_ms": int(sum(latencies)),
        "max_ms": int(max(latencies, default=0)),
        "buckets": buckets,
    }


def _counts(minute: float, key: str, count: int, errors: int = 0):
    return {
        "minute": int(minute) // 60 * 60,
        "worker_id": WORKER_ID,
        "series": "auth",
        "key": key,
        "count": count,
        "errors": errors,
        "sum_ms": 0,
        "max_ms": 0,
        "buckets": [0] * BUCKETS,
    }


def _app(store=None, usage=None):
    insights = SimpleNamespace(store=usage) if usage is not None else None
    return build_app(
        store if store is not None else MemoryOpsStore(), insights=insights
    )


# -- Q5 on the shared usage store ----------------------------------------------------------


async def test_q5_percentiles_by_purpose_and_model_exclude_cache_hits():
    usage = MemoryUsageStore()
    now = time.time()
    usage.call_rows = [
        _call(now - 30, latency_ms=100),
        _call(now - 40, latency_ms=300, status="error"),
        _call(now - 50, latency_ms=500, status="truncated"),
        _call(now - 60, latency_ms=99999, cache_hit=True),  # not LLM latency
        _call(now - 70, latency_ms=2000, purpose="query", model="gpt-4o"),
        _call(now - 7200, latency_ms=50),  # outside the window
    ]
    end = datetime.fromtimestamp(now, timezone.utc)
    start = end - timedelta(hours=1)
    rows = await usage.latency_percentiles(start, end, group="purpose")
    by_key = {r["key"]: r for r in rows}
    assert by_key["extract"]["calls"] == 3
    assert by_key["extract"]["errors"] == 1 and by_key["extract"]["truncated"] == 1
    assert by_key["extract"]["p50_ms"] == 300.0
    assert by_key["extract"]["p99_ms"] < 99999
    assert by_key["query"]["p95_ms"] == 2000.0
    by_model = {
        r["key"]: r for r in await usage.latency_percentiles(start, end, group="model")
    }
    assert set(by_model) == {"gpt-4o-mini", "gpt-4o"}
    series = await usage.latency_percentiles(start, end, group="purpose", bucket_s=60)
    assert all(r["t"] % 60 == 0 for r in series)
    assert sum(r["calls"] for r in series) == 4
    with pytest.raises(ValueError):
        await usage.latency_percentiles(start, end, group="workspace; DROP")


async def test_baseline_reads_the_same_hour_on_earlier_days():
    usage = MemoryUsageStore()
    hour = (time.time() // 3600) * 3600
    usage.call_rows = [
        _call(hour - d * 86400 + 600, latency_ms=1000 * d) for d in range(1, 8)
    ] + [_call(hour - 86400 + 4000, latency_ms=999_999)]  # the next hour
    starts = [
        datetime.fromtimestamp(hour - d * 86400, timezone.utc) for d in range(1, 8)
    ]
    out = await usage.latency_baseline(starts)
    assert out["calls"] == 7 and 6000 < out["p95_ms"] <= 7000


# -- baseline job and its use ----------------------------------------------------------------


async def test_baseline_cycle_stores_the_current_hour_under_the_lease():
    usage = MemoryUsageStore()
    store = MemoryOpsStore()
    now = time.time()
    hour = (now // 3600) * 3600
    usage.call_rows = [
        _call(hour - d * 86400 + 60 + i, latency_ms=2000 + i)
        for d in range(1, 8)
        for i in range(5)
    ]
    entry = await perf.baseline_cycle(
        Held(), store=store, usage_store=lambda: usage, clock=lambda: now
    )
    assert entry["calls"] == 35 and entry["p95_ms"] >= 2000
    stored = await store.kv_get(perf.BASELINE_KEY)
    key = str(datetime.fromtimestamp(now, timezone.utc).hour)
    assert stored["hours"][key]["calls"] == 35
    assert perf.baseline_for(stored, now)["p95_ms"] == entry["p95_ms"]
    # Lost lease: nothing written. No Insights: nothing to compute.
    other = MemoryOpsStore()
    await perf.baseline_cycle(
        Held(False), store=other, usage_store=lambda: usage, clock=lambda: now
    )
    assert await other.kv_get(perf.BASELINE_KEY) is None
    assert (
        await perf.baseline_cycle(Held(), store=store, usage_store=lambda: None) is None
    )


def test_baseline_needs_volume_and_freshness():
    now = time.time()
    hour = str(datetime.fromtimestamp(now, timezone.utc).hour)

    def value(calls, age):
        return {
            "hours": {hour: {"p95_ms": 900.0, "calls": calls, "computed_at": now - age}}
        }

    assert perf.baseline_for(value(50, 60), now)["p95_ms"] == 900.0
    assert perf.baseline_for(value(5, 60), now) is None
    assert perf.baseline_for(value(50, 9 * 86400), now) is None
    assert perf.baseline_for(None, now) is None


@pytest.mark.parametrize(
    "p95,baseline,severity",
    [
        (1500.0, 1000.0, None),
        (2500.0, 1000.0, "warning"),
        (4500.0, 1000.0, "serious"),
        (61_000.0, None, "serious"),
        (30_000.0, None, None),
    ],
)
def test_llm_p95_rule_thresholds(p95, baseline, severity):
    assert perf.llm_p95_severity(p95, baseline) == severity


async def test_passive_llm_card_warns_against_the_baseline():
    now = time.time()
    hour = str(datetime.fromtimestamp(now, timezone.utc).hour)
    store = MemoryOpsStore()
    await store.kv_put(
        perf.BASELINE_KEY,
        {"hours": {hour: {"p95_ms": 1000.0, "calls": 500, "computed_at": now}}},
    )
    usage = MemoryUsageStore()
    usage.call_rows = [_call(now - 10 - i, latency_ms=3000) for i in range(30)]
    probe = PassiveLlm(
        SimpleNamespace(insights=SimpleNamespace(store=usage)),
        "openai",
        None,
        store=store,
    )
    result = await probe(5.0)
    assert result.status == "warn"
    assert (
        result.detail["baseline_p95_ms"] == 1000.0 and result.detail["p95_ratio"] == 3.0
    )
    usage.call_rows = [_call(now - 10 - i, latency_ms=1100) for i in range(30)]
    probe._baseline_at = 0
    assert (await probe(5.0)).status == "ok"


def test_metric_rules():
    items = []

    def add(rule, severity, title, link, **kw):
        items.append((kw.get("item_id") or rule, severity))

    perf.metric_attention(
        add,
        {
            "http": {"count": 200, "errors": 12},
            "deps": {
                "embed": {"count": 100, "errors": 3},
                "rerank": {"count": 10, "errors": 9},
            },
            "signin": {"sso_errors": 25, "lockouts": 0, "token_401": 3},
            "llm_baseline": {"p95_ms": 1000.0},
        },
        {"available": True, "calls": 40, "p95_ms": 2500.0},
        DEFAULTS,
        lambda **kw: "#/",
    )
    assert dict(items) == {
        "llm_p95_15m": "warning",
        "http_5xx_rate_5m": "serious",
        "embed_rerank_error_rate_15m:embed": "warning",
        "signin_failures_15m": "serious",
    }
    items.clear()
    perf.metric_attention(
        add,
        {"http": {"count": 50, "errors": 50}, "deps": {}, "signin": {}},
        {"available": True, "calls": 5, "p95_ms": 90_000.0},
        DEFAULTS,
        lambda **kw: "#/",
    )
    assert items == []  # below the volume floors


def test_metric_rules_follow_threshold_overrides():
    items = []

    def add(rule, severity, title, link, **kw):
        items.append((kw.get("item_id") or rule, severity))

    inputs = {
        "http": {"count": 200, "errors": 12},
        "deps": {"embed": {"count": 100, "errors": 3}},
        "signin": {"sso_errors": 25, "lockouts": 0},
        "llm_baseline": {"p95_ms": 1000.0},
    }
    llm = {"available": True, "calls": 40, "p95_ms": 2500.0}
    looser = {
        **DEFAULTS,
        "llm_p95_ratio_warn": 3.0,
        "http_5xx_warn": 0.1,
        "dep_error_min_calls": 500,
        "signin_sso_warn": 30,
    }
    perf.metric_attention(add, inputs, llm, looser, lambda **kw: "#/")
    assert items == []
    stricter = {**DEFAULTS, "http_5xx_serious": 0.5, "signin_serious": 100}
    perf.metric_attention(add, inputs, llm, stricter, lambda **kw: "#/")
    assert dict(items)["http_5xx_rate_5m"] == "warning"
    assert dict(items)["signin_failures_15m"] == "warning"
    assert perf.llm_p95_severity(2500.0, 1000.0, {"llm_p95_ratio_serious": 2.2}) == (
        "serious"
    )


async def test_kv_thresholds_reach_the_metric_rules():
    store = await _seeded_metric_store()
    app = _app(store)
    service = app.state.ops.service
    merged = await service.thresholds()
    assert merged["signin_sso_warn"] == 5 and merged["http_5xx_min_requests"] == 100
    snapshot = await service.compute_attention_snapshot(
        {"mode": "uvicorn", "expected": 1, "live": 1}, WORKER_ID
    )
    assert "signin_failures_15m" in {i["rule"] for i in snapshot["items"]}
    # 10 SSO errors in 15 min: above the default 5, below an override of 50.
    await store.kv_put("thresholds", {"signin_sso_warn": 50, "http_5xx_warn": "x"})
    service.cache.forget(("thresholds",))
    merged = await service.thresholds()
    assert merged["signin_sso_warn"] == 50 and merged["http_5xx_warn"] == 0.01
    snapshot = await service.compute_attention_snapshot(
        {"mode": "uvicorn", "expected": 1, "live": 1}, WORKER_ID
    )
    assert "signin_failures_15m" not in {i["rule"] for i in snapshot["items"]}
    body = TestClient(app).get("/admin/ops/signin", params={"window": "15m"}).json()
    assert body["state"] == "ok" and body["thresholds"]["signin_sso_warn"] == 50


# -- routes ---------------------------------------------------------------------------------


def test_llm_route_percentiles_and_7d_hourly():
    usage = MemoryUsageStore()
    now = time.time()
    records = [_call(now - 60 * i, latency_ms=100 + i) for i in range(30)]
    records.append(_call(now - 120, latency_ms=10, purpose="query", status="error"))
    # write() stores the calls and their hourly rollup (read by the 7 d window).
    asyncio.run(usage.write(records, []))
    client = TestClient(_app(usage=usage))
    body = client.get("/admin/ops/performance/llm", params={"window": "1h"}).json()
    assert body["insights"] is True and body["approx_7d"] is False
    table = {r["key"]: r for r in body["table"]}
    assert table["extract"]["calls"] == 30 and table["query"]["error_rate"] == 1.0
    assert table["extract"]["p95"] is not None and body["series"]
    assert set(body["queue"]) == {"running", "in_flight", "queued"}
    week = client.get(
        "/admin/ops/performance/llm", params={"window": "7d", "group": "model"}
    ).json()
    assert week["approx_7d"] is True
    assert {r["key"] for r in week["table"]} == {"gpt-4o-mini"}
    assert "avg" in week["table"][0] and "max" in week["table"][0]
    assert "p95" not in week["table"][0]  # percentiles need ≤24 h
    bad = client.get("/admin/ops/performance/llm", params={"window": "2y"})
    assert bad.status_code == 400 and bad.json()["detail"] == "invalid_window"
    assert (
        client.get("/admin/ops/performance/llm", params={"group": "x"}).json()["detail"]
        == "invalid_group"
    )


def test_llm_route_without_insights():
    body = TestClient(_app()).get("/admin/ops/performance/llm").json()
    assert body["insights"] is False and body["table"] == []


async def _seeded_metric_store() -> MemoryOpsStore:
    store = MemoryOpsStore()
    now = time.time()
    # Minutes the views read: every worker has flushed them.
    end = perf.complete_end(now)
    rows = []
    for m in range(1, 11):
        minute = end - m * 60
        rows.append(
            _metric(minute, "http", "GET /documents/paginated", [20, 40, 80], errors=0)
        )
        rows.append(_metric(minute, "http", "<unmatched>", [2]))
        rows.append(_metric(minute, "http", "POST /query", [900, 1500], errors=1))
        rows.append(_metric(minute, "embed", "bge-m3", [30, 35, 40, 45], errors=1))
        rows.append(_metric(minute, "rerank", "bge-reranker", [60]))
        rows.append(_counts(minute, "sso_error", 1, 1))
        rows.append(_counts(minute, "token_401", 2, 2))
    rows.append(_counts(now - 2 * 3600, "lockout", 4, 4))  # older than 15 min
    await store.metrics_write(rows)
    return store


async def test_http_route_aggregates_routes_and_series():
    client = TestClient(_app(await _seeded_metric_store()))
    body = client.get("/admin/ops/performance/http", params={"window": "15m"}).json()
    routes = {r["route"]: r for r in body["routes"]}
    assert routes["POST /query"]["count"] == 20
    assert routes["POST /query"]["err5xx"] == 0.5
    assert routes["POST /query"]["method"] == "POST"
    assert routes["<unmatched>"]["method"] is None
    assert routes["GET /documents/paginated"]["p95"] <= 80
    assert body["top_p95"][0] == "POST /query"
    assert len(body["series"]) == 10 and body["series"][0]["rps"] > 0
    assert body["totals"]["count"] == 60 and body["enabled"] is True
    assert body["available"] is True


async def test_deps_route_series_models_and_vllm():
    store = await _seeded_metric_store()
    store.kv["probe:embed"] = {
        "status": "ok",
        "checked_at": "2026-11-03T00:00:00+00:00",
        "detail": {"running": 3, "waiting": 25, "kv_cache_pct": 41.5},
    }
    app = _app(store)
    app.state.ops.service.probes = SimpleNamespace(
        read_latest=lambda: _latest(store), latest={}, cycle=None
    )
    body = (
        TestClient(app)
        .get("/admin/ops/performance/deps", params={"window": "1h"})
        .json()
    )
    assert body["totals"]["embed"]["count"] == 40
    assert body["totals"]["embed"]["error_rate"] == 0.25
    assert body["summary"] == body["totals"]
    assert body["models"]["rerank"][0]["model"] == "bge-reranker"
    assert len(body["series"]["embed"]) >= 2
    assert body["vllm"]["embed"]["waiting"] == 25
    assert body["vllm"]["rerank"] is None


async def _latest(store):
    return {"embed": store.kv["probe:embed"]}, None


async def test_signin_route_counts_outcomes_only():
    client = TestClient(_app(await _seeded_metric_store()))
    body = client.get("/admin/ops/signin", params={"window": "24h"}).json()
    assert (
        body["sso_errors"] == 10 and body["token_401"] == 20 and body["lockouts"] == 4
    )
    assert body["last_15m"]["lockouts"] == 0 and body["last_15m"]["sso_errors"] == 10
    assert body["series"]
    text = str(body)
    assert "@" not in text


def test_concurrency_sums_live_heartbeats_without_loading_workspaces(monkeypatch):
    monkeypatch.setenv("OPS_PROVIDER_CONCURRENCY_BUDGET", "40")
    store = MemoryOpsStore()
    now = datetime.now(timezone.utc)
    for worker, pid, age, queued in (
        (WORKER_ID, 1, 5, 7),
        (OTHER, 2, 6, 3),
        ("ab" * 16, 3, 900, 99),
    ):
        store.worker_rows[worker] = {
            "worker_id": worker,
            "host": "h",
            "pid": pid,
            "pid_create_time": 1.0,
            "mode": "gunicorn",
            "version": "x",
            "started_at": now - timedelta(hours=1),
            "heartbeat_at": now - timedelta(seconds=age),
            "exited_at": None,
            "superseded": False,
            "leases_held": [],
            "snapshot": {
                "instances": 4,
                "queues": {
                    "llm": {"running": 2, "in_flight": 5, "queued": queued},
                    "embed": {"running": 1, "in_flight": 1, "queued": 0},
                    "rerank": None,
                },
                "llm_roles": {
                    "extract": {"running": 2, "in_flight": 5, "queued": queued}
                },
                "top_instances": [
                    {"workspace": f"ws{pid}", "running": 2, "queued": queued}
                ],
            },
        }
    body = TestClient(_app(store)).get("/admin/ops/performance/concurrency").json()
    assert body["roles"]["llm"] == {"running": 4, "in_flight": 10, "queued": 10}
    assert body["roles"]["rerank"] is None
    assert body["llm_roles"]["extract"]["queued"] == 10
    assert [t["workspace"] for t in body["top"]] == [
        "ws1",
        "ws2",
    ]  # dead worker left out
    assert body["workers"] == 2 and body["instances"] == 8
    assert body["budget"] == {"limit": 40, "in_flight": 10, "ratio": 0.25}


def test_throughput_and_top_workspaces():
    store = MemoryOpsStore()
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    store.run_rows = {
        "r1": {
            "run_id": "r1",
            "workspace": "fsktm",
            "kind": "process",
            "status": "succeeded",
            "started_at": hour_ago - timedelta(minutes=30),
            "finished_at": hour_ago,
            "docs_done": 12,
        },
        "r2": {
            "run_id": "r2",
            "workspace": "space1",
            "kind": "process",
            "status": "running",
            "started_at": now - timedelta(minutes=10),
            "finished_at": None,
            "docs_done": 3,
        },
        "r3": {
            "run_id": "r3",
            "workspace": "fsktm",
            "kind": "email_batch",
            "status": "failed",
            "started_at": hour_ago,
            "finished_at": hour_ago,
            "docs_done": 0,
        },
    }
    store.failure_rows = [
        {"ts": hour_ago, "workspace": "fsktm", "error_kind": "rate_limit"},
        {"ts": hour_ago, "workspace": "fsktm", "error_kind": "rate_limit"},
        {"ts": hour_ago, "workspace": "space1", "error_kind": "cancelled"},
    ]
    usage = MemoryUsageStore()
    usage.call_rows = [_call(now.timestamp() - 30, workspace="hr") for _ in range(3)]
    client = TestClient(_app(store, usage))
    body = client.get("/admin/ops/performance/throughput", params={"days": 1}).json()
    assert sum(d["processed"] for d in body["docs"]) == 12
    assert sum(d["failed"] for d in body["docs"]) == 2  # cancellations excluded
    assert body["email_batches"][0] == {
        "t": body["email_batches"][0]["t"],
        "batches": 1,
        "failed": 1,
    }
    assert sum(p["calls"] for p in body["llm_calls_per_min"]) == 3
    assert (
        client.get("/admin/ops/performance/throughput", params={"days": 99}).status_code
        == 422
    )
    top = client.get(
        "/admin/ops/performance/top-workspaces",
        params={"metric": "processing_minutes", "window": "24h"},
    ).json()
    assert [i["workspace"] for i in top["items"]] == ["fsktm", "space1"]
    assert top["items"][0]["value"] == pytest.approx(30.0, abs=0.1)
    failures = client.get(
        "/admin/ops/performance/top-workspaces", params={"metric": "failures"}
    ).json()
    assert failures["items"] == [{"workspace": "fsktm", "value": 2.0}]
    calls = client.get(
        "/admin/ops/performance/top-workspaces",
        params={"metric": "llm_calls", "window": "1h"},
    ).json()
    assert calls["items"] == [{"workspace": "hr", "value": 3}]
    bad = client.get("/admin/ops/performance/top-workspaces", params={"metric": "x"})
    assert bad.status_code == 400 and bad.json()["detail"] == "invalid_metric"


def test_performance_routes_without_postgres_answer_unavailable():
    client = TestClient(build_app(None, kv_storage="JsonKVStorage"))
    for path in (
        "/admin/ops/performance/http",
        "/admin/ops/performance/deps",
        "/admin/ops/performance/throughput",
        "/admin/ops/performance/top-workspaces",
        "/admin/ops/performance/llm",
        "/admin/ops/performance/concurrency",
        "/admin/ops/signin",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.json()["available"] is False, path


async def test_reads_are_cached_per_window():
    store = await _seeded_metric_store()
    app = _app(store)
    client = TestClient(app)
    calls = {"n": 0}
    original = store.metrics_read

    async def counting(*args, **kwargs):
        calls["n"] += 1
        return await original(*args, **kwargs)

    store.metrics_read = counting
    for _ in range(3):
        client.get("/admin/ops/performance/http", params={"window": "15m"})
    assert calls["n"] == 1
    client.get("/admin/ops/performance/http", params={"window": "1h"})
    assert calls["n"] == 2


async def test_failing_database_answers_503():
    store = await _seeded_metric_store()
    store.fail_reads = RuntimeError("db down")
    response = TestClient(_app(store)).get("/admin/ops/performance/http")
    assert response.status_code == 503
    assert response.json()["detail"] == "data_unavailable"


async def test_attention_snapshot_includes_metric_rules():
    store = await _seeded_metric_store()
    now = time.time()
    # A route of its own: minute rows are insert-only per (minute, worker, key).
    end = perf.complete_end(now)
    rows = [
        _metric(end - m * 60, "http", "POST /documents/upload", [100] * 50, errors=10)
        for m in range(1, 4)
    ]
    await store.metrics_write(rows)
    app = _app(store)
    service = app.state.ops.service
    snapshot = await service.compute_attention_snapshot(
        {"mode": "uvicorn", "expected": 1, "live": 1}, WORKER_ID
    )
    rules = {item["rule"] for item in snapshot["items"]}
    assert "http_5xx_rate_5m" in rules
    assert "signin_failures_15m" in rules
    runbook = {i["rule"]: i["runbook"] for i in snapshot["items"]}
    assert runbook["http_5xx_rate_5m"]


# -- review fixes: rates, vLLM and queue history, the Sign-in card ---------------------------


async def test_rates_divide_by_the_time_the_rows_cover():
    """60 requests in each of the 15 complete minutes read 1.0 rps, not less:
    the current, unflushed minute is not part of the window."""
    store = MemoryOpsStore()
    now = time.time()
    end = perf.complete_end(now)
    rows = [_metric(end - m * 60, "http", "GET /x", [10] * 60) for m in range(1, 16)]
    rows += [_metric(end - m * 60, "embed", "bge-m3", [5] * 30) for m in range(1, 16)]
    await store.metrics_write(rows)
    client = TestClient(_app(store))
    body = client.get("/admin/ops/performance/http", params={"window": "15m"}).json()
    assert body["span_s"] == 900
    assert body["totals"]["rps"] == pytest.approx(1.0)
    assert body["routes"][0]["rps"] == pytest.approx(1.0)
    assert {p["rps"] for p in body["series"]} == {1.0}
    deps = client.get("/admin/ops/performance/deps", params={"window": "15m"}).json()
    assert deps["totals"]["embed"]["per_min"] == pytest.approx(30.0)
    assert {p["per_min"] for p in deps["series"]["embed"]} == {30.0}
    # 24 h starts on an hour: the denominator is the covered span, not 86400.
    day = client.get("/admin/ops/performance/http", params={"window": "24h"}).json()
    assert 86400 <= day["span_s"] < 86400 + 3600
    assert day["totals"]["rps"] == pytest.approx(900 / day["span_s"], abs=1e-4)


def test_covered_seconds_of_partial_buckets():
    assert perf.covered_s(0, 300, 120, 3600) == 180
    assert perf.covered_s(3300, 300, 0, 3480) == 180
    assert perf.covered_s(600, 300, 0, 3600) == 300


async def test_deps_series_carry_vllm_history_from_probe_rows():
    store = await _seeded_metric_store()
    end = perf.complete_end(time.time())
    for m, (running, waiting, kv) in enumerate(
        ((2, 0, 10.0), (4, 30, 55.5), (3, "x", None)), start=1
    ):
        detail = {"running": running, "waiting": waiting}
        if kv is not None:
            detail["kv_cache_pct"] = kv
        store.probe_rows.append(
            {
                "ts": datetime.fromtimestamp(end - m * 60 + 5, timezone.utc),
                "dependency": "embed",
                "status": "ok",
                "latency_ms": 5,
                "error_class": None,
                "detail": detail,
            }
        )
    # Rows outside the window and of other dependencies are not read.
    store.probe_rows.append(
        {
            "ts": datetime.fromtimestamp(end - 7200, timezone.utc),
            "dependency": "embed",
            "status": "ok",
            "latency_ms": 5,
            "error_class": None,
            "detail": {"running": 99, "waiting": 99},
        }
    )
    body = (
        TestClient(_app(store))
        .get("/admin/ops/performance/deps", params={"window": "15m"})
        .json()
    )
    points = {p["t"] // 1000: p for p in body["series"]["embed"]}
    second = points[int(end - 120)]
    assert second["running"] == 4 and second["waiting"] == 30
    assert second["kv_cache"] == 0.555 and second["count"] == 4
    third = points[int(end - 180)]
    assert third["waiting"] is None and third["kv_cache"] is None
    assert all(p.get("waiting") != 99 for p in body["series"]["embed"])
    assert all("running" not in p for p in body["series"]["rerank"])


async def test_deps_view_stands_when_the_vllm_history_fails():
    store = await _seeded_metric_store()

    async def broken(*args, **kwargs):
        raise RuntimeError("probe table missing")

    store.vllm_series = broken
    response = TestClient(_app(store)).get("/admin/ops/performance/deps")
    assert response.status_code == 200
    assert response.json()["totals"]["embed"]["count"] == 40


async def test_llm_queue_series_sums_workers_samples():
    store = MemoryOpsStore()
    now = time.time()
    minute = int(now // 60 * 60)
    for worker, queued in ((WORKER_ID, 7), (OTHER, 3)):
        for m, running in ((1, 2), (2, 5)):
            store.sample_rows[(worker, minute - m * 60)] = {
                "worker_id": worker,
                "minute": minute - m * 60,
                "llm_running": running,
                "llm_in_flight": running + 1,
                "llm_queued": queued * m,
            }
    # A sample from before PR-6 (no queue columns) adds nothing.
    store.sample_rows[("ab" * 16, minute - 60)] = {
        "worker_id": "ab" * 16,
        "minute": minute - 60,
        "cpu_pct": 3.0,
    }
    body = (
        TestClient(_app(store))
        .get("/admin/ops/performance/llm", params={"window": "15m"})
        .json()
    )
    assert body["insights"] is False  # the queue history needs no usage store
    by_t = {p["t"] // 1000: p for p in body["queue_series"]}
    assert by_t[minute - 60] == {
        "t": (minute - 60) * 1000,
        "running": 4,
        "in_flight": 6,
        "queued": 10,
    }
    assert by_t[minute - 120]["queued"] == 20 and by_t[minute - 120]["running"] == 10
    # One worker's maximum per bucket, then the sum over workers.
    hour = (
        TestClient(_app(store))
        .get("/admin/ops/performance/llm", params={"window": "1h"})
        .json()["queue_series"]
    )
    assert sum(p["queued"] for p in hour) <= 30


async def test_heartbeat_sample_carries_the_llm_queue():
    store = MemoryOpsStore()
    row = {
        "worker_id": WORKER_ID,
        "host": "h",
        "pid": 1,
        "mode": "uvicorn",
        "started_at": datetime.now(timezone.utc),
    }
    await store.worker_beat(
        row,
        {
            "cpu_pct": 1.0,
            "llm_running": 3,
            "llm_in_flight": 4,
            "llm_queued": 9,
        },
    )
    (sample,) = store.sample_rows.values()
    assert (sample["llm_running"], sample["llm_queued"]) == (3, 9)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    rows = await store.llm_queue_series(end - timedelta(hours=1), end, 60)
    assert rows[0]["queued"] == 9 and rows[0]["in_flight"] == 4


async def test_signin_card_in_health_overview_and_drawer():
    store = await _seeded_metric_store()
    client = TestClient(_app(store))
    deps = {d["id"]: d for d in client.get("/admin/ops/health").json()["deps"]}
    card = deps["signin"]
    assert card["passive"] is True and card["kind"] == "internal"
    # 10 SSO errors in 15 min: signin_failures_15m warns.
    assert card["status"] == "warn"
    assert card["detail"] == {
        "window": "15m",
        "sso_errors": 10,
        "token_401": 20,
        "lockouts": 0,
    }
    overview = client.get("/admin/ops/overview").json()
    assert "signin" in {d["id"] for d in overview["deps"]}
    drawer = client.get("/admin/ops/health/signin").json()
    assert drawer["dep"]["id"] == "signin" and drawer["detail"]["sso_errors"] == 10


def test_signin_card_absent_without_metrics(monkeypatch):
    monkeypatch.setenv("OPS_METRICS_ENABLED", "false")
    client = TestClient(_app(MemoryOpsStore()))
    deps = client.get("/admin/ops/health").json()["deps"]
    assert "signin" not in {d["id"] for d in deps}
    assert client.get("/admin/ops/health/signin").status_code == 404


def test_signin_dep_is_unknown_when_counts_are_unreadable():
    from lightrag.api.ops.health import signin_dep

    assert signin_dep(None, None, time.time())["status"] == "unknown"
    serious = signin_dep({"sso_errors": 50, "user": "alice"}, "serious", time.time())
    assert serious["status"] == "down" and "user" not in serious["detail"]
