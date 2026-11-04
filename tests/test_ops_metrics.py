"""Operations console performance metrics (PR-6): histograms, the per-worker
flusher, the leased rollup and retention, the HTTP / dependency / sign-in
hooks and their cost on the hot path."""

from __future__ import annotations

import ast
import asyncio
import functools
import inspect
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from lightrag.api.ops import metrics
from lightrag.api.ops.context import RequestIdMiddleware
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.metrics import (
    BUCKET_BOUNDS_MS,
    BUCKETS,
    MAX_KEYS_PER_MINUTE,
    OVERFLOW_KEY,
    UNMATCHED_KEY,
    MetricBins,
    bucket_index,
    estimate_percentiles,
)
from lightrag.api.ops.store import MemoryOpsStore

pytestmark = pytest.mark.offline

WORKER = "0123456789abcdef0123456789abcdef"
REPO = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


class Held:
    """A LeasedLoop stand-in for cycle functions."""

    def __init__(self, held: bool = True):
        self.held = held

    async def still_held(self) -> bool:
        return self.held


@pytest.fixture
def installed():
    """This process's bins (minute 12:00:00 UTC), uninstalled afterwards."""
    clock = Clock(datetime(2026, 11, 3, 12, 0, 5, tzinfo=timezone.utc).timestamp())
    bins = MetricBins(MemoryOpsStore(clock=clock), WORKER, clock=clock)
    metrics.install(bins)
    yield bins
    metrics.install(None)


def _rows(bins: MetricBins, series: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in bins.take(include_current=True):
        if row["series"] == series:
            out[row["key"]] = row
    return out


def _exact(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


# -- histograms -------------------------------------------------------------------------


def test_bucket_bounds_are_20_log_buckets_from_5ms_to_120s():
    assert len(BUCKET_BOUNDS_MS) == BUCKETS == 20
    assert BUCKET_BOUNDS_MS[0] == pytest.approx(5.0)
    assert BUCKET_BOUNDS_MS[-1] == pytest.approx(120_000.0)
    ratios = {round(b / a, 6) for a, b in zip(BUCKET_BOUNDS_MS, BUCKET_BOUNDS_MS[1:])}
    assert len(ratios) == 1  # evenly spaced on a log scale


@pytest.mark.parametrize(
    "ms,index",
    [(0.0, 0), (5.0, 0), (5.01, 1), (8.5, 1), (8.6, 2), (120_000.0, 19), (9e9, 19)],
)
def test_bucket_index_edges(ms, index):
    assert bucket_index(ms) == index


@pytest.mark.parametrize(
    "name,draw",
    [
        ("lognormal", lambda r: r.lognormvariate(5.5, 1.0)),
        ("wide lognormal", lambda r: r.lognormvariate(7.0, 1.8)),
        ("uniform", lambda r: r.uniform(10, 2000)),
        ("exponential", lambda r: r.expovariate(1 / 300)),
    ],
)
def test_percentile_estimates_are_accurate(name, draw):
    rng = random.Random(7)
    values = [draw(rng) for _ in range(20_000)]
    buckets = [0] * BUCKETS
    for v in values:
        buckets[bucket_index(v)] += 1
    estimates = estimate_percentiles(buckets, (0.5, 0.95, 0.99), max(values))
    for fraction, estimate in zip((0.5, 0.95, 0.99), estimates):
        exact = _exact(values, fraction)
        # Interpolation inside a bucket: well under one bucket width (x1.7).
        assert abs(estimate - exact) / exact < 0.2, (name, fraction, estimate, exact)


def test_percentiles_of_a_constant_and_of_nothing():
    buckets = [0] * BUCKETS
    buckets[bucket_index(100.0)] = 50
    p50, p99 = estimate_percentiles(buckets, (0.5, 0.99), 100.0)
    lower = BUCKET_BOUNDS_MS[bucket_index(100.0) - 1]
    assert lower <= p50 <= 100.0 and p99 <= 100.0  # capped at the max
    assert estimate_percentiles([0] * BUCKETS) == [None, None, None]
    over = [0] * BUCKETS
    over[-1] = 10  # all above 120 s: the open bucket runs up to the max
    assert estimate_percentiles(over, (0.99,), 300_000.0)[0] > 120_000.0


# -- per-process bins --------------------------------------------------------------------


def test_observe_counts_errors_sum_max_and_buckets(installed):
    for ms, error in ((10.0, False), (20.0, True), (3.0, False)):
        installed.observe("embed", "bge-m3", ms, error)
    installed.observe("auth", "sso_error", None, True)
    embed = _rows(installed, "embed")["bge-m3"]
    assert embed["count"] == 3 and embed["errors"] == 1
    assert embed["sum_ms"] == 33 and embed["max_ms"] == 20
    assert sum(embed["buckets"]) == 3 and embed["buckets"][0] == 1
    # take() emptied the bins.
    assert installed.take(include_current=True) == []


def test_counts_without_latency_leave_the_histogram_empty(installed):
    installed.observe("auth", "lockout", None, True)
    row = _rows(installed, "auth")["lockout"]
    assert row["count"] == 1 and row["errors"] == 1
    assert row["sum_ms"] == 0 and sum(row["buckets"]) == 0


def test_key_cap_folds_extra_keys_into_other(installed):
    for i in range(MAX_KEYS_PER_MINUTE + 50):
        installed.observe("http", f"GET /r{i}", 1.0)
    keys = _rows(installed, "http")
    assert len(keys) == MAX_KEYS_PER_MINUTE
    assert keys[OVERFLOW_KEY]["count"] == 51
    assert installed.overflowed == 51
    # The cap is per minute: the next minute starts empty.
    installed.clock.now += 60
    installed.observe("http", "GET /new", 1.0)
    assert set(_rows(installed, "http")) == {"GET /new"}


def test_take_returns_only_completed_minutes(installed):
    installed.observe("http", "GET /a", 1.0)
    installed.clock.now += 60
    installed.observe("http", "GET /a", 2.0)
    first = installed.take()
    assert [r["count"] for r in first] == [1]
    assert first[0]["minute"] == int(installed.clock.now) // 60 * 60 - 60
    assert installed.take() == []
    assert len(installed.take(include_current=True)) == 1


async def test_flush_writes_insert_only_and_retries_failures(installed):
    store = installed.store
    installed.observe("http", "GET /a", 4.0)
    installed.clock.now += 60
    assert await installed.flush_once()
    assert len(store.metric_rows) == 1 and installed.rows_written == 1
    # A failing write keeps the rows for the next attempt.
    installed.observe("http", "GET /b", 4.0)
    installed.clock.now += 60
    store.fail_writes = RuntimeError("down")
    assert not await installed.flush_once()
    assert installed.flush_failures == 1 and installed.stats()["pending_rows"] == 1
    store.fail_writes = None
    assert await installed.flush_once()
    assert {r["key"] for r in store.metric_rows} == {"GET /a", "GET /b"}
    # Insert-only: writing the same minute again changes nothing.
    await store.metrics_write(
        [
            {
                "minute": store.metric_rows[0]["minute"].timestamp(),
                "worker_id": WORKER,
                "series": "http",
                "key": "GET /a",
                "count": 99,
                "errors": 0,
                "sum_ms": 0,
                "max_ms": 0,
                "buckets": [0] * BUCKETS,
            }
        ]
    )
    assert [r["count"] for r in store.metric_rows if r["key"] == "GET /a"] == [1]


async def test_pending_rows_are_bounded_and_drops_counted():
    clock = Clock(1_800_000_000.0)
    store = MemoryOpsStore(clock=clock)
    store.fail_writes = RuntimeError("down")
    bins = MetricBins(store, WORKER, clock=clock, max_pending_rows=5)
    for minute in range(4):
        for k in range(3):
            bins.observe("http", f"GET /{k}", 1.0)
        clock.now += 60
        await bins.flush_once()
    assert bins.stats()["pending_rows"] == 5
    assert bins.dropped_rows == 12 - 5


async def test_stop_flushes_the_partial_minute(installed):
    installed.observe("rerank", "bge-reranker", 30.0)
    installed.start()
    await installed.stop()
    assert [r["series"] for r in installed.store.metric_rows] == ["rerank"]


def test_bins_from_another_process_are_ignored(installed):
    installed.pid = -1  # inherited across a fork
    metrics.observe("http", "GET /a", 1.0)
    assert metrics.current() is None
    installed.pid = __import__("os").getpid()
    assert installed.take(include_current=True) == []


# -- rollup and retention -------------------------------------------------------------------


async def _seed_minutes(store: MemoryOpsStore, start: float, minutes: int) -> None:
    rows = []
    for m in range(minutes):
        buckets = [0] * BUCKETS
        buckets[bucket_index(40.0)] = 2
        rows.append(
            {
                "minute": start + m * 60,
                "worker_id": WORKER,
                "series": "http",
                "key": "GET /documents",
                "count": 2,
                "errors": 1 if m % 2 else 0,
                "sum_ms": 80,
                "max_ms": 45,
                "buckets": buckets,
            }
        )
    await store.metrics_write(rows)


async def test_rollup_sums_hours_and_keeps_a_watermark():
    hour = datetime(2026, 11, 3, 9, tzinfo=timezone.utc).timestamp()
    clock = Clock(hour + 2 * 3600 + 600)  # 11:10, so 09:00 and 10:00 are settled
    store = MemoryOpsStore(clock=clock)
    await _seed_minutes(store, hour, 120)
    out = await metrics.rollup_cycle(Held(), store=store, clock=clock)
    assert out["rolled"] == 2
    hours = {k[0]: v for k, v in store.metric_hours.items()}
    first = hours[datetime.fromtimestamp(hour, timezone.utc)]
    assert first["count"] == 120 and first["errors"] == 30 and first["max_ms"] == 45
    assert sum(first["buckets"]) == 120
    assert (await store.kv_get(metrics.ROLLUP_KEY))["through"] == hour + 7200
    # A minute flushed late is still counted: the last hours are re-rolled.
    await _seed_minutes(store, hour + 3600 - 60, 1)  # duplicate PK: ignored
    late = {
        "minute": hour + 30,
        "worker_id": "fedcba9876543210fedcba9876543210",
        "series": "http",
        "key": "GET /documents",
        "count": 5,
        "errors": 0,
        "sum_ms": 50,
        "max_ms": 10,
        "buckets": [5] + [0] * (BUCKETS - 1),
    }
    await store.metrics_write([late])
    await metrics.rollup_cycle(Held(), store=store, clock=clock)
    assert (
        store.metric_hours[
            (datetime.fromtimestamp(hour, timezone.utc), "http", "GET /documents")
        ]["count"]
        == 125
    )


async def test_rollup_watermark_is_written_only_while_the_lease_is_held():
    hour = datetime(2026, 11, 3, 9, tzinfo=timezone.utc).timestamp()
    clock = Clock(hour + 3600 + 600)
    store = MemoryOpsStore(clock=clock)
    await _seed_minutes(store, hour, 10)
    await metrics.rollup_cycle(Held(False), store=store, clock=clock)
    assert await store.kv_get(metrics.ROLLUP_KEY) is None


async def test_retention_deletes_old_minutes_and_hours():
    now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc).timestamp()
    clock = Clock(now)
    store = MemoryOpsStore(clock=clock)
    await _seed_minutes(store, now - 15 * 86400, 3)  # past 14 days
    await _seed_minutes(store, now - 13 * 86400, 2)
    old = datetime.fromtimestamp(now - 91 * 86400, timezone.utc)
    kept = datetime.fromtimestamp(now - 89 * 86400, timezone.utc)
    for when in (old, kept):
        store.metric_hours[(when, "http", "GET /x")] = {
            "count": 1,
            "errors": 0,
            "sum_ms": 1,
            "max_ms": 1,
            "buckets": [1] + [0] * (BUCKETS - 1),
        }
    out = await metrics.rollup_cycle(Held(), store=store, clock=clock)
    assert out["deleted_minutes"] == 3 and out["deleted_hours"] == 1
    assert len(store.metric_rows) == 2
    assert [k[0] for k in store.metric_hours if k[2] == "GET /x"] == [kept]


async def test_rollup_runs_on_one_lease_holder_only():
    clock = Clock(datetime(2026, 11, 3, 12, 10, tzinfo=timezone.utc).timestamp())
    store = MemoryOpsStore(clock=clock)
    ran: list[str] = []

    def cycle_for(name):
        async def cycle(loop):
            ran.append(name)
            await metrics.rollup_cycle(loop, store=store, clock=clock)

        return cycle

    loops = [
        LeasedLoop(
            store,
            metrics.ROLLUP_LEASE,
            holder,
            cycle_for(holder),
            ttl_s=metrics.ROLLUP_LEASE_TTL_S,
            interval_s=metrics.ROLLUP_INTERVAL_S,
            cycle_timeout_s=metrics.ROLLUP_CYCLE_TIMEOUT_S,
        )
        for holder in ("worker-a", "worker-b")
    ]
    assert await loops[0].run_once() is True
    assert await loops[1].run_once() is False
    assert ran == ["worker-a"]


async def test_memory_read_splits_hour_and_minute_rows():
    hour = datetime(2026, 11, 3, 9, tzinfo=timezone.utc).timestamp()
    clock = Clock(hour + 3 * 3600)
    store = MemoryOpsStore(clock=clock)
    await _seed_minutes(store, hour, 180)
    await metrics.rollup_cycle(Held(), store=store, clock=clock)
    cut = datetime.fromtimestamp(hour + 2 * 3600, timezone.utc)
    rows = await store.metrics_read(
        ["http"],
        datetime.fromtimestamp(hour, timezone.utc),
        cut,
        datetime.fromtimestamp(hour + 3 * 3600, timezone.utc),
        3600,
    )
    # Hours 9 and 10 from the rollup, hour 11 from minutes: nothing twice.
    assert [r["count"] for r in rows] == [120, 120, 120]


# -- HTTP hook -------------------------------------------------------------------------------


def _http_app(tmp_path) -> FastAPI:
    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: str):
        return {"id": item_id}

    @app.get("/broken")
    async def broken():
        raise RuntimeError("SECRET boom")

    @app.get("/unavailable")
    async def unavailable():
        raise HTTPException(status_code=503, detail="down")

    @app.get("/private")
    async def private():
        raise HTTPException(status_code=401, detail="Not authenticated")

    @app.post("/login")
    async def login():
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    @app.get("/api/oauth2/callback")
    async def sso_callback():
        # The real route counts its own outcome (sso_error) before raising.
        metrics.auth_event("sso_error")
        raise HTTPException(status_code=401, detail="Invalid state")

    (tmp_path / "a.txt").write_text("x")
    app.mount("/static", StaticFiles(directory=tmp_path), name="static")
    app.add_middleware(
        RequestIdMiddleware, trusted_proxies="", observers=[metrics.http_observer]
    )
    return app


def test_middleware_keys_routes_by_template_and_counts_5xx(installed, tmp_path):
    client = TestClient(_http_app(tmp_path), raise_server_exceptions=False)
    assert client.get("/items/123").status_code == 200
    assert client.get("/items/456").status_code == 200
    assert client.get("/broken").status_code == 500
    assert client.get("/unavailable").status_code == 503
    assert client.get("/static/a.txt").status_code == 200
    http = _rows(installed, "http")
    assert http["GET /items/{item_id}"]["count"] == 2
    assert http["GET /items/{item_id}"]["errors"] == 0
    # An exception and a 503 are both server errors.
    assert http["GET /broken"]["errors"] == 1
    assert http["GET /unavailable"]["errors"] == 1
    # A mount is keyed by its mount point, never the file path.
    assert set(k for k in http if "static" in k) == {"GET <mount>/static"}
    assert not any("123" in k or "a.txt" in k for k in http)


def test_404_stays_404_and_is_keyed_unmatched(installed, tmp_path):
    client = TestClient(_http_app(tmp_path))
    for path in ("/nope/1", "/nope/2", "/does-not-exist"):
        assert client.get(path).status_code == 404
    http = _rows(installed, "http")
    assert set(http) == {UNMATCHED_KEY}
    assert http[UNMATCHED_KEY]["count"] == 3 and http[UNMATCHED_KEY]["errors"] == 0


def test_route_key_cap_holds_through_the_middleware(installed, tmp_path):
    app = FastAPI()
    for i in range(MAX_KEYS_PER_MINUTE + 20):
        app.add_api_route(f"/r{i}", (lambda: {"ok": True}), methods=["GET"])
    app.add_middleware(
        RequestIdMiddleware, trusted_proxies="", observers=[metrics.http_observer]
    )
    client = TestClient(app)
    for i in range(MAX_KEYS_PER_MINUTE + 20):
        client.get(f"/r{i}")
    client.get("/missing")
    http = _rows(installed, "http")
    assert len(http) == MAX_KEYS_PER_MINUTE
    assert OVERFLOW_KEY in http
    assert sum(r["count"] for r in http.values()) == MAX_KEYS_PER_MINUTE + 21


def test_401s_are_counted_except_password_login(installed, tmp_path):
    client = TestClient(_http_app(tmp_path))
    assert client.get("/private").status_code == 401
    assert client.post("/login").status_code == 401
    auth = _rows(installed, "auth")
    assert auth == {"token_401": auth["token_401"]}
    assert auth["token_401"]["count"] == 1


def test_failed_sso_callback_is_counted_once(installed, tmp_path):
    """A 401 from the SSO callback is its sso_error, not also a token_401."""
    client = TestClient(_http_app(tmp_path))
    assert client.get("/api/oauth2/callback").status_code == 401
    auth = _rows(installed, "auth")
    assert set(auth) == {"sso_error"}
    assert auth["sso_error"]["count"] == 1


def test_hooks_are_noops_without_installed_metrics(tmp_path):
    metrics.install(None)
    client = TestClient(_http_app(tmp_path))
    assert client.get("/items/1").status_code == 200
    metrics.observe("http", "GET /x", 1.0)
    metrics.auth_event("lockout")
    assert metrics.current() is None


# -- sign-in counters --------------------------------------------------------------------------


def test_auth_events_accept_fixed_outcomes_only(installed):
    for outcome in ("sso_ok", "sso_error", "lockout", "login_failed"):
        metrics.auth_event(outcome)
    metrics.auth_event("alice@unimas.my")
    metrics.auth_event("login_failed:alice")
    auth = _rows(installed, "auth")
    assert set(auth) == {"sso_ok", "sso_error", "lockout", "login_failed"}
    assert auth["sso_ok"]["errors"] == 0 and auth["sso_error"]["errors"] == 1


def test_server_counts_sign_in_outcomes_without_usernames():
    """Every auth_event call in the server passes a fixed outcome literal."""
    source = (REPO / "lightrag" / "api" / "lightrag_server.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "auth_event"
    ]
    outcomes = set()
    for call in calls:
        assert len(call.args) == 1 and not call.keywords
        assert isinstance(call.args[0], ast.Constant), ast.unparse(call)
        outcomes.add(call.args[0].value)
    assert outcomes <= metrics.AUTH_OUTCOMES
    assert {
        "login_ok",
        "login_failed",
        "login_throttled",
        "lockout",
        "sso_ok",
        "sso_error",
    } <= outcomes


# -- embed / rerank wrapper ----------------------------------------------------------------------


def _rag(tmp_path, embed, rerank):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def llm(*args, **kwargs):
        return "ok"

    return LightRAG(
        working_dir=str(tmp_path),
        embedding_func=EmbeddingFunc(embedding_dim=4, func=embed, model_name="bge-m3"),
        llm_model_func=llm,
        rerank_model_func=rerank,
    )


async def test_dep_wrapper_keeps_queue_stats_on_the_outer_wrapper(installed, tmp_path):
    seen_limits = []

    async def embed(texts, max_token_size=None, **kwargs):
        seen_limits.append(max_token_size)
        return np.zeros((len(texts), 4))

    async def rerank(query, documents, top_n=None, **kwargs):
        raise RuntimeError("rerank down")

    rerank.model_name = "bge-reranker"
    rag = _rag(tmp_path, embed, rerank)
    rag.embedding_func.max_token_size = 512
    try:
        # .func is still the queue wrapper; the queue status is still there.
        assert callable(getattr(rag.embedding_func.func, "get_queue_stats", None))
        assert callable(getattr(rag.embedding_func.func, "shutdown", None))
        assert callable(getattr(rag.rerank_model_func, "get_queue_stats", None))
        assert (await rag.get_embedding_queue_status())["available"] is True
        assert (await rag.get_rerank_queue_status())["available"] is True
        # The signature still reaches EmbeddingFunc through the wrappers.
        assert "max_token_size" in inspect.signature(rag.embedding_func.func).parameters
        result = await rag.embedding_func(["a", "b"])
        assert result.shape == (2, 4) and seen_limits == [512]
        with pytest.raises(RuntimeError, match="rerank down"):
            await rag.rerank_model_func(query="q", documents=["d"])
    finally:
        await rag.embedding_func.func.shutdown()
        await rag.rerank_model_func.shutdown()
    rows = {(r["series"], r["key"]): r for r in installed.take(include_current=True)}
    assert rows[("embed", "bge-m3")]["count"] == 1
    assert rows[("embed", "bge-m3")]["errors"] == 0
    assert rows[("rerank", "bge-reranker")]["errors"] == 1


async def test_dep_wrapper_records_errors_by_model(installed):
    async def rerank(**kwargs):
        raise TimeoutError("vllm")

    async def embed(texts):
        return texts

    wrapped = metrics.wrap_dep_latency(functools.partial(rerank, model="m1"), "rerank")
    other = metrics.wrap_dep_latency(embed, "embed", "bge-m3")
    with pytest.raises(TimeoutError):
        await wrapped()
    assert await other(["x"]) == ["x"]
    rows = installed.take(include_current=True)
    by_series = {(r["series"], r["key"]): r for r in rows}
    assert by_series[("rerank", "m1")]["errors"] == 1
    assert by_series[("embed", "bge-m3")]["errors"] == 0
    # Wrapping twice does not time twice.
    assert metrics.wrap_dep_latency(other, "embed") is other


def test_dep_wrapper_is_not_installed_when_metrics_are_off(monkeypatch):
    monkeypatch.setenv("OPS_METRICS_ENABLED", "false")

    async def embed(texts):
        return texts

    assert metrics.wrap_dep_latency(embed, "embed") is embed
    assert metrics.wrap_dep_latency(None, "rerank") is None


# -- hot-path cost (§3.12: < 50 µs per call; CI-safe margin) ----------------------------------


async def test_wrapper_and_request_overhead_stay_under_50_microseconds(installed):
    async def raw(texts):
        return texts

    wrapped = metrics.wrap_dep_latency(raw, "embed", "bge-m3")
    calls = 5000

    async def timed(fn) -> float:
        started = time.perf_counter()
        for _ in range(calls):
            await fn(["a"])
        return (time.perf_counter() - started) / calls

    await timed(wrapped)  # warm up
    base = min([await timed(raw) for _ in range(3)])
    with_metrics = min([await timed(wrapped) for _ in range(3)])
    wrapper_us = max(0.0, with_metrics - base) * 1e6

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestIdMiddleware(
        inner, trusted_proxies="", observers=[metrics.http_observer]
    )
    plain = RequestIdMiddleware(inner, trusted_proxies="")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/documents",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "route": type("R", (), {"path": "/documents"})(),
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        return None

    async def requests(app) -> float:
        started = time.perf_counter()
        for _ in range(calls):
            await app(dict(scope), receive, send)
        return (time.perf_counter() - started) / calls

    await requests(middleware)
    observed = min([await requests(middleware) for _ in range(3)])
    unobserved = min([await requests(plain) for _ in range(3)])
    request_us = max(0.0, observed - unobserved) * 1e6
    print(
        f"\nmetrics overhead: wrapped call +{wrapper_us:.2f} us, "
        f"request observer +{request_us:.2f} us"
    )
    assert wrapper_us < 50.0
    assert request_us < 50.0
    assert installed.observed >= 2 * calls


def test_auth_event_overhead_is_small(installed):
    started = time.perf_counter()
    for _ in range(20_000):
        metrics.auth_event("token_401")
    per_call_us = (time.perf_counter() - started) / 20_000 * 1e6
    assert per_call_us < 50.0


# -- the env flag ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("", True), ("true", True), ("0", False), ("off", False)],
)
def test_metrics_flag(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("OPS_METRICS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("OPS_METRICS_ENABLED", value)
    assert metrics.metrics_enabled() is expected


def test_flusher_aligns_to_minute_boundaries():
    clock = Clock(1_800_000_030.0)
    bins = MetricBins(None, WORKER, clock=clock)
    delay = bins._next_delay()
    assert 30.0 + 2.0 <= delay <= 30.0 + 6.0
    asyncio.run(bins.flush_once())  # no store: nothing to write, never raises


def test_minute_rows_carry_the_worker_and_epoch_minute(installed):
    installed.observe("embed", "x" * 500, 1.0)
    (row,) = installed.take(include_current=True)
    assert row["worker_id"] == WORKER and row["minute"] % 60 == 0
    assert len(row["key"]) == metrics.KEY_MAX_LEN
    assert (
        datetime.fromtimestamp(row["minute"], timezone.utc)
        <= datetime.fromtimestamp(installed.clock.now, timezone.utc)
        < datetime.fromtimestamp(row["minute"], timezone.utc) + timedelta(minutes=1)
    )
