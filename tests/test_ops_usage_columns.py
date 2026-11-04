"""``error_kind`` and ``request_id`` on LLM usage calls (PR-9).

The role wrapper hands the exception of a failed call to the usage record;
the recorder classifies it (``classify.py``), stamps the request ID from
its ContextVar and drops the exception before buffering. Insights returns
both columns and filters on ``request_id``; ``/correlate`` finds a
request's LLM calls.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from lightrag.api.insights.recorder import UsageRecorder, complete_record
from lightrag.api.insights.service import InsightsService
from lightrag.api.insights.store import MemoryUsageStore
from lightrag.api.ops import context as ops_context
from lightrag.api.ops.store import MemoryOpsStore
from lightrag.usage import (
    UsageRecord,
    add_usage_sink,
    recorded_stream,
    remove_usage_sink,
)
from tests.ops_fixtures import build_app

pytestmark = pytest.mark.offline

REQUEST = "5d1e0c3f9a2b4c6d8e0f1a2b3c4d5e6f"


class RateLimitError(Exception):
    """Stands in for openai.RateLimitError (classified by class name)."""

    status_code = 429


def _record(**kw) -> UsageRecord:
    kw.setdefault("workspace", "fsktm")
    kw.setdefault("purpose", "extract")
    kw.setdefault("operation", "extraction")
    return UsageRecord(**kw)


def test_recorder_classifies_failures_and_stamps_the_request():
    token = ops_context._request_id.set(REQUEST)
    try:
        recorder = UsageRecorder(MemoryUsageStore())
        recorder.record(_record(status="error", error=RateLimitError("429 slow down")))
        recorder.record(_record(status="ok"))
    finally:
        ops_context._request_id.reset(token)
    failed, ok = recorder._records
    assert failed.error_kind == "rate_limited" and failed.request_id == REQUEST
    assert failed.error is None
    assert ok.error_kind is None and ok.request_id == REQUEST


def test_completion_keeps_given_values_and_never_raises():
    record = _record(status="error", error_kind="timeout", request_id="r" * 10)
    assert complete_record(record) is record
    # An error without an exception still gets a kind.
    assert complete_record(_record(status="error")).error_kind == "unknown"
    # Outside a request no ID is invented.
    assert complete_record(_record()).request_id is None


async def test_role_wrapper_passes_the_exception(tmp_path):
    from tests.test_insights import _build_rag, _role

    seen: list[UsageRecord] = []
    add_usage_sink(seen.append)

    async def llm(prompt, **kwargs):
        raise RateLimitError("Error code: 429")

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        with pytest.raises(RateLimitError):
            await rag.role_llm_funcs["query"]("q")
    finally:
        remove_usage_sink(seen.append)
        await rag.finalize_storages()
    [record] = seen
    assert record.status == "error" and isinstance(record.error, RateLimitError)
    assert complete_record(record).error_kind == "rate_limited"


async def test_a_failing_stream_passes_its_exception():
    finished: list[tuple] = []

    async def stream():
        yield "a"
        raise RateLimitError("429")

    def finish(parts, status, error=None):
        finished.append((parts, status, error))

    with pytest.raises(RateLimitError):
        async for _ in recorded_stream(stream(), finish):
            pass
    [(parts, status, error)] = finished
    assert parts == ["a"] and status == "error" and isinstance(error, RateLimitError)


async def test_insights_calls_return_and_filter_the_new_columns():
    store = MemoryUsageStore()
    now = time.time()
    await store.write(
        [
            _record(
                status="error",
                error_kind="rate_limited",
                request_id=REQUEST,
                ts=now - 60,
            ),
            _record(status="ok", request_id="other-request-id", ts=now - 30),
        ],
        [],
    )
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    rows = await store.calls(start, end, request_id=REQUEST)
    assert len(rows) == 1 and rows[0]["error_kind"] == "rate_limited"
    assert "error" not in rows[0]
    service = InsightsService(store, default_rag=None, app_state=SimpleNamespace())
    out = await service.calls(1, {}, request_id=REQUEST)
    assert [(r["error_kind"], r["request_id"]) for r in out] == [
        ("rate_limited", REQUEST)
    ]
    everything = await service.calls(1, {})
    assert {r["request_id"] for r in everything} == {REQUEST, "other-request-id"}


async def test_correlate_finds_the_llm_calls_of_a_request():
    store = MemoryOpsStore()
    now = datetime.now(timezone.utc)
    store.usage_call_rows += [
        {
            "ts": now - timedelta(minutes=2),
            "workspace": "fsktm",
            "request_id": REQUEST,
            "status": "error",
            "error_kind": "rate_limited",
        },
        {
            "ts": now - timedelta(minutes=1),
            "workspace": "fsktm",
            "request_id": REQUEST,
            "status": "ok",
            "error_kind": None,
        },
        {"ts": now, "workspace": "space1", "request_id": "x" * 32, "status": "ok"},
    ]
    app = build_app(store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        body = (await client.get("/admin/ops/correlate", params={"id": REQUEST})).json()
    [match] = [m for m in body["matches"] if m["type"] == "request"]
    assert match["workspace"] == "fsktm"
    assert match["usage_calls"] == {
        "calls": 2,
        "errors": 1,
        "error_kinds": ["rate_limited"],
        "window_hours": None,
    }
    assert match["log_lines"] == 0
    assert "2 LLM calls (1 failed: rate_limited)" in match["label"]
    assert match["link"].startswith("#/")


async def test_correlate_survives_a_failing_usage_lookup(monkeypatch):
    store = MemoryOpsStore()
    store.log_rows.append(
        {
            "id": 1,
            "ts": datetime.now(timezone.utc),
            "level": 30,
            "logger": "lightrag",
            "msg": "x",
            "request_id": REQUEST,
            "workspace": "fsktm",
            "fingerprint": "0" * 16,
        }
    )

    async def broken(request_id, limit):
        raise RuntimeError("column missing")

    monkeypatch.setattr(store, "usage_calls_by_request", broken)
    app = build_app(store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        response = await client.get("/admin/ops/correlate", params={"id": REQUEST})
    assert response.status_code == 200
    [match] = [m for m in response.json()["matches"] if m["type"] == "request"]
    assert match["count"] == 1 and match["usage_calls"] is None


class _CatalogConn:
    """A connection that answers the column catalog and records ALTERs."""

    def __init__(self, columns: set[str], log: list[str]):
        self.columns = columns
        self.log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def transaction(self):
        return self

    async def execute(self, sql, *args):
        self.log.append(sql)
        if "ADD COLUMN IF NOT EXISTS" in sql:
            self.columns.add(sql.split("IF NOT EXISTS")[1].split()[0])

    async def fetch(self, sql, *args):
        assert "information_schema.columns" in sql
        return [{"column_name": c} for c in sorted(self.columns) if c in args[1]]


async def _migrate(columns: set[str]) -> tuple[frozenset, list[str]]:
    from lightrag.api.insights.store import PostgresUsageStore

    log: list[str] = []
    store = PostgresUsageStore()

    async def connection():
        return _CatalogConn(columns, log)

    store._connection = connection
    await store._add_call_columns()
    return store.call_columns, log


async def test_column_migration_alters_only_missing_columns():
    # Both present (every start after the first): no ALTER, so no exclusive
    # lock is queued on the hot calls table.
    present, log = await _migrate({"error_kind", "request_id"})
    assert present == {"error_kind", "request_id"}
    assert not [s for s in log if "ALTER" in s]
    # One missing: only that one is altered, under the lock timeout.
    present, log = await _migrate({"request_id"})
    assert present == {"error_kind", "request_id"}
    alters = [s for s in log if "ALTER" in s]
    assert len(alters) == 1 and "error_kind" in alters[0]
    assert any("lock_timeout" in s for s in log)
    # A pre-PR-9 table gets both.
    present, log = await _migrate(set())
    assert present == {"error_kind", "request_id"}
    assert len([s for s in log if "ALTER" in s]) == 2
