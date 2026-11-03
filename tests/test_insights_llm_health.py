"""Insights store helpers shared with the Operations console."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from lightrag.api.insights.store import (
    CALLS_LIMIT_MAX,
    MemoryUsageStore,
    PostgresUsageStore,
    llm_rates,
)
from lightrag.usage import UsageRecord

pytestmark = pytest.mark.offline


def _record(ts: float, **kw) -> UsageRecord:
    return UsageRecord(workspace="ws", purpose="extract", operation="x", ts=ts, **kw)


def test_llm_rates():
    assert llm_rates({}) == {
        "error_rate": 0.0,
        "truncated_rate": 0.0,
        "abandoned_rate": 0.0,
    }
    assert llm_rates({"calls": 10, "errors": 2, "truncated": 1}) == {
        "error_rate": 0.2,
        "truncated_rate": 0.1,
        "abandoned_rate": 0.0,
    }


async def test_memory_calls_cap_and_cursor():
    store = MemoryUsageStore()
    now = datetime.now(timezone.utc)
    base = now.timestamp()
    store.call_rows = [_record(base - i) for i in range(300)]
    start, end = now - timedelta(hours=1), now + timedelta(minutes=1)
    assert len(await store.calls(start, end)) == 100  # default unchanged
    assert len(await store.calls(start, end, limit=1000)) == CALLS_LIMIT_MAX == 200
    page = await store.calls(start, end, limit=50, before_ts=base - 99.5)
    assert page[0]["ts"] == base - 100 and len(page) == 50
    page = await store.calls(
        start, end, limit=5, before_ts=now - timedelta(seconds=10.5)
    )
    assert page[0]["ts"] == base - 11


async def test_memory_llm_health():
    store = MemoryUsageStore()
    now = datetime.now(timezone.utc)
    base = now.timestamp()
    store.call_rows = [
        _record(base - 1, latency_ms=100),
        _record(base - 2, latency_ms=200, status="error"),
        _record(base - 3, latency_ms=300, status="truncated"),
        _record(base - 4, latency_ms=400),
        _record(base - 5, latency_ms=9999, cache_hit=True),
    ]
    health = await store.llm_health(
        now - timedelta(minutes=1), now + timedelta(seconds=1)
    )
    assert health["calls"] == 4 and health["cache_hits"] == 1
    assert health["errors"] == 1 and health["truncated"] == 1
    assert health["error_rate"] == 0.25
    assert health["p50_ms"] == 250.0
    assert health["p95_ms"] == pytest.approx(385.0)
    empty = await store.llm_health(now - timedelta(days=3), now - timedelta(days=2))
    assert (
        empty["calls"] == 0 and empty["p95_ms"] is None and empty["error_rate"] == 0.0
    )


async def test_postgres_calls_and_health_sql(monkeypatch):
    store = PostgresUsageStore()
    seen = []

    async def rows(sql, params):
        seen.append((sql, params))
        if "percentile_cont(ARRAY" in sql:
            return [
                {
                    "calls": 20,
                    "cache_hits": 1,
                    "errors": 2,
                    "truncated": 0,
                    "abandoned": 1,
                    "pct": [100.0, 900.0, 1200.0],
                }
            ]
        return []

    monkeypatch.setattr(store, "_rows", rows)
    now = datetime.now(timezone.utc)
    await store.calls(
        now - timedelta(hours=1), now, limit=5000, before_ts=now.timestamp()
    )
    sql, params = seen[0]
    assert "ts < $3" in sql and params[-1] == 200
    assert params[2].tzinfo is not None
    health = await store.llm_health(
        now - timedelta(minutes=15), now, filters={"purpose": "extract"}
    )
    sql, params = seen[1]
    assert "NOT cache_hit" in sql and "purpose = $3" in sql
    assert health["error_rate"] == 0.1 and health["p95_ms"] == 900.0
    assert health["abandoned_rate"] == 0.05
