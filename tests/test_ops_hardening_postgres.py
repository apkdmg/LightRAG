"""PR-9 hardening against a real Postgres (pgvector/pgvector:pg16 ships pg_trgm).

- ``pg_trgm`` detection (available, then installed), the background
  builder's GIN trigram index on ``LIGHTRAG_OPS_LOGS(msg)``, and an
  ``EXPLAIN`` of the 7-day text search showing that index;
- the ``error_kind`` / ``request_id`` columns added to an existing
  ``LIGHTRAG_USAGE_CALLS`` table, filled by the recorder path, read by
  Insights and found by request ID (with and without its index);
- the GPU history query over probe rows.

Runs only with ``LIGHTRAG_RUN_INTEGRATION=true``, ``--run-integration`` and
``POSTGRES_*`` pointing at a throwaway database (it drops the pg_trgm
extension and the trigram index it creates).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_db,
    pytest.mark.asyncio(loop_scope="module"),
]

if os.getenv("LIGHTRAG_RUN_INTEGRATION", "false").lower() != "true" or not os.getenv(
    "POSTGRES_HOST"
):
    pytest.skip(
        "needs LIGHTRAG_RUN_INTEGRATION=true and POSTGRES_* settings",
        allow_module_level=True,
    )

import pytest_asyncio  # noqa: E402

from lightrag.api.ops import indexes, logs  # noqa: E402
from lightrag.api.ops.store import (  # noqa: E402
    TABLE_LOGS,
    TABLE_PROBES,
    PostgresOpsStore,
    _log_conditions,
)
from lightrag.usage import UsageRecord  # noqa: E402

LOG_ROWS = int(os.getenv("OPS_IT_TRGM_ROWS", "200000"))
REQUEST = "5d1e0c3f9a2b4c6d8e0f1a2b3c4d5e6f"


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def pg():
    store = PostgresOpsStore()
    await store.initialize()
    async with store.dedicated_connection() as conn:
        await conn.execute(f"DROP INDEX IF EXISTS {indexes.LOG_TRGM_INDEX}")
        await conn.execute("DROP EXTENSION IF EXISTS pg_trgm")
    yield store
    async with store.dedicated_connection() as conn:
        await conn.execute(f"DROP INDEX IF EXISTS {indexes.LOG_TRGM_INDEX}")
    await store.finalize()


async def test_trgm_detection_index_and_explain(pg):
    store = pg
    state = await store.log_trgm_state()
    assert state == {"available": True, "installed": False, "index_valid": False}
    async with store.dedicated_connection() as conn:
        # Not installed and not allowed: nothing is created.
        assert await indexes.prepare_trgm(conn, create=False) is False
        assert indexes.LOG_TRGM_SPEC not in await indexes.index_specs(
            conn, create=False
        )
        # Allowed: created, and the builder adds the trigram index.
        specs = await indexes.index_specs(conn, create=True)
        assert indexes.LOG_TRGM_SPEC in specs

        await conn.execute(f"TRUNCATE {TABLE_LOGS}")
        await conn.execute(
            f"""INSERT INTO {TABLE_LOGS}
                (ts, level, logger, module, lineno, msg, fingerprint)
            SELECT now() - (g * interval '3 seconds'), 30, 'lightrag', 'operate', 1,
                CASE WHEN g % 50000 = 7 THEN 'needle-gateway timeout at hop ' || g
                     ELSE 'routine warning number ' || g || ' for chunk ' || md5(g::text)
                END,
                substr(md5((g % 97)::text), 1, 16)
            FROM generate_series(1, $1) g""",
            LOG_ROWS,
        )
        await conn.execute(f"ANALYZE {TABLE_LOGS}")
        outcome = await indexes.build_indexes(conn, specs)
        assert outcome[indexes.LOG_TRGM_INDEX] == "created"
        valid = await conn.fetchval(
            """SELECT i.indisvalid FROM pg_class c JOIN pg_index i
               ON i.indexrelid = c.oid WHERE c.relname = $1""",
            indexes.LOG_TRGM_INDEX,
        )
        assert valid is True

    store._trgm = None  # the detection is cached for a few minutes
    state = await store.log_trgm_state()
    assert state == {"available": True, "installed": True, "index_valid": True}

    # The 7-day text search the cap now allows uses the GIN index.
    now = datetime.now(timezone.utc)
    f = logs.parse_filter(
        {"from": (now - timedelta(days=7)).isoformat(), "q": "needle-gateway"},
        now,
        text_window_s=logs.MAX_WINDOW_S,
    )
    args: list = []
    where = " AND ".join(_log_conditions(f, args))
    args.append(101)
    async with store.dedicated_connection() as conn:
        plan = await conn.fetchval(
            f"""EXPLAIN (FORMAT JSON) SELECT id, ts, msg FROM {TABLE_LOGS}
            WHERE {where} ORDER BY ts DESC, id DESC LIMIT ${len(args)}""",
            *args,
        )
        text = json.dumps(plan)
        print("trgm plan:", text[:600])
        assert indexes.LOG_TRGM_INDEX in text
        rows = await conn.fetch(
            f"""SELECT msg FROM {TABLE_LOGS} WHERE {where}
            ORDER BY ts DESC, id DESC LIMIT ${len(args)}""",
            *args,
        )
        assert rows and all("needle-gateway" in r["msg"] for r in rows)
    page = await store.logs_page(f, before=None, after=None, limit=10)
    assert page and all("needle-gateway" in r["msg"] for r in page)


async def test_usage_columns_migrate_on_an_existing_table(pg):
    from lightrag.api.insights.recorder import complete_record
    from lightrag.api.insights.store import PostgresUsageStore
    from lightrag.api.ops import context as ops_context

    usage = PostgresUsageStore()
    await usage.initialize()
    try:
        async with pg.dedicated_connection() as conn:
            # The table as it was before PR-9.
            await conn.execute("DROP INDEX IF EXISTS idx_lightrag_usage_calls_req")
            for column in ("error_kind", "request_id"):
                await conn.execute(
                    f"ALTER TABLE LIGHTRAG_USAGE_CALLS DROP COLUMN IF EXISTS {column}"
                )
            await conn.execute(
                "DELETE FROM LIGHTRAG_USAGE_CALLS WHERE workspace = 'it_pr9'"
            )
        await usage.initialize()
        assert usage.call_columns == {"error_kind", "request_id"}

        class RateLimitError(Exception):
            status_code = 429

        token = ops_context._request_id.set(REQUEST)
        try:
            failed = complete_record(
                UsageRecord(
                    workspace="it_pr9",
                    purpose="extract",
                    operation="extraction",
                    status="error",
                    error=RateLimitError("429"),
                )
            )
            ok = complete_record(
                UsageRecord(workspace="it_pr9", purpose="extract", operation="x")
            )
        finally:
            ops_context._request_id.reset(token)
        await usage.write([failed, ok], [])
        start = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(minutes=1)
        rows = await usage.calls(start, end, request_id=REQUEST)
        assert sorted((r["status"], r["error_kind"]) for r in rows) == [
            ("error", "rate_limited"),
            ("ok", None),
        ]

        pg._column_checks.clear()
        pg._index_checks.clear()
        await pg._refresh_tables(force=True)
        found = await pg.usage_calls_by_request(REQUEST, 10)
        assert found[0]["workspace"] == "it_pr9" and found[0]["n"] == 2
        assert found[0]["errors"] == 1 and found[0]["error_kinds"] == ["rate_limited"]
        # Before its index exists, only the last 24 h are searched.
        assert found[0]["window_hours"] == 24
        async with pg.dedicated_connection() as conn:
            outcome = await indexes.build_indexes(conn)
        assert outcome["idx_lightrag_usage_calls_req"] in ("created", "valid")
        pg._index_checks.clear()
        found = await pg.usage_calls_by_request(REQUEST, 10)
        assert found[0]["window_hours"] is None and found[0]["n"] == 2
    finally:
        async with pg.dedicated_connection() as conn:
            await conn.execute(
                "DELETE FROM LIGHTRAG_USAGE_CALLS WHERE workspace = 'it_pr9'"
            )
        await usage.finalize()


async def test_gpu_series_sql(pg):
    async with pg.dedicated_connection() as conn:
        await conn.execute(f"DELETE FROM {TABLE_PROBES} WHERE dependency = 'gpu'")
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        for minutes, util in ((3, 40.0), (2, 60.0), (1, 20.0)):
            await conn.execute(
                f"""INSERT INTO {TABLE_PROBES} (ts, dependency, status, detail)
                VALUES ($1, 'gpu', 'ok', $2::jsonb)""",
                base - timedelta(minutes=minutes),
                json.dumps(
                    {
                        "gpus": [
                            {
                                "index": 0,
                                "util_pct": util,
                                "mem_pct": 70,
                                "temp_c": 60 + minutes,
                                "power_w": 100,
                            },
                            {"index": 1, "util_pct": "n/a"},
                        ]
                    }
                ),
            )
    rows = await pg.gpu_series(
        base - timedelta(hours=1), base + timedelta(minutes=1), 3600
    )
    zero = [r for r in rows if r["gpu"] == 0]
    assert sum(1 for _ in zero) >= 1
    if len(zero) == 1:
        assert zero[0]["util_pct"] == pytest.approx(40.0)
        assert zero[0]["temp_c"] == 63.0
    one = [r for r in rows if r["gpu"] == 1]
    assert all(r["util_pct"] is None for r in one)
