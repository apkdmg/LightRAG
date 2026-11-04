"""Operations console SQL against a real Postgres, at release scale.

Seeds 3000 workspaces x a few hundred doc-status rows (0.5 % in flight, 2 %
failed, 10 % ``dup-*``) with ``copy_records_to_table``, runs the background
index builder, then checks that:

- the backlog (Q1), stale-doc (Q3) and live-candidate (Q6) queries never
  sequentially scan ``LIGHTRAG_DOC_STATUS``, also under
  ``plan_cache_mode = force_generic_plan``;
- the latency budgets hold loosely (Q1 < 50 ms, Q2 page < 150 ms,
  Q3/Q6 < 50 ms; asserted at 3x to absorb a noisy laptop, measured values
  printed);
- the store agrees with ``MemoryOpsStore`` on a small dataset;
- leases, KV helpers, the index builder's invalid-index handling and the
  workspace-deletion hook behave as specified.

Runs only with ``LIGHTRAG_RUN_INTEGRATION=true``, ``--run-integration`` and
``POSTGRES_HOST`` / ``POSTGRES_PORT`` / ``POSTGRES_USER`` /
``POSTGRES_PASSWORD`` / ``POSTGRES_DATABASE`` pointing at a throwaway
database (it truncates ``LIGHTRAG_DOC_STATUS``). ``OPS_IT_WORKSPACES`` and
``OPS_IT_DOCS_PER_WS`` change the seed size.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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

from lightrag.api.ops import indexes  # noqa: E402
from lightrag.api.ops.service import OpsService  # noqa: E402
from lightrag.api.ops.snapshot import NamespaceReader  # noqa: E402
from lightrag.api.ops.store import (  # noqa: E402
    MemoryOpsStore,
    Page,
    PostgresOpsStore,
    deleted_key,
    workspace_hash,
)

WORKSPACES = int(os.getenv("OPS_IT_WORKSPACES", "3000"))
DOCS_PER_WS = int(os.getenv("OPS_IT_DOCS_PER_WS", "300"))
BUDGET_SLACK = 3.0
COLUMNS = [
    "workspace",
    "id",
    "status",
    "file_path",
    "track_id",
    "metadata",
    "error_msg",
    "chunks_count",
    "created_at",
    "updated_at",
]
ERRORS = [
    "RateLimitError: Error code: 429 - request 7f3a9c21 exceeded quota",
    "APITimeoutError: Request timed out after 120 seconds",
    "JSONDecodeError: Expecting value: line 1 column 3712 (char 3711)",
    "ReadTimeout docling:5001 after 300s",
]
CANCELLED = [
    "User cancelled",
    "User cancelled during entity extraction",
    "User cancelled during merge phase",
]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _row(
    ws: str, doc_id: str, status: str, updated: datetime, error=None, metadata=None
):
    return (
        ws,
        doc_id,
        status,
        f"report-{doc_id[-6:]}.pdf",
        f"upload_20261103_101010_{doc_id[-8:]}",
        json.dumps(metadata or {}),
        error,
        3,
        updated - timedelta(minutes=5),
        updated,
    )


def _seed_rows(workspaces: int, per_ws: int, rng: random.Random):
    now = _now()
    statuses_inflight = ["pending", "parsing", "analyzing", "processing"]
    for w in range(workspaces):
        ws = f"user{w:04d}_unimas_my"
        for d in range(per_ws):
            roll = rng.random()
            updated = now - timedelta(minutes=rng.randint(1, 60 * 24 * 30))
            doc_id = f"doc-{w:04d}{d:05d}{rng.getrandbits(32):08x}"
            if roll < 0.005:
                status = statuses_inflight[d % 4]
                updated = now - timedelta(minutes=rng.choice([5, 30, 90, 400]))
                yield _row(ws, doc_id, status, updated)
            elif roll < 0.025:
                if rng.random() < 0.2:
                    error = rng.choice(CANCELLED)
                else:
                    error = rng.choice(ERRORS)
                yield _row(ws, doc_id, "failed", updated, error=error)
            elif roll < 0.125:
                yield _row(
                    ws, f"dup-{doc_id[4:]}", "failed", updated, error="duplicate"
                )
            else:
                yield _row(ws, doc_id, "processed", updated)


async def _copy(conn, rows):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= 50_000:
            await conn.copy_records_to_table(
                "lightrag_doc_status", records=batch, columns=COLUMNS
            )
            batch = []
    if batch:
        await conn.copy_records_to_table(
            "lightrag_doc_status", records=batch, columns=COLUMNS
        )


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def pg():
    from lightrag.api.insights.store import PostgresUsageStore
    from lightrag.api.workspace_registry import PostgresWorkspaceRegistry

    store = PostgresOpsStore()
    await store.initialize()
    usage = PostgresUsageStore()
    await usage.initialize()
    registry = PostgresWorkspaceRegistry()
    await registry.initialize()
    await store._refresh_tables(force=True)
    yield store
    await registry.finalize()
    await usage.finalize()
    await store.finalize()


async def _reset(store: PostgresOpsStore) -> None:
    async with store.dedicated_connection() as conn:
        await conn.execute("TRUNCATE lightrag_doc_status")
        await conn.execute("TRUNCATE lightrag_workspace_owners")
        await conn.execute("TRUNCATE lightrag_usage_events")
        await conn.execute("TRUNCATE lightrag_ops_kv")
        await conn.execute("TRUNCATE lightrag_ops_audit")
        await conn.execute("TRUNCATE lightrag_ops_probes")
        await conn.execute("TRUNCATE lightrag_ops_runs")
        await conn.execute("TRUNCATE lightrag_ops_doc_failures")


# -- small dataset: Postgres and the memory store must agree ---------------------


def _small_dataset(now: datetime):
    """(workspace, id, status, updated_at, error, metadata) rows."""
    rows = []
    for w in range(6):
        ws = f"ws{w}"
        for d in range(12):
            status = [
                "pending",
                "parsing",
                "analyzing",
                "processing",
                "failed",
                "processed",
            ][d % 6]
            updated = now - timedelta(minutes=15 * (d + 1) + w, seconds=30)
            error = None
            metadata = {}
            if status == "failed":
                error = ERRORS[(w + d) % len(ERRORS)] if d % 4 else CANCELLED[w % 3]
                if w == 5:
                    metadata = {"failure": {"error_kind": "provider_budget"}}
            rows.append((ws, f"doc-{w}-{d:02d}", status, updated, error, metadata))
        rows.append((ws, f"dup-{w}-x", "failed", now, "duplicate", {}))
    return rows


async def test_postgres_matches_memory_store(pg):
    await _reset(pg)
    now = _now()
    memory = MemoryOpsStore(clock=lambda: now.replace(tzinfo=timezone.utc).timestamp())
    records = []
    for ws, doc_id, status, updated, error, metadata in _small_dataset(now):
        records.append(_row(ws, doc_id, status, updated, error, metadata))
        memory.add_doc(
            ws,
            doc_id,
            status,
            updated_at=updated,
            created_at=updated - timedelta(minutes=5),
            error_msg=error,
            metadata=metadata,
            file_path=f"report-{doc_id[-6:]}.pdf",
            track_id=f"upload_20261103_101010_{doc_id[-8:]}",
            chunks_count=3,
        )
    async with pg.dedicated_connection() as conn:
        await conn.copy_records_to_table(
            "lightrag_doc_status", records=records, columns=COLUMNS
        )
        await conn.execute(
            "INSERT INTO lightrag_workspace_owners VALUES ('ws1', 'sso:1', 'a.bin@unimas.my', 0, 0)"
        )
    memory.owner_rows["ws1"] = "a.bin@unimas.my"
    await pg._refresh_tables(force=True)

    def norm(value):
        if isinstance(value, list):
            return [norm(v) for v in value]
        if isinstance(value, dict):
            return {k: norm(v) for k, v in value.items()}
        if isinstance(value, datetime):
            return value.replace(tzinfo=None, microsecond=0)
        return value

    assert norm(await pg.backlog_totals()) == norm(await memory.backlog_totals())
    assert await pg.dup_count() == await memory.dup_count() == 6
    assert await pg.cancelled_count() == await memory.cancelled_count() > 0
    for sort in ("oldest", "failed", "inflight", "name", "activity"):
        for desc in (False, True):
            for has in (frozenset(), frozenset({"failed"}), frozenset({"stuck"})):
                kwargs = dict(
                    q=None,
                    has=has,
                    sort=sort,
                    descending=desc,
                    page=Page(limit=4, offset=0),
                    stale_minutes=60,
                    email_workspaces=["ws2"],
                    paused_workspaces=[],
                )
                got = await pg.backlog_page(**kwargs)
                want = await memory.backlog_page(**kwargs)
                assert norm(got) == norm(want), (sort, desc, has)
    search = dict(
        has=frozenset(),
        sort="name",
        descending=False,
        page=Page(limit=10, offset=0),
        stale_minutes=60,
        email_workspaces=["ws2"],
        paused_workspaces=["ws3"],
    )
    rows, total = await pg.backlog_page(q="A.BIN", **search)
    assert (
        total == 1
        and rows[0]["workspace"] == "ws1"
        and rows[0]["owner"] == "a.bin@unimas.my"
    )
    rows, total = await pg.backlog_page(
        q="ws_", **search
    )  # "_" is literal, not a wildcard
    assert total == 0
    rows, total = await pg.backlog_page(
        q=None, **{**search, "has": frozenset({"email"})}
    )
    assert [r["workspace"] for r in rows] == ["ws2"]
    rows, total = await pg.backlog_page(
        q=None, **{**search, "page": Page(limit=10, offset=50)}
    )
    assert rows == [] and total == 6
    assert norm(await pg.stale_docs(60, Page(10, 0))) == norm(
        await memory.stale_docs(60, Page(10, 0))
    )
    assert norm(await pg.live_candidates(50)) == norm(await memory.live_candidates(50))
    assert await pg.pending_candidates(50) == await memory.pending_candidates(50)
    for group in ("signature", "workspace"):
        for include in (False, True):
            got = await pg.legacy_failures(24, group, include)
            want = await memory.legacy_failures(24, group, include)
            assert norm(got["groups"]) == norm(want["groups"]), group
            assert got["cancelled"]["count"] == want["cancelled"]["count"]
        top = (await pg.legacy_failures(24, group, False))["groups"][0]
        key = top["key"]
        got = await pg.failure_docs(24, group, key, Page(5, 0))
        assert norm(got) == norm(await memory.failure_docs(24, group, key, Page(5, 0)))
        # A drill lists exactly the rows its group counted (no classified ones).
        assert got[1] == top["count"], group
    # ws5 holds a classified failure (metadata.failure): its legacy drill
    # leaves it out, as its legacy group count does.
    groups = (await pg.legacy_failures(24, "workspace", False))["groups"]
    counted = next((g["count"] for g in groups if g["key"] == "ws5"), 0)
    assert (await pg.failure_docs(24, "workspace", "ws5", Page(50, 0)))[1] == counted
    got = await pg.failure_docs(24, "cancelled", "cancelled", Page(50, 0))
    assert norm(got) == norm(
        await memory.failure_docs(24, "cancelled", "cancelled", Page(50, 0))
    )
    assert norm(await pg.workspace_counts("ws1")) == norm(
        await memory.workspace_counts("ws1")
    )
    assert norm(await pg.workspace_documents("ws1", "failed", Page(5, 0))) == norm(
        await memory.workspace_documents("ws1", "failed", Page(5, 0))
    )
    doc = await pg.document("ws5", "doc-5-04")
    assert doc["metadata"] == {"failure": {"error_kind": "provider_budget"}}
    assert await pg.owners(["ws1", "ws2"]) == {"ws1": "a.bin@unimas.my"}


async def test_kv_and_leases(pg):
    await _reset(pg)
    await pg.kv_put("thresholds", {"stale_doc_minutes": 30})
    assert await pg.kv_get("thresholds") == {"stale_doc_minutes": 30}
    await pg.kv_merge_many({"resync:ws1": {"first_pending_at": 1.0}})
    await pg.kv_merge_many({"resync:ws1": {"attempts": 2}})
    assert await pg.kv_get("resync:ws1") == {"first_pending_at": 1.0, "attempts": 2}
    assert await pg.kv_cas("action:ws1", None, {"actor": "a"})
    assert not await pg.kv_cas("action:ws1", None, {"actor": "b"})
    assert await pg.kv_cas("action:ws1", {"actor": "a"}, {"actor": "b"})
    assert await pg.kv_get_many(["action:ws1", "missing"]) == {
        "action:ws1": {"actor": "b"}
    }

    # Holder stored, renewal by the holder, takeover only after expiry.
    assert await pg.try_lease("ops:test", "worker-a", 1.0)
    lease = await pg.kv_get("lease:ops:test")
    assert lease["holder"] == "worker-a"
    assert await pg.try_lease("ops:test", "worker-a", 1.0)
    assert not await pg.try_lease("ops:test", "worker-b", 1.0)
    await asyncio.sleep(1.2)
    assert await pg.try_lease("ops:test", "worker-b", 30.0)
    assert not await pg.try_lease("ops:test", "worker-a", 30.0)
    await pg.release_lease("ops:test", "worker-a")  # not the holder: no effect
    assert (await pg.kv_get("lease:ops:test"))["holder"] == "worker-b"
    await pg.release_lease("ops:test", "worker-b")
    assert await pg.kv_get("lease:ops:test") is None


async def test_heartbeat_rows_samples_and_supersede(pg):
    async with pg.dedicated_connection() as conn:
        await conn.execute("TRUNCATE lightrag_ops_workers")
        await conn.execute("TRUNCATE lightrag_ops_worker_samples")
    started = datetime.now(timezone.utc) - timedelta(minutes=10)
    old_id, new_id = "a" * 32, "b" * 32
    row = {
        "worker_id": old_id,
        "host": "it-host",
        "pid": 811,
        "pid_create_time": 1.5,
        "mode": "gunicorn",
        "version": "it",
        "started_at": started,
        "leases_held": ["ops:probes"],
        "snapshot": {"rss": 123, "pool": {"wait_ms": 2.0}},
    }
    await pg.worker_beat(
        row, {"cpu_pct": 12.5, "rss": 123, "loop_lag_ms": 3, "pool_wait_ms": 1}
    )
    await pg.worker_beat(
        row, {"cpu_pct": 20.0, "rss": 456, "loop_lag_ms": 9, "pool_wait_ms": 2}
    )
    async with pg.dedicated_connection() as conn:
        await conn.execute(
            "UPDATE lightrag_ops_workers SET heartbeat_at = now() - interval '2 minutes'"
        )
    new_row = {
        **row,
        "worker_id": new_id,
        "pid": 812,
        "started_at": datetime.now(timezone.utc),
    }
    await pg.worker_beat(new_row)
    # A row whose process still runs is never superseded.
    assert (
        await pg.supersede_workers(
            "it-host", new_id, new_row["started_at"], 45, alive=lambda pid, ct: True
        )
        == 0
    )
    seen = []
    assert (
        await pg.supersede_workers(
            "it-host",
            new_id,
            new_row["started_at"],
            45,
            alive=lambda pid, ct: seen.append((pid, ct)) or False,
        )
        == 1
    )
    assert seen == [(811, 1.5)]
    rows = {r["worker_id"]: r for r in await pg.workers(10)}
    assert rows[old_id]["superseded"] is True and rows[new_id]["superseded"] is False
    assert rows[new_id]["heartbeat_age_s"] < 5
    assert rows[old_id]["snapshot"]["pool"] == {"wait_ms": 2.0}
    assert rows[old_id]["leases_held"] == ["ops:probes"]
    [sample] = await pg.worker_history(old_id, 24, 60)
    assert sample["rss"] == 456 and sample["loop_lag"] == 9
    await pg.worker_exited(new_id)
    exited = {r["worker_id"]: r for r in await pg.workers(10)}
    assert exited[new_id]["exited_at"] is not None
    async with pg.dedicated_connection() as conn:
        await conn.execute(
            "UPDATE lightrag_ops_workers SET heartbeat_at = now() - interval '25 hours'"
            " WHERE worker_id = $1::uuid",
            old_id,
        )
    await pg.cleanup_workers(24, 7)
    assert [r["worker_id"] for r in await pg.workers(10)] == [new_id]
    assert pg.pool_stats()["max"] >= 1
    assert await pg.pool_wait_ms(2.0) >= 0


async def test_probe_history_and_postgres_health(pg):
    async with pg.dedicated_connection() as conn:
        await conn.execute("TRUNCATE lightrag_ops_probes")
    await pg.probe_write(
        [
            {
                "dependency": "neo4j",
                "status": "down",
                "latency_ms": None,
                "error_class": "timeout",
                "detail": {},
            },
            {
                "dependency": "postgres",
                "status": "ok",
                "latency_ms": 2,
                "error_class": None,
                "detail": {"connect_ms": 1},
            },
        ]
    )
    await pg.probe_write(
        [
            {
                "dependency": "neo4j",
                "status": "ok",
                "latency_ms": 12,
                "error_class": None,
                "detail": {},
            }
        ]
    )
    strips = {
        (r["dependency"], r["total"], r["down"])
        for r in await pg.probe_strips(24, 1800)
    }
    assert ("neo4j", 2, 1) in strips and ("postgres", 1, 0) in strips
    series = await pg.probe_series("neo4j", 24, 600)
    assert sum(r["total"] for r in series) == 2
    recent = await pg.probe_recent("neo4j", 50)
    assert [r["status"] for r in recent] == ["ok", "down"]
    assert await pg.probe_error_classes("neo4j", 24) == [{"key": "timeout", "count": 1}]
    await pg.probe_cleanup(7)
    assert len(await pg.probe_recent("neo4j", 50)) == 2

    # A real lock wait: one session holds a row lock, another waits on it.
    async with pg.dedicated_connection() as holder, pg.dedicated_connection() as waiter:
        await holder.execute(
            "CREATE TABLE IF NOT EXISTS ops_it_lock (id int primary key)"
        )
        await holder.execute(
            "INSERT INTO ops_it_lock VALUES (1) ON CONFLICT DO NOTHING"
        )
        tx = holder.transaction()
        await tx.start()
        await holder.execute("UPDATE ops_it_lock SET id = 1 WHERE id = 1")
        blocked = asyncio.create_task(
            waiter.execute("UPDATE ops_it_lock SET id = 1 WHERE id = 1 /* 4242 'x' */")
        )
        await asyncio.sleep(0.5)
        try:
            health = await pg.postgres_health(extras=True, timeout_s=5)
        finally:
            await tx.rollback()
            await blocked
            await holder.execute("DROP TABLE ops_it_lock")
    assert health["latency_ms"] >= 0 and health["max_connections"] > 0
    assert health["connections"] >= 3 and health["db_size_bytes"] > 0
    assert health["db_stats"]["xact_commit"] > 0
    assert {"xact_rollback", "deadlocks", "temp_bytes", "blks_hit"} <= set(
        health["db_stats"]
    )
    waiting = [b for b in health["blocking"] if b["blocked_by"]]
    assert waiting and waiting[0]["statement_type"] == "UPDATE"
    assert len(waiting[0]["fingerprint"]) == 16 and waiting[0]["query_length"] > 20
    assert "query" not in waiting[0]
    simple = await pg.postgres_health(extras=False, timeout_s=5)
    assert set(simple) == {"latency_ms", "connect_ms"}


async def test_forget_workspace(pg):
    await _reset(pg)
    ws = "a_bin_unimas_my"
    for key in (f"resync:{ws}", f"kick:{ws}", f"action:{ws}", f"pause:{ws}:extraction"):
        await pg.kv_put(key, {"x": 1})
    await pg.kv_put(f"pause:{ws}x:extraction", {"keep": True})  # another workspace
    await pg.kv_put(
        "email_queue_summary",
        {
            "items": [
                {"workspace": ws, "pending": 1},
                {"workspace": "other", "pending": 2},
            ]
        },
    )
    async with pg.dedicated_connection() as conn:
        await conn.execute(
            """INSERT INTO lightrag_ops_audit (actor, action, workspace, target, phase)
               VALUES ('admin', 'cancel', $1, $1, 'result')""",
            ws,
        )
    await pg.forget_workspace(ws)
    remaining = await pg.kv_get_many(
        [
            f"resync:{ws}",
            f"kick:{ws}",
            f"action:{ws}",
            f"pause:{ws}:extraction",
            f"pause:{ws}x:extraction",
        ]
    )
    assert remaining == {f"pause:{ws}x:extraction": {"keep": True}}
    assert (await pg.kv_get("email_queue_summary"))["items"] == [
        {"workspace": "other", "pending": 2}
    ]
    async with pg.dedicated_connection() as conn:
        row = await conn.fetchrow(
            "SELECT workspace, target, actor FROM lightrag_ops_audit"
        )
    assert dict(row) == {
        "workspace": workspace_hash(ws),
        "target": workspace_hash(ws),
        "actor": "admin",
    }


# -- run tracking ---------------------------------------------------------------------


def _run_row(run_id: str, ws: str, status: str, started: float, **extra) -> dict:
    return {
        "run_id": run_id,
        "workspace": ws,
        "kind": extra.pop("kind", "process"),
        "trigger": "upload",
        "status": status,
        "worker_id": "0123456789abcdef0123456789abcdef",
        "pid": 7,
        "started_at": started,
        "heartbeat_at": started + 10,
        "progress_at": started + 5,
        "finished_at": started + 20 if status != "running" else None,
        "docs_total": 3,
        "docs_done": 2,
        "docs_failed": 1,
        "stage": "extract",
        "label": "x [3 files]",
        "error_kind": None,
        "error_summary": None,
        "request_id": None,
        **extra,
    }


async def test_run_tracking_matches_memory_store(pg):
    await _reset(pg)
    memory = MemoryOpsStore()
    now = time.time()
    ids = [f"{i:032x}" for i in range(1, 9)]
    rows = [
        _run_row(ids[0], "ws1", "running", now - 600),
        _run_row(ids[1], "ws1", "succeeded", now - 3600, kind="delete"),
        _run_row(ids[2], "ws2", "running", now - 900),
        _run_row(ids[3], "ws2", "failed", now - 7200),
        _run_row(ids[4], "ws3", "running", now - 300),
        _run_row(ids[5], "ws3", "cancelled", now - 40 * 86400),
    ]
    failures = [
        {
            "ts": now - 60 * i,
            "workspace": f"ws{1 + i % 3}",
            "doc_id": f"doc-{i % 4}",
            "track_id": None,
            "run_id": ids[i % 3],
            "stage": ("parse", "extract", "merge")[i % 3],
            "error_kind": ("timeout", "rate_limited", "cancelled", "storage")[i % 4],
            "http_status": 429 if i % 4 == 1 else None,
            "exc_type": "x.Error",
            "message": f"masked message {i}",
            "attempt": 1 + i // 4,
            "model": None if i % 5 == 0 else "gpt-4o-mini",
        }
        for i in range(12)
    ]
    for store in (pg, memory):
        await store.runs_write(rows)
        await store.doc_failures_write(failures)
        # A late running write never reopens a finished run.
        await store.runs_write([_run_row(ids[1], "ws1", "running", now - 3600)])
        assert await store.runs_orphan(
            [ids[2], ids[1]], {ids[2]: "host_gone", ids[1]: "pid_gone"}
        ) == [ids[2]]
        # finish() after orphaning: final status, was_orphaned kept.
        await store.runs_write([_run_row(ids[2], "ws2", "succeeded", now - 900)])
        # A NUL character never fails a write (Postgres text rejects it).
        await store.runs_write(
            [{**_run_row(ids[4], "ws3", "running", now - 300), "label": "a\x00b"}]
        )

    def norm(value):
        if isinstance(value, list):
            return [norm(v) for v in value]
        if isinstance(value, tuple):
            return tuple(norm(v) for v in value)
        if isinstance(value, dict):
            return {
                k: norm(v)
                for k, v in value.items()
                if not k.endswith("_age_s")
                and k != "id"
                and not k.startswith("worker_")
            }
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).replace(microsecond=0)
        if isinstance(value, float):
            return round(value, 3)
        return value

    got = await pg.run_get(ids[2])
    assert got["status"] == "succeeded" and got["was_orphaned"] is True
    # How the owner's death was confirmed, and the host its PID lived on.
    assert got["orphaned_by"] == "host_gone"
    assert got["run_host"] == (await memory.run_get(ids[2]))["run_host"]
    assert (await pg.run_get(ids[1]))["status"] == "succeeded"
    assert norm(await pg.running_runs(10)) == norm(await memory.running_runs(10))
    for kwargs in (
        {"workspace": None, "kind": None, "status": None},
        {"workspace": "ws1", "kind": None, "status": None},
        {"workspace": None, "kind": "delete", "status": "succeeded"},
    ):
        first = await pg.runs_page(**kwargs, before=None, limit=2)
        mfirst = await memory.runs_page(**kwargs, before=None, limit=2)
        assert norm(first) == norm(mfirst)
        if first:
            cursor = (first[-1]["started_at"], first[-1]["run_id"])
            mcursor = (mfirst[-1]["started_at"], mfirst[-1]["run_id"])
            assert norm(await pg.runs_page(**kwargs, before=cursor, limit=10)) == norm(
                await memory.runs_page(**kwargs, before=mcursor, limit=10)
            )
    assert norm(await pg.last_runs(["ws1", "ws2", "nope"])) == norm(
        await memory.last_runs(["ws1", "ws2", "nope"])
    )
    assert norm(await pg.run_failures(ids[0], 50)) == norm(
        await memory.run_failures(ids[0], 50)
    )
    for group in ("error_kind", "stage", "model", "workspace"):
        for include in (False, True):
            got = await pg.failure_groups(24, group, include)
            want = await memory.failure_groups(24, group, include)
            assert norm(got) == norm(want), (group, include)
        key = (await pg.failure_groups(24, group, False))["groups"][0]["key"]
        assert norm(await pg.failure_group_docs(24, group, key, Page(3, 0))) == norm(
            await memory.failure_group_docs(24, group, key, Page(3, 0))
        )
    assert norm(
        await pg.failure_group_docs(24, "error_kind", "cancelled", Page(10, 0))
    ) == norm(
        await memory.failure_group_docs(24, "error_kind", "cancelled", Page(10, 0))
    )
    assert norm(await pg.failures_by_kind("ws1", 24)) == norm(
        await memory.failures_by_kind("ws1", 24)
    )
    assert norm(await pg.doc_failures("ws2", "doc-1", 10)) == norm(
        await memory.doc_failures("ws2", "doc-1", 10)
    )
    end = datetime.now(timezone.utc) + timedelta(minutes=1)
    assert norm(await pg.run_events(end - timedelta(days=1), end, 10)) == norm(
        await memory.run_events(end - timedelta(days=1), end, 10)
    )

    # Pauses: open ones only; merge_existing never creates keys.
    await pg.kv_put("pause:ws1:extraction", {"workspace": "ws1", "cleared_at": None})
    await pg.kv_put("pause:ws2:query", {"workspace": "ws2", "cleared_at": now})
    await pg.kv_merge_existing(
        {"pause:ws1:vision": {"cleared_at": now}, "pause:ws1:extraction": {"x": 1}}
    )
    assert [p["key"] for p in await pg.open_pauses(10)] == ["pause:ws1:extraction"]
    assert await pg.kv_get("pause:ws1:vision") is None
    # Counts are SQL aggregates; pages come from SQL, oldest first.
    for i in range(3):
        value = {
            "workspace": f"p{i}",
            "slot": "extraction",
            "kind": "budget_exhausted",
            "since": now - 100 + i,
            "cleared_at": None,
        }
        await pg.kv_put(f"pause:p{i}:extraction", value)
        await memory.kv_put(f"pause:p{i}:extraction", value)
    await memory.kv_put(
        "pause:ws1:extraction", {"workspace": "ws1", "cleared_at": None}
    )
    assert await pg.pause_counts() == await memory.pause_counts()
    assert (await pg.pause_counts())["total"] == 4
    for offset in (0, 2):
        assert [p["key"] for p in await pg.pauses_page(2, offset)] == [
            p["key"] for p in await memory.pauses_page(2, offset)
        ]

    # Retention and deletion.
    await pg.runs_cleanup(30, 90)
    assert await pg.run_get(ids[5]) is None
    await pg.forget_workspace("ws1")
    assert await pg.last_runs(["ws1"]) == {}
    assert await pg.failures_by_kind("ws1", 24) == []
    # A tombstone keeps other workers' buffered rows from coming back.
    tombstone = await pg.kv_get(deleted_key("ws1"))
    assert isinstance(tombstone["at"], float)

    # The running-runs read can use the partial index.
    async with pg.dedicated_connection() as conn:
        await conn.execute("SET enable_seqscan = off")
        plan = await conn.fetchval(
            """EXPLAIN (FORMAT JSON) SELECT run_id FROM lightrag_ops_runs
            WHERE status = 'running' ORDER BY heartbeat_at"""
        )
    assert "idx_ops_runs_running" in json.dumps(plan)


# -- release scale ------------------------------------------------------------------


def _seq_scans(plan) -> list[str]:
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("Node Type") == "Seq Scan":
                found.append(str(node.get("Relation Name")))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(plan)
    return found


class _Capture(PostgresOpsStore):
    """Records the SQL and arguments of each read."""

    def __init__(self, inner: PostgresOpsStore):
        super().__init__()
        self._db = inner._db
        self._tables = inner._tables
        self._tables_at = inner._tables_at
        self.ready = self.doc_status_ready = True
        self.statements: list[tuple[str, tuple]] = []

    async def _read(self, sql, *args):
        self.statements.append((sql, args))
        return await super()._read(sql, *args)


def _literal(value) -> str:
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


_STATEMENTS = iter(range(1_000_000))


async def _explain(conn, sql: str, args: tuple, generic: bool) -> list[str]:
    name = f"ops_q{next(_STATEMENTS)}"
    await conn.execute(
        f"SET plan_cache_mode = {'force_generic_plan' if generic else 'auto'}"
    )
    await conn.execute(f"PREPARE {name} AS {sql}")
    params = f"({', '.join(_literal(a) for a in args)})" if args else ""
    raw = await conn.fetchval(f"EXPLAIN (FORMAT JSON) EXECUTE {name}{params}")
    await conn.execute(f"DEALLOCATE {name}")
    return _seq_scans(json.loads(raw) if isinstance(raw, str) else raw)


async def _median_ms(call, runs: int = 7) -> float:
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        await call()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


async def test_release_scale_plans_and_budgets(pg):
    await _reset(pg)
    rng = random.Random(42)
    started = time.perf_counter()
    async with pg.dedicated_connection() as conn:
        # Build over the seeded rows, not over an empty table.
        for spec in indexes.OPS_INDEXES:
            await conn.execute(f"DROP INDEX IF EXISTS {spec.name}")
        await _copy(conn, _seed_rows(WORKSPACES, DOCS_PER_WS, rng))
        total = await conn.fetchval("SELECT count(*) FROM lightrag_doc_status")
    seed_s = time.perf_counter() - started

    # The builder runs on its own connection and is idempotent.
    started = time.perf_counter()
    async with pg.dedicated_connection() as conn:
        outcome = await indexes.build_indexes(conn)
    build_s = time.perf_counter() - started
    assert outcome["idx_lightrag_doc_status_inflight"] == "created"
    assert outcome["idx_lightrag_doc_status_dup"] == "created"
    assert outcome["idx_lightrag_doc_status_id"] == "created"
    assert outcome["idx_usage_events_kind_ts"] == "created"
    async with pg.dedicated_connection() as conn:
        again = await indexes.build_indexes(conn)
        assert set(again.values()) <= {"valid", "skipped"}
        # A concurrent build that died leaves an INVALID index: it is rebuilt.
        await conn.execute(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = "
            "'idx_lightrag_doc_status_id'::regclass"
        )
        assert [s.name for s in await indexes.missing_indexes(conn)] == [
            "idx_lightrag_doc_status_id"
        ]
        assert (await indexes.build_indexes(conn))[
            "idx_lightrag_doc_status_id"
        ] == "rebuilt"
        # Another session holding the builder lock makes this one skip.
        async with pg.dedicated_connection() as other:
            await other.fetchval(
                "SELECT pg_advisory_lock($1)", indexes.ADVISORY_LOCK_KEY
            )
            assert set((await indexes.build_indexes(conn)).values()) == {"locked"}
            await other.fetchval(
                "SELECT pg_advisory_unlock($1)", indexes.ADVISORY_LOCK_KEY
            )
        # Steady state: autovacuum keeps the visibility map current in production.
        await conn.execute("VACUUM ANALYZE lightrag_doc_status")

    capture = _Capture(pg)
    await capture.backlog_totals()
    await capture.stale_docs(60, Page(50, 0))
    await capture.live_candidates(50)
    await capture.pending_candidates(50)
    await capture.dup_count()
    await capture.cancelled_count()
    async with pg.dedicated_connection() as conn:
        for sql, args in capture.statements:
            for generic in (False, True):
                scans = await _explain(conn, sql, args, generic)
                assert "lightrag_doc_status" not in [s.lower() for s in scans], (
                    sql.split()[1:4],
                    generic,
                    scans,
                )

    page_args = dict(
        q=None,
        has=frozenset(),
        sort="oldest",
        descending=False,
        page=Page(50, 0),
        stale_minutes=60,
        email_workspaces=[],
        paused_workspaces=[],
    )
    measured = {
        "Q1 backlog totals": await _median_ms(pg.backlog_totals),
        "Q1 dup count": await _median_ms(pg.dup_count),
        "Q1 cancelled count": await _median_ms(pg.cancelled_count),
        "Q2 backlog page": await _median_ms(lambda: pg.backlog_page(**page_args)),
        "Q2 backlog search": await _median_ms(
            lambda: pg.backlog_page(**{**page_args, "q": "user12"})
        ),
        "Q3 stale docs": await _median_ms(lambda: pg.stale_docs(60, Page(50, 0))),
        "Q6 live candidates": await _median_ms(lambda: pg.live_candidates(50)),
        "Q4 legacy failures 24h": await _median_ms(
            lambda: pg.legacy_failures(24, "signature", False)
        ),
        "workspace counts": await _median_ms(
            lambda: pg.workspace_counts("user0042_unimas_my")
        ),
    }
    app_state = SimpleNamespace(
        default_rag=None, ops_server=SimpleNamespace(rag=None), insights=None
    )
    service = OpsService(store=pg, reader=NamespaceReader(), app_state=app_state)
    workers = {"mode": "uvicorn", "expected": 1, "live": 1}
    await service.overview(workers)
    service.cache._entries.pop(("overview", "uvicorn", 1, 1), None)
    measured["/overview warm"] = await _median_ms(
        lambda: _overview_uncached(service, workers)
    )
    measured["/overview cold"] = await _median_ms(
        lambda: _overview_cold(service, workers)
    )
    report = ", ".join(f"{k}={v:.1f}ms" for k, v in measured.items())
    print(
        f"\n[ops-it] rows={total} workspaces={WORKSPACES} seed={seed_s:.1f}s "
        f"index_build={build_s:.1f}s {report}",
        file=sys.stderr,
    )
    budgets = {
        "Q1 backlog totals": 50,
        "Q2 backlog page": 150,
        "Q3 stale docs": 50,
        "Q6 live candidates": 50,
        "/overview warm": 300,
        "/overview cold": 300,
    }
    for name, budget in budgets.items():
        assert measured[name] < budget * BUDGET_SLACK, (name, measured[name])


async def _overview_uncached(service: OpsService, workers: dict) -> None:
    # Warm sub-caches, but recompute the composite each time.
    service.cache._entries.pop(("overview", "uvicorn", 1, 1), None)
    await service.overview(workers)


async def _overview_cold(service: OpsService, workers: dict) -> None:
    # Every query runs: what the first viewer after a cache expiry waits for.
    service.cache.clear()
    await service.overview(workers)
