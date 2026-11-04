"""Operations console alerts against a real Postgres (PR-8).

Proves what the memory store can only imitate: the partial unique index
``idx_ops_alerts_open`` (one open row per fingerprint, also when evaluators
race on separate connections), the conditional-UPDATE email claim (one
winner among concurrent claimers), the resolve claim, acks, keyset paging,
the export window, 180-day retention and the workspace-deletion rewrite.

Runs only with ``LIGHTRAG_RUN_INTEGRATION=true`` and ``POSTGRES_HOST`` /
``POSTGRES_PORT`` / ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` /
``POSTGRES_DATABASE`` pointing at a throwaway database (it truncates
``LIGHTRAG_OPS_ALERTS`` and ``LIGHTRAG_OPS_KV``).
"""

from __future__ import annotations

import asyncio
import os
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

from lightrag.api.ops.alerts import AlertEvaluator  # noqa: E402
from lightrag.api.ops.service import OpsService  # noqa: E402
from lightrag.api.ops.snapshot import NamespaceReader  # noqa: E402
from lightrag.api.ops.store import PostgresOpsStore, workspace_hash  # noqa: E402


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def pg():
    store = PostgresOpsStore()
    await store.initialize()
    yield store
    await store.finalize()


@pytest_asyncio.fixture(loop_scope="module")
async def clean(pg):
    async with pg.dedicated_connection() as conn:
        await conn.execute("TRUNCATE lightrag_ops_alerts RESTART IDENTITY")
        await conn.execute("TRUNCATE lightrag_ops_kv")
    return pg


def _row(fp: str, rule: str = "pg_probe", severity: str = "critical", **scope):
    return {
        "rule": rule,
        "fingerprint": fp,
        "severity": severity,
        "title": f"{rule} title",
        "scope": scope or {"count": 1},
        "link": "#/?ops=1&otab=health",
        "runbook": "check it",
    }


async def test_partial_unique_index_allows_one_open_row(clean):
    store = clean
    async with store.dedicated_connection() as conn:
        index = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_ops_alerts_open'"
        )
        assert "UNIQUE" in index and "resolved_at IS NULL" in index
        insert = """INSERT INTO lightrag_ops_alerts (rule, fingerprint, severity, state,
            title, scope, first_seen, last_seen, resolved_at)
            VALUES ('r', 'fp', 'warning', $1, 't', '{}'::jsonb, now(), now(), $2)"""
        await conn.execute(insert, "resolved", datetime.now(timezone.utc))
        await conn.execute(insert, "resolved", datetime.now(timezone.utc))
        await conn.execute(insert, "firing", None)
        import asyncpg

        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(insert, "firing", None)


async def test_racing_upserts_keep_one_open_row(clean):
    store = clean
    results = await asyncio.gather(
        *(store.alerts_upsert([_row("dependency_down:neo4j")]) for _ in range(12))
    )
    ids = {r[0]["id"] for r in results}
    assert len(ids) == 1
    assert sum(1 for r in results if r[0]["inserted"]) == 1
    open_rows = await store.alerts_open(10)
    assert [r["fingerprint"] for r in open_rows] == ["dependency_down:neo4j"]
    # Touch updates the open row in place.
    [touched] = await store.alerts_upsert(
        [_row("dependency_down:neo4j", severity="warning")]
    )
    assert touched["inserted"] is False and touched["id"] in ids
    row = await store.alert_get(touched["id"])
    assert row["severity"] == "warning" and row["scope"] == {"count": 1}


async def test_resolve_refire_ack_and_paging(clean):
    store = clean
    [first] = await store.alerts_upsert([_row("disk_free", "disk_free")])
    resolved = await store.alerts_resolve(["disk_free"])
    assert [r["id"] for r in resolved] == [first["id"]]
    assert resolved[0]["state"] == "resolved" and resolved[0]["resolved_at"]
    [again] = await store.alerts_upsert([_row("disk_free", "disk_free")])
    assert again["inserted"] and again["id"] != first["id"]
    history = await store.alerts_history("disk_free", 10)
    assert [r["id"] for r in history] == [again["id"], first["id"]]
    code, row = await store.alert_ack(again["id"], "khairil", "on it")
    assert code == "acked" and row["state"] == "firing" and row["acked_by"] == "khairil"
    assert (await store.alert_ack(again["id"], "x", None))[0] == "already_acked"
    assert (await store.alert_ack(first["id"], "x", None))[0] == "resolved"
    assert (await store.alert_ack(999999, "x", None))[0] == "not_found"
    for i in range(5):
        await store.alerts_upsert([_row(f"fp{i}")])
    page = await store.alerts_page(None, None, 3)
    assert [r["id"] for r in page] == sorted([r["id"] for r in page], reverse=True)
    rest = await store.alerts_page(None, page[-1]["id"], 100)
    assert all(r["id"] < page[-1]["id"] for r in rest)
    assert {r["id"] for r in await store.alerts_page("resolved", None, 100)} == {
        first["id"]
    }
    now = datetime.now(timezone.utc)
    window = await store.alerts_window(
        now - timedelta(hours=1), now + timedelta(minutes=1), 100
    )
    assert len(window) == 7


async def test_email_claims_have_one_winner(clean):
    store = clean
    [row] = await store.alerts_upsert([_row("pg_probe")])
    claims = await asyncio.gather(
        *(store.alert_claim_email(row["id"], "open", 3600) for _ in range(8))
    )
    winners = [c for c in claims if c is not None]
    assert len(winners) == 1
    assert await store.alert_claim_email(row["id"], "open", 3600) is None
    # A failed send undoes the claim; the next claim wins again.
    await store.alert_unclaim_email(row["id"], winners[0], None)
    claimed = await store.alert_claim_email(row["id"], "open", 3600)
    assert claimed is not None
    # Resolve claim: only after resolving, only once.
    assert await store.alert_claim_email(row["id"], "resolve", 0) is None
    await store.alerts_resolve(["pg_probe"])
    claims = await asyncio.gather(
        *(store.alert_claim_email(row["id"], "resolve", 0) for _ in range(5))
    )
    assert sum(1 for c in claims if c is not None) == 1
    assert await store.alert_claim_email(row["id"], "resolve", 0) is None


async def test_retention_and_workspace_deletion(clean):
    store = clean
    await store.alerts_upsert(
        [
            _row("old", workspaces=["fsktm", "other"], count=2),
            _row("open", workspace="fsktm"),
        ]
    )
    await store.alerts_resolve(["old"])
    async with store.dedicated_connection() as conn:
        await conn.execute(
            "UPDATE lightrag_ops_alerts SET resolved_at = now() - interval '181 days' "
            "WHERE fingerprint = 'old'"
        )
    await store._refresh_tables(force=True)
    await store.forget_workspace("fsktm")
    rows = {r["fingerprint"]: r for r in await store.alerts_page(None, None, 10)}
    assert rows["old"]["scope"]["workspaces"] == [workspace_hash("fsktm"), "other"]
    assert rows["open"]["scope"]["workspace"] == workspace_hash("fsktm")
    assert await store.alerts_cleanup(180) == 1
    assert [r["fingerprint"] for r in await store.alerts_page(None, None, 10)] == [
        "open"
    ]


async def test_three_evaluators_send_one_email(clean, monkeypatch):
    monkeypatch.setenv("OPS_ALERT_EMAIL", "true")
    store = clean
    sent = []

    def send(settings, recipients, msg):
        sent.append(msg["Subject"])

    settings = SimpleNamespace(
        enabled=True,
        sender="alerts@example.org",
        timeout=5,
        repeat_seconds=86400,
        admin_recipients=("ops@example.org",),
    )
    workers = []
    for _ in range(3):
        service = OpsService(
            store=store, reader=NamespaceReader(), app_state=SimpleNamespace()
        )
        workers.append(
            AlertEvaluator(
                service, process=lambda: {}, send=send, settings=lambda: settings
            )
        )
    snapshot = {
        "at_epoch": time.time(),
        "items": [
            {
                "id": "run_orphaned",
                "rule": "run_orphaned",
                "severity": "critical",
                "title": "2 run(s) lost their worker",
                "params": {"count": 2, "workspaces": ["fsktm"]},
                "link": None,
            }
        ],
        "failed_sections": [],
        "deploy_recent": False,
    }
    outcomes = await asyncio.gather(*(w.apply(dict(snapshot), None) for w in workers))
    assert len(await store.alerts_open(10)) == 1
    counts = await asyncio.gather(*(w.notify(o) for w, o in zip(workers, outcomes)))
    assert sorted(counts) == [0, 0, 1] and sent == ["[LightRAG] CRITICAL run_orphaned"]


async def test_a_refire_within_the_quiet_window_is_carried_not_emailed(
    clean, monkeypatch
):
    """The flapping guard reads the fingerprint's history and claims the new
    row without an email, with the database's own timestamps."""
    monkeypatch.setenv("OPS_ALERT_EMAIL", "true")
    store = clean
    sent = []
    settings = SimpleNamespace(
        enabled=True,
        sender="alerts@example.org",
        timeout=5,
        repeat_seconds=86400,
        admin_recipients=("ops@example.org",),
    )
    service = OpsService(
        store=store, reader=NamespaceReader(), app_state=SimpleNamespace()
    )
    ev = AlertEvaluator(
        service,
        process=lambda: {},
        send=lambda s, r, m: sent.append(m["Subject"]),
        settings=lambda: settings,
    )
    item = {
        "id": "dependency_down:neo4j",
        "rule": "dependency_down",
        "severity": "critical",
        "title": "neo4j is down",
        "params": {"dependency": "neo4j"},
        "link": None,
    }

    def snap(*items):
        return {
            "at_epoch": time.time(),
            "items": [dict(i) for i in items],
            "failed_sections": [],
            "deploy_recent": False,
        }

    previous = None
    for items in ([item], [], [], [item]):
        current = snap(*items)
        outcome = await ev.apply(current, previous)
        await ev.notify(outcome)
        previous = current
    rows = await store.alerts_history("dependency_down:neo4j", 10)
    assert len(rows) == 2 and rows[0]["resolved_at"] is None
    assert rows[0]["last_emailed_at"] is not None  # carried
    assert sent == ["[LightRAG] CRITICAL dependency_down"]
