"""Deleting a workspace removes its ops data and hashes it in kept audit rows."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from lightrag.api.ops.runtime import OpsRuntime
from lightrag.api.ops.service import OpsService
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import (
    MemoryOpsStore,
    PostgresOpsStore,
    deleted_key,
    workspace_hash,
)
from lightrag.usage import add_event_sink, emit_event, remove_event_sink

pytestmark = pytest.mark.offline

WS = "a_bin_unimas_my"


def _seeded() -> MemoryOpsStore:
    store = MemoryOpsStore()
    for key in (f"resync:{WS}", f"kick:{WS}", f"action:{WS}", f"pause:{WS}:extraction"):
        store.kv[key] = {"x": 1}
    store.kv[f"resync:{WS}x"] = {"other": True}
    store.kv["email_queue_summary"] = {
        "items": [{"workspace": WS, "pending": 2}, {"workspace": "other", "pending": 1}]
    }
    store.audit_rows.append(
        {"workspace": WS, "target": WS, "actor": "admin", "action": "cancel"}
    )
    store.workspace_rows[WS].append({"run_id": "r1"})
    store.alert_rows.append(
        {"title": f"Run orphaned in {WS}", "scope": {"workspace": WS}}
    )
    return store


async def test_memory_forget_semantics():
    store = _seeded()
    await store.forget_workspace(WS)
    # A tombstone (hashed name) keeps buffered rows from being written back.
    assert set(store.kv) == {
        f"resync:{WS}x",
        "email_queue_summary",
        deleted_key(WS),
    }
    assert WS not in deleted_key(WS)
    assert store.kv["email_queue_summary"]["items"] == [
        {"workspace": "other", "pending": 1}
    ]
    hashed = workspace_hash(WS)
    assert store.audit_rows == [
        {"workspace": hashed, "target": hashed, "actor": "admin", "action": "cancel"}
    ]
    assert WS not in store.workspace_rows
    assert store.alert_rows[0] == {
        "title": f"Run orphaned in {hashed}",
        "scope": {"workspace": hashed},
    }
    assert len(hashed) == 16 and WS not in str(store.audit_rows)


async def test_workspace_deleted_event_triggers_the_purge():
    store = _seeded()
    service = OpsService(
        store=store, reader=NamespaceReader(), app_state=SimpleNamespace()
    )
    service.cache._entries[("x",)] = (float("inf"), 1)
    runtime = OpsRuntime(
        worker_id="w", pid=0, reader=service.reader, service=service, store=store
    )
    add_event_sink(runtime.on_event)
    try:
        emit_event("workspace_deleted", workspace=WS)
        emit_event("deployment", version="1")  # ignored
        await asyncio.gather(*list(runtime.tasks))
    finally:
        remove_event_sink(runtime.on_event)
    assert f"kick:{WS}" not in store.kv
    assert store.audit_rows[0]["workspace"] == workspace_hash(WS)
    assert len(service.cache) == 0


async def test_purge_is_retried(monkeypatch):
    import lightrag.api.ops.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_FORGET_RETRY_DELAYS_S", (0.0, 0.0))
    store = _seeded()
    failures = {"n": 2}
    original = store.forget_workspace

    async def flaky(workspace):
        if failures["n"]:
            failures["n"] -= 1
            raise ConnectionError("db restarting")
        await original(workspace)

    store.forget_workspace = flaky
    service = OpsService(
        store=store, reader=NamespaceReader(), app_state=SimpleNamespace()
    )
    runtime = OpsRuntime(
        worker_id="w", pid=0, reader=service.reader, service=service, store=store
    )
    await runtime._forget(WS)
    assert f"kick:{WS}" not in store.kv


class _Conn:
    def __init__(self):
        self.statements: list[tuple[str, tuple]] = []

    def transaction(self):
        @asynccontextmanager
        async def tx():
            yield

        return tx()

    async def execute(self, sql, *args):
        self.statements.append((" ".join(sql.split()), args))


async def test_postgres_forget_sql(monkeypatch):
    store = PostgresOpsStore()
    conn = _Conn()

    @asynccontextmanager
    async def fake_conn():
        yield conn

    async def tables(force=False):
        return {"LIGHTRAG_DOC_STATUS", "LIGHTRAG_OPS_RUNS"}

    monkeypatch.setattr(store, "_conn", fake_conn)
    monkeypatch.setattr(store, "_refresh_tables", tables)
    await store.forget_workspace(WS)
    joined = "\n".join(sql for sql, _ in conn.statements)
    kv_delete = next(
        s for s in conn.statements if s[0].startswith("DELETE FROM LIGHTRAG_OPS_KV")
    )
    assert kv_delete[1] == (
        [f"resync:{WS}", f"kick:{WS}", f"action:{WS}"],
        ["pause:a\\_bin\\_unimas\\_my:%"],
    )
    assert "UPDATE LIGHTRAG_OPS_AUDIT SET workspace = $2 WHERE workspace = $1" in joined
    assert "UPDATE LIGHTRAG_OPS_AUDIT SET target = $2 WHERE target = $1" in joined
    assert "DELETE FROM LIGHTRAG_OPS_RUNS WHERE workspace = $1" in joined
    # Tables that do not exist yet are skipped.
    assert "LIGHTRAG_OPS_LOGS" not in joined and "LIGHTRAG_OPS_ALERTS" not in joined
    audit = next(s for s in conn.statements if "SET workspace = $2" in s[0])
    assert audit[1] == (WS, workspace_hash(WS))
    # Audit rows are kept: nothing deletes from the audit table.
    assert "DELETE FROM LIGHTRAG_OPS_AUDIT" not in joined
