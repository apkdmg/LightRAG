"""Ops runtime: per-worker start/stop, degradation, flags."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from lightrag.api.ops import context, runtime
from lightrag.api.ops.store import PostgresOpsStore
from lightrag.usage import _event_sinks

pytestmark = pytest.mark.offline


def _app(**state):
    return SimpleNamespace(state=SimpleNamespace(**state))


async def test_without_postgres_the_service_reports_unavailable():
    app = _app()
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="JsonKVStorage"))
    try:
        assert app.state.ops is rt and rt.store is None
        assert rt.service.available is False
        assert rt.service.unavailable_reason == "postgres_required"
        assert len(rt.worker_id) == 32 and rt.pid == os.getpid()
        assert context.current_worker_id() == rt.worker_id
        assert rt.loops == [] and rt.index_task is None
        # Same process: the runtime is reused, not recreated.
        assert (
            await runtime.start_ops(
                app, None, SimpleNamespace(kv_storage="JsonKVStorage")
            )
            is rt
        )
    finally:
        await runtime.stop_ops(app)
    assert app.state.ops is None
    again = await runtime.start_ops(
        app, None, SimpleNamespace(kv_storage="JsonKVStorage")
    )
    assert again.worker_id != rt.worker_id
    await runtime.stop_ops(app)


async def test_store_failure_never_stops_the_server(monkeypatch):
    async def boom(self):
        raise OSError("connection refused")

    monkeypatch.setattr(PostgresOpsStore, "initialize", boom)
    app = _app()
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="PGKVStorage"))
    assert rt.store is None and rt.service.unavailable_reason == "store_unavailable"
    await runtime.stop_ops(app)


async def test_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("OPS_ENABLED", "false")
    app = _app()
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="PGKVStorage"))
    assert rt.store is None and rt.service.unavailable_reason == "disabled"
    await runtime.stop_ops(app)


async def test_postgres_starts_builder_and_scan_loop(monkeypatch, tmp_path):
    async def ready(self):
        self.ready = self.doc_status_ready = True

    finalized = []

    async def finalize(self):
        finalized.append(True)

    built = asyncio.Event()

    async def fake_builder(store, **kwargs):
        built.set()
        return {"idx": "valid"}

    async def lease(self, name, holder, ttl_s):
        return True

    async def release(self, name, holder):
        return None

    async def kv_get(self, key, default=None):
        return default

    async def kv_put(self, key, value):
        return None

    async def kv_get_many(self, keys):
        return {}

    async def kv_merge_many(self, patches):
        return None

    for name, fn in (
        ("initialize", ready),
        ("finalize", finalize),
        ("try_lease", lease),
        ("release_lease", release),
        ("kv_get", kv_get),
        ("kv_put", kv_put),
        ("kv_get_many", kv_get_many),
        ("kv_merge_many", kv_merge_many),
    ):
        monkeypatch.setattr(PostgresOpsStore, name, fn)
    import lightrag.api.ops.indexes as indexes

    monkeypatch.setattr(indexes, "run_index_builder", fake_builder)
    queue = SimpleNamespace(base_dir=tmp_path / ".email_resync_queue")
    app = _app(email_resync_queue=queue)
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="PGKVStorage"))
    try:
        assert rt.service.available is True
        await asyncio.wait_for(built.wait(), 2)
        await asyncio.sleep(0)
        assert rt.index_outcome == {"idx": "valid"}
        [loop] = rt.loops
        assert loop.name == "ops:scan" and loop.holder == rt.worker_id
        assert rt.on_event in _event_sinks
    finally:
        await runtime.stop_ops(app)
    assert finalized == [True]
    assert rt.on_event not in _event_sinks


async def test_flags_turn_off_builder_and_scan(monkeypatch, tmp_path):
    async def ready(self):
        self.ready = self.doc_status_ready = True

    async def finalize(self):
        return None

    monkeypatch.setattr(PostgresOpsStore, "initialize", ready)
    monkeypatch.setattr(PostgresOpsStore, "finalize", finalize)
    monkeypatch.setenv("OPS_INDEX_BUILD", "0")
    monkeypatch.setenv("OPS_QUEUE_SCAN", "off")
    app = _app(email_resync_queue=SimpleNamespace(base_dir=tmp_path))
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="PGKVStorage"))
    assert rt.index_task is None and rt.loops == []
    await runtime.stop_ops(app)


def test_env_flag():
    assert runtime.env_flag("OPS_TEST_UNSET_FLAG") is True
    os.environ["OPS_TEST_FLAG_X"] = "no"
    try:
        assert runtime.env_flag("OPS_TEST_FLAG_X") is False
    finally:
        del os.environ["OPS_TEST_FLAG_X"]


async def test_processing_views_need_postgres_doc_status(monkeypatch):
    async def ready(self):
        self.ready = self.doc_status_ready = True

    async def finalize(self):
        return None

    monkeypatch.setattr(PostgresOpsStore, "initialize", ready)
    monkeypatch.setattr(PostgresOpsStore, "finalize", finalize)
    monkeypatch.setenv("OPS_INDEX_BUILD", "false")
    app = _app()
    rt = await runtime.start_ops(
        app,
        None,
        SimpleNamespace(
            kv_storage="PGKVStorage", doc_status_storage="JsonDocStatusStorage"
        ),
    )
    assert rt.service.available is True and rt.service.docs_available is False
    assert rt.service.unavailable_reason == "doc_status_not_postgres"
    await runtime.stop_ops(app)
