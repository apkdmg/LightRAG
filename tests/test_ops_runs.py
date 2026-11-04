"""Run tracking: lifecycle, pipeline hooks, failure taxonomy carry-over,
provider pauses, and the guarantees the pipeline relies on (hooks never
raise, never block, never take a lock, and do nothing when switched off)."""

from __future__ import annotations

import asyncio
import os
import time
from uuid import uuid4

import httpx
import numpy as np
import openai
import pytest

from lightrag.api.ops import runs
from lightrag.api.ops.pauses import _role_slots
from lightrag.api.ops.store import MemoryOpsStore
from lightrag.base import DocStatus
from lightrag.kg import shared_storage
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.lightrag import LightRAG
from lightrag.pipeline import PROVIDER_PAUSE_KEY
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

WORKER = "0123456789abcdef0123456789abcdef"
_REQUEST = httpx.Request("POST", "https://router.example.test/v1/chat/completions")


# -- harness --------------------------------------------------------------------


class _Tokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


def _chunk(tokenizer, content, *_args, **_kwargs) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


def _status_error(cls, code: int, message: str = "boom"):
    return cls(
        f"Error code: {code}",
        response=httpx.Response(code, request=_REQUEST),
        body={"error": {"message": message}},
    )


class _LLM:
    """Extraction LLM stub: fails documents whose text contains a marker."""

    def __init__(self):
        self.fail_marker: str | None = None
        self.error: BaseException | None = None
        self.reject = False
        self.calls = 0
        self.seen_status: list[dict] = []
        self.workspace: str | None = None

    async def __call__(self, prompt, *args, **kwargs) -> str:
        self.calls += 1
        if self.workspace is not None:
            status = await get_namespace_data(
                "pipeline_status", workspace=self.workspace
            )
            self.seen_status.append(
                {k: status.get(k) for k in runs.OWNER_KEYS + ("busy",)}
            )
        if self.reject:
            raise _status_error(openai.PermissionDeniedError, 403, "Invalid key")
        text = f"{prompt} {kwargs.get('system_prompt') or ''}"
        if self.fail_marker and self.fail_marker in text:
            raise self.error or _status_error(openai.InternalServerError, 500)
        return "ok"


async def _rag(tmp_path, llm: _LLM) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / f"rag_{uuid4().hex[:6]}"),
        workspace=f"runs_{uuid4().hex[:8]}",
        llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=_embed),
        tokenizer=Tokenizer("test-tokenizer", _Tokenizer()),
        chunking_func=_chunk,
        max_parallel_insert=1,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    llm.workspace = rag.workspace
    return rag


async def _close(rag: LightRAG) -> None:
    task = getattr(rag, "_provider_resume_task", None)
    if task is not None:
        task.cancel()
    await rag.finalize_storages()


async def _docs(rag) -> dict[str, object]:
    return await rag.doc_status.get_docs_by_statuses(list(DocStatus))


async def _status(rag) -> dict:
    return await get_namespace_data("pipeline_status", workspace=rag.workspace)


@pytest.fixture
def store() -> MemoryOpsStore:
    return MemoryOpsStore()


@pytest.fixture
def tracker(store):
    tracker = runs.RunTracker(store, WORKER, interval_s=3600, role_slots=_role_slots)
    runs.install(tracker)
    yield tracker
    runs.install(None)


@pytest.fixture(autouse=True)
def _no_tracker_leak():
    yield
    runs.install(None)


# -- hooks without a tracker -------------------------------------------------------


def test_every_hook_is_a_noop_without_a_tracker():
    runs.install(None)
    assert runs.new_run_id() is None
    assert runs.owner_fields("abc") == {}
    assert runs.clear_fields() == {}
    assert runs.scan_owner_fields() == {}
    assert runs.scan_clear_fields() == {}
    assert runs.start("abc", workspace="w", kind="process") is None
    assert runs.progress("abc", done=1) == {}
    assert runs.finish("abc", "succeeded") is None
    assert (
        runs.doc_failure(workspace="w", doc_id="d", stage="extract", error=None) is None
    )
    assert runs.duplicate_failure("parse") is None
    assert (
        runs.pause_set(
            "w", role="extract", kind="x", status_code=1, since=1, retry_at=2
        )
        is None
    )
    assert runs.resync_result("w", {"attempts": 1}) is None
    assert runs.note_email_ingest() is None
    assert runs.enabled() is False


def test_a_tracker_from_another_process_is_ignored(store):
    runs.install(runs.RunTracker(store, WORKER, pid=os.getpid() + 1))
    assert runs.enabled() is False
    assert runs.new_run_id() is None


# -- lifecycle --------------------------------------------------------------------


async def test_lifecycle_owner_keys_and_flush(tracker, store):
    run_id = runs.new_run_id()
    owner = runs.owner_fields(run_id)
    assert owner["run_id"] == run_id
    assert owner["owner_worker_id"] == WORKER
    assert owner["owner_pid"] == os.getpid()
    assert owner["heartbeat_at"]
    assert runs.clear_fields() == {k: None for k in runs.OWNER_KEYS}

    runs.start(run_id, workspace="fsktm", kind="process", trigger="upload")
    # Progress that is not merged into pipeline_status never uses up the
    # heartbeat throttle; the first merged one writes, the next within 5 s not.
    assert runs.progress(run_id, add_total=3, stage="parse") == {}
    assert "heartbeat_at" in runs.progress(run_id, done=1, heartbeat=True)
    assert runs.progress(run_id, stage="extract", heartbeat=True) == {}
    runs.progress(run_id, failed=1, stage="extract")
    assert await tracker.flush_once()
    row = store.run_rows[run_id]
    assert row["status"] == "running"
    assert (row["docs_total"], row["docs_done"], row["docs_failed"]) == (3, 2, 1)
    assert row["stage"] == "extract" and row["trigger"] == "upload"
    assert row["worker_id"] == WORKER and row["pid"] == os.getpid()
    # The host the PID lives on is recorded with the run.
    assert row["host"] == tracker.host

    runs.finish(run_id, "succeeded", docs_done=3)
    assert await tracker.flush_once()
    row = store.run_rows[run_id]
    assert row["status"] == "succeeded" and row["finished_at"] is not None
    assert row["docs_done"] == 3
    # Finished and written: no longer held in memory.
    assert run_id not in tracker._runs


async def test_final_status_is_never_overwritten(tracker, store):
    run_id = runs.new_run_id()
    runs.start(run_id, workspace="w", kind="delete")
    runs.finish(run_id, "failed", error=RuntimeError("storage gone"))
    await tracker.flush_once()
    assert store.run_rows[run_id]["status"] == "failed"
    assert store.run_rows[run_id]["error_kind"] == "unknown"
    # A late running state (e.g. a retried flush) does not reopen it.
    stale = dict(store.run_rows[run_id], status="running")
    stale["started_at"] = stale["heartbeat_at"] = stale["progress_at"] = time.time()
    stale["finished_at"] = None
    await store.runs_write([stale])
    assert store.run_rows[run_id]["status"] == "failed"


async def test_finish_after_orphaning_sets_was_orphaned(tracker, store):
    run_id = runs.new_run_id()
    runs.start(run_id, workspace="w", kind="process")
    await tracker.flush_once()
    assert await store.runs_orphan([run_id]) == [run_id]
    runs.finish(run_id, "succeeded")
    await tracker.flush_once()
    row = store.run_rows[run_id]
    assert row["status"] == "succeeded"
    assert row["was_orphaned"] is True


async def test_progress_after_orphaning_reopens_the_run(tracker, store):
    run_id = runs.new_run_id()
    runs.start(run_id, workspace="w", kind="process")
    await tracker.flush_once()
    await store.runs_orphan([run_id])
    runs.progress(run_id, done=1)
    await tracker.flush_once()
    row = store.run_rows[run_id]
    assert row["status"] == "running" and row["was_orphaned"] is True


async def test_run_origin_reaches_tasks_created_inside_it(tracker, store):
    async def start_one() -> str:
        run_id = runs.new_run_id()
        runs.start(run_id, workspace="w", kind="process", trigger="upload")
        return run_id

    with runs.run_origin(kind="startup_resume", trigger="startup"):
        task = asyncio.create_task(start_one())
    run_id = await task
    assert runs.current_origin() == (None, None)
    run = tracker._runs[run_id]
    assert run["kind"] == "startup_resume" and run["trigger"] == "startup"
    # A non-process kind keeps its own kind; the origin only names the trigger.
    with runs.run_origin(kind="startup_resume", trigger="resync"):
        other = runs.new_run_id()
        runs.start(other, workspace="w", kind="email_batch")
    assert tracker._runs[other]["kind"] == "email_batch"
    assert tracker._runs[other]["trigger"] == "resync"


async def test_failure_metadata_counts_attempts_without_reads(tracker):
    first = runs.doc_failure(
        workspace="w",
        doc_id="doc-1",
        stage="extract",
        error=_status_error(openai.RateLimitError, 429),
        run_id="r1",
        model=lambda: "gpt-4o-mini",
        file_path="Senate minutes 900101-13-5678.pdf",
    )
    assert first["attempt"] == 1 and first["error_kind"] == "rate_limited"
    assert first["http_status"] == 429 and first["model"] == "gpt-4o-mini"
    second = runs.doc_failure(
        workspace="w",
        doc_id="doc-1",
        stage="merge",
        error=TimeoutError(),
        previous=first,
        run_id="r2",
    )
    assert second["attempt"] == 2
    assert second["first_failed_at"] == first["first_failed_at"]
    assert second["last_run_id"] == "r2"
    await tracker.flush_once()
    rows = tracker.store.failure_rows
    assert [r["attempt"] for r in rows] == [1, 2]
    # The stored message is redacted and the file name masked.
    assert all("900101" not in (r["message"] or "") for r in rows)
    assert all("Senate minutes" not in (r["message"] or "") for r in rows)


# -- the pipeline with run tracking -------------------------------------------------


async def test_pipeline_run_writes_owner_keys_and_a_run_row(tmp_path, tracker, store):
    llm = _LLM()
    rag = await _rag(tmp_path, llm)
    try:
        await rag.apipeline_enqueue_documents(
            ["first document text", "second document text"],
            file_paths=["a.txt", "b.txt"],
        )
        await rag.apipeline_process_enqueue_documents()
        await tracker.flush_once()
        [row] = store.run_rows.values()
        assert row["status"] == "succeeded"
        assert row["kind"] == "process"
        assert row["workspace"] == runs.workspace_of(rag)
        assert (row["docs_total"], row["docs_done"], row["docs_failed"]) == (2, 2, 0)
        assert row["label"] and "files" in row["label"]
        # While the LLM ran, pipeline_status named this run as the owner.
        assert llm.seen_status
        assert all(s["run_id"] == row["run_id"] for s in llm.seen_status)
        assert all(s["owner_worker_id"] == WORKER for s in llm.seen_status)
        assert all(s["owner_pid"] == os.getpid() for s in llm.seen_status)
        # Released together with busy.
        status = await _status(rag)
        assert status["busy"] is False
        assert all(status.get(k) is None for k in runs.OWNER_KEYS)
    finally:
        await _close(rag)


async def test_failure_survives_failed_pending_failed_and_clears_on_success(
    tmp_path, tracker, store
):
    llm = _LLM()
    llm.fail_marker = "flaky"
    rag = await _rag(tmp_path, llm)
    try:
        await rag.apipeline_enqueue_documents(
            ["a flaky document"], file_paths=["f.txt"]
        )
        await rag.apipeline_process_enqueue_documents()
        [(doc_id, doc)] = (await _docs(rag)).items()
        assert doc.status == DocStatus.FAILED
        first = doc.metadata["failure"]
        assert first["error_kind"] == "provider_5xx"
        assert first["stage"] == "extract" and first["attempt"] == 1

        # Retried: FAILED -> PENDING -> ... -> FAILED again, attempt 2.
        await rag.apipeline_process_enqueue_documents()
        doc = (await _docs(rag))[doc_id]
        second = doc.metadata["failure"]
        assert second["attempt"] == 2
        assert second["first_failed_at"] == first["first_failed_at"]
        assert second["last_run_id"] != first["last_run_id"]

        # Fixed: PROCESSED carries no current failure.
        llm.fail_marker = None
        await rag.apipeline_process_enqueue_documents()
        doc = (await _docs(rag))[doc_id]
        assert doc.status == DocStatus.PROCESSED
        assert "failure" not in (doc.metadata or {})

        await tracker.flush_once()
        rows = [r for r in store.failure_rows if r["doc_id"] == doc_id]
        assert [r["attempt"] for r in rows] == [1, 2]
        assert {r["error_kind"] for r in rows} == {"provider_5xx"}
        statuses = sorted(r["status"] for r in store.run_rows.values())
        assert statuses == ["succeeded", "succeeded", "succeeded"]
        failed_runs = [r for r in store.run_rows.values() if r["docs_failed"]]
        assert len(failed_runs) == 2
    finally:
        await _close(rag)


async def test_provider_pause_finishes_the_run_paused_and_persists_the_pause(
    tmp_path, tracker, store
):
    llm = _LLM()
    llm.reject = True
    rag = await _rag(tmp_path, llm)
    ws = runs.workspace_of(rag)
    try:
        await rag.apipeline_enqueue_documents(
            [f"document {i}" for i in range(3)],
            file_paths=[f"d{i}.txt" for i in range(3)],
        )
        await rag.apipeline_process_enqueue_documents()
        await tracker.flush_once()
        [row] = store.run_rows.values()
        assert row["status"] == "paused"
        pause = store.kv[f"pause:{ws}:extraction"]
        assert pause["kind"] == "invalid_key" and pause["status_code"] == 403
        assert pause["cleared_at"] is None and pause["slot"] == "extraction"
        assert [p["workspace"] for p in await store.open_pauses(10)] == [ws]
        # The live pause names the workspace the console filed it under.
        assert (await _status(rag))[PROVIDER_PAUSE_KEY]["ops_workspace"] == ws
        failed = [
            d for d in (await _docs(rag)).values() if d.status == DocStatus.FAILED
        ]
        assert failed[0].metadata["failure"]["error_kind"] == "provider_auth"

        # Window over, key fixed: the probe clears the pause.
        llm.reject = False
        status = await _status(rag)
        async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
            status[PROVIDER_PAUSE_KEY] = {**status[PROVIDER_PAUSE_KEY], "retry_at": 0}
        await rag.apipeline_process_enqueue_documents()
        await tracker.flush_once()
        pause = store.kv[f"pause:{ws}:extraction"]
        assert pause["cleared_at"] is not None
        assert pause["cleared_by"] == "provider_accepted"
        assert await store.open_pauses(10) == []
        assert sorted(r["status"] for r in store.run_rows.values()) == [
            "paused",
            "succeeded",
        ]
    finally:
        await _close(rag)


async def test_duplicates_get_a_duplicate_failure_and_no_history_row(
    tmp_path, tracker, store
):
    rag = await _rag(tmp_path, _LLM())
    try:
        await rag.apipeline_enqueue_documents(["same text"], file_paths=["x.txt"])
        await rag.apipeline_enqueue_documents(["other text"], file_paths=["x.txt"])
        dups = [d for i, d in (await _docs(rag)).items() if i.startswith("dup-")]
        assert dups
        failure = dups[0].metadata["failure"]
        assert failure["error_kind"] == "duplicate" and failure["stage"] == "enqueue"
        await tracker.flush_once()
        assert store.failure_rows == []
    finally:
        await _close(rag)


async def test_tracking_off_writes_nothing(tmp_path, store):
    runs.install(None)
    llm = _LLM()
    llm.fail_marker = "flaky"
    rag = await _rag(tmp_path, llm)
    try:
        await rag.apipeline_enqueue_documents(
            ["a flaky document", "a fine document"], file_paths=["f.txt", "g.txt"]
        )
        await rag.apipeline_process_enqueue_documents()
        status = await _status(rag)
        assert not any(k in status for k in runs.OWNER_KEYS)
        assert all(s.get("run_id") is None for s in llm.seen_status)
        docs = await _docs(rag)
        assert {d.status for d in docs.values()} == {
            DocStatus.FAILED,
            DocStatus.PROCESSED,
        }
        assert all("failure" not in (d.metadata or {}) for d in docs.values())
        assert store.run_rows == {} and store.failure_rows == []
    finally:
        await _close(rag)


# -- guarantees ----------------------------------------------------------------------


async def test_hooks_never_raise_into_the_pipeline(
    tmp_path, tracker, store, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("ops store exploded")

    for name in (
        "begin",
        "update",
        "end",
        "add_failure",
        "add_pause_event",
        "heartbeat_fields",
        "_put_kv",
    ):
        monkeypatch.setattr(runs.RunTracker, name, boom)
    monkeypatch.setattr(runs, "iso", boom)
    monkeypatch.setattr(runs, "classify_failure", boom)
    llm = _LLM()
    llm.fail_marker = "flaky"
    rag = await _rag(tmp_path, llm)
    try:
        await rag.apipeline_enqueue_documents(
            ["a flaky document", "a fine document"], file_paths=["f.txt", "g.txt"]
        )
        await rag.apipeline_process_enqueue_documents()
        docs = await _docs(rag)
        assert sorted(str(d.status.value) for d in docs.values()) == [
            "failed",
            "processed",
        ]
        status = await _status(rag)
        assert status["busy"] is False
        assert tracker.hook_errors > 0
        # Other hooks too.
        runs.note_email_ingest()
        runs.resync_result("w", {"attempts": 1})
        runs.pause_cleared("w", role="extract")
    finally:
        await _close(rag)


async def test_store_down_or_pool_exhausted_never_slows_the_pipeline(tmp_path, store):
    # The ops store hangs (pool exhausted): every write waits until the
    # writer's timeout. Buffers are tiny, so entries are dropped and counted.
    store.write_delay_s = 30.0
    tracker = runs.RunTracker(
        store,
        WORKER,
        interval_s=0.01,
        write_timeout_s=0.05,
        max_runs=1,
        max_failures=1,
        max_kv=1,
    )
    tracker.start()
    runs.install(tracker)
    llm = _LLM()
    llm.fail_marker = "flaky"
    rag = await _rag(tmp_path, llm)
    try:
        await rag.apipeline_enqueue_documents(
            [f"a flaky document {i}" for i in range(3)] + ["a fine document"],
            file_paths=[f"f{i}.txt" for i in range(4)],
        )
        started = time.perf_counter()
        await rag.apipeline_process_enqueue_documents()
        # A second run in the same workspace, while the first is unwritten.
        await rag.apipeline_process_enqueue_documents()
        elapsed = time.perf_counter() - started
        assert elapsed < 10.0
        statuses = sorted(str(d.status.value) for d in (await _docs(rag)).values())
        assert statuses == ["failed", "failed", "failed", "processed"]
        await asyncio.sleep(0.2)
        assert tracker.flush_failures > 0
        assert tracker.dropped_failures > 0
        assert tracker.dropped_runs > 0
        assert store.run_rows == {} and store.failure_rows == []

        # Postgres down instead: writes fail fast, rows stay buffered, the
        # newest win, and they are written once the store is back.
        store.write_delay_s = 0.0
        store.fail_writes = ConnectionRefusedError("postgres down")
        await rag.apipeline_enqueue_documents(["more text"], file_paths=["m.txt"])
        await rag.apipeline_process_enqueue_documents()
        await asyncio.sleep(0.05)
        assert store.run_rows == {}
        store.fail_writes = None
        await asyncio.sleep(0.2)
        assert any(r["status"] == "succeeded" for r in store.run_rows.values())
    finally:
        runs.install(None)
        await tracker.stop(flush_timeout_s=0.1)
        await _close(rag)


class _CountingLock:
    def __init__(self, inner, counter):
        self._inner = inner
        self._counter = counter

    async def __aenter__(self):
        self._counter["enter"] += 1
        return await self._inner.__aenter__()

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


async def _lock_entries(tmp_path, monkeypatch, *, tracking: bool) -> dict:
    import lightrag.lightrag as lightrag_module
    import lightrag.pipeline as pipeline_module

    counter = {"calls": 0, "enter": 0}
    real = shared_storage.get_namespace_lock

    def counting(*args, **kwargs):
        counter["calls"] += 1
        return _CountingLock(real(*args, **kwargs), counter)

    for module in (pipeline_module, lightrag_module, shared_storage):
        if hasattr(module, "get_namespace_lock"):
            monkeypatch.setattr(module, "get_namespace_lock", counting)
    store = MemoryOpsStore()
    runs.install(runs.RunTracker(store, WORKER, interval_s=3600) if tracking else None)
    llm = _LLM()
    llm.fail_marker = "flaky"
    rag = await _rag(tmp_path, llm)
    try:
        await rag.apipeline_enqueue_documents(
            ["a flaky document", "doc two", "doc three"],
            file_paths=["f.txt", "g.txt", "h.txt"],
        )
        counter["calls"] = counter["enter"] = 0
        await rag.apipeline_process_enqueue_documents()
        await rag.apipeline_process_enqueue_documents()
    finally:
        runs.install(None)
        await _close(rag)
        monkeypatch.undo()
    return counter


async def test_tracking_takes_no_extra_lock(tmp_path, monkeypatch):
    off = await _lock_entries(tmp_path, monkeypatch, tracking=False)
    on = await _lock_entries(tmp_path, monkeypatch, tracking=True)
    assert off["enter"] > 0
    assert on == off


def test_hook_modules_never_touch_the_namespace_lock():
    import inspect

    from lightrag.api.ops import classify

    for module in (runs, classify):
        source = inspect.getsource(module)
        assert "get_namespace_lock" not in source
        assert "get_namespace_data" not in source


# -- destructive slots and deletes -------------------------------------------------


async def test_destructive_slot_is_a_run_with_owner_keys(tmp_path, tracker, store):
    import importlib
    import sys

    argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        routes = importlib.import_module("lightrag.api.routers.document_routes")
    finally:
        sys.argv = argv
    rag = await _rag(tmp_path, _LLM())
    try:
        acquired, _ = await routes._acquire_destructive_busy(rag, run_kind="clear")
        assert acquired
        status = await _status(rag)
        run_id = status["run_id"]
        assert run_id and status["owner_worker_id"] == WORKER
        assert status["destructive_busy"] is True
        assert await routes._release_destructive_busy(rag, outcome="failed") == run_id
        status = await _status(rag)
        assert status["busy"] is False and status.get("run_id") is None
        await tracker.flush_once()
        row = store.run_rows[run_id]
        assert row["kind"] == "clear" and row["status"] == "failed"

        # run_kind=None reserves the slot without a run (workspace deletion).
        acquired, _ = await routes._acquire_destructive_busy(rag, run_kind=None)
        assert acquired and (await _status(rag)).get("run_id") is None
        assert await routes._release_destructive_busy(rag) is None
    finally:
        await _close(rag)


async def test_single_document_delete_is_a_delete_run(tmp_path, tracker, store):
    rag = await _rag(tmp_path, _LLM())
    try:
        await rag.apipeline_enqueue_documents(["delete me"], file_paths=["d.txt"])
        await rag.apipeline_process_enqueue_documents()
        [doc_id] = list(await _docs(rag))
        result = await rag.adelete_by_doc_id(doc_id)
        assert result.status == "success"
        await tracker.flush_once()
        deletes = [r for r in store.run_rows.values() if r["kind"] == "delete"]
        assert len(deletes) == 1
        assert deletes[0]["status"] == "succeeded" and deletes[0]["docs_done"] == 1
        status = await _status(rag)
        assert status.get("run_id") is None and status["busy"] is False
    finally:
        await _close(rag)


async def test_cancelled_delete_with_failed_flush_releases_busy(tmp_path, tracker):
    rag = await _rag(tmp_path, _LLM())
    try:
        await rag.apipeline_enqueue_documents(["cancel me"], file_paths=["c.txt"])
        await rag.apipeline_process_enqueue_documents()
        [doc_id] = list(await _docs(rag))

        real_flush = rag._insert_done
        fired = []

        async def cancelled(*_a, **_k):
            fired.append(True)
            raise asyncio.CancelledError()

        async def flush_fails(*a, **k):
            # Only the post-cancel flush in the finally block fails.
            if fired:
                raise RuntimeError("flush failed")
            return await real_flush(*a, **k)

        rag.doc_status.delete = cancelled
        rag._insert_done = flush_fails
        with pytest.raises(asyncio.CancelledError):
            await rag.adelete_by_doc_id(doc_id)
        status = await _status(rag)
        assert status["busy"] is False and status.get("run_id") is None
    finally:
        await _close(rag)


# -- exit markers -------------------------------------------------------------------


def test_exit_markers_round_trip(tmp_path):
    path = runs.write_exit_marker(str(tmp_path), 4242)
    assert path and os.path.exists(path)
    assert oct(os.stat(os.path.dirname(path)).st_mode & 0o777) == "0o700"
    markers = runs.read_exit_markers(str(tmp_path))
    assert set(markers) == {4242} and markers[4242] <= time.time()
    old = os.path.join(runs.exit_marker_dir(str(tmp_path)), "77-1000")
    open(old, "w").close()
    assert runs.prune_exit_markers(str(tmp_path)) == 1
    assert set(runs.read_exit_markers(str(tmp_path))) == {4242}
    # Never raises, even where it cannot write.
    assert runs.write_exit_marker("/nonexistent/\0bad", 1) is None


async def test_store_reads_from_hooks_never_raise(tracker, store):
    store.kv["kick:w"] = {"at": 123.0}
    assert await runs.kick_at("w") == 123.0
    store.fail_reads = ConnectionRefusedError("postgres down")
    assert await runs.kick_at("w") is None

    async def hang(*args, **kwargs):
        await asyncio.sleep(30)

    store.kv_get = hang
    started = time.perf_counter()
    assert await runs.kick_at("w") is None
    assert time.perf_counter() - started < runs.KICK_READ_TIMEOUT_S + 1


async def test_tracker_stats_reach_the_heartbeat(tracker):
    from types import SimpleNamespace

    from lightrag.api.ops.heartbeat import Heartbeat
    from lightrag.api.ops.snapshot import NamespaceReader

    tracker.dropped_failures = 3
    heartbeat = Heartbeat(
        worker_id=WORKER,
        mode="uvicorn",
        version="",
        store=None,
        reader=NamespaceReader(),
        app_state=SimpleNamespace(),
        leases=lambda: [],
        data_rag=lambda: None,
    )
    heartbeat.run_tracker = tracker
    snapshot = await heartbeat.collect()
    assert snapshot["run_tracking"]["dropped_failures"] == 3
    await heartbeat.lag.stop()


# -- review fixes ----------------------------------------------------------------


async def test_run_id_reaches_the_pipeline_stage_workers(tmp_path, tracker, store):
    """The processing loop's run_id (log correlation) is set in the context
    the parse/analyze/extract workers are created from."""
    from lightrag.api.ops.context import current_run_id

    seen: list = []

    class _Recording(_LLM):
        async def __call__(self, prompt, *args, **kwargs):
            seen.append(current_run_id())
            return await super().__call__(prompt, *args, **kwargs)

    rag = await _rag(tmp_path, _Recording())
    try:
        await rag.apipeline_enqueue_documents(["a document"], file_paths=["a.txt"])
        await rag.apipeline_process_enqueue_documents()
        await tracker.flush_once()
        [row] = store.run_rows.values()
        assert seen and set(seen) == {row["run_id"]}
        assert current_run_id() is None
    finally:
        await _close(rag)


async def test_heartbeat_reaches_pipeline_status_despite_stage_progress(
    tmp_path, tracker, store
):
    """Stage-only progress calls (parse, analyze, merge, labels) never use up
    the 5 s throttle the extract and done calls write heartbeat_at with."""
    clock = [1000.0]
    tracker.clock = lambda: clock[0]
    run_id = runs.new_run_id()
    runs.start(run_id, workspace="w", kind="process")
    for stage in ("parse", "analyze", "merge"):
        assert runs.progress(run_id, stage=stage) == {}
    assert runs.progress(run_id, stage="extract", heartbeat=True) == {
        "heartbeat_at": runs.iso(1000.0)
    }
    clock[0] += 6
    runs.progress(run_id, stage="merge")
    assert "heartbeat_at" in runs.progress(run_id, done=1, heartbeat=True)


async def test_deleted_workspace_rows_are_never_written_back(tracker, store):
    """Rows buffered before a workspace's deletion (a flush failing during
    an outage, or another worker's buffer) are dropped, not written later."""
    run_id = runs.new_run_id()
    runs.start(run_id, workspace="gone", kind="process")
    runs.doc_failure(
        workspace="gone", doc_id="d1", stage="extract", error=RuntimeError("x")
    )
    runs.pause_set("gone", role="extract", kind="k", status_code=1, since=1, retry_at=2)
    runs.resync_result("gone", {"attempts": 1})
    keep = runs.new_run_id()
    runs.start(keep, workspace="kept", kind="process")
    store.fail_writes = ConnectionRefusedError("postgres down")
    assert not await tracker.flush_once()

    # This worker deletes the workspace (as OpsRuntime._forget does): its
    # buffers drop the rows, then the stored ops data goes.
    await tracker.forget_flushed("gone")
    store.fail_writes = None
    await store.forget_workspace("gone")
    assert await tracker.flush_once()
    assert {r["workspace"] for r in store.run_rows.values()} == {"kept"}
    assert store.failure_rows == []
    assert not [k for k in store.kv if "gone" in k]
    assert tracker.stats()["dropped_deleted"] >= 2  # the run and the failure row

    # Another worker deleted a workspace this tracker still buffers rows for.
    late = runs.new_run_id()
    runs.start(late, workspace="elsewhere", kind="process")
    runs.resync_result("elsewhere", {"attempts": 2})
    await store.forget_workspace("elsewhere")
    assert await tracker.flush_once()
    assert late not in store.run_rows
    assert "resync:elsewhere" not in store.kv

    # A workspace created again under the same name later is tracked.
    tracker.clock = lambda: time.time() + 5
    again = runs.new_run_id()
    runs.start(again, workspace="elsewhere", kind="process")
    assert await tracker.flush_once()
    assert again in store.run_rows


async def test_one_rejected_row_does_not_hold_back_the_batch(tracker, store):
    """A row the database rejects (data error) is dropped and counted; the
    rest of the batch is written instead of being retried forever."""

    class _DataError(Exception):
        sqlstate = "22021"  # character_not_in_repertoire

    good = runs.new_run_id()
    bad = runs.new_run_id()
    runs.start(good, workspace="w", kind="process")
    runs.start(bad, workspace="w", kind="process")
    write = store.runs_write

    async def picky(rows):
        if any(r["run_id"] == bad for r in rows):
            raise _DataError("invalid byte sequence")
        await write(rows)

    store.runs_write = picky
    assert await tracker.flush_once()
    assert good in store.run_rows and bad not in store.run_rows
    assert tracker.dropped_bad_rows == 1

    # A connection failure keeps the batch for the next flush instead.
    runs.progress(good, done=1)
    store.fail_writes = ConnectionRefusedError("down")
    store.runs_write = write
    assert not await tracker.flush_once()
    assert good in tracker._dirty


def test_nul_characters_never_reach_postgres_text():
    from lightrag.api.ops.store import pg_text

    assert pg_text("bad \x00 byte", 64) == "bad  byte"
    assert pg_text("\x00", 8) is None
    assert pg_text(None, 8) is None
    assert pg_text("abcdef", 3) == "abc"


async def test_run_less_destructive_slot_names_its_owner(tracker):
    assert runs.owner_fields(None) == {}
    owner = runs.owner_fields(None, owner_only=True)
    assert owner["run_id"] is None and owner["owner_worker_id"] == WORKER
    runs.install(None)
    assert runs.owner_fields(None, owner_only=True) == {}


async def test_saving_a_key_clears_the_pause_under_the_storage_name(tracker, store):
    """The unnamed workspace is "default" in Postgres storage: the pause the
    pipeline filed there is the one a key update clears."""
    import importlib
    import sys

    from lightrag.kg.shared_storage import (
        initialize_pipeline_status,
        initialize_share_data,
    )

    argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        provider_routes = importlib.import_module(
            "lightrag.api.routers.provider_routes"
        )
    finally:
        sys.argv = argv
    raw = f"lift_{uuid4().hex[:8]}"
    initialize_share_data()
    await initialize_pipeline_status(workspace=raw)
    status = await get_namespace_data("pipeline_status", workspace=raw)
    status[PROVIDER_PAUSE_KEY] = {
        "role": "extract",
        "kind": "invalid_key",
        "retry_at": 9e18,
        "ops_workspace": "storage_name",
    }
    store.kv["pause:storage_name:extraction"] = {
        "workspace": "storage_name",
        "slot": "extraction",
        "cleared_at": None,
    }
    assert await provider_routes._lift_provider_pause(raw)
    await tracker.flush_once()
    pause = store.kv["pause:storage_name:extraction"]
    assert pause["cleared_by"] == "key_updated" and pause["cleared_at"]
    assert not [k for k in store.kv if k.startswith(f"pause:{raw}:")]
