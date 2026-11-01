"""Documents a restart left waiting are resumed at startup.

The provider pause and its retry timer live in memory, so after a restart only
this startup pass can get pending documents moving again.
"""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import openai
import pytest

from lightrag.api.startup_resume import (
    RESUMABLE_DOC_STATUSES,
    resume_pending_documents,
)
from lightrag.base import DocStatus
from lightrag.kg.postgres_impl import PGDocStatusStorage
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.lightrag import LightRAG
from lightrag.pipeline import PROVIDER_PAUSE_KEY
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _FakeDocStatus:
    def __init__(self, workspace, counts, listed=None):
        self.workspace = workspace
        self.counts = counts
        self.listed = listed
        self.list_calls = []

    async def get_status_counts(self):
        return self.counts

    async def list_workspaces_with_statuses(self, statuses):
        self.list_calls.append(statuses)
        return self.listed


class _FakeRag:
    def __init__(self, workspace, counts=None, listed=None):
        self.workspace = workspace
        self.doc_status = _FakeDocStatus(workspace, counts or {}, listed)
        self.runs = 0

    async def apipeline_process_enqueue_documents(self):
        self.runs += 1


def _getter(rags):
    async def get_rag(workspace):
        return rags[workspace]

    return get_rag


async def test_resumes_listed_workspaces_with_no_registered_owner():
    default = _FakeRag("space1", {"processed": 3}, listed=["email_a", "email_b"])
    a = _FakeRag("email_a", {"pending": 2})
    b = _FakeRag("email_b", {"processing": 1})
    registry = SimpleNamespace(list_owners=None)  # must not be consulted

    await resume_pending_documents(
        default,
        _getter({"email_a": a, "email_b": b}),
        registry=registry,
        multi_tenant=True,
    )

    assert (a.runs, b.runs, default.runs) == (1, 1, 0)
    assert default.doc_status.list_calls == [list(RESUMABLE_DOC_STATUSES)]


def test_failed_is_not_resumable_but_interrupted_states_are():
    assert DocStatus.FAILED not in RESUMABLE_DOC_STATUSES
    assert {
        DocStatus.PENDING,
        DocStatus.PARSING,
        DocStatus.ANALYZING,
        DocStatus.PROCESSING,
    } == set(RESUMABLE_DOC_STATUSES)


async def test_failed_only_workspace_does_not_start_a_run():
    default = _FakeRag("space1", {"failed": 5, "processed": 9})

    await resume_pending_documents(default, _getter({"space1": default}))

    assert default.runs == 0


async def test_single_tenant_checks_only_its_own_workspace():
    default = _FakeRag("space1", {"pending": 1}, listed=["someone_else"])

    await resume_pending_documents(default, _getter({"space1": default}))

    assert default.runs == 1
    assert default.doc_status.list_calls == []


async def test_backend_without_listing_falls_back_to_registered_owners():
    default = _FakeRag("space1", {"pending": 1}, listed=None)
    owned = _FakeRag("owned", {"pending": 1})

    async def list_owners():
        return [
            SimpleNamespace(workspace_id="owned"),
            SimpleNamespace(workspace_id="space1"),
        ]

    await resume_pending_documents(
        default,
        _getter({"space1": default, "owned": owned}),
        registry=SimpleNamespace(list_owners=list_owners),
        multi_tenant=True,
    )

    assert (default.runs, owned.runs) == (1, 1)


async def test_storage_default_name_maps_back_to_the_default_workspace():
    default = _FakeRag("", {"pending": 1}, listed=["default"])
    default.doc_status.workspace = "default"

    await resume_pending_documents(default, _getter({"": default}), multi_tenant=True)

    assert default.runs == 1


async def test_one_failing_workspace_does_not_stop_the_others():
    default = _FakeRag("space1", listed=["broken", "unbuildable", "healthy"])
    broken = _FakeRag("broken", {"pending": 1})
    healthy = _FakeRag("healthy", {"pending": 1})

    async def explode():
        raise RuntimeError("boom")

    broken.apipeline_process_enqueue_documents = explode

    async def get_rag(workspace):
        if workspace == "unbuildable":
            raise RuntimeError("cannot build instance")
        return {"broken": broken, "healthy": healthy}[workspace]

    await resume_pending_documents(default, get_rag, multi_tenant=True)

    assert healthy.runs == 1


async def test_listing_failure_does_not_escape():
    default = _FakeRag("space1")

    async def explode(_statuses):
        raise RuntimeError("database down")

    default.doc_status.list_workspaces_with_statuses = explode

    await resume_pending_documents(default, _getter({}), multi_tenant=True)


async def test_runs_are_limited_and_cancellation_stops_them():
    names = [f"ws{i}" for i in range(5)]
    default = _FakeRag("space1", listed=names)
    rags = {name: _FakeRag(name, {"pending": 1}) for name in names}
    started = []
    cancelled = []
    release = asyncio.Event()

    def blocking(name):
        async def run():
            started.append(name)
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        return run

    for name, rag in rags.items():
        rag.apipeline_process_enqueue_documents = blocking(name)

    task = asyncio.create_task(
        resume_pending_documents(
            default, _getter(rags), multi_tenant=True, max_concurrent=2
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(started) == 2

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert sorted(cancelled) == sorted(started)
    assert len(started) == 2


async def test_postgres_lists_workspaces_across_the_whole_table():
    captured = {}

    async def query(sql, params, multirows=False):
        captured.update(sql=sql, params=params, multirows=multirows)
        return [{"workspace": "a"}, {"workspace": "b"}]

    storage = object.__new__(PGDocStatusStorage)
    storage.db = SimpleNamespace(query=query)

    found = await storage.list_workspaces_with_statuses([DocStatus.PENDING])

    assert found == ["a", "b"]
    assert "workspace=" not in captured["sql"]
    assert captured["params"] == [["pending"]]
    assert captured["multirows"] is True
    assert await storage.list_workspaces_with_statuses([]) == []


# --------------------------------------------------------------------------- #
# Against a real pipeline (file-based storages, stub provider)
# --------------------------------------------------------------------------- #


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


def _no_op_chunking(tokenizer, content, *_args, **_kwargs) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


class _Provider:
    def __init__(self, accept):
        self.accept = accept

    async def __call__(self, *args, **kwargs) -> str:
        if self.accept:
            return "ok"
        request = httpx.Request("POST", "https://llm.example.com/v1/chat/completions")
        response = httpx.Response(412, request=request)
        raise openai.APIStatusError(
            "monthly spend limit reached",
            response=response,
            body={"error": {"message": "monthly spend limit reached"}},
        )


async def _build_rag(tmp_path, provider, workspace) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace=workspace,
        llm_model_func=provider,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_no_op_chunking,
        max_parallel_insert=1,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    return rag


async def _statuses(rag) -> list[str]:
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    return sorted(str(getattr(d.status, "value", d.status)) for d in docs.values())


async def _pause(rag):
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    return status.get(PROVIDER_PAUSE_KEY)


async def _resume(rag) -> None:
    async def get_rag(_workspace):
        return rag

    await resume_pending_documents(rag, get_rag)


async def test_restart_processes_documents_left_pending(tmp_path):
    workspace = f"resume_{uuid4().hex[:8]}"
    rag = await _build_rag(tmp_path, _Provider(accept=True), workspace)
    try:
        await rag.apipeline_enqueue_documents(
            ["first waiting document", "second waiting document"],
            file_paths=["a.txt", "b.txt"],
        )
        await rag.finalize_storages()

        # A new process only finds the documents on disk.
        rag = await _build_rag(tmp_path, _Provider(accept=True), workspace)
        assert await _statuses(rag) == ["pending", "pending"]

        await _resume(rag)

        assert await _statuses(rag) == ["processed", "processed"]
    finally:
        await rag.finalize_storages()


async def test_restart_with_rejected_key_pauses_again_and_restores_the_retry(tmp_path):
    workspace = f"resume_{uuid4().hex[:8]}"
    rag = await _build_rag(tmp_path, _Provider(accept=False), workspace)
    try:
        await rag.apipeline_enqueue_documents(
            ["first waiting document", "second waiting document"],
            file_paths=["a.txt", "b.txt"],
        )

        await _resume(rag)

        assert (await _pause(rag))["status_code"] == 412
        assert await _statuses(rag) == ["failed", "pending"]
        retry = rag._provider_resume_task
        assert retry is not None and not retry.done()
    finally:
        task = getattr(rag, "_provider_resume_task", None)
        if task is not None:
            task.cancel()
        await rag.finalize_storages()
