"""Email re-syncs that find the pipeline busy are queued, not dropped.

Drives the real EmailIngestionService + DeferredResyncQueue against a LightRAG
with dummy LLM/embedding and file-based storages.
"""

import asyncio
import importlib
import os
import stat
import sys
from uuid import uuid4

import numpy as np
import pytest

# Neutralize pytest argv during the router import (auth handler parses args).
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_email_routes = importlib.import_module("lightrag.api.routers.email_routes")
sys.argv = _original_argv

from lightrag.api.email_resync_queue import DeferredResyncQueue  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    get_namespace_data,
    get_namespace_lock,
)
from lightrag.lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc, Tokenizer  # noqa: E402

EmailIngestionService = _email_routes.EmailIngestionService
ParsedAttachment = _email_routes.ParsedAttachment
ParsedEmail = _email_routes.ParsedEmail

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _no_op_chunking(tokenizer, content, *_args, **_kwargs) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


@pytest.fixture(autouse=True)
def _fast_overwrite_retry(monkeypatch):
    monkeypatch.setattr(_email_routes, "_OVERWRITE_RETRY_BASE_SECONDS", 0.0)


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "rag"),
        workspace=f"resync_{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_no_op_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _queue(tmp_path, rag) -> DeferredResyncQueue:
    async def get_rag(_workspace):
        return rag

    return DeferredResyncQueue(
        str(tmp_path / "rag"), get_rag, min_backoff=0.02, max_backoff=0.05
    )


def _email(*attachment_names: str) -> ParsedEmail:
    return ParsedEmail(
        message_id="<busy-resync@example.com>",
        from_address="sender@example.com",
        to_addresses=["r@example.com"],
        cc_addresses=[],
        subject="Busy pipeline",
        date=None,
        body_text="Master body.",
        body_html=None,
        attachments=[
            ParsedAttachment(
                filename=name,
                content_type="text/plain",
                content=f"content of {name}".encode(),
            )
            for name in attachment_names
        ],
    )


async def _set_busy(rag, busy: bool) -> None:
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
        status["busy"] = busy


async def _bundle_docs(rag) -> dict[str, tuple[str, str]]:
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    return {
        str(st.file_path): (st.track_id, str(getattr(st.status, "value", st.status)))
        for st in docs.values()
        if str(st.file_path).startswith("email_")
    }


async def _wait_drained(queue: DeferredResyncQueue, workspace: str) -> None:
    task = queue._drainers.get(workspace)
    if task is not None:
        await asyncio.wait_for(task, timeout=10)
    if queue._background:
        await asyncio.wait_for(asyncio.gather(*list(queue._background)), timeout=10)


async def test_busy_resync_is_queued_coalesced_and_applied(tmp_path):
    rag = await _build_rag(tmp_path)
    queue = _queue(tmp_path, rag)
    svc = EmailIngestionService(rag, resync_queue=queue)
    try:
        first = await svc.ingest_email(_email("a.txt"), track_id="v1")
        assert first["deferred"] is False

        await _set_busy(rag, True)
        second = await svc.ingest_email(_email("a.txt", "b.txt"), track_id="v2")
        third = await svc.ingest_email(_email(), track_id="v3")
        assert second["deferred"] is True and third["deferred"] is True
        # One queued file per bundle; the newest version replaced the older.
        assert queue.pending_count(rag.workspace) == 1
        # The stored version is untouched while the pipeline is busy.
        assert {t for t, _ in (await _bundle_docs(rag)).values()} == {"v1"}

        await _set_busy(rag, False)
        await _wait_drained(queue, rag.workspace)

        docs = await _bundle_docs(rag)
        bundle = first["bundle_id"]
        assert set(docs) == {f"{bundle}__master.txt"}  # v3 has no attachments
        assert docs[f"{bundle}__master.txt"] == ("v3", "processed")
        assert queue.pending_count(rag.workspace) == 0
    finally:
        await queue.stop()
        await rag.finalize_storages()


async def test_newer_resync_waits_behind_a_queued_one(tmp_path):
    rag = await _build_rag(tmp_path)
    queue = _queue(tmp_path, rag)
    svc = EmailIngestionService(rag, resync_queue=queue)
    try:
        await svc.ingest_email(_email("a.txt"), track_id="v1")
        await _set_busy(rag, True)
        await svc.ingest_email(_email("b.txt"), track_id="v2")
        await queue.stop()  # simulate the drainer not having run yet

        # Pipeline is free, but v2 is still queued: v3 must queue behind it.
        await _set_busy(rag, False)
        queue2 = _queue(tmp_path, rag)
        svc2 = EmailIngestionService(rag, resync_queue=queue2)
        result = await svc2.ingest_email(_email("c.txt"), track_id="v3")
        assert result["deferred"] is True
        await _wait_drained(queue2, rag.workspace)

        tracks = {t for t, _ in (await _bundle_docs(rag)).values()}
        assert tracks == {"v3"}
        await queue2.stop()
    finally:
        await rag.finalize_storages()


async def test_queue_survives_restart_and_files_are_private(tmp_path):
    rag = await _build_rag(tmp_path)
    queue = _queue(tmp_path, rag)
    svc = EmailIngestionService(rag, resync_queue=queue)
    try:
        await svc.ingest_email(_email("a.txt"), track_id="v1")
        await _set_busy(rag, True)
        await svc.ingest_email(_email("b.txt"), track_id="v2")
        await queue.stop()

        directory = tmp_path / "rag" / ".email_resync_queue" / rag.workspace
        (queued,) = list(directory.glob("*.json"))
        assert stat.S_IMODE(os.stat(queued).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700

        # A fresh process picks the queued re-sync up on start-up.
        await _set_busy(rag, False)
        restarted = _queue(tmp_path, rag)
        await restarted.start()
        await _wait_drained(restarted, rag.workspace)
        tracks = {t for t, _ in (await _bundle_docs(rag)).values()}
        assert tracks == {"v2"}
        await restarted.stop()
    finally:
        await rag.finalize_storages()


async def test_malformed_record_is_set_aside(tmp_path):
    rag = await _build_rag(tmp_path)
    queue = _queue(tmp_path, rag)
    try:
        directory = tmp_path / "rag" / ".email_resync_queue" / rag.workspace
        directory.mkdir(parents=True)
        (directory / "deadbeef.json").write_text("{not json")
        queue.ensure_draining(rag.workspace)
        await _wait_drained(queue, rag.workspace)
        assert not list(directory.glob("*.json"))
        assert (directory / "deadbeef.bad").exists()
    finally:
        await queue.stop()
        await rag.finalize_storages()


def test_unsafe_workspace_name_rejected(tmp_path):
    queue = DeferredResyncQueue(str(tmp_path), lambda _ws: None)
    with pytest.raises(ValueError):
        queue.has_pending("../escape", "bundle")


async def test_new_email_refused_during_delete_is_queued(tmp_path):
    rag = await _build_rag(tmp_path)
    queue = _queue(tmp_path, rag)
    svc = EmailIngestionService(rag, resync_queue=queue)
    try:
        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
        async with lock:
            status["busy"] = True
            status["destructive_busy"] = True  # another bundle's overwrite

        result = await svc.ingest_email(_email("a.txt"), track_id="new")
        assert result["deferred"] is True
        assert await _bundle_docs(rag) == {}

        async with lock:
            status["busy"] = False
            status["destructive_busy"] = False
        await _wait_drained(queue, rag.workspace)
        assert {t for t, _ in (await _bundle_docs(rag)).values()} == {"new"}
    finally:
        await queue.stop()
        await rag.finalize_storages()
