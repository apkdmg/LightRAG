"""Integration test for email ingestion: stable identity + overwrite-on-resync.

Drives the real EmailIngestionService.ingest_email against a LightRAG built with
dummy LLM/embedding and file-based storages (no external deps), exercising the
actual ainsert + doc_status + overwrite (adelete_by_doc_id) code paths.
"""

import importlib
import struct
import sys
from uuid import uuid4

import numpy as np
import pytest

# Neutralize pytest argv during the router import (auth handler parses args).
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_email_routes = importlib.import_module("lightrag.api.routers.email_routes")
sys.argv = _original_argv

EmailIngestionService = _email_routes.EmailIngestionService
ParsedAttachment = _email_routes.ParsedAttachment
ParsedEmail = _email_routes.ParsedEmail

from lightrag.base import DocStatus  # noqa: E402
from lightrag.lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc, Tokenizer  # noqa: E402
from lightrag.utils_pipeline import raw_document_content_hash  # noqa: E402

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


def _no_op_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "email_overwrite"),
        workspace=f"emailtest_{uuid4().hex[:8]}",
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


def _make_email(*, attachments) -> ParsedEmail:
    return ParsedEmail(
        message_id="<resync-me@example.com>",
        from_address="sender@example.com",
        to_addresses=["r@example.com"],
        cc_addresses=[],
        subject="Quarterly bundle",
        date=None,
        body_text="This is the master email body.",
        body_html=None,
        inline_images=[
            ParsedAttachment(
                filename="poster.png",
                content_type="image/png",
                content=b"\x89PNG\r\n\x1a\n",
                is_inline=True,
            )
        ],
        attachments=attachments,
    )


async def _bundle_docs(rag) -> dict:
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    return {
        did: st
        for did, st in docs.items()
        if str(getattr(st, "file_path", "")).startswith("email_")
    }


async def test_overwrite_on_resync_and_orphan_removal(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        svc = EmailIngestionService(rag, vision_model_func=None)

        text_att = ParsedAttachment(
            filename="notes.txt",
            content_type="text/plain",
            content=b"real extractable attachment content",
        )
        legacy_doc = ParsedAttachment(
            filename="ethics.doc",
            content_type="application/msword",
            content=b"\xd0\xcf\x11\xe0 legacy binary",
        )

        # --- First ingest -------------------------------------------------- #
        result = await svc.ingest_email(
            _make_email(attachments=[text_att, legacy_doc]),
            track_id="track-first",
        )
        bundle_id = result["bundle_id"]

        docs = await _bundle_docs(rag)
        file_paths = {str(st.file_path) for st in docs.values()}

        # Master + the extractable text attachment are ingested.
        assert f"{bundle_id}__master.txt" in file_paths
        assert any(
            fp.startswith(f"{bundle_id}__attachment_000_") and "notes" in fp
            for fp in file_paths
        )
        # Legacy .doc (unsupported) and the vision-less inline image are skipped.
        assert not any("ethics" in fp for fp in file_paths)
        assert not any("__inline_" in fp for fp in file_paths)
        # No unknown_source, and all share the one track_id.
        assert all(fp != "unknown_source" for fp in file_paths)
        assert all(st.track_id == "track-first" for st in docs.values())

        first_ids = set(docs.keys())
        assert len(first_ids) == 2  # master + 1 attachment

        # --- Re-ingest identical (nothing to do) ---------------------------- #
        second = await svc.ingest_email(
            _make_email(attachments=[text_att, legacy_doc]),
            track_id="track-second",
        )
        docs2 = await _bundle_docs(rag)
        # Same doc_ids, same count, left exactly as stored.
        assert second["unchanged"] is True
        assert set(docs2.keys()) == first_ids
        assert all(st.track_id == "track-first" for st in docs2.values())
        # No FAILED duplicate rows created anywhere.
        all_docs = await rag.doc_status.get_docs_by_statuses([DocStatus.FAILED])
        assert not any(did.startswith("dup-") for did in all_docs)

        # --- Re-ingest with the attachment removed (orphan removal) -------- #
        await svc.ingest_email(
            _make_email(attachments=[legacy_doc]),  # notes.txt gone
            track_id="track-third",
        )
        docs3 = await _bundle_docs(rag)
        file_paths3 = {str(st.file_path) for st in docs3.values()}
        # Only the master remains; the orphaned attachment doc was deleted.
        assert file_paths3 == {f"{bundle_id}__master.txt"}
    finally:
        await rag.finalize_storages()


# --------------------------------------------------------------------------- #
# Re-syncs replace only what changed                                          #
# --------------------------------------------------------------------------- #


class _Vision:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def __call__(self, prompt: str, image_data: str) -> str:
        if self.fail:
            raise RuntimeError("provider down")
        return "A poster announcing the seminar."


def _png() -> bytes:
    # Just enough of a PNG for its 200x100 size to be read from the header.
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", 200, 100)
    )


def _bundle_email(body: str = "Master body.") -> ParsedEmail:
    return ParsedEmail(
        message_id="<partial@example.com>",
        from_address="sender@example.com",
        to_addresses=["r@example.com"],
        cc_addresses=[],
        subject="Seminar",
        date=None,
        body_text=body,
        body_html=None,
        inline_images=[
            ParsedAttachment(
                filename="poster.png",
                content_type="image/png",
                content=_png(),
                is_inline=True,
            )
        ],
        attachments=[
            ParsedAttachment(
                filename="notes.txt",
                content_type="text/plain",
                content=b"Room 2, 10am.",
            )
        ],
    )


def _count_calls(monkeypatch, target, name) -> list:
    calls = []
    original = getattr(target, name)

    async def wrapper(*args, **kwargs):
        calls.append(args[0] if args else kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)
    return calls


async def _tracks(rag) -> dict[str, str]:
    return {
        str(st.file_path).split("__", 1)[1]: st.track_id
        for st in (await _bundle_docs(rag)).values()
    }


async def test_unchanged_resync_deletes_and_indexes_nothing(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        svc = EmailIngestionService(rag, vision_model_func=_Vision())
        await svc.ingest_email(_bundle_email(), track_id="v1")
        deletes = _count_calls(monkeypatch, rag, "adelete_by_doc_id")
        inserts = _count_calls(monkeypatch, rag, "ainsert")

        result = await svc.ingest_email(_bundle_email(), track_id="v2")

        assert result["unchanged"] is True and result["deferred"] is False
        assert result["documents_created"] == 0
        assert deletes == [] and inserts == []
        assert set((await _tracks(rag)).values()) == {"v1"}
    finally:
        await rag.finalize_storages()


async def test_changed_body_replaces_only_the_master(tmp_path, monkeypatch):
    rag = await _build_rag(tmp_path)
    try:
        svc = EmailIngestionService(rag, vision_model_func=_Vision())
        first = await svc.ingest_email(_bundle_email(), track_id="v1")
        deletes = _count_calls(monkeypatch, rag, "adelete_by_doc_id")

        result = await svc.ingest_email(
            _bundle_email("Master body, moved to room 3."), track_id="v2"
        )

        master_id = _email_routes._component_doc_id(f"{first['bundle_id']}__master.txt")
        assert result["unchanged"] is False and result["documents_created"] == 1
        assert deletes == [master_id]
        tracks = await _tracks(rag)
        assert tracks.pop("master.txt") == "v2"
        assert set(tracks.values()) == {"v1"}  # image + attachment untouched
    finally:
        await rag.finalize_storages()


async def test_vision_failure_on_resync_keeps_the_stored_image(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await EmailIngestionService(rag, vision_model_func=_Vision()).ingest_email(
            _bundle_email(), track_id="v1"
        )
        failing = EmailIngestionService(rag, vision_model_func=_Vision(fail=True))
        # The cached description would hide the failure; force a real call.
        await rag.llm_response_cache.drop()

        result = await failing.ingest_email(_bundle_email(), track_id="v2")

        statuses = {c["kind"]: c["status"] for c in result["components"]}
        assert statuses["inline_image"] == "error"
        assert result["unchanged"] is True
        tracks = await _tracks(rag)
        assert any(path.startswith("inline_") for path in tracks)
        assert set(tracks.values()) == {"v1"}
    finally:
        await rag.finalize_storages()


async def test_image_that_failed_before_is_added_without_a_delete(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        await EmailIngestionService(
            rag, vision_model_func=_Vision(fail=True)
        ).ingest_email(_bundle_email(), track_id="v1")
        assert not any(p.startswith("inline_") for p in await _tracks(rag))
        deletes = _count_calls(monkeypatch, rag, "adelete_by_doc_id")
        slots = _count_calls(
            monkeypatch,
            importlib.import_module("lightrag.api.routers.document_routes"),
            "_acquire_destructive_busy",
        )

        result = await EmailIngestionService(
            rag, vision_model_func=_Vision()
        ).ingest_email(_bundle_email(), track_id="v2")

        assert result["documents_created"] == 1
        assert deletes == [] and slots == []
        tracks = await _tracks(rag)
        assert [t for p, t in tracks.items() if p.startswith("inline_")] == ["v2"]
        assert tracks["master.txt"] == "v1"
    finally:
        await rag.finalize_storages()


async def test_failed_document_is_reingested(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        svc = EmailIngestionService(rag, vision_model_func=_Vision())
        first = await svc.ingest_email(_bundle_email(), track_id="v1")
        master_id = _email_routes._component_doc_id(f"{first['bundle_id']}__master.txt")
        stored = await rag.doc_status.get_by_id(master_id)
        await rag.doc_status.upsert(
            {master_id: {**stored, "status": DocStatus.FAILED.value}}
        )

        result = await svc.ingest_email(_bundle_email(), track_id="v2")

        assert result["unchanged"] is False
        master = await rag.doc_status.get_by_id(master_id)
        assert master["track_id"] == "v2"
        assert master["status"] == DocStatus.PROCESSED.value
    finally:
        await rag.finalize_storages()


async def test_planner_hash_matches_the_stored_content_hash(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        # Entities and edge whitespace exercise the sanitizer before hashing.
        content = "  Fish &amp; chips at <b>noon</b> today.  \n"
        await rag.ainsert(input=[content], file_paths=["email_x__master.txt"])
        doc_id = _email_routes._component_doc_id("email_x__master.txt")
        stored = await rag.doc_status.get_by_id(doc_id)
        assert stored["content_hash"] == raw_document_content_hash(content)
    finally:
        await rag.finalize_storages()


def test_plan_bundle_sync_rules():
    plan_bundle_sync = _email_routes.plan_bundle_sync
    doc_id = _email_routes._component_doc_id
    master = ("master text", "b__master.txt")
    image = ("image text", "b__inline_000_a.png.txt")
    stored = {
        doc_id(master[1]): ("processed", raw_document_content_hash(master[0])),
        doc_id(image[1]): ("processed", raw_document_content_hash(image[0])),
    }

    assert plan_bundle_sync(stored, [master, image]).unchanged
    # Still being processed with the same text: nothing to redo.
    pending = {**stored, doc_id(master[1]): ("pending", stored[doc_id(master[1])][1])}
    assert plan_bundle_sync(pending, [master, image]).unchanged

    # A component the email no longer produces is removed...
    plan = plan_bundle_sync(stored, [master])
    assert plan.delete_ids == [doc_id(image[1])] and plan.insert == []
    # ...unless it failed to process this time.
    assert plan_bundle_sync(stored, [master], keep_paths=[image[1]]).unchanged

    # Changed or failed components are replaced; new ones are just added.
    changed = ("new master text", master[1])
    plan = plan_bundle_sync(stored, [changed, image])
    assert plan.delete_ids == [doc_id(master[1])] and plan.insert == [changed]
    failed = {**stored, doc_id(image[1]): ("failed", stored[doc_id(image[1])][1])}
    plan = plan_bundle_sync(failed, [master, image])
    assert plan.delete_ids == [doc_id(image[1])] and plan.insert == [image]
    extra = ("attachment text", "b__attachment_000_n.txt")
    plan = plan_bundle_sync(stored, [master, image, extra])
    assert plan.delete_ids == [] and plan.insert == [extra]
