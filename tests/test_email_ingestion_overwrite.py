"""Integration test for email ingestion: stable identity + overwrite-on-resync.

Drives the real EmailIngestionService.ingest_email against a LightRAG built with
dummy LLM/embedding and file-based storages (no external deps), exercising the
actual ainsert + doc_status + overwrite (adelete_by_doc_id) code paths.
"""

import importlib
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

        # --- Re-ingest identical (overwrite) ------------------------------- #
        await svc.ingest_email(
            _make_email(attachments=[text_att, legacy_doc]),
            track_id="track-second",
        )
        docs2 = await _bundle_docs(rag)
        # Same doc_ids, same count — overwritten, not duplicated.
        assert set(docs2.keys()) == first_ids
        assert all(st.track_id == "track-second" for st in docs2.values())
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
