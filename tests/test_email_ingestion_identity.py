"""Offline unit tests for email ingestion identity + content-quality helpers.

Covers the stable-identity keying (deterministic bundle/message id, collision-free
component file_paths that survive canonicalization) and the content-quality gates
(ExtractionResult ok-gating, HTML→text, safe date parse, address parsing, and the
parse_eml robustness fixes). No DB / network / LLM required.
"""

import importlib
import sys

import pytest

# Importing the routers package triggers import-time arg parsing (auth handler);
# neutralize pytest's argv during the import, mirroring the other API tests.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_email_routes = importlib.import_module("lightrag.api.routers.email_routes")
_utils = importlib.import_module("lightrag.utils")
_utils_pipeline = importlib.import_module("lightrag.utils_pipeline")
sys.argv = _original_argv

compute_mdhash_id = _utils.compute_mdhash_id
normalize_document_file_path = _utils_pipeline.normalize_document_file_path
EmailIngestionService = _email_routes.EmailIngestionService
EmailParser = _email_routes.EmailParser
ExtractionResult = _email_routes.ExtractionResult
ParsedAttachment = _email_routes.ParsedAttachment
_component_file_path = _email_routes._component_file_path
_derive_stable_message_id = _email_routes._derive_stable_message_id
_html_to_text = _email_routes._html_to_text
_safe_parse_date = _email_routes._safe_parse_date
_slug = _email_routes._slug
generate_bundle_id = _email_routes.generate_bundle_id

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Identity: bundle id + message id                                            #
# --------------------------------------------------------------------------- #


def test_generate_bundle_id_deterministic_and_prefixed():
    a = generate_bundle_id("<abc@x>")
    b = generate_bundle_id("<abc@x>")
    assert a == b
    assert a.startswith("email_")
    assert len(a) == len("email_") + 12
    assert generate_bundle_id("<other@x>") != a


def test_derive_stable_message_id_is_deterministic():
    from datetime import datetime, timezone

    d = datetime(2026, 7, 13, tzinfo=timezone.utc)
    one = _derive_stable_message_id("Subject", "a@x", d, "body")
    two = _derive_stable_message_id("Subject", "a@x", d, "body")
    assert one == two
    assert one.endswith("@lightrag.local>")
    # Sensitive to content: a different subject yields a different id.
    assert _derive_stable_message_id("Other", "a@x", d, "body") != one


def test_missing_message_id_yields_stable_bundle_across_parses():
    """Same .eml (no Message-ID) parsed twice → same message_id → same bundle."""
    eml = (
        b"From: sender@example.com\r\n"
        b"Subject: No message id here\r\n"
        b"Date: Mon, 13 Jul 2026 08:00:00 +0000\r\n"
        b"\r\n"
        b"Hello body\r\n"
    )
    a = EmailParser.parse_eml(eml)
    b = EmailParser.parse_eml(eml)
    assert a.message_id == b.message_id
    assert generate_bundle_id(a.message_id) == generate_bundle_id(b.message_id)


# --------------------------------------------------------------------------- #
# Component file_paths: uniqueness + canonicalization stability               #
# --------------------------------------------------------------------------- #


def _doc_id(file_path: str) -> str:
    """Replicate the pipeline's deterministic doc_id derivation for a known source."""
    return compute_mdhash_id(normalize_document_file_path(file_path), prefix="doc-")


def test_component_paths_are_unique_and_canonicalization_stable():
    bundle = "email_deadbeef1234"
    paths = [
        _component_file_path(bundle, "master", 0, None),
        _component_file_path(bundle, "inline", 0, "image001.png"),
        _component_file_path(bundle, "inline", 1, "image001.png"),  # same name, diff idx
        _component_file_path(bundle, "attachment", 0, "report.pdf"),
        _component_file_path(bundle, "attachment", 1, "report.pdf"),
    ]
    # No path separators or hint brackets that canonicalization would mangle.
    for p in paths:
        assert "/" not in p and "[" not in p and "]" not in p
        # Canonicalization is a no-op (stable across re-syncs).
        assert normalize_document_file_path(p) == p
        # Real (non-placeholder) source → deterministic doc_id.
        assert normalize_document_file_path(p) != "unknown_source"

    # All distinct → distinct doc_ids (index disambiguates same-filename items).
    assert len(set(paths)) == len(paths)
    assert len({_doc_id(p) for p in paths}) == len(paths)


def test_component_paths_stable_across_calls():
    bundle = "email_deadbeef1234"
    a = _component_file_path(bundle, "attachment", 2, "Q4 Report [final].pdf")
    b = _component_file_path(bundle, "attachment", 2, "Q4 Report [final].pdf")
    assert a == b
    assert normalize_document_file_path(a) == a  # slug stripped the brackets


def test_slug_sanitizes_hostile_filenames():
    assert "/" not in _slug("../../etc/passwd")
    assert "[" not in _slug("a.[docling].pdf") and "]" not in _slug("a.[docling].pdf")
    assert _slug("") == "file"
    assert _slug(None) == "file"


# --------------------------------------------------------------------------- #
# ExtractionResult gating                                                     #
# --------------------------------------------------------------------------- #


def test_extraction_result_ok_property():
    assert ExtractionResult("hello", "ok").ok is True
    assert ExtractionResult("   ", "ok").ok is False  # whitespace only
    assert ExtractionResult(None, "unsupported", "legacy .doc").ok is False
    assert ExtractionResult("text", "failed").ok is False


async def test_legacy_doc_attachment_is_skipped_not_ingested():
    svc = EmailIngestionService(rag_instance=None, vision_model_func=None)
    att = ParsedAttachment(
        filename="ethics.doc",
        content_type="application/msword",
        content=b"\xd0\xcf\x11\xe0garbage",  # OLE header, not docx
    )
    result = await svc._extract_attachment_content(att)
    assert result.status == "unsupported"
    assert result.ok is False


async def test_inline_image_without_vision_model_is_skipped():
    svc = EmailIngestionService(rag_instance=None, vision_model_func=None)
    img = ParsedAttachment(
        filename="poster.png", content_type="image/png", content=b"\x89PNG", is_inline=True
    )

    from types import SimpleNamespace

    email = SimpleNamespace(
        date=None, subject="Hi", from_address="a@x", inline_images=[img]
    )
    doc = await svc._process_inline_image(img, "email_x", email, 0)
    assert doc is None  # no placeholder doc ingested


async def test_unknown_binary_attachment_is_skipped():
    svc = EmailIngestionService(rag_instance=None, vision_model_func=None)
    att = ParsedAttachment(
        filename="thing.bin",
        content_type="application/octet-stream",
        content=b"\x00\x01\x02",
    )
    result = await svc._extract_attachment_content(att)
    assert result.ok is False


async def test_plain_text_attachment_extracted():
    svc = EmailIngestionService(rag_instance=None, vision_model_func=None)
    att = ParsedAttachment(
        filename="note.txt", content_type="text/plain", content=b"real content here"
    )
    result = await svc._extract_attachment_content(att)
    assert result.ok is True
    assert "real content" in result.text


# --------------------------------------------------------------------------- #
# HTML → text                                                                 #
# --------------------------------------------------------------------------- #


def test_html_to_text_strips_tags_and_scripts():
    html_body = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><p>Hello&nbsp;World</p>"
        "<script>alert(1)</script>"
        "<div>Second line</div></body></html>"
    )
    text = _html_to_text(html_body)
    assert "Hello" in text and "World" in text
    assert "Second line" in text
    assert "alert" not in text  # script dropped
    assert "color:red" not in text  # style dropped
    assert "<" not in text and ">" not in text


def test_html_to_text_empty_inputs():
    assert _html_to_text("") == ""
    assert _html_to_text(None) == ""


def test_master_document_recovers_html_only_body():
    svc = EmailIngestionService(rag_instance=None, vision_model_func=None)
    from types import SimpleNamespace

    email = SimpleNamespace(
        subject="HTML only",
        message_id="<m@x>",
        thread_id=None,
        from_address="a@x",
        to_addresses=[],
        cc_addresses=[],
        date=None,
        body_text="",  # empty text part
        body_html="<p>Important HTML body</p>",
        attachments=[],
        inline_images=[],
    )
    doc = svc._build_master_document(email, "email_x")
    assert "Important HTML body" in doc
    assert "(No text content)" not in doc


# --------------------------------------------------------------------------- #
# Safe date parse                                                             #
# --------------------------------------------------------------------------- #


def test_safe_parse_date_variants():
    assert _safe_parse_date(None) is None
    assert _safe_parse_date("") is None
    assert _safe_parse_date("   ") is None
    assert _safe_parse_date("not a date") is None
    assert _safe_parse_date("2026-07-13T08:00:00").year == 2026
    # Trailing Z (older Python fromisoformat rejects it)
    assert _safe_parse_date("2026-07-13T08:00:00Z").year == 2026
    # RFC-2822 fallback
    assert _safe_parse_date("Mon, 13 Jul 2026 08:00:00 +0000").year == 2026


# --------------------------------------------------------------------------- #
# Address parsing                                                             #
# --------------------------------------------------------------------------- #


def test_parse_addresses_handles_comma_in_display_name():
    addrs = EmailParser._parse_addresses('"Doe, John" <john@x.com>, jane@y.com')
    assert addrs == ["john@x.com", "jane@y.com"]


def test_parse_addresses_empty():
    assert EmailParser._parse_addresses("") == []


# --------------------------------------------------------------------------- #
# parse_eml robustness                                                        #
# --------------------------------------------------------------------------- #


def test_thread_index_used_on_its_own():
    eml = (
        b"From: a@x\r\nSubject: s\r\nMessage-ID: <m@x>\r\n"
        b"Thread-Index: AQHTESTINDEX=\r\n\r\nbody\r\n"
    )
    parsed = EmailParser.parse_eml(eml)
    assert parsed.thread_id == "AQHTESTINDEX="


def test_whitespace_references_does_not_crash():
    eml = (
        b"From: a@x\r\nSubject: s\r\nMessage-ID: <m@x>\r\n"
        b"References:    \r\n\r\nbody\r\n"
    )
    parsed = EmailParser.parse_eml(eml)  # must not raise IndexError
    assert parsed.thread_id is None
