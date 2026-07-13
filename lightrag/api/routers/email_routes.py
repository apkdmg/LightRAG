"""
Email ingestion routes for LightRAG API.

This module provides endpoints for ingesting emails with attachments as
related document bundles, preserving relationships between email body,
inline images, and attachments in the knowledge graph.

Supports large files up to 100MB with:
- Streaming upload to temporary files (avoids memory spikes)
- Thread pool execution for CPU-bound parsing (non-blocking)
- Proper cleanup of temporary files
"""

import asyncio
import atexit
import base64
import email
import hashlib
import html
import logging
import mimetypes
import os
import re
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.utils import getaddresses, parsedate_to_datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field

from lightrag.api.dependencies import resolve_workspace_from_request
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import generate_track_id

logger = logging.getLogger("lightrag.api.email")

# Configuration constants
MAX_EMAIL_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
MAX_ATTACHMENT_SIZE_BYTES = 25 * 1024 * 1024  # 25MB per file (Mode 2)
MAX_TOTAL_UPLOAD_BYTES = MAX_EMAIL_SIZE_BYTES  # aggregate cap across Mode 2 files
STREAMING_CHUNK_SIZE = 64 * 1024  # 64KB chunks for streaming

# Thread pool for CPU-bound parsing operations
_parsing_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="email_parser")
# The pool is module-level and outlives any single request; make sure it is torn
# down on interpreter exit so reloads/tests don't leak worker threads.
atexit.register(_parsing_executor.shutdown, wait=False)

router = APIRouter(prefix="/documents", tags=["email"])


# ============================================================================
# Shared helpers: identity, extraction gating, parsing safety
# ============================================================================


@dataclass
class ExtractionResult:
    """Outcome of extracting text from an attachment or inline image.

    Only ``status == "ok"`` with non-empty ``text`` is ever embedded as
    knowledge-graph content.  Every other status means "no genuine content" —
    the component is skipped rather than ingesting failure/placeholder prose
    (which would otherwise create bogus entities like "Extraction Failure").
    ``note`` is a short, neutral provenance label — never the raw error string.
    """

    text: Optional[str]
    status: str  # "ok" | "empty" | "unsupported" | "failed" | "no_extractor"
    note: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.text and self.text.strip())


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(name: Optional[str], maxlen: int = 40) -> str:
    """Sanitize a filename into a canonicalization-safe basename fragment.

    Strips ``[``/``]`` (which the parser-hint stripper is sensitive to) and any
    path separators (``Path().name``), replaces every other non
    ``[A-Za-z0-9._-]`` character, and caps the length.
    """
    base = Path(name or "").name.replace("[", "").replace("]", "")
    base = _SLUG_RE.sub("_", base).strip("._")
    return base[:maxlen] or "file"


def generate_bundle_id(message_id: str) -> str:
    """Single source of truth for the per-email bundle id (stable in message_id)."""
    return f"email_{hashlib.sha256(message_id.encode()).hexdigest()[:12]}"


def _derive_stable_message_id(
    subject: Optional[str],
    from_address: Optional[str],
    date: Optional[datetime],
    body_text: Optional[str],
) -> str:
    """Deterministic fallback Message-ID for emails lacking the header.

    A random UUID would give the same email a new bundle_id on every re-sync,
    defeating overwrite/dedup — so derive the id from stable content instead.
    """
    basis = "\x1f".join(
        [
            subject or "",
            from_address or "",
            date.isoformat() if date else "",
            (body_text or "")[:8192],
        ]
    )
    return f"<{hashlib.sha256(basis.encode('utf-8')).hexdigest()}@lightrag.local>"


def _component_file_path(
    prefix_base: str, role: str, index: int, filename: Optional[str]
) -> str:
    """Stable, collision-free per-component file_path (drives doc_id).

    Flat basename (no ``/``), no ``[``/``]``, terminal ``.txt`` because the
    ingested content is extracted text, not the original binary.  All components
    of one email share the ``{prefix_base}__`` prefix used for enumeration.
    """
    if role == "master":
        return f"{prefix_base}__master.txt"
    return f"{prefix_base}__{role}_{index:03d}_{_slug(filename)}.txt"


def _safe_parse_date(value: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO-8601 / RFC-2822 date parse. Never raises."""
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        pass
    try:
        return parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        return None


class _HTMLTextExtractor(HTMLParser):
    """Minimal stdlib HTML→text: drop script/style, break on block tags."""

    _BREAK_TAGS = {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    _SKIP_TAGS = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BREAK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html_body: Optional[str]) -> str:
    """Convert an HTML email body to readable plain text (stdlib only)."""
    if not html_body:
        return ""
    try:
        parser = _HTMLTextExtractor()
        parser.feed(html_body)
        parser.close()
        text = html.unescape(parser.get_text())
    except Exception:
        return ""
    # Collapse runs of blank lines / trailing whitespace.
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: List[str] = []
    blank = False
    for ln in lines:
        if ln.strip():
            out.append(ln.strip())
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


# Per-bundle in-process serialization for the delete→enqueue window.
_bundle_locks: Dict[str, asyncio.Lock] = {}
_bundle_locks_guard = asyncio.Lock()


async def _get_bundle_lock(bundle_key: str) -> asyncio.Lock:
    async with _bundle_locks_guard:
        return _bundle_locks.setdefault(bundle_key, asyncio.Lock())


async def _enumerate_bundle_doc_ids(rag: Any, prefix: str) -> List[str]:
    """All doc_ids whose stored file_path starts with the bundle prefix.

    O(N) scan of doc_status (acceptable at current scale; a
    ``get_doc_ids_by_file_path_prefix`` storage method is the flagged follow-up).
    """
    from lightrag.base import DocStatus

    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    result: List[str] = []
    for did, st in docs.items():
        fp = getattr(st, "file_path", None)
        if fp is None and isinstance(st, dict):
            fp = st.get("file_path")
        if isinstance(fp, str) and fp.startswith(prefix):
            result.append(did)
    return result


async def _mark_deletion_job(rag: Any, bundle_key: str, count: int) -> None:
    """Set pipeline_status.job_name so adelete_by_doc_id's busy-join guard admits us.

    The guard requires the running job_name to start with "deleting" and contain
    "document" (see lightrag.py adelete_by_doc_id).
    """
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    ps = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
    msg = f"Overwriting email bundle {bundle_key}"
    async with lock:
        ps.update(
            {
                "job_name": f"Deleting email bundle {bundle_key} ({count} documents)",
                "job_start": datetime.now().isoformat(),
                "docs": count,
                "batchs": count,
                "cur_batch": 0,
                "latest_message": msg,
            }
        )
        if "history_messages" in ps:
            ps["history_messages"][:] = [msg]


# ============================================================================
# Data Models
# ============================================================================


class EmailMetadata(BaseModel):
    """Metadata for a pre-parsed email."""

    message_id: Optional[str] = Field(None, description="Unique message ID")
    from_address: str = Field(..., alias="from", description="Sender email address")
    to_addresses: List[str] = Field(
        default_factory=list, alias="to", description="Recipient email addresses"
    )
    cc_addresses: List[str] = Field(
        default_factory=list, alias="cc", description="CC email addresses"
    )
    subject: str = Field("", description="Email subject")
    date: Optional[str] = Field(None, description="Email date (ISO format)")
    thread_id: Optional[str] = Field(None, description="Thread/conversation ID")
    body_text: Optional[str] = Field(None, description="Plain text body")
    body_html: Optional[str] = Field(None, description="HTML body")

    class Config:
        populate_by_name = True


class EmailIngestionResponse(BaseModel):
    """Response from email ingestion."""

    status: str
    bundle_id: str
    message: str
    track_id: str = Field(
        default="",
        description="Track ID for monitoring background processing status. "
        "Use GET /documents/track_status/{track_id} to check progress.",
    )
    documents_created: int = Field(
        default=0,
        description="Number of documents created (0 when processing in background).",
    )
    email_subject: str
    attachments_processed: int = Field(
        default=0,
        description="Number of attachments processed (0 when processing in background).",
    )
    inline_images_processed: int = Field(
        default=0,
        description="Number of inline images processed (0 when processing in background).",
    )


# ============================================================================
# Internal Data Structures
# ============================================================================


@dataclass
class ParsedAttachment:
    """Represents a parsed email attachment."""

    filename: str
    content_type: str
    content: bytes
    content_id: Optional[str] = None  # For inline images
    is_inline: bool = False


@dataclass
class ParsedEmail:
    """Represents a fully parsed email with all components."""

    message_id: str
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str]
    subject: str
    date: Optional[datetime]
    body_text: str
    body_html: Optional[str]
    inline_images: List[ParsedAttachment] = field(default_factory=list)
    attachments: List[ParsedAttachment] = field(default_factory=list)
    thread_id: Optional[str] = None


# ============================================================================
# Email Parser
# ============================================================================


class EmailParser:
    """Parses .eml files and extracts all components."""

    @staticmethod
    def parse_eml(eml_content: bytes) -> ParsedEmail:
        """
        Parse an .eml file and extract all components.

        Args:
            eml_content: Raw bytes of the .eml file.

        Returns:
            ParsedEmail with all extracted components.
        """
        msg = email.message_from_bytes(eml_content, policy=policy.default)

        # Extract headers
        header_message_id = msg.get("Message-ID", "")
        from_address = msg.get("From", "")
        to_raw = msg.get("To", "")
        cc_raw = msg.get("Cc", "")
        subject = msg.get("Subject", "(No Subject)")
        date_str = msg.get("Date", "")
        # Thread-Index (Outlook) is authoritative on its own; otherwise fall back
        # to the first References id.  Guard split() against empty/whitespace-only
        # References so it never raises IndexError.
        thread_index = msg.get("Thread-Index")
        if thread_index:
            thread_id: Optional[str] = thread_index
        else:
            refs = msg.get("References", "").split()
            thread_id = refs[0] if refs else None

        # Parse To and CC addresses
        to_addresses = EmailParser._parse_addresses(to_raw)
        cc_addresses = EmailParser._parse_addresses(cc_raw)

        # Parse date (tolerant; never raises)
        date = _safe_parse_date(date_str)

        # Extract body and attachments
        body_text = ""
        body_html = None
        inline_images: List[ParsedAttachment] = []
        attachments: List[ParsedAttachment] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get("Content-Disposition", "")
                content_id = part.get("Content-ID", "")

                # Clean content_id (remove < and >)
                if content_id:
                    content_id = content_id.strip("<>")

                if (
                    content_type == "text/plain"
                    and "attachment" not in content_disposition
                ):
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_text += payload.decode(charset, errors="replace")

                elif (
                    content_type == "text/html"
                    and "attachment" not in content_disposition
                ):
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_html = payload.decode(charset, errors="replace")

                elif part.get_payload(decode=True):
                    # This is an attachment or inline image
                    payload = part.get_payload(decode=True)
                    filename = part.get_filename()

                    # Generate filename with proper extension if missing
                    if not filename:
                        # Try to get extension from content-type first
                        ext = get_extension_from_content_type(content_type)
                        # Fallback to magic bytes detection for images
                        if not ext and payload:
                            ext = detect_image_type_from_bytes(payload) or ""
                        filename = (
                            f"inline_{len(attachments) + len(inline_images)}{ext}"
                        )

                    attachment = ParsedAttachment(
                        filename=filename,
                        content_type=content_type,
                        content=payload,
                        content_id=content_id,
                        is_inline="inline" in content_disposition or bool(content_id),
                    )

                    if attachment.is_inline and content_type.startswith("image/"):
                        inline_images.append(attachment)
                    elif (
                        "attachment" in content_disposition
                        or not content_type.startswith("text/")
                    ):
                        attachments.append(attachment)
        else:
            # Single part message
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                if msg.get_content_type() == "text/html":
                    body_html = payload.decode(charset, errors="replace")
                else:
                    body_text = payload.decode(charset, errors="replace")

        # Deterministic fallback only after body_text is known, so emails with no
        # Message-ID header still hash to a stable bundle_id across re-syncs.
        message_id = header_message_id or _derive_stable_message_id(
            subject, from_address, date, body_text
        )

        return ParsedEmail(
            message_id=message_id,
            from_address=from_address,
            to_addresses=to_addresses,
            cc_addresses=cc_addresses,
            subject=subject,
            date=date,
            body_text=body_text,
            body_html=body_html,
            inline_images=inline_images,
            attachments=attachments,
            thread_id=thread_id,
        )

    @staticmethod
    def _parse_addresses(address_string: str) -> List[str]:
        """Parse an address header into normalized email addresses.

        Uses ``email.utils.getaddresses`` so display names containing commas
        (e.g. ``"Doe, John" <john@x>``) are not split into bogus addresses.
        Display names are dropped; only the address is kept.
        """
        if not address_string:
            return []
        return [addr for _name, addr in getaddresses([address_string]) if addr]


# ============================================================================
# Large File Handling Utilities
# ============================================================================


async def stream_upload_to_temp_file(
    upload_file: UploadFile,
    max_size: int = MAX_EMAIL_SIZE_BYTES,
) -> Tuple[str, int]:
    """
    Stream an uploaded file to a temporary file with size validation.

    This avoids loading the entire file into memory, which is critical
    for large files (up to 100MB).

    Args:
        upload_file: The FastAPI UploadFile object.
        max_size: Maximum allowed file size in bytes.

    Returns:
        Tuple of (temp_file_path, actual_size).

    Raises:
        HTTPException: If file exceeds max_size.
    """
    # Create temp file that won't be auto-deleted (we'll clean it up manually)
    fd, temp_path = tempfile.mkstemp(suffix=".eml", prefix="email_upload_")
    os.close(fd)  # Close the file descriptor, we'll use aiofiles

    total_size = 0
    try:
        async with aiofiles.open(temp_path, "wb") as temp_file:
            while True:
                chunk = await upload_file.read(STREAMING_CHUNK_SIZE)
                if not chunk:
                    break

                total_size += len(chunk)
                if total_size > max_size:
                    # Clean up and raise error
                    await temp_file.close()
                    os.unlink(temp_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds maximum allowed size of {max_size // (1024*1024)}MB",
                    )

                await temp_file.write(chunk)

        logger.debug(f"Streamed {total_size} bytes to temp file: {temp_path}")
        return temp_path, total_size

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded file: {str(e)}",
        )


async def parse_eml_from_file_async(file_path: str) -> ParsedEmail:
    """
    Parse an .eml file asynchronously using thread pool.

    Email parsing is CPU-bound (MIME decoding, charset conversion, etc.),
    so we run it in a thread pool to avoid blocking the event loop.

    Args:
        file_path: Path to the .eml file.

    Returns:
        ParsedEmail with all extracted components.
    """
    loop = asyncio.get_event_loop()

    def _parse_sync():
        with open(file_path, "rb") as f:
            content = f.read()
        return EmailParser.parse_eml(content)

    # Run CPU-bound parsing in thread pool
    return await loop.run_in_executor(_parsing_executor, _parse_sync)


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up a temporary file.

    Args:
        file_path: Path to the temporary file to delete.
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {file_path}: {e}")


async def _read_upload_capped(
    upload_file: UploadFile,
    running_total: int,
    per_file_max: int = MAX_ATTACHMENT_SIZE_BYTES,
    aggregate_max: int = MAX_TOTAL_UPLOAD_BYTES,
) -> Tuple[bytes, int]:
    """Read a Mode-2 UploadFile into memory with per-file + aggregate size caps.

    Returns ``(content, new_running_total)``; raises ``HTTPException(413)`` on
    breach so a few large files can't exhaust memory.
    """
    chunks: List[bytes] = []
    file_size = 0
    while True:
        chunk = await upload_file.read(STREAMING_CHUNK_SIZE)
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > per_file_max:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File '{upload_file.filename or '?'}' exceeds per-file "
                    f"limit of {per_file_max // (1024 * 1024)}MB"
                ),
            )
        if running_total + file_size > aggregate_max:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Total upload exceeds aggregate limit of "
                    f"{aggregate_max // (1024 * 1024)}MB"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks), running_total + file_size


# ============================================================================
# Email Ingestion Service
# ============================================================================


class EmailIngestionService:
    """Handles email ingestion with relationship preservation using LightRAG."""

    def __init__(self, rag_instance: Any, vision_model_func: Optional[Any] = None):
        """
        Initialize the email ingestion service.

        Args:
            rag_instance: The LightRAG instance.
            vision_model_func: Optional vision model for image description.
        """
        self.rag = rag_instance
        self.vision_model_func = vision_model_func

    async def ingest_email(
        self,
        parsed_email: ParsedEmail,
        *,
        bundle_id: Optional[str] = None,
        track_id: Optional[str] = None,
        source_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest an email as connected documents in the knowledge graph.

        Every component (master body, inline images, attachments) is given a
        stable, unique ``file_path`` keyed on the bundle so re-synced emails map
        to the same doc_ids.  All components are inserted in one batched
        ``ainsert`` under a single ``track_id``.  On re-sync, prior bundle docs
        are deleted first (overwrite semantics).

        Args:
            parsed_email: The parsed email with all components.
            bundle_id: Precomputed bundle id (defaults to derive-from-message-id).
            track_id: Shared track id for all components (defaults to a new one).
            source_override: Optional client-supplied provenance name; when set,
                it (slugged) replaces the auto-derived bundle name as the
                file_path prefix.

        Returns:
            Dict with ingestion results (bundle_id, track_id, counts, per-component outcomes).
        """
        bundle_id = bundle_id or generate_bundle_id(parsed_email.message_id)
        track_id = track_id or generate_track_id("email")
        # Component file_paths (and thus doc_ids) are keyed on this prefix.
        prefix_base = _slug(source_override) if source_override else bundle_id

        components: List[Tuple[str, str]] = []  # (content, file_path)
        outcomes: List[Dict[str, Any]] = []

        # 1. Master email document (always present).
        components.append(
            (
                self._build_master_document(parsed_email, bundle_id),
                _component_file_path(prefix_base, "master", 0, None),
            )
        )
        outcomes.append({"kind": "master", "filename": None, "status": "ok"})

        # 2. Inline images — only genuinely-described images become content docs.
        for idx, inline_img in enumerate(parsed_email.inline_images):
            try:
                inline_doc = await self._process_inline_image(
                    inline_img, bundle_id, parsed_email, idx
                )
            except Exception as e:
                logger.warning(
                    f"Failed to process inline image {inline_img.filename}: {e}"
                )
                outcomes.append(
                    {
                        "kind": "inline_image",
                        "filename": inline_img.filename,
                        "status": "error",
                        "note": str(e)[:200],
                    }
                )
                continue
            if inline_doc is None:
                outcomes.append(
                    {
                        "kind": "inline_image",
                        "filename": inline_img.filename,
                        "status": "skipped",
                    }
                )
                continue
            components.append(
                (
                    inline_doc,
                    _component_file_path(
                        prefix_base, "inline", idx, inline_img.filename
                    ),
                )
            )
            outcomes.append(
                {
                    "kind": "inline_image",
                    "filename": inline_img.filename,
                    "status": "ok",
                }
            )

        # 3. Attachments — only successfully-extracted attachments become content docs.
        for idx, attachment in enumerate(parsed_email.attachments):
            try:
                attachment_doc = await self._process_attachment(
                    attachment, bundle_id, parsed_email, idx
                )
            except Exception as e:
                logger.warning(
                    f"Failed to process attachment {attachment.filename}: {e}"
                )
                outcomes.append(
                    {
                        "kind": "attachment",
                        "filename": attachment.filename,
                        "status": "error",
                        "note": str(e)[:200],
                    }
                )
                continue
            if attachment_doc is None:
                outcomes.append(
                    {
                        "kind": "attachment",
                        "filename": attachment.filename,
                        "status": "skipped",
                    }
                )
                continue
            components.append(
                (
                    attachment_doc,
                    _component_file_path(
                        prefix_base, "attachment", idx, attachment.filename
                    ),
                )
            )
            outcomes.append(
                {
                    "kind": "attachment",
                    "filename": attachment.filename,
                    "status": "ok",
                }
            )

        # 4. Overwrite-on-resync + single batched insert under one track_id.
        lock = await _get_bundle_lock(f"{self.rag.workspace}:{prefix_base}")
        async with lock:
            await self._overwrite_bundle(prefix_base)
            await self.rag.ainsert(
                input=[content for content, _ in components],
                file_paths=[fp for _, fp in components],
                track_id=track_id,
            )

        inline_ok = sum(
            1
            for o in outcomes
            if o["kind"] == "inline_image" and o["status"] == "ok"
        )
        att_ok = sum(
            1 for o in outcomes if o["kind"] == "attachment" and o["status"] == "ok"
        )
        return {
            "bundle_id": bundle_id,
            "track_id": track_id,
            "documents_created": len(components),
            "email_subject": parsed_email.subject,
            "attachments_processed": att_ok,
            "inline_images_processed": inline_ok,
            "components": outcomes,
        }

    async def _overwrite_bundle(self, prefix_base: str) -> None:
        """Delete all prior docs for this bundle so a re-sync overwrites cleanly.

        Cheap O(1) precheck via the deterministic master doc_id: only re-syncs
        (existing master) pay the full prefix scan + destructive reservation;
        first-time ingests return immediately and go straight to enqueue.
        """
        from lightrag.utils import compute_mdhash_id
        from lightrag.utils_pipeline import normalize_document_file_path

        rag = self.rag
        master_fp = _component_file_path(prefix_base, "master", 0, None)
        master_id = compute_mdhash_id(
            normalize_document_file_path(master_fp), prefix="doc-"
        )
        if await rag.doc_status.get_by_id(master_id) is None:
            return  # first-time ingest: nothing to overwrite

        # Lazy import avoids app-factory circular-import ordering issues.
        from lightrag.api.routers.document_routes import (
            _acquire_destructive_busy,
            _release_destructive_busy,
        )

        prefix = f"{prefix_base}__"
        # Bounded retry: a concurrent non-deletion pipeline job can refuse the
        # destructive slot; back off and retry rather than dropping the email.
        acquired = False
        reason: Optional[str] = None
        for attempt in range(5):
            acquired, reason = await _acquire_destructive_busy(rag)
            if acquired:
                break
            await asyncio.sleep(2.0 * (attempt + 1))
        if not acquired:
            raise RuntimeError(
                f"Cannot overwrite email bundle {prefix_base}: {reason}"
            )
        try:
            existing = await _enumerate_bundle_doc_ids(rag, prefix)
            if existing:
                await _mark_deletion_job(rag, prefix_base, len(existing))
                for did in existing:
                    res = await rag.adelete_by_doc_id(did, delete_llm_cache=False)
                    status = getattr(res, "status", None)
                    if status not in ("success", "not_found"):
                        logger.warning(
                            f"[{prefix_base}] delete {did}: {status} "
                            f"{getattr(res, 'message', '')}"
                        )
        finally:
            await _release_destructive_busy(rag)

    @staticmethod
    def _email_body_text(email: ParsedEmail) -> str:
        """Readable body text: prefer plain text, fall back to HTML→text.

        HTML-only / multipart-alternative emails with an empty text part would
        otherwise lose their entire body.
        """
        if email.body_text and email.body_text.strip():
            return email.body_text.strip()
        html_text = _html_to_text(email.body_html)
        if html_text:
            return html_text
        return "(No text content)"

    def _build_master_document(self, email: ParsedEmail, bundle_id: str) -> str:
        """Create the master document that links everything together."""

        # Format date
        date_str = email.date.isoformat() if email.date else "Unknown"
        subject = email.subject or "(No Subject)"

        # Build attachment list
        all_attachments = email.attachments + email.inline_images
        if all_attachments:
            attachment_list = "\n".join(
                [
                    f"  - {att.filename} ({att.content_type}, {'inline' if att.is_inline else 'attachment'})"
                    for att in all_attachments
                ]
            )
        else:
            attachment_list = "  (No attachments)"

        # Build recipients list
        to_list = ", ".join(email.to_addresses) if email.to_addresses else "(none)"
        cc_list = ", ".join(email.cc_addresses) if email.cc_addresses else "(none)"

        return f"""Email: {subject}
Bundle-ID: {bundle_id}
Message-ID: {email.message_id}
Thread-ID: {email.thread_id or 'N/A'}

From: {email.from_address}
To: {to_list}
Cc: {cc_list}
Subject: {subject}
Date: {date_str}

Content:
{self._email_body_text(email)}

Attachments ({len(all_attachments)} files):
{attachment_list}
"""

    async def _process_inline_image(
        self,
        image: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
    ) -> Optional[str]:
        """Process an inline image and create a document with description."""

        # Describe the image; skip the component entirely if there's no genuine
        # description (no vision model / transient failure) so we never ingest a
        # "Vision processing failed" placeholder as knowledge-graph content.
        result = await self._describe_image(image)
        if not result.ok:
            logger.info(
                f"Skipping inline image {image.filename}: "
                f"{result.status} ({result.note})"
            )
            return None

        date_str = email.date.isoformat() if email.date else "Unknown"
        subject = email.subject or "(No Subject)"

        return f"""EMAIL INLINE IMAGE
Bundle-ID: {bundle_id}
Image {index + 1} of {len(email.inline_images)}: {image.filename}
Content-Type: {image.content_type}
Content-ID: {image.content_id or 'N/A'}
Size: {len(image.content)} bytes

Parent Email: {subject}
From: {email.from_address}
Date: {date_str}

Image Description:
{result.text}
"""

    async def _process_attachment(
        self,
        attachment: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
    ) -> Optional[str]:
        """Process an attachment and create a document with extracted content."""

        # Only ingest genuinely-extracted content; unsupported/failed extractions
        # are skipped (not written as "[Extraction failed: ...]" prose that would
        # otherwise become bogus entities).
        result = await self._extract_attachment_content(attachment)
        if not result.ok:
            logger.info(
                f"Skipping attachment {attachment.filename}: "
                f"{result.status} ({result.note})"
            )
            return None

        date_str = email.date.isoformat() if email.date else "Unknown"
        subject = email.subject or "(No Subject)"
        total_attachments = len(email.attachments)

        return f"""EMAIL ATTACHMENT
Bundle-ID: {bundle_id}
Attachment {index + 1} of {total_attachments}: {attachment.filename}
Content-Type: {attachment.content_type}
Size: {len(attachment.content)} bytes

Parent Email: {subject}
From: {email.from_address}
Date: {date_str}

Content:
{result.text}
"""

    async def _describe_image(self, image: ParsedAttachment) -> ExtractionResult:
        """Describe an image via the vision model. Returns a gated result.

        No-model / empty / error cases return non-ok so the caller skips the
        component instead of ingesting a placeholder.
        """
        if self.vision_model_func is None:
            return ExtractionResult(None, "no_extractor", "no vision model")

        try:
            image_b64 = base64.b64encode(image.content).decode("utf-8")
            prompt = (
                "Describe this image in detail. Include any text, charts, graphs, "
                "diagrams, or important visual elements. If it contains data, "
                "summarize the key information."
            )
            response = await self.vision_model_func(prompt, image_data=image_b64)
            if response and response.strip():
                return ExtractionResult(response, "ok")
            return ExtractionResult(None, "empty", "empty vision response")
        except Exception as e:
            logger.warning(f"Vision model failed for {image.filename}: {e}")
            return ExtractionResult(None, "failed", "vision error")

    async def _extract_attachment_content(
        self, attachment: ParsedAttachment
    ) -> ExtractionResult:
        """Extract text content from an attachment as a gated result."""

        content_type = attachment.content_type.lower().split(";")[0].strip()

        # Handle text-based files
        if content_type in (
            "text/plain",
            "text/csv",
            "text/markdown",
            "application/json",
        ):
            try:
                text = attachment.content.decode("utf-8", errors="replace")
            except Exception:
                return ExtractionResult(None, "failed", "decode error")
            return (
                ExtractionResult(text, "ok")
                if text.strip()
                else ExtractionResult(None, "empty", "empty text file")
            )

        # Handle images
        if content_type.startswith("image/"):
            return await self._describe_image(attachment)

        # Handle PDFs
        if content_type == "application/pdf":
            return await self._extract_pdf_text(attachment)

        # Modern Word (.docx)
        if (
            content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            return await self._extract_docx_text(attachment)

        # Legacy binary Word (.doc): python-docx cannot read it. Record as an
        # unsupported stub rather than ingesting the parser error as content.
        if content_type == "application/msword":
            return ExtractionResult(None, "unsupported", "legacy .doc format")

        if content_type in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ):
            return ExtractionResult(None, "unsupported", "spreadsheet")

        # Unknown / binary types: metadata-only, no content ingested.
        return ExtractionResult(None, "unsupported", content_type or "unknown type")

    async def _extract_pdf_text(self, attachment: ParsedAttachment) -> ExtractionResult:
        """Extract text from a PDF attachment using pypdf (the repo standard)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return ExtractionResult(None, "no_extractor", "pypdf unavailable")

        try:
            reader = PdfReader(BytesIO(attachment.content))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return ExtractionResult(None, "unsupported", "encrypted PDF")

            text_parts = []
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

            if text_parts:
                return ExtractionResult("\n\n".join(text_parts), "ok")
            return ExtractionResult(None, "empty", "no extractable text")
        except Exception as e:
            logger.warning(f"PDF extraction failed for {attachment.filename}: {e}")
            return ExtractionResult(None, "failed", "PDF extraction error")

    async def _extract_docx_text(self, attachment: ParsedAttachment) -> ExtractionResult:
        """Extract text from a DOCX attachment using python-docx."""
        try:
            import docx
        except ImportError:
            return ExtractionResult(None, "no_extractor", "python-docx unavailable")

        try:
            document = docx.Document(BytesIO(attachment.content))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            if paragraphs:
                return ExtractionResult("\n\n".join(paragraphs), "ok")
            return ExtractionResult(None, "empty", "no extractable text")
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {attachment.filename}: {e}")
            return ExtractionResult(None, "failed", "DOCX extraction error")


# ============================================================================
# Email attachment helpers
# ============================================================================


# Magic bytes signatures for common image formats
IMAGE_MAGIC_BYTES = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"BM": ".bmp",
    b"RIFF": ".webp",  # RIFF....WEBP (check for WEBP later)
    b"II*\x00": ".tiff",  # Little-endian TIFF
    b"MM\x00*": ".tiff",  # Big-endian TIFF
}

# Content-type to extension mapping for common types
CONTENT_TYPE_TO_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/x-tiff": ".tiff",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension from filename."""
    return Path(filename).suffix.lower()


def detect_image_type_from_bytes(content: bytes) -> Optional[str]:
    """
    Detect image file type from magic bytes.

    Args:
        content: Raw bytes of the file.

    Returns:
        File extension (e.g., ".png", ".jpg") or None if not detected.
    """
    if not content or len(content) < 8:
        return None

    # Check magic bytes
    for magic, ext in IMAGE_MAGIC_BYTES.items():
        if content.startswith(magic):
            # Special case for WEBP: RIFF....WEBP
            if magic == b"RIFF" and len(content) >= 12:
                if content[8:12] != b"WEBP":
                    continue
            return ext

    return None


def get_extension_from_content_type(content_type: str) -> str:
    """
    Get file extension from content-type header.

    Args:
        content_type: MIME content type (e.g., "image/png").

    Returns:
        File extension (e.g., ".png") or empty string if not found.
    """
    content_type_lower = content_type.lower().split(";")[0].strip()

    # First check our explicit mapping
    if content_type_lower in CONTENT_TYPE_TO_EXTENSION:
        return CONTENT_TYPE_TO_EXTENSION[content_type_lower]

    # Fallback to mimetypes library
    ext = mimetypes.guess_extension(content_type_lower)
    if ext:
        # mimetypes returns ".jpeg" but we prefer ".jpg"
        if ext == ".jpeg":
            return ".jpg"
        return ext

    return ""


# ============================================================================
# Helper Functions
# ============================================================================


async def get_lightrag_for_request(request: Request, rag_instance=None):
    """
    Get the LightRAG instance for the request.

    In single-instance mode, returns the passed rag_instance.
    In multi-tenant mode, resolves the workspace and gets the LightRAG instance.
    """
    workspace_manager = getattr(request.app.state, "workspace_manager", None)

    if workspace_manager is not None:
        # Multi-tenant mode - get workspace-specific LightRAG instance
        workspace = await resolve_workspace_from_request(request)
        return await workspace_manager.get_lightrag_instance(workspace)
    else:
        # Single-instance mode - use the provided rag instance
        return rag_instance


# ============================================================================
# Route Factory
# ============================================================================


def create_email_routes(rag, vision_model_func=None, api_key: Optional[str] = None):
    """
    Create email ingestion routes.

    Args:
        rag: The default LightRAG instance (for single-instance mode).
        vision_model_func: Optional native vision function for inline-image
            description; called as ``vision_model_func(prompt, image_data=<base64>)``.
        api_key: Optional API key for authentication.

    Returns:
        The configured router.
    """
    combined_auth = get_combined_auth_dependency(api_key)
    _default_rag = rag
    _vision_model_func = vision_model_func

    @router.post(
        "/email",
        response_model=EmailIngestionResponse,
        dependencies=[Depends(combined_auth)],
        summary="Ingest email with attachments",
        description="""
Ingest an email with its attachments as a related document bundle.

## Mode 1: Raw .eml File Upload (Recommended)

Upload a complete `.eml` file which contains everything (headers, body, inline images, attachments).
The `.eml` format is the standard RFC 822 email format exported from email clients.

**curl example:**
```bash
curl -X POST "http://localhost:9621/documents/email" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -F "email_file=@/path/to/email.eml"
```

**How to get .eml files:**
- **Outlook**: drag the email from the message list onto a folder or the desktop — this saves a `.eml` file. (Outlook's *File → Save As* produces `.msg`, which is **not** supported — only `.eml` can be ingested.)
- **Gmail**: Open email → Three dots menu → "Download message" (or "Show original" → "Download Original")
- **Thunderbird**: Right-click email → Save As → .eml file
- **Apple Mail**: Drag email to Finder, or File → Save As

---

## Mode 2: Structured JSON Input

For programmatic use when you have parsed email data. Send email metadata as JSON string
with separate file uploads for attachments.

**curl example:**
```bash
curl -X POST "http://localhost:9621/documents/email" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -F 'metadata={"from": "sender@example.com", "to": ["recipient@example.com"], "subject": "Q4 Report", "date": "2024-01-15T10:30:00"}' \\
  -F "body_text=Here is the Q4 report as discussed." \\
  -F "attachments=@/path/to/report.pdf" \\
  -F "attachments=@/path/to/data.xlsx" \\
  -F "inline_images=@/path/to/chart.png"
```

**Metadata JSON fields:**
- `from` (required): Sender email address
- `to` (required): List of recipient addresses
- `subject`: Email subject line
- `date`: ISO format date string (e.g., "2024-01-15T10:30:00")
- `cc`: List of CC addresses
- `message_id`: Unique message identifier
- `thread_id`: Thread/conversation identifier
- `body_html`: HTML body (alternative to body_text form field)

---

## How It Works

The email and all its components are ingested with a shared **Bundle ID** that preserves
relationships in the knowledge graph. This enables queries like:
- "What was in John's Q4 report email?"
- "Show me the attachments from the budget discussion"
- "What charts were included in the marketing presentation email?"

**File size limit:** 100MB for .eml files (streamed to avoid memory issues)

Inline images are described by the native vision model (when configured);
document attachments have their text extracted. All content is inserted into
the LightRAG knowledge graph under the shared Bundle ID.
        """,
    )
    async def ingest_email(
        http_request: Request,
        background_tasks: BackgroundTasks,
        email_file: Optional[UploadFile] = File(
            None,
            description="Complete .eml file containing email with all attachments",
        ),
        metadata: Optional[str] = Form(
            None,
            description="JSON string with email metadata (for structured input mode)",
        ),
        body_text: Optional[str] = Form(
            None,
            description="Plain text email body (for structured input mode)",
        ),
        source: Optional[str] = Form(
            None,
            description="Optional explicit provenance name for this email bundle. "
            "When set, it replaces the auto-derived bundle name as the document "
            "source/file_path prefix (still suffixed per component).",
        ),
        attachments: List[UploadFile] = File(
            default=[],
            description="Attachment files (for structured input mode)",
        ),
        inline_images: List[UploadFile] = File(
            default=[],
            description="Inline image files (for structured input mode)",
        ),
    ):
        """
        Ingest an email with attachments as a related document bundle.

        The email content and attachments are processed and stored with linking
        metadata that preserves their relationships in the knowledge graph.
        """
        try:
            # Resolve the per-workspace LightRAG instance and run the email
            # ingestion service with native inline-image vision description.
            rag_instance = await get_lightrag_for_request(http_request, _default_rag)
            service = EmailIngestionService(rag_instance, _vision_model_func)
            logger.info("Using LightRAG email ingestion service")

            # Parse email based on input mode
            temp_file_path = None
            if email_file and email_file.filename:
                # Mode 1: Raw .eml file - use streaming for large file support
                logger.info(f"Processing .eml file: {email_file.filename}")

                try:
                    # Stream upload to temp file (handles up to 100MB)
                    temp_file_path, file_size = await stream_upload_to_temp_file(
                        email_file, max_size=MAX_EMAIL_SIZE_BYTES
                    )
                    logger.info(
                        f"Uploaded {file_size / (1024*1024):.2f}MB to temp file"
                    )

                    # Parse in thread pool to avoid blocking
                    parsed_email = await parse_eml_from_file_async(temp_file_path)

                finally:
                    # Always clean up temp file
                    if temp_file_path:
                        cleanup_temp_file(temp_file_path)

                # Reject uploads that clearly aren't an email (empty/junk bytes):
                # no sender, no meaningful subject, no body, no attachments.
                if not (
                    parsed_email.from_address
                    or (parsed_email.subject and parsed_email.subject != "(No Subject)")
                    or (parsed_email.body_text and parsed_email.body_text.strip())
                    or parsed_email.body_html
                    or parsed_email.attachments
                    or parsed_email.inline_images
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded file does not appear to be a valid email",
                    )

            elif metadata:
                # Mode 2: Structured input
                logger.info("Processing structured email input")
                import json

                try:
                    meta_dict = json.loads(metadata)
                    email_meta = EmailMetadata(**meta_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid metadata JSON: {str(e)}",
                    )

                # Process attachment + inline files with per-file and aggregate
                # size caps (Mode 2 reads into memory).
                total_bytes = 0
                parsed_attachments = []
                for att_file in attachments:
                    if att_file.filename:
                        content, total_bytes = await _read_upload_capped(
                            att_file, total_bytes
                        )
                        parsed_attachments.append(
                            ParsedAttachment(
                                filename=att_file.filename,
                                content_type=att_file.content_type
                                or "application/octet-stream",
                                content=content,
                                is_inline=False,
                            )
                        )

                parsed_inline = []
                for img_file in inline_images:
                    if img_file.filename:
                        content, total_bytes = await _read_upload_capped(
                            img_file, total_bytes
                        )
                        parsed_inline.append(
                            ParsedAttachment(
                                filename=img_file.filename,
                                content_type=img_file.content_type or "image/png",
                                content=content,
                                is_inline=True,
                            )
                        )

                # Resolve body first so a missing Message-ID yields a stable,
                # content-derived id (not a random UUID that breaks re-sync).
                resolved_body = body_text or email_meta.body_text or ""
                resolved_date = _safe_parse_date(email_meta.date)
                resolved_message_id = email_meta.message_id or _derive_stable_message_id(
                    email_meta.subject,
                    email_meta.from_address,
                    resolved_date,
                    resolved_body,
                )

                # Build ParsedEmail from structured input
                parsed_email = ParsedEmail(
                    message_id=resolved_message_id,
                    from_address=email_meta.from_address,
                    to_addresses=email_meta.to_addresses,
                    cc_addresses=email_meta.cc_addresses,
                    subject=email_meta.subject,
                    date=resolved_date,
                    body_text=resolved_body,
                    body_html=email_meta.body_html,
                    inline_images=parsed_inline,
                    attachments=parsed_attachments,
                    thread_id=email_meta.thread_id,
                )

            else:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide either 'email_file' (.eml) or 'metadata' (structured input)",
                )

            # Generate bundle_id and track_id for background processing. All
            # components enqueue under this single track_id so
            # GET /documents/track_status/{track_id} reflects the whole email.
            bundle_id = generate_bundle_id(parsed_email.message_id)
            track_id = generate_track_id("email")

            # Define background task for email ingestion
            async def process_email_background(
                svc: Any,
                email: ParsedEmail,
                b_id: str,
                t_id: str,
                src: Optional[str],
            ):
                """Background task to ingest email."""
                try:
                    logger.info(
                        f"[{t_id}] Starting background email ingestion for: {email.subject}"
                    )
                    result = await svc.ingest_email(
                        email, bundle_id=b_id, track_id=t_id, source_override=src
                    )
                    logger.info(
                        f"[{t_id}] Email ingestion completed: {result['documents_created']} documents, "
                        f"{result['attachments_processed']} attachments, "
                        f"{result['inline_images_processed']} inline images"
                    )
                except Exception as e:
                    logger.error(f"[{t_id}] Email ingestion failed: {str(e)}")
                    logger.error(traceback.format_exc())

            # Add to background tasks
            background_tasks.add_task(
                process_email_background,
                svc=service,
                email=parsed_email,
                b_id=bundle_id,
                t_id=track_id,
                src=source,
            )

            return EmailIngestionResponse(
                status="success",
                bundle_id=bundle_id,
                message=f"Email '{parsed_email.subject}' accepted. Processing will continue in background. "
                f"Track progress via GET /documents/track_status/{track_id}.",
                track_id=track_id,
                email_subject=parsed_email.subject,
                documents_created=0,
                attachments_processed=0,
                inline_images_processed=0,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error ingesting email: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to ingest email: {str(e)}",
            )

    @router.get(
        "/email/supported-formats",
        dependencies=[Depends(combined_auth)],
        summary="Get supported email formats",
    )
    async def get_supported_formats():
        """Return information about supported email formats and attachment types."""
        return {
            "email_formats": [
                {
                    "extension": ".eml",
                    "description": "Standard email format (RFC 822)",
                    "supports_attachments": True,
                    "supports_inline_images": True,
                }
            ],
            "attachment_types": {
                "fully_supported": [
                    "text/plain",
                    "text/csv",
                    "text/markdown",
                    "application/json",
                    "application/pdf (pypdf)",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document (.docx, python-docx)",
                ],
                "image_types": [
                    "image/png",
                    "image/jpeg",
                    "image/gif",
                    "image/webp",
                ],
                "metadata_only": [
                    "application/msword (legacy .doc)",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Other binary formats",
                ],
            },
            "notes": [
                "Images are described using the vision model if available",
                "PDF text extraction uses pypdf",
                "DOCX (.docx) extraction uses python-docx; legacy .doc is recorded as a metadata stub (not extracted)",
                "Unsupported/failed extractions are skipped, not ingested as placeholder content",
                "All documents include linking metadata for relationship preservation",
                f"Maximum .eml file size: {MAX_EMAIL_SIZE_BYTES // (1024*1024)}MB; "
                f"Mode-2 per-file cap: {MAX_ATTACHMENT_SIZE_BYTES // (1024*1024)}MB",
                "Large files are streamed to disk and parsed in background threads",
            ],
        }

    return router
