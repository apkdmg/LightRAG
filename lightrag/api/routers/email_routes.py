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
import base64
import email
import hashlib
import logging
import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.message import EmailMessage
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
from lightrag.api.routers.document_routes import (
    is_raganything_available,
    get_default_framework,
    resolve_scheme,
    validate_framework_availability,
    generate_track_id,
)

logger = logging.getLogger("lightrag.api.email")

# Configuration constants
MAX_EMAIL_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
STREAMING_CHUNK_SIZE = 64 * 1024  # 64KB chunks for streaming

# Thread pool for CPU-bound parsing operations
_parsing_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="email_parser")

router = APIRouter(prefix="/documents", tags=["email"])


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
        "Use GET /documents/status/{track_id} to check progress.",
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


class EmailAttachmentInfo(BaseModel):
    """Information about a processed attachment."""

    filename: str
    content_type: str
    size_bytes: int
    is_inline: bool


class EmailIngestionDetailedResponse(EmailIngestionResponse):
    """Detailed response including attachment info."""

    attachments: List[EmailAttachmentInfo] = Field(default_factory=list)


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
        message_id = msg.get("Message-ID", "") or EmailParser._generate_message_id()
        from_address = msg.get("From", "")
        to_raw = msg.get("To", "")
        cc_raw = msg.get("Cc", "")
        subject = msg.get("Subject", "(No Subject)")
        date_str = msg.get("Date", "")
        thread_id = msg.get("Thread-Index") or msg.get("References", "").split()[0] if msg.get("References") else None

        # Parse To and CC addresses
        to_addresses = EmailParser._parse_addresses(to_raw)
        cc_addresses = EmailParser._parse_addresses(cc_raw)

        # Parse date
        date = None
        if date_str:
            try:
                from email.utils import parsedate_to_datetime
                date = parsedate_to_datetime(date_str)
            except (ValueError, TypeError):
                pass

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

                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_text += payload.decode(charset, errors="replace")

                elif content_type == "text/html" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body_html = payload.decode(charset, errors="replace")

                elif part.get_payload(decode=True):
                    # This is an attachment or inline image
                    payload = part.get_payload(decode=True)
                    filename = part.get_filename() or f"attachment_{len(attachments) + len(inline_images)}"

                    attachment = ParsedAttachment(
                        filename=filename,
                        content_type=content_type,
                        content=payload,
                        content_id=content_id,
                        is_inline="inline" in content_disposition or bool(content_id),
                    )

                    if attachment.is_inline and content_type.startswith("image/"):
                        inline_images.append(attachment)
                    elif "attachment" in content_disposition or not content_type.startswith("text/"):
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
        """Parse a comma-separated list of email addresses."""
        if not address_string:
            return []
        # Simple parsing - split by comma and clean up
        addresses = []
        for addr in address_string.split(","):
            addr = addr.strip()
            if addr:
                addresses.append(addr)
        return addresses

    @staticmethod
    def _generate_message_id() -> str:
        """Generate a unique message ID."""
        import uuid
        return f"<{uuid.uuid4()}@lightrag.local>"


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

    async def ingest_email(self, parsed_email: ParsedEmail) -> Dict[str, Any]:
        """
        Ingest an email as connected documents in the knowledge graph.

        Strategy:
        1. Create a master document with email metadata + body + attachment summaries
        2. Process each attachment and inline image with linking context
        3. Each component references the master email document

        Args:
            parsed_email: The parsed email with all components.

        Returns:
            Dict with ingestion results.
        """
        bundle_id = self._generate_bundle_id(parsed_email.message_id)
        documents_created = 0

        # 1. Build and ingest master email document
        master_doc = self._build_master_document(parsed_email, bundle_id)
        await self.rag.ainsert(master_doc)
        documents_created += 1

        # 2. Process inline images
        inline_processed = 0
        for idx, inline_img in enumerate(parsed_email.inline_images):
            try:
                inline_doc = await self._process_inline_image(
                    inline_img, bundle_id, parsed_email, idx
                )
                if inline_doc:
                    await self.rag.ainsert(inline_doc)
                    documents_created += 1
                    inline_processed += 1
            except Exception as e:
                logger.warning(f"Failed to process inline image {inline_img.filename}: {e}")

        # 3. Process attachments
        attachments_processed = 0
        for idx, attachment in enumerate(parsed_email.attachments):
            try:
                attachment_doc = await self._process_attachment(
                    attachment, bundle_id, parsed_email, idx
                )
                if attachment_doc:
                    await self.rag.ainsert(attachment_doc)
                    documents_created += 1
                    attachments_processed += 1
            except Exception as e:
                logger.warning(f"Failed to process attachment {attachment.filename}: {e}")

        return {
            "bundle_id": bundle_id,
            "documents_created": documents_created,
            "email_subject": parsed_email.subject,
            "attachments_processed": attachments_processed,
            "inline_images_processed": inline_processed,
        }

    def _generate_bundle_id(self, message_id: str) -> str:
        """Generate a unique bundle ID from the message ID."""
        # Create a short hash of the message ID
        hash_obj = hashlib.sha256(message_id.encode())
        short_hash = hash_obj.hexdigest()[:12]
        return f"email_{short_hash}"

    def _build_master_document(self, email: ParsedEmail, bundle_id: str) -> str:
        """Create the master document that links everything together."""

        # Format date
        date_str = email.date.isoformat() if email.date else "Unknown"

        # Build attachment list
        all_attachments = email.attachments + email.inline_images
        if all_attachments:
            attachment_list = "\n".join([
                f"  - {att.filename} ({att.content_type}, {'inline' if att.is_inline else 'attachment'})"
                for att in all_attachments
            ])
        else:
            attachment_list = "  (No attachments)"

        # Build recipients list
        to_list = ", ".join(email.to_addresses) if email.to_addresses else "(none)"
        cc_list = ", ".join(email.cc_addresses) if email.cc_addresses else "(none)"

        return f"""
================================================================================
EMAIL DOCUMENT
================================================================================
Bundle-ID: {bundle_id}
Message-ID: {email.message_id}
Thread-ID: {email.thread_id or 'N/A'}

FROM: {email.from_address}
TO: {to_list}
CC: {cc_list}
SUBJECT: {email.subject}
DATE: {date_str}

--------------------------------------------------------------------------------
EMAIL BODY:
--------------------------------------------------------------------------------
{email.body_text.strip() if email.body_text else '(No text content)'}

--------------------------------------------------------------------------------
ATTACHMENTS ({len(all_attachments)} files):
--------------------------------------------------------------------------------
{attachment_list}

Note: Each attachment is stored as a separate document with Bundle-ID prefix
'{bundle_id}' for cross-reference. Query using the bundle ID or email subject
to retrieve related content.
================================================================================
"""

    async def _process_inline_image(
        self,
        image: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
    ) -> Optional[str]:
        """Process an inline image and create a document with description."""

        # Try to describe the image using vision model
        description = await self._describe_image(image)

        date_str = email.date.isoformat() if email.date else "Unknown"

        return f"""
================================================================================
EMAIL INLINE IMAGE
================================================================================
Bundle-ID: {bundle_id}
Image-Index: {index + 1} of {len(email.inline_images)}
Filename: {image.filename}
Content-Type: {image.content_type}
Content-ID: {image.content_id or 'N/A'}
Size: {len(image.content)} bytes

PARENT EMAIL CONTEXT:
  From: {email.from_address}
  Subject: {email.subject}
  Date: {date_str}
  Message-ID: {email.message_id}

--------------------------------------------------------------------------------
IMAGE DESCRIPTION:
--------------------------------------------------------------------------------
{description}
================================================================================
"""

    async def _process_attachment(
        self,
        attachment: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
    ) -> Optional[str]:
        """Process an attachment and create a document with extracted content."""

        content = await self._extract_attachment_content(attachment)

        date_str = email.date.isoformat() if email.date else "Unknown"
        total_attachments = len(email.attachments)

        return f"""
================================================================================
EMAIL ATTACHMENT
================================================================================
Bundle-ID: {bundle_id}
Attachment-Index: {index + 1} of {total_attachments}
Filename: {attachment.filename}
Content-Type: {attachment.content_type}
Size: {len(attachment.content)} bytes

PARENT EMAIL CONTEXT:
  From: {email.from_address}
  Subject: {email.subject}
  Date: {date_str}
  Message-ID: {email.message_id}

--------------------------------------------------------------------------------
ATTACHMENT CONTENT:
--------------------------------------------------------------------------------
{content}
================================================================================
"""

    async def _describe_image(self, image: ParsedAttachment) -> str:
        """Generate a description for an image using vision model if available."""

        if self.vision_model_func is None:
            return f"[Image: {image.filename} - {image.content_type}, {len(image.content)} bytes. No vision model available for description.]"

        try:
            # Encode image as base64
            image_b64 = base64.b64encode(image.content).decode("utf-8")

            # Call vision model
            prompt = (
                "Describe this image in detail. Include any text, charts, graphs, "
                "diagrams, or important visual elements. If it contains data, "
                "summarize the key information."
            )

            response = await self.vision_model_func(
                prompt,
                images=[f"data:{image.content_type};base64,{image_b64}"]
            )

            return response if response else f"[Image: {image.filename}]"

        except Exception as e:
            logger.warning(f"Vision model failed for {image.filename}: {e}")
            return f"[Image: {image.filename} - {image.content_type}, {len(image.content)} bytes. Vision processing failed.]"

    async def _extract_attachment_content(self, attachment: ParsedAttachment) -> str:
        """Extract text content from an attachment."""

        content_type = attachment.content_type.lower()

        # Handle text-based files
        if content_type in ("text/plain", "text/csv", "text/markdown", "application/json"):
            try:
                return attachment.content.decode("utf-8", errors="replace")
            except Exception:
                return f"[Text file: {attachment.filename} - Could not decode content]"

        # Handle images
        if content_type.startswith("image/"):
            return await self._describe_image(attachment)

        # Handle PDFs - try basic extraction
        if content_type == "application/pdf":
            return await self._extract_pdf_text(attachment)

        # Handle Office documents
        if content_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ):
            return await self._extract_docx_text(attachment)

        if content_type in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ):
            return f"[Spreadsheet: {attachment.filename} - Excel file, {len(attachment.content)} bytes]"

        # Default for unknown types
        return f"[Attachment: {attachment.filename} - {content_type}, {len(attachment.content)} bytes]"

    async def _extract_pdf_text(self, attachment: ParsedAttachment) -> str:
        """Extract text from a PDF attachment."""
        try:
            import fitz  # PyMuPDF

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(attachment.content)
                tmp.flush()

                doc = fitz.open(tmp.name)
                text_parts = []
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")
                doc.close()

                if text_parts:
                    return "\n\n".join(text_parts)
                return f"[PDF: {attachment.filename} - No extractable text, {len(attachment.content)} bytes]"

        except ImportError:
            return f"[PDF: {attachment.filename} - PyMuPDF not installed for text extraction]"
        except Exception as e:
            logger.warning(f"PDF extraction failed for {attachment.filename}: {e}")
            return f"[PDF: {attachment.filename} - Extraction failed: {str(e)}]"

    async def _extract_docx_text(self, attachment: ParsedAttachment) -> str:
        """Extract text from a DOCX attachment."""
        try:
            import docx
            from io import BytesIO

            doc = docx.Document(BytesIO(attachment.content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            if paragraphs:
                return "\n\n".join(paragraphs)
            return f"[DOCX: {attachment.filename} - No extractable text]"

        except ImportError:
            return f"[DOCX: {attachment.filename} - python-docx not installed for text extraction]"
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {attachment.filename}: {e}")
            return f"[DOCX: {attachment.filename} - Extraction failed: {str(e)}]"


# ============================================================================
# RAGAnything Email Ingestion Service
# ============================================================================


# File extensions supported by RAGAnything's parser
RAGANYTHING_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc", ".docx",
    ".ppt", ".pptx",
    ".xls", ".xlsx",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp",
    ".txt", ".md",
}


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension from filename."""
    return Path(filename).suffix.lower()


def is_raganything_parseable(content_type: str, filename: str) -> bool:
    """Check if this attachment can be parsed by RAGAnything's doc_parser."""
    ext = get_file_extension(filename)
    return ext in RAGANYTHING_SUPPORTED_EXTENSIONS


class EmailIngestionServiceRAGAnything:
    """
    Handles email ingestion with full multimodal support using RAGAnything.

    This service uses RAGAnything's doc_parser to parse document attachments
    (PDF, DOCX, PPTX, images, etc.) and insert_content_list() for ingestion,
    preserving Bundle-ID associations across all content.
    """

    def __init__(self, rag_anything_instance: Any):
        """
        Initialize the RAGAnything email ingestion service.

        Args:
            rag_anything_instance: The RAGAnything instance with doc_parser.
        """
        self.rag_anything = rag_anything_instance
        self.doc_parser = getattr(rag_anything_instance, "doc_parser", None)
        self._temp_files: List[str] = []  # Track temp files for cleanup

    async def ingest_email(self, parsed_email: ParsedEmail) -> Dict[str, Any]:
        """
        Ingest an email with full multimodal processing.

        Strategy:
        1. Build a content list starting with email metadata + body as text
        2. For each attachment:
           - If parseable by RAGAnything (PDF, DOCX, images, etc.):
             Save to temp file, parse with doc_parser, merge results with context
           - If simple text: Add as text content
        3. Call insert_content_list() with the complete content list
        4. Clean up temp files

        Args:
            parsed_email: The parsed email with all components.

        Returns:
            Dict with ingestion results.
        """
        bundle_id = self._generate_bundle_id(parsed_email.message_id)
        content_list: List[Dict[str, Any]] = []
        page_idx = 0

        try:
            # 1. Add email header/body as the first text content
            email_header_content = self._build_email_header_content(parsed_email, bundle_id)
            content_list.append({
                "type": "text",
                "text": email_header_content,
                "page_idx": page_idx,
            })
            page_idx += 1

            # 2. Process inline images
            inline_processed = 0
            for idx, inline_img in enumerate(parsed_email.inline_images):
                try:
                    items, new_page_idx = await self._process_attachment_multimodal(
                        inline_img, bundle_id, parsed_email, idx, page_idx, is_inline=True
                    )
                    content_list.extend(items)
                    page_idx = new_page_idx
                    inline_processed += 1
                except Exception as e:
                    logger.warning(f"Failed to process inline image {inline_img.filename}: {e}")
                    # Add fallback text entry
                    content_list.append({
                        "type": "text",
                        "text": self._build_attachment_fallback_text(
                            inline_img, bundle_id, parsed_email, idx, str(e), is_inline=True
                        ),
                        "page_idx": page_idx,
                    })
                    page_idx += 1

            # 3. Process attachments
            attachments_processed = 0
            for idx, attachment in enumerate(parsed_email.attachments):
                try:
                    items, new_page_idx = await self._process_attachment_multimodal(
                        attachment, bundle_id, parsed_email, idx, page_idx, is_inline=False
                    )
                    content_list.extend(items)
                    page_idx = new_page_idx
                    attachments_processed += 1
                except Exception as e:
                    logger.warning(f"Failed to process attachment {attachment.filename}: {e}")
                    # Add fallback text entry
                    content_list.append({
                        "type": "text",
                        "text": self._build_attachment_fallback_text(
                            attachment, bundle_id, parsed_email, idx, str(e), is_inline=False
                        ),
                        "page_idx": page_idx,
                    })
                    page_idx += 1

            # 4. Insert all content using RAGAnything's insert_content_list
            if content_list:
                await self.rag_anything.insert_content_list(
                    content_list=content_list,
                    file_path=f"email_{bundle_id}",
                    doc_id=bundle_id,
                )

            return {
                "bundle_id": bundle_id,
                "documents_created": len(content_list),
                "email_subject": parsed_email.subject,
                "attachments_processed": attachments_processed,
                "inline_images_processed": inline_processed,
            }

        finally:
            # Clean up all temp files
            self._cleanup_temp_files()

    def _generate_bundle_id(self, message_id: str) -> str:
        """Generate a unique bundle ID from the message ID."""
        hash_obj = hashlib.sha256(message_id.encode())
        short_hash = hash_obj.hexdigest()[:12]
        return f"email_{short_hash}"

    def _build_email_header_content(self, email: ParsedEmail, bundle_id: str) -> str:
        """Build the email header and body as text content."""
        date_str = email.date.isoformat() if email.date else "Unknown"

        all_attachments = email.attachments + email.inline_images
        if all_attachments:
            attachment_list = "\n".join([
                f"  - {att.filename} ({att.content_type}, {'inline' if att.is_inline else 'attachment'})"
                for att in all_attachments
            ])
        else:
            attachment_list = "  (No attachments)"

        to_list = ", ".join(email.to_addresses) if email.to_addresses else "(none)"
        cc_list = ", ".join(email.cc_addresses) if email.cc_addresses else "(none)"

        return f"""
================================================================================
EMAIL DOCUMENT
================================================================================
Bundle-ID: {bundle_id}
Message-ID: {email.message_id}
Thread-ID: {email.thread_id or 'N/A'}

FROM: {email.from_address}
TO: {to_list}
CC: {cc_list}
SUBJECT: {email.subject}
DATE: {date_str}

--------------------------------------------------------------------------------
EMAIL BODY:
--------------------------------------------------------------------------------
{email.body_text.strip() if email.body_text else '(No text content)'}

--------------------------------------------------------------------------------
ATTACHMENTS ({len(all_attachments)} files):
--------------------------------------------------------------------------------
{attachment_list}

Note: Each attachment is processed with full multimodal support using RAGAnything.
Query using the bundle ID '{bundle_id}' or email subject to retrieve related content.
================================================================================
"""

    async def _process_attachment_multimodal(
        self,
        attachment: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
        start_page_idx: int,
        is_inline: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Process an attachment using RAGAnything's multimodal capabilities.

        Returns:
            Tuple of (list of content items, next page_idx)
        """
        content_items: List[Dict[str, Any]] = []
        page_idx = start_page_idx

        content_type = attachment.content_type.lower()
        ext = get_file_extension(attachment.filename)

        # Build context header that will be prepended to parsed content
        context_header = self._build_attachment_context_header(
            attachment, bundle_id, email, index, is_inline
        )

        # Check if this can be parsed by RAGAnything
        if self.doc_parser and is_raganything_parseable(content_type, attachment.filename):
            # Save attachment to temp file
            temp_path = self._save_attachment_to_temp(attachment)

            try:
                # Parse using RAGAnything's doc_parser
                loop = asyncio.get_event_loop()
                parsed_content = await loop.run_in_executor(
                    _parsing_executor,
                    lambda: self.doc_parser.parse_document(temp_path)
                )

                if parsed_content:
                    # Add context header as first item
                    content_items.append({
                        "type": "text",
                        "text": context_header,
                        "page_idx": page_idx,
                    })
                    page_idx += 1

                    # Add all parsed content items with updated page indices
                    for item in parsed_content:
                        # Update page_idx relative to our email document
                        item_copy = item.copy()
                        item_copy["page_idx"] = page_idx

                        # For images, the path needs to be absolute
                        if item_copy.get("type") == "image" and "img_path" in item_copy:
                            img_path = item_copy["img_path"]
                            if not os.path.isabs(img_path):
                                # Make it absolute relative to temp file location
                                item_copy["img_path"] = os.path.join(
                                    os.path.dirname(temp_path), img_path
                                )

                        content_items.append(item_copy)
                        page_idx += 1

                    return content_items, page_idx

            except Exception as e:
                logger.warning(f"RAGAnything parser failed for {attachment.filename}: {e}")
                # Fall through to text-based handling

        # Fallback: Handle as text content
        text_content = await self._extract_attachment_as_text(attachment, content_type)

        content_items.append({
            "type": "text",
            "text": f"{context_header}\n\n{text_content}",
            "page_idx": page_idx,
        })
        page_idx += 1

        return content_items, page_idx

    def _build_attachment_context_header(
        self,
        attachment: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
        is_inline: bool,
    ) -> str:
        """Build context header for an attachment."""
        date_str = email.date.isoformat() if email.date else "Unknown"
        attachment_type = "INLINE IMAGE" if is_inline else "ATTACHMENT"
        total = len(email.inline_images) if is_inline else len(email.attachments)

        return f"""
================================================================================
EMAIL {attachment_type}
================================================================================
Bundle-ID: {bundle_id}
Attachment-Index: {index + 1} of {total}
Filename: {attachment.filename}
Content-Type: {attachment.content_type}
Size: {len(attachment.content)} bytes

PARENT EMAIL CONTEXT:
  From: {email.from_address}
  Subject: {email.subject}
  Date: {date_str}
  Message-ID: {email.message_id}
--------------------------------------------------------------------------------
CONTENT:
--------------------------------------------------------------------------------"""

    def _build_attachment_fallback_text(
        self,
        attachment: ParsedAttachment,
        bundle_id: str,
        email: ParsedEmail,
        index: int,
        error_msg: str,
        is_inline: bool,
    ) -> str:
        """Build fallback text when attachment processing fails."""
        context = self._build_attachment_context_header(
            attachment, bundle_id, email, index, is_inline
        )
        return f"""{context}

[Processing failed: {error_msg}]
[Attachment: {attachment.filename} - {attachment.content_type}, {len(attachment.content)} bytes]
================================================================================
"""

    def _save_attachment_to_temp(self, attachment: ParsedAttachment) -> str:
        """Save attachment to a temporary file and return the path."""
        ext = get_file_extension(attachment.filename) or ".bin"
        fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="email_attach_")

        try:
            os.write(fd, attachment.content)
        finally:
            os.close(fd)

        self._temp_files.append(temp_path)
        logger.debug(f"Saved attachment to temp file: {temp_path}")
        return temp_path

    async def _extract_attachment_as_text(
        self, attachment: ParsedAttachment, content_type: str
    ) -> str:
        """Extract text content from an attachment (fallback method)."""
        # Handle text-based files
        if content_type in ("text/plain", "text/csv", "text/markdown", "application/json"):
            try:
                return attachment.content.decode("utf-8", errors="replace")
            except Exception:
                return f"[Text file: {attachment.filename} - Could not decode content]"

        # For other types, return a placeholder
        return f"[Attachment: {attachment.filename} - {content_type}, {len(attachment.content)} bytes]"

    def _cleanup_temp_files(self) -> None:
        """Clean up all temporary files created during processing."""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_path}: {e}")
        self._temp_files.clear()


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


async def get_raganything_for_request(request: Request, rag_anything_instance=None):
    """
    Get the RAGAnything instance for the request.

    In single-instance mode, returns the passed rag_anything_instance.
    In multi-tenant mode, resolves the workspace and gets the RAGAnything instance.
    """
    workspace_manager = getattr(request.app.state, "workspace_manager", None)

    if workspace_manager is not None:
        # Multi-tenant mode - get workspace-specific RAGAnything instance
        workspace = await resolve_workspace_from_request(request)
        return await workspace_manager.get_raganything_instance(workspace)
    else:
        # Single-instance mode - use the provided rag_anything instance
        return rag_anything_instance


# ============================================================================
# Route Factory
# ============================================================================


def create_email_routes(rag, rag_anything=None, api_key: Optional[str] = None):
    """
    Create email ingestion routes.

    Args:
        rag: The default LightRAG instance (for single-instance mode).
        rag_anything: The default RAGAnything instance (for single-instance mode, optional).
        api_key: Optional API key for authentication.

    Returns:
        The configured router.
    """
    combined_auth = get_combined_auth_dependency(api_key)
    _default_rag = rag
    _default_rag_anything = rag_anything

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
- **Outlook**: Open email → File → Save As → Select "Outlook Message Format - Unicode (*.msg)" or drag to folder
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

---

## Scheme Selection (Unified Parameter)

The `scheme` parameter accepts either:

1. **Numeric scheme ID** (e.g., `scheme=1`): Loads configuration from schemes.json
   - Used by WebUI for pre-configured processing schemes

2. **Framework name** (e.g., `scheme=lightrag` or `scheme=raganything`):
   - `lightrag`: Text-only processing (faster, basic image description)
   - `raganything`: Multimodal processing (enhanced image/attachment handling)

3. **Not specified**: Uses server's default (raganything if configured, else lightrag)

## Backward Compatibility

The `framework` parameter is deprecated but still supported.
Priority order: `scheme` > `framework`
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
        attachments: List[UploadFile] = File(
            default=[],
            description="Attachment files (for structured input mode)",
        ),
        inline_images: List[UploadFile] = File(
            default=[],
            description="Inline image files (for structured input mode)",
        ),
        scheme: Optional[str] = Form(
            None,
            description="Processing scheme: either a numeric scheme ID (e.g., '1') "
            "or a framework name ('lightrag' or 'raganything'). "
            "If not specified, uses server default (raganything if available, else lightrag).",
        ),
        # Backward compatibility alias (deprecated)
        framework: Optional[str] = Form(
            None,
            description="[Deprecated] Alias for 'scheme'. Use 'scheme' parameter instead.",
        ),
    ):
        """
        Ingest an email with attachments as a related document bundle.

        The email content and attachments are processed and stored with linking
        metadata that preserves their relationships in the knowledge graph.
        """
        try:
            # Resolve scheme with backward compatibility
            # Priority: scheme > framework
            effective_scheme = scheme or framework

            resolved = resolve_scheme(
                scheme=effective_scheme,
                request=http_request,
                rag_anything_instance=_default_rag_anything,
            )

            # Validate framework availability
            validate_framework_availability(
                resolved.framework, http_request, _default_rag_anything
            )

            if resolved.scheme_name:
                logger.info(
                    f"Email ingestion using scheme '{resolved.scheme_name}' with framework: {resolved.framework}"
                )
            else:
                logger.info(f"Email ingestion using framework: {resolved.framework}")

            # Create appropriate ingestion service based on framework
            use_raganything = resolved.framework == "raganything"

            if use_raganything:
                # Use RAGAnything for full multimodal processing
                rag_anything_instance = await get_raganything_for_request(
                    http_request, _default_rag_anything
                )
                service = EmailIngestionServiceRAGAnything(rag_anything_instance)
                logger.info("Using RAGAnything multimodal email ingestion service")
            else:
                # Use LightRAG with optional vision model for basic processing
                rag_instance = await get_lightrag_for_request(http_request, _default_rag)

                # Get vision model if available (for image description)
                vision_model_func = None
                workspace_manager = getattr(http_request.app.state, "workspace_manager", None)
                if workspace_manager:
                    shared = getattr(workspace_manager, "_shared", None)
                    if shared:
                        vision_model_func = getattr(shared, "vision_model_func", None)
                elif _default_rag_anything is not None:
                    # Single-instance mode - get vision_model_func from RAGAnything
                    vision_model_func = getattr(
                        _default_rag_anything, "vision_model_func", None
                    )

                service = EmailIngestionService(rag_instance, vision_model_func)
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
                    logger.info(f"Uploaded {file_size / (1024*1024):.2f}MB to temp file")

                    # Parse in thread pool to avoid blocking
                    parsed_email = await parse_eml_from_file_async(temp_file_path)

                finally:
                    # Always clean up temp file
                    if temp_file_path:
                        cleanup_temp_file(temp_file_path)

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

                # Process attachment files
                parsed_attachments = []
                for att_file in attachments:
                    if att_file.filename:
                        content = await att_file.read()
                        parsed_attachments.append(ParsedAttachment(
                            filename=att_file.filename,
                            content_type=att_file.content_type or "application/octet-stream",
                            content=content,
                            is_inline=False,
                        ))

                # Process inline image files
                parsed_inline = []
                for img_file in inline_images:
                    if img_file.filename:
                        content = await img_file.read()
                        parsed_inline.append(ParsedAttachment(
                            filename=img_file.filename,
                            content_type=img_file.content_type or "image/png",
                            content=content,
                            is_inline=True,
                        ))

                # Build ParsedEmail from structured input
                parsed_email = ParsedEmail(
                    message_id=email_meta.message_id or EmailParser._generate_message_id(),
                    from_address=email_meta.from_address,
                    to_addresses=email_meta.to_addresses,
                    cc_addresses=email_meta.cc_addresses,
                    subject=email_meta.subject,
                    date=datetime.fromisoformat(email_meta.date) if email_meta.date else None,
                    body_text=body_text or email_meta.body_text or "",
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

            # Generate bundle_id and track_id for background processing
            bundle_id = hashlib.sha256(parsed_email.message_id.encode()).hexdigest()[:12]
            bundle_id = f"email_{bundle_id}"
            track_id = generate_track_id("email")

            # Define background task for email ingestion
            async def process_email_background(
                svc: Any,
                email: ParsedEmail,
                t_id: str,
            ):
                """Background task to ingest email."""
                try:
                    logger.info(f"[{t_id}] Starting background email ingestion for: {email.subject}")
                    result = await svc.ingest_email(email)
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
                t_id=track_id,
            )

            return EmailIngestionResponse(
                status="success",
                bundle_id=bundle_id,
                message=f"Email '{parsed_email.subject}' accepted. Processing will continue in background.",
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
                    "application/pdf (requires PyMuPDF)",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document (requires python-docx)",
                ],
                "image_types": [
                    "image/png",
                    "image/jpeg",
                    "image/gif",
                    "image/webp",
                ],
                "metadata_only": [
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Other binary formats",
                ],
            },
            "notes": [
                "Images are described using vision model if available",
                "PDF text extraction requires PyMuPDF (fitz)",
                "DOCX extraction requires python-docx",
                "All documents include linking metadata for relationship preservation",
                f"Maximum .eml file size: {MAX_EMAIL_SIZE_BYTES // (1024*1024)}MB",
                "Large files are streamed to disk and parsed in background threads",
            ],
        }

    return router
