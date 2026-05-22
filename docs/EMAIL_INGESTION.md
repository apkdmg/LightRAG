# Email Ingestion

Ingest emails — complete `.eml` files or structured email payloads — together
with their attachments and inline images into the LightRAG knowledge graph as a
linked document bundle. This is an **enterprise-fork feature** (not part of
upstream `HKUDS/LightRAG`), exposed by the API server under `/documents/email`.

## Overview

The email ingestion feature turns an email into a set of connected documents in
the knowledge graph. A single email may contain a body, several attachments, and
inline images; ingesting them as unrelated blobs would lose the fact that they
belong together. Instead, LightRAG:

1. Parses the email into its components (headers, plain-text body, inline
   images, attachments).
2. Builds a **master document** carrying the email metadata, body, and an
   inventory of every attachment.
3. Creates a separate document for each attachment and inline image, with
   extracted text (for documents) or a vision-model description (for images).
4. Stamps **every** document with a shared **Bundle ID** so the
   email-↔-attachment relationships survive in the graph.

This makes graph queries like *"What was in John's Q4 report email?"* or
*"Show me the attachments from the budget discussion"* answerable, because the
retrieval can follow the bundle relationship.

Ingestion is **asynchronous**: the endpoint validates and parses the email
inline, then hands the heavy work (LLM extraction, vision description, text
extraction) to a FastAPI background task. The response returns immediately with
a `track_id` you can poll for progress.

The implementation lives in `lightrag/api/routers/email_routes.py`.

## Exporting emails

Mode 1 (raw upload) needs an `.eml` file — the standard RFC 822 email format.
Most desktop and web email clients can produce one:

- **Outlook** — Open the email → **File → Save As** → choose the message
  format; or drag the email out of Outlook into a folder to get a file on disk.
- **Gmail** — Open the email → the **three-dots menu** → **Download message**
  (equivalently, **Show original → Download Original**) saves a `.eml` file.
- **Thunderbird** — Right-click the email → **Save As** → save it as an `.eml`
  file.
- **Apple Mail** — Drag the email to the Finder, or **File → Save As**.

> The exported file must be a true RFC 822 message (`.eml`). Outlook's
> proprietary `.msg` format is **not** an `.eml` file — if your client only
> offers `.msg`, drag-export to a folder (which yields `.eml`) or use a
> converter, otherwise parsing will fail.

If you do not have an `.eml` file (for example, you already hold parsed email
data from an API or database), use the structured payload mode below instead.

## Ingestion modes

Both modes target the same endpoint, `POST /documents/email`, with
`multipart/form-data`. The server picks the mode by which fields are present:
`email_file` → Mode 1; otherwise `metadata` → Mode 2; neither → `400`.

### Mode 1 — Raw `.eml` upload (recommended)

Upload a single complete `.eml` file in the `email_file` field. The file
contains everything — headers, body, inline images, and attachments — so no
other fields are needed. The server streams the upload to a temporary file
(64 KB chunks, so a 100 MB email never sits fully in memory), parses it in a
worker thread, then deletes the temp file.

Use this mode when you have a real email exported from a mail client and want
the simplest, highest-fidelity path.

### Mode 2 — Structured payload

Supply the email as separate form fields when you have already parsed (or
synthesised) the email data programmatically:

- `metadata` — a JSON **string** with the email headers (see the schema in the
  API reference below).
- `body_text` — the plain-text body (a form field, optional).
- `attachments` — zero or more uploaded attachment files (repeatable field).
- `inline_images` — zero or more uploaded inline image files (repeatable
  field).

Use this mode for integrations that hold structured email data, for synthetic
test fixtures, or when you want to control exactly which parts are ingested.

## API reference

The router is mounted with the `/documents` prefix and tagged `email`. All
endpoints require authentication via the server's combined auth dependency —
send a credential exactly as for any other LightRAG API call (`Authorization:
Bearer <token>`, a per-user API key, or `X-API-Key`; see
[`INTEGRATION_GUIDE.md`](./INTEGRATION_GUIDE.md) §2). Requests are
workspace-scoped: service accounts must add `X-Target-Workspace`.

### `POST /documents/email` — Ingest email with attachments

| | |
|---|---|
| **Method / path** | `POST /documents/email` |
| **Content type** | `multipart/form-data` |
| **Auth** | Required (combined auth) |
| **Response model** | `EmailIngestionResponse` |
| **Success status** | `200 OK` |

**Request fields** (form fields; all individually optional, but at least one of
`email_file` or `metadata` is required):

| Field | Type | Mode | Description |
|---|---|---|---|
| `email_file` | file (`.eml`) | 1 | Complete `.eml` file with all attachments |
| `metadata` | string (JSON) | 2 | JSON string of email metadata (see below) |
| `body_text` | string | 2 | Plain-text email body |
| `attachments` | file[] (repeatable) | 2 | Attachment files |
| `inline_images` | file[] (repeatable) | 2 | Inline image files |

**`metadata` JSON schema** (`EmailMetadata`) — used in Mode 2:

| Key | Type | Required | Notes |
|---|---|---|---|
| `from` | string | yes | Sender email address |
| `to` | string[] | no (default `[]`) | Recipient addresses |
| `cc` | string[] | no (default `[]`) | CC addresses |
| `subject` | string | no (default `""`) | Email subject |
| `date` | string | no | ISO-8601 date, e.g. `2024-01-15T10:30:00` |
| `message_id` | string | no | Unique message ID; auto-generated if omitted |
| `thread_id` | string | no | Thread / conversation ID |
| `body_text` | string | no | Plain-text body (the `body_text` form field takes precedence) |
| `body_html` | string | no | HTML body |

**Response schema** — `EmailIngestionResponse`:

| Field | Type | Description |
|---|---|---|
| `status` | string | `"success"` on acceptance |
| `bundle_id` | string | The shared Bundle ID for this email (e.g. `email_a1b2c3d4e5f6`) |
| `message` | string | Human-readable acceptance message |
| `track_id` | string | Tracking ID for the background job; poll `GET /documents/track_status/{track_id}` |
| `documents_created` | int | Number of documents created — **`0` in the response**, because work runs in the background |
| `email_subject` | string | Subject of the ingested email |
| `attachments_processed` | int | Attachments processed — **`0` in the response** (background) |
| `inline_images_processed` | int | Inline images processed — **`0` in the response** (background) |

> **Background-processing note.** Because ingestion is asynchronous, the
> `documents_created`, `attachments_processed`, and `inline_images_processed`
> fields in the immediate response are always `0`. The real counts are produced
> by the background task and logged under the `track_id`; observe final status
> through `GET /documents/track_status/{track_id}`.

The codebase also defines an extended model **`EmailIngestionDetailedResponse`**
(it inherits every `EmailIngestionResponse` field and adds an `attachments`
array of `EmailAttachmentInfo` objects — `filename`, `content_type`,
`size_bytes`, `is_inline`). It is available for use by detailed-response
variants; the current `POST /documents/email` endpoint returns the base
`EmailIngestionResponse`.

**Status codes:**

| Code | Meaning |
|---|---|
| `200` | Email accepted; background processing started |
| `400` | Neither `email_file` nor `metadata` supplied; or invalid `metadata` JSON |
| `401` / `403` | Authentication / workspace authorization failure |
| `413` | `.eml` upload exceeds the size limit (see *Attachments & inline images*) |
| `500` | Upload could not be written, or an unexpected ingestion error |

### `GET /documents/email/supported-formats` — Supported formats

| | |
|---|---|
| **Method / path** | `GET /documents/email/supported-formats` |
| **Auth** | Required (combined auth) |
| **Response** | JSON describing supported formats |

Returns a static description of what the email ingester accepts. No request
body. The response contains:

- `email_formats` — currently just `.eml` (RFC 822), with
  `supports_attachments` and `supports_inline_images` both `true`.
- `attachment_types` — three buckets: `fully_supported` (text-based formats,
  PDF, DOCX), `image_types` (PNG, JPEG, GIF, WebP), and `metadata_only`
  (Excel and other binary formats — see next section).
- `notes` — operational notes, including the maximum `.eml` size and the
  optional dependencies for PDF/DOCX extraction.

Use this endpoint as a machine-readable capability check before building an
integration.

### Tracking progress

The `track_id` returned by `POST /documents/email` is a standard LightRAG
tracking ID (prefix `email`). Poll it the same way as any other ingestion job:

```bash
curl http://localhost:9621/documents/track_status/<track_id> \
  -H "Authorization: Bearer <token>"
```

## Attachments & inline images

### How parts are classified

When parsing an `.eml`, each MIME part is sorted:

- `text/plain` and `text/html` parts (not marked as attachments) become the
  email **body**.
- A part marked `inline` *or* carrying a `Content-ID`, with an `image/*`
  content type, becomes an **inline image**.
- Anything else marked as an attachment, or any non-text part, becomes an
  **attachment**.

Parts without a filename are given a generated name with an extension inferred
from the `Content-Type` header, falling back to magic-byte detection for images
(PNG, JPEG, GIF, BMP, WebP, TIFF signatures).

### Attachment text extraction

Each attachment is turned into a document whose content depends on its type:

| Content type | Handling |
|---|---|
| `text/plain`, `text/csv`, `text/markdown`, `application/json` | Decoded as UTF-8 text |
| `application/pdf` | Text extracted with **PyMuPDF** (`fitz`) if installed; otherwise a placeholder |
| `.docx` (`...wordprocessingml.document`), `application/msword` | Text extracted with **python-docx** if installed; otherwise a placeholder |
| `image/*` | Described by the vision model (see below) |
| `application/vnd.ms-excel`, `...spreadsheetml.sheet` (Excel) | **Metadata only** — filename and size, no content extraction |
| Any other binary type | **Metadata only** — a placeholder noting filename, type, and size |

PDF and DOCX extraction depend on optional packages (`PyMuPDF`, `python-docx`).
If a package is missing, the attachment is still ingested but its document
carries a placeholder instead of extracted text — ingestion does not fail.

### Inline-image vision (requires VLM)

Inline images — and any `image/*` attachment — are passed to a vision model so
that their visual content (charts, diagrams, embedded text) becomes searchable.

The vision model is wired in **only when `VLM_PROCESS_ENABLE=true`**. The email
router receives a vision function backed by the native `vlm` LLM role; when the
master switch is off, no vision function is supplied and images fall back to a
plain placeholder (`"[Image: <filename> ... No vision model available ...]"`).

To enable inline-image description, configure the VLM in your `.env`:

```bash
VLM_PROCESS_ENABLE=true
# The effective VLM binding must be vision-capable (openai, azure_openai,
# gemini, bedrock, or ollama — lollms is rejected at startup).
VLM_LLM_BINDING=openai
VLM_LLM_MODEL=your_vlm_model
VLM_LLM_BINDING_HOST=https://api.example.com/v1
VLM_LLM_BINDING_API_KEY=your_vlm_api_key
# Optional: per-role concurrency / timeout / image-size cap
# VLM_MAX_ASYNC_LLM=4
# VLM_LLM_TIMEOUT=180
# VLM_MAX_IMAGE_BYTES=5242880
```

If `VLM_LLM_*` is left unset, the `vlm` role falls back to the base `LLM_*`
configuration, which must itself be vision-capable. If a vision call fails at
runtime, the image is still ingested with a "Vision processing failed"
placeholder — one bad image never aborts the bundle.

### Size limit

The maximum `.eml` upload size is defined in code as:

```python
MAX_EMAIL_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
```

This limit is a **fixed compile-time constant** in
`lightrag/api/routers/email_routes.py` — it is **not** read from an environment
variable and is **not configurable** without editing the source. (Note that it
is separate from the general `MAX_UPLOAD_SIZE` env var that governs
`POST /documents/upload`.) The streaming uploader enforces it chunk by chunk; an
oversized `.eml` is rejected with HTTP `413` before the whole file is read.

Mode 2 (structured payload) attachment and inline-image files are read into
memory directly and are **not** subject to `MAX_EMAIL_SIZE_BYTES`; their size is
bounded only by the server's general request limits.

## Bundle IDs

A **Bundle ID** is the identifier that ties an email and all its parts together
in the knowledge graph. It is derived deterministically from the email's
`Message-ID`:

```
bundle_id = "email_" + sha256(message_id)[:12]
```

If the email has no `Message-ID` (or you omit `message_id` in Mode 2), the
server generates a synthetic one (`<uuid@lightrag.local>`) first, so a Bundle ID
always exists. The same input email therefore always yields the same Bundle ID.

### How parts relate in the graph

Every document produced from one email carries the Bundle ID inline in its text:

- The **master document** begins with `Email: <subject>` and a header block
  containing `Bundle-ID:`, `Message-ID:`, `Thread-ID:`, sender/recipients,
  date, body, and a list of all attachments.
- Each **inline-image document** begins with `EMAIL INLINE IMAGE` and repeats
  `Bundle-ID:`, the image index, content type, `Content-ID`, size, the parent
  email subject, and the vision description.
- Each **attachment document** begins with `EMAIL ATTACHMENT` and repeats
  `Bundle-ID:`, the attachment index, content type, size, the parent email
  subject, and the extracted text.

Because the Bundle ID, the parent subject, and the sender appear as text in
every component, LightRAG's entity/relationship extraction links the components
through the graph — the Bundle ID and email subject become shared nodes that all
the bundle's documents connect to.

### Querying a bundle

Query the bundle the same way as any LightRAG knowledge base — there is no
dedicated bundle endpoint. Reference the Bundle ID or the email subject in a
natural-language query:

```bash
curl -X POST http://localhost:9621/query \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"query": "Summarise everything in bundle email_a1b2c3d4e5f6", "mode": "mix"}'
```

```bash
curl -X POST http://localhost:9621/query \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"query": "What attachments were in the Q4 Report email and what did they contain?", "mode": "hybrid"}'
```

`mix` or `hybrid` retrieval modes work best, since they combine graph traversal
(to follow the bundle relationships) with vector search (to find the relevant
content).

## Usage examples

### Mode 1 — Raw `.eml` upload

```bash
curl -X POST "http://localhost:9621/documents/email" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "email_file=@/path/to/email.eml"
```

A service account targeting another workspace adds the workspace header:

```bash
curl -X POST "http://localhost:9621/documents/email" \
  -H "Authorization: Bearer YOUR_SERVICE_TOKEN" \
  -H "X-Target-Workspace: john_doe_unimas_my" \
  -F "email_file=@/path/to/email.eml"
```

Example response:

```json
{
  "status": "success",
  "bundle_id": "email_a1b2c3d4e5f6",
  "message": "Email 'Q4 Report' accepted. Processing will continue in background.",
  "track_id": "email_20260522_103012_abc123",
  "documents_created": 0,
  "email_subject": "Q4 Report",
  "attachments_processed": 0,
  "inline_images_processed": 0
}
```

### Mode 2 — Structured payload

```bash
curl -X POST "http://localhost:9621/documents/email" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F 'metadata={"from": "sender@example.com", "to": ["recipient@example.com"], "subject": "Q4 Report", "date": "2024-01-15T10:30:00"}' \
  -F "body_text=Here is the Q4 report as discussed." \
  -F "attachments=@/path/to/report.pdf" \
  -F "attachments=@/path/to/data.xlsx" \
  -F "inline_images=@/path/to/chart.png"
```

### Check supported formats

```bash
curl "http://localhost:9621/documents/email/supported-formats" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `500` — `Failed to ingest email` right after upload | Malformed / non-RFC-822 `.eml` (e.g. an Outlook `.msg` renamed to `.eml`, or a corrupt file) | Re-export a genuine `.eml` from the mail client (see *Exporting emails*); do not rename a `.msg` file |
| `413` — *File size exceeds maximum allowed size of 100MB* | The `.eml` is larger than the fixed `MAX_EMAIL_SIZE_BYTES` (100 MB) | Reduce the email size (e.g. strip large attachments and ingest them separately), or raise the constant in `email_routes.py` and rebuild |
| Attachment ingested but its document only shows `[Spreadsheet: ...]` or `[Attachment: ...]` | The attachment type is **metadata-only** (Excel, other binary formats) — by design no text is extracted | Convert the file to a supported text format (CSV, PDF, DOCX) before sending, or accept metadata-only indexing |
| Attachment document shows `PyMuPDF not installed` / `python-docx not installed` | The optional PDF/DOCX extraction package is missing on the server | Install the dependency (`PyMuPDF` for PDF, `python-docx` for DOCX) and re-ingest |
| `200` returned but final counts (in `track_status`) show fewer documents than expected — partial failure | One or more attachments / inline images failed during background processing | Per-part failures are caught and logged (search server logs for the `track_id`); the rest of the bundle is still ingested. Fix the offending file and re-ingest the email |
| Inline images appear in the graph only as `[Image: ... No vision model available ...]` | `VLM_PROCESS_ENABLE` is `false`, so no vision function was wired into the email router | Set `VLM_PROCESS_ENABLE=true` and configure `VLM_LLM_*` with a vision-capable binding, then restart the server |
| Server refuses to start: VLM binding does not support image inputs | `VLM_PROCESS_ENABLE=true` but the effective VLM binding is `lollms` (or otherwise not vision-capable) | Set `VLM_LLM_BINDING` (or `LLM_BINDING`) to one of `openai`, `azure_openai`, `gemini`, `bedrock`, `ollama` |
| `400` — *Must provide either 'email_file' or 'metadata'* | The request supplied neither an `.eml` file nor a `metadata` JSON string | Send `email_file` (Mode 1) or `metadata` (Mode 2) |
| `400` — *Invalid metadata JSON* | The `metadata` form field is not valid JSON, or is missing the required `from` field | Fix the JSON; ensure `from` is present (see the `EmailMetadata` schema) |

## Related documentation

- [`INTEGRATION_GUIDE.md`](./INTEGRATION_GUIDE.md) — authentication,
  multi-tenancy, workspace headers, and the rest of the document-ingestion API
  (section 4.3 covers email ingestion at a glance).
- [`LightRAG-API-Server.md`](./LightRAG-API-Server.md) — full API server
  reference, configuration, and deployment.
