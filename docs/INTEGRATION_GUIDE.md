# LightRAG Enterprise Server — Integration Guide

How to integrate external applications, services, and tools with the LightRAG
enterprise server (LightRAG 1.5.0, branch `enterprise-1.5.0`).

This guide covers authentication, multi-tenancy, the REST API surface, the
OpenAI- and Ollama-compatible endpoints, end-to-end integration patterns, and
error handling. Examples assume the server is reachable at
`http://localhost:9621` (the default; `PORT` is configurable).

## Contents

1. [Integration surfaces](#1-integration-surfaces)
2. [Authentication](#2-authentication)
3. [Multi-tenancy & workspaces](#3-multi-tenancy--workspaces)
4. [Core API](#4-core-api)
5. [OpenAI-compatible API](#5-openai-compatible-api)
6. [Ollama-compatible API](#6-ollama-compatible-api)
7. [Admin API](#7-admin-api)
8. [Integration patterns](#8-integration-patterns)
9. [Error reference](#9-error-reference)
10. [Configuration reference](#10-configuration-reference)

---

## 1. Integration surfaces

| Surface | Use it to | Section |
|---|---|---|
| REST API (`/query`, `/documents/*`, `/graph*`) | Index documents, run RAG queries, manage the graph | [4](#4-core-api) |
| OpenAI-compatible API (`/v1/*`) | Plug LightRAG into tools that speak the OpenAI API | [5](#5-openai-compatible-api) |
| Ollama-compatible API (`/api/*`) | Use LightRAG as an "Ollama model" (e.g. Open WebUI) | [6](#6-ollama-compatible-api) |
| Email ingestion (`/documents/email`) | Ingest `.eml` files or structured emails | [4](#4-core-api) |
| Admin API (`/admin/*`) | Manage tenant workspaces (multi-tenancy only) | [7](#7-admin-api) |

Interactive API docs are always available at `/docs` (Swagger) and `/redoc`.

---

## 2. Authentication

### 2.1 Choosing a method

| Integration scenario | Recommended method |
|---|---|
| Interactive user in a browser / the WebUI | **OAuth2 SSO** (Keycloak) |
| Interactive user, SSO disabled | **Username/password → JWT** (`/login`) |
| Your own backend service / automation | **Client Credentials** (Keycloak service account) |
| A script or CI job acting as one user | **Per-user API key** |
| A fully-trusted internal service | **Shared API key** (`X-API-Key`) |

All authenticated requests (except SSO cookies) present credentials in the
**`Authorization: Bearer <token>`** header, except the shared API key which uses
**`X-API-Key`**.

### 2.2 Username / password → JWT

```bash
curl -X POST http://localhost:9621/login \
  -d "username=alice&password=secret"
```
Request is `application/x-www-form-urlencoded` with `username` + `password`.
Response:
```json
{ "access_token": "eyJ…", "token_type": "bearer", "auth_mode": "enabled",
  "role": "user", "core_version": "1.5.0", "api_version": "0295" }
```
Use the token: `Authorization: Bearer eyJ…`. Wrong credentials → **401**.

Accounts come from the `AUTH_ACCOUNTS` env var. If `AUTH_ACCOUNTS` is unset,
authentication is **disabled** — `/login` and `/auth-status` hand out a *guest*
token and every request is allowed.

### 2.3 OAuth2 / Keycloak SSO

Enabled by default (`OAUTH2_ENABLED=true`). Flow:

1. `GET /oauth2/authorize` → `{ "authorization_url": "...", "state": "..." }`.
   The server generates a PKCE challenge tied to `state` (10-minute TTL).
2. Redirect the user to `authorization_url`; they authenticate at Keycloak.
3. Keycloak redirects back to the configured `OAUTH2_REDIRECT_URI` with
   `?code=…&state=…`. Two callback variants:
   - **`GET /oauth2/callback`** (WebUI) — exchanges the code, sets cookies
     (`lightrag_token` HTTP-only, `lightrag_user` readable metadata), and
     302-redirects into the WebUI.
   - **`GET /api/oauth2/callback`** (REST clients) — same exchange but returns
     JSON: `{ access_token, token_type, auth_mode:"sso", role, username, … }`.
4. Use the `access_token` as a normal Bearer token, or rely on the
   `lightrag_token` cookie (browser clients — send `withCredentials`).

The SSO username is the Keycloak `email` (fallback `preferred_username`, `sub`).
`role` is `admin` only if that username is listed in `ADMIN_ACCOUNTS`.

If OAuth2 is enabled but `OAUTH2_CLIENT_ID` / `OAUTH2_CLIENT_SECRET` are unset,
`/oauth2/authorize` returns **503**. See
[KEYCLOAK_SSO_SETUP.md](./KEYCLOAK_SSO_SETUP.md) and
[OAuth2-SSO-Authentication.md](./OAuth2-SSO-Authentication.md).

### 2.4 Client Credentials (service accounts)

For backend services and automation, register a Keycloak **service account**
client and obtain a token directly from Keycloak:

```bash
curl -X POST https://keycloak.example.com/realms/your-realm/protocol/openid-connect/token \
  -d "grant_type=client_credentials" \
  -d "client_id=my-service" -d "client_secret=…"
```
Present the resulting access token as `Authorization: Bearer <token>`. LightRAG
validates it via JWKS (`validate_any_token`) and treats it as an **admin
service account**. Service accounts **must** specify the target workspace with
the `X-Target-Workspace` header on every workspace operation (see §3).

### 2.5 Per-user API keys

A user mints a personal key (no OAuth token juggling):

```bash
curl -X POST http://localhost:9621/api-keys \
  -H "Authorization: Bearer <user-JWT>" \
  -H "Content-Type: application/json" \
  -d '{"name": "ci-pipeline", "expires_in_days": 30}'
```
Response (the full key is shown **once**):
```json
{ "api_key": "sk-lightrag-1a2b3c4d-…", "id": "…", "name": "ci-pipeline",
  "created_at": "…", "expires_at": "…" }
```
The key format is `sk-lightrag-{workspace_hash}-{random}` — the workspace is
**embedded in the key**. Use it exactly like a JWT:
```bash
curl -X POST http://localhost:9621/query \
  -H "Authorization: Bearer sk-lightrag-1a2b3c4d-…" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LightRAG?", "mode": "hybrid"}'
```
The server stores only a hash of the key. Manage keys with `GET /api-keys`
(list, metadata only) and `DELETE /api-keys/{key_id}` (revoke). Service
accounts cannot create per-user keys (**403**).

### 2.6 Shared API key

Set `LIGHTRAG_API_KEY` on the server. Clients send it as **`X-API-Key`**:
```bash
curl http://localhost:9621/documents/status_counts \
  -H "X-API-Key: <LIGHTRAG_API_KEY>" \
  -H "X-Target-Workspace: alice_example_com"
```
A matching `X-API-Key` is treated as an **admin service account** — like
Client Credentials, it must supply `X-Target-Workspace` for workspace
operations.

### 2.7 How the server resolves credentials

`combined_auth` tries, in order: (1) whitelisted path → allow; (2)
`Authorization: Bearer sk-lightrag-…` → per-user API key; (3) `lightrag_token`
cookie as a fallback token source; (4) JWT / Keycloak token via
`validate_any_token`; (5) if neither auth nor API key is configured → allow;
(6) `X-API-Key` shared key. Default public paths (`WHITELIST_PATHS`) are
`/health` and `/api/*`.

### 2.8 Token auto-renewal

When a JWT is close to expiry the server mints a fresh one and returns it in
the **`X-New-Token`** response header (rate-limited to once per user per 60 s).
**Integrators should read `X-New-Token` on every response and, if present,
replace their stored token with it.**

---

## 3. Multi-tenancy & workspaces

Multi-tenancy is enabled by default (`ENABLE_MULTI_TENANCY=true`). Each user
gets an isolated **workspace** — separate knowledge graph, embeddings, and
documents.

- **Normal users** (JWT, SSO, or per-user API key): the workspace is resolved
  automatically from the caller's identity — no header needed. It is the
  `workspace_id` JWT claim, derived from the username/email by
  `sanitize_workspace_id` (lowercase; non-`[a-zA-Z0-9_-]` → `_`; e.g.
  `alice@example.com` → `alice_example_com`).
- **Admins and service accounts** can operate **on behalf of** another
  workspace with the **`X-Target-Workspace`** header:
  ```
  X-Target-Workspace: alice_example_com
  ```
  - A **non-admin** sending this header → **403**.
  - **Service accounts** (shared `X-API-Key` or Client Credentials) have *no*
    personal workspace, so they **must** send `X-Target-Workspace` on every
    workspace operation — omitting it → **400**.
  - Service-account on-behalf-of access is additionally gated by the **OBO
    allowlist** (`.obo_allowlist` file / `OBO_*` env vars). A denied target →
    **401**. See [OBO_ALLOWLIST.md](../lightrag/api/OBO_ALLOWLIST.md).

> Per-user API keys (`sk-lightrag-…`) carry their workspace inside the key and
> behave as a normal user — do **not** send `X-Target-Workspace` with them.

---

## 4. Core API

All core endpoints require auth (§2) and are workspace-scoped (§3).

### 4.1 Query

`POST /query` — RAG query, returns a generated answer.
```bash
curl -X POST http://localhost:9621/query \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"query": "Summarise the Q4 report", "mode": "hybrid"}'
```
`QueryRequest` fields: `query` (str, ≥3 chars), `mode`
(`local`|`global`|`hybrid`|`naive`|`mix`|`bypass`, default `mix`),
`top_k`, `chunk_top_k`, `response_type`, `conversation_history`
(`[{role, content}]`), `user_prompt`, `enable_rerank`, `include_references`
(default `true`), `only_need_context`, `only_need_prompt`, `stream`.

- `POST /query/stream` — same body, streams NDJSON (`application/x-ndjson`).
- `POST /query/data` — returns structured entities/relationships/chunks
  instead of generated text.

### 4.2 Document ingestion

| Endpoint | Body | Notes |
|---|---|---|
| `POST /documents/upload` | `multipart/form-data`, `file` | Max 100 MB; same-name conflict → 409 |
| `POST /documents/text` | JSON `{text, file_source?}` | Insert one text document |
| `POST /documents/texts` | JSON `{texts[], file_sources?[]}` | Insert many |
| `POST /documents/scan` | — | Index new files in the input dir |
| `POST /documents/email` | `multipart/form-data` | See §4.3 |

```bash
curl -X POST http://localhost:9621/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/report.pdf"
```
Ingestion is asynchronous — the response carries a **`track_id`**. Poll
`GET /documents/track_status/{track_id}` for progress, or
`GET /documents/pipeline_status` for the overall pipeline state.

### 4.3 Email ingestion

`POST /documents/email` — `multipart/form-data`, two modes:

- **Mode 1 — raw `.eml`:** `email_file=@email.eml` (RFC 822, ≤100 MB). Headers,
  body, inline images, and attachments are all extracted.
- **Mode 2 — structured:** `metadata` (JSON string: `from`, `to[]`, `subject`,
  …), `body_text`, `attachments` (files), `inline_images` (files).

```bash
curl -X POST http://localhost:9621/documents/email \
  -H "Authorization: Bearer <token>" \
  -F "email_file=@/path/to/email.eml"
```
Inline images are described by the native VLM when `VLM_PROCESS_ENABLE=true`.
All parts share a Bundle ID so their relationships survive in the graph.

### 4.4 Graph

`GET /graphs?label=<x>` (subgraph), `GET /graph/label/list`,
`GET /graph/label/search?q=<x>`, `GET /graph/entity/exists?name=<x>`,
`POST /graph/entity/edit|create`, `POST /graph/relation/edit|create`,
`POST /graph/entities/merge`.

### 4.5 Document management

`GET /documents/paginated` (filtered list), `GET /documents/status_counts`,
`DELETE /documents/delete_document` (by ID), `DELETE /documents` (clear all),
`POST /documents/clear_cache`, `POST /documents/reprocess_failed`.

---

## 5. OpenAI-compatible API

Point any OpenAI-API client at `http://localhost:9621/v1`.

`POST /v1/chat/completions` — standard request: `model`, `messages`
(`[{role, content}]`), `stream`. The last message is the query; earlier
messages become conversation history. Streaming uses SSE
(`data: {…}` … `data: [DONE]`).

```bash
curl -X POST http://localhost:9621/v1/chat/completions \
  -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"model": "lightrag", "messages": [{"role":"user","content":"What is LightRAG?"}], "stream": false}'
```

`GET /v1/models` lists: `lightrag` (default — mix mode), `lightrag-local`,
`lightrag-global`, `lightrag-hybrid`, `lightrag-naive`, `lightrag-mix`.

**Query mode** is chosen by: a message prefix (`/local …`, `/global …`,
`/bypass …`) → else the model-name suffix (`lightrag-global` → global) → else
`mix`.

---

## 6. Ollama-compatible API

Lets tools that speak the Ollama API (e.g. **Open WebUI**) use LightRAG as a
model. Endpoints under `/api`: `GET /api/version`, `GET /api/tags`,
`GET /api/ps`, `POST /api/generate` (direct LLM, no retrieval),
`POST /api/chat` (routes through RAG; `/local`, `/global`, … prefixes select
the mode).

> ⚠️ `/api/*` is in the default `WHITELIST_PATHS`, so the Ollama endpoints are
> **public by default**. To require auth on them, override `WHITELIST_PATHS`
> (e.g. to just `/health`).

---

## 7. Admin API

Available only when `ENABLE_MULTI_TENANCY=true`; all routes require the
**admin** role. Use them to manage tenant workspaces.

| Endpoint | Purpose |
|---|---|
| `GET /admin/workspaces` | List workspaces (paginated) |
| `POST /admin/workspaces` | Pre-create a workspace for a user |
| `GET /admin/workspaces/{id}` | Workspace statistics |
| `DELETE /admin/workspaces/{id}` | Delete a workspace and all its data |
| `POST /admin/workspaces/{id}/impersonate` | Mint a 1-hour impersonation JWT |
| `GET /admin/status` | Admin dashboard status |

---

## 8. Integration patterns

### 8.1 Web application via SSO
Send the user to `GET /oauth2/authorize`, redirect to the returned URL. After
Keycloak login your callback page receives the `lightrag_token` cookie (WebUI
callback) or a JSON token (`/api/oauth2/callback`). Make API calls with that
token; honour `X-New-Token` on responses.

### 8.2 Backend service ingesting into many tenants
Use a Keycloak Client Credentials token (§2.4). Ensure the client is in the
OBO allowlist for the target workspaces. On every call set
`X-Target-Workspace: <tenant>`:
```bash
TOKEN=$(curl -s -X POST https://keycloak…/token -d grant_type=client_credentials \
        -d client_id=ingestor -d client_secret=… | jq -r .access_token)
curl -X POST http://localhost:9621/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Target-Workspace: tenant_a" -F "file=@doc.pdf"
```

### 8.3 Per-user script / CI job
Have the user mint a per-user API key once (§2.5); store it as a secret; use it
as a Bearer token. No `X-Target-Workspace` needed — the key is workspace-bound.

### 8.4 Open WebUI / OpenAI-API tool
Point the tool's "OpenAI API base URL" at `http://your-host:9621/v1` and its
API key at a LightRAG token or per-user API key. Pick the `lightrag-*` model
for the desired retrieval mode.

---

## 9. Error reference

| Status | Meaning |
|---|---|
| **400** | Service account (shared key / Client Credentials) called a workspace endpoint without `X-Target-Workspace`; or an invalid/expired OAuth2 `state`. |
| **401** | Invalid/expired token or API key; auth required but none supplied; OBO target denied for a service account (returned as 401 by design). |
| **403** | Invalid/missing `X-API-Key`; non-admin used `X-Target-Workspace`; non-admin hit `/admin/*`; service account tried to create a per-user key. |
| **409** | Upload conflicts with an existing same-named document. |
| **413** | Upload exceeds `MAX_UPLOAD_SIZE`. |
| **503** | OAuth2 enabled but not configured (`OAUTH2_CLIENT_ID`/`SECRET` missing); identity provider unreachable; a multi-tenant dependency used while multi-tenancy is disabled. |

---

## 10. Configuration reference

Server-side env vars most relevant to integration (full list in `env.example`):

| Variable | Purpose |
|---|---|
| `OAUTH2_ENABLED` | OAuth2/SSO on/off (default `true`) |
| `OAUTH2_CLIENT_ID`, `OAUTH2_CLIENT_SECRET` | Keycloak client credentials (required for SSO) |
| `OAUTH2_ISSUER`, `OAUTH2_*_ENDPOINT`, `OAUTH2_REDIRECT_URI` | Keycloak endpoints |
| `ENABLE_MULTI_TENANCY` | Multi-tenancy on/off (default `true`) |
| `AUTH_ACCOUNTS` | `user:password` pairs for password login |
| `ADMIN_ACCOUNTS` | Usernames granted the admin role |
| `TOKEN_SECRET` | Signs all JWTs — **required** in production |
| `TOKEN_EXPIRE_HOURS`, `TOKEN_AUTO_RENEW` | JWT lifetime / sliding renewal |
| `LIGHTRAG_API_KEY` | Shared `X-API-Key` value |
| `WHITELIST_PATHS` | Public (no-auth) paths — default `/health,/api/*` |
| `OBO_ALLOWLIST_PATH`, `OBO_DEFAULT_POLICY` | On-behalf-of allowlist for service accounts |
| `VLM_PROCESS_ENABLE`, `VLM_LLM_*` | Native multimodal / inline-image vision |

### Related documentation
- [Linux Installation Guide](./LINUX_INSTALLATION_GUIDE.md)
- [Keycloak SSO Setup](./KEYCLOAK_SSO_SETUP.md)
- [OAuth2 SSO Authentication](./OAuth2-SSO-Authentication.md)
- [OBO Allowlist](../lightrag/api/OBO_ALLOWLIST.md)
- [API Server Guide](./LightRAG-API-Server.md)
