# Per-User API Keys

> Long-lived, workspace-scoped credentials that let a single integration authenticate as one user without a browser login or a JWT. **This is an enterprise-fork feature** — it is not part of upstream LightRAG.

## Overview

A per-user API key is a LightRAG-native credential of the form `sk-lightrag-…` that a logged-in user mints for themselves. It is meant for **one integration tied to one user** — for example, a Continue.dev plugin, an n8n workflow, or a personal script that should always operate inside that user's own workspace.

Each key has the user's workspace **encoded directly into the key string**. Because the workspace is part of the credential, the server can resolve it automatically: the caller never sends an `X-Target-Workspace` header, and the key can never reach another workspace.

### How it differs from the shared `LIGHTRAG_API_KEY`

LightRAG also supports a single server-wide secret, `LIGHTRAG_API_KEY`, sent as the `X-API-Key` header. The two mechanisms are very different:

| | Per-user API key | Shared API key (`LIGHTRAG_API_KEY`) |
|---|---|---|
| Header | `Authorization: Bearer sk-lightrag-…` | `X-API-Key: <secret>` |
| Scope | One specific user's workspace, embedded in the key | Server-wide |
| Identity granted | A regular **user** (`role: user`, `auth_mode: user_api_key`) | An **admin service account** (`role: admin`, `auth_mode: api_key`) |
| Workspace selection | Automatic, from the key — `X-Target-Workspace` is **not** used | Must send `X-Target-Workspace` on every workspace operation (OBO-gated) |
| Who creates it | Any authenticated end user, for themselves | Set once by the server operator via env var |
| Revocation | Per key, via `DELETE /api-keys/{id}` | Only by rotating the single shared secret |
| Number of keys | Many per user, each independently named, dated, and revocable | Exactly one per server |
| Storage | SHA-256 hash, per workspace, on disk | Plaintext in server config / env |

### When to use it

- **Use a per-user API key** when a single automated integration acts for exactly one user and should be confined to that user's workspace (personal scripts, IDE plugins, single-tenant workflows).
- **Use the shared `LIGHTRAG_API_KEY`** for a trusted internal service that legitimately needs to act across workspaces and has no Keycloak available.
- For a multi-tenant backend that routes many users into their own workspaces, prefer **Keycloak Client Credentials** (see [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) §2.4).

## How it works

### Key format

A per-user API key has four hyphen-separated segments:

```
sk-lightrag-{workspace_hash}-{random}
```

- `sk-lightrag-` — fixed prefix. The server uses it to recognise the credential and route it through per-user-key validation **before** any JWT decode is attempted (`sk-lightrag-` strings are not JWTs).
- `{workspace_hash}` — the first 8 hex characters of `SHA256(workspace_id)`. This is what lets the server resolve the workspace without an `X-Target-Workspace` header.
- `{random}` — 32 bytes of cryptographically secure randomness, URL-safe encoded (`secrets.token_urlsafe(32)`).

Because both `workspace_hash` and `random` are themselves generated with `token_urlsafe` (which can contain `-`), validation does **not** assume exactly four segments — it splits on `-` and requires **at least four** parts, then reads `parts[2]` as the workspace hash.

### Per-workspace storage

Keys are stored per workspace, on disk, in a JSON file:

```
{WORKING_DIR}/{workspace_id}/.api_keys.json
```

`WORKING_DIR` is the server's working directory (`WORKING_DIR` env var, default `./rag_storage`). The file is created on first key creation, and its parent directory is created if missing. Its shape is:

```json
{
  "keys": [
    {
      "id": "key_a1b2c3d4e5f6a7b8",
      "key_hash": "<sha256 hex of the full key>",
      "key_preview": "Ab3z",
      "name": "n8n integration",
      "created_at": "2026-05-22T10:00:00.000000",
      "last_used_at": "2026-05-22T11:30:00.000000",
      "expires_at": "2026-06-21T10:00:00.000000"
    }
  ]
}
```

### Hashing — the plaintext key is never stored

The full `sk-lightrag-…` string is returned to the caller **exactly once**, in the response to the create request. The server keeps only `SHA256(api_key)` in `key_hash`, plus the last 4 characters in `key_preview` for display. If you lose the key, it cannot be recovered — you must mint a new one and revoke the old.

On every request, the presented key is SHA-256-hashed and compared against the stored `key_hash`. Expiry, if set, is checked at the same time. The `last_used_at` timestamp is updated as a best-effort, fire-and-forget operation; a failure to write it never blocks the request.

### The workspace-hash registry

To turn a `{workspace_hash}` back into a `workspace_id`, the server keeps an in-memory map, `_workspace_hash_registry` (`workspace_hash → workspace_id`). Lookup works in two stages:

1. **Registry hit** — if the hash is already in the in-memory map, the workspace is returned immediately.
2. **Directory scan fallback** — on a miss, the server scans the sub-directories of `WORKING_DIR`, hashes each directory name, and matches it against the requested hash. A successful match is **added to the registry**, so subsequent lookups for that workspace are fast.

The registry is also populated whenever a new key is created (`register_workspace_hash` is called as part of the create flow).

### What happens on server restart

The registry is **in-memory only** and is **not pre-populated at startup** — it starts empty on every boot. This is intentional and does not break anything:

- The first per-user-key request after a restart for a given workspace produces a registry miss, falls back to the directory scan, finds the workspace directory, and re-populates the registry entry. Every later request for that workspace is a fast registry hit.
- Keys themselves survive restarts because they live in the on-disk `.api_keys.json` files, not in memory.

The only operational consequence: if a workspace's directory has been deleted from `WORKING_DIR`, the directory scan can no longer resolve its hash, and keys for that workspace stop validating (the request is rejected as an unknown workspace hash).

### Where it fits in the auth pipeline

The `combined_auth` dependency (in `lightrag/api/utils_api.py`) checks credentials in a fixed order. The per-user API key is handled **second**, right after whitelisted paths and before any JWT or cookie handling:

1. Whitelisted path → allow.
2. `Authorization` header starts with `Bearer sk-lightrag-` → validate as a per-user API key (`validate_user_api_key`). On success the resolved identity is stored on `request.state.api_key_user` and the request is allowed; on failure the request is rejected with **401 `Invalid API key`**.
3. Otherwise, fall through to cookie/JWT/Keycloak validation and finally the shared `X-API-Key`.

A valid per-user key resolves to this identity:

- `username`: `apikey-{workspace_id}`
- `role`: `user` — per-user keys deliberately get the regular user role, **never** admin.
- `workspace_id`: the workspace embedded in the key.
- `metadata.auth_mode`: `user_api_key`.

Because `auth_mode` is `user_api_key` (and **not** `api_key` or `client_credentials`), a per-user key is **not** treated as a service account. It always operates on its own embedded workspace, and any attempt to send `X-Target-Workspace` with it is rejected **403** by the workspace resolver (only admins may do on-behalf operations).

## API reference

All endpoints live under the `/api-keys` router. If the server is started with a path prefix (`LIGHTRAG_API_PREFIX`, e.g. `/api/v1`), prepend it to every path below.

Every endpoint is protected by the standard `combined_auth` dependency, so the **caller must already be authenticated** by some other means — a JWT from `/login`, an OAuth2 / Keycloak token, or the SSO cookie. You cannot use a per-user key (or have one minted) without first being a logged-in user.

The request models are defined in `lightrag/api/routers/apikey_routes.py`.

### Create a key — `POST /api-keys`

Mints a new per-user API key for the **caller's own workspace**.

- **Method / path:** `POST /api-keys`
- **Auth required:** Yes — a valid user JWT or Keycloak user token (via `Authorization: Bearer …` or the `lightrag_token` cookie).
- **Who may call it:** Any authenticated regular user or admin. **Service accounts are rejected.** If the caller's `auth_mode` is `api_key` (shared key) or `client_credentials` (Keycloak service account), the request fails **403 `Service accounts cannot create per-user API keys`** — a service account has no personal workspace to scope a key to.

**Request body** (`CreateApiKeyRequest`, `application/json`):

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | **Yes** | Human-friendly label for the key (e.g. `"My Continue.dev key"`). Shown in listings. |
| `expires_in_days` | integer | No | Lifetime in days from creation. Omit (or `null`) for a key that never expires. |

**Response body** (`ApiKeyResponse`, HTTP `200`):

| Field | Type | Description |
|---|---|---|
| `api_key` | string | The full `sk-lightrag-…` key. **Returned only here, only once.** Store it immediately. |
| `id` | string | Key ID, e.g. `key_a1b2c3d4e5f6a7b8` — use this to revoke the key later. |
| `name` | string | The name you supplied. |
| `created_at` | string | ISO-8601 UTC timestamp of creation. |
| `expires_at` | string \| null | ISO-8601 UTC expiry, or `null` if the key never expires. |

**Status codes:**

| Code | Meaning |
|---|---|
| `200` | Key created; full key returned in `api_key`. |
| `401` | Caller is not authenticated (no/invalid token). |
| `403` | Caller is a service account (shared `X-API-Key` or Keycloak Client Credentials). |
| `422` | Request body is malformed (e.g. `name` missing). |

### List keys — `GET /api-keys`

Lists metadata for every key in the **caller's own workspace**. The key strings themselves are never returned.

- **Method / path:** `GET /api-keys`
- **Auth required:** Yes — a valid user token (same as create).
- **Who may call it:** Any authenticated user. There is no explicit service-account block on this endpoint, but a service account has no personal workspace, so the listing reflects whatever workspace the resolver assigns it.

**Request body:** none.

**Response body** (`ApiKeyListResponse`, HTTP `200`): an object with a single `keys` array, each element an `ApiKeyMetadata`:

| Field | Type | Description |
|---|---|---|
| `id` | string | Key ID — pass this to `DELETE /api-keys/{key_id}`. |
| `name` | string | The key's human-friendly name. |
| `created_at` | string | ISO-8601 UTC creation timestamp. |
| `last_used_at` | string \| null | ISO-8601 UTC timestamp of last successful use, or `null` if never used. |
| `expires_at` | string \| null | ISO-8601 UTC expiry, or `null` if non-expiring. |
| `key_preview` | string | Last 4 characters of the key, prefixed with `...` (e.g. `...Ab3z`), to help you tell keys apart. |

**Status codes:**

| Code | Meaning |
|---|---|
| `200` | Listing returned (an empty `keys` array if the workspace has no keys). |
| `401` | Caller is not authenticated. |

### Delete (revoke) a key — `DELETE /api-keys/{key_id}`

Permanently revokes a key. Once revoked, the key string can no longer authenticate any request.

- **Method / path:** `DELETE /api-keys/{key_id}`
- **Path parameter:** `key_id` — the `id` field from a create response or a list entry (e.g. `key_a1b2c3d4e5f6a7b8`).
- **Auth required:** Yes — a valid user token (same as create).
- **Who may call it:** Any authenticated user. A user can only revoke keys in their **own workspace**: deletion is scoped to `user.workspace_id`, so a key ID belonging to another workspace simply will not be found.

**Request body:** none.

**Response body** (HTTP `200`):

```json
{ "message": "API key 'key_a1b2c3d4e5f6a7b8' has been revoked" }
```

**Status codes:**

| Code | Meaning |
|---|---|
| `200` | Key found and revoked. |
| `401` | Caller is not authenticated. |
| `404` | No key with that `key_id` exists in the caller's workspace. |

## Usage examples

The examples assume a server at `http://localhost:9621` and a user JWT in the shell variable `$USER_JWT` (obtain one from `POST /login` or the OAuth2 flow — see [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) §2.2–2.3).

### Create a key

```bash
curl -X POST http://localhost:9621/api-keys \
  -H "Authorization: Bearer $USER_JWT" \
  -H "Content-Type: application/json" \
  -d '{"name": "n8n integration", "expires_in_days": 30}'
```

Response (the only time you ever see the full key):

```json
{
  "api_key": "sk-lightrag-7f3a9c2e-xK9...redacted...Ab3z",
  "id": "key_a1b2c3d4e5f6a7b8",
  "name": "n8n integration",
  "created_at": "2026-05-22T10:00:00.000000",
  "expires_at": "2026-06-21T10:00:00.000000"
}
```

For a key that never expires, simply omit `expires_in_days`:

```bash
curl -X POST http://localhost:9621/api-keys \
  -H "Authorization: Bearer $USER_JWT" \
  -H "Content-Type: application/json" \
  -d '{"name": "personal script"}'
```

### List keys

```bash
curl http://localhost:9621/api-keys \
  -H "Authorization: Bearer $USER_JWT"
```

```json
{
  "keys": [
    {
      "id": "key_a1b2c3d4e5f6a7b8",
      "name": "n8n integration",
      "created_at": "2026-05-22T10:00:00.000000",
      "last_used_at": "2026-05-22T11:30:00.000000",
      "expires_at": "2026-06-21T10:00:00.000000",
      "key_preview": "...Ab3z"
    }
  ]
}
```

### Delete (revoke) a key

```bash
curl -X DELETE http://localhost:9621/api-keys/key_a1b2c3d4e5f6a7b8 \
  -H "Authorization: Bearer $USER_JWT"
```

```json
{ "message": "API key 'key_a1b2c3d4e5f6a7b8' has been revoked" }
```

### Use a minted key against a normal endpoint

Once you have a key, use it exactly like a Bearer token on any protected endpoint. **No `X-Target-Workspace` header is needed** — the workspace is embedded in the key:

```bash
curl -X POST http://localhost:9621/query \
  -H "Authorization: Bearer sk-lightrag-7f3a9c2e-xK9...redacted...Ab3z" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is in my knowledge base?", "mode": "mix"}'
```

The key also works wherever an OpenAI-compatible client expects an API key, since the server resolves the workspace from the key itself:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9621/v1",
    api_key="sk-lightrag-7f3a9c2e-xK9...redacted...Ab3z",
)
```

## Troubleshooting

| Symptom | Likely cause | What the server returns | Fix |
|---|---|---|---|
| Key suddenly stops working | The key's `expires_at` has passed. | `401 Invalid API key` on the request. | Mint a new key; expired keys cannot be renewed. |
| Key stops working after `DELETE` | The key was revoked. The on-disk hash is gone. | `401 Invalid API key`. | Mint a replacement and update the integration. |
| `401 Invalid API key` for a key you believe is valid | Malformed prefix — the credential does not start with `sk-lightrag-`, or has fewer than 4 hyphen-separated segments. | The pipeline either falls through to JWT validation (and fails there) or returns `401 Invalid API key`. | Re-copy the full, untruncated key including the `sk-lightrag-` prefix. |
| `401 Invalid API key`, unknown workspace hash in server logs | The `{workspace_hash}` segment does not match any workspace directory under `WORKING_DIR` — usually the workspace directory was deleted or `WORKING_DIR` changed. | `401 Invalid API key`; log line `Unknown workspace hash in API key`. | Restore the workspace directory or point the server at the correct `WORKING_DIR`. |
| `403 Service accounts cannot create per-user API keys` on `POST /api-keys` | The caller authenticated with the shared `X-API-Key` or a Keycloak Client Credentials token. Service accounts have no personal workspace. | `403`. | Mint the key while logged in as a real user (JWT or OAuth2 user token). |
| `403 Only admins can perform operations on behalf of other users` | An `X-Target-Workspace` header was sent alongside a per-user key. Per-user keys are regular users, not admins, and are already workspace-scoped. | `403`. | Drop the `X-Target-Workspace` header — the key already targets the right workspace. |
| `404 API key '…' not found` on `DELETE` | The `key_id` is wrong, already revoked, or belongs to a different workspace. | `404`. | List keys with `GET /api-keys` to get the correct `id`. |
| `422 Unprocessable Entity` on `POST /api-keys` | The request body is missing the required `name` field, or a field has the wrong type. | `422` with field-level detail. | Send valid JSON with a string `name`. |
| `401 Not authenticated` / `403 API Key required` calling `/api-keys` | No valid credential was presented to the management endpoint itself. | `401` or `403`. | Authenticate first (the `/api-keys` endpoints require a user token). |
| Key works after a server restart but the first request is slightly slower | Expected. The in-memory workspace-hash registry is empty after a restart; the first request rebuilds the entry via a directory scan. | `200`. | No action needed — subsequent requests are fast registry hits. |

## Related documentation

- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) — full overview of every authentication method (§2.5 covers per-user API keys), workspace selection, and the credential-resolution order.
- [OAuth2-SSO-Authentication.md](./OAuth2-SSO-Authentication.md) — Keycloak SSO and how to obtain the user JWT required to mint a per-user key.
- [LightRAG-API-Server.md](./LightRAG-API-Server.md) — running and configuring the LightRAG API server, including `LIGHTRAG_API_KEY`, `WORKING_DIR`, and `LIGHTRAG_API_PREFIX`.
