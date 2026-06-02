# OBO (On-Behalf-Of) Client Allowlist

> For the big picture of how principals, roles, and data scope fit together, see [AUTH_MODEL.md](AUTH_MODEL.md).

Controls which OAuth2 clients and API keys can perform OBO operations using the `X-Target-Workspace` header.

## Quick Start

1. Copy the starter file [`.obo_allowlist.example`](../../.obo_allowlist.example) (repo root) into your working directory as `.obo_allowlist`
2. Edit your config (see format below)
3. Changes apply automatically within 60 seconds (no restart needed)

## Config File

**Location:** `{working_dir}/.obo_allowlist` (or set `OBO_ALLOWLIST_PATH` env var)

**Format:** `.env`-like syntax with `KEY=VALUE` pairs

```
# Format: [client_id:workspace1,workspace2] or [client_id:*] for all
OBO_ALLOWED_CLIENTS=[backend-service:*],[partner-app:tenant_a,tenant_b]

# Allow shared X-API-Key to do OBO
OBO_API_KEY_ALLOWED=true
OBO_API_KEY_WORKSPACES=*

# Default policy for unlisted clients
OBO_DEFAULT_POLICY=deny

# Grant the global admin role to these service-account client_ids
OBO_ADMIN_CLIENTS=admin-tool
```

## Config Options

| Key | Default | Description |
|-----|---------|-------------|
| `OBO_ALLOWED_CLIENTS` | `` | Client allowlist (see format below) |
| `OBO_API_KEY_ALLOWED` | `false` | Can shared X-API-Key do OBO? |
| `OBO_API_KEY_WORKSPACES` | `` | Workspaces for API key: `*` or `ws1,ws2` |
| `OBO_DEFAULT_POLICY` | `deny` | Action for unlisted clients: `deny` or `allow` |
| `OBO_ADMIN_CLIENTS` | `` | Service-account client_ids granted the global admin role (see below) |

## OBO_ALLOWED_CLIENTS Format

```
[client_id:workspaces],[client_id:workspaces],...
```

Where `workspaces` is either:

- `*` - access to all workspaces
- `workspace1,workspace2` - access to specific workspaces only

### Examples

**Single client with full access:**

```
OBO_ALLOWED_CLIENTS=[backend-service:*]
```

**Single client with restricted access:**

```
OBO_ALLOWED_CLIENTS=[partner-app:tenant_a,tenant_b]
```

**Multiple clients:**

```
OBO_ALLOWED_CLIENTS=[backend-service:*],[partner-app:tenant_a,tenant_b],[admin-tool:*]
```

## OBO_ADMIN_CLIENTS (service-account admin role)

`OBO_ADMIN_CLIENTS` grants the **global `admin` role** to the listed
service-account client_ids (the token's `azp` / `clientId` claim from a
client-credentials / service-account token).

```
# Comma-separated client_ids. No wildcard ("*" is NOT supported for admin).
OBO_ADMIN_CLIENTS=n8n,backend-service
```

- **Format:** comma-separated `client_id` values; whitespace is stripped and
  empty entries are dropped. There is **no workspace dimension** and **no `*`
  wildcard** — admin is a flat, global allowlist (kept deliberately strict).
- **Hot-reloaded:** like the rest of `.obo_allowlist`, changes apply within
  60 seconds (no restart). It flows through the same file-mtime reload path.
- **Default (empty):** NO service account is granted admin; client-credentials
  tokens authenticate as a normal `user`.

### Role vs scope (admin vs OBO)

These two controls answer different questions:

- **`OBO_ADMIN_CLIENTS` → role.** Grants the `admin` role, which gates the
  global admin routes (`admin_routes.py`). This is a process-wide capability.
- **`OBO_ALLOWED_CLIENTS` → scope.** Gates *which workspace data* a client may
  touch via the `X-Target-Workspace` header (per-workspace OBO).

A client can have one, both, or neither.

### Precedence with the deprecated env var

The admin allowlist is resolved with this precedence (first match wins; once a
source is present it is authoritative — later sources are ignored):

1. `OBO_ADMIN_CLIENTS` key in the `.obo_allowlist` file.
2. `OBO_ADMIN_CLIENTS` environment variable.
3. `OAUTH2_SERVICE_ACCOUNT_ADMIN_CLIENTS` environment variable
   (**DEPRECATED**; logs a one-time warning when used). Migrate this to
   `OBO_ADMIN_CLIENTS` in `.obo_allowlist`.
4. Empty (no admin).

Note: if the file defines `OBO_ADMIN_CLIENTS` (even with a different/empty
list), the file is authoritative and the env vars are ignored for admin.

### Example: both controls together

```
OBO_DEFAULT_POLICY=deny
OBO_ALLOWED_CLIENTS=[n8n:space1]   # which workspace data n8n may touch
OBO_ADMIN_CLIENTS=n8n              # grant n8n the admin role (global admin routes)
```

## Full Examples

### Allow one service full access

```
OBO_ALLOWED_CLIENTS=[my-backend:*]
OBO_API_KEY_ALLOWED=true
OBO_API_KEY_WORKSPACES=*
OBO_DEFAULT_POLICY=deny
```

### Multiple services with different permissions

```
OBO_ALLOWED_CLIENTS=[internal-api:*],[partner-x:partner_workspace],[partner-y:ws1,ws2,ws3]
OBO_API_KEY_ALLOWED=false
OBO_DEFAULT_POLICY=deny
```

### Backward compatible (allow all - not recommended)

```
OBO_DEFAULT_POLICY=allow
OBO_API_KEY_ALLOWED=true
OBO_API_KEY_WORKSPACES=*
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OBO_ALLOWLIST_PATH` | `{working_dir}/.obo_allowlist` | Config file path |
| `OBO_DEFAULT_POLICY` | `deny` | Fallback if no config file |
| `OBO_ADMIN_CLIENTS` | `` | Fallback admin allowlist if not set in the file |
| `OAUTH2_SERVICE_ACCOUNT_ADMIN_CLIENTS` | `` | DEPRECATED admin allowlist fallback (use `OBO_ADMIN_CLIENTS`) |

## Behavior

- **No config file**: Uses `OBO_DEFAULT_POLICY` env var (default: `deny`)
- **Client not in list**: Uses `OBO_DEFAULT_POLICY` from config
- **Denied request**: Returns `401 Unauthorized` (not 403, to avoid info leakage)
- **Hot-reload**: File checked every 60 seconds, reloaded if modified

## What This Affects

| Auth Method | Affected? | Notes |
|-------------|-----------|-------|
| OAuth2 Client Credentials | Yes | Checked against `OBO_ALLOWED_CLIENTS` |
| Shared X-API-Key | Yes | Checked against `OBO_API_KEY_ALLOWED` |
| Per-user API keys (`sk-lightrag-*`) | No | These embed workspace, don't use OBO |
| Regular user tokens | No | Users access their own workspace |
