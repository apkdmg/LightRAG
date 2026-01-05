# OBO (On-Behalf-Of) Client Allowlist

Controls which OAuth2 clients and API keys can perform OBO operations using the `X-Target-Workspace` header.

## Quick Start

1. Create `.obo_allowlist` in your working directory
2. Add your config (see format below)
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
```

## Config Options

| Key | Default | Description |
|-----|---------|-------------|
| `OBO_ALLOWED_CLIENTS` | `` | Client allowlist (see format below) |
| `OBO_API_KEY_ALLOWED` | `false` | Can shared X-API-Key do OBO? |
| `OBO_API_KEY_WORKSPACES` | `` | Workspaces for API key: `*` or `ws1,ws2` |
| `OBO_DEFAULT_POLICY` | `deny` | Action for unlisted clients: `deny` or `allow` |

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
