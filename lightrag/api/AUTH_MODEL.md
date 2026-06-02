# Authentication & Authorization Model

This document describes how LightRAG's API server decides **who** a caller is
(authentication), **what role** they get (authorization), and **what data** they
can reach (workspace scope). It is the companion to [OBO_ALLOWLIST.md](OBO_ALLOWLIST.md),
which covers the on-behalf-of (OBO) client allowlist in detail.

## Two-axis model

Every request resolves to a single principal. A principal has a *type* (how it
authenticated) and, from that, a *role* and a *data scope*.

| Principal type | How authenticated | How role is decided | How data scope is decided | Config home |
| --- | --- | --- | --- | --- |
| Human (password) | `POST /login` with username/password (`AUTH_ACCOUNTS`); issues a LightRAG JWT | `admin` if the typed username is in `ADMIN_ACCOUNTS` (case-insensitive), else `user` | Workspace derived from the username | `.env`: `AUTH_ACCOUNTS`, `ADMIN_ACCOUNTS` |
| Human (SSO) | OAuth2/OIDC Authorization Code flow via Keycloak; ID token validated, then a LightRAG JWT is issued | `admin` if **email OR preferred_username OR sub** matches `ADMIN_ACCOUNTS` (case-insensitive), else `user` | Workspace derived from email (fallback preferred_username) | `.env`: `OAUTH2_*`, `ADMIN_ACCOUNTS` |
| Raw Keycloak user token | `Authorization: Bearer <keycloak access token>` validated via JWKS | `admin` if **preferred_username OR email** matches `ADMIN_ACCOUNTS` (case-insensitive), else `user` | Workspace derived from email (fallback preferred_username) | `.env`: `OAUTH2_*`, `ADMIN_ACCOUNTS` |
| Service account (client credentials) | Keycloak client-credentials token (`Authorization: Bearer ...`) | `admin` only if `client_id` is in `OBO_ADMIN_CLIENTS`, else `user` (default: no service account is admin) | `service_account`; may target other workspaces via `X-Target-Workspace` if allowed by `OBO_ALLOWED_CLIENTS` | `.obo_allowlist`: `OBO_ADMIN_CLIENTS`, `OBO_ALLOWED_CLIENTS` |
| Global API key | `X-API-Key: <LIGHTRAG_API_KEY>` | `LIGHTRAG_API_KEY_ROLE` (`admin` by default, can be set to `user`) | `service_account`; may do OBO via `X-Target-Workspace` if `OBO_API_KEY_ALLOWED=true` | `.env`: `LIGHTRAG_API_KEY`, `LIGHTRAG_API_KEY_ROLE`; `.obo_allowlist`: `OBO_API_KEY_*` |
| Per-user API key | Per-user key resolved to that user's identity | Inherits that user's role | That user's workspace | Per-user key store |
| Guest | No auth configured (or `auth_mode=disabled`) | `guest` | Default workspace | n/a |

## Rule of thumb

- **Humans → `ADMIN_ACCOUNTS`** (in `.env`). One case-insensitive list of email
  addresses and/or usernames; matched the same way across `/login`, SSO, and raw
  Keycloak user tokens.
- **Machines → `.obo_allowlist`**: `OBO_ADMIN_CLIENTS` decides the *role* of a
  service account, `OBO_ALLOWED_CLIENTS` decides its *workspace scope* (OBO via
  `X-Target-Workspace`), and `OBO_API_KEY_*` governs the global API key's OBO
  behaviour.
- **Global API key role → `LIGHTRAG_API_KEY_ROLE`** (`admin` by default; set to
  `user` so a leaked shared key cannot act as an admin).

## What the `admin` role actually gates

The `admin` role gates **only the `/admin` routes** (see
`routers/admin_routes.py`), and **only when multi-tenancy is enabled**.

Critically: with `ENABLE_MULTI_TENANCY=false`:

- The admin routes are **not mounted** (`create_admin_routes()` is only included
  when `args.enable_multi_tenancy` is true).
- The OBO / `X-Target-Workspace` mechanism **gates nothing** — all data routes
  operate on the single default workspace.

So in a single-tenant deployment, the `admin` vs `user` distinction does not
restrict access to the data/query/document routes at all. **Single-tenant
deployments must control access at authentication time** (e.g. who has a valid
account, a valid SSO identity, or the API key) rather than relying on the role.

## See also

- [OBO_ALLOWLIST.md](OBO_ALLOWLIST.md) — service-account / API-key OBO allowlist,
  admin-client grants, and hot-reload behaviour.
