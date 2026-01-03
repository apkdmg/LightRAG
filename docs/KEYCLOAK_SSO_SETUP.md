# Keycloak OAuth2 SSO Setup Guide

This guide explains how to configure Keycloak as an OAuth2 Single Sign-On (SSO) provider for LightRAG. This enables users to log in using their organization's identity provider while preserving the existing username/password authentication option.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Keycloak Client Configuration](#keycloak-client-configuration)
- [LightRAG Configuration](#lightrag-configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

---

## Overview

LightRAG supports dual authentication:
- **Username/Password**: Traditional login using credentials defined in `AUTH_ACCOUNTS`
- **OAuth2 SSO**: Single Sign-On via Keycloak (or compatible OpenID Connect providers)

Both methods can be enabled simultaneously, giving users the choice of how to authenticate.

### How It Works

```
┌─────────────────────────────────────┐
│           Login Page                │
│  ┌─────────────────────────────────┐│
│  │  [Sign in with SSO]             ││ ← Keycloak OAuth2 (PKCE)
│  └─────────────────────────────────┘│
│  ──────── Or continue with ──────── │
│  ┌─────────────────────────────────┐│
│  │  Username: [___________]        ││ ← Existing password auth
│  │  Password: [___________]        ││
│  │  [        Login        ]        ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### Authentication Flow

1. User clicks "Sign in with SSO"
2. Browser redirects to Keycloak login page
3. User authenticates with Keycloak
4. Keycloak redirects back to LightRAG with authorization code
5. LightRAG exchanges code for tokens and validates the ID token
6. LightRAG creates a local JWT for the session
7. User is logged in with their Keycloak identity

---

## Prerequisites

- Admin access to your Keycloak server
- LightRAG server with HTTPS enabled (required for production)
- Network connectivity between LightRAG and Keycloak servers

---

## Keycloak Client Configuration

### Step 1: Access Keycloak Admin Console

1. Navigate to your Keycloak admin URL (e.g., `https://your-keycloak-server/admin`)
2. Log in with admin credentials
3. Select your realm from the dropdown (top-left)

### Step 2: Create a New Client

1. Navigate to **Clients** in the left sidebar
2. Click **Create client**

#### General Settings

| Field | Value | Description |
|-------|-------|-------------|
| Client type | OpenID Connect | Standard OAuth2/OIDC protocol |
| Client ID | `lightrag-web` | Unique identifier for your app |
| Name | LightRAG Web Application | Human-readable name |
| Description | LightRAG Knowledge Graph RAG System | Optional description |

3. Click **Next**

#### Capability Config

| Setting | Value | Description |
|---------|-------|-------------|
| Client authentication | **OFF** | Makes this a public client (required for SPAs) |
| Authorization | OFF | Not needed for authentication-only |
| Standard flow | **ON** ✓ | Required for Authorization Code flow |
| Direct access grants | OFF | Not used |
| Implicit flow | OFF | Deprecated, not recommended |
| Service accounts roles | OFF | Not needed for user authentication |

4. Click **Next**

#### Login Settings

Replace `https://your-lightrag-server` with your actual LightRAG URL:

| Field | Value |
|-------|-------|
| Root URL | `https://your-lightrag-server` |
| Home URL | `https://your-lightrag-server` |
| Valid redirect URIs | `https://your-lightrag-server/*` |
| Valid post logout redirect URIs | `https://your-lightrag-server/*` |
| Web origins | `https://your-lightrag-server` |

> **Important**: The redirect URI pattern `/*` allows all paths. For tighter security, you can use `https://your-lightrag-server/oauth2/callback` specifically.

5. Click **Save**

### Step 3: Configure PKCE (Proof Key for Code Exchange)

PKCE is required for public clients to prevent authorization code interception attacks.

1. Go to the **Settings** tab of your newly created client
2. Scroll to **Advanced** tab
3. Find **Advanced Settings** section
4. Set **Proof Key for Code Exchange Code Challenge Method** to **S256**
5. Click **Save**

### Step 4: Verify Client Scopes

1. Go to the **Client scopes** tab
2. Ensure these scopes are assigned (Default or Optional):
   - `openid` - Required for OIDC
   - `profile` - Provides name, preferred_username
   - `email` - Provides email address

These are typically assigned by default. If missing, click **Add client scope** to add them.

### Step 5: Note Your Realm's OIDC Endpoints

You can find all endpoints at your realm's well-known configuration:

```
https://your-keycloak-server/realms/YOUR_REALM/.well-known/openid-configuration
```

Key endpoints you'll need:
- Authorization endpoint
- Token endpoint
- Userinfo endpoint
- JWKS URI

---

## LightRAG Configuration

Add the following to your `.env` file:

```bash
# =============================================================================
# OAUTH2/KEYCLOAK SSO CONFIGURATION
# =============================================================================

# Enable OAuth2 SSO (set to false to disable SSO button)
OAUTH2_ENABLED=true

# Keycloak client credentials
# Use the Client ID you created in Keycloak
OAUTH2_CLIENT_ID=lightrag-web

# Client secret - only needed for confidential clients
# Leave commented out for public clients (recommended for SPAs)
# OAUTH2_CLIENT_SECRET=your-client-secret

# Keycloak OpenID Connect endpoints
# Replace YOUR_KEYCLOAK_SERVER and YOUR_REALM with your values
OAUTH2_ISSUER=https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM
OAUTH2_AUTHORIZATION_ENDPOINT=https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/protocol/openid-connect/auth
OAUTH2_TOKEN_ENDPOINT=https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/protocol/openid-connect/token
OAUTH2_USERINFO_ENDPOINT=https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/protocol/openid-connect/userinfo
OAUTH2_JWKS_URI=https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/protocol/openid-connect/certs

# OAuth2 callback URL - MUST match Keycloak valid redirect URIs exactly
OAUTH2_REDIRECT_URI=https://YOUR_LIGHTRAG_SERVER/oauth2/callback

# OAuth2 scopes to request (space-separated)
OAUTH2_SCOPES=openid profile email
```

### Example Configuration

For a LightRAG server at `https://cloudengine.unimas.my` with Keycloak at `https://id.unimas.my`:

```bash
OAUTH2_ENABLED=true
OAUTH2_CLIENT_ID=lightrag-web
OAUTH2_ISSUER=https://id.unimas.my/realms/UNIMAS
OAUTH2_AUTHORIZATION_ENDPOINT=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/auth
OAUTH2_TOKEN_ENDPOINT=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/token
OAUTH2_USERINFO_ENDPOINT=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/userinfo
OAUTH2_JWKS_URI=https://id.unimas.my/realms/UNIMAS/protocol/openid-connect/certs
OAUTH2_REDIRECT_URI=https://cloudengine.unimas.my/oauth2/callback
OAUTH2_SCOPES=openid profile email
```

### Granting Admin Access to SSO Users

By default, all SSO users receive the "user" role. To grant admin privileges to specific SSO users, add their username (email) to `ADMIN_ACCOUNTS`:

```bash
# Comma-separated list of admin usernames
# Works for both password and SSO users
ADMIN_ACCOUNTS=admin,manager,john.doe@example.com,jane.smith@example.com
```

When these users log in via SSO, they will automatically receive admin privileges.

---

## Testing

### Step 1: Restart LightRAG

After updating your `.env` file, restart the LightRAG server:

```bash
# If running directly
lightrag-server

# If using Docker
docker-compose restart lightrag
```

### Step 2: Verify SSO Button Appears

1. Navigate to your LightRAG login page
2. You should see "Sign in with SSO" button above the username/password form
3. If the button doesn't appear, check that `OAUTH2_ENABLED=true`

### Step 3: Test the Login Flow

1. Click "Sign in with SSO"
2. You should be redirected to Keycloak login page
3. Enter your Keycloak credentials
4. After successful authentication, you should be redirected back to LightRAG
5. Verify you're logged in with your Keycloak username

### Step 4: Verify Multi-Tenancy (if enabled)

If `ENABLE_MULTI_TENANCY=true`, each SSO user gets their own isolated workspace:
- Workspace ID is derived from username (e.g., `john.doe@example.com` → `john_doe_example_com`)
- Documents and knowledge graphs are isolated per user

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| SSO button not visible | OAuth2 disabled | Set `OAUTH2_ENABLED=true` in `.env` and restart |
| "Invalid redirect URI" | URI mismatch | Ensure `OAUTH2_REDIRECT_URI` exactly matches Keycloak's Valid redirect URIs |
| "Invalid client" | Wrong client ID | Verify `OAUTH2_CLIENT_ID` matches Keycloak client ID exactly |
| "PKCE verification failed" | PKCE not configured | Set PKCE method to `S256` in Keycloak Advanced settings |
| CORS errors | Missing web origin | Add your LightRAG URL to Web origins in Keycloak |
| "Invalid state" | Session expired | State tokens expire after 10 minutes; try logging in again |
| "ID token issuer mismatch" | Wrong issuer URL | Verify `OAUTH2_ISSUER` matches your realm URL exactly |

### Debug Logging

Enable debug logging to troubleshoot issues:

```bash
LOG_LEVEL=DEBUG
```

Check the logs for OAuth2-related messages:
```
lightrag.api.oauth2 - Generated authorization URL with state: abc123...
lightrag.api.oauth2 - Exchanging authorization code for tokens...
lightrag.api.oauth2 - ID token validated for user: john.doe@example.com
```

### Testing Keycloak Connectivity

Verify LightRAG can reach Keycloak endpoints:

```bash
# Test JWKS endpoint
curl https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/protocol/openid-connect/certs

# Test well-known configuration
curl https://YOUR_KEYCLOAK_SERVER/realms/YOUR_REALM/.well-known/openid-configuration
```

---

## Security Considerations

### Public vs Confidential Clients

LightRAG uses a **public client** configuration because:
- The frontend is a Single Page Application (SPA)
- JavaScript code is visible to users in the browser
- Client secrets cannot be safely stored in browser code

Security is maintained through:
- **PKCE (S256)**: Prevents authorization code interception
- **State parameter**: Prevents CSRF attacks
- **JWKS validation**: Verifies ID tokens using Keycloak's public keys

### Recommendations

1. **Always use HTTPS** in production for both LightRAG and Keycloak
2. **Restrict redirect URIs** to specific paths rather than wildcards when possible
3. **Keep Keycloak updated** to receive security patches
4. **Monitor login attempts** in Keycloak's admin console
5. **Use short token lifetimes** and implement token refresh if needed

### Session Management

- SSO login creates a local JWT token stored in the browser
- Logging out of LightRAG only clears the local session
- It does **not** end the Keycloak session (single logout not implemented)
- Users remain logged into Keycloak and can re-authenticate without entering credentials

---

## Additional Resources

- [Keycloak Documentation](https://www.keycloak.org/documentation)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [PKCE RFC 7636](https://tools.ietf.org/html/rfc7636)
- [OpenID Connect Core](https://openid.net/specs/openid-connect-core-1_0.html)
