# OAuth2/SSO Authentication Guide

This document describes the OAuth2/SSO authentication flow in LightRAG, supporting Keycloak and other OpenID Connect providers.

## Overview

LightRAG supports two authentication methods:
1. **Local Authentication** - Username/password via `/login` endpoint
2. **OAuth2/SSO Authentication** - Keycloak or other OIDC providers

## Security Best Practices

This implementation follows industry best practices for OAuth2 in Single Page Applications (SPAs):

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **PKCE** | S256 challenge method | Protects against authorization code interception |
| **HTTP-only Cookies** | Token stored in `httponly` cookie | Prevents XSS attacks from stealing tokens |
| **SameSite Cookies** | `samesite=lax` | CSRF protection |
| **Secure Cookies** | `secure=true` in production | Prevents token interception over HTTP |
| **State Parameter** | Cryptographically random | CSRF protection during OAuth2 flow |
| **JWKS Validation** | ID token validated via Keycloak JWKS | Ensures token authenticity |

## Configuration

### Environment Variables

Add these to your `.env` file to enable OAuth2/SSO:

```env
# Enable OAuth2 SSO authentication
OAUTH2_ENABLED=true

# Keycloak client credentials
OAUTH2_CLIENT_ID=lightrag-web
# OAUTH2_CLIENT_SECRET=          # Only for confidential clients

# Keycloak OpenID Connect endpoints
OAUTH2_ISSUER=https://your-keycloak-server/realms/YOUR_REALM
OAUTH2_AUTHORIZATION_ENDPOINT=https://your-keycloak-server/realms/YOUR_REALM/protocol/openid-connect/auth
OAUTH2_TOKEN_ENDPOINT=https://your-keycloak-server/realms/YOUR_REALM/protocol/openid-connect/token
OAUTH2_USERINFO_ENDPOINT=https://your-keycloak-server/realms/YOUR_REALM/protocol/openid-connect/userinfo
OAUTH2_JWKS_URI=https://your-keycloak-server/realms/YOUR_REALM/protocol/openid-connect/certs

# OAuth2 callback URL - MUST be the FULL URL (not just the path!)
# This URL must match exactly what you configure in Keycloak's "Valid redirect URIs"
# For local development: http://localhost:8020/oauth2/callback
# For production: https://your-domain.com/oauth2/callback
OAUTH2_REDIRECT_URI=http://localhost:8020/oauth2/callback

# OAuth2 scopes to request (space-separated)
OAUTH2_SCOPES=openid profile email
```

### Keycloak Client Setup

1. Create a new client in your Keycloak realm
2. Set **Client ID** to match `OAUTH2_CLIENT_ID`
3. Set **Valid Redirect URIs** to your callback URL (e.g., `http://localhost:8020/oauth2/callback`)
4. Enable **Standard Flow** (Authorization Code Flow)
5. For public clients (recommended for SPAs), enable **PKCE** with S256 challenge method

## API Endpoints

### OAuth2 Configuration Status

```
GET /oauth2/config
```

Returns OAuth2 configuration status (does not expose secrets).

**Response:**
```json
{
  "oauth2_enabled": true,
  "oauth2_provider": "keycloak"
}
```

### Initiate OAuth2 Login

```
GET /oauth2/authorize
```

Initiates the OAuth2 authorization flow. Returns the Keycloak authorization URL for redirect.

**Response:**
```json
{
  "authorization_url": "https://keycloak-server/realms/REALM/protocol/openid-connect/auth?client_id=...&redirect_uri=...&state=...",
  "state": "random-state-string"
}
```

### OAuth2 Callback (WebUI)

```http
GET /oauth2/callback?code=AUTHORIZATION_CODE&state=STATE
```

Handles OAuth2 callback from Keycloak for browser-based WebUI clients.

**Behavior:**
- Sets JWT token in HTTP-only secure cookie (`lightrag_token`)
- Sets user metadata in readable cookie (`lightrag_user`) for frontend display
- Redirects to `/webui/#/oauth2/callback?success=true`

**Security Features:**
- Token is stored in HTTP-only cookie (JavaScript cannot access it)
- Cookie uses `SameSite=Lax` for CSRF protection
- Cookie uses `Secure` flag when SSL is enabled

**Success redirect:**

```text
/webui/#/oauth2/callback?success=true
```

**Cookies set:**
- `lightrag_token` (HTTP-only): Contains the JWT access token
- `lightrag_user` (readable): Contains user metadata JSON

**Error redirect:**

```text
/webui/#/oauth2/callback?error=auth_failed&error_description=Error+message
```

### OAuth2 Callback (REST API)

```
GET /api/oauth2/callback?code=AUTHORIZATION_CODE&state=STATE
```

Handles OAuth2 callback for REST API clients. Returns JSON response.

**Success Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "auth_mode": "sso",
  "role": "user",
  "username": "user@example.com",
  "core_version": "1.4.8.2",
  "api_version": "0223",
  "webui_title": "LightRAG",
  "webui_description": "Simple and Fast Graph Based RAG System"
}
```

**Error Response:**
```json
{
  "detail": "Error message describing the failure"
}
```

## Authentication Flows

### WebUI Flow (Browser-based)

```
┌─────────┐     ┌─────────────┐     ┌──────────┐     ┌─────────────┐
│  User   │     │   WebUI     │     │ LightRAG │     │  Keycloak   │
└────┬────┘     └──────┬──────┘     └─────┬────┘     └──────┬──────┘
     │                 │                  │                 │
     │ Click SSO Login │                  │                 │
     │────────────────>│                  │                 │
     │                 │                  │                 │
     │                 │ GET /oauth2/authorize             │
     │                 │─────────────────>│                 │
     │                 │                  │                 │
     │                 │ {authorization_url, state}        │
     │                 │<─────────────────│                 │
     │                 │                  │                 │
     │ Redirect to Keycloak               │                 │
     │<────────────────│                  │                 │
     │                 │                  │                 │
     │ Login with Keycloak credentials    │                 │
     │─────────────────────────────────────────────────────>│
     │                 │                  │                 │
     │ Redirect to /oauth2/callback?code=...&state=...     │
     │<─────────────────────────────────────────────────────│
     │                 │                  │                 │
     │ GET /oauth2/callback?code=...&state=...             │
     │────────────────────────────────────>│                 │
     │                 │                  │                 │
     │                 │                  │ Exchange code   │
     │                 │                  │────────────────>│
     │                 │                  │                 │
     │                 │                  │ {id_token, ...} │
     │                 │                  │<────────────────│
     │                 │                  │                 │
     │ Redirect to /webui/#/oauth2/callback?access_token=...
     │<───────────────────────────────────│                 │
     │                 │                  │                 │
     │ Frontend parses token and completes login           │
     │────────────────>│                  │                 │
     │                 │                  │                 │
     │ Redirect to home page              │                 │
     │<────────────────│                  │                 │
```

### REST API Flow (Programmatic)

For REST API clients that need OAuth2 authentication:

1. **Get Authorization URL:**
   ```bash
   curl http://localhost:8020/oauth2/authorize
   ```

2. **Complete OAuth2 flow in browser** (user authenticates with Keycloak)

3. **Configure Keycloak** to redirect to `/api/oauth2/callback` instead of `/oauth2/callback`

4. **Exchange code for token:**
   ```bash
   curl "http://localhost:8020/api/oauth2/callback?code=AUTH_CODE&state=STATE"
   ```

5. **Use the token** in subsequent requests:
   ```bash
   curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" http://localhost:8020/health
   ```

## User Roles

SSO users are assigned roles based on the `ADMIN_ACCOUNTS` configuration:

- Users listed in `ADMIN_ACCOUNTS` receive the `admin` role
- All other SSO users receive the `user` role

```env
# Define admin users (comma-separated usernames/emails)
ADMIN_ACCOUNTS=admin@example.com,manager@example.com
```

## Automation Access (n8n, Backend Services)

For automation tools like n8n that need to access user workspaces, LightRAG supports multiple authentication methods with on-behalf access.

### Authentication Methods for Automation

| Method | On-Behalf Support | Use Case |
|--------|-------------------|----------|
| LightRAG JWT | Yes (if admin) | WebUI admin managing users |
| HTTP-only Cookie | Yes (if admin) | WebUI SSO admin |
| Keycloak Access Token (User) | Yes (if in ADMIN_ACCOUNTS) | Mobile admin apps |
| **Keycloak Client Credentials** | **Yes (service account)** | **n8n, backend services (OAuth2 standard)** |
| X-API-Key | Yes (implicit admin) | Simple automation, legacy |

### Using X-API-Key with On-Behalf Access

The simplest method for automation. The API key holder is treated as a service account with admin privileges.

**IMPORTANT**: When using X-API-Key, the `X-Target-Workspace` header is **MANDATORY** for any endpoint that works with workspaces.

```bash
curl -X POST "http://lightrag-server/query" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "X-Target-Workspace: user@example.com" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

### Using OAuth2 Client Credentials (Recommended)

This is the OAuth2-standard approach with token rotation and Keycloak audit logs.

**Step 1: Get service token from Keycloak**

```bash
curl -X POST "https://keycloak-server/realms/REALM/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=lightrag-service" \
  -d "client_secret=SERVICE_SECRET"
```

**Step 2: Use token with on-behalf header**

**IMPORTANT**: When using Client Credentials, the `X-Target-Workspace` header is **MANDATORY** for any endpoint that works with workspaces.

```bash
curl -X POST "http://lightrag-server/query" \
  -H "Authorization: Bearer <service_token>" \
  -H "X-Target-Workspace: user@example.com" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

### Using Direct Keycloak Access Tokens

Mobile apps and backend services can also use Keycloak access tokens directly:

```bash
# User authenticates with Keycloak and gets access token
curl -X POST "http://lightrag-server/query" \
  -H "Authorization: Bearer <keycloak_access_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

### n8n Workflow Examples

#### Using X-API-Key (Simple)

```text
HTTP Request Node:
  Method: POST
  URL: http://lightrag-server/documents/upload
  Headers:
    X-API-Key: ${LIGHTRAG_API_KEY}
    X-Target-Workspace: ${user_email}
  Body: (file upload)
```

#### Using Client Credentials (Recommended)

```text
HTTP Request Node 1 - Get Service Token:
  Method: POST
  URL: https://keycloak-server/realms/REALM/protocol/openid-connect/token
  Body (Form):
    grant_type: client_credentials
    client_id: lightrag-n8n-service
    client_secret: ${KEYCLOAK_CLIENT_SECRET}
  Output: {{ $json.access_token }}

HTTP Request Node 2 - Call LightRAG:
  Method: POST
  URL: http://lightrag-server/documents/upload
  Headers:
    Authorization: Bearer {{ $node["Get Token"].json.access_token }}
    X-Target-Workspace: ${user_email}
  Body: (file upload)
```

### Backend Service Example (Python)

```python
import httpx

# Get service token from Keycloak
token_response = httpx.post(
    "https://keycloak/realms/REALM/protocol/openid-connect/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "lightrag-backend-service",
        "client_secret": "SERVICE_SECRET",
    }
)
access_token = token_response.json()["access_token"]

# Call LightRAG on behalf of a user
response = httpx.post(
    "http://lightrag-server/query",
    headers={
        "Authorization": f"Bearer {access_token}",
        "X-Target-Workspace": "user@example.com",  # On-behalf access
    },
    json={"query": "What is RAG?"}
)
```

### Keycloak Setup for Client Credentials

In Keycloak admin console:

1. Create a new client (e.g., `lightrag-n8n-service`)
2. Set **Client authentication**: ON (confidential client)
3. Enable **Service accounts roles**: ON
4. Set **Valid redirect URIs**: (not needed for client credentials)
5. Copy the **Client secret** from Credentials tab

## Security Considerations

1. **PKCE (Proof Key for Code Exchange):** LightRAG uses PKCE with S256 challenge method for secure authorization code exchange, protecting against code interception attacks.

2. **State Parameter:** A cryptographically secure state parameter is used for CSRF protection.

3. **Token Validation:** ID tokens are validated using Keycloak's JWKS (JSON Web Key Set) to ensure authenticity.

4. **Local JWT:** After successful OAuth2 authentication, LightRAG issues its own JWT token for subsequent API requests, allowing consistent authorization across all endpoints.

## Troubleshooting

### Common Issues

1. **"Invalid or expired state parameter"**
   - The OAuth2 flow took too long (state expires after 10 minutes)
   - The user navigated away and back during authentication
   - Solution: Try logging in again

2. **"No ID token received from identity provider"**
   - Keycloak client is not configured to return ID tokens
   - Solution: Ensure `openid` scope is requested and client is configured correctly

3. **"ID token audience mismatch"**
   - The `OAUTH2_CLIENT_ID` doesn't match Keycloak's client ID
   - Solution: Verify client ID configuration

4. **"ID token issuer mismatch"**
   - The `OAUTH2_ISSUER` doesn't match Keycloak's issuer URL
   - Solution: Verify issuer URL (check Keycloak's OpenID Configuration endpoint)

5. **Page shows JSON instead of redirecting (WebUI)**
   - Old backend version returning JSON instead of redirect
   - Solution: Update LightRAG and restart the server

### Debug Logging

Enable debug logging to troubleshoot OAuth2 issues:

```env
LOG_LEVEL=DEBUG
```

This will log detailed information about:
- Authorization URL generation
- State storage and retrieval
- Token exchange requests
- ID token validation
