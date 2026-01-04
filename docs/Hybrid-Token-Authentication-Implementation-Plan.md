# Hybrid Token Validation & Streamlined On-Behalf Access Plan

## Overview

This plan implements:

1. **Hybrid token validation** - Support multiple token types simultaneously
2. **Streamlined on-behalf access** - Allow X-API-Key and OAuth2 Client Credentials to perform on-behalf operations for automation

### Authentication Methods Summary

| Method | On-Behalf Support | Use Case |
|--------|-------------------|----------|
| LightRAG JWT | Yes (if admin) | WebUI admin managing users |
| HTTP-only Cookie | Yes (if admin) | WebUI SSO admin |
| Keycloak Access Token (User) | Yes (if in ADMIN_ACCOUNTS) | Mobile admin apps |
| **Keycloak Client Credentials** | **Yes (service account)** | **n8n, backend services (OAuth2 standard)** |
| X-API-Key | Yes (implicit admin) | Simple automation, legacy |

### Two Options for Automation (n8n)

**Option 1: OAuth2 Client Credentials (Recommended)**

```bash
# 1. Get service token from Keycloak
curl -X POST "https://keycloak/realms/REALM/protocol/openid-connect/token" \
  -d "grant_type=client_credentials" \
  -d "client_id=lightrag-service" \
  -d "client_secret=SERVICE_SECRET"

# 2. Use token with on-behalf header
curl -H "Authorization: Bearer <service_token>" \
     -H "X-Target-Workspace: user@example.com" \
     http://lightrag-server/query
```

**Option 2: X-API-Key (Simple)**

```http
X-API-Key: <your-api-key>
X-Target-Workspace: <target-user-workspace>
```

Client Credentials is the OAuth2-standard approach with token rotation and Keycloak audit logs.

## Current Problem

```text
Current Flow:
Request with X-API-Key → combined_dependency() validates key → Returns (no user context)
                                                                    ↓
Request hits get_current_user() → No token → 401 Unauthorized
```

X-API-Key validation doesn't provide user context, so endpoints requiring user info fail.

## Target Architecture

```text
New Flow:
Request with X-API-Key → combined_dependency() validates key
                                    ↓
                         Stores "api_key_user" in request.state
                                    ↓
Request hits get_current_user() → Checks request.state for api_key_user
                                    ↓
                         Returns ServiceAccount UserInfo (admin role)
                                    ↓
get_current_workspace() → Checks X-Target-Workspace → Uses target or default
```

## Implementation Steps

### Step 1: Create Service Account UserInfo for X-API-Key

When X-API-Key is validated, create a "service account" user with admin privileges.

**File**: [utils_api.py](lightrag/api/utils_api.py)

```python
# In combined_dependency(), after API key validation succeeds:
if api_key_configured and api_key_header_value and api_key_header_value == api_key:
    # Create service account user info and store in request state
    request.state.api_key_user = {
        "username": "api_key_service_account",
        "role": "admin",  # API key holder has admin privileges
        "workspace_id": "service_account",
        "metadata": {"auth_mode": "api_key"},
    }
    return  # API key validation successful
```

### Step 2: Update get_current_user to Check API Key User

**File**: [dependencies.py](lightrag/api/dependencies.py)

```python
async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserInfo:
    # Check if request was authenticated via API key
    api_key_user = getattr(request.state, "api_key_user", None)
    if api_key_user:
        return UserInfo(
            username=api_key_user["username"],
            role=api_key_user["role"],
            workspace_id=api_key_user["workspace_id"],
            metadata=api_key_user["metadata"],
        )

    # Otherwise, validate token as usual
    if not token:
        token = request.cookies.get("lightrag_token")

    return await _resolve_user(token)
```

### Step 3: Update get_current_workspace - Require X-Target-Workspace for Service Accounts

**IMPORTANT**: For X-API-Key and Client Credentials, there's no user context to derive a default workspace.
The `X-Target-Workspace` header is **MANDATORY** for these auth methods when accessing workspace endpoints.

**File**: [dependencies.py](lightrag/api/dependencies.py)

```python
async def get_current_workspace(request: Request, user: UserInfo) -> str:
    target_workspace = request.headers.get(TARGET_WORKSPACE_HEADER)

    # Check if this is a service account (API key or Client Credentials)
    auth_mode = user.metadata.get("auth_mode", "")
    is_service_account = auth_mode in ("api_key", "client_credentials")

    if target_workspace:
        # On-behalf operation - admin or service account only
        if user.role != "admin":
            raise HTTPException(403, "Only admins can perform on-behalf operations")
        return sanitize_workspace_id(target_workspace)

    # Service accounts MUST provide X-Target-Workspace for workspace endpoints
    if is_service_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Target-Workspace header is required when using API key or Client Credentials authentication",
        )

    # Regular user - use own workspace
    return user.workspace_id
```

**Key Point**: Service accounts (X-API-Key and Client Credentials) don't have a personal workspace, so `X-Target-Workspace` is mandatory for any endpoint that works with workspaces.

### Step 4: Add Keycloak Access Token Validation (oauth2.py)

**File**: [oauth2.py](lightrag/api/oauth2.py)

```python
def validate_access_token(self, access_token: str) -> Dict[str, Any]:
    """
    Validate Keycloak access token using JWKS.

    Supports both:
    - User access tokens (from Authorization Code flow)
    - Service account tokens (from Client Credentials flow)

    Unlike ID tokens, access tokens may not have audience claim for the client.
    We validate: signature, issuer, expiration.
    """
    try:
        signing_key = self.jwks_client.get_signing_key_from_jwt(access_token)
        payload = jwt.decode(
            access_token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=self.config.issuer,
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": False,  # Access tokens may not have client as audience
                "verify_iss": True,
            },
        )
        return payload
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid access token: {e}")

def is_service_account_token(self, payload: Dict[str, Any]) -> bool:
    """
    Check if the token is from Client Credentials grant (service account).

    Client Credentials tokens have:
    - No 'preferred_username' (or it's the client_id)
    - 'clientId' claim present
    - 'azp' (authorized party) equals the client_id
    """
    # Check for service account indicators
    client_id = payload.get("clientId") or payload.get("azp")
    preferred_username = payload.get("preferred_username", "")

    # Service account usernames typically follow pattern: service-account-<client_id>
    if preferred_username.startswith("service-account-"):
        return True

    # Or no preferred_username but has clientId
    if not preferred_username and client_id:
        return True

    return False
```

### Step 5: Create Hybrid Token Validator (auth.py)

**File**: [auth.py](lightrag/api/auth.py)

```python
from .oauth2 import get_keycloak_client
from .config import global_args

def _is_admin_user(username: str) -> bool:
    """Check if username is in ADMIN_ACCOUNTS."""
    admin_accounts = global_args.admin_accounts
    if not admin_accounts:
        return False
    admins = [a.strip().lower() for a in admin_accounts.split(",")]
    return username.lower() in admins

async def validate_any_token(token: str) -> dict:
    """
    Validate token as LightRAG JWT or Keycloak access token.

    Supports:
    - LightRAG JWT (from /login endpoint)
    - Keycloak user access token (from Authorization Code flow)
    - Keycloak service account token (from Client Credentials flow)

    Returns standardized user info dict.
    """
    # 1. Try LightRAG JWT first (fast, local validation)
    try:
        return auth_handler.validate_token(token)
    except HTTPException:
        pass

    # 2. Try Keycloak access token if OAuth2 is enabled
    keycloak_client = get_keycloak_client()
    if keycloak_client:
        try:
            payload = keycloak_client.validate_access_token(token)

            # Check if this is a service account (Client Credentials)
            if keycloak_client.is_service_account_token(payload):
                # Service accounts get admin role for on-behalf operations
                client_id = payload.get("clientId") or payload.get("azp")
                return {
                    "username": f"service-account-{client_id}",
                    "role": "admin",  # Service accounts have admin privileges
                    "workspace_id": "service_account",
                    "metadata": {
                        "auth_mode": "client_credentials",
                        "client_id": client_id,
                        "scope": payload.get("scope", ""),
                    },
                }

            # Regular user access token
            username = payload.get("preferred_username") or payload.get("sub")
            role = "admin" if _is_admin_user(username) else "user"

            return {
                "username": username,
                "role": role,
                "workspace_id": sanitize_workspace_id(username),
                "metadata": {
                    "auth_mode": "keycloak_direct",
                    "email": payload.get("email"),
                },
            }
        except HTTPException:
            pass

    # All validation failed
    raise HTTPException(status_code=401, detail="Invalid token")
```

**Key Point**: Client Credentials tokens (service accounts) automatically get `role: "admin"`, enabling them to use `X-Target-Workspace` for on-behalf operations without being in ADMIN_ACCOUNTS.

### Step 6: Update _resolve_user to Use Hybrid Validation

**File**: [dependencies.py](lightrag/api/dependencies.py)

```python
async def _resolve_user(token: Optional[str]) -> UserInfo:
    from .auth import auth_handler, validate_any_token

    if not auth_handler.accounts:
        return UserInfo(username="guest", role="guest", ...)

    if not token:
        raise HTTPException(401, "Not authenticated")

    # Use hybrid validation
    try:
        payload = await validate_any_token(token)
    except HTTPException:
        raise

    return UserInfo(
        username=payload["username"],
        role=payload["role"],
        workspace_id=payload["workspace_id"],
        metadata=payload.get("metadata", {}),
    )
```

### Step 7: Update Documentation

**File**: [docs/OAuth2-SSO-Authentication.md](docs/OAuth2-SSO-Authentication.md)

Add section for automation (n8n) usage:

```markdown
## Automation Access (n8n, Backend Services)

For automation tools like n8n that need to access user workspaces:

### Using X-API-Key with On-Behalf Access

\`\`\`bash
curl -X POST "http://lightrag-server/query" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "X-Target-Workspace: user@example.com" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
\`\`\`

The API key holder is trusted as a service account with admin privileges,
allowing access to any user's workspace via X-Target-Workspace header.

### Using Direct Keycloak Tokens

Mobile apps and backend services can also use Keycloak access tokens directly:

\`\`\`bash
# User authenticates with Keycloak and gets access token
curl -X POST "http://lightrag-server/query" \
  -H "Authorization: Bearer <keycloak_access_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
\`\`\`
```

## Critical Files to Modify

1. [utils_api.py](lightrag/api/utils_api.py) - Store API key user in request.state
2. [dependencies.py](lightrag/api/dependencies.py) - Check request.state for API key user
3. [oauth2.py](lightrag/api/oauth2.py) - Add `validate_access_token()` method
4. [auth.py](lightrag/api/auth.py) - Add `validate_any_token()` function
5. [docs/OAuth2-SSO-Authentication.md](docs/OAuth2-SSO-Authentication.md) - Documentation

## Usage Examples

### n8n Workflow Example (Client Credentials - Recommended)

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

### n8n Workflow Example (X-API-Key - Simple)

```text
HTTP Request Node:
  Method: POST
  URL: http://lightrag-server/documents/upload
  Headers:
    X-API-Key: ${LIGHTRAG_API_KEY}
    X-Target-Workspace: ${user_email}
  Body: (file upload)
```

### Mobile App Example

```text
1. User logs in via Keycloak (mobile SDK)
2. Get access_token from Keycloak
3. Call LightRAG API with: Authorization: Bearer <access_token>
4. LightRAG validates token via JWKS
5. User accesses their workspace
```

### Backend Service Example (Client Credentials)

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

## Security Considerations

1. **X-API-Key = Full Admin**: Treat API key as root credential
2. **Client Credentials = Service Admin**: Service accounts get admin role automatically
3. **X-Target-Workspace Mandatory**: Service accounts (X-API-Key and Client Credentials) MUST provide X-Target-Workspace header for workspace endpoints - they have no default workspace
4. **Token Precedence**: Try LightRAG JWT first (faster, no network)
5. **JWKS Caching**: Already implemented in PyJWKClient (1 hour cache)
6. **Issuer Validation**: Only accept tokens from configured OAUTH2_ISSUER
7. **Admin from ADMIN_ACCOUNTS**: Regular Keycloak users get admin role only if in list
8. **Keycloak Client Setup**: Create a "confidential" client for service accounts with Client Credentials enabled

## Keycloak Setup for Client Credentials

In Keycloak admin console:

1. Create a new client (e.g., `lightrag-n8n-service`)
2. Set **Client authentication**: ON (confidential client)
3. Enable **Service accounts roles**: ON
4. Set **Valid redirect URIs**: (not needed for client credentials)
5. Copy the **Client secret** from Credentials tab

## Testing Checklist

- [ ] LightRAG JWT login still works
- [ ] Cookie-based auth for WebUI SSO still works
- [ ] Keycloak user access token accepted for mobile apps
- [ ] Keycloak admin user can do on-behalf operations
- [ ] **X-API-Key without X-Target-Workspace returns 400 error for workspace endpoints**
- [ ] **X-API-Key + X-Target-Workspace works for on-behalf operations**
- [ ] **Client Credentials without X-Target-Workspace returns 400 error for workspace endpoints**
- [ ] **Client Credentials + X-Target-Workspace works for on-behalf operations**
- [ ] Invalid tokens properly rejected
- [ ] Expired tokens properly rejected
