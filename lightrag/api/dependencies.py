"""
FastAPI Dependencies for Multi-Tenant LightRAG Server.

This module provides dependency injection functions for workspace resolution,
authentication, and RAG instance retrieval in the multi-tenant architecture.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from .workspace_manager import WorkspaceManager, sanitize_workspace_id

logger = logging.getLogger("lightrag.api.dependencies")

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)

# Header name for admin on-behalf operations
TARGET_WORKSPACE_HEADER = "X-Target-Workspace"


@dataclass
class UserInfo:
    """Information about the authenticated user."""

    username: str
    """The username of the authenticated user."""

    role: str
    """The role of the user (user, admin, guest)."""

    workspace_id: str
    """The user's workspace ID (derived from username)."""

    metadata: dict
    """Additional metadata from the token."""


def get_workspace_manager(request: Request) -> WorkspaceManager:
    """
    Get the WorkspaceManager instance from app state.

    Args:
        request: The FastAPI request object.

    Returns:
        The WorkspaceManager instance.

    Raises:
        HTTPException: If multi-tenancy is not enabled.
    """
    workspace_manager = getattr(request.app.state, "workspace_manager", None)
    if workspace_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-tenancy is not enabled on this server",
        )
    return workspace_manager


def _extract_token_from_request(request: Request) -> Optional[str]:
    """
    Extract the Bearer token from the Authorization header or HTTP-only cookie.

    Security: This function supports both header-based auth (for REST clients)
    and cookie-based auth (for browser-based WebUI after OAuth2 SSO login).
    The cookie-based approach is more secure for SPAs as it prevents XSS attacks
    from accessing the token.

    Args:
        request: The FastAPI request object.

    Returns:
        The token string if present, None otherwise.
    """
    # First, check Authorization header (for REST API clients)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix

    # Second, check HTTP-only cookie (for browser-based WebUI with SSO)
    # This is the secure way to store tokens for SPAs
    token_cookie = request.cookies.get("lightrag_token")
    if token_cookie:
        return token_cookie

    return None


async def resolve_user_from_request(request: Request) -> UserInfo:
    """
    Resolve the current user from a request without using FastAPI's DI.

    This function manually extracts the token from the Authorization header
    and validates it. Use this when calling from helper functions that are
    not part of FastAPI's dependency injection chain.

    Args:
        request: The FastAPI request object.

    Returns:
        UserInfo containing the authenticated user's details.

    Raises:
        HTTPException: If authentication fails or token is invalid.
    """
    # Check if request was authenticated via API key
    api_key_user = getattr(request.state, "api_key_user", None)
    if api_key_user:
        return UserInfo(
            username=api_key_user["username"],
            role=api_key_user["role"],
            workspace_id=api_key_user["workspace_id"],
            metadata=api_key_user["metadata"],
        )

    token = _extract_token_from_request(request)
    return await _resolve_user(token, request)


async def resolve_workspace_from_request(request: Request) -> str:
    """
    Resolve the workspace ID from a request without using FastAPI's DI.

    This function handles both normal operations (user's own workspace) and
    admin on-behalf operations via the X-Target-Workspace header.

    Use this when calling from helper functions that are not part of
    FastAPI's dependency injection chain.

    IMPORTANT: Service accounts (X-API-Key and Client Credentials) MUST provide
    the X-Target-Workspace header.

    Args:
        request: The FastAPI request object.

    Returns:
        The workspace ID to use for this request.

    Raises:
        HTTPException: If authentication fails, token is invalid,
            a non-admin tries to use on-behalf operations, or
            a service account doesn't provide X-Target-Workspace.
    """
    user = await resolve_user_from_request(request)
    target_workspace = request.headers.get(TARGET_WORKSPACE_HEADER)

    # Check if this is a service account (API key or Client Credentials)
    auth_mode = user.metadata.get("auth_mode", "")
    is_service_account = auth_mode in ("api_key", "client_credentials")

    if target_workspace:
        # On-behalf operation - admin only
        if user.role != "admin":
            logger.warning(
                f"Non-admin user '{user.username}' attempted on-behalf operation "
                f"for workspace '{target_workspace}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can perform operations on behalf of other users",
            )

        sanitized_target = sanitize_workspace_id(target_workspace)
        logger.info(
            f"Admin '{user.username}' operating on behalf of workspace: {sanitized_target}"
        )
        return sanitized_target

    # Service accounts MUST provide X-Target-Workspace
    if is_service_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Target-Workspace header is required when using API key or Client Credentials authentication",
        )

    # Normal operation - use own workspace
    return user.workspace_id


async def _resolve_user(
    token: Optional[str], request: Optional[Request] = None
) -> UserInfo:
    """
    Core user resolution logic shared by DI and non-DI paths.

    Supports hybrid token validation:
    - LightRAG JWT tokens (from /login endpoint)
    - Keycloak access tokens (if OAuth2 is enabled)

    Args:
        token: The JWT token (or None).
        request: Optional request object (for future extensions).

    Returns:
        UserInfo containing the authenticated user's details.

    Raises:
        HTTPException: If authentication fails or token is invalid.
    """
    from .auth import auth_handler, validate_any_token

    # Check if auth is configured
    if not auth_handler.accounts:
        # Auth disabled - return guest user
        return UserInfo(
            username="guest",
            role="guest",
            workspace_id="guest",
            metadata={"auth_mode": "disabled"},
        )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Use hybrid token validation (LightRAG JWT or Keycloak access token)
    try:
        payload = validate_any_token(token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("username", "")
    role = payload.get("role", "user")
    workspace_id = payload.get("workspace_id") or sanitize_workspace_id(username)
    metadata = payload.get("metadata", {})

    return UserInfo(
        username=username,
        role=role,
        workspace_id=workspace_id,
        metadata=metadata,
    )


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserInfo:
    """
    Extract and validate the current user from the JWT token.

    This is a FastAPI dependency - use resolve_user_from_request() when
    calling from non-DI contexts.

    Supports:
    - API key authentication (via request.state.api_key_user)
    - Authorization header (for REST API clients)
    - HTTP-only cookie (for browser-based WebUI with SSO)

    Args:
        request: The FastAPI request object.
        token: The JWT token from the Authorization header (injected by FastAPI).

    Returns:
        UserInfo containing the authenticated user's details.

    Raises:
        HTTPException: If authentication fails or token is invalid.
    """
    # Check if request was authenticated via API key (set by combined_dependency)
    api_key_user = getattr(request.state, "api_key_user", None)
    if api_key_user:
        return UserInfo(
            username=api_key_user["username"],
            role=api_key_user["role"],
            workspace_id=api_key_user["workspace_id"],
            metadata=api_key_user["metadata"],
        )

    # If no token from header, check cookie (for SSO cookie-based auth)
    if not token:
        token = request.cookies.get("lightrag_token")

    return await _resolve_user(token, request)


async def get_current_workspace(
    request: Request,
    user: UserInfo = Depends(get_current_user),
) -> str:
    """
    Resolve the current workspace for the request.

    This supports admin on-behalf operations via the X-Target-Workspace header.
    For regular users, returns their own workspace.
    For admins with the header, returns the target workspace.

    IMPORTANT: Service accounts (X-API-Key and Client Credentials) MUST provide
    the X-Target-Workspace header for any workspace endpoint, as they don't have
    a personal workspace.

    Args:
        request: The FastAPI request object.
        user: The authenticated user info.

    Returns:
        The workspace ID to use for this request.

    Raises:
        HTTPException: If a non-admin tries to use on-behalf operations,
            or if a service account doesn't provide X-Target-Workspace.
    """
    target_workspace = request.headers.get(TARGET_WORKSPACE_HEADER)

    # Check if this is a service account (API key or Client Credentials)
    auth_mode = user.metadata.get("auth_mode", "")
    is_service_account = auth_mode in ("api_key", "client_credentials")

    if target_workspace:
        # On-behalf operation - admin only
        if user.role != "admin":
            logger.warning(
                f"Non-admin user '{user.username}' attempted on-behalf operation "
                f"for workspace '{target_workspace}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can perform operations on behalf of other users",
            )

        sanitized_target = sanitize_workspace_id(target_workspace)
        logger.info(
            f"Admin '{user.username}' operating on behalf of workspace: {sanitized_target}"
        )
        return sanitized_target

    # Service accounts MUST provide X-Target-Workspace for workspace endpoints
    if is_service_account:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Target-Workspace header is required when using API key or Client Credentials authentication",
        )

    # Normal operation - use own workspace
    return user.workspace_id


async def get_rag_instance(
    workspace: str = Depends(get_current_workspace),
    workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
):
    """
    Get the LightRAG instance for the current workspace.

    This is the main dependency for route handlers that need to perform
    RAG operations. It handles workspace resolution and instance caching.

    Args:
        workspace: The resolved workspace ID.
        workspace_manager: The WorkspaceManager instance.

    Returns:
        The LightRAG instance for the workspace.
    """
    return await workspace_manager.get_instance(workspace)


async def require_admin(
    user: UserInfo = Depends(get_current_user),
) -> UserInfo:
    """
    Dependency that requires the user to have admin role.

    Args:
        user: The authenticated user info.

    Returns:
        The user info if they are an admin.

    Raises:
        HTTPException: If the user is not an admin.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# Dependency aliases for cleaner imports
CurrentUser = Depends(get_current_user)
CurrentWorkspace = Depends(get_current_workspace)
RAGInstance = Depends(get_rag_instance)
AdminRequired = Depends(require_admin)
