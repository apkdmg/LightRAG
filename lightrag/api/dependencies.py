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


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
) -> UserInfo:
    """
    Extract and validate the current user from the JWT token.

    Args:
        request: The FastAPI request object.
        token: The JWT token from the Authorization header.

    Returns:
        UserInfo containing the authenticated user's details.

    Raises:
        HTTPException: If authentication fails or token is invalid.
    """
    from .auth import auth_handler

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

    # Validate token
    try:
        payload = auth_handler.validate_token(token)
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


async def get_current_workspace(
    request: Request,
    user: UserInfo = Depends(get_current_user),
) -> str:
    """
    Resolve the current workspace for the request.

    This supports admin on-behalf operations via the X-Target-Workspace header.
    For regular users, returns their own workspace.
    For admins with the header, returns the target workspace.

    Args:
        request: The FastAPI request object.
        user: The authenticated user info.

    Returns:
        The workspace ID to use for this request.

    Raises:
        HTTPException: If a non-admin tries to use on-behalf operations.
    """
    target_workspace = request.headers.get(TARGET_WORKSPACE_HEADER)

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
