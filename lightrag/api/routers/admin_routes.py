"""
Admin routes for workspace management in multi-tenant LightRAG.

This module provides administrative APIs for managing workspaces,
including creation, deletion, listing, and impersonation.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from lightrag.api.dependencies import (
    get_workspace_manager,
    require_admin,
    UserInfo,
)
from lightrag.api.workspace_manager import WorkspaceManager, sanitize_workspace_id
from lightrag.api.auth import auth_handler

logger = logging.getLogger("lightrag.api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# Request/Response Models


class CreateWorkspaceRequest(BaseModel):
    """Request to create a new workspace."""

    username: str = Field(
        ...,
        min_length=1,
        description="Username for the workspace owner",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata for the workspace",
    )


class WorkspaceResponse(BaseModel):
    """Response containing workspace information."""

    workspace_id: str
    owner_username: str
    created_at: float
    last_accessed_at: float
    access_count: int
    is_active: bool


class WorkspaceListResponse(BaseModel):
    """Response containing a list of workspaces."""

    workspaces: List[WorkspaceResponse]
    total: int
    page: int
    page_size: int


class ImpersonationTokenResponse(BaseModel):
    """Response containing an impersonation token."""

    access_token: str
    token_type: str = "bearer"
    workspace_id: str
    target_username: str
    expires_in_hours: int


class WorkspaceStatsResponse(BaseModel):
    """Response containing workspace statistics."""

    workspace_id: str
    owner_username: str
    is_active: bool
    created_at: float
    last_accessed_at: float
    access_count: int


class DeleteWorkspaceResponse(BaseModel):
    """Response for workspace deletion."""

    status: str
    message: str
    workspace_id: str


# Routes


def create_admin_routes():
    """Create admin routes for workspace management."""

    @router.post(
        "/workspaces",
        response_model=WorkspaceResponse,
        dependencies=[Depends(require_admin)],
    )
    async def create_workspace(
        request: CreateWorkspaceRequest,
        workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
    ):
        """
        Create a new workspace for a user.

        This endpoint allows admins to pre-create workspaces for users
        before they log in for the first time.

        Args:
            request: The workspace creation request containing username.

        Returns:
            WorkspaceResponse: Information about the created workspace.
        """
        try:
            workspace_id = sanitize_workspace_id(request.username)

            # Check if workspace already exists
            existing = workspace_manager.get_workspace_info(workspace_id)
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Workspace already exists for user: {request.username}",
                )

            workspace_info = await workspace_manager.create_workspace(
                workspace_id=workspace_id,
                owner_username=request.username,
            )

            logger.info(f"Admin created workspace for user: {request.username}")

            return WorkspaceResponse(
                workspace_id=workspace_info.workspace_id,
                owner_username=workspace_info.owner_username,
                created_at=workspace_info.created_at,
                last_accessed_at=workspace_info.last_accessed_at,
                access_count=workspace_info.access_count,
                is_active=workspace_info.is_active,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating workspace: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create workspace: {str(e)}",
            )

    @router.delete(
        "/workspaces/{workspace_id}",
        response_model=DeleteWorkspaceResponse,
        dependencies=[Depends(require_admin)],
    )
    async def delete_workspace(
        workspace_id: str,
        workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
    ):
        """
        Delete a workspace and all its data.

        WARNING: This permanently deletes all data in the workspace.

        Args:
            workspace_id: The workspace identifier to delete.

        Returns:
            DeleteWorkspaceResponse: Status of the deletion.
        """
        try:
            # Sanitize the workspace_id
            sanitized_id = sanitize_workspace_id(workspace_id)

            deleted = await workspace_manager.delete_workspace(sanitized_id)

            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Workspace not found: {workspace_id}",
                )

            logger.info(f"Admin deleted workspace: {sanitized_id}")

            return DeleteWorkspaceResponse(
                status="success",
                message=f"Workspace '{sanitized_id}' deleted successfully",
                workspace_id=sanitized_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting workspace: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete workspace: {str(e)}",
            )

    @router.get(
        "/workspaces",
        response_model=WorkspaceListResponse,
        dependencies=[Depends(require_admin)],
    )
    async def list_workspaces(
        page: int = 1,
        page_size: int = 50,
        workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
    ):
        """
        List all workspaces with pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.

        Returns:
            WorkspaceListResponse: Paginated list of workspaces.
        """
        try:
            workspaces, total = workspace_manager.list_workspaces(
                page=page,
                page_size=page_size,
            )

            return WorkspaceListResponse(
                workspaces=[
                    WorkspaceResponse(
                        workspace_id=w.workspace_id,
                        owner_username=w.owner_username,
                        created_at=w.created_at,
                        last_accessed_at=w.last_accessed_at,
                        access_count=w.access_count,
                        is_active=w.is_active,
                    )
                    for w in workspaces
                ],
                total=total,
                page=page,
                page_size=page_size,
            )
        except Exception as e:
            logger.error(f"Error listing workspaces: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list workspaces: {str(e)}",
            )

    @router.get(
        "/workspaces/{workspace_id}",
        response_model=WorkspaceStatsResponse,
        dependencies=[Depends(require_admin)],
    )
    async def get_workspace_stats(
        workspace_id: str,
        workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
    ):
        """
        Get statistics for a specific workspace.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            WorkspaceStatsResponse: Statistics about the workspace.
        """
        try:
            sanitized_id = sanitize_workspace_id(workspace_id)
            workspace_info = workspace_manager.get_workspace_info(sanitized_id)

            if not workspace_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Workspace not found: {workspace_id}",
                )

            return WorkspaceStatsResponse(
                workspace_id=workspace_info.workspace_id,
                owner_username=workspace_info.owner_username,
                is_active=workspace_info.is_active,
                created_at=workspace_info.created_at,
                last_accessed_at=workspace_info.last_accessed_at,
                access_count=workspace_info.access_count,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting workspace stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get workspace stats: {str(e)}",
            )

    @router.post(
        "/workspaces/{workspace_id}/impersonate",
        response_model=ImpersonationTokenResponse,
        dependencies=[Depends(require_admin)],
    )
    async def impersonate_user(
        workspace_id: str,
        admin_user: UserInfo = Depends(require_admin),
        workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
    ):
        """
        Generate an impersonation token to act as another user.

        This creates a token that allows the admin to perform operations
        on behalf of the target user. The token includes metadata indicating
        it's an impersonation token.

        Args:
            workspace_id: The workspace/username to impersonate.
            admin_user: The authenticated admin user.

        Returns:
            ImpersonationTokenResponse: Token for impersonation.
        """
        try:
            sanitized_id = sanitize_workspace_id(workspace_id)

            # Check if workspace exists
            workspace_info = workspace_manager.get_workspace_info(sanitized_id)
            if not workspace_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Workspace not found: {workspace_id}",
                )

            # Create impersonation token
            expire_hours = 1  # Short-lived impersonation tokens
            token = auth_handler.create_token(
                username=workspace_info.owner_username,
                role="user",  # Impersonated user has regular user role
                custom_expire_hours=expire_hours,
                workspace_id=sanitized_id,
                metadata={
                    "impersonated_by": admin_user.username,
                    "impersonation": True,
                },
            )

            logger.info(
                f"Admin '{admin_user.username}' generated impersonation token "
                f"for workspace: {sanitized_id}"
            )

            return ImpersonationTokenResponse(
                access_token=token,
                token_type="bearer",
                workspace_id=sanitized_id,
                target_username=workspace_info.owner_username,
                expires_in_hours=expire_hours,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating impersonation token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate impersonation token: {str(e)}",
            )

    @router.get(
        "/status",
        dependencies=[Depends(require_admin)],
    )
    async def get_admin_status(
        workspace_manager: WorkspaceManager = Depends(get_workspace_manager),
    ):
        """
        Get admin dashboard status information.

        Returns:
            Dict containing system status and workspace statistics.
        """
        try:
            _, total_workspaces = workspace_manager.list_workspaces(page=1, page_size=1)

            return {
                "status": "healthy",
                "multi_tenancy_enabled": True,
                "active_instances": workspace_manager.active_instance_count,
                "max_instances": workspace_manager.max_instances,
                "total_workspaces": total_workspaces,
            }
        except Exception as e:
            logger.error(f"Error getting admin status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get admin status: {str(e)}",
            )

    return router
