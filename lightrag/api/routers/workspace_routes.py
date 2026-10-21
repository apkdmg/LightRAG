"""
Self-service workspace deletion.

``DELETE /workspace`` lets a workspace owner permanently delete their own
workspace and everything in it (see ``workspace_deletion``). The workspace is
always the caller's own, taken from their session; the request carries no
workspace ID, so it cannot be pointed at anyone else's.

Only an interactive owner session may call it: not per-user or global API
keys, service accounts, raw Keycloak tokens or admin impersonation tokens. The
sign-in must be recent, and the body must repeat a fixed confirmation phrase.

Registered only when multi-tenancy is enabled. Fork-only and self-contained.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from lightrag.api.dependencies import (
    TARGET_WORKSPACE_HEADER,
    UserInfo,
    get_current_user,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.workspace_deletion import WorkspaceBusyError, purge_app_workspace
from lightrag.api.workspace_registry import local_identity, sso_identity

logger = logging.getLogger("lightrag.api.workspace_routes")

DELETE_CONFIRMATION = "DELETE MY WORKSPACE"

# How long after signing in an owner may delete their workspace; older
# sessions must sign in again first.
RECENT_SIGN_IN_SECONDS = 15 * 60


class DeleteOwnWorkspaceRequest(BaseModel):
    confirm: str


class DeleteOwnWorkspaceResponse(BaseModel):
    status: str
    message: str


class OwnWorkspaceResponse(BaseModel):
    workspace_id: str
    can_delete: bool
    """Whether this session is the owner's and may delete the workspace."""
    sign_in_seconds_left: int
    """How long deletion stays allowed before signing in again; 0 = now."""


def _session_identity(user: UserInfo) -> Optional[str]:
    """The registry identity behind an interactive owner session, or None
    for any other kind of credential."""
    metadata = user.metadata or {}
    if metadata.get("impersonation"):
        return None
    auth_mode = metadata.get("auth_mode")
    if auth_mode == "enabled":
        return local_identity(user.username)
    if auth_mode == "sso" and metadata.get("keycloak_sub"):
        return sso_identity(metadata["keycloak_sub"])
    return None


def _sign_in_seconds_left(user: UserInfo) -> int:
    if not user.session_started_at:
        return 0
    elapsed = time.time() - user.session_started_at
    return max(0, int(RECENT_SIGN_IN_SECONDS - elapsed))


async def _is_owner(request: Request, user: UserInfo, identity: str) -> bool:
    registry = getattr(request.app.state, "workspace_registry", None)
    owner = await registry.get_by_workspace(user.workspace_id) if registry else None
    return owner is not None and owner.identity == identity


def create_workspace_routes(api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(prefix="/workspace", tags=["workspace"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "",
        response_model=OwnWorkspaceResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_own_workspace(
        request: Request, user: UserInfo = Depends(get_current_user)
    ):
        """The caller's workspace and whether they may delete it now."""
        identity = _session_identity(user)
        can_delete = identity is not None and await _is_owner(request, user, identity)
        return OwnWorkspaceResponse(
            workspace_id=user.workspace_id,
            can_delete=can_delete,
            sign_in_seconds_left=_sign_in_seconds_left(user) if can_delete else 0,
        )

    @router.delete(
        "",
        response_model=DeleteOwnWorkspaceResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_own_workspace(
        body: DeleteOwnWorkspaceRequest,
        request: Request,
        response: Response,
        user: UserInfo = Depends(get_current_user),
    ):
        """
        Permanently delete your own workspace and all of its data.

        Removes every document, chunk, entity, relation, vector, graph node,
        cached LLM response, uploaded file, API key and provider credential of
        the workspace, then signs you out. This cannot be undone.

        Requires a WebUI session (password or SSO) started within the last
        15 minutes, and ``{"confirm": "DELETE MY WORKSPACE"}`` as the body.
        """
        if request.headers.get(TARGET_WORKSPACE_HEADER):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{TARGET_WORKSPACE_HEADER} is not accepted here; "
                "a workspace can only be deleted by its owner",
            )
        identity = _session_identity(user)
        if identity is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the workspace owner, signed in to the WebUI, can "
                "delete a workspace",
            )
        if body.confirm != DELETE_CONFIRMATION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Confirm by sending "{DELETE_CONFIRMATION}"',
            )
        if _sign_in_seconds_left(user) <= 0:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Sign in again to confirm deleting your workspace",
            )

        if not await _is_owner(request, user, identity):
            logger.warning(
                f"Refused deletion of workspace '{user.workspace_id}' by "
                f"{identity}: not its registered owner"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the workspace owner can delete it",
            )

        logger.warning(
            f"Owner {identity} ({user.username}) is deleting workspace "
            f"'{user.workspace_id}'"
        )
        try:
            result = await purge_app_workspace(request.app, user.workspace_id)
        except WorkspaceBusyError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        if not result.ok:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Deleting the workspace did not finish; some data may "
                "remain. Try again, or contact an administrator.",
            )

        response.delete_cookie(key="lightrag_token", path="/")
        response.delete_cookie(key="lightrag_user", path="/")
        return DeleteOwnWorkspaceResponse(
            status="success",
            message="Your workspace and all of its data have been deleted",
        )

    return router
