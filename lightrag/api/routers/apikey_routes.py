"""
Per-User API Key Management Routes for LightRAG.

This module provides endpoints for users to generate and manage their own API keys.
These API keys are tied to the user's workspace and can be used with OpenAI-compatible
clients without needing X-Target-Workspace header.

API Key Format: sk-lightrag-{workspace_hash}-{random_32_chars}
The workspace is embedded in the key, so the server can resolve it automatically.
"""

import hashlib
import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from lightrag.api.dependencies import get_current_user, UserInfo
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.config import global_args

logger = logging.getLogger("lightrag.api.apikey")


# Pydantic models
class CreateApiKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str  # User-friendly name for the key (e.g., "My Continue.dev key")
    expires_in_days: Optional[int] = None  # Optional expiration in days


class ApiKeyResponse(BaseModel):
    """Response when creating a new API key."""

    api_key: str  # The full API key (only returned once)
    id: str  # Key ID for reference
    name: str  # User-friendly name
    created_at: str  # ISO timestamp
    expires_at: Optional[str] = None  # ISO timestamp if set


class ApiKeyMetadata(BaseModel):
    """Metadata about an API key (without the actual key)."""

    id: str
    name: str
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    key_preview: str  # Last 4 characters of the key


class ApiKeyListResponse(BaseModel):
    """Response listing all API keys."""

    keys: List[ApiKeyMetadata]


# Global registry for workspace hash -> workspace_id mapping
# This is populated at startup and when new keys are created
_workspace_hash_registry: Dict[str, str] = {}


def _get_workspace_dir(workspace_id: str) -> Path:
    """Get the workspace directory path."""
    working_dir = Path(global_args.working_dir)
    return working_dir / workspace_id


def _get_api_keys_file(workspace_id: str) -> Path:
    """Get the path to the API keys file for a workspace."""
    workspace_dir = _get_workspace_dir(workspace_id)
    return workspace_dir / ".api_keys.json"


def _hash_workspace_id(workspace_id: str) -> str:
    """
    Create a short hash of the workspace ID for embedding in the API key.

    Returns first 8 characters of SHA256 hash.
    """
    return hashlib.sha256(workspace_id.encode()).hexdigest()[:8]


def _hash_api_key(api_key: str) -> str:
    """Hash an API key for storage (never store plaintext)."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def _load_api_keys(workspace_id: str) -> Dict:
    """Load API keys data for a workspace."""
    keys_file = _get_api_keys_file(workspace_id)
    if keys_file.exists():
        try:
            with open(keys_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading API keys file: {e}")
            return {"keys": []}
    return {"keys": []}


def _save_api_keys(workspace_id: str, data: Dict) -> None:
    """Save API keys data for a workspace."""
    keys_file = _get_api_keys_file(workspace_id)
    # Ensure workspace directory exists
    keys_file.parent.mkdir(parents=True, exist_ok=True)
    with open(keys_file, "w") as f:
        json.dump(data, f, indent=2)


def generate_api_key(workspace_id: str) -> str:
    """
    Generate a new API key with embedded workspace.

    Format: sk-lightrag-{workspace_hash}-{random_32_chars}
    """
    workspace_hash = _hash_workspace_id(workspace_id)
    random_part = secrets.token_urlsafe(32)
    return f"sk-lightrag-{workspace_hash}-{random_part}"


def register_workspace_hash(workspace_id: str) -> None:
    """Register a workspace hash in the global registry."""
    workspace_hash = _hash_workspace_id(workspace_id)
    _workspace_hash_registry[workspace_hash] = workspace_id
    logger.debug(f"Registered workspace hash: {workspace_hash} -> {workspace_id}")


def lookup_workspace_by_hash(workspace_hash: str) -> Optional[str]:
    """
    Look up a workspace ID by its hash.

    First checks the in-memory registry, then scans workspace directories.
    """
    # Check registry first
    if workspace_hash in _workspace_hash_registry:
        return _workspace_hash_registry[workspace_hash]

    # Scan workspace directories if not in registry
    working_dir = Path(global_args.working_dir)
    if working_dir.exists():
        for workspace_dir in working_dir.iterdir():
            if workspace_dir.is_dir():
                workspace_id = workspace_dir.name
                if _hash_workspace_id(workspace_id) == workspace_hash:
                    # Add to registry for faster future lookups
                    _workspace_hash_registry[workspace_hash] = workspace_id
                    return workspace_id

    return None


def verify_api_key_hash(workspace_id: str, api_key: str) -> Optional[Dict]:
    """
    Verify an API key hash exists in workspace and return key metadata.

    Returns key data if valid, None otherwise.
    """
    data = _load_api_keys(workspace_id)
    key_hash = _hash_api_key(api_key)

    for key_data in data.get("keys", []):
        if key_data.get("key_hash") == key_hash:
            # Check expiration
            expires_at = key_data.get("expires_at")
            if expires_at:
                if datetime.fromisoformat(expires_at) < datetime.utcnow():
                    logger.warning(f"API key expired for workspace {workspace_id}")
                    return None
            return key_data

    return None


def update_key_last_used(workspace_id: str, api_key: str) -> None:
    """Update the last_used_at timestamp for an API key."""
    data = _load_api_keys(workspace_id)
    key_hash = _hash_api_key(api_key)

    for key_data in data.get("keys", []):
        if key_data.get("key_hash") == key_hash:
            key_data["last_used_at"] = datetime.utcnow().isoformat()
            _save_api_keys(workspace_id, data)
            return


def validate_user_api_key(api_key: str) -> Optional[Dict]:
    """
    Validate a per-user API key and return user info.

    Key format: sk-lightrag-{workspace_hash}-{random}

    Returns:
        dict with username, role, workspace_id, metadata if valid
        None if invalid
    """
    if not api_key.startswith("sk-lightrag-"):
        return None

    parts = api_key.split("-")
    if len(parts) < 4:
        return None

    workspace_hash = parts[2]
    workspace_id = lookup_workspace_by_hash(workspace_hash)

    if not workspace_id:
        logger.warning(f"Unknown workspace hash in API key: {workspace_hash}")
        return None

    # Verify key hash exists in workspace
    key_data = verify_api_key_hash(workspace_id, api_key)
    if not key_data:
        logger.warning(f"Invalid API key for workspace: {workspace_id}")
        return None

    # Update last_used_at (fire and forget, don't block on this)
    try:
        update_key_last_used(workspace_id, api_key)
    except Exception as e:
        logger.debug(f"Failed to update last_used_at: {e}")

    return {
        "username": f"apikey-{workspace_id}",
        "role": "user",  # Per-user API keys get user role, not admin
        "workspace_id": workspace_id,
        "metadata": {
            "auth_mode": "user_api_key",
            "key_id": key_data.get("id"),
            "key_name": key_data.get("name"),
        },
    }


def create_apikey_routes(api_key: Optional[str] = None) -> APIRouter:
    """
    Create API key management routes.

    Args:
        api_key: Optional shared API key for authentication

    Returns:
        FastAPI router with API key management endpoints
    """
    router = APIRouter(prefix="/api-keys", tags=["api-keys"])

    # Create combined auth dependency
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "",
        response_model=ApiKeyResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def create_api_key(
        request: CreateApiKeyRequest,
        user: UserInfo = Depends(get_current_user),
    ):
        """
        Create a new API key for the current user's workspace.

        The API key is only returned once - it cannot be retrieved later.
        Store it securely.
        """
        workspace_id = user.workspace_id

        # Check if user is a service account (they shouldn't create keys)
        auth_mode = user.metadata.get("auth_mode", "")
        if auth_mode in ("api_key", "client_credentials"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Service accounts cannot create per-user API keys",
            )

        # Generate the key
        new_key = generate_api_key(workspace_id)
        key_id = f"key_{secrets.token_hex(8)}"
        created_at = datetime.utcnow().isoformat()

        # Calculate expiration if specified
        expires_at = None
        if request.expires_in_days:
            from datetime import timedelta

            expires_at = (
                datetime.utcnow() + timedelta(days=request.expires_in_days)
            ).isoformat()

        # Load existing keys
        data = _load_api_keys(workspace_id)

        # Add new key (store only the hash)
        key_entry = {
            "id": key_id,
            "key_hash": _hash_api_key(new_key),
            "key_preview": new_key[-4:],  # Last 4 chars for identification
            "name": request.name,
            "created_at": created_at,
            "last_used_at": None,
            "expires_at": expires_at,
        }
        data["keys"].append(key_entry)

        # Save
        _save_api_keys(workspace_id, data)

        # Register workspace hash for fast lookups
        register_workspace_hash(workspace_id)

        logger.info(f"Created API key '{request.name}' for workspace {workspace_id}")

        return ApiKeyResponse(
            api_key=new_key,  # Only time the full key is returned
            id=key_id,
            name=request.name,
            created_at=created_at,
            expires_at=expires_at,
        )

    @router.get(
        "",
        response_model=ApiKeyListResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_api_keys(
        user: UserInfo = Depends(get_current_user),
    ):
        """
        List all API keys for the current user's workspace.

        Note: The actual keys are not returned, only metadata.
        """
        workspace_id = user.workspace_id
        data = _load_api_keys(workspace_id)

        keys = []
        for key_data in data.get("keys", []):
            keys.append(
                ApiKeyMetadata(
                    id=key_data["id"],
                    name=key_data["name"],
                    created_at=key_data["created_at"],
                    last_used_at=key_data.get("last_used_at"),
                    expires_at=key_data.get("expires_at"),
                    key_preview=f"...{key_data.get('key_preview', '????')}",
                )
            )

        return ApiKeyListResponse(keys=keys)

    @router.delete(
        "/{key_id}",
        dependencies=[Depends(combined_auth)],
    )
    async def revoke_api_key(
        key_id: str,
        user: UserInfo = Depends(get_current_user),
    ):
        """
        Revoke (delete) an API key.

        Once revoked, the key can no longer be used for authentication.
        """
        workspace_id = user.workspace_id
        data = _load_api_keys(workspace_id)

        # Find and remove the key
        original_count = len(data.get("keys", []))
        data["keys"] = [k for k in data.get("keys", []) if k.get("id") != key_id]

        if len(data["keys"]) == original_count:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{key_id}' not found",
            )

        _save_api_keys(workspace_id, data)
        logger.info(f"Revoked API key '{key_id}' for workspace {workspace_id}")

        return {"message": f"API key '{key_id}' has been revoked"}

    return router
