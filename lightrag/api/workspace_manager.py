"""
Workspace Manager for Multi-Tenant LightRAG Server.

Provides workspace management with LRU caching of LightRAG instances, enabling a
single server to serve multiple users with isolated workspaces.

Instances are produced by an injected ``instance_factory`` callable, so this
module stays fully decoupled from how a LightRAG instance is constructed — the
server owns that logic and passes a factory in.
"""

import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from lightrag import LightRAG

logger = logging.getLogger("lightrag.api.workspace_manager")


@dataclass
class WorkspaceInfo:
    """Information about a workspace."""

    workspace_id: str
    """Unique identifier for the workspace (derived from username)."""

    owner_username: str
    """Username of the workspace owner."""

    created_at: float
    """Unix timestamp when the workspace was created."""

    last_accessed_at: float
    """Unix timestamp when the workspace was last accessed."""

    access_count: int
    """Number of times the workspace has been accessed."""

    is_active: bool
    """Whether the workspace is currently loaded in memory."""


def sanitize_workspace_id(username: str) -> str:
    """
    Convert a username to a valid workspace ID.

    Replaces invalid characters with underscores and ensures the ID
    is safe for use in file paths and database identifiers.

    Args:
        username: The username to sanitize.

    Returns:
        A sanitized workspace ID string.
    """
    # Replace @ and . with underscores, remove other special chars
    workspace_id = re.sub(r"[^a-zA-Z0-9_-]", "_", username.lower())
    # Remove consecutive underscores
    workspace_id = re.sub(r"_+", "_", workspace_id)
    # Remove leading/trailing underscores
    workspace_id = workspace_id.strip("_")
    # Ensure non-empty
    if not workspace_id:
        workspace_id = "default"
    return workspace_id


class WorkspaceManager:
    """
    Manages LightRAG instances for multiple workspaces with LRU caching.

    Maintains a pool of active LightRAG instances, creating new ones on demand
    via the injected ``instance_factory`` and evicting the least-recently-used
    instance when the pool reaches capacity. Native multimodal processing is a
    property of each LightRAG instance (LightRAG 1.5.0), so the manager itself
    is engine-agnostic.

    Attributes:
        max_instances: Maximum number of LightRAG instances to keep in memory.
        ttl_minutes: Time-to-live for inactive instances (not currently enforced).
    """

    def __init__(
        self,
        instance_factory: Callable[[str], LightRAG],
        max_instances: int = 100,
        ttl_minutes: int = 60,
    ):
        """
        Initialize the WorkspaceManager.

        Args:
            instance_factory: Callable that builds an un-initialized LightRAG
                instance for a given workspace id. The manager calls
                ``initialize_storages()`` on the result and owns its lifecycle
                (eviction / shutdown call ``finalize_storages()``).
            max_instances: Maximum number of instances to keep in memory.
            ttl_minutes: Time-to-live for inactive instances.
        """
        self._instance_factory = instance_factory
        self._max_instances = max_instances
        self._ttl_minutes = ttl_minutes

        # LRU cache: workspace_id -> (LightRAG, created_at, last_accessed, access_count)
        self._instances: OrderedDict[str, Tuple[LightRAG, float, float, int]] = (
            OrderedDict()
        )
        self._lock = asyncio.Lock()

        # Track workspace metadata (persists even after eviction)
        self._workspace_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"WorkspaceManager initialized with max_instances={max_instances}, "
            f"ttl_minutes={ttl_minutes}"
        )

    @property
    def active_instance_count(self) -> int:
        """Return the number of active LightRAG instances in memory."""
        return len(self._instances)

    @property
    def max_instances(self) -> int:
        """Return the maximum number of instances allowed."""
        return self._max_instances

    async def _create_instance(self, workspace_id: str) -> LightRAG:
        """Build and initialize a new LightRAG instance for a workspace."""
        rag = self._instance_factory(workspace_id)
        await rag.initialize_storages()
        logger.info(
            f"LightRAG instance created and initialized for workspace: {workspace_id}"
        )
        return rag

    async def get_instance(self, workspace_id: str) -> LightRAG:
        """
        Get or create a LightRAG instance for the specified workspace.

        If an instance exists in the cache it is moved to the most-recently-used
        position; otherwise a new instance is created and LRU eviction runs if
        the pool is at capacity.

        Args:
            workspace_id: The workspace identifier (typically a sanitized username).

        Returns:
            The LightRAG instance for the workspace.
        """
        async with self._lock:
            if workspace_id in self._instances:
                # Move to end (most recently used) and update access time
                rag, created_at, _, count = self._instances.pop(workspace_id)
                self._instances[workspace_id] = (
                    rag,
                    created_at,
                    time.time(),
                    count + 1,
                )
                logger.debug(f"Cache hit for workspace: {workspace_id}")
                return rag

            logger.info(f"Creating new LightRAG instance for workspace: {workspace_id}")
            rag = await self._create_instance(workspace_id)

            # Evict LRU if over capacity
            while len(self._instances) >= self._max_instances:
                await self._evict_lru()

            now = time.time()
            self._instances[workspace_id] = (rag, now, now, 1)

            # Update metadata
            if workspace_id not in self._workspace_metadata:
                self._workspace_metadata[workspace_id] = {
                    "created_at": now,
                    "owner_username": workspace_id,  # may be set properly by caller
                }
            self._workspace_metadata[workspace_id]["last_accessed_at"] = now

            return rag

    async def get_lightrag_instance(self, workspace_id: str) -> LightRAG:
        """
        Get the LightRAG instance for a workspace.

        Retained as a distinct name for callers (e.g. graph routes) that
        previously needed to distinguish RAG engines; with native multimodal it
        is simply equivalent to :meth:`get_instance`.
        """
        return await self.get_instance(workspace_id)

    async def _evict_lru(self) -> None:
        """
        Evict the least recently used instance from the cache.

        This properly finalizes the storage before removing the instance.
        """
        if not self._instances:
            return

        # Get oldest (first item in OrderedDict)
        workspace_id, (rag, created_at, last_accessed, count) = next(
            iter(self._instances.items())
        )

        logger.info(
            f"Evicting LRU workspace: {workspace_id} "
            f"(created: {created_at:.0f}, last_access: {last_accessed:.0f}, "
            f"access_count: {count})"
        )

        # Finalize storages before eviction
        try:
            await rag.finalize_storages()
        except Exception as e:
            logger.error(f"Error finalizing storage for workspace {workspace_id}: {e}")

        del self._instances[workspace_id]

    async def create_workspace(
        self, workspace_id: str, owner_username: str
    ) -> WorkspaceInfo:
        """
        Create a new workspace and pre-initialize its instance (admin API).

        Args:
            workspace_id: The workspace identifier.
            owner_username: The username of the workspace owner.

        Returns:
            Information about the created workspace.
        """
        now = time.time()

        # Store metadata
        self._workspace_metadata[workspace_id] = {
            "created_at": now,
            "last_accessed_at": now,
            "owner_username": owner_username,
        }

        # Pre-initialize the instance
        await self.get_instance(workspace_id)

        return WorkspaceInfo(
            workspace_id=workspace_id,
            owner_username=owner_username,
            created_at=now,
            last_accessed_at=now,
            access_count=1,
            is_active=True,
        )

    async def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace, finalizing and removing its instance (admin API).

        Args:
            workspace_id: The workspace identifier to delete.

        Returns:
            True if the workspace was found and deleted, False otherwise.
        """
        async with self._lock:
            # Remove from active instances if present
            if workspace_id in self._instances:
                rag, _, _, _ = self._instances.pop(workspace_id)
                try:
                    await rag.finalize_storages()
                except Exception as e:
                    logger.error(
                        f"Error finalizing storage during deletion for {workspace_id}: {e}"
                    )

            # Remove metadata
            if workspace_id in self._workspace_metadata:
                del self._workspace_metadata[workspace_id]
                logger.info(f"Deleted workspace: {workspace_id}")
                return True

            return False

    def list_workspaces(
        self, page: int = 1, page_size: int = 50
    ) -> Tuple[List[WorkspaceInfo], int]:
        """
        List all known workspaces with pagination.

        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.

        Returns:
            Tuple of (list of WorkspaceInfo, total count).
        """
        all_workspaces = []
        for workspace_id, metadata in self._workspace_metadata.items():
            is_active = workspace_id in self._instances
            access_count = 0
            if is_active:
                _, _, _, access_count = self._instances[workspace_id]

            all_workspaces.append(
                WorkspaceInfo(
                    workspace_id=workspace_id,
                    owner_username=metadata.get("owner_username", workspace_id),
                    created_at=metadata.get("created_at", 0),
                    last_accessed_at=metadata.get("last_accessed_at", 0),
                    access_count=access_count,
                    is_active=is_active,
                )
            )

        # Sort by last_accessed_at descending
        all_workspaces.sort(key=lambda w: w.last_accessed_at, reverse=True)

        # Paginate
        total = len(all_workspaces)
        start = (page - 1) * page_size
        end = start + page_size

        return all_workspaces[start:end], total

    def get_workspace_info(self, workspace_id: str) -> Optional[WorkspaceInfo]:
        """
        Get information about a specific workspace.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            WorkspaceInfo if found, None otherwise.
        """
        if workspace_id not in self._workspace_metadata:
            return None

        metadata = self._workspace_metadata[workspace_id]
        is_active = workspace_id in self._instances
        access_count = 0
        if is_active:
            _, _, _, access_count = self._instances[workspace_id]

        return WorkspaceInfo(
            workspace_id=workspace_id,
            owner_username=metadata.get("owner_username", workspace_id),
            created_at=metadata.get("created_at", 0),
            last_accessed_at=metadata.get("last_accessed_at", 0),
            access_count=access_count,
            is_active=is_active,
        )

    async def shutdown(self) -> None:
        """
        Gracefully shut down the workspace manager.

        Finalizes all active instances and clears the cache.
        """
        logger.info("Shutting down WorkspaceManager...")

        async with self._lock:
            for workspace_id, (rag, _, _, _) in list(self._instances.items()):
                try:
                    logger.info(f"Finalizing workspace: {workspace_id}")
                    await rag.finalize_storages()
                except Exception as e:
                    logger.error(
                        f"Error finalizing workspace {workspace_id} during shutdown: {e}"
                    )

            self._instances.clear()

        logger.info("WorkspaceManager shutdown complete")
