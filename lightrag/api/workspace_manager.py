"""
Workspace Manager for Multi-Tenant LightRAG Server.

This module provides workspace management with LRU caching of LightRAG instances,
enabling a single server to serve multiple users with isolated workspaces while
sharing stateless components like embedding and LLM functions.

Supports both LightRAG and RAGAnything engines via the EngineType configuration.
"""

import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lightrag import LightRAG
from lightrag.base import EmbeddingFunc
from lightrag.utils import Tokenizer

logger = logging.getLogger("lightrag.api.workspace_manager")


class EngineType(Enum):
    """Supported RAG engine types."""

    LIGHTRAG = "lightrag"
    RAGANYTHING = "raganything"


@dataclass
class SharedComponents:
    """
    Stateless components shared across all workspace LightRAG instances.

    These components are created once during server startup and injected into
    every LightRAG instance to save memory and resources.
    """

    embedding_func: EmbeddingFunc
    """Embedding function shared across all workspaces."""

    llm_model_func: Callable[..., Any]
    """LLM function shared across all workspaces."""

    rerank_model_func: Optional[Callable[..., Any]] = None
    """Optional rerank function shared across all workspaces."""

    tokenizer: Optional[Tokenizer] = None
    """Tokenizer shared across all workspaces."""

    # RAGAnything-specific components (only used when engine_type is RAGANYTHING)
    vision_model_func: Optional[Callable[..., Any]] = None
    """Vision model function for RAGAnything multimodal processing."""

    raganything_embedding_func: Optional[EmbeddingFunc] = None
    """Optional separate embedding function for RAGAnything (uses embedding_func if None)."""


@dataclass
class WorkspaceConfig:
    """
    Configuration for creating LightRAG or RAGAnything instances.

    This captures all the non-workspace-specific configuration needed
    to create new instances.
    """

    # Engine selection
    engine_type: EngineType = EngineType.LIGHTRAG
    """Which RAG engine to use (lightrag or raganything)."""

    working_dir: str = "./rag_storage"
    kv_storage: str = "JsonKVStorage"
    vector_storage: str = "NanoVectorDBStorage"
    graph_storage: str = "NetworkXStorage"
    doc_status_storage: str = "JsonDocStatusStorage"

    # LLM settings
    llm_model_name: str = "gpt-4o-mini"
    llm_model_max_async: int = 4
    llm_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_llm_timeout: int = 120

    # Embedding settings
    default_embedding_timeout: int = 60

    # Chunking settings
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Summary settings
    summary_max_tokens: int = 500
    summary_context_size: int = 4000
    summary_language: str = "English"

    # Storage settings
    vector_db_storage_cls_kwargs: Dict[str, Any] = field(default_factory=dict)
    enable_llm_cache: bool = True
    enable_llm_cache_for_entity_extract: bool = True

    # Query settings
    max_parallel_insert: int = 2
    max_graph_nodes: int = 1000

    # Additional parameters
    addon_params: Dict[str, Any] = field(default_factory=dict)
    ollama_server_infos: Optional[Any] = None

    # RAGAnything-specific settings
    raganything_parser: str = "mineru"
    """Parser for RAGAnything: 'mineru' or 'docling'."""

    raganything_parse_method: str = "auto"
    """Parse method for RAGAnything: 'auto', 'ocr', or 'txt'."""

    raganything_enable_image_processing: bool = True
    """Enable image processing in RAGAnything."""

    raganything_enable_table_processing: bool = True
    """Enable table processing in RAGAnything."""

    raganything_enable_equation_processing: bool = True
    """Enable equation processing in RAGAnything."""


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
    Manages LightRAG/RAGAnything instances for multiple workspaces with LRU caching.

    This class maintains a pool of active RAG instances, creating new
    ones on demand and evicting least-recently-used instances when the pool
    reaches capacity. Supports both LightRAG and RAGAnything engines.

    Attributes:
        max_instances: Maximum number of RAG instances to keep in memory.
        ttl_minutes: Time-to-live for inactive instances (not currently enforced).
        engine_type: The RAG engine type being used (lightrag or raganything).
    """

    def __init__(
        self,
        shared_components: SharedComponents,
        workspace_config: WorkspaceConfig,
        max_instances: int = 100,
        ttl_minutes: int = 60,
    ):
        """
        Initialize the WorkspaceManager.

        Args:
            shared_components: Stateless components to share across instances.
            workspace_config: Configuration for creating new RAG instances.
            max_instances: Maximum number of instances to keep in memory.
            ttl_minutes: Time-to-live for inactive instances.
        """
        self._shared = shared_components
        self._config = workspace_config
        self._max_instances = max_instances
        self._ttl_minutes = ttl_minutes

        # LRU cache: workspace_id -> (RAG instance, created_at, last_accessed, access_count)
        # Instance can be LightRAG or RAGAnything depending on engine_type
        self._instances: OrderedDict[str, Tuple[Any, float, float, int]] = OrderedDict()
        self._lock = asyncio.Lock()

        # Separate cache for RAGAnything instances when in hybrid mode
        # This allows having both LightRAG and RAGAnything per workspace
        self._raganything_instances: OrderedDict[str, Tuple[Any, float, float, int]] = (
            OrderedDict()
        )
        self._raganything_lock = asyncio.Lock()

        # Track workspace metadata (persists even after eviction)
        self._workspace_metadata: Dict[str, Dict[str, Any]] = {}

        # RAGAnything components (lazy-loaded)
        self._raganything_class = None
        self._raganything_config_class = None

        engine_name = workspace_config.engine_type.value
        logger.info(
            f"WorkspaceManager initialized with engine={engine_name}, "
            f"max_instances={max_instances}, ttl_minutes={ttl_minutes}"
        )

    @property
    def active_instance_count(self) -> int:
        """Return the number of active RAG instances in memory."""
        return len(self._instances)

    @property
    def max_instances(self) -> int:
        """Return the maximum number of instances allowed."""
        return self._max_instances

    @property
    def engine_type(self) -> EngineType:
        """Return the engine type being used."""
        return self._config.engine_type

    async def get_instance(self, workspace_id: str) -> Any:
        """
        Get or create a RAG instance for the specified workspace.

        If an instance exists in the cache, it's moved to the end (most recently used).
        If not, a new instance is created, and LRU eviction occurs if needed.

        Args:
            workspace_id: The workspace identifier (typically sanitized username).

        Returns:
            The RAG instance (LightRAG or RAGAnything) for the workspace.
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

            # Create new instance
            engine_name = self._config.engine_type.value
            logger.info(f"Creating new {engine_name} instance for workspace: {workspace_id}")
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
                    "owner_username": workspace_id,  # Will be set properly by caller
                }
            self._workspace_metadata[workspace_id]["last_accessed_at"] = now

            return rag

    async def get_lightrag_instance(self, workspace_id: str) -> "LightRAG":
        """
        Get or create a LightRAG instance for the specified workspace.

        This always returns a LightRAG instance, regardless of engine_type.
        Useful when both LightRAG and RAGAnything are needed per workspace.

        Args:
            workspace_id: The workspace identifier (typically sanitized username).

        Returns:
            The LightRAG instance for the workspace.
        """
        async with self._lock:
            if workspace_id in self._instances:
                instance_data = self._instances[workspace_id]
                rag = instance_data[0]
                # If engine_type is RAGANYTHING, the cached instance is RAGAnything
                # which wraps a LightRAG - extract the underlying LightRAG
                if self._config.engine_type == EngineType.RAGANYTHING:
                    # RAGAnything has a .lightrag attribute
                    rag = getattr(rag, "lightrag", rag)
                # Move to end (most recently used) and update access time
                created_at = instance_data[1]
                count = instance_data[3]
                self._instances.pop(workspace_id)
                self._instances[workspace_id] = (
                    instance_data[0],  # Keep original instance
                    created_at,
                    time.time(),
                    count + 1,
                )
                logger.debug(f"LightRAG cache hit for workspace: {workspace_id}")
                return rag

            # Create new LightRAG instance
            logger.info(f"Creating new LightRAG instance for workspace: {workspace_id}")
            rag = await self._create_lightrag_instance(workspace_id)

            # Evict LRU if over capacity
            while len(self._instances) >= self._max_instances:
                await self._evict_lru()

            now = time.time()
            self._instances[workspace_id] = (rag, now, now, 1)

            # Update metadata
            if workspace_id not in self._workspace_metadata:
                self._workspace_metadata[workspace_id] = {
                    "created_at": now,
                    "owner_username": workspace_id,
                }
            self._workspace_metadata[workspace_id]["last_accessed_at"] = now

            return rag

    async def get_raganything_instance(self, workspace_id: str) -> Any:
        """
        Get or create a RAGAnything instance for the specified workspace.

        This is separate from the main instance cache to allow having both
        LightRAG and RAGAnything instances per workspace for hybrid mode.

        Args:
            workspace_id: The workspace identifier (typically sanitized username).

        Returns:
            The RAGAnything instance for the workspace.

        Raises:
            ImportError: If RAGAnything is not installed.
            ValueError: If vision_model_func is not configured.
        """
        async with self._raganything_lock:
            if workspace_id in self._raganything_instances:
                # Move to end (most recently used) and update access time
                rag_anything, created_at, _, count = self._raganything_instances.pop(
                    workspace_id
                )
                self._raganything_instances[workspace_id] = (
                    rag_anything,
                    created_at,
                    time.time(),
                    count + 1,
                )
                logger.debug(f"RAGAnything cache hit for workspace: {workspace_id}")
                return rag_anything

            # Create new RAGAnything instance
            logger.info(
                f"Creating new RAGAnything instance for workspace: {workspace_id}"
            )
            rag_anything = await self._create_raganything_instance_standalone(
                workspace_id
            )

            # Evict LRU if over capacity (same limit as main instances)
            while len(self._raganything_instances) >= self._max_instances:
                await self._evict_raganything_lru()

            now = time.time()
            self._raganything_instances[workspace_id] = (rag_anything, now, now, 1)

            return rag_anything

    async def _evict_raganything_lru(self) -> None:
        """Evict the least recently used RAGAnything instance from the cache."""
        if not self._raganything_instances:
            return

        # Get oldest (first item in OrderedDict)
        workspace_id, (rag_anything, created_at, last_accessed, count) = next(
            iter(self._raganything_instances.items())
        )

        logger.info(
            f"Evicting LRU RAGAnything workspace: {workspace_id} "
            f"(created: {created_at:.0f}, last_access: {last_accessed:.0f}, "
            f"access_count: {count})"
        )

        # RAGAnything wraps LightRAG - finalize the underlying LightRAG
        try:
            lightrag = getattr(rag_anything, "lightrag", None)
            if lightrag:
                await lightrag.finalize_storages()
        except Exception as e:
            logger.error(
                f"Error finalizing RAGAnything storage for workspace {workspace_id}: {e}"
            )

        del self._raganything_instances[workspace_id]

    async def _create_raganything_instance_standalone(self, workspace_id: str) -> Any:
        """
        Create a new standalone RAGAnything instance for a workspace.

        This creates both a LightRAG and RAGAnything instance for the workspace,
        suitable for hybrid mode where both frameworks are available.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            A newly initialized RAGAnything instance.
        """
        # Lazy import RAGAnything
        if self._raganything_class is None:
            try:
                from raganything import RAGAnything, RAGAnythingConfig

                self._raganything_class = RAGAnything
                self._raganything_config_class = RAGAnythingConfig
            except ImportError as e:
                raise ImportError(
                    "RAGAnything is not installed. Please run 'pip install raganything' "
                    "to enable multimodal document processing."
                ) from e

        # Validate required components
        if self._shared.vision_model_func is None:
            raise ValueError(
                "vision_model_func is required for RAGAnything. "
                "Please configure SharedComponents with a vision model function."
            )

        config = self._config

        # Get or create the underlying LightRAG instance (uses the main cache)
        lightrag_instance = await self.get_lightrag_instance(workspace_id)

        # Create RAGAnything config
        raganything_config = self._raganything_config_class(
            working_dir=config.working_dir,
            parser=config.raganything_parser,
            parse_method=config.raganything_parse_method,
            enable_image_processing=config.raganything_enable_image_processing,
            enable_table_processing=config.raganything_enable_table_processing,
            enable_equation_processing=config.raganything_enable_equation_processing,
        )

        # Use RAGAnything-specific embedding func if provided
        embedding_func = (
            self._shared.raganything_embedding_func
            if self._shared.raganything_embedding_func is not None
            else self._shared.embedding_func
        )

        # Create RAGAnything instance wrapping LightRAG
        rag_anything = self._raganything_class(
            lightrag=lightrag_instance,
            config=raganything_config,
            llm_model_func=self._shared.llm_model_func,
            vision_model_func=self._shared.vision_model_func,
            embedding_func=embedding_func,
        )

        logger.info(
            f"Standalone RAGAnything instance created for workspace: {workspace_id}"
        )
        return rag_anything

    async def _create_instance(self, workspace_id: str) -> Any:
        """
        Create a new RAG instance for a workspace.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            A newly initialized RAG instance (LightRAG or RAGAnything).
        """
        if self._config.engine_type == EngineType.RAGANYTHING:
            return await self._create_raganything_instance(workspace_id)
        else:
            return await self._create_lightrag_instance(workspace_id)

    async def _create_lightrag_instance(self, workspace_id: str) -> LightRAG:
        """
        Create a new LightRAG instance for a workspace.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            A newly initialized LightRAG instance.
        """
        config = self._config

        rag = LightRAG(
            working_dir=config.working_dir,
            workspace=workspace_id,
            # Shared components
            embedding_func=self._shared.embedding_func,
            llm_model_func=self._shared.llm_model_func,
            rerank_model_func=self._shared.rerank_model_func,
            tokenizer=self._shared.tokenizer,
            # Storage configuration
            kv_storage=config.kv_storage,
            vector_storage=config.vector_storage,
            graph_storage=config.graph_storage,
            doc_status_storage=config.doc_status_storage,
            # LLM settings
            llm_model_name=config.llm_model_name,
            llm_model_max_async=config.llm_model_max_async,
            llm_model_kwargs=config.llm_model_kwargs,
            default_llm_timeout=config.default_llm_timeout,
            # Embedding settings
            default_embedding_timeout=config.default_embedding_timeout,
            # Chunking settings
            chunk_token_size=config.chunk_token_size,
            chunk_overlap_token_size=config.chunk_overlap_token_size,
            tiktoken_model_name=config.tiktoken_model_name,
            # Summary settings
            summary_max_tokens=config.summary_max_tokens,
            summary_context_size=config.summary_context_size,
            # Storage settings
            vector_db_storage_cls_kwargs=config.vector_db_storage_cls_kwargs,
            enable_llm_cache=config.enable_llm_cache,
            enable_llm_cache_for_entity_extract=config.enable_llm_cache_for_entity_extract,
            # Query settings
            max_parallel_insert=config.max_parallel_insert,
            max_graph_nodes=config.max_graph_nodes,
            # Additional parameters
            addon_params=config.addon_params,
            ollama_server_infos=config.ollama_server_infos,
        )

        # Initialize storages asynchronously
        await rag.initialize_storages()

        logger.info(f"LightRAG instance created and initialized for workspace: {workspace_id}")
        return rag

    async def _create_raganything_instance(self, workspace_id: str) -> Any:
        """
        Create a new RAGAnything instance for a workspace.

        RAGAnything wraps LightRAG and adds multimodal document processing capabilities.

        Args:
            workspace_id: The workspace identifier.

        Returns:
            A newly initialized RAGAnything instance.

        Raises:
            ImportError: If raganything package is not installed.
            ValueError: If required components (vision_model_func) are not configured.
        """
        # Lazy import RAGAnything to avoid import errors when not installed
        if self._raganything_class is None:
            try:
                from raganything import RAGAnything, RAGAnythingConfig

                self._raganything_class = RAGAnything
                self._raganything_config_class = RAGAnythingConfig
            except ImportError as e:
                raise ImportError(
                    "RAGAnything is not installed. Please run 'pip install raganything' "
                    "to enable multimodal document processing."
                ) from e

        # Validate required components
        if self._shared.vision_model_func is None:
            raise ValueError(
                "vision_model_func is required for RAGAnything engine. "
                "Please configure SharedComponents with a vision model function."
            )

        config = self._config

        # First, create the underlying LightRAG instance
        lightrag_instance = await self._create_lightrag_instance(workspace_id)

        # Create RAGAnything config
        raganything_config = self._raganything_config_class(
            working_dir=config.working_dir,
            parser=config.raganything_parser,
            parse_method=config.raganything_parse_method,
            enable_image_processing=config.raganything_enable_image_processing,
            enable_table_processing=config.raganything_enable_table_processing,
            enable_equation_processing=config.raganything_enable_equation_processing,
        )

        # Use RAGAnything-specific embedding func if provided, otherwise use shared
        embedding_func = (
            self._shared.raganything_embedding_func
            if self._shared.raganything_embedding_func is not None
            else self._shared.embedding_func
        )

        # Create RAGAnything instance wrapping LightRAG
        rag_anything = self._raganything_class(
            lightrag=lightrag_instance,
            config=raganything_config,
            llm_model_func=self._shared.llm_model_func,
            vision_model_func=self._shared.vision_model_func,
            embedding_func=embedding_func,
        )

        logger.info(
            f"RAGAnything instance created and initialized for workspace: {workspace_id}"
        )
        return rag_anything

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
        Create a new workspace (admin API).

        This creates the workspace metadata and optionally pre-initializes
        the LightRAG instance.

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
        Delete a workspace and all its data (admin API).

        This removes the instance from the cache and can optionally
        clean up the persistent storage.

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
        Gracefully shutdown the workspace manager.

        This finalizes all active instances and clears the cache.
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
