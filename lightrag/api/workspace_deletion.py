"""
Permanent deletion of a workspace and everything stored for it.

Used by the owner's ``DELETE /workspace`` and the admin's
``DELETE /admin/workspaces/{id}``. The purge removes:

- every storage of the workspace's LightRAG instance (documents, chunks,
  entities, relations, vectors, graph, doc status and the LLM response cache),
  plus backend objects named after the workspace (Milvus collections, Neo4j
  indexes) where the backend provides ``drop_workspace_schema``;
- its input directory (uploaded and parsed files) and its working directory
  (file-based storages and per-user API keys);
- its BYO provider credentials;
- its ownership binding, replaced by a tombstone so sessions issued before the
  deletion stop working.

It runs under the pipeline's destructive slot, like ``/documents/clear``, and
the workspace answers HTTP 409 to other requests meanwhile. If a storage cannot
be dropped, nothing else is removed and ownership is kept, so the deletion can
simply be retried.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lightrag.api.config import validate_tenant_isolation_configuration
from lightrag.api.workspace_manager import WorkspaceManager
from lightrag.api.workspace_registry import WorkspaceRegistry

logger = logging.getLogger("lightrag.api.workspace_deletion")


class WorkspaceBusyError(Exception):
    """The workspace's pipeline is processing, scanning or enqueuing."""


@dataclass
class WorkspacePurgeResult:
    workspace_id: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _storages(rag: Any) -> list[Any]:
    """The instance's storages, as initialized by ``initialize_storages``."""
    return [
        storage
        for storage in (
            rag.full_docs,
            rag.text_chunks,
            rag.full_entities,
            rag.full_relations,
            rag.entity_chunks,
            rag.relation_chunks,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.chunks_vdb,
            rag.chunk_entity_relation_graph,
            rag.llm_response_cache,
            rag.doc_status,
        )
        if storage is not None
    ]


def _check_isolation(workspace_id: str, storages: list[Any]) -> None:
    """Refuse to drop anything a storage might share with other workspaces."""
    validate_tenant_isolation_configuration(
        argparse.Namespace(enable_multi_tenancy=True)
    )
    for storage in storages:
        if getattr(storage, "workspace", None) != workspace_id:
            raise RuntimeError(
                f"{storage.__class__.__name__} is not scoped to workspace "
                f"'{workspace_id}'; refusing to drop it"
            )


def _workspace_subdir(base_dir: str | Path, workspace_id: str) -> Path:
    base = Path(base_dir).resolve()
    path = (base / workspace_id).resolve()
    if path.parent != base:
        raise RuntimeError(f"Workspace path escapes {base}: {path}")
    return path


async def _drop_storages(storages: list[Any], errors: list[str]) -> None:
    async def drop(storage: Any) -> None:
        result = await storage.drop()
        # Most backends report failure in the result rather than raising
        if isinstance(result, dict) and result.get("status") == "error":
            raise RuntimeError(result.get("message") or "drop failed")
        remove_schema = getattr(storage, "drop_workspace_schema", None)
        if remove_schema is not None:
            await remove_schema()

    results = await asyncio.gather(
        *(drop(storage) for storage in storages), return_exceptions=True
    )
    for storage, result in zip(storages, results):
        if isinstance(result, Exception):
            errors.append(f"{storage.__class__.__name__}/{storage.namespace}: {result}")


async def _clear_pipeline_history(rag: Any) -> None:
    """Drop the in-memory job log, which names the workspace's files."""
    from lightrag.exceptions import PipelineNotInitializedError
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    try:
        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    except PipelineNotInitializedError:
        return
    async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
        if "history_messages" in status:
            del status["history_messages"][:]
        status["latest_message"] = "Workspace deleted"
        status["docs"] = 0


def _remove_files(
    workspace_id: str,
    working_dir: str | Path,
    input_dir: str | Path,
    provider_store: Optional[Any],
    errors: list[str],
) -> None:
    for base_dir in (input_dir, working_dir):
        try:
            path = _workspace_subdir(base_dir, workspace_id)
            if path.exists():
                shutil.rmtree(path)
        except Exception as e:
            errors.append(f"remove {base_dir}/{workspace_id}: {e}")

    if provider_store is not None:
        try:
            provider_store.delete(workspace_id)
        except Exception as e:
            errors.append(f"provider credentials: {e}")

    from lightrag.api.routers.apikey_routes import forget_workspace_keys

    forget_workspace_keys(workspace_id)


async def purge_workspace(
    workspace_id: str,
    *,
    workspace_manager: WorkspaceManager,
    registry: Optional[WorkspaceRegistry],
    working_dir: str | Path,
    input_dir: str | Path,
    provider_store: Optional[Any] = None,
) -> WorkspacePurgeResult:
    """Permanently delete a workspace's data and release its ID.

    Raises:
        ValueError: ``workspace_id`` is not a valid workspace ID.
        WorkspaceBusyError: the workspace's pipeline is in use; retry later.
        WorkspaceDeletionInProgress: another deletion of it is running.
    """
    # Sanitized IDs only (older ones may contain "-"): no path tricks
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", workspace_id or ""):
        raise ValueError(f"Invalid workspace ID: {workspace_id!r}")

    from lightrag.api.routers.document_routes import (
        _acquire_destructive_busy,
        _release_destructive_busy,
    )

    result = WorkspacePurgeResult(workspace_id)
    owner = await registry.get_by_workspace(workspace_id) if registry else None

    rag = await workspace_manager.begin_deletion(workspace_id)
    try:
        acquired, reason = await _acquire_destructive_busy(rag)
        if not acquired:
            raise WorkspaceBusyError(reason)
        try:
            storages = _storages(rag)
            try:
                _check_isolation(workspace_id, storages)
            except (RuntimeError, ValueError) as e:
                result.errors.append(str(e))
            else:
                await _drop_storages(storages, result.errors)
            if result.ok:
                await _clear_pipeline_history(rag)
        finally:
            await _release_destructive_busy(rag)

        if result.ok:
            # Close before removing files: finalizing file-based storages may
            # flush their (now empty) state back into the working directory.
            await workspace_manager.discard_instance(workspace_id)
            _remove_files(
                workspace_id, working_dir, input_dir, provider_store, result.errors
            )
        if result.ok and registry is not None:
            await registry.retire(workspace_id, owner.identity if owner else None)
    finally:
        await workspace_manager.end_deletion(workspace_id)

    if result.ok:
        logger.warning(f"Workspace '{workspace_id}' and all its data deleted")
    else:
        logger.error(
            f"Deleting workspace '{workspace_id}' failed: {'; '.join(result.errors)}"
        )
    return result


async def workspace_exists(app: Any, workspace_id: str) -> bool:
    """Whether a workspace is bound, served, or has files on this server."""
    from lightrag.api.config import global_args

    registry = getattr(app.state, "workspace_registry", None)
    if registry is not None and await registry.get_by_workspace(workspace_id):
        return True
    if app.state.workspace_manager.is_known(workspace_id):
        return True
    return any(
        (Path(base_dir) / workspace_id).is_dir()
        for base_dir in (global_args.working_dir, global_args.input_dir)
    )


async def purge_app_workspace(app: Any, workspace_id: str) -> WorkspacePurgeResult:
    """:func:`purge_workspace` with the server's own manager, registry,
    provider store and directories."""
    from lightrag.api.config import global_args

    return await purge_workspace(
        workspace_id,
        workspace_manager=app.state.workspace_manager,
        registry=getattr(app.state, "workspace_registry", None),
        working_dir=global_args.working_dir,
        input_dir=global_args.input_dir,
        provider_store=getattr(app.state, "workspace_provider_store", None),
    )
