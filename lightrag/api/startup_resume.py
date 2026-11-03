"""Resume documents a restart left waiting.

Pipeline runs, the provider-rejection pause and its retry timer all live in
memory, so a restart stops them while the documents stay PENDING (or stranded
mid-pipeline). Without a new upload, scan or email nothing would start them
again, so the server looks for such documents once at startup.

FAILED documents alone do not trigger a run: retrying them on every restart
would spend LLM budget on documents that may never succeed. Like any other run,
a resumed run does retry the workspace's FAILED documents.

Fork-only and self-contained to keep upstream merges trivial.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from lightrag.api.ops import runs as ops_runs
from lightrag.base import DocStatus
from lightrag.pipeline import _INFLIGHT_DOC_STATUSES
from lightrag.utils import logger

RESUMABLE_DOC_STATUSES = tuple(
    status for status in _INFLIGHT_DOC_STATUSES if status is not DocStatus.FAILED
)

# Each run holds a slot until its workspace's queue is empty, so one large
# backlog cannot take every LLM slot from the other workspaces.
MAX_CONCURRENT_RESUMES = 4


async def _candidate_workspaces(
    default_rag: Any, registry: Any, multi_tenant: bool
) -> list[str]:
    if not multi_tenant:
        return [default_rag.workspace]

    # Doc status is the only complete source: email ingestion creates workspaces
    # with no registered owner and no input directory.
    found = await default_rag.doc_status.list_workspaces_with_statuses(
        list(RESUMABLE_DOC_STATUSES)
    )
    if found is None:
        owners = await registry.list_owners() if registry is not None else []
        logger.info(
            "Startup resume: this doc status backend cannot list workspaces; "
            "checking the default workspace and registered owners only"
        )
        return list(
            dict.fromkeys(
                [default_rag.workspace, *(owner.workspace_id for owner in owners)]
            )
        )

    # Storage may resolve an unnamed default workspace to its own name.
    storage_default = default_rag.doc_status.workspace
    return list(
        dict.fromkeys(
            default_rag.workspace if workspace == storage_default else workspace
            for workspace in found
        )
    )


async def resume_pending_documents(
    default_rag: Any,
    get_rag: Callable[[str], Awaitable[Any]],
    *,
    registry: Optional[Any] = None,
    multi_tenant: bool = False,
    max_concurrent: int = MAX_CONCURRENT_RESUMES,
) -> None:
    """Start the pipeline for every workspace with documents still waiting.

    One workspace failing does not affect the others. A workspace whose provider
    still rejects requests pauses again, which also restores its retry timer.
    """
    try:
        workspaces = await _candidate_workspaces(default_rag, registry, multi_tenant)
    except Exception as e:
        logger.error(f"Startup resume could not list workspaces: {e}")
        return

    semaphore = asyncio.Semaphore(max_concurrent)

    async def resume(workspace: str) -> None:
        async with semaphore:
            try:
                rag = await get_rag(workspace)
                counts = await rag.doc_status.get_status_counts()
                waiting = sum(
                    counts.get(status.value, 0) for status in RESUMABLE_DOC_STATUSES
                )
                if not waiting:
                    return
                logger.info(
                    f"Startup resume: processing {waiting} waiting document(s) "
                    f"in workspace '{workspace}'"
                )
                # Runs started here are recorded as startup resumes.
                with ops_runs.run_origin(kind="startup_resume", trigger="startup"):
                    await rag.apipeline_process_enqueue_documents()
            except Exception as e:
                logger.error(f"Startup resume failed for '{workspace}': {e}")

    await asyncio.gather(*(resume(workspace) for workspace in workspaces))
