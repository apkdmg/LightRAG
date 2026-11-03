"""Starts and stops the Operations console backend with the API server.

Called from the server lifespan, so it runs once per worker process (under
Gunicorn after the fork). It creates this worker's ``worker_id`` (never at
import: Gunicorn preloads the app, and module state would be shared by every
forked worker), the shared-state reader, and, on Postgres KV storage, the
ops store, the background index builder and the leased singleton loops.

Nothing here may stop the server from starting or serving: every failure is
logged and the console reports what is unavailable instead.

Switches (all default on): ``OPS_ENABLED`` (the whole backend),
``OPS_INDEX_BUILD`` (the background index builder) and ``OPS_QUEUE_SCAN``
(the email re-sync queue scan).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from lightrag.api.ops.context import set_worker_id
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.pauses import OwnProviderCache
from lightrag.api.ops.service import OpsService
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.usage import SystemEvent, add_event_sink, remove_event_sink

logger = logging.getLogger("lightrag.api.ops")

POSTGRES_KV_STORAGE = "PGKVStorage"
POSTGRES_DOC_STATUS_STORAGE = "PGDocStatusStorage"
_FORGET_RETRY_DELAYS_S = (5.0, 30.0, 120.0)


def env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class OpsRuntime:
    worker_id: str
    pid: int
    reader: NamespaceReader
    service: OpsService
    store: Any = None
    loops: list[LeasedLoop] = field(default_factory=list)
    tasks: set[asyncio.Task] = field(default_factory=set)
    index_task: Optional[asyncio.Task] = None
    index_outcome: Optional[dict[str, str]] = None

    def _spawn(self, coro: Any, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    # -- workspace deletion --------------------------------------------------
    def on_event(self, event: SystemEvent) -> None:
        """Event sink: a deleted workspace's ops data is removed in the background."""
        if (
            event.kind != "workspace_deleted"
            or not event.workspace
            or self.store is None
        ):
            return
        try:
            self._spawn(self._forget(event.workspace), "ops-forget-workspace")
        except RuntimeError:  # no running loop
            logger.warning(
                "Ops data of a deleted workspace could not be scheduled for removal"
            )

    async def _forget(self, workspace: str) -> None:
        for attempt, delay in enumerate((0.0, *_FORGET_RETRY_DELAYS_S), start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                await self.store.forget_workspace(workspace)
                self.service.cache.clear()
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "Removing a deleted workspace's ops data failed (attempt %d, %s)",
                    attempt,
                    type(e).__name__,
                )
        logger.error(
            "A deleted workspace's ops data could not be removed; retry the deletion"
        )


async def start_ops(app: Any, rag: Any, args: Any) -> Optional[OpsRuntime]:
    """Create this worker's ops runtime and set ``app.state.ops``."""
    current = getattr(app.state, "ops", None)
    if isinstance(current, OpsRuntime) and current.pid == os.getpid():
        return current
    try:
        return await _start(app, args)
    except Exception:
        logger.error("Operations console backend could not start", exc_info=True)
        app.state.ops = None
        return None


async def _start(app: Any, args: Any) -> OpsRuntime:
    worker_id = uuid4().hex
    set_worker_id(worker_id)
    reader = NamespaceReader()
    store = None
    reason: Optional[str] = None
    if getattr(args, "kv_storage", "") != POSTGRES_KV_STORAGE:
        reason = "postgres_required"
        logger.info(
            "Operations console: durable views need LIGHTRAG_KV_STORAGE=PGKVStorage"
        )
    elif not env_flag("OPS_ENABLED"):
        reason = "disabled"
        logger.info("Operations console: disabled by OPS_ENABLED=false")
    else:
        from lightrag.api.ops.store import PostgresOpsStore

        candidate = PostgresOpsStore(getattr(args, "vector_storage", None))
        try:
            await candidate.initialize()
            # Doc-status SQL only describes the documents when they live there.
            doc_status = getattr(
                args, "doc_status_storage", POSTGRES_DOC_STATUS_STORAGE
            )
            if doc_status != POSTGRES_DOC_STATUS_STORAGE:
                candidate.doc_status_ready = False
                logger.info(
                    "Operations console: processing views need "
                    "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage"
                )
            store = candidate
        except Exception as e:
            reason = "store_unavailable"
            logger.error(
                "Operations console store could not start (%s)", type(e).__name__
            )
            try:
                await candidate.finalize()
            except Exception:
                pass
    provider_store = getattr(app.state, "workspace_provider_store", None)
    service = OpsService(
        store=store,
        reader=reader,
        app_state=app.state,
        unavailable_reason=reason,
        own_providers=OwnProviderCache(provider_store)
        if provider_store is not None
        else None,
    )
    runtime = OpsRuntime(
        worker_id=worker_id,
        pid=os.getpid(),
        reader=reader,
        service=service,
        store=store,
    )
    app.state.ops = runtime
    if store is None:
        return runtime

    add_event_sink(runtime.on_event)
    if env_flag("OPS_INDEX_BUILD"):
        from lightrag.api.ops.indexes import run_index_builder

        async def build() -> None:
            runtime.index_outcome = await run_index_builder(store)

        runtime.index_task = runtime._spawn(build(), "ops-index-builder")
    queue = getattr(app.state, "email_resync_queue", None)
    if queue is not None and env_flag("OPS_QUEUE_SCAN"):
        from lightrag.api.ops import queue_scan

        async def scan(loop: LeasedLoop) -> None:
            await queue_scan.scan_cycle(loop, store=store, base_dir=queue.base_dir)

        loop = LeasedLoop(
            store,
            queue_scan.SCAN_LEASE,
            worker_id,
            scan,
            ttl_s=queue_scan.SCAN_LEASE_TTL_S,
            interval_s=queue_scan.SCAN_INTERVAL_S,
            cycle_timeout_s=queue_scan.SCAN_CYCLE_TIMEOUT_S,
            # Spread workers so they do not all race for the lease at once.
            initial_delay_s=random.uniform(2.0, 10.0),
        )
        loop.start()
        runtime.loops.append(loop)
    logger.info(
        "Operations console: processing views on Postgres (worker %s)", worker_id[:8]
    )
    return runtime


async def stop_ops(app: Any) -> None:
    runtime = getattr(app.state, "ops", None)
    if not isinstance(runtime, OpsRuntime):
        return
    remove_event_sink(runtime.on_event)
    for loop in runtime.loops:
        try:
            await loop.stop()
        except Exception:
            logger.debug("Ops loop %s did not stop cleanly", loop.name, exc_info=True)
    tasks = list(runtime.tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    runtime.reader.close()
    if runtime.store is not None:
        try:
            await runtime.store.finalize()
        except Exception:
            logger.debug("Ops store did not close cleanly", exc_info=True)
    app.state.ops = None
