"""Time-bounded reads of shared pipeline state for the Operations console.

Under Gunicorn every key read of the shared ``pipeline_status`` is a blocking
round trip to the ``multiprocessing.Manager``. Done on the event loop, a slow
or dead Manager would stall the whole worker. :class:`NamespaceReader` runs
:func:`lightrag.kg.shared_storage.snapshot_namespace` on its own single thread
with a 2 s deadline and at most one call in flight; while a call is stuck the
Manager is reported as unresponsive and pages fall back to SQL only. In
single-process mode the state is a plain dict and is read directly.

The reader never creates a namespace and never takes a storage lock.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from lightrag.kg import shared_storage

logger = logging.getLogger("lightrag.api.ops")

SNAPSHOT_TIMEOUT_S = 2.0
PIPELINE_NAMESPACE = "pipeline_status"

# What the processing views read for each candidate workspace.
LIVE_KEYS = (
    "busy",
    "destructive_busy",
    "scanning",
    "scanning_exclusive",
    "pending_enqueues",
    "request_pending",
    "cancellation_requested",
    "job_name",
    "job_start",
    "docs",
    "batchs",
    "cur_batch",
    "latest_message",
    "provider_rejection",
    "run_id",
    "owner_worker_id",
    "owner_pid",
    "heartbeat_at",
    "scan_owner_worker_id",
    "scan_started_at",
)
PAUSE_KEYS = ("provider_rejection",)


@dataclass
class SnapshotBatch:
    """Snapshots by workspace; ``responsive`` is False when the Manager timed out."""

    data: dict[str, Optional[dict[str, Any]]] = field(default_factory=dict)
    responsive: bool = True


class NamespaceReader:
    """One per process (created by the ops runtime, never at import)."""

    def __init__(self, timeout_s: float = SNAPSHOT_TIMEOUT_S):
        self._timeout_s = timeout_s
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_pid: Optional[int] = None
        self._inflight = threading.Event()
        self._inflight_since: Optional[float] = None
        self._gate: Optional[asyncio.Lock] = None
        self._gate_loop: Optional[asyncio.AbstractEventLoop] = None
        self.unresponsive_since: Optional[float] = None
        self.last_rtt_ms: Optional[float] = None

    # -- state -------------------------------------------------------------
    @property
    def state(self) -> str:
        """``ok``, ``unresponsive`` (a call timed out or is still stuck) or ``local``.

        A call that is merely in flight is normal; it only counts once it has
        run past the deadline (its own caller then records the timeout, but a
        concurrent reader may look first).
        """
        if not shared_storage.is_multiprocess():
            return "local"
        if self.unresponsive_since is not None:
            return "unresponsive"
        since = self._inflight_since
        if (
            since is not None
            and self._inflight.is_set()
            and time.monotonic() - since > self._timeout_s
        ):
            return "unresponsive"
        return "ok"

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            # A stuck Manager call cannot be interrupted; do not wait for it.
            executor.shutdown(wait=False, cancel_futures=True)

    # -- internals -----------------------------------------------------------
    def _pool(self) -> ThreadPoolExecutor:
        pid = os.getpid()
        if self._executor is None or self._executor_pid != pid:
            # A forked child must not reuse its parent's (threadless) executor.
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ops-manager-read"
            )
            self._executor_pid = pid
            self._inflight.clear()
        return self._executor

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._gate is None or self._gate_loop is not loop:
            self._gate = asyncio.Lock()
            self._gate_loop = loop
        return self._gate

    @staticmethod
    def _read_many(
        workspaces: list[str],
        keys: tuple[str, ...],
        list_slices: Optional[dict[str, tuple]],
    ) -> tuple[dict[str, Optional[dict[str, Any]]], int]:
        """Snapshots by workspace, plus how many reads raised."""
        out: dict[str, Optional[dict[str, Any]]] = {}
        failures = 0
        for workspace in workspaces:
            try:
                out[workspace] = shared_storage.snapshot_namespace(
                    PIPELINE_NAMESPACE, workspace, keys, list_slices=list_slices
                )
            except Exception as e:  # a broken proxy for one workspace
                logger.warning(
                    "Ops snapshot read failed for one workspace (%s)",
                    type(e).__name__,
                )
                out[workspace] = None
                failures += 1
        return out, failures

    def _run_job(self, job) -> Any:
        try:
            return job()
        finally:
            self._inflight.clear()

    # -- public ----------------------------------------------------------------
    async def snapshot_many(
        self,
        workspaces: Iterable[str],
        keys: Iterable[str] = LIVE_KEYS,
        *,
        list_slices: Optional[dict[str, tuple]] = None,
    ) -> SnapshotBatch:
        """Read ``keys`` for each workspace in one bounded job."""
        names = list(dict.fromkeys(w for w in workspaces if w is not None))
        wanted = tuple(keys)
        denied = set(wanted) - shared_storage.SNAPSHOT_KEYS[PIPELINE_NAMESPACE]
        if denied:
            raise ValueError(f"Keys not readable by snapshot: {sorted(denied)}")
        if not names:
            return SnapshotBatch()
        if not shared_storage.is_multiprocess():
            data, failures = self._read_many(names, wanted, list_slices)
            return SnapshotBatch(data, responsive=failures < len(names))

        async with self._lock():
            if self._inflight.is_set():
                # The previous call is still stuck in the Manager.
                return SnapshotBatch(responsive=False)
            pool = self._pool()
            self._inflight_since = time.monotonic()
            self._inflight.set()
            started = time.perf_counter()
            try:
                future = pool.submit(
                    self._run_job,
                    lambda: self._read_many(names, wanted, list_slices),
                )
            except RuntimeError:
                self._inflight.clear()
                return SnapshotBatch(responsive=False)
            try:
                data, failures = await asyncio.wait_for(
                    asyncio.wrap_future(future), timeout=self._timeout_s
                )
            except asyncio.TimeoutError:
                if self.unresponsive_since is None:
                    self.unresponsive_since = time.time()
                    logger.warning(
                        "Shared pipeline state did not answer within %.1fs; "
                        "Operations pages show SQL data only",
                        self._timeout_s,
                    )
                return SnapshotBatch(responsive=False)
            except Exception as e:
                logger.warning("Ops snapshot read failed (%s)", type(e).__name__)
                return SnapshotBatch(responsive=False)
            self.last_rtt_ms = (time.perf_counter() - started) * 1000.0
            if failures and failures == len(names):
                # Every read raised: the Manager is gone rather than slow.
                if self.unresponsive_since is None:
                    self.unresponsive_since = time.time()
                return SnapshotBatch(data, responsive=False)
            if self.unresponsive_since is not None:
                logger.info("Shared pipeline state answers again")
                self.unresponsive_since = None
            return SnapshotBatch(data)

    async def snapshot(
        self,
        workspace: str,
        keys: Iterable[str] = LIVE_KEYS,
        *,
        list_slices: Optional[dict[str, tuple]] = None,
    ) -> tuple[Optional[dict[str, Any]], bool]:
        """One workspace: ``(snapshot or None, responsive)``."""
        batch = await self.snapshot_many([workspace], keys, list_slices=list_slices)
        return batch.data.get(workspace), batch.responsive
