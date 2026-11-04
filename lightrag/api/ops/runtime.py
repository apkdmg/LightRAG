"""Starts and stops the Operations console backend with the API server.

Called from the server lifespan, so it runs once per worker process (under
Gunicorn after the fork). It creates this worker's ``worker_id`` (never at
import: Gunicorn preloads the app, and module state would be shared by every
forked worker), the shared-state reader, and, on Postgres KV storage, the
ops store, the background index builder and the leased singleton loops.

Nothing here may stop the server from starting or serving: every failure is
logged and the console reports what is unavailable instead.

It also starts this worker's heartbeat (every process) and the dependency
probes (a leased singleton on Postgres; local to each process without it),
and, on Postgres, run tracking (this worker's buffered run writer), the
leased reconcile + attention cycle, the structured log sink (this worker's
handler and flusher) and the leased log retention.

Switches (all default on): ``OPS_ENABLED`` (the whole backend),
``OPS_INDEX_BUILD`` (the background index builder), ``OPS_QUEUE_SCAN``
(the email re-sync queue scan), ``OPS_PROBES_ENABLED`` (dependency
probes) and ``OPS_RUN_TRACKING`` (run and failure tracking; off makes
every pipeline hook a no-op). ``OPS_LOG_SINK`` (default **off** until the
write volume has been measured) stores WARNING+ log lines in
``LIGHTRAG_OPS_LOGS``; off means no handler is attached at all.
``OPS_METRICS_ENABLED`` (default on) records HTTP, embed, rerank and
sign-in metrics (this worker's histograms and flusher, plus the leased
hourly rollup/retention and LLM p95 baseline jobs); off makes every
hot-path hook a no-op.
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
from lightrag.api.ops.heartbeat import Heartbeat
from lightrag.api.ops.leases import LeasedLoop, LocalLeases
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
    heartbeat: Optional[Heartbeat] = None
    probes: Any = None
    run_tracker: Any = None
    log_sink: Any = None
    metrics: Any = None

    def leases_held(self) -> list[str]:
        return [
            loop.name for loop in self.loops if loop.held and loop.store is self.store
        ]

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
        sink = self.log_sink
        if sink is not None:
            # Same rule as the run rows: nothing buffered for the workspace
            # is written after its stored lines are deleted.
            try:
                await sink.forget_flushed(workspace)
            except Exception as e:
                logger.warning(
                    "Dropping a deleted workspace's buffered log lines failed (%s)",
                    type(e).__name__,
                )
        tracker = self.run_tracker
        if tracker is not None:
            # Rows buffered for the workspace are dropped before its stored
            # ops data is removed, so no flush writes them back afterwards.
            try:
                await tracker.forget_flushed(workspace)
            except Exception as e:
                logger.warning(
                    "Dropping a deleted workspace's buffered ops rows failed (%s)",
                    type(e).__name__,
                )
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
        return await _start(app, rag, args)
    except Exception:
        logger.error("Operations console backend could not start", exc_info=True)
        app.state.ops = None
        return None


def _process(app: Any) -> dict[str, Any]:
    """Process mode and expected workers, as the ops routes report them."""
    from lightrag.api.routers import ops_routes

    info = getattr(app.state, "ops_server", None)
    mode = ops_routes.process_mode()
    return {
        "mode": mode,
        "expected": ops_routes._expected_workers(
            mode, info.workers if info is not None else None
        ),
        "live": None,
    }


def _start_heartbeat(app: Any, rag: Any, args: Any, runtime: OpsRuntime) -> None:
    info = getattr(app.state, "ops_server", None)
    heartbeat = Heartbeat(
        worker_id=runtime.worker_id,
        mode=_process(app)["mode"],
        version=getattr(info, "version", "") or "",
        store=runtime.store,
        reader=runtime.reader,
        app_state=app.state,
        leases=runtime.leases_held,
        data_rag=lambda: getattr(app.state, "default_rag", None) or rag,
        host_paths=_disk_paths(args),
    )
    heartbeat.start()
    runtime.heartbeat = heartbeat
    runtime.service.heartbeat = heartbeat


def _disk_paths(args: Any) -> list[str]:
    paths = [getattr(args, "working_dir", None), os.getenv("LOG_DIR") or os.getcwd()]
    return [p for p in paths if isinstance(p, str) and p]


def _start_probes(app: Any, args: Any, runtime: OpsRuntime) -> None:
    from lightrag.api.ops import probes

    specs = probes.build_specs(
        args,
        store=runtime.store,
        app_state=app.state,
        disk_paths=_disk_paths(args),
    )
    runner = probes.ProbeRunner(specs, store=runtime.store, holder=runtime.worker_id)

    async def cycle(loop: LeasedLoop) -> None:
        await runner.run_cycle(loop)

    loop = LeasedLoop(
        runtime.store if runtime.store is not None else LocalLeases(),
        probes.PROBE_LEASE,
        runtime.worker_id,
        cycle,
        ttl_s=probes.PROBE_LEASE_TTL_S,
        interval_s=probes.PROBE_INTERVAL_S,
        cycle_timeout_s=probes.PROBE_CYCLE_TIMEOUT_S,
        initial_delay_s=random.uniform(1.0, 5.0),
    )
    loop.start()
    runtime.loops.append(loop)
    runtime.probes = runner
    runtime.service.probes = runner
    logger.info(
        "Operations console: probing %d dependencies (%s)",
        sum(1 for s in specs if s.checkable),
        "leased" if runtime.store is not None else "this process only",
    )


def _start_run_tracking(app: Any, runtime: OpsRuntime) -> None:
    """Install this worker's run tracker (pipeline hooks write through it)."""
    from lightrag.api.ops import runs
    from lightrag.api.ops.pauses import _role_slots

    own = runtime.service.own_providers

    async def byo(workspace: str, slot: str) -> bool:
        if own is None:
            return False
        slots = await own.slots([workspace])
        return slot in slots.get(workspace, frozenset())

    tracker = runs.RunTracker(
        runtime.store,
        runtime.worker_id,
        byo_resolver=byo,
        role_slots=_role_slots,
    )
    tracker.start()
    runs.install(tracker)
    runtime.run_tracker = tracker
    runtime.service.run_tracking = True
    if runtime.heartbeat is not None:
        runtime.heartbeat.run_tracker = tracker
    logger.info("Operations console: run tracking on")


def _start_log_sink(app: Any, runtime: OpsRuntime) -> None:
    """Attach this worker's structured log handler and start its flusher."""
    from lightrag.api.ops import log_sink

    level = max(
        logging.DEBUG,
        log_sink.level_from(os.getenv("OPS_LOG_LEVEL"), log_sink.DEFAULT_LEVEL),
    )
    sink = log_sink.OpsLogSink(
        runtime.store,
        runtime.worker_id,
        level=level,
        workers=_process(app)["expected"],
    )
    sink.start()
    sink.attach()
    runtime.log_sink = sink
    if runtime.heartbeat is not None:
        runtime.heartbeat.log_sink = sink
    logger.info(
        "Operations console: storing %s+ log lines", logging.getLevelName(level)
    )


def _start_log_retention(runtime: OpsRuntime) -> None:
    from lightrag.api.ops import log_sink

    days = log_sink.retention_days()

    async def cycle(loop: LeasedLoop) -> None:
        await log_sink.retention_cycle(loop, store=runtime.store, days=days)

    loop = LeasedLoop(
        runtime.store,
        log_sink.RETENTION_LEASE,
        runtime.worker_id,
        cycle,
        ttl_s=log_sink.RETENTION_LEASE_TTL_S,
        interval_s=log_sink.RETENTION_INTERVAL_S,
        cycle_timeout_s=log_sink.RETENTION_CYCLE_TIMEOUT_S,
        initial_delay_s=random.uniform(60.0, 300.0),
    )
    loop.start()
    runtime.loops.append(loop)


def _start_metrics(runtime: OpsRuntime) -> None:
    """This worker's histograms and flusher, plus the leased hourly jobs."""
    from lightrag.api.ops import metrics

    bins = metrics.MetricBins(runtime.store, runtime.worker_id)
    bins.start()
    metrics.install(bins)
    runtime.metrics = bins

    async def rollup(loop: LeasedLoop) -> None:
        await metrics.rollup_cycle(loop, store=runtime.store)

    loop = LeasedLoop(
        runtime.store,
        metrics.ROLLUP_LEASE,
        runtime.worker_id,
        rollup,
        ttl_s=metrics.ROLLUP_LEASE_TTL_S,
        interval_s=metrics.ROLLUP_INTERVAL_S,
        cycle_timeout_s=metrics.ROLLUP_CYCLE_TIMEOUT_S,
        initial_delay_s=random.uniform(120.0, 420.0),
    )
    loop.start()
    runtime.loops.append(loop)
    logger.info("Operations console: performance metrics on")


def _start_baselines(app: Any, runtime: OpsRuntime) -> None:
    """Hourly 7-day same-hour LLM p95 baseline (needs Token Insights)."""
    from lightrag.api.ops import perf

    def usage_store() -> Any:
        insights = getattr(app.state, "insights", None)
        return getattr(insights, "store", None)

    async def cycle(loop: LeasedLoop) -> None:
        await perf.baseline_cycle(loop, store=runtime.store, usage_store=usage_store)

    loop = LeasedLoop(
        runtime.store,
        perf.BASELINE_LEASE,
        runtime.worker_id,
        cycle,
        ttl_s=perf.BASELINE_LEASE_TTL_S,
        interval_s=perf.BASELINE_INTERVAL_S,
        cycle_timeout_s=perf.BASELINE_CYCLE_TIMEOUT_S,
        initial_delay_s=random.uniform(60.0, 180.0),
    )
    loop.start()
    runtime.loops.append(loop)


def _start_reconcile(app: Any, runtime: OpsRuntime) -> None:
    from lightrag.api.ops import reconcile

    async def cycle(loop: LeasedLoop) -> None:
        await reconcile.attention_cycle(
            loop, service=runtime.service, process=lambda: _process(app)
        )

    loop = LeasedLoop(
        runtime.store,
        reconcile.RECONCILE_LEASE,
        runtime.worker_id,
        cycle,
        ttl_s=reconcile.RECONCILE_LEASE_TTL_S,
        interval_s=reconcile.RECONCILE_INTERVAL_S,
        cycle_timeout_s=reconcile.RECONCILE_CYCLE_TIMEOUT_S,
        initial_delay_s=random.uniform(5.0, 15.0),
    )
    loop.start()
    runtime.loops.append(loop)


async def _start(app: Any, rag: Any, args: Any) -> OpsRuntime:
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
    service.worker_id = worker_id
    app.state.ops = runtime
    if env_flag("OPS_ENABLED"):
        # Each part is optional: a failure leaves the rest running.
        try:
            _start_heartbeat(app, rag, args, runtime)
        except Exception:
            logger.error("Operations console heartbeat could not start", exc_info=True)
        if env_flag("OPS_PROBES_ENABLED"):
            try:
                _start_probes(app, args, runtime)
            except Exception:
                logger.error("Operations console probes could not start", exc_info=True)
        else:
            logger.info(
                "Operations console: probes disabled by OPS_PROBES_ENABLED=false"
            )
    if store is None:
        return runtime
    if env_flag("OPS_RUN_TRACKING"):
        try:
            _start_run_tracking(app, runtime)
        except Exception:
            logger.error(
                "Operations console run tracking could not start", exc_info=True
            )
    else:
        logger.info(
            "Operations console: run tracking disabled by OPS_RUN_TRACKING=false"
        )
    try:
        _start_reconcile(app, runtime)
    except Exception:
        logger.error("Operations console attention loop could not start", exc_info=True)
    if env_flag("OPS_LOG_SINK", default=False):
        try:
            _start_log_sink(app, runtime)
        except Exception:
            logger.error("Operations console log sink could not start", exc_info=True)
    else:
        logger.info(
            "Operations console: log sink off (OPS_LOG_SINK=false); log views "
            "show stored lines only"
        )
    try:
        _start_log_retention(runtime)
    except Exception:
        logger.error("Operations console log retention could not start", exc_info=True)
    if env_flag("OPS_METRICS_ENABLED"):
        try:
            _start_metrics(runtime)
        except Exception:
            logger.error("Operations console metrics could not start", exc_info=True)
        try:
            _start_baselines(app, runtime)
        except Exception:
            logger.error(
                "Operations console LLM baselines could not start", exc_info=True
            )
    else:
        logger.info(
            "Operations console: performance metrics disabled by "
            "OPS_METRICS_ENABLED=false"
        )

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
    if runtime.log_sink is not None:
        # Detached first, so nothing logged while stopping is buffered.
        try:
            await runtime.log_sink.stop()
        except Exception:
            logger.debug("Ops log sink did not stop cleanly", exc_info=True)
    if runtime.metrics is not None:
        from lightrag.api.ops import metrics

        if metrics.current() is runtime.metrics:
            metrics.install(None)
        try:
            await runtime.metrics.stop()
        except Exception:
            logger.debug("Ops metrics did not stop cleanly", exc_info=True)
    if runtime.run_tracker is not None:
        from lightrag.api.ops import runs

        if runs.tracker() is runtime.run_tracker:
            runs.install(None)
        try:
            await runtime.run_tracker.stop()
        except Exception:
            logger.debug("Ops run tracker did not stop cleanly", exc_info=True)
    if runtime.heartbeat is not None:
        try:
            await runtime.heartbeat.stop()
        except Exception:
            logger.debug("Ops heartbeat did not stop cleanly", exc_info=True)
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
    if runtime.probes is not None:
        runtime.probes.close()
    if runtime.store is not None:
        try:
            await runtime.store.finalize()
        except Exception:
            logger.debug("Ops store did not close cleanly", exc_info=True)
    app.state.ops = None
