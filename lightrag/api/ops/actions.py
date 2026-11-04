"""Admin actions of the Operations console, and the audited reveal.

Every action (cancel, run now, drain the email queue, release a stale busy
slot) follows the same sequence (§3.9):

1. ``require_admin`` at the router, ``require_live_admin`` (and
   ``require_operator`` where marked), a JSON body and the CSRF checks: done
   by the route before anything here runs.
2. The per-workspace action lease ``action:<ws>`` (60 s; holder, actor,
   action). Another holder means ``in_progress_by``, and nothing else
   happens (no audit row: nothing was attempted).
3. The intent audit row, written synchronously. When it cannot be written
   the action is refused with 503 and nothing changes.
4. The preconditions, checked again on the server.
5. The ``pipeline_status`` lock, only through
   :func:`~lightrag.kg.shared_storage.try_acquire_namespace_lock` with a
   2 s timeout, for the shortest possible time and with no I/O inside it. A
   timeout is the ``lock_unavailable`` result (audited like any other).
6. The result audit row.
7. At most 10 actions per minute per actor (checked first, host-wide).

Business outcomes (refusals included) are ``result`` codes in a 200 body;
HTTP errors are reserved for bad input, permissions, the rate limit and an
unavailable database. A no-op returns its cause (``not_busy``,
``nothing_to_process``, ``nothing_pending``).

The reveal (``POST /reveal``) is not a state change: it takes no lease and
no lock, but it is rate limited, refused without its intent audit row, reads
one doc-status row by primary key and returns the value still run through
the redactor (secrets, emails, IC and phone numbers stay masked; only the
file-name masking is lifted).

All of it is off unless ``OPS_ACTIONS_ENABLED`` is set (the routes answer
404 then).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from lightrag.api.ops import runs as run_hooks
from lightrag.api.ops.audit import actor_of, write_audit
from lightrag.api.ops.heartbeat import process_alive
from lightrag.api.ops.reconcile import _epoch, owner_check
from lightrag.api.ops.redact import redact
from lightrag.api.ops.service import (
    OpsDataUnavailable,
    OpsNotFound,
    OpsService,
    _job_start_iso,
    validate_workspace,
)
from lightrag.kg import shared_storage

logger = logging.getLogger("lightrag.api.ops")

ACTIONS_FLAG = "OPS_ACTIONS_ENABLED"
PIPELINE_NAMESPACE = "pipeline_status"
LEASE_TTL_S = 60.0
LOCK_TIMEOUT_S = 2.0
RATE_LIMIT = 10
RATE_WINDOW_S = 60.0
DRAIN_INTERVAL_S = 60.0
# The owner of a busy slot is dead only after this long without a heartbeat
# (and OS confirmation). A slot without owner keys is never released.
OWNER_STALE_S = 60.0
WORKERS_LIMIT = 500
REASON_MIN = 3
REASON_MAX = 512
REVEAL_KINDS = ("doc_file_path", "doc_error", "job_name")
REVEAL_MAX_CHARS = 2048
RELEASE_KEYS = (
    "busy",
    "destructive_busy",
    "request_pending",
    "run_id",
    "owner_worker_id",
    "owner_pid",
    "heartbeat_at",
    "job_start",
    "job_name",
)
# Compared under the lock with the snapshot whose owner was confirmed dead:
# any change means the slot was taken again (or its owner is alive).
RELEASE_CAS_KEYS = (
    "run_id",
    "owner_worker_id",
    "owner_pid",
    "heartbeat_at",
    "job_start",
    "job_name",
)
RELEASED_BY = "Released by admin {actor}"
CANCEL_MESSAGE = "Pipeline cancellation requested by an administrator"

# Strong references to the pipeline runs started by "Run pipeline now".
_run_tasks: set[asyncio.Task] = set()


def actions_enabled() -> bool:
    """``OPS_ACTIONS_ENABLED`` (default off)."""
    value = os.getenv(ACTIONS_FLAG)
    if value is None or not value.strip():
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


class ActionRefused(Exception):
    """An HTTP-level refusal: ``status`` and a fixed ``code``."""

    def __init__(self, status: int, code: str, *, retry_after: Optional[float] = None):
        super().__init__(code)
        self.status = status
        self.code = code
        self.retry_after = retry_after


def clean_reason(reason: Any) -> str:
    """The reason as stored: trimmed, redacted, 3 to 512 characters."""
    text = redact(str(reason or "").strip(), REASON_MAX)
    if len(text) < REASON_MIN:
        raise ActionRefused(400, "invalid_reason")
    return text


@dataclass
class ActionCall:
    """One action request, after the route's guards."""

    user: Any
    request: Any
    action: str
    workspace: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None


def _result(code: str, **extra: Any) -> dict[str, Any]:
    return {"result": code, **extra}


def _cas_value(key: str, value: Any) -> Any:
    """A slot key as the release CAS compares it (times normalised to UTC)."""
    if key == "job_start":
        return _job_start_iso(value)
    return value or None


def _lock_unavailable(waited_s: float) -> dict[str, Any]:
    # A lock that stays held is usually held by a worker that died inside
    # the critical section: only a restart frees it.
    return _result(
        "lock_unavailable",
        waited_s=round(waited_s, 3),
        guidance="restart_container",
    )


class OpsActions:
    """The console's actions over one worker's :class:`OpsService`."""

    def __init__(
        self,
        service: OpsService,
        *,
        lock_timeout_s: float = LOCK_TIMEOUT_S,
        clock: Callable[[], float] = time.time,
        host: Optional[str] = None,
        alive: Callable[[Any, Any], bool] = process_alive,
        markers: Optional[Callable[[], dict[int, float]]] = None,
    ):
        self.service = service
        self.lock_timeout_s = lock_timeout_s
        self.clock = clock
        self.host = host or socket.gethostname()
        self.alive = alive
        self._markers = markers or run_hooks.read_exit_markers

    # -- plumbing -------------------------------------------------------------
    @property
    def store(self) -> Any:
        if not self.service.available:
            raise ActionRefused(503, "data_unavailable")
        return self.service.store

    async def check_rate(self, user: Any) -> None:
        """Step 7, checked before anything else: 10 per minute per actor,
        host-wide (counted in the ops KV)."""
        actor = (actor_of(user)["actor"] or "-").lower()
        try:
            allowed, retry_after = await self.store.rate_hit(
                f"ratelimit:actions:{actor}", RATE_LIMIT, RATE_WINDOW_S, self.clock()
            )
        except ActionRefused:
            raise
        except Exception as e:
            logger.warning("Ops action rate limit unavailable (%s)", type(e).__name__)
            raise ActionRefused(503, "data_unavailable") from e
        if not allowed:
            raise ActionRefused(429, "rate_limited", retry_after=max(1.0, retry_after))

    async def _audit(self, call: ActionCall, phase: str, **fields: Any) -> int:
        return await write_audit(
            self.store,
            user=call.user,
            action=call.action,
            phase=phase,
            request=call.request,
            workspace=call.workspace,
            target=call.target or call.workspace,
            reason=call.reason,
            **fields,
        )

    async def _result_audit(
        self, call: ActionCall, intent_id: int, outcome: str, extra: dict[str, Any]
    ) -> Optional[int]:
        try:
            return await self._audit(
                call,
                "result",
                params={**call.params, **extra, "intent_id": intent_id},
                outcome=outcome,
            )
        except Exception as e:
            # The action already happened; its intent row records the attempt.
            logger.warning(
                "Ops %s result audit failed (%s)", call.action, type(e).__name__
            )
            return None

    async def perform(
        self, call: ActionCall, body: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Steps 2-7 around ``body`` (steps 4-5, returns ``{"result": ...}``)."""
        store = self.store
        await self.check_rate(call.user)
        key = f"action:{call.workspace}"
        holder = uuid4().hex
        identity = actor_of(call.user)
        try:
            taken, current = await store.claim_action(
                key,
                holder,
                LEASE_TTL_S,
                {
                    "actor": identity["actor"],
                    "action": call.action,
                    "since": self.clock(),
                },
            )
        except Exception as e:
            logger.warning("Ops action lease unavailable (%s)", type(e).__name__)
            raise ActionRefused(503, "data_unavailable") from e
        if not taken:
            current = current or {}
            return _result(
                "in_progress_by",
                in_progress_by={
                    "actor": current.get("actor"),
                    "action": current.get("action"),
                    "since": run_hooks.iso(current.get("since")),
                },
                audit_id=None,
            )
        try:
            try:
                intent_id = await self._audit(call, "intent", params=call.params)
            except Exception as e:
                logger.warning(
                    "Ops %s refused: intent audit failed (%s)",
                    call.action,
                    type(e).__name__,
                )
                raise ActionRefused(503, "audit_unavailable") from e
            try:
                outcome = await body()
            except ActionRefused as e:
                await self._result_audit(call, intent_id, e.code, {})
                raise
            except asyncio.CancelledError:
                await asyncio.shield(
                    self._result_audit(call, intent_id, "cancelled", {})
                )
                raise
            except Exception:
                await self._result_audit(call, intent_id, "error", {})
                raise
            code = str(outcome.get("result") or "ok")
            audit_extra = {
                k: v
                for k, v in outcome.items()
                if k in ("would_process", "pending", "released", "how", "owner_state")
            }
            result_id = await self._result_audit(call, intent_id, code, audit_extra)
            logger.info(
                "Ops action %s on %s by %s: %s",
                call.action,
                call.workspace,
                identity["actor"],
                code,
            )
            return {**outcome, "audit_id": intent_id, "result_audit_id": result_id}
        finally:
            self.service.cache.forget(("workspace", call.workspace))
            try:
                await store.release_action(key, holder)
            except Exception as e:
                # It expires on its own after LEASE_TTL_S.
                logger.warning("Ops action lease release failed (%s)", type(e).__name__)

    # -- shared pieces ---------------------------------------------------------
    async def _counts(self, workspace: str) -> dict[str, int]:
        """Q2 for one workspace: what a run would pick up."""
        if not self.service.docs_available:
            raise ActionRefused(503, "data_unavailable")
        raw = await self.service._q(lambda: self.store.workspace_counts(workspace))
        statuses = raw.get("statuses") or {}
        cancelled = int(raw.get("cancelled") or 0)
        in_flight = sum(
            int(statuses.get(s) or 0) for s in ("parsing", "analyzing", "processing")
        )
        return {
            "pending": int(statuses.get("pending") or 0),
            # User-cancelled documents are failed documents (and retried).
            "failed": int(statuses.get("failed") or 0) + cancelled,
            "cancelled": cancelled,
            "in_flight": in_flight,
        }

    @staticmethod
    def _total(counts: dict[str, int]) -> int:
        return counts["pending"] + counts["failed"] + counts["in_flight"]

    def _pause_active(self, pause: Any) -> bool:
        return (
            isinstance(pause, dict)
            and bool(pause)
            and self.clock() < float(pause.get("retry_at") or 0)
        )

    def _runner(self, workspace: str) -> tuple[str, Any]:
        """``("default", rag)`` for the server's own workspace, ``("manager",
        manager)`` for another one; refuses when this server cannot run it."""
        state = self.service.app_state
        data_rag = self.service._data_rag()
        namespace = self.service.namespace_of(workspace)
        if data_rag is not None and namespace == (
            getattr(data_rag, "workspace", None) or ""
        ):
            return "default", data_rag
        manager = getattr(state, "workspace_manager", None)
        if manager is None:
            raise ActionRefused(404, "workspace_not_served")
        return "manager", manager

    # -- cancel ------------------------------------------------------------------
    async def cancel(self, call: ActionCall) -> dict[str, Any]:
        """Soft cancel, as ``POST /documents/cancel_pipeline`` does it, without
        loading the workspace."""
        namespace = self.service.namespace_of(call.workspace)

        async def body() -> dict[str, Any]:
            async with shared_storage.try_acquire_namespace_lock(
                PIPELINE_NAMESPACE, namespace, timeout=self.lock_timeout_s
            ) as lock:
                if not lock.acquired:
                    return _lock_unavailable(lock.waited_s)
                data = shared_storage.existing_namespace_data(
                    PIPELINE_NAMESPACE, namespace
                )
                if data is None or not data.get("busy", False):
                    return _result("not_busy")
                data["cancellation_requested"] = True
                data["latest_message"] = CANCEL_MESSAGE
                history = data.get("history_messages")
                if history is not None:
                    history.append(CANCEL_MESSAGE)
            return _result("cancellation_requested")

        return await self.perform(call, body)

    # -- run pipeline now ----------------------------------------------------------
    async def run_preview(self, workspace: str) -> dict[str, Any]:
        """The dry run: counts and live flags; changes nothing."""
        counts = await self._counts(workspace)
        snap, responsive = await self.service.reader.snapshot(
            self.service.namespace_of(workspace),
            ("busy", "destructive_busy", "provider_rejection"),
        )
        snap = snap or {}
        pause = snap.get("provider_rejection")
        paused = self._pause_active(pause)
        return {
            "dry_run": True,
            "would_process": counts,
            "total": self._total(counts),
            "live": {
                "busy": bool(snap.get("busy")),
                "destructive_busy": bool(snap.get("destructive_busy")),
                "paused": paused,
                "pause_retry_at": run_hooks.iso(pause.get("retry_at"))
                if paused
                else None,
            },
            "manager": "ok" if responsive else "unresponsive",
        }

    async def run(self, call: ActionCall, override_pause: bool) -> dict[str, Any]:
        """Start the pipeline for one workspace in the background.

        It retries every pending and failed document of the workspace,
        user-cancelled ones included (the dialog copy says so).
        """
        namespace = self.service.namespace_of(call.workspace)

        async def body() -> dict[str, Any]:
            counts = await self._counts(call.workspace)
            if self._total(counts) == 0:
                return _result("nothing_to_process", would_process=counts)
            kind, runner = self._runner(call.workspace)
            # A refusal is decided on a snapshot first, so it loads nothing;
            # it is decided again under the lock below.
            snap, _ = await self.service.reader.snapshot(
                namespace, ("busy", "destructive_busy", "provider_rejection")
            )
            refusal = self._run_refusal(snap, counts, override_pause)
            if refusal is not None:
                return refusal
            # Load the workspace before the lock (it is I/O, and the one
            # explicit single-workspace instantiation the console allows): once
            # the pause is overridden under the lock, the run must start, so
            # nothing that can fail may come between the two.
            if kind == "default":
                rag = runner
            else:
                from lightrag.api.workspace_manager import WorkspaceDeletionInProgress

                try:
                    rag = await runner.get_instance(call.workspace)
                except WorkspaceDeletionInProgress:
                    return _result(
                        "refused_busy",
                        reason="workspace_deleting",
                        would_process=counts,
                    )
            overridden = False
            async with shared_storage.try_acquire_namespace_lock(
                PIPELINE_NAMESPACE, namespace, timeout=self.lock_timeout_s
            ) as lock:
                if not lock.acquired:
                    return _lock_unavailable(lock.waited_s)
                data = shared_storage.existing_namespace_data(
                    PIPELINE_NAMESPACE, namespace
                )
                refusal = self._run_refusal(data, counts, override_pause)
                if refusal is not None:
                    return refusal
                pause = data.get("provider_rejection") if data is not None else None
                if self._pause_active(pause):
                    # Retry the provider now: the next document becomes the
                    # probe; a rejection pauses again, a success clears it.
                    # Replace the dict (Manager-backed state).
                    data["provider_rejection"] = {
                        **pause,
                        "retry_at": self.clock(),
                        "probe_doc": None,
                    }
                    overridden = True
            # Right after the lock and with nothing that can fail in between:
            # the run task takes the slot itself (or queues "process again").
            self._start_pipeline(rag)
            return _result("started", would_process=counts, pause_overridden=overridden)

        return await self.perform(call, body)

    def _run_refusal(
        self, data: Any, counts: dict[str, int], override_pause: bool
    ) -> Optional[dict[str, Any]]:
        """The refusal of "Run now" for these live flags, or None."""
        if not data:
            return None
        if data.get("destructive_busy", False):
            return _result("refused_busy", reason="destructive", would_process=counts)
        if data.get("busy", False):
            return _result("refused_busy", reason="processing", would_process=counts)
        pause = data.get("provider_rejection")
        if self._pause_active(pause) and not override_pause:
            return _result(
                "refused_paused",
                pause_retry_at=run_hooks.iso(pause.get("retry_at")),
                would_process=counts,
            )
        return None

    @staticmethod
    def _start_pipeline(rag: Any) -> None:
        from lightrag.usage import run_in_usage_scope

        async def run() -> None:
            try:
                await run_in_usage_scope(
                    rag.apipeline_process_enqueue_documents(), source="admin"
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("Pipeline run started by an admin failed", exc_info=True)

        # Runs started here are labelled as started by an admin.
        with run_hooks.run_origin(trigger="admin"):
            task = asyncio.create_task(run(), name="ops-admin-run")
        _run_tasks.add(task)
        task.add_done_callback(_run_tasks.discard)

    # -- drain the email queue now ---------------------------------------------------
    async def drain_email(self, call: ActionCall) -> dict[str, Any]:
        """Ask the drainer (any process) to retry now; start one here if this
        process has none. Never takes the queue's flock and never touches
        queued or ``.bad`` files."""
        queue_workspace = self.service.namespace_of(call.workspace)

        async def body() -> dict[str, Any]:
            queue = getattr(self.service.app_state, "email_resync_queue", None)
            if queue is None:
                return _result("queue_unavailable")
            try:
                pending = await asyncio.to_thread(queue.pending_count, queue_workspace)
            except ValueError:
                return _result("queue_unavailable")
            if not pending:
                return _result("nothing_pending", pending=0)
            now = self.clock()
            claimed = await self.service._q(
                lambda: self.store.kv_put_if_older(
                    f"kick:{queue_workspace}",
                    {"at": now, "worker_id": self.service.worker_id},
                    now - DRAIN_INTERVAL_S,
                )
            )
            if not claimed:
                return _result(
                    "rate_limited", pending=pending, retry_after=DRAIN_INTERVAL_S
                )
            if queue.kick(queue_workspace):
                return _result("kicked", pending=pending)
            queue.ensure_draining(queue_workspace)
            return _result("started_local", pending=pending)

        return await self.perform(call, body)

    # -- release a stale busy slot (Gunicorn only) ------------------------------------
    async def _owner_state(
        self, workspace: str, snap: dict[str, Any]
    ) -> tuple[str, Optional[str]]:
        """``(state, how)`` of the busy slot's owner, confirmed outside the lock:
        ``dead`` (with how), ``alive``, ``unresponsive`` or ``unknown``."""
        store = self.store
        now = self.clock()
        heartbeat = _epoch(snap.get("heartbeat_at"))
        owner_worker_id = snap.get("owner_worker_id")
        if owner_worker_id:
            if heartbeat is not None and now - heartbeat < OWNER_STALE_S:
                return "alive", None
            workers = await self.service._q(lambda: store.workers(WORKERS_LIMIT))
            worker = next(
                (w for w in workers if w.get("worker_id") == owner_worker_id), None
            )
            markers = {}
            try:
                markers = self._markers()
            except Exception:
                markers = {}
            # The Manager holding this slot belongs to this Gunicorn master, so
            # its owner ran on this host.
            return owner_check(
                worker=worker,
                pid=snap.get("owner_pid"),
                worker_id=owner_worker_id,
                since=_epoch(snap.get("job_start")),
                my_worker_id=self.service.worker_id,
                host=self.host,
                markers=markers,
                stale_s=OWNER_STALE_S,
                alive=self.alive,
                owner_host=self.host,
            )
        # No owner: run tracking is off (or its hook failed when the slot
        # was taken). No signal lists every workspace a live worker is
        # running (heartbeats report only the busiest instances, and an
        # evicted instance can still be running), so death cannot be
        # confirmed and the slot is never released (§3.9 fallback, narrowed).
        return "unknown", "no_owner_keys"

    async def release_busy(
        self,
        call: ActionCall,
        *,
        mode: str,
        expected_run_id: Optional[str],
        expected_job_start: Optional[str],
    ) -> dict[str, Any]:
        """Clear ``busy`` left by a dead worker (Gunicorn only).

        Never touches ``scanning``, ``scanning_exclusive``, ``pending_enqueues``
        or ``request_pending``; when a "process again" request was queued the
        result offers "Run now".
        """
        namespace = self.service.namespace_of(call.workspace)
        actor = actor_of(call.user)["actor"]

        def refused(reason: str, **extra: Any) -> dict[str, Any]:
            return _result(reason, released=False, reason=reason, **extra)

        async def body() -> dict[str, Any]:
            if mode != "gunicorn" or not shared_storage.is_multiprocess():
                return refused("single_process")
            snap, responsive = await self.service.reader.snapshot(
                namespace, RELEASE_KEYS
            )
            if not responsive:
                return refused("lock_unavailable", guidance="restart_container")
            if not snap or not snap.get("busy"):
                return refused("not_busy")
            expected_start = _job_start_iso(expected_job_start)
            if (snap.get("run_id") or None) != (
                expected_run_id or None
            ) or _job_start_iso(snap.get("job_start")) != expected_start:
                # The slot the operator looked at is not the one there now.
                return refused("cas_mismatch")
            state, how = await self._owner_state(call.workspace, snap)
            if state in ("alive", "unresponsive"):
                # A live but frozen worker needs a restart, not a release.
                return refused("owner_alive", owner_state=state)
            if state != "dead":
                return refused("owner_unknown", owner_state=state)
            async with shared_storage.try_acquire_namespace_lock(
                PIPELINE_NAMESPACE, namespace, timeout=self.lock_timeout_s
            ) as lock:
                if not lock.acquired:
                    return {
                        **_lock_unavailable(lock.waited_s),
                        "released": False,
                        "reason": "lock_unavailable",
                    }
                data = shared_storage.existing_namespace_data(
                    PIPELINE_NAMESPACE, namespace
                )
                if data is None or not data.get("busy", False):
                    return refused("not_busy")
                # CAS against the snapshot whose owner was confirmed dead (it
                # already matched the operator's run_id and job_start). The
                # owner keys and heartbeat_at are rewritten by every new
                # owner, so a slot taken again since then never matches.
                if any(
                    _cas_value(key, data.get(key)) != _cas_value(key, snap.get(key))
                    for key in RELEASE_CAS_KEYS
                ):
                    return refused("cas_mismatch")
                run_id = data.get("run_id")
                request_pending = bool(data.get("request_pending", False))
                data.update(
                    {
                        "busy": False,
                        "destructive_busy": False,
                        "cancellation_requested": False,
                        "run_id": None,
                        "owner_worker_id": None,
                        "owner_pid": None,
                        "latest_message": RELEASED_BY.format(actor=actor),
                    }
                )
            # Outside the lock: the run row and the log line.
            orphaned = False
            if run_id:
                try:
                    orphaned = bool(
                        await self.store.runs_orphan(
                            [run_id], {run_id: "admin_release"}
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Released run could not be marked orphaned (%s)",
                        type(e).__name__,
                    )
            logger.warning(
                "Ops: admin %s released the busy pipeline slot of %s (owner %s)",
                actor,
                call.workspace,
                how,
            )
            return _result(
                "released",
                released=True,
                how=how,
                run_now_offered=request_pending,
                run_marked_orphaned=orphaned,
            )

        return await self.perform(call, body)

    # -- reveal ----------------------------------------------------------------------
    async def reveal(
        self, call: ActionCall, kind: str, item_id: Optional[str]
    ) -> dict[str, Any]:
        """One field, unmasked but still secret-redacted. Audited; refused
        without its intent row."""
        if kind not in REVEAL_KINDS:
            raise ActionRefused(400, "invalid_kind")
        store = self.store
        await self.check_rate(call.user)
        try:
            intent_id = await write_audit(
                store,
                user=call.user,
                action=call.action,
                phase="intent",
                request=call.request,
                workspace=call.workspace,
                target=call.target,
                params=call.params,
                reason=call.reason,
            )
        except Exception as e:
            logger.warning("Ops reveal refused: audit failed (%s)", type(e).__name__)
            raise ActionRefused(503, "audit_unavailable") from e
        try:
            value = await self._reveal_value(call.workspace, kind, item_id)
            outcome = "ok" if value is not None else "empty"
        except OpsNotFound:
            await self._result_audit(call, intent_id, "not_found", {})
            raise
        except Exception:
            await self._result_audit(call, intent_id, "error", {})
            raise
        result_id = await self._result_audit(call, intent_id, outcome, {})
        return {
            "kind": kind,
            "value": value,
            "audit_id": intent_id,
            "result_audit_id": result_id,
        }

    async def _reveal_value(
        self, workspace: str, kind: str, item_id: Optional[str]
    ) -> Optional[str]:
        if kind == "job_name":
            snap, _ = await self.service.reader.snapshot(
                self.service.namespace_of(workspace), ("job_name", "run_id")
            )
            if not snap or not snap.get("job_name"):
                raise OpsNotFound(workspace)
            if item_id and snap.get("run_id") != item_id:
                # The job changed since the page was drawn.
                raise OpsNotFound(workspace)
            raw = snap.get("job_name")
        else:
            if not item_id:
                raise ActionRefused(400, "invalid_id")
            if not self.service.docs_available:
                raise ActionRefused(503, "data_unavailable")
            row = await self.service._q(lambda: self.store.document(workspace, item_id))
            if row is None:
                raise OpsNotFound(item_id)
            raw = row.get("file_path") if kind == "doc_file_path" else row.get("error")
        if raw is None or raw == "":
            return None
        return redact(raw, REVEAL_MAX_CHARS, mask_paths=False)


def build_call(
    *,
    user: Any,
    request: Any,
    action: str,
    workspace: str,
    reason: Any,
    params: Optional[dict[str, Any]] = None,
    target: Optional[str] = None,
) -> ActionCall:
    """Validate the workspace and the reason (400 on bad input)."""
    try:
        validate_workspace(workspace)
    except Exception as e:
        raise ActionRefused(400, "invalid_workspace") from e
    return ActionCall(
        user=user,
        request=request,
        action=action,
        workspace=workspace,
        reason=clean_reason(reason),
        params=dict(params or {}),
        target=target,
    )


__all__ = [
    "ACTIONS_FLAG",
    "ActionCall",
    "ActionRefused",
    "OpsActions",
    "OpsDataUnavailable",
    "actions_enabled",
    "build_call",
]
