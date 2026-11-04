"""Run tracking for the Operations console.

A *run* is one holder of a workspace's pipeline: a processing loop, a clear
or delete job, an email re-sync batch, a scan. Each run gets a ``run_id`` and
a row in ``LIGHTRAG_OPS_RUNS`` (start, progress, finish); each document that
fails gets a classified row in ``LIGHTRAG_OPS_DOC_FAILURES`` and a
``metadata.failure`` entry on its doc-status row.

The hooks in this module are called from the ingestion pipeline and the
document routes, so they follow strict rules:

- they are synchronous and never await: rows go into bounded in-memory
  buffers and a per-process flusher writes them every 2 s;
- they never raise: every hook swallows its own errors (counted, logged once);
- they take no lock. Owner keys for ``pipeline_status`` are *returned* as a
  dict that the caller merges into the update it already makes under the
  existing ``pipeline_status`` lock (:func:`owner_fields`,
  :func:`clear_fields`);
- when a buffer is full the oldest entry is dropped and counted; a slow or
  unreachable Postgres only delays the flusher, never the pipeline.

Tracking runs only in a server process that installed a :class:`RunTracker`
(the ops runtime does, unless ``OPS_RUN_TRACKING=false``); everywhere else
(library use, tests, tracking switched off) every hook is a no-op and no
owner key is written.

Stdlib only, like the rest of :mod:`lightrag.api.ops` that core code imports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterator, Optional
from uuid import uuid4

from lightrag.api.ops.classify import FailureClass, classify_failure

logger = logging.getLogger("lightrag.api.ops")

RUN_KINDS = (
    "process",
    "email_batch",
    "delete",
    "clear",
    "email_delete",
    "scan",
    "startup_resume",
    "admin",
)
RUN_STATUSES = ("running", "succeeded", "failed", "cancelled", "paused", "orphaned")
FINAL_STATUSES = ("succeeded", "failed", "cancelled", "paused")
TRIGGERS = (
    "upload",
    "email",
    "resync",
    "scan",
    "insert",
    "reprocess",
    "startup",
    "admin",
    "api",
    "provider",
    "other",
)

FLUSH_INTERVAL_S = 2.0
WRITE_TIMEOUT_S = 10.0
# ``heartbeat_at`` in pipeline_status is refreshed at most this often per
# workspace (each write is a Manager round trip under Gunicorn).
HEARTBEAT_EVERY_S = 5.0
EMAIL_INGEST_EVERY_S = 60.0
KICK_READ_TIMEOUT_S = 2.0
MAX_RUNS = 2000
MAX_FAILURES = 5000
MAX_KV = 2000
MAX_PAUSE_EVENTS = 1000
LABEL_MAX = 128
SUMMARY_MAX = 512
MESSAGE_MAX = 1024
_RAW_MESSAGE_MAX = 4096
EXIT_MARKER_SUBDIR = os.path.join("ops", "exited")
# Workspaces deleted in this process are remembered this long (rows buffered
# before the deletion are dropped, never written back).
FORGET_KEEP_S = 3600.0
# Postgres error classes caused by the data itself (22 data exception, 23
# integrity violation): retrying the same batch can never succeed.
_DATA_ERROR_SQLSTATES = ("22", "23")

# Owner keys written into pipeline_status (see AGENTS.md).
OWNER_KEYS = ("run_id", "owner_worker_id", "owner_pid", "heartbeat_at")
SCAN_OWNER_KEYS = ("scan_owner_worker_id", "scan_started_at")
# Per-worker shares of the ``pending_enqueues`` counter (OQ-12): written next
# to the counter, in the same locked block; the counter stays authoritative.
PENDING_OWNERS_KEY = "pending_enqueue_owners"

_TRACK_TRIGGERS = {
    "upload": "upload",
    "insert": "insert",
    "enqueue": "insert",
    "email": "email",
    "scan": "scan",
    "error": "reprocess",
}
_JOB_FILES = re.compile(r"^(?P<prefix>.*?)(?:\.\.\.)?\[(?P<count>\d+) files\]$", re.S)

_tracker: Optional["RunTracker"] = None
_origin: ContextVar[Optional[tuple[Optional[str], Optional[str]]]] = ContextVar(
    "lightrag_run_origin", default=None
)
_warned: set[str] = set()


def _now() -> float:
    return time.time()


def iso(epoch: Optional[float]) -> Optional[str]:
    if not isinstance(epoch, (int, float)) or epoch <= 0:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def job_label(job_name: Any) -> Optional[str]:
    """A job name safe to show: the file name a batch name embeds is masked.

    Processing batches are named ``<first 20 chars of the first path>...[N
    files]``; the cut-off name may hold spaces and no extension, so it is
    masked as a whole rather than left to the token redactor.
    """
    from lightrag.api.ops.redact import mask_path, redact

    if not isinstance(job_name, str) or not job_name or job_name == "-":
        return None
    try:
        match = _JOB_FILES.match(job_name)
        if match:
            prefix = match.group("prefix").strip()
            label = (
                "unknown source"
                if prefix in ("", "unknown_source")
                else mask_path(prefix)
            )
            return f"{label} [{match.group('count')} files]"[:LABEL_MAX]
        return redact(job_name, LABEL_MAX)
    except Exception:
        return None


def workspace_of(rag: Any) -> str:
    """The workspace name doc-status rows carry for ``rag``.

    Storage may resolve an unnamed workspace to its own default name
    (``default`` on Postgres), and ops rows must join with doc-status rows,
    so the storage's name wins over the instance's. Never raises.
    """
    try:
        name = getattr(getattr(rag, "doc_status", None), "workspace", None)
        if isinstance(name, str) and name:
            return name
        name = getattr(rag, "workspace", "")
        return name if isinstance(name, str) else ""
    except Exception:
        return ""


def trigger_from_track_id(track_id: Any) -> Optional[str]:
    if not isinstance(track_id, str) or "_" not in track_id:
        return None
    return _TRACK_TRIGGERS.get(track_id.split("_", 1)[0].lower())


def _key_workspace(key: str) -> Optional[str]:
    """The workspace a buffered KV key belongs to (``resync:<ws>``,
    ``pause:<ws>:<slot>``), or None for a global key."""
    if key.startswith("resync:"):
        return key[len("resync:") :]
    if key.startswith("pause:"):
        return key[len("pause:") :].rsplit(":", 1)[0]
    return None


def _is_data_error(error: BaseException) -> bool:
    """Whether a write failed on the rows themselves, not on the database."""
    if isinstance(error, (ValueError, TypeError, UnicodeError)):
        return True
    state = getattr(error, "sqlstate", None)
    return isinstance(state, str) and state[:2] in _DATA_ERROR_SQLSTATES


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _warned:
        return
    _warned.add(key)
    try:
        logger.warning(message, *args)
    except Exception:  # pragma: no cover - logging must not raise
        pass


@dataclass
class _PendingFailure:
    row: dict[str, Any]
    raw_message: str
    file_path: Optional[str]


class RunTracker:
    """Per-process buffers and flusher for run rows, failure rows and KV."""

    def __init__(
        self,
        store: Any,
        worker_id: str,
        *,
        pid: Optional[int] = None,
        clock: Callable[[], float] = _now,
        interval_s: float = FLUSH_INTERVAL_S,
        write_timeout_s: float = WRITE_TIMEOUT_S,
        max_runs: int = MAX_RUNS,
        max_failures: int = MAX_FAILURES,
        max_kv: int = MAX_KV,
        byo_resolver: Optional[Callable[[str, str], Awaitable[bool]]] = None,
        role_slots: Optional[Callable[[], dict[str, str]]] = None,
    ):
        self.store = store
        self.worker_id = worker_id
        self.pid = pid if pid is not None else os.getpid()
        # The host the heartbeat records for this worker; run rows carry it
        # so the reconciler never checks a PID that lives on another host.
        self.host = socket.gethostname()[:255]
        self.clock = clock
        self.interval_s = interval_s
        self.write_timeout_s = write_timeout_s
        self.max_runs = max_runs
        self.max_failures = max_failures
        self.max_kv = max_kv
        self.byo_resolver = byo_resolver
        self.role_slots = role_slots
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._dirty: set[str] = set()
        self._failures: deque[_PendingFailure] = deque()
        self._kv: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        # When each buffered KV value was first buffered (deletion checks).
        self._kv_at: dict[str, float] = {}
        self._pauses: deque[dict[str, Any]] = deque()
        # Workspaces deleted while rows for them may still be buffered.
        self._forgotten: dict[str, float] = {}
        self._heartbeat_at: dict[str, float] = {}
        self._email_ingest_at = 0.0
        self._task: Optional[asyncio.Task] = None
        self._flush_lock: Optional[asyncio.Lock] = None
        # Counters (exported in the heartbeat snapshot).
        self.dropped_runs = 0
        self.dropped_failures = 0
        self.dropped_kv = 0
        self.dropped_deleted = 0
        self.dropped_bad_rows = 0
        self.hook_errors = 0
        self.flush_failures = 0
        self.runs_written = 0
        self.failures_written = 0
        self.last_flush_ok_at: Optional[float] = None
        self.last_flush_error: Optional[str] = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="ops-run-tracker")

    async def stop(self, flush_timeout_s: float = 5.0) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            await asyncio.wait_for(self.flush_once(), timeout=flush_timeout_s)
        except Exception as e:
            logger.debug("Ops run tracker final flush failed (%s)", type(e).__name__)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            try:
                await self.flush_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover - flush_once guards itself
                logger.debug("Ops run tracker flush failed (%s)", type(e).__name__)

    def stats(self) -> dict[str, Any]:
        return {
            "runs_buffered": len(self._dirty),
            "runs_tracked": len(self._runs),
            "failures_buffered": len(self._failures),
            "kv_buffered": len(self._kv) + len(self._pauses),
            "dropped_runs": self.dropped_runs,
            "dropped_failures": self.dropped_failures,
            "dropped_kv": self.dropped_kv,
            "dropped_deleted": self.dropped_deleted,
            "dropped_bad_rows": self.dropped_bad_rows,
            "hook_errors": self.hook_errors,
            "flush_failures": self.flush_failures,
            "runs_written": self.runs_written,
            "failures_written": self.failures_written,
            "last_flush_ok_at": iso(self.last_flush_ok_at),
            "last_flush_error": self.last_flush_error,
        }

    # -- buffers ---------------------------------------------------------------
    def _mark(self, run_id: str) -> None:
        self._dirty.add(run_id)
        self._runs.move_to_end(run_id)
        while len(self._runs) > self.max_runs:
            oldest = next(iter(self._runs))
            # Finished runs go first; a running one only when nothing else is left.
            victim = next(
                (rid for rid, r in self._runs.items() if r["status"] != "running"),
                oldest,
            )
            self._runs.pop(victim, None)
            self._dirty.discard(victim)
            self.dropped_runs += 1

    def _put_kv(self, key: str, op: str, value: Any) -> None:
        current = self._kv.get(key)
        if (
            op in ("merge", "merge_existing")
            and current is not None
            and isinstance(current[1], dict)
        ):
            # A put followed by a merge stays a put of the merged value.
            value = {**current[1], **value}
            op = current[0]
        self._kv[key] = (op, value)
        self._kv.move_to_end(key)
        self._kv_at.setdefault(key, self.clock())
        while len(self._kv) > self.max_kv:
            dropped, _ = self._kv.popitem(last=False)
            self._kv_at.pop(dropped, None)
            self.dropped_kv += 1

    # -- run lifecycle (called through the module hooks) ------------------------
    def begin(
        self,
        run_id: str,
        *,
        workspace: str,
        kind: str,
        trigger: Optional[str],
        label: Optional[str] = None,
        docs_total: int = 0,
        request_id: Optional[str] = None,
    ) -> None:
        now = self.clock()
        self._runs[run_id] = {
            "run_id": run_id,
            "workspace": workspace or "",
            "kind": kind if kind in RUN_KINDS else "admin",
            "trigger": (trigger or "other")[:32],
            "status": "running",
            "worker_id": self.worker_id,
            "pid": self.pid,
            "started_at": now,
            "heartbeat_at": now,
            "progress_at": now,
            "finished_at": None,
            "docs_total": max(0, int(docs_total or 0)),
            "docs_done": 0,
            "docs_failed": 0,
            "stage": None,
            "label": label[:LABEL_MAX] if label else None,
            "error_kind": None,
            "error_summary": None,
            "request_id": (request_id or None) and request_id[:64],
            "host": self.host,
        }
        self._mark(run_id)

    def update(
        self,
        run_id: str,
        *,
        add_total: int = 0,
        docs_total: Optional[int] = None,
        done: int = 0,
        failed: int = 0,
        stage: Optional[str] = None,
        label: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        run = self._runs.get(run_id)
        if run is None or run["status"] != "running":
            return None
        now = self.clock()
        if docs_total is not None:
            run["docs_total"] = max(0, int(docs_total))
        if add_total:
            run["docs_total"] += max(0, int(add_total))
        if done or failed:
            run["docs_done"] += max(0, int(done)) + max(0, int(failed))
            run["docs_failed"] += max(0, int(failed))
            run["progress_at"] = now
        if stage:
            run["stage"] = stage[:16]
            run["progress_at"] = now
        if label:
            run["label"] = label[:LABEL_MAX]
        if trigger:
            run["trigger"] = trigger[:32]
        run["heartbeat_at"] = now
        self._mark(run_id)
        return run

    def heartbeat_fields(self, workspace: str) -> dict[str, Any]:
        """``heartbeat_at`` for pipeline_status, at most every 5 s per workspace."""
        now = self.clock()
        last = self._heartbeat_at.get(workspace, 0.0)
        if now - last < HEARTBEAT_EVERY_S:
            return {}
        self._heartbeat_at[workspace] = now
        if len(self._heartbeat_at) > self.max_runs:
            self._heartbeat_at.pop(next(iter(self._heartbeat_at)))
        return {"heartbeat_at": iso(now)}

    def end(
        self,
        run_id: str,
        outcome: str,
        *,
        error: Optional[BaseException] = None,
        docs_done: Optional[int] = None,
    ) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return
        now = self.clock()
        run["status"] = outcome if outcome in FINAL_STATUSES else "failed"
        run["finished_at"] = now
        run["heartbeat_at"] = now
        if docs_done is not None:
            run["docs_done"] = max(run["docs_done"], int(docs_done))
        if error is not None:
            failure = classify_failure(error)
            run["error_kind"] = failure.error_kind
            try:
                from lightrag.api.ops.redact import redact

                run["error_summary"] = redact(str(error), SUMMARY_MAX) or None
            except Exception:
                run["error_summary"] = None
        self._mark(run_id)
        self._heartbeat_at.pop(run.get("workspace") or "", None)

    def add_failure(
        self,
        *,
        workspace: str,
        doc_id: str,
        track_id: Optional[str],
        run_id: Optional[str],
        stage: str,
        failure: FailureClass,
        message: str,
        attempt: int,
        model: Optional[str],
        file_path: Optional[str],
    ) -> None:
        row = {
            "ts": self.clock(),
            "workspace": workspace or "",
            "doc_id": doc_id[:255],
            "track_id": (track_id or None) and str(track_id)[:255],
            "run_id": run_id,
            "stage": stage[:16],
            "error_kind": failure.error_kind,
            "http_status": failure.http_status,
            "exc_type": failure.exc_type or None,
            "attempt": max(1, int(attempt)),
            "model": (model or None) and str(model)[:255],
        }
        self._failures.append(
            _PendingFailure(row, (message or "")[:_RAW_MESSAGE_MAX], file_path)
        )
        while len(self._failures) > self.max_failures:
            self._failures.popleft()
            self.dropped_failures += 1

    def add_pause_event(self, event: dict[str, Any]) -> None:
        self._pauses.append(event)
        while len(self._pauses) > MAX_PAUSE_EVENTS:
            self._pauses.popleft()
            self.dropped_kv += 1

    # -- workspace deletion ------------------------------------------------------
    def forget(self, workspace: str) -> None:
        """Drop everything buffered for a deleted ``workspace``.

        Rows created before now are never written for it again (a flush that
        fails keeps its batch, so the check repeats at every flush); rows of
        a workspace created later under the same name are kept.
        """
        now = self.clock()
        self._forgotten[workspace] = now
        cutoff = now - FORGET_KEEP_S
        for name in [w for w, at in self._forgotten.items() if at < cutoff]:
            self._forgotten.pop(name, None)
        self._drop_workspace(workspace, now)

    async def forget_flushed(self, workspace: str, timeout_s: float = 30.0) -> None:
        """:meth:`forget` once no flush is writing (so none of the dropped
        rows is in flight); without waiting longer than ``timeout_s``."""
        guard = self._guard()
        try:
            await asyncio.wait_for(guard.acquire(), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.forget(workspace)
            return
        try:
            self.forget(workspace)
        finally:
            guard.release()

    def _drop_workspace(self, workspace: str, before: float) -> int:
        dropped = 0
        for run_id in [
            rid
            for rid, run in self._runs.items()
            if run["workspace"] == workspace and run["started_at"] <= before
        ]:
            self._runs.pop(run_id, None)
            self._dirty.discard(run_id)
            dropped += 1
        kept = deque(
            f
            for f in self._failures
            if f.row["workspace"] != workspace or f.row["ts"] > before
        )
        dropped += len(self._failures) - len(kept)
        self._failures = kept
        pauses = deque(
            e
            for e in self._pauses
            if e.get("workspace") != workspace or e.get("at", 0) > before
        )
        dropped += len(self._pauses) - len(pauses)
        self._pauses = pauses
        for key in [
            k
            for k in self._kv
            if _key_workspace(k) == workspace and self._kv_at.get(k, 0) <= before
        ]:
            self._kv.pop(key, None)
            self._kv_at.pop(key, None)
            dropped += 1
        self.dropped_deleted += dropped
        return dropped

    def _buffered_workspaces(self) -> set[str]:
        names = {
            self._runs[rid]["workspace"] for rid in self._dirty if rid in self._runs
        }
        names.update(f.row["workspace"] for f in self._failures)
        names.update(e.get("workspace") or "" for e in self._pauses)
        names.update(w for w in map(_key_workspace, self._kv) if w is not None)
        return names

    async def _drop_deleted(self) -> None:
        """Drop rows of workspaces deleted here or by another worker since
        they were buffered (KV tombstones written by ``forget_workspace``)."""
        names = self._buffered_workspaces()
        if not names:
            return
        from lightrag.api.ops.store import deleted_key

        keys = {deleted_key(name): name for name in names}
        try:
            found = await self._timed(self.store.kv_get_many(list(keys)))
        except asyncio.CancelledError:
            raise
        except Exception:
            found = {}  # the writes below fail too and keep their rows
        deleted = dict(self._forgotten)
        for key, value in (found or {}).items():
            at = value.get("at") if isinstance(value, dict) else None
            name = keys.get(key)
            if name is not None and isinstance(at, (int, float)):
                deleted[name] = max(float(at), deleted.get(name, 0.0))
        for name in names:
            if name in deleted:
                self._drop_workspace(name, deleted[name])

    # -- flushing ----------------------------------------------------------------
    def _guard(self) -> asyncio.Lock:
        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()
        return self._flush_lock

    async def _timed(self, coro: Awaitable[Any]) -> Any:
        return await asyncio.wait_for(coro, timeout=self.write_timeout_s)

    async def flush_once(self) -> bool:
        """Write what is buffered. Returns True when everything was written."""
        async with self._guard():
            ok = True
            await self._drop_deleted()
            ok &= await self._flush_runs()
            ok &= await self._flush_failures()
            ok &= await self._flush_pauses()
            ok &= await self._flush_kv()
            if ok:
                self.last_flush_ok_at = self.clock()
                self.last_flush_error = None
            return ok

    def _failed(self, what: str, error: BaseException) -> None:
        self.flush_failures += 1
        self.last_flush_error = type(error).__name__
        _warn_once(
            f"flush:{what}:{type(error).__name__}",
            "Ops run tracking could not write %s (%s); keeping them buffered",
            what,
            type(error).__name__,
        )

    async def _flush_runs(self) -> bool:
        if not self._dirty:
            return True
        ids = [rid for rid in self._runs if rid in self._dirty]
        rows = [dict(self._runs[rid]) for rid in ids]
        self._dirty.difference_update(ids)
        try:
            await self._timed(self.store.runs_write(rows))
        except asyncio.CancelledError:
            self._dirty.update(rid for rid in ids if rid in self._runs)
            raise
        except Exception as e:
            if not _is_data_error(e):
                self._dirty.update(rid for rid in ids if rid in self._runs)
                self._failed("runs", e)
                return False
            # One bad row must not hold back the others: write them one by
            # one and drop (and count) the rows the database rejects.
            rows = await self._write_each(self.store.runs_write, rows, "runs", e)
            if rows is None:
                self._dirty.update(rid for rid in ids if rid in self._runs)
                return False
        self.runs_written += len(rows)
        written = {row["run_id"]: row for row in rows}
        for rid, row in written.items():
            current = self._runs.get(rid)
            # Finished and written (and not changed since): forget it.
            if (
                current is not None
                and current["status"] != "running"
                and rid not in self._dirty
                and current == row
            ):
                self._runs.pop(rid, None)
        return True

    async def _flush_failures(self) -> bool:
        if not self._failures:
            return True
        from lightrag.api.ops.redact import redact_pipeline

        batch = list(self._failures)
        self._failures.clear()
        rows = []
        for item in batch:
            row = dict(item.row)
            try:
                row["message"] = (
                    redact_pipeline(item.raw_message, MESSAGE_MAX, (item.file_path,))
                    or None
                )
            except Exception:
                row["message"] = None
            rows.append(row)
        try:
            await self._timed(self.store.doc_failures_write(rows))
        except asyncio.CancelledError:
            self._failures.extendleft(reversed(batch))
            raise
        except Exception as e:
            if _is_data_error(e):
                written = await self._write_each(
                    self.store.doc_failures_write, rows, "failure rows", e
                )
                if written is not None:
                    self.failures_written += len(written)
                    return True
            self._failures.extendleft(reversed(batch))
            while len(self._failures) > self.max_failures:
                self._failures.popleft()
                self.dropped_failures += 1
            self._failed("failure rows", e)
            return False
        self.failures_written += len(rows)
        return True

    async def _write_each(
        self,
        write: Callable[[list[dict[str, Any]]], Awaitable[Any]],
        rows: list[dict[str, Any]],
        what: str,
        error: BaseException,
    ) -> Optional[list[dict[str, Any]]]:
        """Write ``rows`` one at a time after a data error in their batch.

        Returns the rows written (rejected ones are dropped and counted), or
        None when a write fails for another reason (the batch is kept).
        """
        written: list[dict[str, Any]] = []
        for row in rows:
            try:
                await self._timed(write([row]))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not _is_data_error(e):
                    self._failed(what, e)
                    return None
                self.dropped_bad_rows += 1
                continue
            written.append(row)
        _warn_once(
            f"bad-row:{what}:{type(error).__name__}",
            "Ops run tracking dropped %s the database rejected (%s)",
            what,
            type(error).__name__,
        )
        return written

    async def _flush_pauses(self) -> bool:
        if not self._pauses:
            return True
        from lightrag.api.ops.redact import redact

        slots = {}
        if self.role_slots is not None:
            try:
                slots = self.role_slots() or {}
            except Exception:
                slots = {}
        events = list(self._pauses)
        self._pauses.clear()
        for event in events:
            workspace = event.get("workspace") or ""
            role = event.get("role")
            slot = slots.get(role, role or "unknown")
            if event["op"] == "set":
                byo = None
                if self.byo_resolver is not None:
                    try:
                        byo = bool(
                            await asyncio.wait_for(
                                self.byo_resolver(workspace, slot), timeout=5.0
                            )
                        )
                    except Exception:
                        byo = None
                previous = event.get("previous_role")
                if previous and slots.get(previous, previous) != slot:
                    old = slots.get(previous, previous)
                    self._put_kv(
                        f"pause:{workspace}:{old}",
                        "merge_existing",
                        {"cleared_at": event["at"], "cleared_by": "replaced"},
                    )
                self._put_kv(
                    f"pause:{workspace}:{slot}",
                    "put",
                    {
                        "workspace": workspace,
                        "slot": slot,
                        "role": role,
                        "kind": event.get("kind"),
                        "status_code": event.get("status_code"),
                        "byo": byo,
                        "since": event.get("since"),
                        "retry_at": event.get("retry_at"),
                        "message": redact(event.get("message"), 300) or None,
                        "set_at": event["at"],
                        "cleared_at": None,
                    },
                )
            elif event["op"] == "retry":
                self._put_kv(
                    f"pause:{workspace}:{slot}",
                    "merge_existing",
                    {"retry_at": event.get("retry_at")},
                )
            else:  # clear
                keys = (
                    [f"pause:{workspace}:{slot}"]
                    if role
                    else [
                        f"pause:{workspace}:{name}"
                        for name in sorted({*slots.values(), "unknown"})
                    ]
                )
                for key in keys:
                    self._put_kv(
                        key,
                        "merge_existing",
                        {
                            "cleared_at": event["at"],
                            "cleared_by": event.get("reason") or "cleared",
                        },
                    )
        return True

    async def _flush_kv(self) -> bool:
        if not self._kv:
            return True
        batch = list(self._kv.items())
        batch_at = dict(self._kv_at)
        self._kv.clear()
        self._kv_at.clear()
        puts = [(k, v) for k, (op, v) in batch if op == "put"]
        merges = {k: v for k, (op, v) in batch if op == "merge"}
        existing = {k: v for k, (op, v) in batch if op == "merge_existing"}
        try:
            for key, value in puts:
                await self._timed(self.store.kv_put(key, value))
            if merges:
                await self._timed(self.store.kv_merge_many(merges))
            if existing:
                await self._timed(self.store.kv_merge_existing(existing))
        except asyncio.CancelledError:
            self._restore_kv(batch, batch_at)
            raise
        except Exception as e:
            self._restore_kv(batch, batch_at)
            self._failed("ops values", e)
            return False
        return True

    def _restore_kv(
        self, batch: list[tuple[str, tuple[str, Any]]], batch_at: dict[str, float]
    ) -> None:
        # Newer values buffered meanwhile win over the ones that failed.
        newer, newer_at = self._kv, self._kv_at
        now = self.clock()
        self._kv = OrderedDict(batch)
        self._kv_at = {k: batch_at.get(k, now) for k, _ in batch}
        for key, (op, value) in newer.items():
            fresh = key not in self._kv_at
            self._put_kv(key, op, value)
            if fresh and key in newer_at:
                self._kv_at[key] = newer_at[key]
        while len(self._kv) > self.max_kv:
            dropped, _ = self._kv.popitem(last=False)
            self._kv_at.pop(dropped, None)
            self.dropped_kv += 1

    # -- reads used by drainers ---------------------------------------------------
    async def kick_at(self, workspace: str) -> Optional[float]:
        value = await asyncio.wait_for(
            self.store.kv_get(f"kick:{workspace}"), timeout=KICK_READ_TIMEOUT_S
        )
        if isinstance(value, dict):
            value = value.get("at")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None


# -- installation ------------------------------------------------------------------


def install(tracker: Optional[RunTracker]) -> None:
    """Make ``tracker`` this process's run tracker (``None`` switches tracking off)."""
    global _tracker
    _tracker = tracker


def tracker() -> Optional[RunTracker]:
    """This process's tracker; never one inherited across a fork."""
    current = _tracker
    if current is None or current.pid != os.getpid():
        return None
    return current


def enabled() -> bool:
    return tracker() is not None


def _hook(default: Any = None) -> Callable:
    """Decorator: a hook never raises; any error is counted and returns ``default``."""

    def wrap(fn: Callable) -> Callable:
        def inner(*args: Any, **kwargs: Any) -> Any:
            current = tracker()
            if current is None:
                return default() if callable(default) else default
            try:
                return fn(current, *args, **kwargs)
            except Exception as e:
                current.hook_errors += 1
                _warn_once(
                    f"hook:{fn.__name__}:{type(e).__name__}",
                    "Ops run tracking hook %s failed (%s); the pipeline is unaffected",
                    fn.__name__,
                    type(e).__name__,
                )
                return default() if callable(default) else default

        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner

    return wrap


# -- origin of the next run (who asked) ------------------------------------------------


@contextmanager
def run_origin(
    kind: Optional[str] = None, trigger: Optional[str] = None
) -> Iterator[None]:
    """Label runs started in this context (and tasks created in it)."""
    token = None
    try:
        token = _origin.set((kind, trigger))
    except Exception:  # pragma: no cover - ContextVar.set does not fail
        pass
    try:
        yield
    finally:
        if token is not None:
            try:
                _origin.reset(token)
            except ValueError:
                pass


def current_origin() -> tuple[Optional[str], Optional[str]]:
    value = _origin.get()
    return value if value else (None, None)


# -- hooks: owner keys ---------------------------------------------------------------------


@_hook(None)
def new_run_id(t: RunTracker) -> Optional[str]:
    """A fresh run ID, or None when tracking is off. Pure: no I/O."""
    return uuid4().hex


@_hook(dict)
def owner_fields(
    t: RunTracker, run_id: Optional[str], *, owner_only: bool = False
) -> dict[str, Any]:
    """Owner keys to merge into the busy-acquire update (under the existing lock).

    ``owner_only`` records the owner of a slot taken without a run (a
    workspace purge), so it is never reported as busy without an owner.
    """
    if not run_id and not owner_only:
        return {}
    return {
        "run_id": run_id or None,
        "owner_worker_id": t.worker_id,
        "owner_pid": t.pid,
        "heartbeat_at": iso(t.clock()),
    }


@_hook(dict)
def clear_fields(t: RunTracker) -> dict[str, Any]:
    """Owner keys reset, to merge into the update that releases ``busy``."""
    return {key: None for key in OWNER_KEYS}


@_hook(dict)
def scan_owner_fields(t: RunTracker) -> dict[str, Any]:
    return {"scan_owner_worker_id": t.worker_id, "scan_started_at": iso(t.clock())}


@_hook(dict)
def scan_clear_fields(t: RunTracker) -> dict[str, Any]:
    return {key: None for key in SCAN_OWNER_KEYS}


def pending_owners(value: Any) -> dict[str, int]:
    """A ``pending_enqueue_owners`` value as ``{worker_id: count > 0}``."""
    if not isinstance(value, dict):
        return {}
    return {
        str(worker): int(count)
        for worker, count in value.items()
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }


@_hook(dict)
def pending_reserve_fields(t: RunTracker, current: Any) -> dict[str, Any]:
    """This worker's share of ``pending_enqueues`` plus one, to merge into the
    update that increments the counter (under the existing lock).

    ``current`` is the map read in the same locked block. The whole value is
    replaced, never mutated in place (Manager-backed shared state).
    """
    owners = pending_owners(current)
    owners[t.worker_id] = owners.get(t.worker_id, 0) + 1
    return {PENDING_OWNERS_KEY: owners}


@_hook(dict)
def pending_release_fields(t: RunTracker, current: Any) -> dict[str, Any]:
    """This worker's share minus one (clamped; the entry goes at zero), to
    merge into the update that decrements the counter."""
    owners = pending_owners(current)
    count = owners.get(t.worker_id, 0)
    if count <= 0:
        return {}
    if count == 1:
        owners.pop(t.worker_id)
    else:
        owners[t.worker_id] = count - 1
    return {PENDING_OWNERS_KEY: owners}


# -- hooks: run lifecycle ------------------------------------------------------------------


@_hook(None)
def start(
    t: RunTracker,
    run_id: Optional[str],
    *,
    workspace: str,
    kind: str,
    trigger: Optional[str] = None,
    label: Optional[str] = None,
    docs_total: int = 0,
) -> Optional[str]:
    """Record a run that just took the slot. ``kind``/``trigger`` default to
    the surrounding :func:`run_origin`."""
    if not run_id:
        return None
    origin_kind, origin_trigger = current_origin()
    request_id = None
    try:
        from lightrag.api.ops.context import current_request_id

        request_id = current_request_id()
    except Exception:
        pass
    t.begin(
        run_id,
        workspace=workspace,
        kind=origin_kind if (origin_kind and kind == "process") else kind,
        trigger=origin_trigger or trigger,
        label=label,
        docs_total=docs_total,
        request_id=request_id,
    )
    return run_id


@_hook(dict)
def progress(
    t: RunTracker,
    run_id: Optional[str],
    *,
    add_total: int = 0,
    docs_total: Optional[int] = None,
    done: int = 0,
    failed: int = 0,
    stage: Optional[str] = None,
    label: Optional[str] = None,
    trigger: Optional[str] = None,
    heartbeat: bool = False,
) -> dict[str, Any]:
    """Update a run.

    With ``heartbeat=True`` (callers that merge the result into their own
    ``pipeline_status`` update) it returns ``{"heartbeat_at": ...}`` when the
    per-workspace 5 s throttle allows a write, else ``{}``; other calls never
    use up the throttle and always return ``{}``.
    """
    if not run_id:
        return {}
    run = t.update(
        run_id,
        add_total=add_total,
        docs_total=docs_total,
        done=done,
        failed=failed,
        stage=stage,
        label=label,
        trigger=trigger,
    )
    if run is None or not heartbeat:
        return {}
    return t.heartbeat_fields(run["workspace"])


@_hook(None)
def finish(
    t: RunTracker,
    run_id: Optional[str],
    outcome: str,
    *,
    error: Optional[BaseException] = None,
    docs_done: Optional[int] = None,
) -> None:
    """Record the end of a run (succeeded, failed, cancelled or paused)."""
    if run_id:
        t.end(run_id, outcome, error=error, docs_done=docs_done)


@_hook(None)
def run_trigger_for(
    t: RunTracker, track_ids: Any, default: str = "other"
) -> Optional[str]:
    """Trigger of a processing run: the origin context, else the track IDs."""
    _, origin = current_origin()
    if origin:
        return origin
    for track_id in track_ids or ():
        trigger = trigger_from_track_id(track_id)
        if trigger:
            return trigger
    return default


# -- hooks: failures -----------------------------------------------------------------------


def _iso_now(t: RunTracker) -> Optional[str]:
    return iso(t.clock())


@_hook(None)
def doc_failure(
    t: RunTracker,
    *,
    workspace: str,
    doc_id: str,
    stage: str,
    error: Optional[BaseException],
    previous: Any = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    model: Any = None,
    file_path: Optional[str] = None,
    message: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Classify a document failure and buffer its history row.

    Returns the ``metadata.failure`` value to merge into the FAILED upsert
    (``None`` when tracking is off). ``attempt`` follows the failure carried
    over from the previous attempt (``previous``), so no read is needed.
    ``model`` may be a callable, evaluated only here.
    """
    failure = classify_failure(error)
    prior = previous if isinstance(previous, dict) else {}
    try:
        attempt = int(prior.get("attempt") or 0) + 1
    except (TypeError, ValueError):
        attempt = 1
    if callable(model):
        try:
            model = model()
        except Exception:
            model = None
    now_iso = _iso_now(t)
    text = message if message is not None else ("" if error is None else str(error))
    t.add_failure(
        workspace=workspace,
        doc_id=doc_id,
        track_id=track_id,
        run_id=run_id,
        stage=stage,
        failure=failure,
        message=text,
        attempt=attempt,
        model=model if isinstance(model, str) else None,
        file_path=file_path,
    )
    if run_id:
        t.update(run_id, failed=1, stage=stage)
    return {
        "stage": stage,
        "error_kind": failure.error_kind,
        "http_status": failure.http_status,
        "exc_type": failure.exc_type or None,
        "attempt": attempt,
        "first_failed_at": prior.get("first_failed_at") or now_iso,
        "failed_at": now_iso,
        "last_run_id": run_id,
        "model": model if isinstance(model, str) else None,
    }


@_hook(None)
def duplicate_failure(
    t: RunTracker, stage: str, run_id: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """``metadata.failure`` for a duplicate row (no failure history row)."""
    now_iso = _iso_now(t)
    return {
        "stage": stage,
        "error_kind": "duplicate",
        "http_status": None,
        "exc_type": None,
        "attempt": 1,
        "first_failed_at": now_iso,
        "failed_at": now_iso,
        "last_run_id": run_id,
        "model": None,
    }


@_hook(None)
def doc_done(t: RunTracker, run_id: Optional[str]) -> None:
    """One document of the run finished successfully."""
    if run_id:
        t.update(run_id, done=1)


# -- hooks: provider pauses, email queue, ingest ------------------------------------------


@_hook(None)
def pause_set(
    t: RunTracker,
    workspace: str,
    *,
    role: Optional[str],
    kind: Optional[str],
    status_code: Optional[int],
    since: Optional[float],
    retry_at: Optional[float],
    message: Optional[str] = None,
    previous_role: Optional[str] = None,
) -> None:
    t.add_pause_event(
        {
            "op": "set",
            "at": t.clock(),
            "workspace": workspace or "",
            "role": role,
            "kind": kind,
            "status_code": status_code,
            "since": since,
            "retry_at": retry_at,
            "message": (message or "")[:1000] or None,
            "previous_role": previous_role,
        }
    )


@_hook(None)
def pause_retry(
    t: RunTracker, workspace: str, *, role: Optional[str], retry_at: Optional[float]
) -> None:
    t.add_pause_event(
        {
            "op": "retry",
            "at": t.clock(),
            "workspace": workspace or "",
            "role": role,
            "retry_at": retry_at,
        }
    )


@_hook(None)
def pause_cleared(
    t: RunTracker, workspace: str, *, role: Optional[str], reason: str = "cleared"
) -> None:
    t.add_pause_event(
        {
            "op": "clear",
            "at": t.clock(),
            "workspace": workspace or "",
            "role": role,
            "reason": reason,
        }
    )


@_hook(None)
def resync_result(t: RunTracker, workspace: str, patch: dict[str, Any]) -> None:
    """Merge a drain result into KV ``resync:<ws>`` (keeps ``first_pending_at``)."""
    t._put_kv(f"resync:{workspace}", "merge", dict(patch))


@_hook(None)
def note_email_ingest(t: RunTracker) -> None:
    """KV ``email:last_ingest_at``, at most once a minute per process."""
    now = t.clock()
    if now - t._email_ingest_at < EMAIL_INGEST_EVERY_S:
        return
    t._email_ingest_at = now
    t._put_kv("email:last_ingest_at", "put", {"at": now, "worker_id": t.worker_id})


async def kick_at(workspace: str) -> Optional[float]:
    """When ``kick:<ws>`` was last written (any process), or None. Never raises."""
    current = tracker()
    if current is None:
        return None
    try:
        return await current.kick_at(workspace)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _warn_once(
            f"kick:{type(e).__name__}",
            "Ops kick check failed (%s); drainers keep their backoff",
            type(e).__name__,
        )
        return None


# -- child_exit markers (Gunicorn master) --------------------------------------------------


def exit_marker_dir(log_dir: Optional[str] = None) -> str:
    base = log_dir or os.getenv("LOG_DIR") or os.getcwd()
    return os.path.join(base, EXIT_MARKER_SUBDIR)


def write_exit_marker(log_dir: Optional[str], pid: int) -> Optional[str]:
    """Record that worker ``pid`` exited (``<pid>-<exit epoch>``). Never raises.

    Runs in the Gunicorn master's ``child_exit`` hook: synchronous and tiny.
    """
    try:
        directory = exit_marker_dir(log_dir)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        path = os.path.join(directory, f"{int(pid)}-{int(time.time())}")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
        return path
    except Exception:
        return None


def read_exit_markers(log_dir: Optional[str] = None) -> dict[int, float]:
    """Latest exit time per pid from the marker directory."""
    out: dict[int, float] = {}
    try:
        entries = os.scandir(exit_marker_dir(log_dir))
    except OSError:
        return out
    with entries:
        for entry in entries:
            pid_text, _, at_text = entry.name.partition("-")
            try:
                pid, at = int(pid_text), float(at_text)
            except ValueError:
                continue
            if at > out.get(pid, 0.0):
                out[pid] = at
    return out


def prune_exit_markers(
    log_dir: Optional[str] = None, max_age_s: float = 172800.0
) -> int:
    removed = 0
    cutoff = time.time() - max_age_s
    try:
        entries = os.scandir(exit_marker_dir(log_dir))
    except OSError:
        return 0
    with entries:
        for entry in entries:
            _, _, at_text = entry.name.partition("-")
            try:
                at = float(at_text)
            except ValueError:
                continue
            if at < cutoff:
                try:
                    os.unlink(entry.path)
                    removed += 1
                except OSError:
                    pass
    return removed
