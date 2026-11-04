"""Leased host-wide reconciliation of runs, busy flags and reservations.

One worker per host (the holder of ``ops:reconcile``) runs this cycle once a
minute:

1. **Orphaned runs.** A running run becomes ``orphaned`` only when its owner
   is confirmed dead: the owner worker's heartbeat is at least
   ``orphan_seconds`` (60 s) old **and** the operating system says the
   process is gone (``psutil``: the PID does not exist, or it belongs to a
   process that started at another time), or the Gunicorn master left a
   ``child_exit`` marker for it, or the worker recorded its own exit
   (graceful shutdown) or was superseded after such a check. A stale
   heartbeat on a live process is a wedged worker: it is reported
   (``unresponsive``), never orphaned. A PID is only ever checked on the
   host that owns it (run rows carry their host). Owners on another host
   are orphaned only when that whole host is gone: none of its workers has
   beaten for ``HOST_GONE_S`` (a container recreated by a deploy after a
   SIGKILL comes back under a new host name). How death was confirmed is
   stored with the run (``orphaned_by``).
2. **Shared-state checks** over at most 200 candidate workspaces (running
   runs, open pauses, in-flight documents by recency, waiting email
   re-syncs), read through the bounded snapshot reader:
   ``busy_without_run`` (``busy`` with no live run or owner, seen that way
   for ``busy_without_run_minutes`` by this reconciler) and, in multi-worker
   mode only, ``reservation_without_owner`` (``scanning`` whose owner is
   dead or unknown, ``pending_enqueues`` held for longer than
   ``reservation_minutes`` while no live run holds the workspace). Results
   go to KV ``busy_without_run`` and ``reservations``. Persisted provider
   pauses that shared state no longer holds are closed.
3. **Retention**, hourly: runs after 30 days, failure rows after 90 days,
   ``child_exit`` markers after two days.

The attention rules read these KV results; they are evaluated (and the
attention snapshot written) by the alerts evaluator alone, under its own
lease ``ops:alerts`` (:mod:`lightrag.api.ops.alerts`).

Reconciliation **never** clears ``busy`` or a reservation: it only reports
them, with restart guidance.
"""

from __future__ import annotations

import logging
import socket
from typing import Any, Callable, Optional

from lightrag.api.ops import runs as run_hooks
from lightrag.api.ops.heartbeat import process_alive
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.service import (
    BUSY_WITHOUT_RUN_KEY,
    RESERVATIONS_KEY,
)

logger = logging.getLogger("lightrag.api.ops")

RECONCILE_LEASE = "ops:reconcile"
RECONCILE_LEASE_TTL_S = 120.0
RECONCILE_INTERVAL_S = 60.0
# The cycle runs a handful of cached, 3 s-bounded reads and one bounded
# snapshot batch (2 s).
RECONCILE_CYCLE_TIMEOUT_S = 90.0

MAX_CANDIDATES = 200
# A host none of whose workers has beaten for this long is gone (its owners'
# runs are orphaned from another host). Far above any wedged-loop pause.
HOST_GONE_S = 900.0
MAX_RUN_LOOKUPS = 20
RUNNING_LIMIT = 500
OPEN_PAUSES_LIMIT = 500
CLEANUP_EVERY_S = 3600.0
RUN_RETENTION_DAYS = 30
FAILURE_RETENTION_DAYS = 90
# A marker written this long before a worker's recorded start is not its own.
MARKER_MARGIN_S = 5.0
RECONCILE_KEYS = (
    "busy",
    "destructive_busy",
    "run_id",
    "owner_worker_id",
    "owner_pid",
    "job_name",
    "job_start",
    "heartbeat_at",
    "scanning",
    "scanning_exclusive",
    "scan_owner_worker_id",
    "scan_started_at",
    "pending_enqueues",
    "provider_rejection",
)

_state: dict[str, float] = {"last_cleanup": 0.0}


def _epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        try:
            from lightrag.api.ops.store import as_utc

            moment = as_utc(value)
            return moment.timestamp() if moment else None
        except Exception:
            return None
    if isinstance(value, str) and value:
        try:
            from datetime import datetime, timezone

            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment.timestamp()
        except ValueError:
            return None
    return None


def _death_reason(pid: Any) -> str:
    """How the OS confirmed ``pid`` gone: no such process, or another one."""
    try:
        import psutil

        if isinstance(pid, int) and pid > 0 and psutil.pid_exists(pid):
            return "create_time"
    except Exception:
        pass
    return "pid_gone"


def owner_check(
    *,
    worker: Optional[dict[str, Any]],
    pid: Any,
    worker_id: Optional[str],
    since: Optional[float],
    my_worker_id: Optional[str],
    host: str,
    markers: dict[int, float],
    stale_s: float,
    alive: Callable[[Any, Any], bool] = process_alive,
    owner_host: Optional[str] = None,
    owner_age_s: Optional[float] = None,
    host_ages: Optional[dict[str, float]] = None,
    gone_s: float = HOST_GONE_S,
) -> tuple[str, Optional[str]]:
    """``(state, how)``: ``alive``, ``unresponsive`` (stale but running),
    ``dead`` (confirmed; ``how`` says by what) or ``unknown`` (cannot be
    confirmed from this host).

    ``worker`` is the owner's row from ``LIGHTRAG_OPS_WORKERS`` (None when
    there is none); ``since`` is when the owned thing started (epoch);
    ``owner_host`` is the host recorded with the owned thing and
    ``owner_age_s`` the age of its own last sign of life; ``host_ages`` maps
    each host to the age of the freshest beat of any of its workers.
    """
    if worker_id and my_worker_id and worker_id == my_worker_id:
        return "alive", None
    if worker is not None:
        age = worker.get("heartbeat_age_s")
        if isinstance(age, (int, float)) and age < stale_s:
            return "alive", None
        # The process recorded its own exit (graceful shutdown, e.g. a
        # deploy that recreated the container), or a newer worker on the
        # same host superseded it after the OS confirmed it gone.
        if worker.get("exited_at") is not None:
            return "dead", "exited"
        if worker.get("superseded"):
            return "dead", "superseded"
        worker_host = worker.get("host") or owner_host or host
        if worker_host != host:
            return _remote_owner(worker_host, age, host_ages, gone_s)
        row_pid = worker.get("pid", pid)
        started = _epoch(worker.get("started_at")) or since
        marker = markers.get(row_pid) if isinstance(row_pid, int) else None
        if marker is not None and (
            started is None or marker >= started - MARKER_MARGIN_S
        ):
            return "dead", "exit_marker"
        if alive(row_pid, worker.get("pid_create_time")):
            return "unresponsive", None
        return "dead", _death_reason(row_pid)
    # No worker row (never written, or removed a day after its last beat).
    # A PID means nothing on another host, or on an unknown one.
    if not owner_host:
        return "unknown", None
    if owner_host != host:
        return _remote_owner(owner_host, owner_age_s, host_ages, gone_s)
    if not isinstance(pid, int) or pid <= 0:
        return "unknown", None
    marker = markers.get(pid)
    if marker is not None and (since is None or marker >= since - MARKER_MARGIN_S):
        return "dead", "exit_marker"
    if alive(pid, None):
        return "unresponsive", None
    return "dead", _death_reason(pid)


def _remote_owner(
    owner_host: str,
    age: Any,
    host_ages: Optional[dict[str, float]],
    gone_s: float,
) -> tuple[str, Optional[str]]:
    """An owner on another host: dead only when that host is gone."""
    if host_ages is None or not isinstance(age, (int, float)) or age < gone_s:
        return "unknown", None
    freshest = host_ages.get(owner_host)
    if freshest is None or freshest >= gone_s:
        return "dead", "host_gone"
    return "unknown", None


def owner_state(**kwargs: Any) -> str:
    """The state part of :func:`owner_check`."""
    return owner_check(**kwargs)[0]


def host_ages(workers: Any) -> dict[str, float]:
    """The freshest beat age per host, from worker rows."""
    out: dict[str, float] = {}
    for row in workers or ():
        name = row.get("host")
        age = row.get("heartbeat_age_s")
        if isinstance(name, str) and isinstance(age, (int, float)):
            out[name] = min(float(age), out.get(name, float("inf")))
    return out


def _worker_view(row: dict[str, Any]) -> dict[str, Any]:
    """A run row's ``worker_*`` columns as a worker row."""
    return {
        "host": row.get("worker_host"),
        "pid": row.get("pid"),
        "pid_create_time": row.get("worker_pid_create_time"),
        "started_at": row.get("worker_started_at"),
        "heartbeat_age_s": row.get("worker_heartbeat_age_s"),
        "exited_at": row.get("worker_exited_at"),
        "superseded": row.get("worker_superseded"),
    }


async def reconcile_runs(
    store: Any,
    *,
    my_worker_id: Optional[str],
    host: str,
    stale_s: float,
    markers: dict[int, float],
    alive: Callable[[Any, Any], bool] = process_alive,
    workers: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Orphan the running runs whose owner is confirmed dead.

    ``workers`` (all worker rows) lets owners on a host that is gone be
    orphaned; without it they stay unknown.
    """
    running = await store.running_runs(RUNNING_LIMIT)
    ages = host_ages(workers) if workers is not None else None
    dead: dict[str, Optional[str]] = {}
    unresponsive: list[str] = []
    for row in running:
        age = row.get("heartbeat_age_s")
        if row.get("worker_host") is None and row.get("worker_heartbeat_age_s") is None:
            worker = None
            # Without a worker row the run's own beat must be stale too.
            if not isinstance(age, (int, float)) or age < stale_s:
                continue
        else:
            worker = _worker_view(row)
        state, how = owner_check(
            worker=worker,
            pid=row.get("pid"),
            worker_id=row.get("worker_id"),
            since=_epoch(row.get("started_at")),
            my_worker_id=my_worker_id,
            host=host,
            markers=markers,
            stale_s=stale_s,
            alive=alive,
            owner_host=row.get("run_host"),
            owner_age_s=age,
            host_ages=ages,
        )
        if state == "dead":
            dead[row["run_id"]] = how
        elif state == "unresponsive":
            unresponsive.append(row["run_id"])
    orphaned = (
        await store.runs_orphan(list(dead), {k: v for k, v in dead.items() if v})
        if dead
        else []
    )
    if orphaned:
        logger.warning(
            "Ops reconcile: %d run(s) orphaned after their owner process exited",
            len(orphaned),
        )
    return {
        "running": running,
        "orphaned": orphaned,
        "unresponsive": unresponsive,
    }


async def _candidates(
    store: Any, service: Any, running: list[dict[str, Any]], pauses: list[dict]
) -> list[str]:
    ordered: list[str] = []

    def add(workspace: Any) -> None:
        if isinstance(workspace, str) and workspace not in ordered:
            ordered.append(workspace)

    for row in running:
        add(row.get("workspace"))
    for pause in pauses:
        add(pause.get("workspace"))
    if service.docs_available and len(ordered) < MAX_CANDIDATES:
        for row in await store.live_candidates(MAX_CANDIDATES):
            add(row.get("workspace"))
    try:
        queue = await service.queue_summary()
    except Exception:
        queue = {}
    for item in (queue or {}).get("items") or []:
        if isinstance(item, dict) and item.get("pending"):
            add(item.get("workspace"))
    if service.docs_available and len(ordered) < MAX_CANDIDATES:
        for workspace in await store.pending_candidates(MAX_CANDIDATES):
            add(workspace)
    return ordered[:MAX_CANDIDATES]


async def check_shared_state(
    store: Any,
    service: Any,
    *,
    running: Optional[list[dict[str, Any]]],
    workers: dict[str, dict[str, Any]],
    my_worker_id: Optional[str],
    host: str,
    markers: dict[int, float],
    thresholds: dict[str, Any],
    now: float,
    alive: Callable[[Any, Any], bool] = process_alive,
    reservations_checked: bool = True,
) -> dict[str, Any]:
    """busy-without-run, reservations without owner, stale persisted pauses.

    ``running`` is None when the running runs could not be read: busy flags
    cannot be matched to runs then, so the previous busy-without-run result
    is kept as it was. ``reservations_checked`` is False in single-process
    mode, where a reservation cannot outlive its owner.
    """
    pauses = await store.open_pauses(OPEN_PAUSES_LIMIT)
    workspaces = await _candidates(store, service, running or [], pauses)
    names = {ws: service.namespace_of(ws) for ws in workspaces}
    batch = await service.reader.snapshot_many(list(names.values()), RECONCILE_KEYS)
    previous = await store.kv_get(RESERVATIONS_KEY) or {}
    previous = previous if isinstance(previous, dict) else {}
    prior_busy = await store.kv_get(BUSY_WITHOUT_RUN_KEY) or {}
    prior_busy = prior_busy if isinstance(prior_busy, dict) else {}
    if not batch.responsive:
        return {
            "responsive": False,
            "busy_without_run": {**prior_busy, "responsive": False},
            "reservations": {**previous, "responsive": False},
            "closed_pauses": 0,
        }
    seen_before = previous.get("seen") if isinstance(previous.get("seen"), dict) else {}
    busy_seen_before = (
        prior_busy.get("seen") if isinstance(prior_busy.get("seen"), dict) else {}
    )
    running_ids = {r["run_id"]: r for r in running or []}
    stale_s = float(thresholds.get("orphan_seconds") or 60)
    grace_s = float(thresholds.get("busy_without_run_minutes") or 2) * 60
    hold_s = float(thresholds.get("reservation_minutes") or 5) * 60

    def live_owner(worker_id: Any, since: Optional[float]) -> bool:
        """A worker ID whose process is not confirmed dead."""
        if not worker_id:
            return False
        state = owner_state(
            worker=workers.get(worker_id),
            pid=(workers.get(worker_id) or {}).get("pid"),
            worker_id=worker_id,
            since=since,
            my_worker_id=my_worker_id,
            host=host,
            markers=markers,
            stale_s=stale_s,
            alive=alive,
            owner_host=(workers.get(worker_id) or {}).get("host"),
        )
        return state != "dead" and (worker_id in workers or state == "alive")

    busy_items: list[dict[str, Any]] = []
    reservations: list[dict[str, Any]] = []
    seen: dict[str, dict[str, float]] = {}
    busy_seen: dict[str, dict[str, Any]] = {}
    lookups = 0
    for ws in workspaces:
        snap = batch.data.get(names[ws])
        if not snap:
            continue
        job_start = _epoch(snap.get("job_start"))
        run_id = snap.get("run_id")
        # The workspace's busy slot is held by a live run (or, for a run-less
        # slot such as a workspace purge, by a live owner).
        held_live = bool(
            snap.get("busy")
            and (
                (run_id and (running is None or run_id in running_ids))
                or (not run_id and live_owner(snap.get("owner_worker_id"), job_start))
            )
        )
        if running is not None and snap.get("busy") and not held_live:
            reason = None
            if not run_id:
                reason = "no_run"
            elif lookups < MAX_RUN_LOOKUPS:
                lookups += 1
                row = await store.run_get(run_id)
                if row is None:
                    reason = "run_unknown"
                elif row.get("status") not in ("running", "orphaned"):
                    # Finished but still busy: the release did not happen.
                    # (A running row missing from ``running`` started after
                    # that read or is beyond its limit: it owns the slot;
                    # an orphaned one is reported as orphaned.)
                    reason = "run_finished"
            if reason:
                key = run_id or ""
                prior = busy_seen_before.get(ws)
                first = (
                    float(prior.get("at"))
                    if isinstance(prior, dict)
                    and prior.get("key") == key
                    and isinstance(prior.get("at"), (int, float))
                    else now
                )
                busy_seen[ws] = {"key": key, "at": first}
                # Measured from when this reconciler first saw it: a
                # destructive slot never refreshes job_start.
                if now - first >= grace_s:
                    busy_items.append(
                        {
                            "workspace": ws,
                            "reason_code": reason,
                            "run_id": run_id or None,
                            "destructive": bool(snap.get("destructive_busy")),
                            "since": run_hooks.iso(first),
                            "age_s": now - first,
                            "owner_worker_id": snap.get("owner_worker_id"),
                            "job": run_hooks.job_label(snap.get("job_name")),
                        }
                    )
        elif running is None and isinstance(busy_seen_before.get(ws), dict):
            busy_seen[ws] = busy_seen_before[ws]
        if not reservations_checked:
            continue
        prior = seen_before.get(ws) if isinstance(seen_before.get(ws), dict) else {}
        marks: dict[str, float] = {}
        if snap.get("scanning"):
            first = float(prior.get("scanning") or now)
            marks["scanning"] = first
            owner = snap.get("scan_owner_worker_id")
            started = _epoch(snap.get("scan_started_at")) or first
            reason = None
            if owner:
                state = owner_state(
                    worker=workers.get(owner),
                    pid=(workers.get(owner) or {}).get("pid"),
                    worker_id=owner,
                    since=started,
                    my_worker_id=my_worker_id,
                    host=host,
                    markers=markers,
                    stale_s=stale_s,
                    alive=alive,
                    owner_host=(workers.get(owner) or {}).get("host"),
                )
                if state == "dead":
                    reason = "owner_dead"
                elif owner not in workers and now - first >= hold_s:
                    # No heartbeat row at all for the owner, for a while.
                    reason = "owner_unknown"
            elif now - first >= hold_s:
                reason = "no_owner"
            if reason:
                reservations.append(
                    {
                        "workspace": ws,
                        "kind": "scanning",
                        "scanning": True,
                        "reason_code": reason,
                        "owner_worker_id": owner or None,
                        "since": run_hooks.iso(started),
                        "age_s": max(0.0, now - started),
                        "count": None,
                        "pending_enqueues": None,
                        "exclusive": bool(snap.get("scanning_exclusive")),
                    }
                )
        pending = snap.get("pending_enqueues")
        # An upload or insert holds its reservation for its whole processing
        # run: only a hold with no live run behind it counts, and the hold
        # time restarts whenever a live run holds the workspace.
        if isinstance(pending, int) and pending > 0 and not held_live:
            first = float(prior.get("pending_enqueues") or now)
            marks["pending_enqueues"] = first
            if now - first >= hold_s:
                reservations.append(
                    {
                        "workspace": ws,
                        "kind": "pending_enqueues",
                        "scanning": False,
                        "reason_code": "held",
                        "owner_worker_id": None,
                        "since": run_hooks.iso(first),
                        "age_s": now - first,
                        "count": pending,
                        "pending_enqueues": pending,
                        "exclusive": False,
                    }
                )
        if marks:
            seen[ws] = marks
    closed = await _close_stale_pauses(store, pauses, batch, names, now)
    at = run_hooks.iso(now)
    capped = len(workspaces) >= MAX_CANDIDATES
    if running is None:
        busy_result = {**prior_busy, "seen": busy_seen, "responsive": True}
    else:
        busy_result = {
            "at": at,
            "at_epoch": now,
            "items": busy_items,
            "seen": busy_seen,
            "checked": len(workspaces),
            "capped": capped,
            "responsive": True,
        }
    return {
        "responsive": True,
        "busy_without_run": busy_result,
        "reservations": {
            "at": at,
            "at_epoch": now,
            "items": reservations,
            "seen": seen,
            "checked": len(workspaces) if reservations_checked else 0,
            "capped": capped,
            "responsive": True,
            "checked_mode": "all" if reservations_checked else "single_process",
        },
        "closed_pauses": closed,
    }


async def _close_stale_pauses(
    store: Any, pauses: list[dict], batch: Any, names: dict[str, str], now: float
) -> int:
    """Close persisted pauses that shared state no longer holds."""
    from lightrag.api.ops.pauses import _role_slots

    slots = _role_slots()
    patches: dict[str, dict[str, Any]] = {}
    for pause in pauses:
        ws = pause.get("workspace")
        name = names.get(ws) if isinstance(ws, str) else None
        if name is None or name not in batch.data:
            continue
        live = (batch.data.get(name) or {}).get("provider_rejection")
        live_slot = (
            slots.get(live.get("role"), live.get("role") or "unknown")
            if isinstance(live, dict) and live
            else None
        )
        if live_slot is None or live_slot != pause.get("slot"):
            patches[pause["key"]] = {"cleared_at": now, "cleared_by": "reconcile"}
    if patches:
        await store.kv_merge_existing(patches)
    return len(patches)


def _markers() -> dict[int, float]:
    try:
        return run_hooks.read_exit_markers()
    except Exception:
        return {}


async def _maybe_cleanup(store: Any, now: float) -> None:
    if now - _state["last_cleanup"] < CLEANUP_EVERY_S:
        return
    _state["last_cleanup"] = now
    try:
        await store.runs_cleanup(RUN_RETENTION_DAYS, FAILURE_RETENTION_DAYS)
    except Exception as e:
        logger.warning("Ops run retention failed (%s)", type(e).__name__)
    try:
        run_hooks.prune_exit_markers()
    except Exception:
        pass


async def reconcile_cycle(
    loop: LeasedLoop,
    *,
    service: Any,
    process: Callable[[], dict[str, Any]],
    host: Optional[str] = None,
    alive: Callable[[Any, Any], bool] = process_alive,
) -> dict[str, Any]:
    """Run tracking checks (when tracking is on) and write their KV results."""
    store = service.store
    now = service.clock()
    # The same host name the heartbeat records for each worker.
    host = host or socket.gethostname()[:255]
    out: dict[str, Any] = {}
    if not getattr(service, "run_tracking", False):
        return out
    thresholds = await service.thresholds()
    markers = _markers()
    stale_s = float(thresholds.get("orphan_seconds") or 60)
    try:
        process_info = process() or {}
    except Exception:
        process_info = {}
    # Reservations only outlive their owner when another worker can die.
    multi_worker = process_info.get("mode") != "uvicorn"
    worker_rows: Optional[list[dict[str, Any]]]
    try:
        worker_rows = await store.workers(500)
    except Exception as e:
        logger.warning("Ops reconcile could not read workers (%s)", type(e).__name__)
        worker_rows = None
    running: Optional[list[dict[str, Any]]]
    try:
        result = await reconcile_runs(
            store,
            my_worker_id=service.worker_id,
            host=host,
            stale_s=stale_s,
            markers=markers,
            alive=alive,
            workers=worker_rows,
        )
        out["orphaned"] = result["orphaned"]
        out["unresponsive"] = result["unresponsive"]
        running = [
            r for r in result["running"] if r["run_id"] not in result["orphaned"]
        ]
        if len(result["running"]) >= RUNNING_LIMIT:
            # Busy flags cannot be matched against a truncated list.
            running = None
    except Exception as e:
        logger.warning("Ops reconcile of runs failed (%s)", type(e).__name__)
        running = None
    try:
        if worker_rows is None:
            raise RuntimeError("workers unavailable")
        workers = {row["worker_id"]: row for row in worker_rows}
        checks = await check_shared_state(
            store,
            service,
            running=running,
            workers=workers,
            my_worker_id=service.worker_id,
            host=host,
            markers=markers,
            thresholds=thresholds,
            now=now,
            alive=alive,
            reservations_checked=multi_worker,
        )
        if await loop.still_held():
            await store.kv_put(BUSY_WITHOUT_RUN_KEY, checks["busy_without_run"])
            await store.kv_put(RESERVATIONS_KEY, checks["reservations"])
        out["busy_without_run"] = len(checks["busy_without_run"].get("items") or [])
        out["reservations"] = len(checks["reservations"].get("items") or [])
        out["closed_pauses"] = checks["closed_pauses"]
    except Exception as e:
        logger.warning("Ops shared-state checks failed (%s)", type(e).__name__)
    await _maybe_cleanup(store, now)
    for key in (
        ("stuck_runs",),
        ("busy_without_run",),
        ("reservations",),
        ("pauses",),
        ("active",),
    ):
        service.cache.forget(key)
    return out
