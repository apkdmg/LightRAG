"""Per-worker heartbeat for the Operations console.

Every server process (one under Uvicorn, N under Gunicorn) runs one
:class:`Heartbeat`. Every 15 s it takes a snapshot of its own process and
writes it to ``LIGHTRAG_OPS_WORKERS`` on a connection of its own (never the
shared pool, so pool starvation cannot make a live worker look dead); once a
minute it also writes a sample row to ``LIGHTRAG_OPS_WORKER_SAMPLES`` for the
sparklines.

The snapshot holds whitelisted numbers only: memory, CPU, file descriptors,
threads, event-loop lag, the shared pool's size and acquire wait, the shared
state Manager's round-trip time (multi-worker only), queue sums over the
loaded LightRAG instances, the Insights recorder's and the log sink's health,
the email re-sync drainers and, once a minute, container and disk usage.
Never command lines, environment or arguments.

Liveness rules (read side, see :mod:`lightrag.api.ops.service`):

- a worker is alive while its heartbeat is under 45 s old;
- rows of workers from an earlier boot on the same host (started well before
  this worker, and whose process is gone) that stopped beating are marked
  ``superseded`` and leave the expected count. Each beat checks again, since a
  replaced worker only turns stale 45 s after its last beat; a hung sibling
  of the same boot is never superseded, it stays a dead worker to show;
- rows are deleted 24 h after their last beat, samples after 7 days.

Without Postgres the heartbeat still samples, so the console can show the
process that answers. Nothing here may raise into the server.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

from lightrag.kg import shared_storage

logger = logging.getLogger("lightrag.api.ops")

HEARTBEAT_INTERVAL_S = 15.0
ALIVE_WITHIN_S = 45.0
SAMPLE_INTERVAL_S = 60.0
ROW_RETENTION_HOURS = 24
SAMPLE_RETENTION_DAYS = 7
CLEANUP_INTERVAL_S = 3600.0
WRITE_TIMEOUT_S = 10.0
# Longer than the pg_pool_wait serious threshold (5 s), so an exhausted pool
# is measured as such rather than capped below it.
POOL_SAMPLE_TIMEOUT_S = 6.0
# Gunicorn forks its workers within a second or two; rows started less than
# this before us belong to the same boot and are never superseded.
SAME_BOOT_MARGIN_S = 30.0
LAG_TICK_S = 0.5
DROP_WINDOW_S = 300.0
TOP_INSTANCES = 10
MAX_DRAINERS = 20
# Yield to the event loop while summing queues over many loaded instances.
_QUEUE_YIELD_EVERY = 50
_QUEUE_KEYS = ("running", "in_flight", "queued")
_STATUS_KEYS = (
    "queue_name",
    "max_async",
    "max_queue_size",
    "queued",
    "running",
    "in_flight",
    "worker_count",
    "initialized",
    "submitted_total",
    "completed_total",
    "failed_total",
    "cancelled_total",
    "rejected_total",
    "available",
)
_CGROUP = "/sys/fs/cgroup"


def _iso(epoch: Optional[float]) -> Optional[str]:
    if not isinstance(epoch, (int, float)) or epoch <= 0:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), digits)


class LoopLagMonitor:
    """Event-loop lag: how late a short sleep wakes up.

    A task sleeps ``tick_s`` in a loop and records how much later than asked
    it woke. :meth:`take` returns the latest lag and the worst one since the
    previous call. The cost is two wake-ups a second.
    """

    def __init__(self, tick_s: float = LAG_TICK_S):
        self.tick_s = tick_s
        self.last_ms: Optional[float] = None
        self._window_max_ms: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.sleep(self.tick_s)
            lag_ms = max(0.0, (loop.time() - started - self.tick_s) * 1000.0)
            self.last_ms = lag_ms
            if self._window_max_ms is None or lag_ms > self._window_max_ms:
                self._window_max_ms = lag_ms

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="ops-loop-lag")

    def take(self) -> tuple[Optional[float], Optional[float]]:
        worst, self._window_max_ms = self._window_max_ms, None
        return self.last_ms, worst

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class ProcessSampler:
    """This process's RSS, CPU, file descriptors and threads.

    Uses psutil when it is installed; otherwise ``/proc`` and ``os.times``
    where available, and ``None`` for what cannot be measured.
    """

    def __init__(self) -> None:
        self._process: Any = None
        try:
            import psutil

            self._process = psutil.Process()
            self._process.cpu_percent(None)  # first call primes the counter
        except Exception:
            self._process = None
        self._prev_times: Optional[tuple[float, float]] = None
        self.create_time: Optional[float] = None
        if self._process is not None:
            try:
                self.create_time = float(self._process.create_time())
            except Exception:
                self.create_time = None

    def sample(self) -> dict[str, Any]:
        if self._process is not None:
            try:
                return self._psutil_sample()
            except Exception:
                pass
        return self._fallback_sample()

    def _psutil_sample(self) -> dict[str, Any]:
        process = self._process
        out: dict[str, Any] = {
            "rss": int(process.memory_info().rss),
            "cpu_pct": _round(process.cpu_percent(None)),
            "threads": int(process.num_threads()),
            "fds": None,
        }
        num_fds = getattr(process, "num_fds", None)
        if callable(num_fds):
            try:
                out["fds"] = int(num_fds())
            except Exception:
                out["fds"] = None
        return out

    def _fallback_sample(self) -> dict[str, Any]:
        rss = fds = None
        try:
            with open("/proc/self/statm", "rb") as fh:
                rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            rss = None
        try:
            fds = len(os.listdir("/proc/self/fd"))
        except OSError:
            fds = None
        times = os.times()
        busy = times.user + times.system
        now = time.monotonic()
        cpu = None
        if self._prev_times is not None:
            wall = now - self._prev_times[1]
            if wall > 0:
                cpu = _round((busy - self._prev_times[0]) / wall * 100.0)
        self._prev_times = (busy, now)
        return {
            "rss": rss,
            "cpu_pct": cpu,
            "fds": fds,
            "threads": threading.active_count(),
        }


class HostSampler:
    """Container memory and CPU (cgroup v2), host load and disk use.

    Load average is the host's, not the container's, and is labelled so.
    """

    def __init__(self, paths: Iterable[str] = ()):
        self.paths = [p for p in dict.fromkeys(paths) if p]
        self._prev_cpu: Optional[tuple[int, float]] = None

    @staticmethod
    def _read(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="ascii") as fh:
                return fh.read().strip()
        except (OSError, UnicodeDecodeError):
            return None

    def _cpu_pct(self) -> Optional[float]:
        stat = self._read(f"{_CGROUP}/cpu.stat")
        if not stat:
            return None
        usage = None
        for line in stat.splitlines():
            name, _, value = line.partition(" ")
            if name == "usage_usec" and value.strip().isdigit():
                usage = int(value)
                break
        if usage is None:
            return None
        now = time.monotonic()
        previous, self._prev_cpu = self._prev_cpu, (usage, now)
        if previous is None or now <= previous[1]:
            return None
        return _round((usage - previous[0]) / ((now - previous[1]) * 1e6) * 100.0)

    def sample(self) -> dict[str, Any]:
        current = self._read(f"{_CGROUP}/memory.current")
        limit = self._read(f"{_CGROUP}/memory.max")
        try:
            load = [round(v, 2) for v in os.getloadavg()]
        except (OSError, AttributeError):
            load = None
        disks = []
        seen: set[int] = set()
        for path in self.paths:
            try:
                device = os.stat(path).st_dev
                if device in seen:
                    continue
                seen.add(device)
                usage = shutil.disk_usage(path)
            except OSError:
                continue
            disks.append(
                {
                    "path": path,
                    "used": int(usage.used),
                    "total": int(usage.total),
                    "free_pct": _round(usage.free / usage.total * 100.0)
                    if usage.total
                    else None,
                }
            )
        return {
            "mem_used": int(current) if current and current.isdigit() else None,
            "mem_limit": int(limit) if limit and limit.isdigit() else None,
            "cpu_pct": self._cpu_pct(),
            "load": load,
            "load_scope": "host",
            "disks": disks,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
        }


class _DropWindow:
    """Increase of a cumulative counter over the last five minutes."""

    def __init__(self, window_s: float = DROP_WINDOW_S):
        self.window_s = window_s
        self._points: deque[tuple[float, int]] = deque()

    def add(self, now: float, total: int) -> int:
        self._points.append((now, total))
        while self._points and now - self._points[0][0] > self.window_s:
            self._points.popleft()
        return max(0, total - self._points[0][1])


def _queue_counts(stats: Any) -> Optional[dict[str, int]]:
    if not isinstance(stats, dict) or not stats.get("available"):
        return None
    return {k: int(stats.get(k) or 0) for k in _QUEUE_KEYS}


def _status_view(stats: Any) -> Optional[dict[str, Any]]:
    """A queue status in the shape the WebUI's StatusCard reads."""
    if not isinstance(stats, dict):
        return None
    return {k: stats[k] for k in _STATUS_KEYS if k in stats}


class Heartbeat:
    """One per process, created by the ops runtime after the fork."""

    def __init__(
        self,
        *,
        worker_id: str,
        mode: str,
        version: str,
        store: Any,
        reader: Any,
        app_state: Any,
        leases: Callable[[], list[str]],
        data_rag: Callable[[], Any],
        host_paths: Iterable[str] = (),
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ):
        self.worker_id = worker_id
        self.mode = mode
        self.version = version
        self.store = store
        self.reader = reader
        self.app_state = app_state
        self._leases = leases
        self._data_rag = data_rag
        self.interval_s = interval_s
        self.host = socket.gethostname()[:255]
        self.pid = os.getpid()
        self.started_at = datetime.now(timezone.utc)
        self.lag = LoopLagMonitor()
        self.process = ProcessSampler()
        self.host_sampler = HostSampler(host_paths)
        self._lost = _DropWindow()
        self._dropped = _DropWindow()
        self._sink_dropped = _DropWindow()
        self._sink_limited = _DropWindow()
        self._sink_failures = _DropWindow()
        self.snapshot: dict[str, Any] = {}
        self.last_beat_at: Optional[float] = None
        self.last_write_ok_at: Optional[float] = None
        self.beats = 0
        self.write_failures = 0
        # Set by the ops runtime when run tracking runs in this process.
        self.run_tracker: Any = None
        # Set by the ops runtime when the structured log sink runs here.
        self.log_sink: Any = None
        self._failing = False
        self._last_sample_at = 0.0
        self._last_host_at = 0.0
        self._last_cleanup_at = 0.0
        self._host: Optional[dict[str, Any]] = None
        self._window_pool_wait: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    # -- collection -----------------------------------------------------------
    async def _pool(self) -> Optional[dict[str, Any]]:
        if self.store is None:
            return None
        stats = self.store.pool_stats()
        if stats is None:
            return None
        out = {**stats, "wait_ms": None, "saturated": False}
        try:
            wait = await self.store.pool_wait_ms(POOL_SAMPLE_TIMEOUT_S)
            out["wait_ms"] = _round(wait)
        except (asyncio.TimeoutError, TimeoutError):
            out["wait_ms"] = POOL_SAMPLE_TIMEOUT_S * 1000.0
            out["saturated"] = True
        except Exception as e:
            logger.debug("Ops pool wait sample failed (%s)", type(e).__name__)
        wait_ms = out["wait_ms"]
        if wait_ms is not None and (
            self._window_pool_wait is None or wait_ms > self._window_pool_wait
        ):
            self._window_pool_wait = wait_ms
        return out

    async def _manager(self) -> dict[str, Any]:
        if not shared_storage.is_multiprocess():
            return {"state": "local", "rtt_ms": None}
        rag = self._data_rag()
        workspace = getattr(rag, "workspace", "") or ""
        _, responsive = await self.reader.snapshot(workspace, ("busy",))
        return {
            "state": self.reader.state,
            "rtt_ms": _round(self.reader.last_rtt_ms) if responsive else None,
        }

    def _instances(self) -> list[Any]:
        found: list[Any] = []
        seen: set[int] = set()
        rag = self._data_rag()
        manager = getattr(self.app_state, "workspace_manager", None)
        loaded: list[Any] = []
        if manager is not None:
            try:
                loaded = list(manager.loaded_instances())
            except Exception:
                loaded = []
        for candidate in [rag, *loaded]:
            if candidate is not None and id(candidate) not in seen:
                seen.add(id(candidate))
                found.append(candidate)
        return found

    async def _queues(self) -> dict[str, Any]:
        sums = {
            role: dict.fromkeys(_QUEUE_KEYS, 0) for role in ("llm", "embed", "rerank")
        }
        seen_role = dict.fromkeys(sums, False)
        top: list[dict[str, Any]] = []
        instances = self._instances()
        primary: dict[str, Any] = {}
        for index, rag in enumerate(instances):
            if index and index % _QUEUE_YIELD_EVERY == 0:
                await asyncio.sleep(0)
            per_instance = dict.fromkeys(("running", "queued"), 0)
            try:
                llm = await rag.get_llm_queue_status()
                embed = await rag.get_embedding_queue_status()
                rerank = await rag.get_rerank_queue_status()
            except Exception:
                continue
            if index == 0:
                primary = {
                    "llm_queue_status": {
                        role: _status_view(stats) for role, stats in (llm or {}).items()
                    },
                    "embedding_queue_status": _status_view(embed),
                    "rerank_queue_status": _status_view(rerank),
                }
            parts = [("llm", s) for s in (llm or {}).values()]
            parts += [("embed", embed), ("rerank", rerank)]
            for role, stats in parts:
                counts = _queue_counts(stats)
                if counts is None:
                    continue
                seen_role[role] = True
                for key in _QUEUE_KEYS:
                    sums[role][key] += counts[key]
                per_instance["running"] += counts["running"]
                per_instance["queued"] += counts["queued"]
            if per_instance["running"] or per_instance["queued"]:
                top.append(
                    {"workspace": getattr(rag, "workspace", "") or "", **per_instance}
                )
        top.sort(key=lambda t: (-(t["queued"] + t["running"]), t["workspace"]))
        return {
            "instances": len(instances),
            "queues": {role: sums[role] if seen_role[role] else None for role in sums},
            "top_instances": top[:TOP_INSTANCES],
            **primary,
        }

    def _recorder(self, now: float) -> Optional[dict[str, Any]]:
        recorder = getattr(self.app_state, "insights_recorder", None)
        health = getattr(recorder, "health", None)
        if not callable(health):
            return None
        data = health()
        lost = int(data.get("lost_rows") or 0)
        dropped = int(data.get("dropped_events") or 0)
        last_ok = data.get("last_flush_ok_at")
        return {
            "buffer": int(data.get("buffer") or 0),
            "last_flush_ok_at": _iso(last_ok),
            "last_flush_age_s": _round(now - last_ok) if last_ok else None,
            "flush_failures": int(data.get("flush_failures") or 0),
            "lost_rows": lost,
            "dropped_events": dropped,
            "lost_5m": self._lost.add(now, lost),
            "dropped_5m": self._dropped.add(now, dropped),
            "running": bool(data.get("running")),
        }

    def _log_sink(self, now: float) -> Optional[dict[str, Any]]:
        """The structured log sink's health (None when it is off here)."""
        sink = self.log_sink
        if sink is None:
            return None
        stats = sink.stats()
        dropped = int(stats.get("dropped") or 0)
        limited = int(stats.get("rate_limited") or 0)
        failures = int(stats.get("write_failures") or 0)
        return {
            "level": stats.get("level"),
            "attached": bool(stats.get("attached")),
            "buffer": int(stats.get("buffer") or 0),
            "written": int(stats.get("written") or 0),
            "dropped": dropped,
            "rate_limited": limited,
            "write_failures": failures,
            "bad_rows": int(stats.get("bad_rows") or 0),
            "last_flush_ok_at": stats.get("last_flush_ok_at"),
            "last_flush_age_s": stats.get("last_flush_age_s"),
            "last_flush_error": stats.get("last_flush_error"),
            "dropped_5m": self._sink_dropped.add(now, dropped),
            "rate_limited_5m": self._sink_limited.add(now, limited),
            "write_failures_5m": self._sink_failures.add(now, failures),
        }

    def _drainers(self) -> list[dict[str, Any]]:
        queue = getattr(self.app_state, "email_resync_queue", None)
        states = getattr(queue, "drainer_states", None)
        if callable(states):
            return [
                {
                    "workspace": d["workspace"],
                    "state": d.get("state") or "draining",
                    "backoff_until": _iso(d.get("backoff_until")),
                }
                for d in states()[:MAX_DRAINERS]
            ]
        active = getattr(queue, "active_drainers", None)
        if not callable(active):
            return []
        return [
            {"workspace": ws, "state": "draining", "backoff_until": None}
            for ws in active()[:MAX_DRAINERS]
        ]

    async def collect(self) -> dict[str, Any]:
        """This process's snapshot; each part fails on its own."""
        now = time.time()
        snapshot: dict[str, Any] = {}
        try:
            snapshot.update(self.process.sample())
        except Exception as e:
            logger.debug("Ops process sample failed (%s)", type(e).__name__)
        last_lag, worst_lag = self.lag.take()
        snapshot["loop_lag_ms"] = _round(last_lag)
        snapshot["loop_lag_max_ms"] = _round(worst_lag)
        try:
            snapshot["pool"] = await self._pool()
        except Exception:
            snapshot["pool"] = None
        try:
            manager = await self._manager()
        except Exception:
            manager = {"state": "unknown", "rtt_ms": None}
        snapshot["manager_state"] = manager["state"]
        snapshot["manager_rtt_ms"] = manager["rtt_ms"]
        try:
            snapshot.update(await self._queues())
        except Exception as e:
            logger.debug("Ops queue sums failed (%s)", type(e).__name__)
        try:
            snapshot["recorder"] = self._recorder(now)
        except Exception:
            snapshot["recorder"] = None
        try:
            snapshot["log_sink"] = self._log_sink(now)
        except Exception:
            snapshot["log_sink"] = None
        try:
            tracker = self.run_tracker
            snapshot["run_tracking"] = tracker.stats() if tracker is not None else None
        except Exception:
            snapshot["run_tracking"] = None
        try:
            snapshot["drainers"] = self._drainers()
        except Exception:
            snapshot["drainers"] = []
        if self._host is None or now - self._last_host_at >= SAMPLE_INTERVAL_S:
            self._last_host_at = now
            try:
                self._host = await asyncio.to_thread(self.host_sampler.sample)
            except Exception:
                self._host = None
        snapshot["host"] = self._host
        snapshot["beat_failures"] = self.write_failures
        return snapshot

    # -- writing ---------------------------------------------------------------
    def row(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        try:
            leases = sorted(self._leases())
        except Exception:
            leases = []
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "pid": self.pid,
            "pid_create_time": self.process.create_time,
            "mode": self.mode,
            "version": self.version,
            "started_at": self.started_at,
            "leases_held": leases,
            "snapshot": snapshot,
        }

    async def beat_once(self) -> bool:
        """Collect and (with a store) write one beat. Never raises."""
        try:
            snapshot = await self.collect()
        except Exception as e:  # pragma: no cover - collect guards each part
            logger.warning("Ops heartbeat could not sample (%s)", type(e).__name__)
            snapshot = {}
        now = time.time()
        self.snapshot = snapshot
        self.last_beat_at = now
        self.beats += 1
        if self.store is None:
            return True
        sample = None
        if now - self._last_sample_at >= SAMPLE_INTERVAL_S:
            sample = {
                "cpu_pct": snapshot.get("cpu_pct"),
                "rss": snapshot.get("rss"),
                "loop_lag_ms": _int(snapshot.get("loop_lag_max_ms")),
                "pool_wait_ms": _int(self._window_pool_wait),
            }
        try:
            await asyncio.wait_for(
                self.store.worker_beat(self.row(snapshot), sample),
                timeout=WRITE_TIMEOUT_S,
            )
            await asyncio.wait_for(
                self.store.supersede_workers(
                    self.host,
                    self.worker_id,
                    self.started_at - timedelta(seconds=SAME_BOOT_MARGIN_S),
                    ALIVE_WITHIN_S,
                    alive=process_alive,
                ),
                timeout=WRITE_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.write_failures += 1
            if not self._failing:
                self._failing = True
                logger.warning(
                    "Ops heartbeat could not be written (%s); retrying every %.0fs",
                    type(e).__name__,
                    self.interval_s,
                )
            return False
        if self._failing:
            self._failing = False
            logger.info("Ops heartbeat is written again")
        self.last_write_ok_at = now
        if sample is not None:
            self._last_sample_at = now
            self._window_pool_wait = None
        if now - self._last_cleanup_at >= CLEANUP_INTERVAL_S:
            self._last_cleanup_at = now
            try:
                await asyncio.wait_for(
                    self.store.cleanup_workers(
                        ROW_RETENTION_HOURS, SAMPLE_RETENTION_DAYS
                    ),
                    timeout=WRITE_TIMEOUT_S,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Ops worker cleanup failed (%s)", type(e).__name__)
        return True

    async def _run(self) -> None:
        while True:
            await self.beat_once()
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        self.lag.start()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="ops-heartbeat")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.lag.stop()
        if self.store is not None and self.last_write_ok_at is not None:
            try:
                await asyncio.wait_for(
                    self.store.worker_exited(self.worker_id), timeout=3.0
                )
            except Exception as e:
                logger.debug("Ops worker exit not recorded (%s)", type(e).__name__)

    def local_row(self) -> dict[str, Any]:
        """This process as a worker row, for views without the database."""
        age = None if self.last_beat_at is None else time.time() - self.last_beat_at
        row = self.row(self.snapshot)
        row.update(
            {
                "heartbeat_at": _iso(self.last_beat_at),
                "heartbeat_age_s": age,
                "exited_at": None,
                "superseded": False,
            }
        )
        return row


def process_alive(pid: Any, create_time: Any) -> bool:
    """Whether ``pid`` on this host is still the process that wrote the row.

    Our own pid is never another worker's (a reused pid after a restart).
    With psutil the start time must match too; without it an existing pid
    counts as alive, which only ever keeps a row from being superseded.
    """
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return False
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            started = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            return False
        except Exception:
            return True  # exists, but not ours to inspect
        if isinstance(create_time, (int, float)):
            return abs(float(started) - float(create_time)) < 1.0
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _int(value: Any) -> Optional[int]:
    return None if value is None else int(round(float(value)))
