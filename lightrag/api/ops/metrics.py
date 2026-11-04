"""Latency and error metrics for the Operations console's Performance tab.

:class:`MetricBins` keeps, per process, one fixed 20-bucket log histogram per
``(minute, series, key)``:

- ``http``: every HTTP request, keyed ``"<METHOD> <route template>"`` (never
  the raw path, so no IDs), ``<unmatched>`` when no route matched (404s) and
  ``"<METHOD> <mount>/root"`` for mounted apps (the WebUI's static files);
  5xx responses and exceptions count as errors;
- ``embed`` / ``rerank``: every embedding and rerank call, keyed by model,
  timed *inside* the priority queue wrapper (so queue wait is not latency);
  a raised exception counts as an error;
- ``auth``: sign-in outcomes only (:data:`AUTH_OUTCOMES`), counts without
  latency. Never a username.

Each series holds at most :data:`MAX_KEYS_PER_MINUTE` keys per minute; more
distinct keys fold into ``<other>``. A per-process flusher writes every
completed minute to ``LIGHTRAG_OPS_METRICS_MIN`` (insert-only, primary key
``(minute, worker_id, series, key)``) about once a minute; one worker per host
rolls minutes up into ``LIGHTRAG_OPS_METRICS_HOUR`` hourly and deletes rows
past retention (14 d minutes, 90 d hours) in batches, under the
``ops:retention`` lease.

The hot-path hooks (:func:`observe`, :func:`count`, :func:`http_observer`,
the wrapper from :func:`wrap_dep_latency`) never raise, never await, never do
I/O and take only a short in-process lock. Nothing is recorded unless the ops
runtime installed a :class:`MetricBins` in this process
(``OPS_METRICS_ENABLED``, default on, and the ops store on Postgres); without
one every hook returns at once.

Stdlib only: core code (``lightrag.py``) imports the dependency wrapper.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import random
import threading
import time
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence

logger = logging.getLogger("lightrag.api.ops")

SERIES_HTTP = "http"
SERIES_EMBED = "embed"
SERIES_RERANK = "rerank"
SERIES_AUTH = "auth"
SERIES = (SERIES_HTTP, SERIES_EMBED, SERIES_RERANK, SERIES_AUTH)

# 20 log-spaced upper bounds from 5 ms to 120 s (ratio ~1.70 per bucket).
# Bucket i holds values in (bound[i-1], bound[i]]; bucket 0 holds [0, 5 ms];
# anything above 120 s lands in the last bucket (max_ms keeps the real max).
BUCKETS = 20
BUCKET_MIN_MS = 5.0
BUCKET_MAX_MS = 120_000.0
BUCKET_BOUNDS_MS: tuple[float, ...] = tuple(
    BUCKET_MIN_MS * (BUCKET_MAX_MS / BUCKET_MIN_MS) ** (i / (BUCKETS - 1))
    for i in range(BUCKETS)
)

UNMATCHED_KEY = "<unmatched>"
OVERFLOW_KEY = "<other>"
MOUNT_PREFIX = "<mount>"
MAX_KEYS_PER_MINUTE = 200
KEY_MAX_LEN = 160
DEFAULT_MODEL_KEY = "default"

# Sign-in outcomes (the only keys of the auth series) and which are failures.
AUTH_OUTCOMES = frozenset(
    {
        "login_ok",
        "login_failed",
        "login_throttled",
        "lockout",
        "sso_ok",
        "sso_error",
        "token_401",
    }
)
AUTH_FAILURES = frozenset(
    {"login_failed", "login_throttled", "lockout", "sso_error", "token_401"}
)
# Routes that count their own sign-in outcome: a password-login 401 is
# login_failed, an SSO callback 401 is sso_error. Neither is a failed token.
_LOGIN_PATHS = frozenset({"/login", "/api/oauth2/callback", "/oauth2/callback"})

FLUSH_INTERVAL_S = 60.0
FLUSH_JITTER_S = (2.0, 6.0)
WRITE_TIMEOUT_S = 10.0
# Rows kept for a retry while writes fail (about 30 busy minutes); older
# rows are dropped first and counted.
MAX_PENDING_ROWS = 20_000

ROLLUP_LEASE = "ops:retention"
ROLLUP_INTERVAL_S = 3600.0
ROLLUP_CYCLE_TIMEOUT_S = 300.0
ROLLUP_LEASE_TTL_S = 600.0
ROLLUP_KEY = "metrics_rollup"
# Hours re-rolled every cycle so minutes flushed late are still counted.
ROLLUP_REDO_HOURS = 2
# A gap after downtime is caught up at most this far back.
ROLLUP_CATCHUP_HOURS = 48
# A minute is in the hour table once its hour ended this long ago.
ROLLUP_SETTLE_S = 300.0
MINUTE_RETENTION_DAYS = 14
HOUR_RETENTION_DAYS = 90


def metrics_enabled() -> bool:
    """``OPS_METRICS_ENABLED`` (default on)."""
    value = os.getenv("OPS_METRICS_ENABLED")
    if value is None or not value.strip():
        return True
    return value.strip().lower() in ("1", "true", "yes", "on")


# -- histograms ------------------------------------------------------------------


def bucket_index(ms: float) -> int:
    """The bucket a latency falls in (clamped to the last one)."""
    index = bisect_left(BUCKET_BOUNDS_MS, ms)
    return index if index < BUCKETS else BUCKETS - 1


def bucket_lower_ms(index: int) -> float:
    return 0.0 if index <= 0 else BUCKET_BOUNDS_MS[index - 1]


def merge_buckets(target: list[int], source: Sequence[Any]) -> list[int]:
    """Add ``source`` into ``target`` element-wise (both 20 long)."""
    for i in range(min(len(target), len(source))):
        target[i] += int(source[i] or 0)
    return target


def estimate_percentiles(
    buckets: Sequence[Any],
    fractions: Iterable[float] = (0.5, 0.95, 0.99),
    max_ms: Optional[float] = None,
) -> list[Optional[float]]:
    """Percentiles from a histogram.

    The rank is located in its bucket and interpolated inside it:
    geometrically for the log-spaced buckets, linearly in the first one
    (0..5 ms). The open-ended last bucket runs up to the observed maximum.
    Every estimate is capped at ``max_ms`` when given. The error is bounded
    by one bucket width (a factor of ~1.7) and is much smaller for smooth
    distributions.
    """
    counts = [max(0, int(c or 0)) for c in buckets][:BUCKETS]
    total = sum(counts)
    out: list[Optional[float]] = []
    for fraction in fractions:
        if total <= 0:
            out.append(None)
            continue
        rank = min(max(float(fraction), 0.0), 1.0) * total
        cumulative = 0
        estimate: Optional[float] = None
        for index, c in enumerate(counts):
            if c <= 0:
                continue
            if cumulative + c >= rank or index == len(counts) - 1:
                position = (rank - cumulative) / c
                position = min(max(position, 0.0), 1.0)
                lower = bucket_lower_ms(index)
                upper = BUCKET_BOUNDS_MS[index]
                if index == BUCKETS - 1 and max_ms and max_ms > upper:
                    upper = float(max_ms)
                if lower <= 0.0:
                    estimate = upper * position
                else:
                    estimate = lower * (upper / lower) ** position
                break
            cumulative += c
        if estimate is not None and max_ms is not None and max_ms >= 0:
            estimate = min(estimate, float(max_ms))
        out.append(None if estimate is None else round(estimate, 1))
    return out


# -- per-process bins ------------------------------------------------------------


class _Bin:
    __slots__ = ("count", "errors", "sum_ms", "max_ms", "buckets")

    def __init__(self) -> None:
        self.count = 0
        self.errors = 0
        self.sum_ms = 0.0
        self.max_ms = 0.0
        self.buckets = [0] * BUCKETS


def clean_key(key: Any) -> str:
    text = str(key) if key is not None else DEFAULT_MODEL_KEY
    text = text.strip() or DEFAULT_MODEL_KEY
    return text[:KEY_MAX_LEN]


class MetricBins:
    """This process's histograms and their flusher (one per worker)."""

    def __init__(
        self,
        store: Any,
        worker_id: str,
        *,
        clock: Callable[[], float] = time.time,
        interval_s: float = FLUSH_INTERVAL_S,
        write_timeout_s: float = WRITE_TIMEOUT_S,
        max_keys: int = MAX_KEYS_PER_MINUTE,
        max_pending_rows: int = MAX_PENDING_ROWS,
        pid: Optional[int] = None,
    ):
        self.store = store
        self.worker_id = worker_id
        self.clock = clock
        self.interval_s = interval_s
        self.write_timeout_s = write_timeout_s
        self.max_keys = max_keys
        self.max_pending_rows = max_pending_rows
        self.pid = pid if pid is not None else os.getpid()
        self._lock = threading.Lock()
        # minute (epoch seconds) -> series -> key -> bin
        self._minutes: dict[int, dict[str, dict[str, _Bin]]] = {}
        self._pending: list[dict[str, Any]] = []
        self._task: Optional[asyncio.Task] = None
        self._flush_lock: Optional[asyncio.Lock] = None
        # Counters (exported in the heartbeat snapshot).
        self.observed = 0
        self.overflowed = 0
        self.dropped_rows = 0
        self.flush_failures = 0
        self.rows_written = 0
        self.last_flush_ok_at: Optional[float] = None
        self.last_flush_error: Optional[str] = None

    # -- hot path ----------------------------------------------------------------
    def observe(
        self, series: str, key: Any, ms: Optional[float], error: bool = False
    ) -> None:
        """Count one event; ``ms`` None counts without a latency (auth)."""
        minute = int(self.clock()) // 60 * 60
        key = clean_key(key)
        with self._lock:
            by_series = self._minutes.get(minute)
            if by_series is None:
                by_series = self._minutes[minute] = {}
            keys = by_series.get(series)
            if keys is None:
                keys = by_series[series] = {}
            b = keys.get(key)
            if b is None:
                if len(keys) >= self.max_keys - 1 and key != OVERFLOW_KEY:
                    self.overflowed += 1
                    key = OVERFLOW_KEY
                    b = keys.get(key)
                if b is None:
                    b = keys[key] = _Bin()
            b.count += 1
            if error:
                b.errors += 1
            if ms is not None:
                value = ms if ms > 0.0 else 0.0
                b.sum_ms += value
                if value > b.max_ms:
                    b.max_ms = value
                index = bisect_left(BUCKET_BOUNDS_MS, value)
                b.buckets[index if index < BUCKETS else BUCKETS - 1] += 1
            self.observed += 1

    # -- flushing --------------------------------------------------------------------
    def take(self, include_current: bool = False) -> list[dict[str, Any]]:
        """Remove and return rows for every completed minute (all with
        ``include_current``, at shutdown)."""
        current = int(self.clock()) // 60 * 60
        with self._lock:
            ready = [m for m in self._minutes if include_current or m < current]
            taken = [(m, self._minutes.pop(m)) for m in sorted(ready)]
        rows: list[dict[str, Any]] = []
        for minute, by_series in taken:
            for series, keys in by_series.items():
                for key, b in keys.items():
                    rows.append(
                        {
                            "minute": minute,
                            "worker_id": self.worker_id,
                            "series": series,
                            "key": key,
                            "count": b.count,
                            "errors": b.errors,
                            "sum_ms": int(round(b.sum_ms)),
                            "max_ms": int(math.ceil(b.max_ms)),
                            "buckets": list(b.buckets),
                        }
                    )
        return rows

    def snapshot_rows(self) -> list[dict[str, Any]]:
        """Rows not yet written (current minute included), without removing
        them: lets a reader include the last partial minute of this process."""
        with self._lock:
            items = [
                (m, s, k, b.count, b.errors, b.sum_ms, b.max_ms, list(b.buckets))
                for m, by_series in self._minutes.items()
                for s, keys in by_series.items()
                for k, b in keys.items()
            ]
        return [
            {
                "minute": m,
                "worker_id": self.worker_id,
                "series": s,
                "key": k,
                "count": c,
                "errors": e,
                "sum_ms": int(round(total)),
                "max_ms": int(math.ceil(mx)),
                "buckets": buckets,
            }
            for m, s, k, c, e, total, mx, buckets in items
        ]

    def _queue(self, rows: list[dict[str, Any]]) -> None:
        self._pending.extend(rows)
        overflow = len(self._pending) - self.max_pending_rows
        if overflow > 0:
            # The oldest minutes go first: newer ones matter more on a chart.
            del self._pending[:overflow]
            self.dropped_rows += overflow

    async def flush_once(self, include_current: bool = False) -> bool:
        """Write completed minutes (and earlier failed rows). Never raises."""
        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()
        async with self._flush_lock:
            try:
                self._queue(self.take(include_current))
            except Exception as e:  # pragma: no cover - take only copies
                logger.debug(
                    "Ops metrics could not be collected (%s)", type(e).__name__
                )
            rows, self._pending = self._pending, []
            if not rows:
                self.last_flush_ok_at = self.clock()
                return True
            if self.store is None:
                return True
            try:
                await asyncio.wait_for(
                    self.store.metrics_write(rows), timeout=self.write_timeout_s
                )
            except asyncio.CancelledError:
                self._pending = rows + self._pending
                raise
            except Exception as e:
                self.flush_failures += 1
                if self.last_flush_error is None:
                    logger.warning(
                        "Ops metrics could not be written (%s); keeping %d rows "
                        "for the next attempt",
                        type(e).__name__,
                        len(rows),
                    )
                self.last_flush_error = type(e).__name__
                self._pending = rows + self._pending
                if len(self._pending) > self.max_pending_rows:
                    self._queue([])
                return False
            if self.last_flush_error is not None:
                logger.info("Ops metrics are written again")
            self.last_flush_error = None
            self.rows_written += len(rows)
            self.last_flush_ok_at = self.clock()
            return True

    def _next_delay(self) -> float:
        """Sleep until just after the next minute boundary (jittered per
        worker so they do not all write at once)."""
        now = self.clock()
        to_boundary = 60.0 - (now % 60.0)
        delay = to_boundary + random.uniform(*FLUSH_JITTER_S)
        return max(1.0, min(delay, self.interval_s + FLUSH_JITTER_S[1]))

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._next_delay())
            try:
                await self.flush_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover - flush_once guards itself
                logger.debug("Ops metrics flush failed (%s)", type(e).__name__)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="ops-metrics")

    async def stop(self, flush_timeout_s: float = 5.0) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            # Nothing is observed after shutdown: the partial minute goes too.
            await asyncio.wait_for(
                self.flush_once(include_current=True), timeout=flush_timeout_s
            )
        except Exception as e:
            logger.debug("Ops metrics final flush failed (%s)", type(e).__name__)

    def stats(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            buffered = sum(
                len(keys)
                for by_series in self._minutes.values()
                for keys in by_series.values()
            )
        return {
            "buffer": buffered + len(self._pending),
            "pending_rows": len(self._pending),
            "observed": self.observed,
            "overflowed": self.overflowed,
            "dropped_rows": self.dropped_rows,
            "flush_failures": self.flush_failures,
            "rows_written": self.rows_written,
            "last_flush_age_s": None
            if self.last_flush_ok_at is None
            else round(max(0.0, now - self.last_flush_ok_at), 1),
            "last_flush_error": self.last_flush_error,
        }


# -- process-wide hooks -------------------------------------------------------------

_bins: Optional[MetricBins] = None


def install(bins: Optional[MetricBins]) -> None:
    """Make ``bins`` this process's metrics (``None`` turns every hook off)."""
    global _bins
    _bins = bins


def current() -> Optional[MetricBins]:
    """This process's bins; never ones inherited across a fork."""
    bins = _bins
    if bins is None or bins.pid != os.getpid():
        return None
    return bins


def observe(series: str, key: Any, ms: Optional[float], error: bool = False) -> None:
    bins = _bins
    if bins is None:
        return
    try:
        if bins.pid != os.getpid():
            return
        bins.observe(series, key, ms, error)
    except Exception:  # pragma: no cover - a metric must never break a call
        pass


def auth_event(outcome: str) -> None:
    """Count one sign-in outcome (a fixed key from :data:`AUTH_OUTCOMES`).

    Never pass anything user-derived: unknown outcomes are ignored, so a
    username cannot become a key by mistake.
    """
    if _bins is None or outcome not in AUTH_OUTCOMES:
        return
    observe(SERIES_AUTH, outcome, None, outcome in AUTH_FAILURES)


def route_key(scope: dict) -> str:
    """``"<METHOD> <template>"`` for a matched route, ``<unmatched>`` else."""
    method = str(scope.get("method") or "GET")[:10]
    route = scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return f"{method} {path}"
    if scope.get("endpoint") is not None:
        # A mounted app (static files): its mount point, never the file path.
        root = str(scope.get("root_path") or "")
        return f"{method} {MOUNT_PREFIX}{root or '/'}"
    return UNMATCHED_KEY


def http_observer(scope: dict, status: int, duration_s: float) -> None:
    """``RequestIdMiddleware`` observer: route latency, 5xx and auth 401s."""
    if _bins is None:
        return
    try:
        key = route_key(scope)
        observe(SERIES_HTTP, key, duration_s * 1000.0, status >= 500)
        if status == 401:
            route = scope.get("route")
            if getattr(route, "path", None) not in _LOGIN_PATHS:
                auth_event("token_401")
    except Exception:  # pragma: no cover - metrics must never break a request
        pass


def _model_key(func: Any, explicit: Optional[str]) -> str:
    if explicit:
        return clean_key(explicit)
    for candidate in (func, getattr(func, "func", None)):
        name = getattr(candidate, "model_name", None)
        if isinstance(name, str) and name:
            return clean_key(name)
    keywords = getattr(func, "keywords", None)
    if isinstance(keywords, dict):
        for name in ("model", "embed_model", "model_name"):
            value = keywords.get(name)
            if isinstance(value, str) and value:
                return clean_key(value)
    return DEFAULT_MODEL_KEY


def wrap_dep_latency(
    func: Optional[Callable[..., Awaitable[Any]]],
    series: str,
    model: Optional[str] = None,
) -> Optional[Callable[..., Awaitable[Any]]]:
    """``_record_dep_latency``: time each call of an embedding or rerank function.

    Applied to the raw function *before* ``priority_limit_async_func_call``
    wraps it, so the timing excludes queue wait and ``get_queue_stats`` /
    ``shutdown`` stay on the outer queue wrapper. ``functools.wraps`` keeps
    the signature (``EmbeddingFunc`` inspects it) and ``__wrapped__``.
    Returns ``func`` unchanged when it is None, already wrapped, or
    ``OPS_METRICS_ENABLED`` is off.
    """
    if func is None or getattr(func, "_ops_dep_series", None) or not callable(func):
        return func
    if not metrics_enabled():
        return func
    key = _model_key(func, model)

    @functools.wraps(func)
    async def _record_dep_latency(*args: Any, **kwargs: Any) -> Any:
        bins = _bins
        if bins is None:
            return await func(*args, **kwargs)
        started = time.perf_counter()
        failed = True
        try:
            result = await func(*args, **kwargs)
            failed = False
            return result
        finally:
            # A cancellation here is the queue's execution timeout (or a
            # shutdown): the dependency did not answer, so it is an error.
            observe(series, key, (time.perf_counter() - started) * 1000.0, failed)

    _record_dep_latency._ops_dep_series = series  # type: ignore[attr-defined]
    return _record_dep_latency


# -- rollup and retention (leased, one worker per host) -----------------------------


def _hour_floor(epoch: float) -> datetime:
    return datetime.fromtimestamp(int(epoch) // 3600 * 3600, timezone.utc)


async def rollup_cycle(
    loop: Any,
    *,
    store: Any,
    clock: Callable[[], float] = time.time,
    minute_days: int = MINUTE_RETENTION_DAYS,
    hour_days: int = HOUR_RETENTION_DAYS,
) -> dict[str, Any]:
    """Roll complete hours into the hour table, then apply retention.

    The last :data:`ROLLUP_REDO_HOURS` complete hours are recomputed every
    cycle (an upsert), so minutes written late are still counted; a gap since
    the stored watermark is caught up (at most :data:`ROLLUP_CATCHUP_HOURS`).
    """
    now = clock()
    # Only hours that ended a few minutes ago: every worker flushed them.
    end = _hour_floor(now - ROLLUP_SETTLE_S)
    state = await store.kv_get(ROLLUP_KEY, None)
    through = state.get("through") if isinstance(state, dict) else None
    earliest = end - timedelta(hours=ROLLUP_CATCHUP_HOURS)
    start = end - timedelta(hours=ROLLUP_REDO_HOURS)
    if isinstance(through, (int, float)):
        watermark = datetime.fromtimestamp(float(through), timezone.utc)
        start = min(start, max(watermark, earliest))
    else:
        start = earliest
    out: dict[str, Any] = {"rolled": 0, "deleted_minutes": 0, "deleted_hours": 0}
    if start < end:
        out["rolled"] = await store.metrics_rollup(start, end)
        if await loop.still_held():
            await store.kv_put(ROLLUP_KEY, {"through": end.timestamp(), "at": now})
    minutes, hours = await store.metrics_cleanup(minute_days, hour_days)
    out["deleted_minutes"], out["deleted_hours"] = minutes, hours
    if minutes or hours:
        logger.info(
            "Ops metrics retention removed %d minute and %d hour rows", minutes, hours
        )
    return out


async def rollup_through(store: Any) -> Optional[float]:
    """End (epoch) of the last hour rolled up, if any."""
    state = await store.kv_get(ROLLUP_KEY, None)
    through = state.get("through") if isinstance(state, dict) else None
    return float(through) if isinstance(through, (int, float)) else None
