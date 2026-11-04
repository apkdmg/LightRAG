"""Structured WARNING+ log sink for the Operations console (``LIGHTRAG_OPS_LOGS``).

:class:`OpsLogHandler` is attached to ``lightrag`` (which does not
propagate), ``uvicorn.error``, ``gunicorn.error`` and the root logger, where
only an allowlist of libraries (``openai``, ``httpx``, ``asyncpg``,
``pymilvus``, ``neo4j``) is taken. Its ``emit()``:

- redacts first (the same :class:`RedactFilter` as the file and console
  handlers, messages capped at 2 KB and tracebacks at 8 KB), so nothing
  unredacted is ever buffered;
- fingerprints the call site (logger, module, line, exception type, level);
- spends one token of that fingerprint's bucket (20 a minute by default);
  without a token the record is only counted, and the count rides on the
  fingerprint's next stored row (``repeat_count``);
- appends one row to a bounded buffer (20 000 rows); when it is full the row
  is dropped and counted.

It does no I/O, takes no lock beyond the handler's own, never raises, and
never records what the sink itself logs (``lightrag.api.ops.sink``) or
anything logged while a flush is writing (so a failing write cannot feed
itself).

:class:`OpsLogSink` owns the handler, this process's flusher (every 2 s, or
sooner when 1000 rows wait; at most 50 rows/s in steady state, bursts of up
to 1000 after a quiet spell) and the counters the heartbeat exports. The
budget, the burst and the per-call-site limit are server-wide: under
Gunicorn each of N workers takes 1/N of them. The
flusher coalesces identical records of a batch (same call site and context)
into one row with a ``repeat_count``, writes with ``COPY``, keeps a failed
batch (up to a cap) for the next attempt with backoff, and never writes rows
buffered for a workspace before its deletion.

Retention (14 days by default, ``OPS_LOG_RETENTION_DAYS``) runs hourly as the
leased singleton ``ops:log-retention``, together with a weekly audit that
counts stored rows still matching a secret, email, IC, phone or file-name
pattern (counts only).

The sink is created per worker by the ops runtime (never at import) and only
when ``OPS_LOG_SINK=true``; ``OPS_LOG_LEVEL`` (default ``WARNING``) sets the
lowest level stored.

**INFO capture** (FR-26): an operator can store INFO lines of one
allowlisted module for at most 60 minutes. The capture lives in KV
``log_capture`` (``{module, until, id}``); every worker's flusher reads it
every 10 s, so all workers capture, and each handler checks ``until`` on
every record, so a capture ends on time even when the database cannot be
read. While no capture runs the handler keeps its normal level, so INFO
records are rejected before any work. Captured lines go through the same
redaction, per-call-site limit and write budget, plus a cap per capture
(20 000 lines server-wide, 1/N per worker).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict, deque
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from uuid import UUID

from lightrag.api.ops.context import (
    CONSOLE_LOG_FORMAT,
    ContextFilter,
    ContextFormatter,
    current_worker_id,
    request_unauthenticated,
)
from lightrag.api.ops.redact import (
    EXC_MAX_CHARS,
    MSG_MAX_CHARS,
    RedactFilter,
    fingerprint,
    normalise_message,
    redact_pipeline,
    redact_traceback,
)

SINK_LOGGER_NAME = "lightrag.api.ops.sink"
logger = logging.getLogger(SINK_LOGGER_NAME)

DEFAULT_LEVEL = logging.WARNING
MAX_BUFFER = 20000
RATE_PER_MIN = 20
FLUSH_INTERVAL_S = 2.0
FLUSH_POLL_S = 0.25
BATCH_ROWS = 1000
ROWS_PER_S = 50.0
WRITE_TIMEOUT_S = 10.0
MAX_RETRY_ROWS = 5000
MAX_BACKOFF_S = 60.0
MAX_FINGERPRINTS = 5000
DEFAULT_RETENTION_DAYS = 14
# Log lines about a workspace logged this soon after its deletion (teardown
# chatter) are dropped with it.
DELETION_GRACE_S = 30.0
FORGET_KEEP_S = 3600.0

RETENTION_LEASE = "ops:log-retention"
RETENTION_INTERVAL_S = 3600.0
RETENTION_CYCLE_TIMEOUT_S = 300.0
RETENTION_LEASE_TTL_S = 600.0
AUDIT_DAYS = 7
AUDIT_EVERY_S = 7 * 86400.0
AUDIT_KEY = "log_redaction_audit"

# Loggers the handler is attached to directly.
ATTACH_LOGGERS = ("lightrag", "uvicorn.error", "gunicorn.error")
# Third-party loggers taken from the root logger.
ROOT_ALLOWLIST = ("openai", "httpx", "asyncpg", "pymilvus", "neo4j")
_ACCEPTED = (*ATTACH_LOGGERS, *ROOT_ALLOWLIST)

# Per-record context a carry-over row (rate-limited counts) leaves empty.
_RECORD_CONTEXT = (
    "workspace",
    "request_id",
    "client_request_id",
    "run_id",
    "track_id",
    "doc_id",
    "source",
)
SUPPRESSED_OPERATION = "suppressed"

_TRACEBACK_MARK = "Traceback (most recent call last):"
_DOC_ID_IN_TEXT = re.compile(r"(?<![0-9A-Za-z-])(?:doc|dup)-[0-9a-f]{32}(?![0-9a-f])")
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]{0,127})(?::|$)")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")

# INFO capture (FR-26): the KV key, its limits, and the modules (a record's
# ``module``, i.e. the file name) whose INFO lines may be captured.
CAPTURE_KEY = "log_capture"
CAPTURE_MAX_MINUTES = 60
CAPTURE_POLL_S = 10.0
CAPTURE_MAX_ROWS = 20000
CAPTURE_MODULES = (
    "pipeline",
    "operate",
    "lightrag",
    "utils_pipeline",
    "document_routes",
    "email_routes",
    "email_resync_queue",
    "startup_resume",
    "workspace_manager",
    "llm_roles",
    "openai",
    "provider_rejection",
    "provider_alerts",
    "postgres_impl",
    "milvus_impl",
    "neo4j_impl",
)


def capture_view(value: Any, now: float) -> Optional[dict[str, Any]]:
    """The capture in KV while it runs, as the console shows it."""
    if not isinstance(value, dict):
        return None
    until = value.get("until")
    module = value.get("module")
    if not isinstance(until, (int, float)) or until <= now or not module:
        return None
    return {
        "module": module,
        "until": _iso(until),
        "until_epoch": float(until),
        "started_at": _iso(value.get("started_at")),
        "expires_at": _iso(until),
        "minutes": value.get("minutes"),
        "actor": value.get("actor"),
        "by": value.get("actor"),
        "reason": value.get("reason"),
        "id": value.get("id"),
    }


# Set inside the flusher's task: records logged while a flush writes (by the
# database driver, the pool, the store) are never fed back into the sink.
_in_flush: ContextVar[bool] = ContextVar("lightrag_ops_log_flush", default=False)


def _accepted(name: str) -> bool:
    if name == SINK_LOGGER_NAME or name.startswith(SINK_LOGGER_NAME + "."):
        return False
    return any(name == root or name.startswith(root + ".") for root in _ACCEPTED)


def _context(record: logging.LogRecord, field: str, limit: int) -> Optional[str]:
    value = getattr(record, field, None)
    if value is None or value == "-" or value == "":
        return None
    return str(value)[:limit]


def _uuid_hex(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if _HEX32.match(value):
        return value
    try:
        return UUID(value).hex
    except (ValueError, AttributeError, TypeError):
        return None


def level_from(value: Any, default: int = DEFAULT_LEVEL) -> int:
    """A level number from ``WARNING``/``error``/``30``; ``default`` otherwise."""
    if isinstance(value, int):
        return value
    text = str(value or "").strip().upper()
    if not text:
        return default
    if text.isdigit():
        return int(text)
    if text == "WARN":
        text = "WARNING"
    number = logging.getLevelName(text)
    return number if isinstance(number, int) else default


class _Bucket:
    __slots__ = ("tokens", "at", "suppressed", "template")

    def __init__(self, tokens: float, at: float):
        self.tokens = tokens
        self.at = at
        self.suppressed = 0
        # The latest stored row of this fingerprint (already redacted), to
        # carry rate-limited counts when no newer row comes.
        self.template: Optional[dict[str, Any]] = None


class OpsLogHandler(logging.Handler):
    """Buffers redacted, fingerprinted, rate-limited rows; never does I/O."""

    def __init__(
        self,
        *,
        level: int = DEFAULT_LEVEL,
        max_buffer: int = MAX_BUFFER,
        rate_per_min: float = RATE_PER_MIN,
        clock: Callable[[], float] = time.time,
        worker_id: Optional[str] = None,
    ):
        super().__init__(level)
        # The configured level; lowered to INFO only while a capture runs.
        self.base_level = level
        self.capture: Optional[tuple[str, float, str]] = None
        self.capture_cap = CAPTURE_MAX_ROWS
        self.capture_rows = 0
        self.capture_dropped = 0
        self.max_buffer = max_buffer
        self.rate_per_min = float(rate_per_min)
        self.clock = clock
        self.worker_id = worker_id
        self.pid = os.getpid()
        self.buffer: deque[dict[str, Any]] = deque()
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._local = threading.local()
        # The same hygiene filters as the file and console handlers; each
        # marks the record, so the work is done once whichever handler runs
        # first.
        self.addFilter(_Gate(self))
        self.addFilter(RedactFilter())
        self.addFilter(ContextFilter())
        # Counters (cumulative; the heartbeat exports them).
        self.accepted = 0
        self.dropped = 0
        self.rate_limited = 0
        self.self_suppressed = 0
        self.errors = 0
        # WARNING+ lines logged while serving a request that presented no
        # valid credential (rate-limited ones included): the
        # unauth_warning_lines_5m rule (R24).
        self.unauth_warnings = 0

    # -- INFO capture ---------------------------------------------------------------
    def set_capture(
        self, module: Optional[str], until: float, capture_id: str = ""
    ) -> None:
        """Capture ``module``'s records below the configured level until
        ``until`` (epoch); no module, or a past ``until``, ends it."""
        if not module or until <= self.clock() or module not in CAPTURE_MODULES:
            self.end_capture()
            return
        current = self.capture
        if current is None or current[2] != capture_id:
            self.capture_rows = 0
        self.capture = (module, float(until), capture_id)
        self.setLevel(min(self.base_level, logging.INFO))

    def end_capture(self) -> None:
        self.capture = None
        if self.level != self.base_level:
            self.setLevel(self.base_level)

    def captures(self, record: logging.LogRecord) -> bool:
        """Whether a record below the configured level is captured."""
        capture = self.capture
        if capture is None or record.levelno < logging.INFO:
            return False
        module, until, _ = capture
        if self.clock() >= until:
            self.end_capture()
            return False
        if record.module != module or not record.name.startswith("lightrag"):
            return False
        if self.capture_rows >= self.capture_cap:
            self.capture_dropped += 1
            return False
        self.capture_rows += 1
        return True

    # -- logging.Handler ------------------------------------------------------------
    def handle(self, record: logging.LogRecord) -> bool:
        try:
            return bool(super().handle(record))
        except Exception:
            # Filters ran before emit(); nothing may escape into the caller.
            self.errors += 1
            return False

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        self.errors += 1

    def emit(self, record: logging.LogRecord) -> None:
        local = self._local
        if getattr(local, "active", False):
            self.self_suppressed += 1
            return
        local.active = True
        try:
            self._emit(record)
        except Exception:
            self.errors += 1
        finally:
            local.active = False

    # -- the work -------------------------------------------------------------------
    def _emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_lightrag_ops_sunk", False):
            return
        record._lightrag_ops_sunk = True
        if _in_flush.get():
            self.self_suppressed += 1
            return
        if record.levelno >= logging.WARNING and request_unauthenticated():
            self.unauth_warnings += 1
        msg, exc, exc_type = _texts(record)
        lineno = int(record.lineno or 0)
        module = (record.module or "")[:128]
        # The normalised message breaks ties only when no call site is known.
        tie = None if lineno and module else normalise_message(msg)
        fp = fingerprint(record.name, module, lineno, exc_type, record.levelno, tie)
        now = self.clock()
        bucket = self._bucket(fp, now)
        if bucket.tokens < 1.0:
            bucket.suppressed += 1
            self.rate_limited += 1
            return
        if len(self.buffer) >= self.max_buffer:
            self.dropped += 1
            return
        bucket.tokens -= 1.0
        doc_id = _context(record, "doc_id", 255)
        if doc_id is None:
            match = _DOC_ID_IN_TEXT.search(msg)
            doc_id = match.group(0) if match else None
        row = {
            "ts": float(record.created),
            "level": int(record.levelno),
            "logger": record.name[:128],
            "worker_id": _uuid_hex(self.worker_id or current_worker_id()),
            "pid": record.process or self.pid,
            "module": module or None,
            "lineno": lineno or None,
            "workspace": _context(record, "workspace", 255),
            "request_id": _context(record, "request_id", 64),
            "client_request_id": _context(record, "client_request_id", 64),
            "run_id": _uuid_hex(_context(record, "run_id", 64)),
            "track_id": _context(record, "track_id", 255),
            "doc_id": doc_id,
            "operation": _context(record, "operation", 64),
            "source": _context(record, "source", 32),
            "msg": msg,
            "exc_type": exc_type,
            "exc": exc,
            "fingerprint": fp,
            "repeat_count": 1 + bucket.suppressed,
        }
        bucket.suppressed = 0
        bucket.template = row
        self.buffer.append(row)
        self.accepted += 1

    def _bucket(self, fp: str, now: float) -> _Bucket:
        capacity = self.rate_per_min
        bucket = self._buckets.get(fp)
        if bucket is None:
            bucket = _Bucket(capacity, now)
            self._buckets[fp] = bucket
            if len(self._buckets) > MAX_FINGERPRINTS:
                # The least recently seen call site; its pending count is lost
                # only when thousands of newer call sites follow it.
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(fp)
            elapsed = max(0.0, now - bucket.at)
            bucket.tokens = min(capacity, bucket.tokens + elapsed * capacity / 60.0)
            bucket.at = now
        return bucket

    def take_suppressed(self, now: float, room: int) -> list[dict[str, Any]]:
        """At most ``room`` rows carrying rate-limited counts that no newer
        row picked up; the other counts stay pending for a later flush.

        Only for call sites with a token to spend (their storm is over); a
        storm still running carries its count on its next stored row. The
        suppressed lines may come from other requests, documents or
        workspaces than the last stored one, so a carry-over row keeps only
        the call site (logger, module, line, level, exception type, and the
        last stored message with its variable parts collapsed) and none of
        the per-record context.
        """
        out: list[dict[str, Any]] = []
        if room <= 0:
            return out
        for fp, bucket in list(self._buckets.items()):
            if len(out) >= room:
                break
            if not bucket.suppressed or bucket.template is None:
                continue
            elapsed = max(0.0, now - bucket.at)
            tokens = min(
                self.rate_per_min, bucket.tokens + elapsed * self.rate_per_min / 60.0
            )
            if tokens < 1.0:
                continue
            bucket.tokens, bucket.at = tokens - 1.0, now
            row = dict(bucket.template)
            for field in _RECORD_CONTEXT:
                row[field] = None
            row["operation"] = SUPPRESSED_OPERATION
            # The message with its variable parts (IDs, names, numbers)
            # collapsed: the lines it stands for differ in those.
            row["msg"] = normalise_message(row["msg"])
            row["exc"] = None
            row["ts"] = now
            row["repeat_count"] = bucket.suppressed
            bucket.suppressed = 0
            out.append(row)
        return out

    def forget(self, workspace: str) -> int:
        """Drop buffered rows and pending counts of a deleted workspace."""
        kept = deque(r for r in list(self.buffer) if r.get("workspace") != workspace)
        dropped = len(self.buffer) - len(kept)
        self.buffer = kept
        for bucket in self._buckets.values():
            if (bucket.template or {}).get("workspace") == workspace:
                # The template's message may name the workspace; its pending
                # count (lines from any workspace) is dropped with it.
                bucket.template = None
                bucket.suppressed = 0
        return dropped


class _Gate(logging.Filter):
    """First filter: skips what the sink never takes before any work is done."""

    def __init__(self, handler: "OpsLogHandler"):
        super().__init__()
        self.handler = handler

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_lightrag_ops_sunk", False):
            return False
        if not _accepted(record.name):
            return False
        if record.levelno < self.handler.base_level:
            return self.handler.captures(record)
        return True


def _texts(record: logging.LogRecord) -> tuple[str, Optional[str], Optional[str]]:
    """``(msg, exc, exc_type)``, redacted and capped (the filters ran first)."""
    message = record.getMessage()
    if not getattr(record, "_lightrag_redacted", False):  # pragma: no cover
        message = redact_pipeline(message, EXC_MAX_CHARS)
    exc_parts = []
    if record.exc_text:
        exc_parts.append(record.exc_text)
    if record.stack_info:
        exc_parts.append(record.stack_info)
    exc_type = None
    if record.exc_info and record.exc_info[0] is not None:
        exc_type = record.exc_info[0].__name__
    marker = message.find(_TRACEBACK_MARK)
    if marker >= 0:
        # logger.error(traceback.format_exc()): the traceback is the exc.
        trace = message[marker:]
        head = message[:marker].strip()
        lines = [line for line in trace.splitlines() if line.strip()]
        last = lines[-1].strip() if lines else ""
        if exc_type is None:
            match = _EXC_LINE.match(last)
            exc_type = match.group(1).rsplit(".", 1)[-1] if match else None
        exc_parts.insert(0, trace)
        message = head or last or _TRACEBACK_MARK
    if "\r" in message:
        message = message.replace("\r\n", "\n").replace("\r", "\n")
    msg = redact_pipeline(message, MSG_MAX_CHARS)
    exc = None
    if exc_parts:
        exc = redact_traceback("\n".join(exc_parts), EXC_MAX_CHARS) or None
    return msg, exc, (exc_type or None) and exc_type[:128]


class _LastResortStandIn(logging.StreamHandler):
    """Stands in for ``logging.lastResort`` while the sink sits on the root.

    A handler on the root logger means Python no longer prints otherwise
    unhandled WARNING+ records to stderr; this one keeps doing that, redacted.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)
        self.setLevel(logging.WARNING)
        self.setFormatter(ContextFormatter(CONSOLE_LOG_FORMAT))
        self.addFilter(RedactFilter())
        self.addFilter(ContextFilter())


def _coalesce_key(row: dict[str, Any]) -> tuple:
    """Rows of one batch with the same key become one row (``repeat_count``).

    The same call site *and* the same context (workspace, request, run,
    document, message), so coalescing never costs correlation.
    """
    return (
        row["fingerprint"],
        row.get("workspace"),
        row.get("request_id"),
        row.get("run_id"),
        row.get("doc_id"),
        row["msg"],
    )


class OpsLogSink:
    """This process's log handler, flusher and counters."""

    def __init__(
        self,
        store: Any,
        worker_id: str,
        *,
        level: int = DEFAULT_LEVEL,
        clock: Callable[[], float] = time.time,
        interval_s: float = FLUSH_INTERVAL_S,
        poll_s: float = FLUSH_POLL_S,
        batch_rows: int = BATCH_ROWS,
        rows_per_s: float = ROWS_PER_S,
        max_buffer: int = MAX_BUFFER,
        rate_per_min: float = RATE_PER_MIN,
        write_timeout_s: float = WRITE_TIMEOUT_S,
        max_retry_rows: int = MAX_RETRY_ROWS,
        workers: int = 1,
    ):
        self.store = store
        self.worker_id = worker_id
        self.clock = clock
        self.interval_s = interval_s
        self.poll_s = poll_s
        # The write budget (NFR-6: 50 rows/s), the burst and the per-call-site
        # limit are for the whole server: each of N Gunicorn workers takes
        # its share.
        self.workers = max(1, int(workers or 1))
        self.batch_rows = max(1, int(batch_rows) // self.workers)
        self.rows_per_s = float(rows_per_s) / self.workers
        rate_per_min = max(1.0, float(rate_per_min) / self.workers)
        self.write_timeout_s = write_timeout_s
        self.max_retry_rows = max_retry_rows
        self.level = level
        self.handler = OpsLogHandler(
            level=level,
            max_buffer=max_buffer,
            rate_per_min=rate_per_min,
            clock=clock,
            worker_id=worker_id,
        )
        self.handler.capture_cap = max(1, CAPTURE_MAX_ROWS // self.workers)
        self._capture_polled = 0.0
        self._attached: list[logging.Logger] = []
        self._stand_in: Optional[logging.Handler] = None
        self._task: Optional[asyncio.Task] = None
        self._flush_lock: Optional[asyncio.Lock] = None
        # Coalesced rows a failed write left behind, written first next time.
        self._retry: list[dict[str, Any]] = []
        self._row_tokens = float(self.batch_rows)
        self._tokens_at = clock()
        self._last_flush = 0.0
        self._backoff_until = 0.0
        self._failures_in_row = 0
        self._forgotten: dict[str, float] = {}
        # Counters (with the handler's, exported by stats()).
        self.written = 0
        self.write_failures = 0
        self.bad_rows = 0
        self.dropped_deleted = 0
        self.dropped_retry = 0
        self.last_flush_ok_at: Optional[float] = None
        self.last_flush_error: Optional[str] = None

    # -- attach / detach ---------------------------------------------------------------
    def attach(self) -> None:
        """Attach the handler to the sink's loggers (idempotent)."""
        if self._attached:
            return
        root = logging.getLogger()
        targets = [logging.getLogger(name) for name in ATTACH_LOGGERS] + [root]
        if not root.handlers:
            # Keep unhandled third-party warnings on stderr (see the class).
            self._stand_in = _LastResortStandIn()
            root.addHandler(self._stand_in)
        for target in targets:
            if self.handler not in target.handlers:
                target.addHandler(self.handler)
            self._attached.append(target)

    def detach(self) -> None:
        for target in self._attached:
            try:
                target.removeHandler(self.handler)
            except Exception:  # pragma: no cover - removeHandler does not fail
                pass
        self._attached = []
        if self._stand_in is not None:
            logging.getLogger().removeHandler(self._stand_in)
            self._stand_in = None

    @property
    def attached(self) -> bool:
        return bool(self._attached)

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="ops-log-sink")

    async def stop(self, flush_timeout_s: float = 5.0) -> None:
        self.detach()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        token = _in_flush.set(True)
        try:
            # What is left is written without the rate budget, a few batches
            # at most, within the timeout.
            self._row_tokens = float("inf")
            self._backoff_until = 0.0
            await asyncio.wait_for(self._drain(), timeout=flush_timeout_s)
        except Exception as e:
            logger.debug("Ops log sink final flush failed (%s)", type(e).__name__)
        finally:
            _in_flush.reset(token)
            self.handler.close()

    async def _drain(self, batches: int = 5) -> None:
        for _ in range(batches):
            if not await self.flush_once():
                return
            if not self.handler.buffer and not self._retry:
                return

    # -- INFO capture -------------------------------------------------------------------
    def apply_capture(self, value: Any) -> Optional[dict[str, Any]]:
        """Follow the capture stored in KV (or end it); returns its view."""
        view = capture_view(value, self.clock())
        if view is None:
            self.handler.end_capture()
        else:
            self.handler.set_capture(
                view["module"], view["until_epoch"], str(view.get("id") or "")
            )
        return view

    async def poll_capture(self) -> None:
        """Read KV ``log_capture``; an unreadable store keeps the current
        capture, which still ends at its ``until``."""
        self._capture_polled = self.clock()
        try:
            value = await asyncio.wait_for(
                self.store.kv_get(CAPTURE_KEY, None), timeout=self.write_timeout_s
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return
        self.apply_capture(value)

    async def _run(self) -> None:
        # Everything this task logs or triggers is kept out of the sink.
        _in_flush.set(True)
        while True:
            await asyncio.sleep(self.poll_s)
            now = self.clock()
            if now - self._capture_polled >= CAPTURE_POLL_S:
                try:
                    await self.poll_capture()
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - poll_capture guards itself
                    pass
            waiting = len(self.handler.buffer) + len(self._retry)
            due = now - self._last_flush >= self.interval_s
            if not waiting and not due:
                continue
            if not due and waiting < self.batch_rows:
                continue
            if now < self._backoff_until:
                continue
            try:
                await self.flush_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # pragma: no cover - flush_once guards itself
                logger.debug("Ops log sink flush failed (%s)", type(e).__name__)

    # -- flushing --------------------------------------------------------------------------
    def _guard(self) -> asyncio.Lock:
        if self._flush_lock is None:
            self._flush_lock = asyncio.Lock()
        return self._flush_lock

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._tokens_at)
        self._tokens_at = now
        if self._row_tokens != float("inf"):
            self._row_tokens = min(
                float(self.batch_rows), self._row_tokens + elapsed * self.rows_per_s
            )

    def _take_batch(self, now: float) -> list[dict[str, Any]]:
        """Up to the row budget: retried rows first, then coalesced new rows."""
        self._refill(now)
        allowed = int(min(self._row_tokens, float(self.batch_rows)))
        if allowed <= 0:
            return []
        batch = self._retry[:allowed]
        self._retry = self._retry[allowed:]
        room = allowed - len(batch)
        if room > 0:
            buffer = self.handler.buffer
            merged: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
            # Pop raw rows while the coalesced batch has room; a row that
            # merges into one already taken is always taken too.
            while buffer:
                row = buffer[0]
                key = _coalesce_key(row)
                if key not in merged and len(merged) >= room:
                    break
                buffer.popleft()
                current = merged.get(key)
                if current is None:
                    merged[key] = dict(row)
                else:
                    current["repeat_count"] += int(row.get("repeat_count") or 1)
                    current["ts"] = max(current["ts"], row["ts"])
            batch.extend(merged.values())
            room = allowed - len(batch)
            if room > 0:
                batch.extend(self.handler.take_suppressed(now, room))
        return batch

    async def flush_once(self) -> bool:
        """Write one budget's worth of rows. True when nothing failed."""
        async with self._guard():
            now = self.clock()
            self._last_flush = now
            batch = self._take_batch(now)
            if not batch:
                self._flushed_ok()
                return True
            batch = await self._drop_deleted(batch)
            if not batch:
                self._flushed_ok()
                return True
            try:
                await asyncio.wait_for(
                    self.store.logs_write(batch), timeout=self.write_timeout_s
                )
            except asyncio.CancelledError:
                self._keep(batch)
                raise
            except Exception as e:
                if _is_data_error(e):
                    return await self._write_each(batch, e)
                self._failed(batch, e)
                return False
            self._written(len(batch))
            return True

    def _written(self, rows: int) -> None:
        if self._row_tokens != float("inf"):
            self._row_tokens = max(0.0, self._row_tokens - rows)
        self.written += rows
        self._flushed_ok()

    def _flushed_ok(self) -> None:
        """A flush pass that failed nothing, an empty one included: the
        health rule reads a quiet sink as fresh, not as stuck."""
        self.last_flush_ok_at = self.clock()
        self.last_flush_error = None
        self._failures_in_row = 0
        self._backoff_until = 0.0

    def _keep(self, batch: list[dict[str, Any]]) -> None:
        self._retry = batch + self._retry
        overflow = len(self._retry) - self.max_retry_rows
        if overflow > 0:
            # The oldest rows go first: the newest say what is happening now.
            self._retry = self._retry[overflow:]
            self.dropped_retry += overflow

    def _failed(self, batch: list[dict[str, Any]], error: BaseException) -> None:
        self._keep(batch)
        self.write_failures += 1
        self.last_flush_error = type(error).__name__
        self._failures_in_row += 1
        delay = min(MAX_BACKOFF_S, 2.0 ** min(self._failures_in_row, 6))
        self._backoff_until = self.clock() + delay
        if self._failures_in_row == 1:
            # Logged on the sink's own logger, which the sink never takes.
            logger.warning(
                "Ops log sink could not write %d rows (%s); keeping them buffered",
                len(batch),
                type(error).__name__,
            )

    async def _write_each(
        self, batch: list[dict[str, Any]], error: BaseException
    ) -> bool:
        """One bad row must not hold back the others."""
        written = 0
        for index, row in enumerate(batch):
            try:
                await asyncio.wait_for(
                    self.store.logs_write([row]), timeout=self.write_timeout_s
                )
                written += 1
            except asyncio.CancelledError:
                self._keep(batch[index:])
                raise
            except Exception as e:
                if not _is_data_error(e):
                    self._failed(batch[index:], e)
                    if written:
                        self._written(written)
                    return False
                self.bad_rows += 1
        if written:
            self._written(written)
        logger.warning(
            "Ops log sink dropped %d rows the database rejected (%s)",
            len(batch) - written,
            type(error).__name__,
        )
        return True

    # -- workspace deletion --------------------------------------------------------------------
    def forget(self, workspace: str) -> None:
        now = self.clock()
        self._forgotten[workspace] = now
        cutoff = now - FORGET_KEEP_S
        for name in [w for w, at in self._forgotten.items() if at < cutoff]:
            self._forgotten.pop(name, None)
        dropped = self.handler.forget(workspace)
        kept = [r for r in self._retry if r.get("workspace") != workspace]
        dropped += len(self._retry) - len(kept)
        self._retry = kept
        self.dropped_deleted += dropped

    async def forget_flushed(self, workspace: str, timeout_s: float = 30.0) -> None:
        """:meth:`forget` once no flush is writing, so none of the dropped
        rows is in flight; without waiting longer than ``timeout_s``."""
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

    async def _drop_deleted(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop rows of workspaces deleted (here or by another worker) after
        the rows were logged, plus a short grace for teardown chatter."""
        names = {r["workspace"] for r in batch if r.get("workspace")}
        if not names:
            return batch
        from lightrag.api.ops.store import deleted_key

        deleted = {name: at for name, at in self._forgotten.items() if name in names}
        keys = {deleted_key(name): name for name in names}
        try:
            found = await asyncio.wait_for(
                self.store.kv_get_many(list(keys)), timeout=self.write_timeout_s
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            found = {}  # the write below fails too and keeps the rows
        for key, value in (found or {}).items():
            at = value.get("at") if isinstance(value, dict) else None
            name = keys.get(key)
            if name is not None and isinstance(at, (int, float)):
                deleted[name] = max(float(at), deleted.get(name, 0.0))
        if not deleted:
            return batch
        kept = [
            r
            for r in batch
            if r.get("workspace") not in deleted
            or r["ts"] > deleted[r["workspace"]] + DELETION_GRACE_S
        ]
        self.dropped_deleted += len(batch) - len(kept)
        return kept

    # -- health ----------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        handler = self.handler
        now = self.clock()
        last_ok = self.last_flush_ok_at
        capture = handler.capture
        return {
            "enabled": True,
            "attached": self.attached,
            "workers": self.workers,
            "level": logging.getLevelName(self.level),
            "capture": {
                "module": capture[0],
                "until": _iso(capture[1]),
                "rows": handler.capture_rows,
                "dropped": handler.capture_dropped,
            }
            if capture is not None
            else None,
            "buffer": len(handler.buffer) + len(self._retry),
            "accepted": handler.accepted,
            "written": self.written,
            "dropped": handler.dropped + self.dropped_retry,
            "rate_limited": handler.rate_limited,
            "self_suppressed": handler.self_suppressed,
            "unauth_warnings": handler.unauth_warnings,
            "write_failures": self.write_failures,
            "bad_rows": self.bad_rows,
            "dropped_deleted": self.dropped_deleted,
            "errors": handler.errors,
            "last_flush_ok_at": _iso(last_ok),
            "last_flush_age_s": round(now - last_ok, 1) if last_ok else None,
            "last_flush_error": self.last_flush_error,
        }


def _iso(epoch: Optional[float]) -> Optional[str]:
    if not isinstance(epoch, (int, float)) or epoch <= 0:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def _is_data_error(error: BaseException) -> bool:
    from lightrag.api.ops.runs import _is_data_error as is_data_error

    return is_data_error(error)


# -- retention (leased) --------------------------------------------------------------


def retention_days() -> int:
    try:
        days = int(os.getenv("OPS_LOG_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS)))
    except ValueError:
        return DEFAULT_RETENTION_DAYS
    return max(1, days)


async def retention_cycle(
    loop: Any, *, store: Any, days: int, clock: Callable[[], float] = time.time
) -> dict[str, Any]:
    """Delete old rows; once a week, count rows that still look unredacted."""
    out: dict[str, Any] = {"deleted": 0}
    token = _in_flush.set(True)
    try:
        out["deleted"] = await store.logs_cleanup(days)
        if out["deleted"]:
            logger.info("Ops log retention removed %d rows", out["deleted"])
        last = await store.kv_get(AUDIT_KEY, None)
        last_at = last.get("at") if isinstance(last, dict) else None
        now = clock()
        if not isinstance(last_at, (int, float)) or now - last_at >= AUDIT_EVERY_S:
            counts = await store.logs_redaction_audit(AUDIT_DAYS)
            matches = {
                k: int(v) for k, v in (counts or {}).items() if k != "rows" and v
            }
            result = {
                "at": now,
                "days": AUDIT_DAYS,
                "rows": int((counts or {}).get("rows") or 0),
                "matches": matches,
                "clean": not matches,
            }
            if await loop.still_held():
                await store.kv_put(AUDIT_KEY, result)
            out["audit"] = result
            if matches:
                logger.warning(
                    "Ops log redaction audit found rows that look unredacted: %s",
                    ", ".join(f"{k}={v}" for k, v in sorted(matches.items())),
                )
    finally:
        _in_flush.reset(token)
    return out
