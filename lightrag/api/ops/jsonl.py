"""Per-process JSONL log files and the Logs tab's live INFO tail (§3.8.6).

``OPS_JSONL_LOGS=true`` (default off) gives every server process its own
redacted, structured log file in ``LOG_DIR``: ``lightrag-<pid>.jsonl``, one
JSON object per line, INFO and above, rotated by size
(``OPS_JSONL_MAX_BYTES``, default 20 MB, ``OPS_JSONL_BACKUP_COUNT`` older
files, default 2) and created owner-only (0600, never through a symlink).
A file per PID means Gunicorn workers never share or race on a file, and
rotation stays a per-process affair. The handler sits next to the human
``lightrag.log`` on the same loggers as the structured sink (``lightrag``,
``uvicorn.error``, ``gunicorn.error`` and the root logger's library
allowlist) and runs the same redaction and context filters, so nothing
unredacted is ever written. It needs no database.

``GET /admin/ops/logs/tail`` (admin) reads the newest lines back:

- only files whose name matches ``lightrag-<pid>.jsonl[.N]`` are considered,
  found by listing ``LOG_DIR`` (never a name from the request), opened with
  ``O_NOFOLLOW`` and checked to be regular files, so no path can be
  traversed or redirected;
- the sources are the files of live processes, plus those of processes that
  exited in the last :data:`RECENT_EXIT_S` (the lines right before a crash
  are the ones worth seeing). ``LOG_DIR`` outlives containers and PIDs are
  reused, so a file counts as live only while its PID runs *and* that
  process started before the file's last write (psutil create time): a
  previous container's file whose PID now belongs to another process is
  treated as exited (and pruned once old);
- each file is read backwards from its end, at most :data:`MAX_READ_BYTES`
  per source (the newest rotated file too, while the current one is short),
  never whole;
- each source's lines are already in time order, so a k-way merge by
  timestamp yields the newest ``lines`` across processes;
- every message and traceback is run through the redactor again on read.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import re
import stat
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Iterable, Iterator, Optional

from lightrag.api.ops.context import ContextFilter
from lightrag.api.ops.redact import (
    EXC_MAX_CHARS,
    MSG_MAX_CHARS,
    RedactFilter,
    redact_pipeline,
    redact_traceback,
)

logger = logging.getLogger("lightrag.api.ops")

FLAG = "OPS_JSONL_LOGS"
FILE_PREFIX = "lightrag-"
FILE_SUFFIX = ".jsonl"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 2
MAX_BACKUP_COUNT = 10
DEFAULT_LEVEL = logging.INFO
FILE_MODE = 0o600
# Files of exited processes stay readable by the tail this long, and are
# deleted at startup after DEAD_KEEP_S.
RECENT_EXIT_S = 15 * 60.0
DEAD_KEEP_S = 3 * 86400.0
# Tail bounds.
TAIL_DEFAULT_LINES = 200
TAIL_MAX_LINES = 1000
MAX_READ_BYTES = 512 * 1024
MAX_SOURCES = 32
READ_CHUNK = 64 * 1024
# The longest line the writer can produce: msg and exc are capped in
# characters, and JSON escaping can spend up to 6 bytes on one (\u00XX for a
# control character; non-ASCII text is 2-4 bytes of UTF-8), plus the capped
# logger, module, exc_type and context fields. Longer lines are not ours.
LINE_MAX_BYTES = 6 * (MSG_MAX_CHARS + EXC_MAX_CHARS) + 24 * 1024
# A process that started this long after a file's last write cannot have
# written it (create times and mtimes come from different clocks' rounding).
START_SLACK_S = 2.0

# Loggers the handler goes on, and the libraries taken from the root logger
# (the structured sink's choice, so both stores see the same records).
ATTACH_LOGGERS = ("lightrag", "uvicorn.error", "gunicorn.error")
ROOT_ALLOWLIST = ("openai", "httpx", "asyncpg", "pymilvus", "neo4j")
_ACCEPTED = (*ATTACH_LOGGERS, *ROOT_ALLOWLIST)
# Lines the console itself writes about reading the files stay out of them.
_EXCLUDED = ("lightrag.api.ops.sink",)

_FILE_NAME = re.compile(r"^lightrag-(\d{1,10})\.jsonl(?:\.(\d{1,2}))?$")
_CONTEXT = (
    "worker_id",
    "workspace",
    "request_id",
    "run_id",
    "track_id",
    "doc_id",
    "operation",
    "source",
)


def jsonl_enabled() -> bool:
    value = os.getenv(FLAG)
    return bool(value) and value.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError:
        return default
    return max(low, min(high, value))


def log_dir() -> str:
    return os.getenv("LOG_DIR") or os.getcwd()


def file_name(pid: int) -> str:
    return f"{FILE_PREFIX}{int(pid)}{FILE_SUFFIX}"


def _accepted(name: str) -> bool:
    if any(name == x or name.startswith(x + ".") for x in _EXCLUDED):
        return False
    return any(name == root or name.startswith(root + ".") for root in _ACCEPTED)


def _context(record: logging.LogRecord, field: str) -> Optional[str]:
    value = getattr(record, field, None)
    if value is None or value in ("-", ""):
        return None
    return str(value)[:255]


class _Accept(logging.Filter):
    """First filter: only the console's loggers (the root sees everything),
    and each record once (a record of an attached logger that propagates
    reaches the root's copy of the handler too)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_lightrag_jsonl", False) or not _accepted(record.name):
            return False
        record._lightrag_jsonl = True
        return True


class JsonlFileHandler(RotatingFileHandler):
    """One process's redacted JSON-lines log, created 0600, rotated by size."""

    def __init__(
        self,
        path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        level: int = DEFAULT_LEVEL,
    ):
        super().__init__(
            path,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        self.setLevel(level)
        self.addFilter(_Accept())
        # The same hygiene filters as the text file and the structured sink;
        # each marks the record, so the work is done once.
        self.addFilter(RedactFilter())
        self.addFilter(ContextFilter())
        self.written = 0
        self.errors = 0

    def _open(self):  # noqa: D401 - logging API
        """Open owner-only, refusing a symlink planted at the path."""
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(self.baseFilename, flags, FILE_MODE)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(f"{self.baseFilename} is not a regular file")
            if stat.S_IMODE(info.st_mode) != FILE_MODE:
                os.fchmod(fd, FILE_MODE)
        except Exception:
            os.close(fd)
            raise
        return os.fdopen(fd, "a", encoding=self.encoding)

    def format(self, record: logging.LogRecord) -> str:
        from lightrag.api.ops.log_sink import _texts

        msg, exc, exc_type = _texts(record)
        entry: dict[str, Any] = {
            "ts": round(float(record.created), 6),
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": logging.getLevelName(record.levelno),
            "levelno": int(record.levelno),
            "logger": record.name[:128],
            "module": (record.module or "")[:128] or None,
            "lineno": int(record.lineno or 0) or None,
            "pid": record.process or os.getpid(),
        }
        for field in _CONTEXT:
            value = _context(record, field)
            if value is not None:
                entry[field] = value
        entry["msg"] = msg
        if exc_type:
            entry["exc_type"] = exc_type[:128]
        if exc:
            entry["exc"] = exc
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    def emit(self, record: logging.LogRecord) -> None:
        errors = self.errors
        try:
            super().emit(record)
        except Exception:
            self.errors += 1
        if self.errors == errors:
            self.written += 1

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        # Never print a record (possibly unredacted on a failure path) to stderr.
        self.errors += 1


class JsonlLogs:
    """Owns this process's handler (attach, detach, stats)."""

    def __init__(self, handler: JsonlFileHandler):
        self.handler = handler
        self._attached: list[logging.Logger] = []

    def attach(self) -> None:
        if self._attached:
            return
        for target in [logging.getLogger(n) for n in ATTACH_LOGGERS] + [
            logging.getLogger()
        ]:
            if self.handler not in target.handlers:
                target.addHandler(self.handler)
            self._attached.append(target)

    def detach(self) -> None:
        for target in self._attached:
            target.removeHandler(self.handler)
        self._attached = []
        try:
            self.handler.close()
        except Exception:
            pass

    def stats(self) -> dict[str, Any]:
        return {
            "path": os.path.basename(self.handler.baseFilename),
            "written": self.handler.written,
            "errors": self.handler.errors,
            "level": logging.getLevelName(self.handler.level),
        }


def start_jsonl(
    directory: Optional[str] = None, pid: Optional[int] = None
) -> JsonlLogs:
    """Create and attach this process's handler (raises on a bad LOG_DIR)."""
    directory = directory or log_dir()
    os.makedirs(directory, mode=0o700, exist_ok=True)
    prune_dead_files(directory)
    handler = JsonlFileHandler(
        os.path.join(directory, file_name(pid or os.getpid())),
        max_bytes=_env_int(
            "OPS_JSONL_MAX_BYTES", DEFAULT_MAX_BYTES, 1024 * 1024, 1024**3
        ),
        backup_count=_env_int(
            "OPS_JSONL_BACKUP_COUNT", DEFAULT_BACKUP_COUNT, 1, MAX_BACKUP_COUNT
        ),
    )
    logs = JsonlLogs(handler)
    logs.attach()
    return logs


# -- reading back -----------------------------------------------------------------------


def pid_started(pid: int) -> Optional[float]:
    """When the process holding ``pid`` started (epoch seconds), None when no
    such process runs (same PID namespace). 0.0 when it runs but its start
    time is unreadable, so it is never taken for a newer process."""
    if pid <= 0:
        return None
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a server dependency
        psutil = None
    if psutil is not None:
        try:
            return float(psutil.Process(pid).create_time())
        except psutil.NoSuchProcess:
            return None
        except Exception:
            return 0.0
    try:  # pragma: no cover
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return 0.0
    except OSError:
        return None
    return 0.0


def writer_alive(pid: int, last_write: float) -> bool:
    """Whether the process that wrote a file (last modified ``last_write``)
    still runs: its PID is taken and the holder started before that write.
    A PID reused after a restart (``LOG_DIR`` is a persisted volume) belongs
    to a process that started later and never wrote the old file."""
    started = pid_started(pid)
    return started is not None and started <= last_write + START_SLACK_S


def _list_files(directory: str) -> dict[int, list[tuple[int, str, float, int]]]:
    """``{pid: [(rotation, name, mtime, size)]}`` of regular JSONL files,
    current file first. Only names matching the pattern are listed."""
    found: dict[int, list[tuple[int, str, float, int]]] = {}
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return found
    for entry in entries:
        match = _FILE_NAME.match(entry.name)
        if not match:
            continue
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        pid = int(match.group(1))
        rotation = int(match.group(2) or 0)
        found.setdefault(pid, []).append(
            (rotation, entry.name, info.st_mtime, info.st_size)
        )
    for files in found.values():
        files.sort()
    return found


def prune_dead_files(
    directory: str,
    *,
    now: Optional[float] = None,
    alive: Callable[[int, float], bool] = writer_alive,
) -> int:
    """Delete the files of processes that are gone and quiet for days."""
    now = time.time() if now is None else now
    removed = 0
    for pid, files in _list_files(directory).items():
        newest = max(f[2] for f in files)
        if pid == os.getpid() or alive(pid, newest):
            continue
        if now - newest < DEAD_KEEP_S:
            continue
        for _, name, _, _ in files:
            try:
                os.unlink(os.path.join(directory, name))
                removed += 1
            except OSError:
                pass
    return removed


def _open_regular(directory: str, name: str) -> Optional[int]:
    """A read-only fd for ``directory/name`` if it is a regular file (never a
    symlink); None otherwise. ``name`` always comes from :func:`_list_files`."""
    if not _FILE_NAME.match(name) or os.sep in name or name.startswith("."):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(os.path.join(directory, name), flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        return None
    return fd


def read_tail_lines(fd: int, budget: int) -> tuple[list[bytes], int, bool]:
    """Complete lines from the end of a file, newest last, reading at most
    ``budget`` bytes backwards. Returns ``(lines, bytes_read, truncated)``;
    ``truncated`` is True when older content was left unread."""
    size = os.fstat(fd).st_size
    if size <= 0 or budget <= 0:
        return [], 0, size > 0
    start = max(0, size - budget)
    read = 0
    chunks: list[bytes] = []
    position = size
    while position > start:
        step = min(READ_CHUNK, position - start)
        position -= step
        chunks.append(os.pread(fd, step, position))
        read += step
    data = b"".join(reversed(chunks))
    truncated = start > 0
    if truncated:
        # The first line is cut where the read began: drop it.
        cut = data.find(b"\n")
        data = data[cut + 1 :] if cut >= 0 else b""
    lines = [line for line in data.split(b"\n") if line.strip()]
    return lines, read, truncated


def _parse(line: bytes) -> Optional[dict[str, Any]]:
    if len(line) > LINE_MAX_BYTES:
        return None
    try:
        entry = json.loads(line)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("ts"), (int, float)):
        return None
    return entry


def entry_view(entry: dict[str, Any]) -> dict[str, Any]:
    """One tail line as returned: known fields only, redacted again."""
    level = entry.get("levelno")
    level = int(level) if isinstance(level, (int, float)) else 0
    out: dict[str, Any] = {
        "ts": datetime.fromtimestamp(float(entry["ts"]), timezone.utc).isoformat(),
        "level": level,
        "level_name": logging.getLevelName(level) if level else entry.get("level"),
        "logger": str(entry.get("logger") or "")[:128],
        "module": _text(entry.get("module"), 128),
        "lineno": entry.get("lineno") if isinstance(entry.get("lineno"), int) else None,
        "pid": entry.get("pid") if isinstance(entry.get("pid"), int) else None,
    }
    for field in _CONTEXT:
        out[field] = _text(entry.get(field), 255)
    out["msg"] = redact_pipeline(str(entry.get("msg") or ""), MSG_MAX_CHARS)
    out["exc_type"] = _text(entry.get("exc_type"), 128)
    exc = entry.get("exc")
    out["exc"] = redact_traceback(str(exc), EXC_MAX_CHARS) if exc else None
    return out


def _text(value: Any, limit: int) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)[:limit]


def _matches(entry: dict[str, Any], level: int, module: Optional[str]) -> bool:
    levelno = entry.get("levelno")
    if not isinstance(levelno, (int, float)) or levelno < level:
        return False
    if module:
        name = str(entry.get("logger") or "")
        if entry.get("module") != module and not (
            name == module or name.startswith(module + ".")
        ):
            return False
    return True


def _source_lines(
    directory: str, files: list[tuple[int, str, float, int]], budget: int
) -> tuple[list[dict[str, Any]], int, bool]:
    """Parsed entries of one process (oldest first) from its current file
    and, while budget remains, its newest rotated file."""
    entries: list[dict[str, Any]] = []
    total = 0
    truncated = False
    for _, name, _, _ in files[:2]:
        if budget - total <= 0:
            truncated = True
            break
        fd = _open_regular(directory, name)
        if fd is None:
            continue
        try:
            raw, read, cut = read_tail_lines(fd, budget - total)
        finally:
            os.close(fd)
        total += read
        parsed = [e for e in (_parse(line) for line in raw) if e is not None]
        # Older content goes first.
        entries = parsed + entries
        if cut:
            truncated = True
            break
    return entries, total, truncated


def _newest_first(entries: list[dict[str, Any]]) -> Iterator[tuple[float, int, dict]]:
    for position in range(len(entries) - 1, -1, -1):
        entry = entries[position]
        yield (-float(entry["ts"]), position, entry)


def tail(
    directory: Optional[str] = None,
    *,
    lines: int = TAIL_DEFAULT_LINES,
    level: int = DEFAULT_LEVEL,
    module: Optional[str] = None,
    now: Optional[float] = None,
    alive: Callable[[int, float], bool] = writer_alive,
    max_read_bytes: int = MAX_READ_BYTES,
) -> dict[str, Any]:
    """The newest ``lines`` entries across live (and recently exited)
    processes, newest first. Blocking: run it in a worker thread."""
    directory = directory or log_dir()
    now = time.time() if now is None else now
    lines = max(1, min(int(lines), TAIL_MAX_LINES))
    listed = _list_files(directory)
    candidates: list[tuple[float, int, bool, list]] = []
    for pid, files in listed.items():
        newest = max(f[2] for f in files)
        is_alive = pid == os.getpid() or alive(pid, newest)
        if not is_alive and now - newest > RECENT_EXIT_S:
            continue
        candidates.append((newest, pid, is_alive, files))
    candidates.sort(reverse=True)
    sources = []
    streams: list[Iterable[tuple[float, int, dict]]] = []
    for newest, pid, is_alive, files in candidates[:MAX_SOURCES]:
        entries, read, truncated = _source_lines(directory, files, max_read_bytes)
        # A process's lines are in write order; threads can interleave
        # timestamps by a hair, so each source is sorted before the merge.
        kept = sorted(
            (e for e in entries if _matches(e, level, module)),
            key=lambda e: float(e["ts"]),
        )
        sources.append(
            {
                "pid": pid,
                "alive": is_alive,
                "bytes_read": read,
                "truncated": truncated,
                "lines": len(kept),
                "modified_at": datetime.fromtimestamp(newest, timezone.utc).isoformat(),
            }
        )
        streams.append(_newest_first(kept))
    merged: list[dict[str, Any]] = []
    for _, _, entry in heapq.merge(*streams, key=lambda item: item[0]):
        merged.append(entry_view(entry))
        if len(merged) >= lines:
            break
    return {
        "items": merged,
        "sources": sources,
        "sources_capped": len(candidates) > MAX_SOURCES,
        "lines": lines,
        "newest_ts": merged[0]["ts"] if merged else None,
    }


__all__ = [
    "FLAG",
    "JsonlFileHandler",
    "JsonlLogs",
    "jsonl_enabled",
    "prune_dead_files",
    "read_tail_lines",
    "start_jsonl",
    "tail",
    "writer_alive",
]
