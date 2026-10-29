"""Durable queue for email re-syncs that arrive while the pipeline is busy.

Re-syncing an email that is already stored first deletes its previous
documents, which needs the workspace's exclusive (destructive) pipeline slot.
While documents are being processed that slot is unavailable. The endpoint has
already answered "accepted", so instead of dropping the update the prepared
documents are written here and applied as soon as the pipeline is free.

Layout: ``<working_dir>/.email_resync_queue/<workspace>/<sha256(bundle)>.json``,
files ``0600`` in ``0700`` directories (the same trust boundary as the
workspace's other stored content). One file per bundle: a newer re-sync of the
same email replaces a queued older one, so the latest version wins. A per-
workspace ``flock`` keeps multiple server processes from draining twice.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from lightrag.utils import logger

_SUBDIR = ".email_resync_queue"
_DEFAULT_WORKSPACE_DIR = "_default"
_SAFE_WORKSPACE = re.compile(r"[A-Za-z0-9_.\-]{1,128}")
_RECORD_VERSION = 1

GetRag = Callable[[str], Awaitable[Any]]


class DeferredResyncQueue:
    def __init__(
        self,
        working_dir: str,
        get_rag: GetRag,
        *,
        min_backoff: float = 5.0,
        max_backoff: float = 120.0,
        batch_size: int = 10,
    ):
        self._base = Path(working_dir) / _SUBDIR
        self._get_rag = get_rag
        self._min_backoff = min_backoff
        self._max_backoff = max_backoff
        self._batch_size = batch_size
        self._drainers: dict[str, asyncio.Task] = {}
        self._background: set[asyncio.Task] = set()
        self._closed = False

    # -- paths -----------------------------------------------------------
    def _dir(self, workspace: str) -> Path:
        name = workspace or _DEFAULT_WORKSPACE_DIR
        if not _SAFE_WORKSPACE.fullmatch(name) or name in (".", ".."):
            raise ValueError(f"Unsafe workspace name for resync queue: {name!r}")
        return self._base / name

    def _path(self, workspace: str, prefix_base: str) -> Path:
        digest = hashlib.sha256(prefix_base.encode("utf-8")).hexdigest()[:40]
        return self._dir(workspace) / f"{digest}.json"

    # -- public API ------------------------------------------------------
    def has_pending(self, workspace: str, prefix_base: str) -> bool:
        """True when an older re-sync of this bundle is still queued.

        Callers must then queue the new version too, so versions apply in order.
        """
        return self._path(workspace, prefix_base).exists()

    def pending_count(self, workspace: str) -> int:
        directory = self._dir(workspace)
        return len(list(directory.glob("*.json"))) if directory.exists() else 0

    async def defer(
        self,
        workspace: str,
        prefix_base: str,
        track_id: str,
        components: list[tuple[str, str]],
    ) -> None:
        record = {
            "version": _RECORD_VERSION,
            "id": uuid4().hex,
            "workspace": workspace,
            "prefix_base": prefix_base,
            "track_id": track_id,
            "components": [[content, fp] for content, fp in components],
            "queued_at": time.time(),
        }
        path = self._path(workspace, prefix_base)
        await asyncio.to_thread(_write_private_json, path, record)
        logger.warning(
            f"[{prefix_base}] Pipeline busy: email queued "
            f"({len(components)} document(s)); it is applied when the "
            f"pipeline is free"
        )
        self.ensure_draining(workspace)

    def ensure_draining(self, workspace: str) -> None:
        if self._closed:
            return
        task = self._drainers.get(workspace)
        if task is not None and not task.done():
            return
        self._drainers[workspace] = asyncio.create_task(self._drain(workspace))

    async def start(self) -> None:
        """Resume draining anything left over from a previous run."""
        if not self._base.exists():
            return
        for directory in self._base.iterdir():
            if directory.is_dir() and any(directory.glob("*.json")):
                workspace = (
                    "" if directory.name == _DEFAULT_WORKSPACE_DIR else directory.name
                )
                logger.info(
                    f"Resuming {len(list(directory.glob('*.json')))} queued email "
                    f"re-sync(s) for workspace '{directory.name}'"
                )
                self.ensure_draining(workspace)

    async def stop(self) -> None:
        self._closed = True
        tasks = list(self._drainers.values()) + list(self._background)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- draining --------------------------------------------------------
    async def _drain(self, workspace: str) -> None:
        backoff = self._min_backoff
        directory = self._dir(workspace)
        while not self._closed:
            paths = sorted(directory.glob("*.json"), key=_mtime)
            if not paths:
                return
            progressed = False
            lock_fd = _try_flock(directory)
            if lock_fd is not None:
                try:
                    rag = await self._get_rag(workspace)
                    progressed = await self._apply_batch(
                        rag, workspace, paths[: self._batch_size]
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Queued email re-sync drain failed: {e}")
                finally:
                    _release_flock(lock_fd)
            if progressed:
                backoff = self._min_backoff
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    async def _apply_batch(self, rag: Any, workspace: str, paths: list[Path]) -> bool:
        """Delete the old documents of each queued bundle, then enqueue the new.

        Returns False when the destructive slot is unavailable (retry later).
        """
        # Lazy imports: email_routes imports this module.
        from lightrag.api.routers.document_routes import (
            _acquire_destructive_busy,
            _release_destructive_busy,
        )
        from lightrag.api.routers.email_routes import (
            _bundle_doc_ids,
            _get_bundle_lock,
            _mark_deletion_job,
        )
        from lightrag.base import DocStatus
        from lightrag.parser_routing import resolve_chunk_options

        acquired, _reason = await _acquire_destructive_busy(rag)
        if not acquired:
            return False

        loaded: list[tuple[Path, dict]] = []
        try:
            all_docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
            records = [(p, r) for p in paths if (r := _load_record(p)) is not None]
            stale = {
                path: _bundle_doc_ids(all_docs, f"{record['prefix_base']}__")
                for path, record in records
            }
            await _mark_deletion_job(
                rag,
                f"queue ({len(records)} re-synced emails)",
                sum(len(ids) for ids in stale.values()),
            )
            for path, record in records:
                prefix_base = record["prefix_base"]
                lock = await _get_bundle_lock(f"{rag.workspace}:{prefix_base}")
                async with lock:
                    for did in stale[path]:
                        res = await rag.adelete_by_doc_id(did, delete_llm_cache=False)
                        status = getattr(res, "status", None)
                        if status not in ("success", "not_found"):
                            logger.warning(
                                f"[{prefix_base}] delete {did}: {status} "
                                f"{getattr(res, 'message', '')}"
                            )
                loaded.append((path, record))
        finally:
            await _release_destructive_busy(rag)

        chunk_opts = resolve_chunk_options(rag.addon_params)
        for path, record in loaded:
            prefix_base = record["prefix_base"]
            lock = await _get_bundle_lock(f"{rag.workspace}:{prefix_base}")
            async with lock:
                components = record["components"]
                await rag.apipeline_enqueue_documents(
                    [content for content, _ in components],
                    file_paths=[fp for _, fp in components],
                    track_id=record["track_id"],
                    chunk_options=chunk_opts,
                )
                # A newer re-sync may have replaced the file meanwhile; keep it.
                current = _load_record(path)
                if current is not None and current.get("id") == record["id"]:
                    path.unlink(missing_ok=True)
            logger.info(f"[{prefix_base}] Queued email re-sync applied")

        if loaded:
            task = asyncio.create_task(rag.apipeline_process_enqueue_documents())
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return True


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _write_private_json(path: Path, record: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _load_record(path: Path) -> Optional[dict]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        # Keep the bytes for inspection but stop retrying them forever.
        logger.error(f"Unreadable queued email re-sync {path.name}: {e}")
        path.rename(path.with_suffix(".bad"))
        return None
    if (
        record.get("version") != _RECORD_VERSION
        or not isinstance(record.get("components"), list)
        or not record.get("prefix_base")
    ):
        logger.error(f"Malformed queued email re-sync {path.name}; set aside")
        path.rename(path.with_suffix(".bad"))
        return None
    return record


def _try_flock(directory: Path) -> Optional[int]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(directory / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_flock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
