"""Email re-sync queue depth, scanned once a minute for the whole host.

The queue keeps one file per email bundle under
``<working_dir>/.email_resync_queue/<workspace>/`` (see
:mod:`lightrag.api.email_resync_queue`). The scan only lists directories: it
counts ``*.json`` (waiting) and ``*.bad`` (unreadable, set aside) files per
workspace. It never opens a file, so queued content can never reach the
dashboard, and it never takes the drainers' ``flock``.

It runs as a leased singleton (one worker per host per cycle) in a thread, and
publishes ``email_queue_summary`` to ``LIGHTRAG_OPS_KV``. Because a queue file
is rewritten with a fresh timestamp on every defer, how long a workspace has
been waiting is tracked separately: ``first_pending_at`` is set when the scan
first sees pending files and cleared when the count drops to zero, both in
the summary and in ``resync:<ws>``. Re-syncs keep arriving while the queue
drains, so the queue being non-empty is not a stall: ``last_progress_at`` is
the last scan that saw a workspace's count fall (or first saw it pending), and
``longest_stalled_s`` is how long the slowest workspace has gone without one.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from lightrag.api.email_resync_queue import workspace_for_dir

SCAN_LEASE = "ops:scan"
SCAN_LEASE_TTL_S = 90.0
SCAN_INTERVAL_S = 60.0
# Directory walk plus three small KV statements; well under the lease TTL.
SCAN_CYCLE_TIMEOUT_S = 60.0
SUMMARY_KEY = "email_queue_summary"
MAX_ITEMS = 5000


@dataclass(frozen=True)
class QueueCounts:
    pending: int
    bad: int


def scan_queue_dir(base: Path) -> dict[str, QueueCounts]:
    """Counts by workspace for every workspace with waiting or bad files.

    Lists entries only (``os.scandir``); no file is opened or locked.
    Dot-files (``.lock``, temporary ``.*.tmp`` writes) are ignored.
    """
    out: dict[str, QueueCounts] = {}
    try:
        top = os.scandir(base)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return out
    with top:
        for entry in top:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            workspace = workspace_for_dir(entry.name)
            if workspace is None:
                continue
            pending = bad = 0
            try:
                with os.scandir(entry.path) as files:
                    for item in files:
                        name = item.name
                        if name.startswith("."):
                            continue
                        if name.endswith(".json"):
                            pending += 1
                        elif name.endswith(".bad"):
                            bad += 1
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                continue
            if pending or bad:
                out[workspace] = QueueCounts(pending, bad)
    return out


def build_summary(
    counts: dict[str, QueueCounts],
    first_pending: dict[str, float],
    *,
    now: float,
    holder: str,
    last_progress: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    progress = last_progress or {}
    items = []
    for workspace in sorted(counts, key=lambda w: (-counts[w].pending, w)):
        c = counts[workspace]
        items.append(
            {
                "workspace": workspace,
                "pending": c.pending,
                "bad": c.bad,
                "first_pending_at": first_pending.get(workspace) if c.pending else None,
                "last_progress_at": progress.get(workspace) if c.pending else None,
            }
        )
    starts = [i["first_pending_at"] for i in items if i["first_pending_at"]]
    progressed = [i["last_progress_at"] for i in items if i["last_progress_at"]]
    return {
        "scanned_at": now,
        "holder": holder,
        "total": sum(c.pending for c in counts.values()),
        "workspaces": sum(1 for c in counts.values() if c.pending),
        "bad": sum(c.bad for c in counts.values()),
        "longest_pending_s": max(0.0, now - min(starts)) if starts else None,
        "longest_stalled_s": max(0.0, now - min(progressed)) if progressed else None,
        "truncated": len(items) > MAX_ITEMS,
        "items": items[:MAX_ITEMS],
    }


async def scan_cycle(
    loop: Any,
    *,
    store: Any,
    base_dir: Path,
    clock: Callable[[], float] = time.time,
) -> Optional[dict[str, Any]]:
    """One scan; returns the published summary (None when the lease was lost)."""
    counts = await asyncio.to_thread(scan_queue_dir, base_dir)
    # The walk can take a while on a slow disk: write only while still holder.
    if not await loop.still_held():
        return None
    previous = await store.kv_get(SUMMARY_KEY) or {}
    previous_items = {
        item["workspace"]: item
        for item in previous.get("items") or []
        if isinstance(item, dict) and isinstance(item.get("workspace"), str)
    }
    known: dict[str, float] = {
        w: item["first_pending_at"]
        for w, item in previous_items.items()
        if item.get("first_pending_at")
    }
    progress: dict[str, float] = {
        w: item["last_progress_at"]
        for w, item in previous_items.items()
        if item.get("last_progress_at")
    }
    now = clock()
    started = [w for w, c in counts.items() if c.pending and w not in known]
    if started:
        # A lost or truncated summary must not reset how long they have waited.
        stored = await store.kv_get_many([f"resync:{w}" for w in started])
        for workspace in started:
            value = stored.get(f"resync:{workspace}") or {}
            if isinstance(value, dict) and value.get("first_pending_at"):
                known[workspace] = value["first_pending_at"]
                if value.get("last_progress_at"):
                    progress[workspace] = value["last_progress_at"]
    patches: dict[str, dict[str, Any]] = {}
    first_pending: dict[str, float] = {}
    last_progress: dict[str, float] = {}
    for workspace, c in counts.items():
        if not c.pending:
            continue
        first = known.get(workspace)
        last = progress.get(workspace) or first
        before = previous_items.get(workspace, {}).get("pending")
        if first is None:
            first = last = now
            patches[f"resync:{workspace}"] = {
                "first_pending_at": now,
                "last_progress_at": now,
            }
        elif isinstance(before, int) and c.pending < before:
            # Fewer files than a minute ago: a drain went through.
            last = now
            patches[f"resync:{workspace}"] = {"last_progress_at": now}
        first_pending[workspace] = first
        last_progress[workspace] = last
    for workspace in known:
        if workspace not in first_pending:
            patches[f"resync:{workspace}"] = {
                "first_pending_at": None,
                "last_progress_at": None,
            }
    summary = build_summary(
        counts,
        first_pending,
        now=now,
        holder=loop.holder,
        last_progress=last_progress,
    )
    if patches:
        await store.kv_merge_many(patches)
    await store.kv_put(SUMMARY_KEY, summary)
    return summary
