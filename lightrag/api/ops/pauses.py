"""Provider pauses for the Operations console, read from a capped candidate set.

A pause lives in each workspace's shared ``pipeline_status`` (key
``provider_rejection``), so finding every paused workspace would mean reading
every workspace's shared state. Instead at most :data:`MAX_CANDIDATES`
workspaces are checked, chosen in SQL: first those whose latest
pause-related event is a rejection that paused ingestion (Insights events),
then those with pending documents, most recent first. The result is labelled
approximate. Pauses are grouped by slot, and pauses on a workspace's own key
(BYO) are included and flagged.

This reader is independent of the provider routes (mounted only with
workspace providers enabled) and never decrypts more than the capped
candidates' provider records; a record is decrypted again only when its file
changes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from lightrag.api.ops.redact import redact
from lightrag.api.ops.snapshot import PAUSE_KEYS, NamespaceReader

logger = logging.getLogger("lightrag.api.ops")

MAX_CANDIDATES = 50
EVENT_DAYS = 7
SLOTS = ("extraction", "query", "vision")


def _role_slots() -> dict[str, str]:
    try:
        from lightrag.api.workspace_providers import SLOT_ROLES
    except Exception:  # pragma: no cover - optional dependency missing
        return {
            "extract": "extraction",
            "keyword": "extraction",
            "query": "query",
            "vlm": "vision",
        }
    return {role: slot for slot, roles in SLOT_ROLES.items() for role in roles}


def _iso_epoch(value: Any) -> Optional[str]:
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return None


class OwnProviderCache:
    """Which slots a workspace runs on its own key, cached by record mtime."""

    def __init__(self, provider_store: Any):
        self._store = provider_store
        self._cache: dict[str, tuple[float, frozenset[str]]] = {}

    def _read(self, workspace: str) -> frozenset[str]:
        mtime = self._store.record_mtime(workspace)
        if mtime is None:
            self._cache.pop(workspace, None)
            return frozenset()
        cached = self._cache.get(workspace)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        config = self._store.get(workspace)
        slots = frozenset(
            name
            for name in SLOTS
            if config is not None and config.slot(name).is_active()
        )
        self._cache[workspace] = (mtime, slots)
        if len(self._cache) > 4 * MAX_CANDIDATES:
            self._cache.pop(next(iter(self._cache)))
        return slots

    async def slots(self, workspaces: list[str]) -> dict[str, frozenset[str]]:
        def read_all() -> dict[str, frozenset[str]]:
            out = {}
            for workspace in workspaces:
                try:
                    out[workspace] = self._read(workspace)
                except Exception as e:
                    logger.warning(
                        "Ops pause reader could not read a provider record (%s)",
                        type(e).__name__,
                    )
                    out[workspace] = frozenset()
            return out

        if not workspaces:
            return {}
        return await asyncio.to_thread(read_all)


async def read_pauses(
    store: Any,
    reader: NamespaceReader,
    *,
    namespace_of: Callable[[str], str],
    own_providers: Optional[OwnProviderCache] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Pauses among the capped candidates, grouped by slot and kind."""
    candidates = list(await store.paused_candidates(EVENT_DAYS, MAX_CANDIDATES))
    if len(candidates) < MAX_CANDIDATES:
        for workspace in await store.pending_candidates(MAX_CANDIDATES):
            if workspace not in candidates:
                candidates.append(workspace)
            if len(candidates) >= MAX_CANDIDATES:
                break
    names = {workspace: namespace_of(workspace) for workspace in candidates}
    batch = await reader.snapshot_many(list(names.values()), PAUSE_KEYS)
    role_slots = _role_slots()
    current = now if now is not None else datetime.now(timezone.utc).timestamp()

    paused: list[tuple[str, str, dict[str, Any]]] = []
    for workspace in candidates:
        snap = batch.data.get(names[workspace]) or {}
        pause = snap.get("provider_rejection")
        if not isinstance(pause, dict) or not pause:
            continue
        slot = role_slots.get(pause.get("role"), pause.get("role") or "unknown")
        paused.append((workspace, slot, pause))

    own = (
        await own_providers.slots([w for w, _, _ in paused])
        if own_providers is not None
        else {}
    )
    items = []
    by_slot_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for workspace, slot, pause in paused:
        kind = pause.get("kind") or "unknown"
        retry_at = pause.get("retry_at")
        items.append(
            {
                "workspace": workspace,
                "slot": slot,
                "role": pause.get("role"),
                "kind": kind,
                "status_code": pause.get("status_code"),
                "byo": slot in own.get(workspace, frozenset()),
                "since": _iso_epoch(pause.get("since")),
                "retry_at": _iso_epoch(retry_at),
                "retry_due": isinstance(retry_at, (int, float)) and retry_at <= current,
                "message": redact(pause.get("message"), 300) or None,
            }
        )
        by_slot_kind[slot][kind] += 1
    items.sort(key=lambda i: (i["since"] or "", i["workspace"]))
    return {
        "items": items,
        "by_slot_kind": {slot: dict(kinds) for slot, kinds in by_slot_kind.items()},
        "total": len(items),
        "checked": len(candidates),
        "capped": len(candidates) >= MAX_CANDIDATES,
        "responsive": batch.responsive,
    }
