"""The one audit writer of the Operations console (``LIGHTRAG_OPS_AUDIT``).

Every export, reveal, action and log search goes through :func:`write_audit`.
Rows name the actor, how they signed in, who impersonated them (refused for
mutations, but recorded), the action, an intent or result phase, and a hash
of the client address; never filter values or other free text a caller
typed, which are stored as a hash plus the field names (FR-75).

:class:`SearchAudit` records log searches at most once per actor and filter
hash every five minutes, so a following Logs tab (one poll every 5 s) does
not write an audit row per poll.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from lightrag.api.ops.context import current_request_id

logger = logging.getLogger("lightrag.api.ops")

SEARCH_DEDUP_S = 300.0
SEARCH_DEDUP_MAX = 4096
_AUTH_KIND_MAX = 16


def actor_of(user: Any) -> dict[str, Optional[str]]:
    """``actor``, ``auth_kind`` and ``impersonated_by`` of a signed-in user."""
    metadata = getattr(user, "metadata", None) or {}
    auth_kind = metadata.get("auth_mode") or "token"
    impersonated = metadata.get("impersonated_by")
    return {
        "actor": str(getattr(user, "username", "") or "-")[:255],
        "auth_kind": str(auth_kind)[:_AUTH_KIND_MAX],
        "impersonated_by": str(impersonated)[:255] if impersonated else None,
    }


def client_ip_hash(request: Any) -> Optional[str]:
    client = getattr(request, "client", None) if request is not None else None
    host = getattr(client, "host", None)
    if not host:
        return None
    return hashlib.sha256(str(host).encode("utf-8")).hexdigest()[:16]


def filters_digest(filters: dict[str, Any]) -> dict[str, Any]:
    """``{filter_hash, fields}`` for a set of filters: values never stored."""
    present = {
        str(k): v for k, v in (filters or {}).items() if v is not None and v != ""
    }
    canonical = json.dumps(present, sort_keys=True, default=str)
    return {
        "filter_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "fields": sorted(present),
    }


async def write_audit(
    store: Any,
    *,
    user: Any,
    action: str,
    phase: str = "result",
    request: Any = None,
    workspace: Optional[str] = None,
    target: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    reason: Optional[str] = None,
    outcome: Optional[str] = None,
) -> int:
    """Write one audit row and return its id. Raises when it cannot be written:
    the caller decides whether the action may go on without it."""
    row = {
        **actor_of(user),
        "action": action,
        "phase": phase,
        "workspace": workspace,
        "target": target,
        "params": params or {},
        "reason": reason,
        "outcome": outcome,
        "request_id": current_request_id(),
        "client_ip_hash": client_ip_hash(request),
    }
    return await store.audit_write(row)


class SearchAudit:
    """Audits searches once per actor and filter hash per five minutes."""

    def __init__(
        self,
        window_s: float = SEARCH_DEDUP_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.window_s = window_s
        self.clock = clock
        self._seen: OrderedDict[tuple[str, str, str], float] = OrderedDict()
        self._lock = threading.Lock()
        self.failures = 0

    def _due(self, key: tuple[str, str, str]) -> bool:
        now = self.clock()
        with self._lock:
            last = self._seen.get(key)
            if last is not None and now - last < self.window_s:
                return False
            self._seen[key] = now
            self._seen.move_to_end(key)
            while len(self._seen) > SEARCH_DEDUP_MAX:
                self._seen.popitem(last=False)
            return True

    def _undo(self, key: tuple[str, str, str]) -> None:
        with self._lock:
            self._seen.pop(key, None)

    async def record(
        self,
        store: Any,
        *,
        user: Any,
        action: str,
        filters: dict[str, Any],
        request: Any = None,
        outcome: Optional[str] = "ok",
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        """Best effort: a search is never refused because its audit failed."""
        digest = filters_digest(filters)
        key = (actor_of(user)["actor"] or "-", action, digest["filter_hash"])
        if not self._due(key):
            return None
        try:
            return await write_audit(
                store,
                user=user,
                action=action,
                request=request,
                params={**digest, **(extra or {})},
                outcome=outcome,
            )
        except Exception as e:
            # Not deduplicated, so the next search tries again.
            self._undo(key)
            self.failures += 1
            logger.warning(
                "Ops search audit could not be written (%s)", type(e).__name__
            )
            return None
