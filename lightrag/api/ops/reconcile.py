"""Leased host-wide evaluation of the attention rules.

One worker per host (the holder of ``ops:reconcile``) evaluates every
attention rule once a minute and stores the result in ``LIGHTRAG_OPS_KV``
(``attention``). The Overview of every worker reads that snapshot instead of
evaluating the rules per request, so all workers show the same list and rules
that must hold for consecutive evaluations (``workers_below_expected``,
``pg_pool_wait``, ``pg_connections``, ``manager_rtt``) keep
their streaks across requests and workers. When the snapshot is missing or
older than three cycles, the Overview falls back to evaluating the rules
itself.

Orphaned runs, busy-without-run and reservation checks join this cycle with
run tracking; it never clears ``busy`` or reservations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.service import ATTENTION_KEY

logger = logging.getLogger("lightrag.api.ops")

RECONCILE_LEASE = "ops:reconcile"
RECONCILE_LEASE_TTL_S = 120.0
RECONCILE_INTERVAL_S = 60.0
# The cycle runs a handful of cached, 3 s-bounded reads.
RECONCILE_CYCLE_TIMEOUT_S = 90.0


async def attention_cycle(
    loop: LeasedLoop, *, service: Any, process: Callable[[], dict[str, Any]]
) -> None:
    """Evaluate the rules and publish the snapshot while the lease is ours."""
    snapshot = await service.compute_attention_snapshot(process(), loop.holder)
    if not await loop.still_held():
        return
    await service.store.kv_put(ATTENTION_KEY, snapshot)
    service.cache.forget(("attention",))
