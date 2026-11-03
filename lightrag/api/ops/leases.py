"""Host-wide singleton loops for the Operations console.

Under Gunicorn every worker runs the same background loops; a lease makes sure
only one of them does the work each cycle. A lease is a ``lease:<name>`` row
in ``LIGHTRAG_OPS_KV`` holding ``{holder, exp}``: it can be taken when it is
free or expired, and renewed by its holder. Each lease outlasts its loop's
worst-case cycle (the sum of the cycle's timeouts), the holder renews it at
the start of every cycle, and a cycle that cannot renew does not run. Long
cycles call :meth:`LeasedLoop.still_held` before writing their results.

A loop never raises into the server: every failure is logged and the next
cycle tries again.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("lightrag.api.ops")

Cycle = Callable[["LeasedLoop"], Awaitable[None]]


def being_cancelled() -> bool:
    """Whether the running task itself has a pending cancellation request.

    Distinguishes a real ``task.cancel()`` from a ``CancelledError`` that
    leaked out of an awaited future someone else cancelled. Python 3.10 has
    no ``Task.cancelling()``; there every CancelledError counts as real.
    """
    task = asyncio.current_task()
    if task is None:
        return True
    cancelling = getattr(task, "cancelling", None)
    if cancelling is None:
        return True
    return cancelling() > 0


class LeasedLoop:
    def __init__(
        self,
        store: Any,
        name: str,
        holder: str,
        cycle: Cycle,
        *,
        ttl_s: float,
        interval_s: float,
        cycle_timeout_s: float,
        initial_delay_s: float = 0.0,
    ):
        if ttl_s <= cycle_timeout_s:
            raise ValueError(
                f"Lease {name}: ttl {ttl_s}s must outlast the cycle timeout {cycle_timeout_s}s"
            )
        self.store = store
        self.name = name
        self.holder = holder
        self._cycle = cycle
        self.ttl_s = ttl_s
        self.interval_s = interval_s
        self.cycle_timeout_s = cycle_timeout_s
        self.initial_delay_s = initial_delay_s
        self.held = False
        self.cycles = 0
        self.failures = 0
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    async def renew(self) -> bool:
        """Take or renew the lease; False (never raises) when it is not ours."""
        try:
            self.held = bool(
                await self.store.try_lease(self.name, self.holder, self.ttl_s)
            )
        except Exception as e:
            logger.warning(
                "Ops lease %s could not be renewed (%s)", self.name, type(e).__name__
            )
            self.held = False
        return self.held

    async def still_held(self) -> bool:
        """For long cycles: renew before writing, and stop when it fails."""
        return await self.renew()

    async def run_once(self) -> bool:
        """One cycle if the lease is ours. Returns whether the cycle ran."""
        if not await self.renew():
            return False
        try:
            await asyncio.wait_for(self._cycle(self), timeout=self.cycle_timeout_s)
            self.cycles += 1
        except asyncio.CancelledError:
            if being_cancelled():
                raise
            # A future cancelled by someone else inside the cycle: one bad
            # cycle must not end a singleton loop for the life of the process.
            self.failures += 1
            logger.warning("Ops loop %s cycle was cancelled from inside", self.name)
        except asyncio.TimeoutError:
            self.failures += 1
            logger.warning(
                "Ops loop %s exceeded its %.0fs cycle timeout",
                self.name,
                self.cycle_timeout_s,
            )
        except Exception:
            self.failures += 1
            logger.warning("Ops loop %s failed", self.name, exc_info=True)
        return True

    async def _run(self) -> None:
        if self.initial_delay_s:
            await asyncio.sleep(self.initial_delay_s)
        while not self._stopping:
            await self.run_once()
            await asyncio.sleep(self.interval_s)

    def start(self) -> asyncio.Task:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name=f"ops-{self.name}")
        return self._task

    async def stop(self) -> None:
        self._stopping = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.held:
            try:
                await self.store.release_lease(self.name, self.holder)
            except Exception as e:
                logger.debug(
                    "Ops lease %s release failed (%s)", self.name, type(e).__name__
                )
            self.held = False


class LocalLeases:
    """Stand-in lease store for a loop that runs without the ops database.

    Every lease is granted: without shared storage each process does the
    work for itself (only the probes run this way).
    """

    async def try_lease(self, name: str, holder: str, ttl_s: float) -> bool:
        return True

    async def release_lease(self, name: str, holder: str) -> None:
        return None
