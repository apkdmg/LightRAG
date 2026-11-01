"""Buffers usage records and writes them in batches.

A write failure never reaches the LLM call that produced the record. Records are
kept and retried; once the buffer is full, new ones are counted as lost so the
page can show what share of usage is missing.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

from lightrag.usage import (
    SystemEvent,
    UsageRecord,
    add_event_sink,
    add_usage_sink,
    remove_event_sink,
    remove_usage_sink,
)
from lightrag.utils import logger

from .store import UsageStore

FLUSH_SECONDS = 5.0
BATCH_SIZE = 2000
MAX_BUFFER = 50_000
LOST_ROWS_KEY = "lost_rows"


def deleted_workspace_label(workspace: str, ts: float) -> str:
    """A short hash in place of the name; it stays unique if the ID is reused."""
    digest = hashlib.sha256(f"{workspace}:{ts}".encode()).hexdigest()
    return f"deleted-{digest[:8]}"


class UsageRecorder:
    def __init__(
        self,
        store: UsageStore,
        *,
        flush_seconds: float = FLUSH_SECONDS,
        max_buffer: int = MAX_BUFFER,
    ):
        self.store = store
        self._flush_seconds = flush_seconds
        self._max_buffer = max_buffer
        self._records: list[UsageRecord] = []
        self._events: list[SystemEvent] = []
        self._lost = 0
        self._task: Optional[asyncio.Task] = None

    def record(self, record: UsageRecord) -> None:
        if len(self._records) >= self._max_buffer:
            self._lost += 1
            return
        self._records.append(record)

    def event(self, event: SystemEvent) -> None:
        if len(self._events) < self._max_buffer:
            self._events.append(event)

    def start(self) -> None:
        add_usage_sink(self.record)
        add_event_sink(self.event)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        remove_usage_sink(self.record)
        remove_event_sink(self.event)
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await self.flush()

    async def _write_event(self, event: SystemEvent) -> None:
        if event.kind == "workspace_deleted" and event.workspace:
            label = deleted_workspace_label(event.workspace, event.ts)
            await self.store.anonymize_workspace(event.workspace, label)
            event = SystemEvent(kind=event.kind, workspace=label, ts=event.ts)
        await self.store.write([], [event])

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._flush_seconds)
            await self.flush()

    async def flush(self) -> None:
        while self._records or self._events or self._lost:
            records = self._records[:BATCH_SIZE]
            events = self._events[:BATCH_SIZE]
            try:
                # Records first, so buffered usage is anonymized with the rest.
                await self.store.write(records, [])
                del self._records[: len(records)]
                for event in events:
                    await self._write_event(event)
                    self._events.remove(event)
                if self._lost:
                    total = await self.store.kv_get(LOST_ROWS_KEY, 0)
                    await self.store.kv_set(LOST_ROWS_KEY, (total or 0) + self._lost)
                    self._lost = 0
            except Exception as e:
                logger.warning(f"Usage write failed, will retry: {e}")
                return
