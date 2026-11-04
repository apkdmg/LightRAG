"""Buffers usage records and writes them in batches.

A write failure never reaches the LLM call that produced the record. Records are
kept and retried; once the buffer is full, new ones are counted as lost so the
page can show what share of usage is missing. System events dropped for the
same reason are counted too.

The read-only accessors (:meth:`UsageRecorder.health` and the properties) let
the Operations console report the recorder's state through each worker's
heartbeat.

Each record is completed as it arrives, in the caller's context: a failed
call gets its ``error_kind`` from the Operations console's failure taxonomy
(:func:`lightrag.api.ops.classify.classify_failure`), and every call the API
request ID it served (the request-ID ContextVar), so ``/correlate`` can find
the LLM calls of a request. The exception itself is dropped before the
record is buffered.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
from typing import Any, Optional

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


def complete_record(record: UsageRecord) -> UsageRecord:
    """``record`` with ``error_kind`` and ``request_id`` filled in and the
    exception dropped. Never raises: on any problem the record is kept as it
    came, minus the exception."""
    try:
        from lightrag.api.ops.classify import classify_failure
        from lightrag.api.ops.context import current_request_id

        error_kind = record.error_kind
        if record.status == "error" and error_kind is None:
            error_kind = classify_failure(record.error).error_kind
        request_id = record.request_id or current_request_id()
        if (
            record.error is None
            and error_kind == record.error_kind
            and request_id == record.request_id
        ):
            return record
        return dataclasses.replace(
            record,
            error=None,
            error_kind=(error_kind or None) and str(error_kind)[:32],
            request_id=(request_id or None) and str(request_id)[:64],
        )
    except Exception as e:
        logger.warning(f"Could not classify a usage record: {type(e).__name__}")
        try:
            return dataclasses.replace(record, error=None)
        except Exception:
            return record


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
        # Cumulative counters for health reporting; never reset.
        self._lost_total = 0
        self._dropped_events = 0
        self._flush_failures = 0
        self._last_flush_ok_at: Optional[float] = None
        self._task: Optional[asyncio.Task] = None

    def record(self, record: UsageRecord) -> None:
        if len(self._records) >= self._max_buffer:
            self._lost += 1
            self._lost_total += 1
            return
        self._records.append(complete_record(record))

    def event(self, event: SystemEvent) -> None:
        if len(self._events) < self._max_buffer:
            self._events.append(event)
        else:
            self._dropped_events += 1

    # -- health (read-only) --------------------------------------------------
    @property
    def buffer_len(self) -> int:
        """Records and events waiting to be written."""
        return len(self._records) + len(self._events)

    @property
    def last_flush_ok_at(self) -> Optional[float]:
        """Epoch seconds of the last flush that left nothing unwritten."""
        return self._last_flush_ok_at

    @property
    def flush_failures(self) -> int:
        """Flushes that failed since start (each is retried)."""
        return self._flush_failures

    @property
    def dropped_events(self) -> int:
        """System events dropped since start because the buffer was full."""
        return self._dropped_events

    @property
    def lost_rows(self) -> int:
        """Usage records lost since start because the buffer was full."""
        return self._lost_total

    def health(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer_len,
            "last_flush_ok_at": self._last_flush_ok_at,
            "flush_failures": self._flush_failures,
            "lost_rows": self._lost_total,
            "dropped_events": self._dropped_events,
            "running": self._task is not None and not self._task.done(),
        }

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
                self._flush_failures += 1
                logger.warning(f"Usage write failed, will retry: {e}")
                return
        self._last_flush_ok_at = time.time()
