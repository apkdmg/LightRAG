"""Estimate past usage from the prompts saved in the LLM cache.

This is the only record of usage from before recording began. It is an estimate:
it can't cover images, name the model or say which key paid. Rows are marked as
estimated and stay out of anomaly detection.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Optional

from lightrag.usage import (
    EXTRACT_PURPOSE,
    KEYWORD_PURPOSE,
    QUERY_PURPOSE,
    UsageRecord,
)
from lightrag.utils import logger

from .service import BACKFILL_KEY
from .store import BACKFILL_SOURCE, UNKNOWN_MODEL, UsageStore

_BATCH = 500
_CACHE_TYPE = {
    "extract": (EXTRACT_PURPOSE, "extraction"),
    "summary": (EXTRACT_PURPOSE, "summary"),
    "analysis": (EXTRACT_PURPOSE, "table_analysis"),
    "keywords": (KEYWORD_PURPOSE, "mix"),
    "query": (QUERY_PURPOSE, "mix"),
}


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value or time.time())


def _estimate(
    rows: list[dict[str, Any]], tokenizer: Any, before: float
) -> list[UsageRecord]:
    out = []
    for row in rows:
        mapped = _CACHE_TYPE.get(row.get("cache_type") or "")
        prompt = row.get("original_prompt") or ""
        ts = _timestamp(row.get("create_time"))
        # Image prompts are text only, so a word count would badly undercount.
        if mapped is None or "<vlm_images>" in prompt or ts >= before:
            continue
        out.append(
            UsageRecord(
                workspace=row["workspace"],
                purpose=mapped[0],
                operation=mapped[1],
                source=BACKFILL_SOURCE,
                model=UNKNOWN_MODEL,
                prompt_tokens=len(tokenizer.encode(prompt)),
                completion_tokens=len(tokenizer.encode(row.get("return_value") or "")),
                estimated=True,
                ts=ts,
            )
        )
    return out


async def run_backfill(store: UsageStore, tokenizer: Any) -> dict[str, Any]:
    """Run once. Only entries older than the start of recording are counted."""
    done = await store.kv_get(BACKFILL_KEY)
    if done and done.get("status") in ("running", "done"):
        return done
    first = await store.first_recorded_day()
    before = (
        datetime(first.year, first.month, first.day, tzinfo=store.tz).timestamp()
        if first
        else time.time()
    )
    state: dict[str, Any] = {"status": "running", "started_at": time.time(), "rows": 0}
    await store.kv_set(BACKFILL_KEY, state)
    after: Optional[tuple[str, str]] = None
    try:
        while True:
            rows = await store.llm_cache_batch(after, _BATCH)
            if not rows:
                break
            after = (rows[-1]["workspace"], rows[-1]["id"])
            # Counting tokens on long prompts would stall the event loop.
            records = await asyncio.to_thread(_estimate, rows, tokenizer, before)
            if records:
                await store.write(records, [])
            state["rows"] += len(records)
            await store.kv_set(BACKFILL_KEY, state)
        state.update(status="done", finished_at=time.time())
    except Exception as e:
        logger.error(f"Usage backfill failed: {e}")
        state.update(status="failed")
    await store.kv_set(BACKFILL_KEY, state)
    return state
