"""Background builder for the indexes the Operations console reads through.

The ops queries need a few indexes on large existing tables (doc status,
usage calls and events). Building them at startup would delay serving (as
``check_tables`` does with the pagination indexes, which this module never
touches), so they are built once, in the background, after the server is up:

- on a dedicated connection holding a session advisory lock, so one process
  builds while the others skip;
- with ``CREATE INDEX CONCURRENTLY`` under a ``lock_timeout``, so writers are
  never blocked behind it;
- an index that exists but is invalid (an interrupted concurrent build) is
  dropped and rebuilt, unless ``pg_stat_progress_create_index`` shows it is
  still being built;
- inside the ``OPS_INDEX_BUILD_WINDOW`` (local time, e.g. ``01:00-05:00``)
  when one is set, otherwise shortly after startup.

Queries work without these indexes, only slower. A build that fails (often a
lock timeout behind a long transaction) leaves an INVALID index that writers
would still maintain, so it is dropped at once; the builder then tries again
later (``OPS_INDEX_BUILD_RETRY_S``, doubling, ``OPS_INDEX_BUILD_ATTEMPTS`` in
all, each attempt inside the window) and finally on the next start.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, tzinfo
from typing import Any, Optional

logger = logging.getLogger("lightrag.api.ops")

LOCK_TIMEOUT_MS = 5000
# A concurrent build on a few million rows takes minutes; cap it anyway.
BUILD_STATEMENT_TIMEOUT_MS = 30 * 60 * 1000
DEFAULT_START_DELAY_S = 120.0
DEFAULT_RETRY_DELAY_S = 30 * 60.0
DEFAULT_ATTEMPTS = 4
ADVISORY_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"lightrag:ops:index-builder").digest()[:8], "big", signed=True
)
_WINDOW = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


@dataclass(frozen=True)
class IndexSpec:
    name: str
    table: str
    definition: str  # everything after "ON"

    def create_sql(self) -> str:
        return (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {self.name} ON {self.definition}"
        )


# Spec §3.5. Status lists are literals so the planner can prove the queries'
# predicates imply the partial-index predicates.
OPS_INDEXES: tuple[IndexSpec, ...] = (
    IndexSpec(
        "idx_lightrag_doc_status_inflight",
        "LIGHTRAG_DOC_STATUS",
        "LIGHTRAG_DOC_STATUS (status, updated_at, workspace) "
        "WHERE status IN ('pending','parsing','analyzing','processing','failed') "
        "AND id NOT LIKE 'dup-%'",
    ),
    IndexSpec(
        "idx_lightrag_doc_status_dup",
        "LIGHTRAG_DOC_STATUS",
        "LIGHTRAG_DOC_STATUS (workspace) WHERE id LIKE 'dup-%'",
    ),
    IndexSpec(
        "idx_lightrag_doc_status_id",
        "LIGHTRAG_DOC_STATUS",
        "LIGHTRAG_DOC_STATUS (id)",
    ),
    IndexSpec(
        "idx_lightrag_usage_calls_doc",
        "LIGHTRAG_USAGE_CALLS",
        "LIGHTRAG_USAGE_CALLS (workspace, doc_id, ts DESC) WHERE doc_id IS NOT NULL",
    ),
    IndexSpec(
        "idx_lightrag_usage_calls_err",
        "LIGHTRAG_USAGE_CALLS",
        "LIGHTRAG_USAGE_CALLS (ts DESC) WHERE status <> 'ok'",
    ),
    IndexSpec(
        "idx_usage_events_kind_ts",
        "LIGHTRAG_USAGE_EVENTS",
        "LIGHTRAG_USAGE_EVENTS (kind, ts DESC)",
    ),
)


def parse_window(value: Optional[str]) -> Optional[tuple[dtime, dtime]]:
    """``"01:00-05:00"`` -> (start, end); empty or malformed -> None."""
    if not value:
        return None
    match = _WINDOW.match(value)
    if not match:
        logger.warning(
            "OPS_INDEX_BUILD_WINDOW %r is not HH:MM-HH:MM; ignoring it", value
        )
        return None
    h1, m1, h2, m2 = (int(g) for g in match.groups())
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        logger.warning("OPS_INDEX_BUILD_WINDOW %r is out of range; ignoring it", value)
        return None
    return dtime(h1, m1), dtime(h2, m2)


def seconds_until_window(window: Optional[tuple[dtime, dtime]], now: datetime) -> float:
    """0 inside the window (which may wrap midnight), else seconds to its start."""
    if window is None:
        return 0.0
    start, end = window
    current = now.timetz().replace(tzinfo=None)
    if start <= end:
        inside = start <= current < end
    else:
        inside = current >= start or current < end
    if inside:
        return 0.0
    target = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


async def _index_state(conn: Any, name: str) -> Optional[dict[str, Any]]:
    row = await conn.fetchrow(
        """SELECT c.oid, i.indisvalid, i.indisready,
               EXISTS (SELECT 1 FROM pg_stat_progress_create_index p
                       WHERE p.index_relid = c.oid) AS building
           FROM pg_class c JOIN pg_index i ON i.indexrelid = c.oid
           WHERE c.relname = $1 AND c.relkind = 'i'""",
        name.lower(),
    )
    return dict(row) if row else None


async def _drop_failed_build(conn: Any, spec: IndexSpec) -> None:
    """Drop what a failed concurrent build left behind (an INVALID index).

    An invalid index that reached ``indisready`` is updated on every write
    yet never used, so it must not wait for the next attempt. One that
    another session is still building is left alone.
    """
    try:
        state = await _index_state(conn, spec.name)
        if state is None or state["indisvalid"] or state["building"]:
            return
        await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {spec.name}")
        logger.info(
            "Dropped the invalid ops index %s left by a failed build", spec.name
        )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(
            "Ops index %s is invalid and could not be dropped (%s); the next "
            "attempt drops it",
            spec.name,
            type(e).__name__,
        )


async def missing_indexes(conn: Any, specs=OPS_INDEXES) -> list[IndexSpec]:
    """Specs whose table exists but whose index is absent or invalid."""
    out = []
    for spec in specs:
        if await conn.fetchval("SELECT to_regclass($1)", spec.table.lower()) is None:
            continue
        state = await _index_state(conn, spec.name)
        if state is None or not state["indisvalid"]:
            out.append(spec)
    return out


async def build_indexes(conn: Any, specs=OPS_INDEXES) -> dict[str, str]:
    """Build what is missing on ``conn`` (a dedicated, non-pooled connection).

    Returns the outcome per index name: ``valid``, ``created``, ``rebuilt``,
    ``building`` (another session is building it), ``skipped`` (no table),
    ``locked`` (another process holds the builder lock) or ``failed``.
    """
    outcome: dict[str, str] = {}
    if not await conn.fetchval("SELECT pg_try_advisory_lock($1)", ADVISORY_LOCK_KEY):
        return {spec.name: "locked" for spec in specs}
    try:
        await conn.execute(f"SET lock_timeout = {LOCK_TIMEOUT_MS}")
        await conn.execute(f"SET statement_timeout = {BUILD_STATEMENT_TIMEOUT_MS}")
        for spec in specs:
            try:
                if (
                    await conn.fetchval("SELECT to_regclass($1)", spec.table.lower())
                    is None
                ):
                    outcome[spec.name] = "skipped"
                    continue
                state = await _index_state(conn, spec.name)
                if state is not None and state["indisvalid"]:
                    outcome[spec.name] = "valid"
                    continue
                rebuilt = False
                if state is not None:
                    if state["building"]:
                        outcome[spec.name] = "building"
                        continue
                    # Left INVALID by an interrupted concurrent build.
                    logger.warning("Ops index %s is invalid; rebuilding it", spec.name)
                    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {spec.name}")
                    rebuilt = True
                logger.info("Building ops index %s in the background", spec.name)
                await conn.execute(spec.create_sql())
                state = await _index_state(conn, spec.name)
                if state is None or not state["indisvalid"]:
                    outcome[spec.name] = "failed"
                    logger.warning("Ops index %s did not become valid", spec.name)
                    await _drop_failed_build(conn, spec)
                else:
                    outcome[spec.name] = "rebuilt" if rebuilt else "created"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                outcome[spec.name] = "failed"
                logger.warning(
                    "Ops index %s could not be built (%s); queries stay slower "
                    "until it is retried",
                    spec.name,
                    type(e).__name__,
                )
                await _drop_failed_build(conn, spec)
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_KEY)
        except Exception:
            pass
    return outcome


def _local_tz() -> tzinfo:
    from lightrag.api.insights.store import insights_timezone

    return insights_timezone()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


async def run_index_builder(
    store: Any,
    *,
    delay_s: Optional[float] = None,
    window: Optional[str] = None,
    retry_delay_s: Optional[float] = None,
    attempts: Optional[int] = None,
) -> Optional[dict[str, str]]:
    """The startup task: wait, then build; retry failures with backoff.

    Returns the last outcome (None when no attempt reached the database).
    Never raises except on cancel.
    """
    if delay_s is None:
        delay_s = _env_float("OPS_INDEX_BUILD_DELAY_S", DEFAULT_START_DELAY_S)
    if retry_delay_s is None:
        retry_delay_s = _env_float("OPS_INDEX_BUILD_RETRY_S", DEFAULT_RETRY_DELAY_S)
    if attempts is None:
        attempts = int(_env_float("OPS_INDEX_BUILD_ATTEMPTS", DEFAULT_ATTEMPTS))
    attempts = max(1, attempts)
    parsed = parse_window(
        window if window is not None else os.getenv("OPS_INDEX_BUILD_WINDOW")
    )
    await asyncio.sleep(max(0.0, delay_s))
    outcome: Optional[dict[str, str]] = None
    for attempt in range(1, attempts + 1):
        outcome, retry = await _build_once(store, parsed)
        if not retry:
            return outcome
        if attempt < attempts:
            wait = max(0.0, retry_delay_s) * (2 ** (attempt - 1))
            logger.info(
                "Ops index build will be retried in %.0f min (attempt %d of %d)",
                wait / 60,
                attempt + 1,
                attempts,
            )
            await asyncio.sleep(wait)
    logger.warning(
        "Ops indexes are still incomplete after %d attempts; the next start "
        "tries again",
        attempts,
    )
    return outcome


async def _build_once(
    store: Any, parsed: Optional[tuple[dtime, dtime]]
) -> tuple[Optional[dict[str, str]], bool]:
    """One attempt: ``(outcome, retry)``. Never raises except on cancel."""
    try:
        async with store.dedicated_connection() as conn:
            if not await missing_indexes(conn):
                return {spec.name: "valid" for spec in OPS_INDEXES}, False
        wait = seconds_until_window(parsed, datetime.now(_local_tz()))
        if wait > 0:
            logger.info(
                "Ops indexes missing; building them in the off-peak window in %.0f min",
                wait / 60,
            )
            await asyncio.sleep(wait)
        async with store.dedicated_connection() as conn:
            outcome = await build_indexes(conn)
        built = sorted(n for n, o in outcome.items() if o in ("created", "rebuilt"))
        if built:
            logger.info("Ops indexes built: %s", ", ".join(built))
        # "locked": another process builds (and retries) them.
        return outcome, "failed" in outcome.values()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Ops index builder did not run (%s)", type(e).__name__)
        return None, True
