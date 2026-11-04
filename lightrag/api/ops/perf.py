"""Read models of the Performance tab and the Sign-in card, and the hourly
LLM p95 baseline.

Sources (§3.6 of ``docs/ADMIN_OPERATIONS_DASHBOARD.md``):

- LLM latency: ``LIGHTRAG_USAGE_CALLS`` through the shared Insights store
  helper (Q5, percentiles, windows up to 24 h); the 7-day window reads the
  hourly usage rollup and shows average and max only (``approx_7d``).
- HTTP, embed, rerank and sign-in: the ops histograms
  (:mod:`lightrag.api.ops.metrics`). Windows up to 1 h read minute rows;
  longer ones read hour rows where the rollup has run and minutes after it.
- Concurrency: the queue sums every worker already publishes in its
  heartbeat (no workspace is loaded, nothing is looped over here).
- Throughput and top workspaces: ``LIGHTRAG_OPS_RUNS``,
  ``LIGHTRAG_OPS_DOC_FAILURES`` and the usage tables, ``ORDER BY ... LIMIT``.

Every read goes through the service's bounded single-flight cache with a TTL
by window (30 s for 15 m / 1 h, 2 min for 24 h, 5 min for 7 d), so the
database load does not grow with the number of open tabs.

The baseline job (lease ``ops:baselines``, hourly) stores the 7-day
same-hour LLM p95 in ``LIGHTRAG_OPS_KV`` under ``llm_p95_baseline``; the LLM
health card and the ``llm_p95_15m`` attention rule compare the last 15
minutes with it.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from lightrag.api.ops import metrics
from lightrag.api.ops.metrics import (
    SERIES_AUTH,
    SERIES_EMBED,
    SERIES_HTTP,
    SERIES_RERANK,
    UNMATCHED_KEY,
    estimate_percentiles,
    merge_buckets,
)

logger = logging.getLogger("lightrag.api.ops")

WINDOWS: dict[str, int] = {"15m": 900, "1h": 3600, "24h": 86400, "7d": 7 * 86400}
# Chart resolution per window (hour multiples where hour rows are read).
BUCKET_S: dict[str, int] = {"15m": 60, "1h": 300, "24h": 3600, "7d": 10800}
CACHE_TTL_S: dict[str, float] = {"15m": 30.0, "1h": 30.0, "24h": 120.0, "7d": 300.0}
# Windows read from minute rows only; longer ones use the hourly rollup.
MINUTE_WINDOWS = ("15m", "1h")
# LLM percentiles need individual calls: at most 24 h (FR-40).
LLM_PERCENTILE_WINDOWS = ("15m", "1h", "24h")
LLM_GROUPS = ("purpose", "model")
TOP_METRICS = ("llm_calls", "processing_minutes", "failures", "runs")
TOP_LIMIT = 20
TOP_INSTANCES = 10
HTTP_TOP_P95 = 10
THROUGHPUT_MAX_DAYS = 30
THROUGHPUT_CALLS_HOURS = 24
THROUGHPUT_CALLS_BUCKET_S = 300
CONCURRENCY_TTL_S = 5.0
THROUGHPUT_TTL_S = 120.0
TOP_TTL_S = 60.0
BASELINE_TTL_S = 300.0
RULES_TTL_S = 30.0

BASELINE_KEY = "llm_p95_baseline"
BASELINE_LEASE = "ops:baselines"
BASELINE_INTERVAL_S = 3600.0
BASELINE_CYCLE_TIMEOUT_S = 120.0
BASELINE_LEASE_TTL_S = 300.0
BASELINE_DAYS = 7
# A baseline needs this many calls in its 7 windows to be compared with.
BASELINE_MIN_CALLS = 20
# A stored hour older than this is not used (the job stopped running).
BASELINE_MAX_AGE_S = 8 * 86400.0

# §1.7 defaults of the rules that read these metrics. They are part of the
# service's DEFAULT_THRESHOLDS, so the ``thresholds`` KV overrides them like
# every other rule's; the windows are in the rule names and stay fixed.
METRIC_THRESHOLDS: dict[str, Any] = {
    "llm_p95_ratio_warn": 2.0,
    "llm_p95_ratio_serious": 4.0,
    "llm_p95_serious_ms": 60_000,
    "http_5xx_warn": 0.01,
    "http_5xx_serious": 0.05,
    "http_5xx_min_requests": 100,
    "dep_error_warn": 0.02,
    "dep_error_serious": 0.10,
    "dep_error_min_calls": 50,
    "signin_sso_warn": 5,
    "signin_lockout_warn": 3,
    "signin_serious": 20,
}
HTTP_5XX_WINDOW_S = 300
DEP_ERROR_WINDOW_S = 900
SIGNIN_WINDOW_S = 900

# Each worker writes a minute's rows at the next minute boundary plus a 2-6 s
# jitter: a window ends at the last minute every worker has written, so a
# rate divides counts by the time they actually cover.
MINUTE_SETTLE_S = metrics.FLUSH_JITTER_S[1] + 4.0


class PerfBadRequest(ValueError):
    """An invalid parameter; ``code`` is the client-facing error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def check_window(window: str) -> str:
    if window not in WINDOWS:
        raise PerfBadRequest("invalid_window")
    return window


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc)


def _hour_floor(epoch: float) -> float:
    return float(int(epoch) // 3600 * 3600)


def _ms(epoch_s: int) -> int:
    return int(epoch_s) * 1000


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _rate(part: float, whole: float) -> Optional[float]:
    return round(part / whole, 4) if whole else None


def complete_end(now: float) -> float:
    """End (epoch) of the last minute whose rows every worker has flushed."""
    return float(int(now - MINUTE_SETTLE_S) // 60 * 60)


def covered_s(t: float, bucket: float, start: float, end: float) -> float:
    """Seconds of bucket ``[t, t + bucket)`` inside ``[start, end)``: the
    first and last buckets of a window may be partial."""
    seconds = min(t + bucket, end) - max(t, start)
    return seconds if seconds > 0 else float(bucket)


def merge_thresholds(defaults: Mapping[str, Any], stored: Any) -> dict[str, Any]:
    """``defaults`` with the positive numbers of the ``thresholds`` KV value
    on top (each cast to its default's type); anything else is ignored."""
    merged = dict(defaults)
    if isinstance(stored, dict):
        for key, default in defaults.items():
            value = stored.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            ):
                merged[key] = type(default)(value)
    return merged


def _threshold(thresholds: Optional[Mapping[str, Any]], key: str) -> Any:
    value = (thresholds or {}).get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return METRIC_THRESHOLDS[key]


def summarize(agg: dict[str, Any]) -> dict[str, Any]:
    """Count, errors, rates and percentiles of one summed histogram."""
    count = int(agg.get("count") or 0)
    errors = int(agg.get("errors") or 0)
    max_ms = agg.get("max_ms")
    p50, p95, p99 = estimate_percentiles(
        agg.get("buckets") or [], (0.5, 0.95, 0.99), max_ms
    )
    return {
        "count": count,
        "errors": errors,
        "error_rate": _rate(errors, count),
        "avg_ms": _round(int(agg.get("sum_ms") or 0) / count) if count else None,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "max_ms": int(max_ms) if count and max_ms is not None else None,
    }


def _empty_agg() -> dict[str, Any]:
    return {"count": 0, "errors": 0, "sum_ms": 0, "max_ms": 0, "buckets": [0] * 20}


def _add(agg: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    agg["count"] += int(row.get("count") or 0)
    agg["errors"] += int(row.get("errors") or 0)
    agg["sum_ms"] += int(row.get("sum_ms") or 0)
    agg["max_ms"] = max(int(agg["max_ms"] or 0), int(row.get("max_ms") or 0))
    merge_buckets(agg["buckets"], row.get("buckets") or [])
    return agg


def group_rows(rows: list[dict[str, Any]], *fields: str) -> dict[tuple, dict[str, Any]]:
    """Sum histogram rows by ``fields`` (e.g. ``"t"``, ``"key"``)."""
    out: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(f) for f in fields)
        agg = out.get(key)
        if agg is None:
            agg = out[key] = _empty_agg()
        _add(agg, row)
    return out


def split_route(key: str) -> tuple[Optional[str], str]:
    """``"GET /documents"`` -> ``("GET", "/documents")``."""
    if key == UNMATCHED_KEY or " " not in key:
        return None, key
    method, path = key.split(" ", 1)
    return method, path


def baseline_for(value: Any, now: float) -> Optional[dict[str, Any]]:
    """The stored baseline for the current UTC hour, when usable."""
    if not isinstance(value, dict):
        return None
    hours = value.get("hours")
    if not isinstance(hours, dict):
        return None
    entry = hours.get(str(datetime.fromtimestamp(now, timezone.utc).hour))
    if not isinstance(entry, dict):
        return None
    p95 = entry.get("p95_ms")
    calls = entry.get("calls")
    at = entry.get("computed_at")
    if (
        not isinstance(p95, (int, float))
        or p95 <= 0
        or not isinstance(calls, int)
        or calls < BASELINE_MIN_CALLS
        or not isinstance(at, (int, float))
        or now - at > BASELINE_MAX_AGE_S
    ):
        return None
    return {"p95_ms": float(p95), "calls": calls, "computed_at": at}


def llm_p95_severity(
    p95_ms: Optional[float],
    baseline_ms: Optional[float],
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """``llm_p95_15m``: > 2x the baseline warns; > 4x or > 60 s is serious
    (the §1.7 defaults; ``thresholds`` carries the KV overrides)."""
    if not isinstance(p95_ms, (int, float)):
        return None
    ratio = (
        p95_ms / baseline_ms
        if isinstance(baseline_ms, (int, float)) and baseline_ms > 0
        else None
    )
    if p95_ms > _threshold(thresholds, "llm_p95_serious_ms") or (
        ratio is not None and ratio > _threshold(thresholds, "llm_p95_ratio_serious")
    ):
        return "serious"
    if ratio is not None and ratio > _threshold(thresholds, "llm_p95_ratio_warn"):
        return "warning"
    return None


def signin_severity(
    signin: Optional[Mapping[str, Any]],
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """``signin_failures_15m``: SSO callback errors > 5 or lockouts > 3 warn,
    either > 20 is serious. 401s alone are normal (expired tokens)."""
    if not signin:
        return None
    sso = int(signin.get("sso_errors") or 0)
    lockouts = int(signin.get("lockouts") or 0)
    if sso <= _threshold(thresholds, "signin_sso_warn") and lockouts <= _threshold(
        thresholds, "signin_lockout_warn"
    ):
        return None
    if max(sso, lockouts) > _threshold(thresholds, "signin_serious"):
        return "serious"
    return "warning"


# -- baseline job ------------------------------------------------------------------


async def baseline_cycle(
    loop: Any,
    *,
    store: Any,
    usage_store: Callable[[], Any],
    clock: Callable[[], float] = time.time,
) -> Optional[dict[str, Any]]:
    """Compute the current UTC hour's 7-day same-hour LLM p95 (one query)."""
    usage = usage_store()
    if usage is None:
        return None
    now = clock()
    hour = _hour_floor(now)
    starts = [_utc(hour - d * 86400) for d in range(1, BASELINE_DAYS + 1)]
    result = await usage.latency_baseline(starts)
    stored = await store.kv_get(BASELINE_KEY, None)
    hours = dict(stored.get("hours") or {}) if isinstance(stored, dict) else {}
    entry = {
        "p95_ms": _round(result.get("p95_ms")),
        "calls": int(result.get("calls") or 0),
        "days": BASELINE_DAYS,
        "computed_at": now,
    }
    hours[str(_utc(hour).hour)] = entry
    if await loop.still_held():
        await store.kv_put(BASELINE_KEY, {"hours": hours, "updated_at": now})
    return entry


# -- views ---------------------------------------------------------------------------


class PerformanceViews:
    """The Performance and Sign-in read models over one ``OpsService``."""

    def __init__(self, service: Any):
        self.service = service

    # -- plumbing ----------------------------------------------------------------
    @property
    def _store(self) -> Any:
        return self.service.store

    def _usage(self) -> Any:
        insights = getattr(self.service.app_state, "insights", None)
        return getattr(insights, "store", None)

    def _now(self) -> float:
        return float(self.service.clock())

    async def _cached(self, key: tuple, ttl: float, compute) -> Any:
        return await self.service.cache.get(
            ("perf", *key), ttl, lambda: self.service._q(compute)
        )

    async def _metric_rows(
        self, series: list[str], window: str, now: float
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Histogram rows of ``series`` for ``window`` and how they were read.

        The range ends at the last minute every worker has flushed (the
        current minute is never written yet). Minute windows cover exactly
        their span; longer ones start on an hour (hour rows cannot be split),
        so ``span_s``, the time the rows cover, is what rates divide by.
        """
        bucket = BUCKET_S[window]
        end = complete_end(now)
        if window in MINUTE_WINDOWS:
            start = end - WINDOWS[window]
            cut = start
        else:
            start = _hour_floor(end - WINDOWS[window])
            through = await metrics.rollup_through(self._store)
            cut = min(max(through or start, start), end)
        rows = await self._store.metrics_read(
            series, _utc(start), _utc(cut), _utc(end), bucket
        )
        source = {
            "bucket_s": bucket,
            "from": _utc(start).isoformat(),
            "to": _utc(end).isoformat(),
            "span_s": int(end - start),
            "hourly_until": _utc(cut).isoformat() if cut > start else None,
        }
        return rows, source

    @staticmethod
    def _bucket_s(source: dict[str, Any], t: float) -> float:
        start = datetime.fromisoformat(source["from"]).timestamp()
        end = datetime.fromisoformat(source["to"]).timestamp()
        return covered_s(t, source["bucket_s"], start, end)

    # -- HTTP ----------------------------------------------------------------------
    async def http(self, window: str) -> dict[str, Any]:
        check_window(window)
        now = self._now()

        async def load() -> dict[str, Any]:
            rows, source = await self._metric_rows([SERIES_HTTP], window, now)
            span = source["span_s"]
            routes = []
            for (key,), agg in group_rows(rows, "key").items():
                summary = summarize(agg)
                method, path = split_route(key)
                routes.append(
                    {
                        "route": key,
                        "method": method,
                        "path": path,
                        "rps": round(summary["count"] / span, 4),
                        "err5xx": summary["error_rate"],
                        **summary,
                    }
                )
            routes.sort(key=lambda r: (-r["count"], r["route"]))
            series = []
            for (t,), agg in sorted(group_rows(rows, "t").items()):
                summary = summarize(agg)
                series.append(
                    {
                        "t": _ms(t),
                        "rps": round(summary["count"] / self._bucket_s(source, t), 4),
                        "err5xx": summary["error_rate"],
                        **summary,
                    }
                )
            total = summarize(_add_all(rows))
            top = sorted(
                (r for r in routes if r["p95"] is not None),
                key=lambda r: (-(r["p95"] or 0), r["route"]),
            )[:HTTP_TOP_P95]
            return {
                "window": window,
                **source,
                "routes": routes,
                "top_p95": [r["route"] for r in top],
                "series": series,
                "totals": {
                    **total,
                    "rps": round(total["count"] / span, 4),
                    "err5xx": total["error_rate"],
                },
            }

        body = await self._cached(("http", window), CACHE_TTL_S[window], load)
        return {**body, "enabled": metrics.metrics_enabled()}

    # -- embed and rerank -------------------------------------------------------------
    async def deps(self, window: str) -> dict[str, Any]:
        check_window(window)
        now = self._now()

        async def load() -> dict[str, Any]:
            rows, source = await self._metric_rows(
                [SERIES_EMBED, SERIES_RERANK], window, now
            )
            span = source["span_s"]
            out: dict[str, Any] = {
                "window": window,
                **source,
                "series": {SERIES_EMBED: [], SERIES_RERANK: []},
                "models": {SERIES_EMBED: [], SERIES_RERANK: []},
                "totals": {},
            }
            points: dict[tuple[str, int], dict[str, Any]] = {}
            for (series, t), agg in group_rows(rows, "series", "t").items():
                summary = summarize(agg)
                points[(series, int(t))] = {
                    "t": _ms(t),
                    "per_min": round(
                        summary["count"] * 60.0 / self._bucket_s(source, t), 2
                    ),
                    **summary,
                }
            # vLLM running/waiting/KV cache from the probe rows of the same
            # range, so a saturated vLLM can be told from slow calls here.
            for row in await self._vllm_rows(source):
                key = (row["dependency"], int(row["t"]))
                point = points.get(key)
                if point is None:
                    point = points[key] = {
                        "t": _ms(row["t"]),
                        "per_min": 0.0,
                        **summarize(_empty_agg()),
                    }
                kv_pct = row.get("kv_cache_pct")
                point.update(
                    {
                        "running": _round(row.get("running")),
                        "waiting": _round(row.get("waiting")),
                        # A 0..1 fraction (the probe reports percent).
                        "kv_cache": None if kv_pct is None else round(kv_pct / 100, 4),
                    }
                )
            for (series, _), point in sorted(points.items()):
                out["series"].setdefault(series, []).append(point)
            for (series, key), agg in sorted(group_rows(rows, "series", "key").items()):
                out["models"].setdefault(series, []).append(
                    {"model": key, **summarize(agg)}
                )
            for name in (SERIES_EMBED, SERIES_RERANK):
                agg = _add_all([r for r in rows if r["series"] == name])
                summary = summarize(agg)
                out["totals"][name] = {
                    **summary,
                    "per_min": round(summary["count"] * 60.0 / span, 2),
                }
            return out

        body = await self._cached(("deps", window), CACHE_TTL_S[window], load)
        return {
            **body,
            # The same window totals under the name the charts read.
            "summary": body["totals"],
            "vllm": await self._vllm(),
            "enabled": metrics.metrics_enabled(),
        }

    async def _vllm_rows(self, source: dict[str, Any]) -> list[dict[str, Any]]:
        """Probe rows' vLLM metrics per bucket over the window (none on error:
        the latency view still stands)."""
        try:
            return await self._store.vllm_series(
                datetime.fromisoformat(source["from"]),
                datetime.fromisoformat(source["to"]),
                source["bucket_s"],
            )
        except Exception as e:
            logger.debug("Ops vLLM history unreadable (%s)", type(e).__name__)
            return []

    async def _vllm(self) -> dict[str, Any]:
        """vLLM running/waiting/KV cache from the latest probe results."""
        out: dict[str, Any] = {SERIES_EMBED: None, SERIES_RERANK: None}
        try:
            records, _, _ = await self.service._probe_records()
        except Exception:
            return out
        for dep in (SERIES_EMBED, SERIES_RERANK):
            record = records.get(dep)
            if not isinstance(record, dict):
                continue
            detail = record.get("detail") or {}
            out[dep] = {
                "status": record.get("status"),
                "checked_at": record.get("checked_at"),
                "running": detail.get("running"),
                "waiting": detail.get("waiting"),
                "kv_cache_pct": detail.get("kv_cache_pct"),
                "metrics": detail.get("metrics"),
            }
        return out

    # -- LLM -------------------------------------------------------------------------
    async def llm(self, window: str, group: str) -> dict[str, Any]:
        check_window(window)
        if group not in LLM_GROUPS:
            raise PerfBadRequest("invalid_group")
        concurrency = await self.concurrency()
        llm_queue = concurrency["roles"].get("llm") or {}
        per_role = concurrency.get("llm_roles") or {}
        base = {
            "window": window,
            "group": group,
            "approx_7d": window not in LLM_PERCENTILE_WINDOWS,
            "queue": {
                "running": llm_queue.get("running"),
                "in_flight": llm_queue.get("in_flight"),
                "queued": llm_queue.get("queued"),
            },
        }
        now = self._now()
        base["queue_series"] = await self._llm_queue_series(window, now)
        usage = self._usage()
        if usage is None:
            return {**base, "insights": False, "series": [], "table": []}
        if window in LLM_PERCENTILE_WINDOWS:
            body = await self._cached(
                ("llm", window, group),
                CACHE_TTL_S[window],
                lambda: self._llm_percentiles(usage, window, group, now),
            )
        else:
            body = await self._cached(
                ("llm", window, group),
                CACHE_TTL_S[window],
                lambda: self._llm_hourly(usage, window, group, now),
            )
        table = []
        for row in body["table"]:
            queued = (per_role.get(row["key"]) or {}) if group == "purpose" else {}
            table.append(
                {
                    **row,
                    "queued": queued.get("queued"),
                    "running": queued.get("running"),
                }
            )
        return {**base, "insights": True, **body, "table": table}

    async def _llm_queue_series(self, window: str, now: float) -> list[dict[str, Any]]:
        """Summed LLM running/in-flight/queued per bucket from the workers'
        minute samples (the second axis of the LLM charts). Empty without the
        ops store or when it cannot answer: the latency charts still stand."""
        if not self.service.available:
            return []
        bucket = BUCKET_S[window]
        if window in LLM_PERCENTILE_WINDOWS:
            start, end = now - WINDOWS[window], now
        else:
            start, end = _hour_floor(now - WINDOWS[window]), _hour_floor(now) + 3600

        async def load() -> list[dict[str, Any]]:
            rows = await self._store.llm_queue_series(_utc(start), _utc(end), bucket)
            return [
                {
                    "t": _ms(r["t"]),
                    "running": _opt_int(r.get("running")),
                    "in_flight": _opt_int(r.get("in_flight")),
                    "queued": _opt_int(r.get("queued")),
                }
                for r in rows
            ]

        try:
            return await self._cached(("llm_queue", window), CACHE_TTL_S[window], load)
        except Exception as e:
            logger.debug("Ops LLM queue history unreadable (%s)", type(e).__name__)
            return []

    async def _llm_percentiles(
        self, usage: Any, window: str, group: str, now: float
    ) -> dict[str, Any]:
        end = _utc(now)
        start = _utc(now - WINDOWS[window])
        bucket = BUCKET_S[window]
        series_rows = await usage.latency_percentiles(
            start, end, group=group, bucket_s=bucket
        )
        table_rows = await usage.latency_percentiles(start, end, group=group)
        span_min = WINDOWS[window] / 60.0
        return {
            "bucket_s": bucket,
            "series": [
                {
                    "t": _ms(r["t"]),
                    "key": r["key"],
                    "calls": r["calls"],
                    "errors": r["errors"],
                    "p50": _round(r["p50_ms"]),
                    "p95": _round(r["p95_ms"]),
                    "p99": _round(r["p99_ms"]),
                }
                for r in series_rows
            ],
            "table": [
                {
                    "key": r["key"],
                    "calls": r["calls"],
                    "calls_per_min": round(r["calls"] / span_min, 2),
                    "errors": r["errors"],
                    "error_rate": _rate(r["errors"], r["calls"]),
                    "truncated_rate": _rate(r["truncated"], r["calls"]),
                    "abandoned_rate": _rate(r["abandoned"], r["calls"]),
                    "p50": _round(r["p50_ms"]),
                    "p95": _round(r["p95_ms"]),
                    "p99": _round(r["p99_ms"]),
                }
                for r in sorted(table_rows, key=lambda r: (-r["calls"], r["key"]))
            ],
        }

    async def _llm_hourly(
        self, usage: Any, window: str, group: str, now: float
    ) -> dict[str, Any]:
        """7 d: hourly rollups give average and max only (labelled approx)."""
        end = _utc(_hour_floor(now) + 3600)
        start = _utc(_hour_floor(now - WINDOWS[window]))
        bucket = BUCKET_S[window]
        rows = await usage.rollup("hour", start, end, group_by=("bucket", group))
        series: dict[tuple[int, str], dict[str, float]] = {}
        totals: dict[str, dict[str, float]] = {}
        for row in rows:
            moment = row["bucket"]
            t = int(moment.timestamp()) // bucket * bucket
            key = row.get(group) or ""
            for target in (
                series.setdefault((t, key), _llm_acc()),
                totals.setdefault(key, _llm_acc()),
            ):
                target["calls"] += float(row.get("calls") or 0)
                target["errors"] += float(row.get("errors") or 0)
                target["truncated"] += float(row.get("truncated") or 0)
                target["abandoned"] += float(row.get("abandoned") or 0)
                target["latency_ms_sum"] += float(row.get("latency_ms_sum") or 0)
                target["max"] = max(
                    target["max"], float(row.get("latency_ms_max") or 0)
                )
        span_min = WINDOWS[window] / 60.0
        return {
            "bucket_s": bucket,
            "series": [
                {
                    "t": _ms(t),
                    "key": key,
                    "calls": int(acc["calls"]),
                    "errors": int(acc["errors"]),
                    "avg": _avg(acc),
                    "max": _round(acc["max"]) if acc["calls"] else None,
                }
                for (t, key), acc in sorted(series.items())
                if acc["calls"]
            ],
            "table": [
                {
                    "key": key,
                    "calls": int(acc["calls"]),
                    "calls_per_min": round(acc["calls"] / span_min, 2),
                    "errors": int(acc["errors"]),
                    "error_rate": _rate(acc["errors"], acc["calls"]),
                    "truncated_rate": _rate(acc["truncated"], acc["calls"]),
                    "abandoned_rate": _rate(acc["abandoned"], acc["calls"]),
                    "avg": _avg(acc),
                    "max": _round(acc["max"]) if acc["calls"] else None,
                }
                for key, acc in sorted(totals.items(), key=lambda kv: -kv[1]["calls"])
                if acc["calls"]
            ],
        }

    # -- concurrency -------------------------------------------------------------------
    async def concurrency(self) -> dict[str, Any]:
        """Queue sums over every live worker's heartbeat snapshot (≤15 s old)."""
        try:
            items = await self.service._worker_items()
        except Exception:
            items = None
        alive = [i for i in items or [] if i.get("alive")]
        roles: dict[str, Optional[dict[str, int]]] = {
            "llm": None,
            SERIES_EMBED: None,
            SERIES_RERANK: None,
        }
        llm_roles: dict[str, dict[str, int]] = {}
        top: list[dict[str, Any]] = []
        instances = 0
        for item in alive:
            snapshot = item.get("snapshot") or {}
            instances += int(snapshot.get("instances") or 0)
            for role, counts in (snapshot.get("queues") or {}).items():
                if role not in roles or not isinstance(counts, dict):
                    continue
                target = roles[role] or {"running": 0, "in_flight": 0, "queued": 0}
                for name in target:
                    target[name] += int(counts.get(name) or 0)
                roles[role] = target
            for role, counts in (snapshot.get("llm_roles") or {}).items():
                if not isinstance(counts, dict):
                    continue
                target = llm_roles.setdefault(
                    str(role), {"running": 0, "in_flight": 0, "queued": 0}
                )
                for name in target:
                    target[name] += int(counts.get(name) or 0)
            for entry in snapshot.get("top_instances") or []:
                if isinstance(entry, dict):
                    top.append(
                        {
                            "workspace": entry.get("workspace") or "",
                            "running": int(entry.get("running") or 0),
                            "queued": int(entry.get("queued") or 0),
                            "pid": item.get("pid"),
                            "worker_id": item.get("worker_id"),
                        }
                    )
        top.sort(key=lambda t: (-t["queued"], -t["running"], t["workspace"]))
        out: dict[str, Any] = {
            "roles": roles,
            "llm_roles": llm_roles,
            "top": top[:TOP_INSTANCES],
            "instances": instances,
            "workers": len(alive),
            "heartbeats": items is not None,
        }
        budget = provider_budget()
        if budget:
            used = (roles["llm"] or {}).get("in_flight")
            out["budget"] = {
                "limit": budget,
                "in_flight": used,
                "ratio": round(used / budget, 3) if isinstance(used, int) else None,
            }
        return out

    # -- throughput -------------------------------------------------------------------
    async def throughput(self, days: int) -> dict[str, Any]:
        if not 1 <= days <= THROUGHPUT_MAX_DAYS:
            raise PerfBadRequest("invalid_days")
        now = self._now()

        async def load() -> dict[str, Any]:
            end = _utc(_hour_floor(now) + 3600)
            start = _utc(_hour_floor(now) + 3600 - days * 86400)
            data = await self._store.throughput(start, end, 3600)
            out = {
                "days": days,
                "bucket_s": 3600,
                "docs": [{**r, "t": _ms(r["t"])} for r in data["docs"]],
                "email_batches": [
                    {**r, "t": _ms(r["t"])} for r in data["email_batches"]
                ],
                "llm_calls_per_min": [],
                "llm_calls_bucket_s": THROUGHPUT_CALLS_BUCKET_S,
            }
            usage = self._usage()
            if usage is not None:
                hours = min(days * 24, THROUGHPUT_CALLS_HOURS)
                rows = await usage.call_counts(
                    _utc(now - hours * 3600), _utc(now), THROUGHPUT_CALLS_BUCKET_S
                )
                per = THROUGHPUT_CALLS_BUCKET_S / 60.0
                out["llm_calls_per_min"] = [
                    {
                        "t": _ms(r["t"]),
                        "value": round(r["calls"] / per, 2),
                        "calls_per_min": round(r["calls"] / per, 2),
                        "calls": r["calls"],
                        "errors": r["errors"],
                    }
                    for r in rows
                ]
            out["insights"] = usage is not None
            return out

        return await self._cached(("throughput", days), THROUGHPUT_TTL_S, load)

    # -- top workspaces ---------------------------------------------------------------
    async def top_workspaces(self, metric: str, window: str) -> dict[str, Any]:
        check_window(window)
        if metric not in TOP_METRICS:
            raise PerfBadRequest("invalid_metric")
        now = self._now()

        async def load() -> dict[str, Any]:
            end = _utc(now)
            start = _utc(now - WINDOWS[window])
            if metric == "llm_calls":
                usage = self._usage()
                if usage is None:
                    return {"items": [], "insights": False}
                hourly = window not in LLM_PERCENTILE_WINDOWS
                if hourly:
                    start = _utc(_hour_floor(now - WINDOWS[window]))
                    end = _utc(_hour_floor(now) + 3600)
                items = await usage.top_workspace_calls(
                    start, end, limit=TOP_LIMIT, hourly=hourly
                )
                return {"items": items, "insights": True}
            items = await self._store.top_workspaces(metric, start, end, TOP_LIMIT)
            return {"items": items}

        body = await self._cached(
            ("top", metric, window), max(TOP_TTL_S, CACHE_TTL_S[window]), load
        )
        return {"metric": metric, "window": window, **body}

    # -- sign-in ------------------------------------------------------------------------
    async def signin(self, window: str) -> dict[str, Any]:
        check_window(window)
        now = self._now()

        async def load() -> dict[str, Any]:
            rows, source = await self._metric_rows([SERIES_AUTH], window, now)
            recent_rows, _ = (
                (rows, source)
                if window == "15m"
                else await self._metric_rows([SERIES_AUTH], "15m", now)
            )
            series: dict[int, dict[str, int]] = {}
            for row in rows:
                slot = series.setdefault(int(row["t"]), {})
                slot[row["key"]] = slot.get(row["key"], 0) + int(row["count"])
            return {
                "window": window,
                **source,
                **_signin_totals(rows),
                "last_15m": _signin_totals(recent_rows),
                "series": [
                    {"t": _ms(t), **_signin_totals_from(counts)}
                    for t, counts in sorted(series.items())
                ],
            }

        body = await self._cached(("signin", window), CACHE_TTL_S[window], load)
        thresholds = await self.service.thresholds()
        severity = signin_severity(body.get("last_15m"), thresholds)
        return {
            **body,
            # signin_failures_15m over the last 15 minutes, with the KV
            # thresholds (the card does not repeat the rule in the browser).
            "state": {"warning": "warning", "serious": "serious"}.get(severity, "ok"),
            "thresholds": {
                k: thresholds.get(k, METRIC_THRESHOLDS[k])
                for k in ("signin_sso_warn", "signin_lockout_warn", "signin_serious")
            },
            "enabled": metrics.metrics_enabled(),
        }

    # -- rule inputs ----------------------------------------------------------------------
    async def rule_inputs(self) -> Optional[dict[str, Any]]:
        """Last complete minutes of HTTP, dependency and sign-in metrics, and
        the LLM baseline, for the attention rules (cached 30 s)."""
        if not self.service.available:
            return None
        now = self._now()

        async def load() -> dict[str, Any]:
            end = complete_end(now)
            longest = max(HTTP_5XX_WINDOW_S, DEP_ERROR_WINDOW_S, SIGNIN_WINDOW_S)
            rows = await self._store.metrics_read(
                [SERIES_HTTP, SERIES_EMBED, SERIES_RERANK, SERIES_AUTH],
                _utc(end - longest),
                _utc(end - longest),
                _utc(end),
                60,
            )
            http_rows = [
                r
                for r in rows
                if r["series"] == SERIES_HTTP and r["t"] >= end - HTTP_5XX_WINDOW_S
            ]
            http = _add_all(http_rows)
            deps = {
                name: _add_all(
                    [
                        r
                        for r in rows
                        if r["series"] == name and r["t"] >= end - DEP_ERROR_WINDOW_S
                    ]
                )
                for name in (SERIES_EMBED, SERIES_RERANK)
            }
            auth = _signin_totals(
                [
                    r
                    for r in rows
                    if r["series"] == SERIES_AUTH and r["t"] >= end - SIGNIN_WINDOW_S
                ]
            )
            baseline = baseline_for(await self._store.kv_get(BASELINE_KEY, None), now)
            return {
                "http": {"count": http["count"], "errors": http["errors"]},
                "deps": {
                    k: {"count": v["count"], "errors": v["errors"]}
                    for k, v in deps.items()
                },
                "signin": auth,
                "llm_baseline": baseline,
            }

        try:
            return await self._cached(("rule_inputs",), RULES_TTL_S, load)
        except Exception as e:
            logger.debug("Ops metric rule inputs failed (%s)", type(e).__name__)
            return None


def _add_all(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = _empty_agg()
    for row in rows:
        _add(agg, row)
    return agg


def _llm_acc() -> dict[str, float]:
    return {
        "calls": 0.0,
        "errors": 0.0,
        "truncated": 0.0,
        "abandoned": 0.0,
        "latency_ms_sum": 0.0,
        "max": 0.0,
    }


def _avg(acc: dict[str, float]) -> Optional[float]:
    return _round(acc["latency_ms_sum"] / acc["calls"]) if acc["calls"] else None


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _signin_totals_from(counts: dict[str, int]) -> dict[str, int]:
    return {
        "sso_errors": int(counts.get("sso_error", 0)),
        "sso_ok": int(counts.get("sso_ok", 0)),
        "token_401": int(counts.get("token_401", 0)),
        "lockouts": int(counts.get("lockout", 0)),
        "login_failed": int(counts.get("login_failed", 0)),
        "login_throttled": int(counts.get("login_throttled", 0)),
        "login_ok": int(counts.get("login_ok", 0)),
    }


def _signin_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["key"]] = counts.get(row["key"], 0) + int(row.get("count") or 0)
    return _signin_totals_from(counts)


def provider_budget() -> Optional[int]:
    """``OPS_PROVIDER_CONCURRENCY_BUDGET``: the provider's concurrent-call
    limit, when known (a ratio is shown only then)."""
    raw = (os.getenv("OPS_PROVIDER_CONCURRENCY_BUDGET") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


# -- attention rules over these metrics ----------------------------------------------


def metric_attention(
    add: Callable[..., None],
    inputs: Optional[dict[str, Any]],
    llm: dict[str, Any],
    thresholds: Mapping[str, Any],
    link: Callable[..., str],
) -> None:
    """``llm_p95_15m``, ``http_5xx_rate_5m``, ``embed_rerank_error_rate_15m``
    and ``signin_failures_15m`` (§1.7), with the merged ``thresholds``."""
    baseline = (inputs or {}).get("llm_baseline")
    llm_min_calls = thresholds.get("llm_min_calls", 20)
    if llm.get("available") and (llm.get("calls") or 0) >= llm_min_calls:
        p95 = llm.get("p95_ms")
        base_ms = baseline.get("p95_ms") if isinstance(baseline, dict) else None
        severity = llm_p95_severity(p95, base_ms, thresholds)
        if severity:
            ratio = (
                round(p95 / base_ms, 2)
                if isinstance(base_ms, (int, float)) and base_ms > 0
                else None
            )
            add(
                "llm_p95_15m",
                severity,
                f"LLM p95 {p95 / 1000:.1f} s over 15 min"
                + (f" ({ratio:.1f}x the usual for this hour)" if ratio else ""),
                link(otab="performance", osub="llm"),
                params={
                    "p95_ms": p95,
                    "baseline_ms": base_ms,
                    "ratio": ratio,
                    "calls": llm.get("calls"),
                },
            )
    if not inputs:
        return
    http = inputs.get("http") or {}
    count = int(http.get("count") or 0)
    if count >= _threshold(thresholds, "http_5xx_min_requests"):
        rate = int(http.get("errors") or 0) / count
        if rate > _threshold(thresholds, "http_5xx_warn"):
            add(
                "http_5xx_rate_5m",
                "serious"
                if rate > _threshold(thresholds, "http_5xx_serious")
                else "warning",
                f"{rate:.1%} of requests failed with a server error over 5 min",
                link(otab="performance", osub="http"),
                params={"rate": round(rate, 4), "requests": count},
            )
    for name, totals in (inputs.get("deps") or {}).items():
        calls = int(totals.get("count") or 0)
        if calls < _threshold(thresholds, "dep_error_min_calls"):
            continue
        rate = int(totals.get("errors") or 0) / calls
        if rate > _threshold(thresholds, "dep_error_warn"):
            add(
                "embed_rerank_error_rate_15m",
                "serious"
                if rate > _threshold(thresholds, "dep_error_serious")
                else "warning",
                f"{'Embedding' if name == SERIES_EMBED else 'Rerank'} error rate "
                f"{rate:.0%} over 15 min",
                link(otab="performance", osub="deps"),
                item_id=f"embed_rerank_error_rate_15m:{name}",
                params={"series": name, "rate": round(rate, 4), "calls": calls},
            )
    signin = inputs.get("signin") or {}
    sso = int(signin.get("sso_errors") or 0)
    lockouts = int(signin.get("lockouts") or 0)
    severity = signin_severity(signin, thresholds)
    if severity:
        add(
            "signin_failures_15m",
            severity,
            f"Sign-in problems over 15 min: {sso} SSO error(s), {lockouts} lockout(s)",
            link(otab="health", osub="signin"),
            params={
                "sso_errors": sso,
                "lockouts": lockouts,
                "token_401": int(signin.get("token_401") or 0),
            },
        )
