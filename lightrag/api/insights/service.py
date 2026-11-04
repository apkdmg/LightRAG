"""Builds what the Insights page shows, from the usage store and the system.

Costs are worked out here, at read time, so a price change applies to history.
Provider-reported costs are used where they exist. Tokens on a model with no
price are counted and reported rather than treated as free.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from math import ceil
from statistics import median
from typing import Any, Awaitable, Callable, Iterable, Optional

from lightrag.usage import (
    BYO_BILLING,
    EXTRACT_PURPOSE,
    KEYWORD_PURPOSE,
    PURPOSES,
    QUERY_PURPOSE,
    VISION_PURPOSE,
)
from lightrag.utils import logger

from . import analytics as an
from .recorder import LOST_ROWS_KEY
from .store import DIMENSIONS, SUM_METRICS, UsageStore, llm_rates

SETTINGS_KEY = "settings"
ANOMALY_STATE_KEY = "anomaly_state"
BACKFILL_KEY = "backfill"
HISTORY_DAYS = 56
CACHE_SECONDS = 30
MERGE_IN_THREAD_ROWS = 5000
# Turning a page shouldn't wait seconds because the last click was a minute ago.
WORKSPACE_ROWS_STALE_SECONDS = 300
WORKSPACE_PAGE_SIZE = 50
WORKSPACE_OPTION_LIMIT = 20
PURPOSE_TOP_WORKSPACES = 20
PURPOSE_BYO_NAMES = 5
# Friday to Monday still counts as the same run of unusual days.
RUN_GAP_DAYS = 3
SEVERITY_ORDER = {"warning": 1, "serious": 2, "critical": 3}
INGEST_PURPOSES = (EXTRACT_PURPOSE, VISION_PURPOSE)
QUESTION_PURPOSES = (KEYWORD_PURPOSE, QUERY_PURPOSE)
SLOT_FOR_PURPOSE = {
    EXTRACT_PURPOSE: "extraction",
    KEYWORD_PURPOSE: "extraction",
    QUERY_PURPOSE: "query",
    VISION_PURPOSE: "vision",
}
# What ``_status_from_rows`` gives a workspace with no rows to compare.
_NO_HISTORY_STATUS = an.status_code(None)
DEFAULT_SETTINGS: dict[str, Any] = {
    "prices": {},
    "budget": {"amount": None, "start": None},
    "rate_limit_tpm": None,
    "myr_rate": None,
    "invoices": {},
}


def _anomaly_days(anomaly: dict[str, Any]) -> list[str]:
    return anomaly.get("run_days") or [anomaly["day"]]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _tokens(row: dict[str, Any]) -> float:
    return row.get("prompt_tokens", 0) + row.get("completion_tokens", 0)


def row_work(purpose: str, operation: str, row: dict[str, Any]) -> float:
    """Units of work: chunks, images, questions or answers, cache hits included."""
    if purpose == EXTRACT_PURPOSE and operation != "extraction":
        return 0
    return row.get("calls", 0) - row.get("errors", 0) + row.get("cache_hits", 0)


def row_cost(row: dict[str, Any], prices: dict[str, Any]) -> tuple[float, float]:
    """Return (cost, tokens with no price) for a row that names a model."""
    price = prices.get(row.get("model") or "")
    cost = row.get("reported_cost", 0) or 0.0
    prompt = row.get("priced_prompt_tokens", 0)
    completion = row.get("priced_completion_tokens", 0)
    if not price:
        return cost, prompt + completion
    cost += prompt / 1e6 * float(price.get("prompt") or 0)
    cost += completion / 1e6 * float(price.get("completion") or 0)
    return cost, 0


class InsightsService:
    def __init__(
        self,
        store: UsageStore,
        *,
        default_rag: Any = None,
        app_state: Any = None,
        now: Optional[Callable[[], datetime]] = None,
    ):
        self.store = store
        self.default_rag = default_rag
        self.app_state = app_state
        self._now = now
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._building: dict[tuple, asyncio.Lock] = {}
        self._refreshes: set[asyncio.Task] = set()

    async def _cached(
        self,
        key: tuple,
        build: Callable[[], Awaitable[Any]],
        *,
        stale_seconds: int = 0,
    ) -> Any:
        """``build`` once per CACHE_SECONDS, however many requests ask at the same time.

        The page asks for the overview and a table together, and both need the
        same expensive result. With ``stale_seconds``, a result that old is
        returned straight away while a new one builds in the background.
        """
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]
        if cached and cached[0] + stale_seconds > now:
            if key not in self._building:
                task = asyncio.create_task(self._build_cached(key, build))
                self._refreshes.add(task)
                task.add_done_callback(self._refresh_done)
            return cached[1]
        return await self._build_cached(key, build)

    async def _build_cached(
        self, key: tuple, build: Callable[[], Awaitable[Any]]
    ) -> Any:
        async with self._building.setdefault(key, asyncio.Lock()):
            cached = self._cache.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            try:
                value = await build()
            finally:
                self._building.pop(key, None)
            self._cache[key] = (time.monotonic() + CACHE_SECONDS, value)
        return value

    def _refresh_done(self, task: asyncio.Task) -> None:
        self._refreshes.discard(task)
        if not task.cancelled() and task.exception():
            logger.warning(f"Could not refresh Insights data: {task.exception()}")

    def now(self) -> datetime:
        return self._now() if self._now else datetime.now(self.store.tz)

    def today(self) -> date:
        return self.now().date()

    def day_start(self, day: date) -> datetime:
        return datetime(day.year, day.month, day.day, tzinfo=self.store.tz)

    # ------------------------------------------------------------------ data

    async def settings(self) -> dict[str, Any]:
        saved = await self.store.kv_get(SETTINGS_KEY, {}) or {}
        return {**DEFAULT_SETTINGS, **saved}

    async def save_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        current = await self.settings()
        current.update(patch)
        await self.store.kv_set(SETTINGS_KEY, current)
        self._cache.clear()
        return current

    async def _rows(
        self,
        start: Any,
        end: Any,
        groups: Iterable[str],
        *,
        filters: Optional[dict[str, str]] = None,
        grain: str = "day",
        exclude_backfill: bool = False,
    ) -> list[dict[str, Any]]:
        """Rollup rows by ``groups`` with tokens, cost and units of work added."""
        groups = tuple(groups)
        key = (
            start,
            end,
            groups,
            tuple(sorted((filters or {}).items())),
            grain,
            exclude_backfill,
        )
        cached = self._cache.get(key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        prices = (await self.settings())["prices"]
        wanted = list(
            dict.fromkeys([*groups, "model", "billing", "purpose", "operation"])
        )
        raw = await self.store.rollup(
            grain,
            start,
            end,
            filters=filters,
            group_by=wanted,
            exclude_backfill=exclude_backfill,
        )
        # A year of every workspace is a lot of rows to add up on the event loop.
        if len(raw) > MERGE_IN_THREAD_ROWS:
            rows = await asyncio.to_thread(self._merge_rows, raw, groups, prices)
        else:
            rows = self._merge_rows(raw, groups, prices)
        self._cache[key] = (time.monotonic() + CACHE_SECONDS, rows)
        if len(self._cache) > 256:
            self._cache.clear()
        return rows

    @staticmethod
    def _merge_rows(
        raw: list[dict[str, Any]], groups: tuple[str, ...], prices: dict[str, Any]
    ) -> list[dict[str, Any]]:
        merged: dict[tuple, dict[str, Any]] = {}
        for row in raw:
            out = merged.setdefault(
                tuple(row[g] for g in groups),
                {
                    **{g: row[g] for g in groups},
                    **dict.fromkeys(SUM_METRICS, 0),
                    "latency_ms_max": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "byo_cost": 0.0,
                    "byo_tokens": 0,
                    "unpriced_tokens": 0,
                    "work": 0,
                },
            )
            for name in SUM_METRICS:
                out[name] += row[name]
            out["latency_ms_max"] = max(out["latency_ms_max"], row["latency_ms_max"])
            tokens = _tokens(row)
            cost, unpriced = row_cost(row, prices)
            out["tokens"] += tokens
            out["cost"] += cost
            out["unpriced_tokens"] += unpriced
            out["work"] += row_work(row["purpose"], row["operation"], row)
            if row["billing"] == BYO_BILLING:
                out["byo_cost"] += cost
                out["byo_tokens"] += tokens
        return list(merged.values())

    @staticmethod
    def _series(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[date, float]]]:
        """{purpose: {metric: {day: value}}} from rows by bucket and purpose."""
        series: dict[str, dict[str, dict[date, float]]] = {
            p: defaultdict(dict) for p in PURPOSES
        }
        for row in rows:
            s = series.get(row["purpose"])
            if s is None:
                continue
            day = row["bucket"]
            attempts = row["calls"] + row["cache_hits"]
            s["tokens"][day] = row["tokens"]
            s["cost"][day] = row["cost"]
            s["byo_cost"][day] = row["byo_cost"]
            s["work"][day] = row["work"]
            s["hits"][day] = row["cache_hits"]
            s["attempts"][day] = attempts
            if attempts:
                s["cache"][day] = row["cache_hits"] / attempts
        return series

    async def learning(self) -> dict[str, Any]:
        first = await self.store.first_recorded_day()
        today = self.today()
        days = (today - first).days + 1 if first else 0
        backfill = await self.store.kv_get(BACKFILL_KEY)
        return {
            "recording_since": first.isoformat() if first else None,
            "days_collected": days,
            "anomalies_ready": days >= an.ANOMALY_MIN_DAYS,
            "anomalies_in_days": max(0, an.ANOMALY_MIN_DAYS - days),
            "forecast_ready": days >= an.FORECAST_MIN_DAYS,
            "forecast_in_days": max(0, an.FORECAST_MIN_DAYS - days),
            "forecast_low_confidence": days < an.FORECAST_CONFIDENT_DAYS,
            "backfill": backfill,
        }

    def _usual_by_now_hours(
        self, hourly: list[dict[str, Any]], day: date, upto_hour: int
    ) -> dict[str, dict[int, list[tuple[date, float]]]]:
        by: dict[str, dict[int, list[tuple[date, float]]]] = {
            p: defaultdict(list) for p in PURPOSES
        }
        for row in hourly:
            bucket = row["bucket"].astimezone(self.store.tz)
            d = bucket.date()
            if d == day or d.weekday() != day.weekday() or row["purpose"] not in by:
                continue
            by[row["purpose"]][bucket.hour].append((d, row["tokens"]))
        return by

    async def _today_status(
        self, workspace: Optional[str]
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[int, float]],
        dict[str, dict[int, float]],
    ]:
        """Each purpose against its usual by this hour, plus the hourly shapes."""
        today = self.today()
        start = self.day_start(today - timedelta(days=HISTORY_DAYS))
        end = self.day_start(today + timedelta(days=1))
        filters = {"workspace": workspace} if workspace else None
        rows = await self._rows(
            start,
            end,
            ["bucket", "purpose"],
            filters=filters,
            grain="hour",
            exclude_backfill=True,
        )
        return self._status_from_rows(rows)

    def _status_from_rows(
        self, rows: list[dict[str, Any]]
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[int, float]],
        dict[str, dict[int, float]],
    ]:
        """The status calculation, from hourly rows by bucket and purpose."""
        now = self.now()
        today = now.date()
        actual: dict[str, dict[int, float]] = {p: defaultdict(float) for p in PURPOSES}
        for row in rows:
            bucket = row["bucket"].astimezone(self.store.tz)
            if bucket.date() == today and row["purpose"] in actual:
                actual[row["purpose"]][bucket.hour] += row["tokens"]
        history = self._usual_by_now_hours(rows, today, now.hour)
        status: dict[str, dict[str, Any]] = {}
        usual_hours: dict[str, dict[int, float]] = {}
        weeks_seen = {
            d for p in PURPOSES for samples in history[p].values() for d, _ in samples
        }
        for p in PURPOSES:
            sample_days = sorted(weeks_seen)
            usual_hours[p] = {}
            for hour, samples in history[p].items():
                values = dict(samples)
                filled = [values.get(d, 0) for d in sample_days]
                usual_hours[p][hour] = median(filled) if filled else 0
            if len(sample_days) < an.MIN_USUAL_SAMPLES:
                status[p] = an.status_code(None)
                continue
            so_far = sum(actual[p].get(h, 0) for h in range(now.hour + 1))
            # The median of each day's total so far. Summing hourly medians would
            # read near zero for a quiet workspace, whose calls fall in odd hours.
            by_day: dict[date, float] = defaultdict(float)
            for hour, samples in history[p].items():
                if hour <= now.hour:
                    for d, tokens in samples:
                        by_day[d] += tokens
            usual = median([by_day.get(d, 0) for d in sample_days])
            if usual < an.MIN_TOKENS_TO_FLAG and so_far < an.MIN_TOKENS_TO_FLAG:
                status[p] = an.status_code(1.0)
            elif usual:
                status[p] = an.status_code(so_far / usual)
            else:
                status[p] = {"code": "unusual", "ratio": None, "level": 1}
        return status, actual, usual_hours

    def _current_models(self) -> dict[str, str]:
        rag = self.default_rag
        if rag is None:
            return {}
        try:
            return {p: rag.get_llm_role_config(p).get("model") or "" for p in PURPOSES}
        except Exception:
            return {}

    def workspace_keys(self, workspace: Optional[str]) -> dict[str, dict[str, Any]]:
        """Who pays for each purpose. API keys are never read here."""
        system = self._current_models()
        store = getattr(self.app_state, "workspace_provider_store", None)
        masked: dict[str, Any] = {}
        if workspace and store is not None:
            try:
                masked = store.get_masked(workspace)
            except Exception as e:
                logger.warning(f"Could not read provider config: {e}")
        out: dict[str, dict[str, Any]] = {}
        for p in PURPOSES:
            slot = masked.get(SLOT_FOR_PURPOSE[p]) or {}
            if slot.get("active"):
                out[p] = {"billing": BYO_BILLING, "model": slot.get("model") or ""}
            else:
                out[p] = {"billing": "system", "model": system.get(p, "")}
        return out

    async def anomalies_open(self, workspace: Optional[str]) -> list[dict[str, Any]]:
        return [
            a
            for a in (await self.anomalies(workspace=workspace))["items"]
            if a["state"] == "open"
        ]

    # -------------------------------------------------------------- overview

    async def overview(
        self,
        range_days: int,
        workspace: Optional[str] = None,
        purpose: Optional[str] = None,
        *,
        end: Optional[date] = None,
    ) -> dict[str, Any]:
        today = end or self.today()
        start = today - timedelta(days=range_days - 1)
        prev_start = start - timedelta(days=range_days)
        history_start = min(
            prev_start, today - timedelta(days=range_days + HISTORY_DAYS)
        )
        filters = {"workspace": workspace} if workspace else None
        rows = await self._rows(
            history_start, today, ["bucket", "purpose"], filters=filters
        )
        series = self._series(rows)
        learning = await self.learning()
        status, hours_actual, hours_usual = await self._today_status(workspace)
        active = [purpose] if purpose else list(PURPOSES)
        period = [start + timedelta(days=i) for i in range(range_days)]
        previous = [prev_start + timedelta(days=i) for i in range(range_days)]
        first = (
            date.fromisoformat(learning["recording_since"])
            if learning["recording_since"]
            else None
        )

        def total(metric: str, days: list[date], purposes: Iterable[str]) -> float:
            return sum(series[p][metric].get(d, 0) for p in purposes for d in days)

        keys = self.workspace_keys(workspace)
        others: dict[str, dict[date, float]] = {}
        if workspace:
            whole = await self._rows(start, today, ["purpose"])
            others = {r["purpose"]: r for r in whole}

        all_tokens = total("tokens", period, PURPOSES) or 1
        all_cost = total("cost", period, PURPOSES) or 1
        purposes_out = []
        for p in PURPOSES:
            tokens = total("tokens", period, [p])
            work = total("work", period, [p])
            attempts = total("attempts", period, [p])
            usual_tokens = [an.usual_value(series[p]["tokens"], d) for d in period]
            usual_work = [an.usual_value(series[p]["work"], d) for d in period]
            usual_per = None
            if all(v is not None for v in usual_tokens) and sum(
                v or 0 for v in usual_work
            ):
                usual_per = sum(usual_tokens) / sum(v or 0 for v in usual_work)
            per_unit = tokens / work if work else None
            others_per = others_ratio = None
            if workspace and p in others:
                rest_work = others[p]["work"] - work
                if rest_work > 0:
                    others_per = (others[p]["tokens"] - tokens) / rest_work
                    if others_per and per_unit:
                        others_ratio = per_unit / others_per
            bars = (
                [hours_actual[p].get(h, 0) for h in range(24)]
                if range_days == 1
                else [series[p]["tokens"].get(d, 0) for d in period]
            )
            purposes_out.append(
                {
                    "id": p,
                    "tokens": tokens,
                    "cost": _round(total("cost", period, [p])),
                    "byo_cost": _round(total("byo_cost", period, [p])),
                    "share_tokens": _round(tokens / all_tokens, 4),
                    "share_cost": _round(total("cost", period, [p]) / all_cost, 4),
                    "status": status[p],
                    "work": work,
                    "per_unit": per_unit,
                    "per_unit_usual": usual_per,
                    "others_per_unit": others_per,
                    "others_ratio": others_ratio,
                    "cache_rate": _round(total("hits", period, [p]) / attempts, 3)
                    if attempts
                    else None,
                    "model": keys[p]["model"],
                    "billing": keys[p]["billing"],
                    "series": bars,
                }
            )

        tokens_now = total("tokens", period, active)
        cost_now = total("cost", period, active)
        tokens_before = total("tokens", previous, active)
        cost_before = total("cost", previous, active)
        hits = total("hits", period, active)
        calls = total("attempts", period, active) - hits
        saved = (tokens_now / calls * hits) if calls else 0
        period_rows = await self._rows(start, today, ["purpose"], filters=filters)
        quality = {
            "estimated_calls": sum(r["estimated_calls"] for r in period_rows),
            "not_measured_calls": sum(r["not_measured"] for r in period_rows),
            "calls": sum(r["calls"] for r in period_rows),
            "lost_rows": await self.store.kv_get(LOST_ROWS_KEY, 0) or 0,
        }
        chart = await self._chart(
            range_days,
            period,
            active,
            series,
            hours_actual,
            hours_usual,
            workspace,
            first,
        )
        out = {
            "available": True,
            "scope": {
                "range": range_days,
                "start": start.isoformat(),
                "end": today.isoformat(),
                "workspace": workspace,
                "purpose": purpose,
                "now_hour": self.now().hour,
            },
            "learning": learning,
            "quality": quality,
            "totals": {
                "tokens": tokens_now,
                "cost": _round(cost_now),
                "byo_cost": _round(total("byo_cost", period, active)),
                "unpriced_tokens": sum(
                    r["unpriced_tokens"] for r in period_rows if r["purpose"] in active
                ),
                "saved_tokens": saved,
                "work": total("work", period, active),
                "attempts": total("attempts", period, active),
                "hits": hits,
                "tokens_change": (tokens_now - tokens_before) / tokens_before
                if tokens_before
                else None,
                "cost_change": (cost_now - cost_before) / cost_before
                if cost_before
                else None,
                "usual_work": sum(
                    an.usual_value(series[p]["work"], d) or 0
                    for p in active
                    for d in period
                ),
                "usual_tokens": sum(
                    an.usual_value(series[p]["tokens"], d) or 0
                    for p in active
                    for d in period
                ),
            },
            "purposes": purposes_out,
            "chart": chart,
            "attention": await self._attention(workspace, status),
        }
        if not workspace and not purpose:
            out["models"] = await self._model_table(start, today)
            out["workspaces"] = (await self.workspaces(range_days, page_size=10))[
                "rows"
            ]
            out["efficiency"] = await self._efficiency(start, today)
        if workspace is None:
            forecast = await self.forecast()
            out["budget"] = forecast.get("budget")
        return out

    async def _chart(
        self,
        range_days: int,
        period: list[date],
        active: list[str],
        series: dict[str, dict[str, dict[date, float]]],
        hours_actual: dict[str, dict[int, float]],
        hours_usual: dict[str, dict[int, float]],
        workspace: Optional[str],
        first: Optional[date],
    ) -> dict[str, Any]:
        lens = active[0] if len(active) == 1 else None
        anomalies = (await self.anomalies(workspace=workspace, purpose=lens))["items"]
        worst: dict[str, str] = {}
        for a in anomalies:
            for day in _anomaly_days(a):
                if SEVERITY_ORDER[a["severity"]] > SEVERITY_ORDER.get(
                    worst.get(day, ""), 0
                ):
                    worst[day] = a["severity"]
        start_dt = self.day_start(period[0] - timedelta(days=0))
        end_dt = self.day_start(period[-1] + timedelta(days=1))
        events = await self.store.events(start_dt, end_dt, workspace=workspace)
        pins: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            d = e["ts"].astimezone(self.store.tz).date().isoformat()
            pins[d].append({"kind": e["kind"], "purpose": e.get("purpose")})
        prices = await self._blended_prices(period[0], period[-1], workspace)
        today = self.today()
        if range_days == 1:
            points = []
            now_hour = self.now().hour
            for h in range(24):
                points.append(
                    {
                        "key": f"{today.isoformat()}T{h:02d}",
                        "day": today.isoformat(),
                        "hour": h,
                        "future": h > now_hour,
                        "tokens": {p: hours_actual[p].get(h, 0) for p in PURPOSES},
                        "cost": {
                            p: hours_actual[p].get(h, 0) / 1e6 * prices.get(p, 0)
                            for p in PURPOSES
                        },
                        "usual": sum(hours_usual[p].get(h, 0) for p in active),
                    }
                )
            return {
                "grain": "hour",
                "points": points,
                "pins": pins.get(today.isoformat(), []),
            }
        points = []
        for d in period:
            key = d.isoformat()
            usual = [an.usual_value(series[p]["tokens"], d) for p in active]
            points.append(
                {
                    "key": key,
                    "day": key,
                    "partial": d == today,
                    "gap": first is None or d < first,
                    "tokens": {p: series[p]["tokens"].get(d, 0) for p in PURPOSES},
                    "cost": {p: _round(series[p]["cost"].get(d, 0)) for p in PURPOSES},
                    "usual": sum(u for u in usual if u) if any(usual) else None,
                    "anomaly": worst.get(key),
                    "pins": pins.get(key, []),
                }
            )
        return {"grain": "day", "points": points}

    async def _blended_prices(
        self, start: date, end: date, workspace: Optional[str]
    ) -> dict[str, float]:
        rows = await self._rows(
            start,
            end,
            ["purpose"],
            filters={"workspace": workspace} if workspace else None,
        )
        return {
            r["purpose"]: (r["cost"] / r["tokens"] * 1e6 if r["tokens"] else 0)
            for r in rows
        }

    async def _attention(
        self, workspace: Optional[str], status: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        above = [p for p in PURPOSES if status[p]["level"] == 1]
        open_items = await self.anomalies_open(workspace)
        reason = None
        today = self.today()
        if above:
            rows = await self._rows(
                today,
                today,
                ["workspace", "source"],
                filters={"workspace": workspace} if workspace else None,
            )
            total = sum(r["tokens"] for r in rows)
            top = max(rows, key=lambda r: r["tokens"], default=None)
            if top and total and top["tokens"] / total >= 0.5:
                reason = {"workspace": top["workspace"], "source": top["source"]}
        documents = None
        if workspace:
            documents = (await self.store.document_status_counts()).get(workspace, {})
        return {
            "above": above,
            "reason": reason,
            "open": len(open_items),
            "critical": sum(1 for a in open_items if a["severity"] == "critical"),
            "documents": documents,
        }

    async def _model_table(self, start: date, end: date) -> list[dict[str, Any]]:
        rows = await self._rows(start, end, ["model", "purpose", "billing"])
        by_model: dict[str, dict[str, Any]] = {}
        purpose_totals: dict[str, float] = defaultdict(float)
        for r in rows:
            if not r["tokens"] and not r["calls"]:
                continue
            purpose_totals[r["purpose"]] += r["tokens"]
            m = by_model.setdefault(
                r["model"],
                {
                    "id": r["model"],
                    "tokens": 0,
                    "byo": False,
                    "by_purpose": defaultdict(float),
                },
            )
            m["tokens"] += r["tokens"]
            m["by_purpose"][r["purpose"]] += r["tokens"]
            m["byo"] = m["byo"] or r["billing"] == BYO_BILLING
        for m in by_model.values():
            m["by_purpose"] = {
                p: _round(v / purpose_totals[p], 4) if purpose_totals[p] else 0
                for p, v in m["by_purpose"].items()
            }
        out = sorted(by_model.values(), key=lambda m: -m["tokens"])
        priced = set((await self.settings())["prices"])
        for model in priced - set(by_model):
            out.append(
                {"id": model, "tokens": 0, "byo": False, "by_purpose": {}, "idle": True}
            )
        return out

    async def _efficiency(self, start: date, end: date) -> dict[str, Any]:
        rows = await self._rows(start, end, ["purpose"])
        settings = await self.settings()
        peak = await self.store.peak_tokens_per_minute(
            self.day_start(max(start, self.today() - timedelta(days=89))),
            self.day_start(end + timedelta(days=1)),
        )
        return {
            "wasted": {
                r["purpose"]: {
                    "tokens": r["wasted_tokens"],
                    "share": _round(r["wasted_tokens"] / r["tokens"], 4)
                    if r["tokens"]
                    else 0,
                    "truncated": r["truncated"],
                    "errors": r["errors"],
                    "abandoned": r["abandoned"],
                }
                for r in rows
            },
            "headroom": {
                "limit_tpm": settings.get("rate_limit_tpm"),
                "peak_tpm": sum(r["tokens"] for r in peak),
                "minute": peak[0]["minute"].isoformat() if peak else None,
                "by_purpose": {r["purpose"]: r["tokens"] for r in peak},
            },
        }

    # --------------------------------------------------------------- purpose

    async def purpose_page(
        self,
        purpose: str,
        range_days: int,
        workspace: Optional[str] = None,
        *,
        end: Optional[date] = None,
    ) -> dict[str, Any]:
        today = self.today()
        start = (end or today) - timedelta(days=range_days - 1)
        filters = {"purpose": purpose}
        if workspace:
            filters["workspace"] = workspace
        ops = await self._rows(start, end or today, ["operation"], filters=filters)
        all_ws = await self._rows(
            start, end or today, ["workspace"], filters={"purpose": purpose}
        )
        models = await self._rows(
            start, end or today, ["model", "billing"], filters=filters
        )
        totals = await self._rows(start, end or today, [], filters=filters)
        total = totals[0] if totals else {}
        tokens = total.get("tokens", 0)
        status, _, _ = await self._today_status(workspace)
        whole_work = sum(r["work"] for r in all_ws)
        whole_tokens = sum(r["tokens"] for r in all_ws)
        average = whole_tokens / whole_work if whole_work else 0
        p95 = await self.store.latency_p95(
            self.day_start(max(start, today - timedelta(days=89))),
            self.day_start(today + timedelta(days=1)),
            filters=filters,
        )
        today_rows = await self._rows(today, today, [], filters=filters)
        today_row = today_rows[0] if today_rows else {}
        today_rates = llm_rates(today_row)
        queues = await self.live()
        keys = self.workspace_keys(workspace)
        byo_share = total.get("byo_tokens", 0) / tokens if tokens else 0
        byo_workspaces = []
        if not workspace:
            byo_rows = await self._rows(
                start,
                today,
                ["workspace"],
                filters={"purpose": purpose, "billing": BYO_BILLING},
            )
            byo_workspaces = sorted(
                (r for r in byo_rows if r["tokens"]), key=lambda r: -r["tokens"]
            )
            byo_workspaces = [r["workspace"] for r in byo_workspaces]
        return {
            "id": purpose,
            "workspace": workspace,
            "model": keys[purpose]["model"],
            "billing": keys[purpose]["billing"],
            "byo_share": _round(byo_share, 4),
            "byo_workspaces": byo_workspaces[:PURPOSE_BYO_NAMES],
            "byo_workspace_count": len(byo_workspaces),
            "status": status[purpose],
            "operations": sorted(
                (
                    {
                        "id": r["operation"],
                        "tokens": r["tokens"],
                        "cost": _round(r["cost"]),
                    }
                    for r in ops
                    if r["tokens"] or r["cache_hits"]
                ),
                key=lambda r: -r["tokens"],
            ),
            "workspaces": sorted(
                (
                    {
                        "id": r["workspace"],
                        "tokens": r["tokens"],
                        "work": r["work"],
                        "per_unit": r["tokens"] / r["work"] if r["work"] else None,
                        "ratio": (r["tokens"] / r["work"]) / average
                        if r["work"] and average
                        else None,
                    }
                    for r in all_ws
                ),
                key=lambda r: -r["tokens"],
            )[:PURPOSE_TOP_WORKSPACES],
            "workspace_count": len(all_ws),
            "models": sorted(
                (
                    {
                        "id": r["model"],
                        "byo": r["billing"] == BYO_BILLING,
                        "tokens": r["tokens"],
                        "share": _round(r["tokens"] / tokens, 4) if tokens else 0,
                        "latency_avg_ms": r["latency_ms_sum"] / r["calls"]
                        if r["calls"]
                        else None,
                        "error_rate": _round(r["errors"] / r["calls"], 4)
                        if r["calls"]
                        else 0,
                    }
                    for r in models
                    if r["tokens"]
                ),
                key=lambda r: -r["tokens"],
            ),
            "split": {
                "prompt": total.get("prompt_tokens", 0),
                "answer": total.get("completion_tokens", 0)
                - total.get("reasoning_tokens", 0),
                "reasoning": total.get("reasoning_tokens", 0),
                "provider_cached": total.get("cached_tokens", 0),
            },
            "health": {
                "queue": queues["purposes"].get(purpose),
                "p95_ms": p95.get(purpose),
                "error_rate": _round(today_rates["error_rate"], 4),
                "truncated_rate": _round(today_rates["truncated_rate"], 4),
                "tokens_today": today_row.get("tokens", 0),
                "not_measured": total.get("not_measured", 0),
            },
            "model_change": await self.model_change_check(purpose, workspace),
        }

    async def model_change_check(
        self, purpose: str, workspace: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Tokens and cost per unit in the 7 days either side of the last change."""
        today = self.today()
        events = await self.store.events(
            self.day_start(today - timedelta(days=90)),
            self.day_start(today + timedelta(days=1)),
            workspace=workspace,
        )
        change = next(
            (
                e
                for e in events
                if e["kind"] == "model_change" and e.get("purpose") == purpose
            ),
            None,
        )
        if change is None:
            return None
        day = change["ts"].astimezone(self.store.tz).date()
        detail = change.get("detail") or {}
        out: dict[str, Any] = {
            "day": day.isoformat(),
            "from": detail.get("old"),
            "to": detail.get("new"),
            "ready": False,
            "ready_on": (day + timedelta(days=7)).isoformat(),
        }
        if today < day + timedelta(days=7):
            return out
        filters = {"purpose": purpose}
        if workspace:
            filters["workspace"] = workspace

        async def stats(first: date, last: date) -> Optional[dict[str, float]]:
            rows = await self._rows(first, last, [], filters=filters)
            if not rows or not rows[0]["work"]:
                return None
            return {
                "tokens": rows[0]["tokens"] / rows[0]["work"],
                "cost": rows[0]["cost"] / rows[0]["work"],
            }

        before = await stats(day - timedelta(days=7), day - timedelta(days=1))
        after = await stats(day, day + timedelta(days=6))
        if before and after:
            out.update(ready=True, before=before, after=after)
        return out

    # ------------------------------------------------------------ workspaces

    async def workspaces(
        self,
        range_days: int,
        purpose: Optional[str] = None,
        *,
        end: Optional[date] = None,
        query: str = "",
        show: str = "all",
        sort: str = "cost",
        page: int = 1,
        page_size: int = WORKSPACE_PAGE_SIZE,
    ) -> dict[str, Any]:
        """One page of the workspace table, filtered and sorted across all of them."""
        everyone = await self._workspace_rows(range_days, purpose, end=end)
        wanted = query.strip().lower()
        matching = [
            r
            for r in everyone
            if (
                show == "all"
                or (show == "byo" and r["byo_for"])
                or (show == "look" and r["needs_look"])
            )
            and (
                not wanted
                or wanted in r["id"].lower()
                or wanted in (r["owner"] or "").lower()
            )
        ]
        # Ties go by name, so a row can't move between pages from one call to the next.
        matching.sort(key=lambda r: (r[sort] is None, -(r[sort] or 0), r["id"]))
        last_page = max(ceil(len(matching) / page_size), 1)
        page = min(page, last_page)
        return {
            "rows": matching[(page - 1) * page_size : page * page_size],
            "total": len(matching),
            "page": page,
            "page_size": page_size,
            "counts": {
                "workspaces": len(everyone),
                "byo": sum(1 for r in everyone if r["byo_for"]),
                "look": sum(1 for r in everyone if r["needs_look"]),
            },
        }

    async def _workspace_rows(
        self,
        range_days: int,
        purpose: Optional[str] = None,
        *,
        end: Optional[date] = None,
    ) -> list[dict[str, Any]]:
        """Every workspace's row. Each page, sort and search reads this one list."""
        today = end or self.today()
        return await self._cached(
            ("workspace_rows", today, range_days, purpose),
            lambda: self._build_workspace_rows(range_days, purpose, today),
            stale_seconds=WORKSPACE_ROWS_STALE_SECONDS,
        )

    async def _build_workspace_rows(
        self, range_days: int, purpose: Optional[str], today: date
    ) -> list[dict[str, Any]]:
        start = today - timedelta(days=range_days - 1)
        prev_start = start - timedelta(days=range_days)
        filters = {"purpose": purpose} if purpose else None
        current = await self._rows(start, today, ["workspace"], filters=filters)
        before = {
            r["workspace"]: r
            for r in await self._rows(
                prev_start, start - timedelta(days=1), ["workspace"], filters=filters
            )
        }
        mix = await self._rows(start, today, ["workspace", "purpose"])
        by_mix: dict[str, dict[str, float]] = defaultdict(dict)
        questions: dict[str, float] = defaultdict(float)
        for r in mix:
            by_mix[r["workspace"]][r["purpose"]] = r["tokens"]
            if r["purpose"] == QUERY_PURPOSE:
                questions[r["workspace"]] += r["work"]
        documents: dict[str, float] = defaultdict(float)
        for r in await self.store.document_days(start, today):
            documents[r["workspace"]] += r["documents"]
        statuses = await self.store.document_status_counts()
        last_active = await self.store.last_activity()
        owners = await self._owners()
        known = set(await self.store.known_workspaces()) | set(owners)
        total_cost = sum(r["cost"] for r in current) or 1
        by_workspace = {r["workspace"]: r for r in current}
        open_by: dict[str, int] = defaultdict(int)
        for a in (await self.anomalies())["items"]:
            if a["state"] == "open" and a["workspace"]:
                open_by[a["workspace"]] += 1
        worst = await self._worst_statuses()
        byo = await self._byo_purposes()
        rows = []
        for ws in sorted(known):
            r = by_workspace.get(ws, {})
            old = before.get(ws, {}).get("tokens", 0)
            tokens = r.get("tokens", 0)
            deleted = ws.startswith("deleted-")
            docs = statuses.get(ws, {})
            status = (
                {"code": "deleted", "ratio": None, "level": 0}
                if deleted
                else worst.get(ws, _NO_HISTORY_STATUS)
            )
            look = not deleted and bool(
                open_by[ws] or docs.get("failed") or status["level"] != 0
            )
            rows.append(
                {
                    "id": ws,
                    "deleted": deleted,
                    "owner": owners.get(ws),
                    "tokens": tokens,
                    "cost": _round(r.get("cost", 0)),
                    "share_cost": _round(r.get("cost", 0) / total_cost, 4),
                    "change": (tokens - old) / old if old else None,
                    "mix": by_mix.get(ws, {}),
                    "documents": documents.get(ws, 0),
                    "questions": questions.get(ws, 0),
                    "byo_for": [] if deleted else byo.get(ws, []),
                    "last_active": last_active.get(ws).isoformat()
                    if last_active.get(ws)
                    else None,
                    "status": status,
                    "open_anomalies": open_by[ws],
                    "failed_documents": docs.get("failed", 0),
                    "waiting_documents": docs.get("pending", 0)
                    + docs.get("processing", 0),
                    "needs_look": look,
                }
            )
        return rows

    async def workspace_options(
        self, query: str = "", limit: int = WORKSPACE_OPTION_LIMIT
    ) -> dict[str, Any]:
        """Live workspaces for the picker, highest spend first."""
        options = await self._cached(
            ("workspace_options",), self._build_workspace_options
        )
        wanted = query.strip().lower()
        matching = [
            o
            for o in options
            if not wanted
            or wanted in o["id"].lower()
            or wanted in (o["owner"] or "").lower()
        ]
        return {"options": matching[:limit], "total": len(matching)}

    async def _build_workspace_options(self) -> list[dict[str, Any]]:
        today = self.today()
        spend = {
            r["workspace"]: r["cost"]
            for r in await self._rows(today - timedelta(days=89), today, ["workspace"])
        }
        owners = await self._owners()
        known = set(await self.store.known_workspaces()) | set(owners)
        return [
            {"id": ws, "owner": owners.get(ws)}
            for ws in sorted(known, key=lambda w: (-spend.get(w, 0), w))
            if ws and not ws.startswith("deleted-")
        ]

    async def _byo_purposes(self) -> dict[str, list[str]]:
        """Workspace -> the purposes it pays for itself, for all of them at once."""
        store = getattr(self.app_state, "workspace_provider_store", None)
        if store is None:
            return {}
        try:
            slots = await asyncio.to_thread(store.workspaces_with_own_provider)
        except Exception as e:
            logger.warning(f"Could not read provider configs: {e}")
            return {}
        out: dict[str, list[str]] = defaultdict(list)
        for p in PURPOSES:
            for ws in slots.get(SLOT_FOR_PURPOSE[p], []):
                out[ws].append(p)
        return dict(out)

    async def _worst_statuses(self) -> dict[str, dict[str, Any]]:
        """``_worst_status`` for every workspace, without a query for each.

        Status only compares today with the same weekday in earlier weeks, so
        those days are all it reads. A workspace with no rows there is left out.
        """
        today = self.today()
        by_workspace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for weeks in range(HISTORY_DAYS // 7 + 1):
            day = today - timedelta(days=7 * weeks)
            for row in await self._rows(
                self.day_start(day),
                self.day_start(day + timedelta(days=1)),
                ["workspace", "bucket", "purpose"],
                grain="hour",
                exclude_backfill=True,
            ):
                by_workspace[row["workspace"]].append(row)
        worst = {}
        for ws, rows in by_workspace.items():
            status, _, _ = self._status_from_rows(rows)
            worst[ws] = max(status.values(), key=lambda s: abs(s["level"]))
        return worst

    async def _owners(self) -> dict[str, str]:
        registry = getattr(self.app_state, "workspace_registry", None)
        if registry is None:
            return {}
        try:
            return {
                o.workspace_id: o.email or o.identity
                for o in await registry.list_owners()
            }
        except Exception as e:
            logger.warning(f"Could not list workspace owners: {e}")
            return {}

    async def _worst_status(self, workspace: str) -> dict[str, Any]:
        status, _, _ = await self._today_status(workspace)
        return max(status.values(), key=lambda s: abs(s["level"]))

    async def workspace_page(
        self, workspace: str, range_days: int, *, end: Optional[date] = None
    ) -> dict[str, Any]:
        today = end or self.today()
        start = today - timedelta(days=range_days - 1)
        filters = {"workspace": workspace}
        owners = await self._owners()
        keys = self.workspace_keys(workspace)
        rows = await self._rows(start, today, ["workspace"])
        ordered = sorted(rows, key=lambda r: -r["cost"])
        total_cost = sum(r["cost"] for r in rows) or 1
        mine = next((r for r in rows if r["workspace"] == workspace), None)
        rank = (
            next(i for i, r in enumerate(ordered) if r["workspace"] == workspace) + 1
            if mine
            else None
        )
        known = await self.store.known_workspaces()
        status = await self._worst_status(workspace)
        docs_now = [r for r in await self.store.document_days(start, today, workspace)]
        history_days = await self.store.document_days(
            today - timedelta(days=range_days + HISTORY_DAYS), today, workspace
        )
        doc_history: dict[date, float] = defaultdict(float)
        for r in history_days:
            doc_history[r["day"]] += r["documents"]
        period = [start + timedelta(days=i) for i in range(range_days)]
        usual_docs = sum(an.usual_value(doc_history, d) or 0 for d in period)
        q_rows = await self._rows(
            today - timedelta(days=range_days + HISTORY_DAYS),
            today,
            ["bucket"],
            filters={"workspace": workspace, "purpose": QUERY_PURPOSE},
        )
        q_history = {r["bucket"]: r["work"] for r in q_rows}
        statuses = (await self.store.document_status_counts()).get(workspace, {})
        sources = await self._rows(start, today, ["source"], filters=filters)
        first_ts = self.day_start(max(start, today - timedelta(days=89)))
        end_ts = self.day_start(today + timedelta(days=1))
        items = await self._labelled_items(first_ts, end_ts, filters, limit=6)
        return {
            "id": workspace,
            "owner": owners.get(workspace),
            "last_active": _iso((await self.store.last_activity()).get(workspace)),
            "status": status,
            "keys": {p: keys[p] for p in PURPOSES},
            "share_cost": _round((mine or {}).get("cost", 0) / total_cost, 4),
            "rank": rank,
            "workspace_count": len(known),
            "documents": sum(r["documents"] for r in docs_now),
            "usual_documents": usual_docs,
            "questions": sum(q_history.get(d, 0) for d in period),
            "usual_questions": sum(an.usual_value(q_history, d) or 0 for d in period),
            "live": {
                "waiting": statuses.get("pending", 0) + statuses.get("preprocessed", 0),
                "processing": statuses.get("processing", 0)
                + statuses.get("analyzing", 0)
                + statuses.get("parsing", 0),
                "failed": statuses.get("failed", 0),
                "processed": statuses.get("processed", 0),
                "questions_today": q_history.get(today, 0),
            },
            "sources": sorted(
                (
                    {
                        "id": r["source"],
                        "tokens": r["tokens"],
                        "cost": _round(r["cost"]),
                    }
                    for r in sources
                    if r["tokens"]
                ),
                key=lambda r: -r["tokens"],
            ),
            "items": items,
            "events": await self._workspace_events(workspace),
        }

    async def _labelled_items(
        self,
        start: datetime,
        end: datetime,
        filters: dict[str, str],
        limit: int = 50,
        kinds: tuple[str, ...] = ("document", "client"),
    ) -> list[dict[str, Any]]:
        """Top documents and API clients, with document names masked here."""
        prices = (await self.settings())["prices"]
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for kind in kinds:
            for r in await self.store.top_items(
                kind, start, end, filters=filters, limit=500
            ):
                row = merged.setdefault(
                    (kind, r["workspace"], r["item"]),
                    {
                        "kind": kind,
                        "workspace": r["workspace"],
                        "item": r["item"],
                        "tokens": 0,
                        "cost": 0.0,
                        "calls": 0,
                        "cache_hits": 0,
                    },
                )
                cost, _ = row_cost(r, prices)
                row["tokens"] += _tokens(r)
                row["cost"] += cost
                row["calls"] += r["calls"]
                row["cache_hits"] += r["cache_hits"]
        rows = sorted(merged.values(), key=lambda r: -r["tokens"])[:limit]
        by_workspace: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            if r["kind"] == "document":
                by_workspace[r["workspace"]].append(r["item"])
        paths: dict[tuple[str, str], str] = {}
        for ws, ids in by_workspace.items():
            for doc_id, path in (await self.store.document_paths(ws, ids)).items():
                paths[(ws, doc_id)] = path
        out = []
        for r in rows:
            entry = {**r, "cost": _round(r["cost"])}
            if r["kind"] == "document":
                entry["doc_id"] = r["item"]
                entry["display"] = an.document_label(
                    r["item"], paths.get((r["workspace"], r["item"]))
                )
                del entry["item"]
            else:
                entry["display"] = {"kind": "client", "label": r["item"], "code": ""}
            out.append(entry)
        return out

    async def _workspace_events(self, workspace: str) -> list[dict[str, Any]]:
        today = self.today()
        rows = []
        for a in (await self.anomalies(workspace=workspace))["items"]:
            rows.append(
                {
                    "day": a["day"],
                    "kind": a["kind"],
                    "type": "anomaly",
                    "state": a["state"],
                    "severity": a["severity"],
                    "shared": a["workspace"] is None,
                    "purposes": [p["id"] for p in a["purposes"]],
                }
            )
        for e in await self.store.events(
            self.day_start(today - timedelta(days=90)),
            self.day_start(today + timedelta(days=1)),
            workspace=workspace,
            limit=50,
        ):
            rows.append(
                {
                    "day": e["ts"].astimezone(self.store.tz).date().isoformat(),
                    "kind": e["kind"],
                    "type": "event",
                    "shared": e.get("workspace") is None,
                    "purposes": [e["purpose"]] if e.get("purpose") else [],
                }
            )
        rows.sort(key=lambda r: r["day"], reverse=True)
        return rows[:12]

    # --------------------------------------------------------------- explore

    async def breakdown(
        self,
        group: str,
        range_days: int,
        filters: dict[str, str],
        day: Optional[str] = None,
        *,
        end: Optional[date] = None,
    ) -> dict[str, Any]:
        today = self.today()
        end = end or today
        start = end - timedelta(days=range_days - 1)
        if day:
            start = end = date.fromisoformat(day)
        clean = {k: v for k, v in filters.items() if k in DIMENSIONS and v}
        if group in ("document", "client"):
            items = await self._labelled_items(
                self.day_start(max(start, today - timedelta(days=89))),
                self.day_start(end + timedelta(days=1)),
                clean,
                limit=200,
                kinds=(group,) if group == "client" else ("document",),
            )
            return {
                "group": group,
                "rows": [
                    {
                        "key": r.get("doc_id") or r["item"],
                        "display": r["display"],
                        "workspace": r["workspace"],
                        "tokens": r["tokens"],
                        "calls": r["calls"],
                        "cache_hits": r["cache_hits"],
                        "cost": r["cost"],
                    }
                    for r in items
                ],
            }
        if group not in DIMENSIONS:
            raise ValueError(f"Unknown group: {group}")
        rows = await self._rows(start, end, [group], filters=clean or None)
        return {
            "group": group,
            "rows": sorted(
                (
                    {
                        "key": r[group],
                        "tokens": r["tokens"],
                        "calls": r["calls"],
                        "cache_hits": r["cache_hits"],
                        "cost": _round(r["cost"]),
                        "byo_cost": _round(r["byo_cost"]),
                        "unpriced_tokens": r["unpriced_tokens"],
                    }
                    for r in rows
                    if r["tokens"] or r["calls"] or r["cache_hits"]
                ),
                key=lambda r: -r["tokens"],
            ),
        }

    async def calls(
        self,
        range_days: int,
        filters: dict[str, str],
        doc_id: Optional[str] = None,
        client: Optional[str] = None,
        day: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Individual calls: counts and timing only, never any text. A failed
        call carries its ``error_kind``; ``request_id`` is the API request it
        served (and filters to one)."""
        today = self.today()
        start = today - timedelta(days=min(range_days, 90) - 1)
        end = today
        if day:
            start = end = date.fromisoformat(day)
        rows = await self.store.calls(
            self.day_start(start),
            self.day_start(end + timedelta(days=1)),
            filters={k: v for k, v in filters.items() if k in DIMENSIONS and v} or None,
            doc_id=doc_id,
            client=client,
            request_id=request_id,
        )
        return [
            {
                "ts": datetime.fromtimestamp(r["ts"], self.store.tz).isoformat(),
                **{
                    k: r.get(k)
                    for k in (
                        "workspace",
                        "purpose",
                        "operation",
                        "source",
                        "model",
                        "billing",
                        "prompt_tokens",
                        "completion_tokens",
                        "reasoning_tokens",
                        "cached_tokens",
                        "latency_ms",
                        "status",
                        "cache_hit",
                        "estimated",
                        "measured",
                        "error_kind",
                        "request_id",
                    )
                },
            }
            for r in rows
        ]

    # -------------------------------------------------------------- why, day

    async def explain_day(
        self, day: date, workspace: Optional[str] = None
    ) -> dict[str, Any]:
        filters = {"workspace": workspace} if workspace else None
        history_start = day - timedelta(days=HISTORY_DAYS)
        rows = await self._rows(
            history_start, day, ["bucket", "purpose"], filters=filters
        )
        series = self._series(rows)
        learning = (await self.learning())["anomalies_ready"]
        purposes = []
        for p in PURPOSES:
            s = series[p]
            tokens = s["tokens"].get(day, 0)
            work = s["work"].get(day, 0)
            usual = an.usual_value(s["tokens"], day)
            usual_work = an.usual_value(s["work"], day)
            cache = s["cache"].get(day)
            usual_cache = an.usual_value(s["cache"], day)
            purposes.append(
                {
                    "id": p,
                    "tokens": tokens,
                    "usual": usual,
                    "cost": _round(s["cost"].get(day, 0)),
                    "work": work,
                    "usual_work": usual_work,
                    "per_unit": tokens / work if work else None,
                    "usual_per_unit": usual / usual_work
                    if usual and usual_work
                    else None,
                    "cache_rate": cache,
                    "usual_cache_rate": usual_cache,
                    "verdict": an.verdict(
                        tokens, usual, work, usual_work, cache, usual_cache
                    ),
                }
            )
        start = self.day_start(day)
        end = self.day_start(day + timedelta(days=1))
        hourly = await self._rows(
            start, end, ["bucket", "purpose"], filters=filters, grain="hour"
        )
        hours: dict[int, dict[str, float]] = defaultdict(dict)
        for r in hourly:
            hours[r["bucket"].astimezone(self.store.tz).hour][r["purpose"]] = r[
                "tokens"
            ]
        _, _, usual_hours = (
            await self._today_status(workspace)
            if day == self.today()
            else (None, None, {})
        )
        events = await self.store.events(start, end, workspace=workspace)
        contributors = []
        if not workspace:
            ws_rows = await self._rows(
                history_start, day, ["bucket", "workspace"], grain="day"
            )
            by_ws: dict[str, dict[date, float]] = defaultdict(dict)
            for r in ws_rows:
                by_ws[r["workspace"]][r["bucket"]] = r["tokens"]
            for ws, values in by_ws.items():
                u = an.usual_value(values, day) or 0
                delta = values.get(day, 0) - u
                if values.get(day, 0) or u:
                    contributors.append(
                        {
                            "id": ws,
                            "tokens": values.get(day, 0),
                            "usual": u,
                            "delta": delta,
                        }
                    )
            contributors.sort(key=lambda c: -abs(c["delta"]))
        anomalies = [
            a
            for a in (await self.anomalies(workspace=workspace))["items"]
            if day.isoformat() in _anomaly_days(a)
        ]
        return {
            "day": day.isoformat(),
            "workspace": workspace,
            "has_baseline": learning,
            "partial": day == self.today(),
            "purposes": purposes,
            "hours": [
                {
                    "hour": h,
                    "tokens": hours.get(h, {}),
                    "usual": sum(
                        (usual_hours or {}).get(p, {}).get(h, 0) for p in PURPOSES
                    )
                    if usual_hours
                    else None,
                }
                for h in range(24)
            ],
            "events": [
                {
                    "at": e["ts"].astimezone(self.store.tz).strftime("%H:%M"),
                    "kind": e["kind"],
                    "workspace": e.get("workspace"),
                    "purpose": e.get("purpose"),
                    "detail": _safe_event_detail(e.get("detail")),
                }
                for e in events
            ],
            "workspaces": contributors[:6],
            "anomalies": anomalies,
        }

    # -------------------------------------------------------------- anomalies

    async def anomalies(
        self, workspace: Optional[str] = None, purpose: Optional[str] = None
    ) -> dict[str, Any]:
        """Unusual days by purpose, grouped by cause. Estimates are left out."""
        detected = await self._cached(("anomalies",), self._detect_anomalies)
        state = await self.store.kv_get(ANOMALY_STATE_KEY, {}) or {}
        items = []
        for a in detected:
            if workspace and a["workspace"] not in (None, workspace):
                continue
            if purpose and purpose not in [p["id"] for p in a["purposes"]]:
                continue
            saved = state.get(a["id"]) or {}
            items.append({**a, "state": saved.get("state", "open")})
        items.sort(key=lambda a: a["day"], reverse=True)
        learning = await self.learning()
        return {
            "ready": learning["anomalies_ready"],
            "days_left": learning["anomalies_in_days"],
            "items": items,
        }

    async def set_anomaly_state(self, anomaly_id: str, state: str, user: str) -> None:
        current = await self.store.kv_get(ANOMALY_STATE_KEY, {}) or {}
        if state == "open":
            current.pop(anomaly_id, None)
        else:
            current[anomaly_id] = {"state": state, "by": user, "at": time.time()}
        await self.store.kv_set(ANOMALY_STATE_KEY, current)

    async def calendar(
        self, workspace: Optional[str] = None, purpose: Optional[str] = None
    ) -> list[dict[str, Any]]:
        today = self.today()
        start = today - timedelta(days=91)
        filters: dict[str, str] = {}
        if workspace:
            filters["workspace"] = workspace
        if purpose:
            filters["purpose"] = purpose
        rows = await self._rows(start, today, ["bucket"], filters=filters or None)
        by_day = {r["bucket"]: r["tokens"] for r in rows}
        flagged = {
            day: a["severity"]
            for a in (await self.anomalies(workspace=workspace, purpose=purpose))[
                "items"
            ]
            for day in _anomaly_days(a)
        }
        return [
            {
                "day": (start + timedelta(days=i)).isoformat(),
                "tokens": by_day.get(start + timedelta(days=i), 0),
                "severity": flagged.get((start + timedelta(days=i)).isoformat()),
            }
            for i in range(92)
        ]

    async def _detect_anomalies(self) -> list[dict[str, Any]]:
        today = self.today()
        first = await self.store.first_recorded_day()
        if first is None or (today - first).days + 1 < an.ANOMALY_MIN_DAYS:
            return []
        window = 92
        start = today - timedelta(days=window + HISTORY_DAYS)
        rows = await self._rows(
            start, today, ["bucket", "purpose", "workspace"], exclude_backfill=True
        )
        check_from = max(
            first + timedelta(days=an.ANOMALY_MIN_DAYS - 1),
            today - timedelta(days=window),
        )
        events = await self.store.events(
            self.day_start(check_from),
            self.day_start(today + timedelta(days=1)),
            limit=2000,
        )
        events_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rejected: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
        for e in events:
            key = e["ts"].astimezone(self.store.tz).date().isoformat()
            events_by_day[key].append(
                {
                    "day": key,
                    "at": e["ts"].astimezone(self.store.tz).strftime("%H:%M"),
                    "kind": e["kind"],
                    "workspace": e.get("workspace"),
                    "purpose": e.get("purpose"),
                }
            )
            if e["kind"] == "provider_rejection":
                rejected[(key, e.get("workspace"))] = e

        # Thousands of workspaces take seconds to go through, and other requests
        # have to keep moving meanwhile.
        frames, out = await asyncio.to_thread(
            self._flag_days, rows, today, check_from, events_by_day
        )
        seen_days = {(a["day"], a["workspace"]) for a in out}
        for (day_key, ws), event in rejected.items():
            detail = event.get("detail") or {}
            if (day_key, ws) in seen_days:
                for a in out:
                    if (a["day"], a["workspace"]) == (day_key, ws):
                        a["kind"] = "provider_rejected"
                        a["run_key"] = (ws, "provider_rejected", event.get("purpose"))
                        a["severity"] = "critical"
                continue
            out.append(
                {
                    "id": f"{day_key}:{ws or 'system'}:rejected",
                    "day": day_key,
                    "run_key": (ws, "provider_rejected", event.get("purpose")),
                    "workspace": ws,
                    "kind": "provider_rejected",
                    "severity": "critical",
                    "purposes": [
                        {"id": event.get("purpose") or EXTRACT_PURPOSE, "ratio": None}
                    ],
                    "extra_cost": 0,
                    "detail": {
                        "status": detail.get("status"),
                        "byo": detail.get("byo"),
                    },
                    "events": events_by_day.get(day_key, []),
                }
            )
        out = self._merge_runs(out)
        out.extend(await self._live_anomalies(today, frames))
        out.extend(await self._new_model_anomalies(today, check_from))
        return out

    def _flag_days(
        self,
        rows: list[dict[str, Any]],
        today: date,
        check_from: date,
        events_by_day: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[Optional[str], Any], list[dict[str, Any]]]:
        """Each workspace's series, and the unusual days found in them."""
        frames: dict[Optional[str], dict[str, dict[str, dict[date, float]]]] = {
            None: {p: defaultdict(dict) for p in PURPOSES}
        }
        day_cost: dict[date, float] = defaultdict(float)
        cost_by: dict[tuple[Optional[str], str, date], float] = {}
        for r in rows:
            ws, p, d = r["workspace"], r["purpose"], r["bucket"]
            if p not in PURPOSES:
                continue
            frames.setdefault(ws, {q: defaultdict(dict) for q in PURPOSES})
            attempts = r["calls"] + r["cache_hits"]
            for scope in (None, ws):
                s = frames[scope][p]
                s["tokens"][d] = s["tokens"].get(d, 0) + r["tokens"]
                s["work"][d] = s["work"].get(d, 0) + r["work"]
                s["hits"][d] = s["hits"].get(d, 0) + r["cache_hits"]
                s["attempts"][d] = s["attempts"].get(d, 0) + attempts
                cost_by[(scope, p, d)] = cost_by.get((scope, p, d), 0) + r["cost"]
            day_cost[d] += r["cost"]
        out: list[dict[str, Any]] = []
        # Yesterday and earlier are judged on complete days; today waits, since
        # an unfinished day always looks low.
        system_flagged: set[tuple[date, str]] = set()
        for scope, frame in frames.items():
            # Most workspaces are too small to flag on any day, so don't try each one.
            checks = [
                (
                    p,
                    frame[p]["tokens"],
                    frame[p]["work"],
                    {
                        day: frame[p]["hits"].get(day, 0) / a
                        for day, a in frame[p]["attempts"].items()
                        if a
                    },
                )
                for p in PURPOSES
                if max(frame[p]["tokens"].values(), default=0) >= an.MIN_TOKENS_TO_FLAG
            ]
            if not checks:
                continue
            d = check_from
            while d < today:
                flags = []
                for p, tokens, work, cache in checks:
                    flag = an.detect_purpose_day(p, d, tokens, work, cache)
                    if flag is None or (scope is not None and (d, p) in system_flagged):
                        continue
                    flags.append(flag)
                if scope is None:
                    system_flagged.update((d, f["purpose"]) for f in flags)
                if flags:
                    extra = sum(
                        max(0.0, f["tokens"] - f["usual"])
                        / (f["tokens"] or 1)
                        * cost_by.get((scope, f["purpose"], d), 0)
                        for f in flags
                    )
                    workspace = scope
                    if scope is None:
                        workspace = self._main_contributor(frames, flags, d)
                    # A small workspace doubling is not worth an alert.
                    if scope is None or extra >= 0.02 * max(day_cost.get(d, 0), 1e-9):
                        out.append(
                            self._anomaly(
                                d,
                                workspace,
                                flags,
                                extra,
                                day_cost.get(d, 0),
                                events_by_day.get(d.isoformat(), []),
                                scope,
                            )
                        )
                d += timedelta(days=1)
        return frames, out

    @staticmethod
    def _merge_runs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One item per run of days, so a lasting change isn't new every morning.

        A run keeps its first day's ID: marking it as expected, or having been
        alerted about it, still holds while it goes on.
        """
        runs: dict[Any, dict[str, Any]] = {}
        out: list[dict[str, Any]] = []
        for a in sorted(items, key=lambda a: a["day"]):
            key = a.pop("run_key")
            run = runs.get(key)
            gap = (
                (date.fromisoformat(a["day"]) - date.fromisoformat(run["day"])).days
                if run
                else None
            )
            if run is None or gap > RUN_GAP_DAYS:
                runs[key] = {**a, "run_days": [a["day"]], "events": list(a["events"])}
                out.append(runs[key])
                continue
            run["run_days"].append(a["day"])
            run["day"] = a["day"]
            run["extra_cost"] = _round(run["extra_cost"] + a["extra_cost"])
            run["events"] += [e for e in a["events"] if e not in run["events"]]
            run["purposes"] = a["purposes"]
            if SEVERITY_ORDER[a["severity"]] > SEVERITY_ORDER[run["severity"]]:
                run["severity"] = a["severity"]
            if a["workspace"] != run["workspace"]:
                run["workspace"] = None
        return out

    @staticmethod
    def _main_contributor(
        frames: dict[Optional[str], Any], flags: list[dict[str, Any]], d: date
    ) -> Optional[str]:
        """The workspace behind most of the extra tokens, if one clearly is."""
        excess: dict[str, float] = defaultdict(float)
        for ws, frame in frames.items():
            if ws is None:
                continue
            for f in flags:
                values = frame[f["purpose"]]["tokens"]
                excess[ws] += values.get(d, 0) - (an.usual_value(values, d) or 0)
        total = sum(v for v in excess.values() if v > 0)
        if not total:
            return None
        ws, value = max(excess.items(), key=lambda kv: kv[1])
        return ws if value / total >= 0.6 else None

    @staticmethod
    def _anomaly(
        d: date,
        workspace: Optional[str],
        flags: list[dict[str, Any]],
        extra: float,
        day_cost: float,
        events: list[dict[str, Any]],
        scope: Optional[str],
    ) -> dict[str, Any]:
        top = max(flags, key=lambda f: abs(f["tokens"] - f["usual"]))
        return {
            "id": f"{d.isoformat()}:{workspace or 'system'}",
            "day": d.isoformat(),
            "run_key": (scope, top["kind"], top["purpose"]),
            "workspace": workspace,
            "kind": top["kind"],
            "severity": an.severity(flags, extra, day_cost),
            "purposes": [
                {
                    "id": f["purpose"],
                    "ratio": f["ratio"],
                    "tokens": f["tokens"],
                    "usual": f["usual"],
                }
                for f in flags
            ],
            "extra_cost": _round(extra),
            "events": events,
        }

    async def _live_anomalies(
        self, today: date, frames: dict[Optional[str], Any]
    ) -> list[dict[str, Any]]:
        """Extraction going silent while documents wait."""
        statuses = await self.store.document_status_counts()
        now = self.now()
        since = now - timedelta(hours=2)
        rows = await self._rows(
            since.replace(minute=0, second=0, microsecond=0),
            now + timedelta(hours=1),
            ["workspace"],
            filters={"purpose": EXTRACT_PURPOSE},
            grain="hour",
        )
        recent = {r["workspace"]: r["calls"] + r["cache_hits"] for r in rows}
        out = []
        for ws, counts in statuses.items():
            waiting = counts.get("pending", 0) + counts.get("processing", 0)
            if waiting and not recent.get(ws) and ws in frames:
                out.append(
                    {
                        "id": f"{today.isoformat()}:{ws}:silent",
                        "day": today.isoformat(),
                        "workspace": ws,
                        "kind": "silent",
                        "severity": "serious",
                        "purposes": [{"id": EXTRACT_PURPOSE, "ratio": None}],
                        "extra_cost": 0,
                        "detail": {"waiting": waiting},
                        "events": [],
                    }
                )
        return out

    async def _new_model_anomalies(
        self, today: date, since: date
    ) -> list[dict[str, Any]]:
        rows = await self._rows(
            today - timedelta(days=HISTORY_DAYS + 7),
            today,
            ["bucket", "model", "purpose"],
            exclude_backfill=True,
        )
        first_seen: dict[tuple[str, str], date] = {}
        for r in rows:
            if not r["model"] or not r["tokens"]:
                continue
            key = (r["model"], r["purpose"])
            if key not in first_seen or r["bucket"] < first_seen[key]:
                first_seen[key] = r["bucket"]
        floor = rows and min(r["bucket"] for r in rows)
        out = []
        for (model, p), seen in first_seen.items():
            if seen > since and seen > floor and today - seen <= timedelta(days=7):
                out.append(
                    {
                        "id": f"{seen.isoformat()}:model:{model}:{p}",
                        "day": seen.isoformat(),
                        "workspace": None,
                        "kind": "new_model",
                        "severity": "warning",
                        "purposes": [{"id": p, "ratio": None}],
                        "extra_cost": 0,
                        "detail": {"model": model},
                        "events": [],
                    }
                )
        return out

    # ------------------------------------------------------------------ work

    async def work(
        self,
        side: str,
        range_days: int,
        workspace: Optional[str] = None,
        purpose: Optional[str] = None,
        *,
        end: Optional[date] = None,
    ) -> dict[str, Any]:
        """Work against tokens, for one side of the system at a time."""
        today = end or self.today()
        range_days = max(7, range_days)
        start = today - timedelta(days=range_days - 1)
        sides = INGEST_PURPOSES if side == "ingest" else QUESTION_PURPOSES
        active = [purpose] if purpose in sides else list(sides)
        filters = {"workspace": workspace} if workspace else None
        rows = await self._rows(start, today, ["bucket", "purpose"], filters=filters)
        tokens: dict[date, dict[str, float]] = defaultdict(dict)
        units: dict[date, float] = defaultdict(float)
        for r in rows:
            if r["purpose"] not in active:
                continue
            tokens[r["bucket"]][r["purpose"]] = r["tokens"]
            if side == "questions" and r["purpose"] == (active[-1]):
                units[r["bucket"]] += r["work"]
            if purpose == VISION_PURPOSE and r["purpose"] == VISION_PURPOSE:
                units[r["bucket"]] += r["work"]
        doc_rows = await self.store.document_days(start, today, workspace)
        documents: dict[date, float] = defaultdict(float)
        docs_by_source: dict[str, float] = defaultdict(float)
        for r in doc_rows:
            documents[r["day"]] += r["documents"]
            docs_by_source[r["source"]] += r["documents"]
        if side == "ingest" and purpose != VISION_PURPOSE:
            units = documents
        days = []
        for i in range(range_days):
            d = start + timedelta(days=i)
            total = sum(tokens[d].values())
            days.append(
                {
                    "day": d.isoformat(),
                    "partial": d == today,
                    "tokens": tokens[d],
                    "units": units.get(d, 0),
                    "per_unit": total / units[d] if units.get(d) else None,
                }
            )
        by_source = await self._rows(
            start, today, ["source", "purpose"], filters=filters
        )
        table: dict[str, dict[str, Any]] = {}
        for r in by_source:
            if r["purpose"] not in sides or not (r["tokens"] or r["work"]):
                continue
            row = table.setdefault(
                r["source"],
                {
                    "id": r["source"],
                    "tokens": {},
                    "total": 0,
                    "cost": 0.0,
                    "hits": 0,
                    "attempts": 0,
                    "wasted": 0,
                    "work": 0,
                },
            )
            row["tokens"][r["purpose"]] = r["tokens"]
            row["total"] += r["tokens"]
            row["cost"] += r["cost"]
            row["hits"] += r["cache_hits"]
            row["attempts"] += r["calls"] + r["cache_hits"]
            row["wasted"] += r["wasted_tokens"]
            if r["purpose"] in (
                EXTRACT_PURPOSE if side == "ingest" else QUERY_PURPOSE,
            ):
                row["work"] += r["work"]
        sources = []
        for row in table.values():
            unit = docs_by_source.get(row["id"], 0) if side == "ingest" else row["work"]
            sources.append(
                {
                    "id": row["id"],
                    "tokens": row["tokens"],
                    "total": row["total"],
                    "cost": _round(row["cost"]),
                    "units": unit,
                    "per_unit": row["total"] / unit if unit else None,
                    "cache_rate": _round(row["hits"] / row["attempts"], 3)
                    if row["attempts"]
                    else None,
                    "wasted_share": _round(row["wasted"] / row["total"], 4)
                    if row["total"]
                    else 0,
                }
            )
        sources.sort(key=lambda r: -r["total"])
        return {"side": side, "purposes": active, "days": days, "sources": sources}

    # -------------------------------------------------------------- forecast

    async def forecast(
        self,
        doc_change: float = 0.0,
        question_change: float = 0.0,
        bulk_documents: int = 0,
        swap: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> dict[str, Any]:
        """13 weeks per purpose. Totals include BYO, with each share shown."""
        learning = await self.learning()
        settings = await self.settings()
        today = self.today()
        last_full = today - timedelta(days=1)
        start = last_full - timedelta(days=181)
        rows = await self._rows(start, last_full, ["bucket", "purpose"])
        series = self._series(rows)
        first_days = [r["bucket"] for r in rows if r["tokens"]]
        out: dict[str, Any] = {
            "ready": learning["forecast_ready"],
            "learning": learning,
        }
        if not first_days or not learning["forecast_ready"]:
            out["budget"] = None
            return out
        first_day = min(first_days)
        span = [
            first_day + timedelta(days=i)
            for i in range((last_full - first_day).days + 1)
        ]
        recent = [d for d in span if d > last_full - timedelta(days=28)]
        docs28 = (
            sum(
                r["documents"]
                for r in await self.store.document_days(recent[0], last_full)
            )
            if recent
            else 0
        )
        plans: dict[str, dict[str, Any]] = {}
        for p in PURPOSES:
            s = series[p]
            daily = [s["tokens"].get(d, 0) for d in span]
            tok28 = sum(s["tokens"].get(d, 0) for d in recent)
            cost28 = sum(s["cost"].get(d, 0) for d in recent)
            byo28 = sum(s["byo_cost"].get(d, 0) for d in recent)
            price = cost28 / tok28 * 1e6 if tok28 else 0
            if swap == p:
                price *= 0.6
            ingest = p in INGEST_PURPOSES
            growth = doc_change if ingest else question_change
            bulk = bulk_documents * (tok28 / docs28) if ingest and docs28 else 0
            plan = an.forecast_series(daily, growth, bulk)
            plan.update(
                price=price,
                byo_share=byo28 / cost28 if cost28 else 0,
                tok28=tok28,
                cost28=cost28,
                work28=sum(s["work"].get(d, 0) for d in recent),
                actual=self._weekly(daily),
            )
            plans[p] = plan
        active = [purpose] if purpose else list(PURPOSES)
        weeks = min(13, len(plans[PURPOSES[0]]["actual"]))
        actual = (
            [sum(plans[p]["actual"][-weeks + i] for p in active) for i in range(weeks)]
            if weeks
            else []
        )
        sys_actual = (
            [
                sum(
                    plans[p]["actual"][-weeks + i] * (1 - plans[p]["byo_share"])
                    for p in active
                )
                for i in range(weeks)
            ]
            if weeks
            else []
        )
        points = []
        for k in range(an.FORECAST_WEEKS):
            points.append(
                {
                    "week": k + 1,
                    "value": sum(plans[p]["points"][k] for p in active),
                    "low": sum(plans[p]["low"][k] for p in active),
                    "high": sum(plans[p]["high"][k] for p in active),
                    "system": sum(
                        plans[p]["points"][k] * (1 - plans[p]["byo_share"])
                        for p in active
                    ),
                }
            )

        def week_cost(k: int, *, byo: bool = True, high: bool = False) -> float:
            total = 0.0
            for p in PURPOSES:
                value = plans[p]["high" if high else "points"][k]
                share = 1.0 if byo else 1 - plans[p]["byo_share"]
                total += value * share / 1e6 * plans[p]["price"]
            return total

        def part(fn: Callable[[int], float], days: int) -> float:
            whole, extra = divmod(days, 7)
            return sum(fn(k) for k in range(whole)) + (
                fn(whole) * extra / 7 if extra else 0
            )

        cost_of = lambda days, byo=True: part(lambda k: week_cost(k, byo=byo), days)  # noqa: E731
        cost90 = cost_of(90)
        system90 = cost_of(90, False)
        high90 = part(lambda k: week_cost(k, high=True), 90)
        out.update(
            week_ending=[
                (last_full - timedelta(days=7 * (weeks - 1 - i))).isoformat()
                for i in range(weeks)
            ],
            actual=actual,
            actual_system=sys_actual,
            points=points,
            next30={
                "tokens": sum(self._part_list(plans[p]["points"], 30) for p in active),
                "low": sum(self._part_list(plans[p]["low"], 30) for p in active),
                "high": sum(self._part_list(plans[p]["high"], 30) for p in active),
                "cost": _round(cost_of(30)),
                "byo_cost": _round(cost_of(30) - cost_of(30, False)),
            },
            next90={
                "tokens": sum(self._part_list(plans[p]["points"], 90) for p in active),
                "low": sum(self._part_list(plans[p]["low"], 90) for p in active),
                "high": sum(self._part_list(plans[p]["high"], 90) for p in active),
                "cost": _round(cost90),
                "byo_cost": _round(cost90 - system90),
                "high_cost": _round(high90),
            },
            rows=[
                {
                    "id": p,
                    "grows_with": "documents" if p in INGEST_PURPOSES else "questions",
                    "tokens": self._part_list(plans[p]["points"], 90),
                    "low": self._part_list(plans[p]["low"], 90),
                    "high": self._part_list(plans[p]["high"], 90),
                    "cost": _round(
                        self._part_list(plans[p]["points"], 90)
                        / 1e6
                        * plans[p]["price"]
                    ),
                    "byo_share": _round(plans[p]["byo_share"], 4),
                    "share_cost": _round(
                        self._part_list(plans[p]["points"], 90)
                        / 1e6
                        * plans[p]["price"]
                        / (cost90 or 1),
                        4,
                    ),
                }
                for p in PURPOSES
            ],
            units=self._unit_costs(plans, docs28),
        )
        out["budget"] = await self._budget(settings, plans, week_cost, high90)
        out["bill"] = await self._bill_check(settings)
        out["myr_rate"] = settings.get("myr_rate")
        return out

    @staticmethod
    def _weekly(daily: list[float]) -> list[float]:
        weeks: list[float] = []
        for end in range(len(daily), 6, -7):
            weeks.insert(0, sum(daily[end - 7 : end]))
        return weeks

    @staticmethod
    def _part_list(values: list[float], days: int) -> float:
        whole, extra = divmod(days, 7)
        total = sum(values[:whole])
        if extra and whole < len(values):
            total += values[whole] * extra / 7
        return total

    @staticmethod
    def _unit_costs(plans: dict[str, Any], docs28: float) -> dict[str, Any]:
        e, v, k, q = (plans[p] for p in PURPOSES)
        return {
            "document": {
                "tokens": (e["tok28"] + v["tok28"]) / docs28,
                "cost": (e["cost28"] + v["cost28"]) / docs28,
            }
            if docs28
            else None,
            "image": {
                "tokens": v["tok28"] / v["work28"],
                "cost": v["cost28"] / v["work28"],
            }
            if v["work28"]
            else None,
            "question": {
                "tokens": (k["tok28"] / k["work28"] if k["work28"] else 0)
                + (q["tok28"] / q["work28"] if q["work28"] else 0),
                "cost": (k["cost28"] / k["work28"] if k["work28"] else 0)
                + (q["cost28"] / q["work28"] if q["work28"] else 0),
            }
            if q["work28"]
            else None,
        }

    async def _budget(
        self,
        settings: dict[str, Any],
        plans: dict[str, Any],
        week_cost: Callable[..., float],
        high90: float,
    ) -> Optional[dict[str, Any]]:
        budget = settings.get("budget") or {}
        amount = budget.get("amount")
        if amount is None:
            return {"set": False, "suggested_purchase": None}
        today = self.today()
        since = date.fromisoformat(budget["start"]) if budget.get("start") else today
        spent = (
            sum(r["cost"] for r in await self._rows(since, today, []))
            if since <= today
            else 0
        )
        left = float(amount) - spent
        weeks = range(an.FORECAST_WEEKS)
        with_byo = an.days_until_spent(left, [week_cost(k) for k in weeks])
        without = an.days_until_spent(left, [week_cost(k, byo=False) for k in weeks])

        def to_date(days: Optional[float]) -> Optional[str]:
            return (
                None
                if days is None
                else (today + timedelta(days=round(days))).isoformat()
            )

        return {
            "set": True,
            "amount": float(amount),
            "start": since.isoformat(),
            "spent": _round(spent),
            "left": _round(left),
            "runs_out": to_date(with_byo),
            "days_left": None if with_byo is None else round(with_byo),
            "runs_out_without_byo": to_date(without),
            "suggested_purchase": an.suggested_purchase(high90, left),
        }

    async def _bill_check(self, settings: dict[str, Any]) -> list[dict[str, Any]]:
        """Your key's usage against each invoice. Invoices don't include BYO."""
        out = []
        for month, invoice in sorted(
            (settings.get("invoices") or {}).items(), reverse=True
        )[:12]:
            year, number = map(int, month.split("-"))
            first = date(year, number, 1)
            last = (first + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            rows = await self._rows(first, last, [], filters={"billing": "system"})
            mine = rows[0]["cost"] if rows else 0.0
            out.append(
                {
                    "month": month,
                    "system_cost": _round(mine),
                    "invoice": float(invoice),
                    "difference": (float(invoice) - mine) / mine if mine else None,
                }
            )
        return out

    # ------------------------------------------------------------------ live

    async def live(self) -> dict[str, Any]:
        """Queues of the instances already in memory. Never loads a workspace."""
        instances: list[Any] = []
        if self.default_rag is not None:
            instances.append(self.default_rag)
        manager = getattr(self.app_state, "workspace_manager", None)
        if manager is not None:
            instances.extend(manager.loaded_instances())
        totals: dict[str, dict[str, int]] = {
            p: {"running": 0, "queued": 0, "max_async": 0, "instances": 0}
            for p in PURPOSES
        }
        for rag in instances:
            try:
                status = await rag.get_llm_queue_status()
            except Exception:
                continue
            for p in PURPOSES:
                item = status.get(p) or {}
                totals[p]["running"] += item.get("running", 0) or 0
                totals[p]["queued"] += item.get("queued", 0) or 0
                totals[p]["max_async"] += item.get("max_async", 0) or 0
                totals[p]["instances"] += 1
        return {"purposes": totals, "loaded_workspaces": len(instances)}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _safe_event_detail(detail: Optional[dict[str, Any]]) -> dict[str, Any]:
    allowed = ("old", "new", "status", "byo", "slot", "version", "paused")
    return {k: v for k, v in (detail or {}).items() if k in allowed}
