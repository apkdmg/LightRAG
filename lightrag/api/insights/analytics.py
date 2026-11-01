"""Plain calculations behind the dashboard, with no LLM and no storage access.

Everything here is deterministic, so an explanation or a forecast can be repeated
and costs nothing. Inputs are rows of numbers; explanations come back as codes
and parameters, which the WebUI turns into translated text.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from functools import lru_cache
from statistics import median
from typing import Any, Iterable, Optional

USUAL_WEEKS = 8
MIN_USUAL_SAMPLES = 2
ANOMALY_MIN_DAYS = 14
FORECAST_MIN_DAYS = 28
FORECAST_CONFIDENT_DAYS = 56
FORECAST_WEEKS = 13
MIN_TOKENS_TO_FLAG = 2_000

_EMAIL_PATH = re.compile(
    r"^email_(?P<bundle>[0-9a-f]{4,})__(?P<part>master|inline|attachment)"
    r"(?:_\d+_(?P<name>.*))?$"
)


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def mask_name(name: str) -> str:
    """``Senate minutes, September.pdf`` -> ``Sen•••er.pdf``.

    The first three and last two characters and the file type stay. Numbers are
    always hidden, because file names can contain IC or matric numbers. The dot
    count is fixed, so the length stays hidden too.
    """
    base = name.rsplit("/", 1)[-1].strip()
    stem, dot, extension = base.rpartition(".")
    if not dot or not stem or len(extension) > 8:
        stem, extension = base, ""
    head = stem[:1] if len(stem) <= 5 else stem[:3]
    head = "".join(ch for ch in head if not ch.isdigit())
    tail = stem[-2:] if len(stem) > 5 else ""
    if _has_digit(tail):
        tail = ""
    return f"{head}•••{tail}" + (f".{extension.lower()}" if extension else "")


def document_label(doc_id: str, path: Optional[str]) -> dict[str, Any]:
    """How a document appears on the page. The full name never leaves the server."""
    hex_id = doc_id[4:] if doc_id.startswith("doc-") else doc_id
    code = hex_id[:4]
    if not path:
        return {"kind": "deleted", "label": "", "code": code}
    match = _EMAIL_PATH.match(path.removesuffix(".txt"))
    if match:
        bundle = match.group("bundle")[:4]
        if match.group("part") == "master":
            return {"kind": "email", "label": "", "code": bundle}
        kind = (
            "email_attachment" if match.group("part") == "attachment" else "email_image"
        )
        return {
            "kind": kind,
            "label": mask_name(match.group("name") or ""),
            "code": bundle,
        }
    return {"kind": "document", "label": mask_name(path), "code": code}


@lru_cache(maxsize=1024)
def _same_weekdays(day: date) -> tuple[date, ...]:
    """The same weekday in the previous 8 weeks. Asked for millions of times a scan."""
    return tuple(day - timedelta(days=7 * n) for n in range(1, USUAL_WEEKS + 1))


def usual_value(history: dict[date, float], day: date) -> Optional[float]:
    """Median for the same weekday over the previous 8 weeks, if there's enough."""
    samples = [history[d] for d in _same_weekdays(day) if d in history]
    return median(samples) if len(samples) >= MIN_USUAL_SAMPLES else None


def verdict(
    tokens: float,
    usual: Optional[float],
    work: float,
    usual_work: Optional[float],
    cache_rate: Optional[float] = None,
    usual_cache_rate: Optional[float] = None,
) -> str:
    """Why a purpose differs from usual, in one code."""
    if not usual:
        return "unknown"
    token_ratio = tokens / usual
    if abs(token_ratio - 1) < 0.15:
        return "normal"
    work_ratio = work / usual_work if usual_work else 1.0
    per_ratio = token_ratio / work_ratio if work_ratio else token_ratio
    if token_ratio > 1:
        cache_drop = (
            usual_cache_rate - cache_rate
            if cache_rate is not None and usual_cache_rate is not None
            else 0
        )
        if cache_drop > 0.1 and work_ratio < 1.2:
            return "cache_misses"
        return "more_work" if work_ratio >= per_ratio else "more_per_unit"
    return "less_work" if work_ratio <= per_ratio else "less_per_unit"


def status_code(ratio: Optional[float]) -> dict[str, Any]:
    """Today's status for a purpose, from today so far against usual by now."""
    if ratio is None:
        return {"code": "learning", "ratio": None, "level": 0}
    if ratio >= 1.5:
        return {"code": "above", "ratio": round(ratio, 1), "level": 1}
    if ratio <= 0.5:
        return {"code": "below", "ratio": round(ratio, 1), "level": -1}
    return {"code": "normal", "ratio": round(ratio, 1), "level": 0}


def _spread(history: list[float]) -> float:
    if len(history) < 3:
        return 0.0
    middle = median(history)
    return median(abs(x - middle) for x in history) * 1.4826


def detect_purpose_day(
    purpose: str,
    day: date,
    tokens: dict[date, float],
    work: dict[date, float],
    cache: dict[date, float],
) -> Optional[dict[str, Any]]:
    """Flag one purpose on one day against its own baseline, or return None."""
    today = tokens.get(day, 0.0)
    usual = usual_value(tokens, day)
    if usual is None:
        return None
    if max(today, usual) < MIN_TOKENS_TO_FLAG:
        return None
    samples = [tokens[d] for d in _same_weekdays(day) if d in tokens]
    ratio = today / usual if usual else math.inf
    distance = (today - usual) / (_spread(samples) + 0.05 * usual + 1)
    usual_work = usual_value(work, day)
    usual_cache = usual_value(cache, day)
    work_today = work.get(day, 0.0)
    kind = None
    if ratio >= 2 and distance >= 3:
        kind = verdict(
            today, usual, work_today, usual_work, cache.get(day), usual_cache
        )
    elif ratio <= 0.3 and usual >= MIN_TOKENS_TO_FLAG:
        kind = "work_stopped"
    elif usual_work and work_today:
        per_ratio = (today / work_today) / (usual / usual_work)
        if per_ratio >= 1.5 and work_today / usual_work < 1.3 and distance >= 2:
            kind = "more_per_unit"
    if kind is None or kind in ("normal", "unknown"):
        return None
    return {
        "purpose": purpose,
        "kind": kind,
        "ratio": round(ratio, 2),
        "tokens": today,
        "usual": usual,
    }


def severity(flags: list[dict[str, Any]], extra_cost: float, day_cost: float) -> str:
    top = max((f["ratio"] for f in flags), default=1)
    kinds = {f["kind"] for f in flags}
    if "work_stopped" in kinds and len(flags) > 1:
        return "critical"
    if top >= 5 or (day_cost and extra_cost >= 0.5 * day_cost and top >= 2.5):
        return "critical"
    if top >= 3 or (day_cost and extra_cost >= 0.25 * day_cost):
        return "serious"
    return "warning"


def forecast_series(
    daily: list[float], growth: float = 0.0, bulk_tokens: float = 0.0
) -> dict[str, Any]:
    """Project 13 weeks from complete daily totals, oldest first.

    Weekly totals smooth out the weekday pattern. The trend is damped and
    clamped, so a single busy month can't project runaway growth. The range
    widens with distance and with how noisy the history has been.
    """
    weeks: list[float] = []
    for end in range(len(daily), 6, -7):
        weeks.insert(0, sum(daily[end - 7 : end]))
    weeks = weeks[-26:]
    if not weeks:
        return {"points": [], "low": [], "high": [], "trend": 0.0, "noise": 0.0}
    recent = weeks[-4:]
    earlier = weeks[-8:-4]
    level = sum(recent) / len(recent)
    trend = 0.0
    if earlier and sum(earlier) > 0 and level > 0:
        trend = (level / (sum(earlier) / len(earlier))) ** (1 / 4) - 1
    trend = max(-0.02, min(0.03, trend))
    noise = 0.05
    if len(weeks) >= 4 and level > 0:
        deviations = [abs(w - level) / level for w in weeks[-8:]]
        noise = max(0.05, min(0.4, median(deviations)))
    points: list[float] = []
    low: list[float] = []
    high: list[float] = []
    for k in range(1, FORECAST_WEEKS + 1):
        value = level * (1 + trend) ** k * (1 + growth * min(k / 4, 1))
        if k == 4:
            value += bulk_tokens
        points.append(value)
        low.append(value * max(0.0, 1 - noise - 0.012 * k))
        high.append(value * (1 + noise * 1.3 + 0.016 * k))
    return {"points": points, "low": low, "high": high, "trend": trend, "noise": noise}


def days_until_spent(
    budget_left: float, weekly_cost: Iterable[float]
) -> Optional[float]:
    """Days until ``budget_left`` runs out, or None if it lasts past the forecast."""
    left = budget_left
    days = 0.0
    for cost in weekly_cost:
        per_day = cost / 7
        if per_day <= 0:
            days += 7
            continue
        if left <= per_day * 7:
            return days + left / per_day
        left -= per_day * 7
        days += 7
    return None


def suggested_purchase(
    high_cost: float, budget_left: float, step: float = 50.0
) -> float:
    return max(0.0, math.ceil((high_cost - budget_left) / step) * step)
