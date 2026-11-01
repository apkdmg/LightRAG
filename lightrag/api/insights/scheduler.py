"""Background work: retention, anomaly and budget alerts, the weekly digest.

A lease in the store makes each job run on one worker only. Alerts go through
the same SMTP settings as provider rejection alerts. They never contain document
names, prompts or answers.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Any, Callable, Optional

from lightrag.api.provider_alerts import SmtpSettings, send_email_sync
from lightrag.usage import emit_event
from lightrag.utils import get_env_value, logger

from .brief import PURPOSE_NAMES
from .service import InsightsService

TICK_SECONDS = 300
ALERTED_KEY = "alerted_anomalies"
BUDGET_ALERT_KEY = "budget_alert"
DIGEST_KEY = "digest_week"
MODELS_KEY = "system_models"
VERSION_KEY = "system_version"
BUDGET_WARNING_DAYS = (30, 14, 7, 3)
ALERT_SEVERITIES = ("critical", "serious")
DIGEST_WEEKDAY = 0  # Monday
DIGEST_HOUR = 8

KIND_TEXT = {
    "more_work": "more work than usual",
    "more_per_unit": "more tokens for each unit of work",
    "cache_misses": "the LLM cache stopped helping",
    "work_stopped": "work stopped",
    "provider_rejected": "the provider rejected a key or a spend limit was reached",
    "silent": "documents are waiting but Extraction has gone quiet",
    "new_model": "a new model appeared",
}


def _names(purposes: list[dict[str, Any]]) -> str:
    return ", ".join(PURPOSE_NAMES.get(p["id"], p["id"]) for p in purposes)


class InsightsScheduler:
    def __init__(
        self,
        service: InsightsService,
        *,
        settings: Optional[SmtpSettings] = None,
        send: Callable[..., None] = send_email_sync,
    ):
        self.service = service
        self.store = service.store
        self.smtp = settings or SmtpSettings.from_env()
        self._send = send
        self._link = get_env_value("INSIGHTS_PUBLIC_URL", "").rstrip("/")
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as e:
                logger.warning(f"Insights background job failed: {e}")
            await asyncio.sleep(TICK_SECONDS)

    async def tick(self) -> None:
        if await self.store.try_lease("retention", 23 * 3600):
            await self.store.purge()
        if await self.store.try_lease("alerts", TICK_SECONDS - 10):
            await self.send_anomaly_alerts()
            await self.send_budget_warning()
        now = self.service.now()
        if now.weekday() == DIGEST_WEEKDAY and now.hour >= DIGEST_HOUR:
            await self.send_digest()

    def _ready(self) -> bool:
        return bool(self.smtp.enabled and self.smtp.admin_recipients)

    async def _email(self, subject: str, body: str) -> bool:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.smtp.sender
        message["To"] = ", ".join(self.smtp.admin_recipients)
        message["Message-ID"] = make_msgid(
            domain=self.smtp.sender.split("@")[-1] or None
        )
        if self._link:
            body += (
                f"\n\nOpen Insights: {self._link}/webui/#/?insights=1&itab=anomalies"
            )
        message.set_content(body)
        try:
            await asyncio.to_thread(
                self._send, self.smtp, list(self.smtp.admin_recipients), message
            )
        except Exception as e:
            logger.error(f"Insights email failed: {e}")
            return False
        return True

    async def send_anomaly_alerts(self) -> list[str]:
        """One email for everything new, so a single cause can't flood the inbox."""
        if not self._ready():
            return []
        items = (await self.service.anomalies())["items"]
        sent_before = set(await self.store.kv_get(ALERTED_KEY, []) or [])
        cutoff = (self.service.today() - timedelta(days=3)).isoformat()
        fresh = [
            a
            for a in items
            if a["state"] == "open"
            and a["severity"] in ALERT_SEVERITIES
            and a["id"] not in sent_before
            and a["day"] >= cutoff
        ]
        if not fresh:
            return []
        lines = [
            f"- {a['day']} [{a['severity']}] {_names(a['purposes'])}: "
            f"{KIND_TEXT.get(a['kind'], a['kind'])}"
            + (f" (workspace {a['workspace']})" if a["workspace"] else "")
            + (f", about ${a['extra_cost']:.2f} extra" if a.get("extra_cost") else "")
            for a in fresh
        ]
        word = "anomaly" if len(fresh) == 1 else "anomalies"
        ok = await self._email(
            f"[LightRAG] {len(fresh)} token usage {word} to review",
            "Token usage looks unusual:\n\n" + "\n".join(lines),
        )
        if not ok:
            return []
        known = sent_before | {a["id"] for a in fresh}
        await self.store.kv_set(ALERTED_KEY, sorted(known)[-500:])
        return [a["id"] for a in fresh]

    async def send_budget_warning(self) -> Optional[int]:
        """Warn at each threshold once. The email gives both dates."""
        if not self._ready():
            return None
        budget = (await self.service.forecast()).get("budget") or {}
        days = budget.get("days_left")
        if not budget.get("set") or days is None:
            return None
        reached = min((d for d in BUDGET_WARNING_DAYS if days <= d), default=None)
        if reached is None:
            return None
        state = await self.store.kv_get(BUDGET_ALERT_KEY, {}) or {}
        same_budget = (
            state.get("start") == budget["start"]
            and state.get("amount") == budget["amount"]
        )
        if same_budget and state.get("threshold", 10**6) <= reached:
            return None
        ok = await self._email(
            f"[LightRAG] Token budget runs out around {budget['runs_out']}",
            f"At the current pace the budget runs out around {budget['runs_out']}, "
            f"in about {days} days. That counts usage on workspace (BYO) keys.\n\n"
            f"Without BYO usage it would last until "
            f"{budget.get('runs_out_without_byo') or 'after the forecast period'}. "
            "Your Requesty spend limit only sees your own key, so that date is the "
            "better guide to when your key could stop.\n\n"
            f"Budget: ${budget['amount']:,.2f}. Left: ${budget['left']:,.2f}.",
        )
        if not ok:
            return None
        await self.store.kv_set(
            BUDGET_ALERT_KEY,
            {
                "start": budget["start"],
                "amount": budget["amount"],
                "threshold": reached,
            },
        )
        return reached

    async def send_digest(self) -> bool:
        if not self._ready():
            return False
        today = self.service.today()
        week = f"{today.isocalendar()[0]}-W{today.isocalendar()[1]}"
        if await self.store.kv_get(DIGEST_KEY) == week:
            return False
        if not await self.store.try_lease(f"digest:{week}", 3600):
            return False
        ok = await self._email(
            f"[LightRAG] Weekly token usage, week {today.isocalendar()[1]}",
            await self.digest_text(today),
        )
        if ok:
            await self.store.kv_set(DIGEST_KEY, week)
        return ok

    async def digest_text(self, today: date) -> str:
        end = today - timedelta(days=1)
        start = end - timedelta(days=6)
        now_rows = {
            r["purpose"]: r for r in await self.service._rows(start, end, ["purpose"])
        }
        before_rows = {
            r["purpose"]: r
            for r in await self.service._rows(
                start - timedelta(days=7), start - timedelta(days=1), ["purpose"]
            )
        }
        lines = [f"Token usage, {start.isoformat()} to {end.isoformat()}", ""]
        for purpose, name in PURPOSE_NAMES.items():
            row = now_rows.get(purpose, {})
            old = before_rows.get(purpose, {}).get("tokens", 0)
            tokens = row.get("tokens", 0)
            change = f"{(tokens - old) / old * 100:+.0f}%" if old else "new"
            lines.append(
                f"{name:<12}{tokens / 1e6:>9.1f}M tokens  ${row.get('cost', 0):>9,.2f}  {change}"
            )
        open_items = [
            a for a in (await self.service.anomalies())["items"] if a["state"] == "open"
        ]
        lines += ["", f"Open anomalies: {len(open_items)}"]
        lines += [
            f"- {a['day']} [{a['severity']}] {_names(a['purposes'])}: "
            f"{KIND_TEXT.get(a['kind'], a['kind'])}"
            for a in open_items[:10]
        ]
        budget = (await self.service.forecast()).get("budget") or {}
        if budget.get("set"):
            lines += [
                "",
                f"Budget runs out around {budget.get('runs_out') or 'later than the forecast'}.",
            ]
        return "\n".join(lines)


async def note_startup(service: InsightsService, version: str) -> None:
    """Log system model changes and deployments, which only show at a restart."""
    store = service.store
    if not await store.try_lease("startup", 60):
        return
    known = await store.kv_get(MODELS_KEY)
    current = service._current_models()
    if known and current:
        for purpose, model in current.items():
            old = known.get(purpose)
            if old and model and old != model:
                emit_event("model_change", purpose=purpose, old=old, new=model)
    if current:
        await store.kv_set(MODELS_KEY, current)
    previous = await store.kv_get(VERSION_KEY)
    if previous and previous != version:
        emit_event("deployment", version=version)
    await store.kv_set(VERSION_KEY, version)
