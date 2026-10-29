"""Email the admins when an LLM provider rejects a key or a spend limit is hit.

Registered as a :mod:`lightrag.provider_rejection` listener by the API server.
Recipients are ``ALERT_EMAIL_TO`` (system admins) plus, when the rejected key
is the workspace's own (BYO) key, the workspace owner, because only they can
fix it on their Providers page.

The email names the workspace, slot, provider host/model and the provider's
error message. It never contains API keys or document content.

Configuration (``.env``):
    SMTP_HOST, SMTP_PORT (587), SMTP_SECURITY (starttls | ssl | none),
    SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM, SMTP_TIMEOUT (15),
    ALERT_EMAIL_TO (comma-separated), ALERT_EMAIL_REPEAT_HOURS (24).
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any, Awaitable, Callable, Optional

from lightrag.provider_rejection import ProviderRejectionEvent
from lightrag.utils import get_env_value, logger

# LLM role -> owner-facing Providers-page slot.
_ROLE_SLOTS = {
    "extract": "extraction",
    "keyword": "extraction",
    "query": "query",
    "vlm": "vision",
}

_KIND_TEXT = {
    "invalid_key": "rejected the API key",
    "budget_exhausted": "refused the request because the spend limit was reached",
}


@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    security: str
    username: str
    password: str
    sender: str
    timeout: int
    admin_recipients: tuple[str, ...]
    repeat_seconds: int

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.sender)

    @classmethod
    def from_env(cls) -> "SmtpSettings":
        security = get_env_value("SMTP_SECURITY", "starttls").strip().lower()
        if security not in ("starttls", "ssl", "none"):
            logger.warning(
                f"SMTP_SECURITY={security!r} is not starttls|ssl|none; using starttls"
            )
            security = "starttls"
        recipients = tuple(
            a.strip()
            for a in get_env_value("ALERT_EMAIL_TO", "").split(",")
            if a.strip()
        )
        return cls(
            host=get_env_value("SMTP_HOST", "").strip(),
            port=get_env_value("SMTP_PORT", 465 if security == "ssl" else 587, int),
            security=security,
            username=get_env_value("SMTP_USERNAME", "").strip(),
            password=get_env_value("SMTP_PASSWORD", ""),
            sender=get_env_value("SMTP_FROM", "").strip(),
            timeout=get_env_value("SMTP_TIMEOUT", 15, int),
            admin_recipients=recipients,
            repeat_seconds=max(1, get_env_value("ALERT_EMAIL_REPEAT_HOURS", 24, int))
            * 3600,
        )


def send_email_sync(settings: SmtpSettings, to: list[str], msg: EmailMessage) -> None:
    """Blocking SMTP send; run it in a thread."""
    context = ssl.create_default_context()
    if settings.security == "ssl":
        server: smtplib.SMTP = smtplib.SMTP_SSL(
            settings.host, settings.port, timeout=settings.timeout, context=context
        )
    else:
        server = smtplib.SMTP(settings.host, settings.port, timeout=settings.timeout)
    with server:
        if settings.security == "starttls":
            server.starttls(context=context)
        if settings.username:
            server.login(settings.username, settings.password)
        server.send_message(msg, to_addrs=to)


class ProviderRejectionNotifier:
    """Turns provider-rejection events into (rate-limited) admin emails."""

    def __init__(
        self,
        settings: SmtpSettings,
        *,
        is_byo_slot: Callable[[str, str], bool],
        owner_email: Callable[[str], Awaitable[Optional[str]]],
        send: Callable[[SmtpSettings, list[str], EmailMessage], None] = send_email_sync,
    ):
        self._settings = settings
        self._is_byo_slot = is_byo_slot
        self._owner_email = owner_email
        self._send = send
        # (workspace, slot, kind, status) -> last time an email went out.
        self._last_sent: dict[tuple[str, str, str, int], float] = {}
        self._warned_unconfigured = False

    async def __call__(self, event: ProviderRejectionEvent) -> None:
        slot = _ROLE_SLOTS.get(event.role, event.role)
        key = (event.workspace, slot, event.rejection.kind, event.rejection.status_code)
        now = time.time()
        last = self._last_sent.get(key)
        if last is not None and now - last < self._settings.repeat_seconds:
            return

        byo = bool(event.byo_key)
        if event.byo_key is None:
            try:
                byo = self._is_byo_slot(event.workspace, slot)
            except Exception as e:
                logger.warning(f"Provider alert: could not read provider config: {e}")

        recipients = list(self._settings.admin_recipients)
        if byo:
            try:
                owner = await self._owner_email(event.workspace)
            except Exception as e:
                owner = None
                logger.warning(f"Provider alert: owner lookup failed: {e}")
            if owner and owner.lower() not in {r.lower() for r in recipients}:
                recipients.append(owner)

        if not self._settings.enabled or not recipients:
            if not self._warned_unconfigured:
                logger.warning(
                    "Provider rejection alert not emailed: set SMTP_HOST, SMTP_FROM "
                    "and ALERT_EMAIL_TO to enable admin notifications"
                )
                self._warned_unconfigured = True
            return

        msg = build_alert_message(
            self._settings.sender,
            recipients,
            event,
            slot,
            byo,
            repeat_hours=self._settings.repeat_seconds // 3600,
        )
        # Record before sending so concurrent events do not double-send; a
        # failed send is retried on the next event after the repeat window.
        self._last_sent[key] = now
        try:
            await asyncio.to_thread(self._send, self._settings, recipients, msg)
        except Exception as e:
            logger.error(f"Provider rejection alert email failed: {e}")
            return
        logger.info(
            f"Provider rejection alert emailed for workspace '{event.workspace}' "
            f"({slot}) to {len(recipients)} recipient(s)"
        )


def build_alert_message(
    sender: str,
    recipients: list[str],
    event: ProviderRejectionEvent,
    slot: str,
    byo: bool,
    *,
    repeat_hours: int,
) -> EmailMessage:
    rejection = event.rejection
    what = _KIND_TEXT.get(rejection.kind, "rejected the request")
    when = datetime.fromtimestamp(event.occurred_at, timezone.utc)
    whose = (
        "the workspace's own key (set on its Providers page)"
        if byo
        else "the system key from the server configuration (.env)"
    )
    if byo:
        fix = (
            f"The workspace owner should open the Providers page, update the "
            f"{slot} key (or raise its spend limit with the provider), and save "
            f"with the test enabled. Ingestion resumes as soon as it is saved."
        )
    else:
        fix = (
            "An administrator should replace the LLM API key (or raise its spend "
            "limit with the provider) in the server .env and restart the "
            "LightRAG container."
        )
    effect = (
        f"Document ingestion in this workspace is paused, so no further calls "
        f"are made with this key. Waiting documents stay pending and are not "
        f"lost. LightRAG retries automatically every "
        f"{event.retry_after_minutes} minutes."
        if event.paused
        else "Requests that use this key fail until it is fixed."
    )

    msg = EmailMessage()
    msg["Subject"] = (
        f"[LightRAG] LLM provider {what} - workspace {event.workspace} ({slot})"
    )
    msg["From"] = formataddr(("LightRAG alerts", sender))
    msg["To"] = ", ".join(recipients)
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1] or None)
    msg.set_content(
        f"""The LLM provider {what}.

Workspace: {event.workspace}
Slot:      {slot} ({event.role} role)
Key:       {whose}
Provider:  {event.host or "unknown"}
Model:     {event.model or "unknown"}
Error:     HTTP {rejection.status_code}: {rejection.message}
Time:      {when:%Y-%m-%d %H:%M} UTC

{effect}

What to do: {fix}

No further email is sent about this problem in this workspace for the
next {repeat_hours} hour(s) unless the error changes.
"""
    )
    return msg


def create_provider_rejection_notifier(
    app_state: Any,
) -> ProviderRejectionNotifier:
    """Build the notifier from env + the app's provider store and registry."""
    settings = SmtpSettings.from_env()

    def is_byo_slot(workspace: str, slot: str) -> bool:
        store = getattr(app_state, "workspace_provider_store", None)
        if store is None or slot not in ("extraction", "query", "vision"):
            return False
        config = store.get(workspace)
        return bool(config and config.slot(slot).is_active())

    async def owner_email(workspace: str) -> Optional[str]:
        registry = getattr(app_state, "workspace_registry", None)
        if registry is None:
            return None
        owner = await registry.get_by_workspace(workspace)
        return owner.email if owner else None

    if settings.enabled and settings.admin_recipients:
        logger.info(
            f"Provider rejection alerts: emailing {len(settings.admin_recipients)} "
            f"admin(s) via {settings.host}:{settings.port}"
        )
    return ProviderRejectionNotifier(
        settings, is_byo_slot=is_byo_slot, owner_email=owner_email
    )
