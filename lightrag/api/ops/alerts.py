"""Alerts of the Operations console: rules, the leased evaluator and its email.

One worker per host (the holder of the renewable lease ``ops:alerts``, 120 s)
runs :class:`AlertEvaluator` once a minute. Each cycle:

1. **Evaluates every attention rule once for the host** and stores the result
   in KV ``attention``. This evaluator is the snapshot's only writer: the
   Overview of every worker reads it, and evaluates the rules itself only
   when the snapshot is missing or older than three cycles.
2. **Drives** ``LIGHTRAG_OPS_ALERTS`` (when ``OPS_ALERTS_ENABLED``, default
   on): an item fires a row, or touches the open row of its fingerprint
   (insert-or-touch; the partial unique index on ``fingerprint WHERE
   resolved_at IS NULL`` makes a second open row impossible, also when two
   evaluators race). A fingerprint missing from two consecutive evaluations
   resolves its row (a flapping guard: one missed evaluation never resolves
   and re-fires an incident); a fingerprint seen again later fires a new
   row. A rule's rows do not resolve while a section that rule reads could
   not be read, and the heartbeat and orphan rules neither fire nor resolve
   for 10 min after a deploy (a ``deployment`` event, or every live worker
   newly started together, at most once an hour so a crash loop is not
   hidden). An ack is recorded in ``acked_by``/``acked_at`` and never
   changes the state.
3. **Emails** (only when ``OPS_ALERT_EMAIL=true``; off by default, so the
   first week is record-only): only critical and serious alerts, never the
   in-app-only rules (``provider_paused`` is emailed by
   :mod:`lightrag.api.provider_alerts` already), not while the rule is muted
   or maintenance mode is on, and not for an acked alert. Every email is
   claimed first with a conditional UPDATE on ``last_emailed_at``, so with N
   workers (or a takeover after a lease expiry) each email goes out once:
   when an incident opens, again every ``ALERT_EMAIL_REPEAT_HOURS`` while it
   stays open, and once when it resolves (only if it was emailed when open).
   The alerts of one cycle go out as one email. Bodies carry rule, severity,
   counts, since, the runbook step and a link to the alert; never log text,
   titles, workspace IDs, owners, file names or secrets. A failed send undoes
   its claims, is counted in KV ``alert_email`` (the console's "alert email
   is not being delivered" banner) and is retried with backoff; a send that
   runs out of time keeps its claims (it may still deliver) and blocks new
   sends until its thread is done. The banner also clears once nothing is
   left to retry and the SMTP check has passed since the failure. Flapping:
   a resolve email waits 10 min, a re-fire within that time continues the
   emailed incident without a new email, and a re-fire after a resolve
   email waits 30 min before it is emailed.
4. **Retention**, hourly: rows resolved more than 180 days ago.

Recipients are ``OPS_ALERT_EMAIL_TO``, falling back to ``ALERT_EMAIL_TO``
(D1), through the SMTP settings of the provider alerts.

Mutes (KV ``alert_mutes``, per rule, at most 7 days), maintenance mode (KV
``maintenance``) and threshold edits (KV ``thresholds``) are admin actions:
live admin, CSRF checks, a reason, an intent audit row before the change (no
row, no change) and a result row after, 10 per minute per admin.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any, Callable, Optional
from urllib.parse import urlencode

from lightrag.api.ops.audit import actor_of, write_audit
from lightrag.api.ops.redact import redact
from lightrag.api.ops.service import (
    ATTENTION_KEY,
    DEFAULT_THRESHOLDS,
    RUNBOOK,
    OpsDataUnavailable,
    OpsService,
)

logger = logging.getLogger("lightrag.api.ops")

ALERTS_FLAG = "OPS_ALERTS_ENABLED"
EMAIL_FLAG = "OPS_ALERT_EMAIL"
EMAIL_TO_ENV = "OPS_ALERT_EMAIL_TO"
PUBLIC_URL_ENV = "OPS_PUBLIC_URL"
RUNBOOK_BASE_ENV = "OPS_RUNBOOK_BASE_URL"

ALERTS_LEASE = "ops:alerts"
ALERTS_LEASE_TTL_S = 120.0
ALERTS_INTERVAL_S = 60.0
# Rule inputs are cached, 3 s-bounded reads; one email send is bounded by the
# SMTP timeout (EMAIL_SEND_TIMEOUT_S at most).
ALERTS_CYCLE_TIMEOUT_S = 100.0

MUTES_KEY = "alert_mutes"
MAINTENANCE_KEY = "maintenance"
EMAIL_STATE_KEY = "alert_email"
MISSING_KEY = "alert_missing"

EMAIL_SEVERITIES = ("critical", "serious")
SEVERITIES = ("warning", "serious", "critical")
# A fingerprint absent for this many consecutive evaluations resolves.
RESOLVE_AFTER_CYCLES = 2
OPEN_LIMIT = 500
RESOLVED_EMAIL_LOOKBACK = 300
RESOLVED_EMAIL_MAX_AGE_S = 86400.0
# Flapping guard for email. A resolve email waits until the incident has
# stayed resolved this long; a fingerprint that fires again within it is the
# same incident to the people who got the opening email, so the new row is
# marked notified without an email and the old row's resolve email is
# dropped. After a resolve email, a re-fire of the fingerprint is emailed
# only once FLAP_COOLDOWN_S have passed since that email.
RESOLVE_EMAIL_QUIET_S = 600.0
FLAP_COOLDOWN_S = 1800.0
FLAP_HISTORY = 20
# History lookups per cycle (one indexed query each).
FLAP_LOOKUPS_MAX = 50
MAX_ALERTS_PER_EMAIL = 20
EMAIL_SEND_TIMEOUT_S = 60.0
EMAIL_SEND_MIN_TIMEOUT_S = 10.0
EMAIL_BACKOFF_BASE_S = 60.0
EMAIL_BACKOFF_MAX_S = 1800.0
RETENTION_DAYS = 180
CLEANUP_EVERY_S = 3600.0
MUTE_MAX_MINUTES = 7 * 24 * 60
MAINTENANCE_MAX_MINUTES = 7 * 24 * 60
FINGERPRINT_MAX = 128
REASON_MIN = 3
REASON_MAX = 512
# Rules emailed elsewhere, or that only matter in the console.
IN_APP_ONLY_RULES = frozenset({"provider_paused"})
# Heartbeat and orphan rules: held (neither fired nor resolved) for the
# deploy window (FR-64).
DEPLOY_SUPPRESSED_RULES = frozenset(
    {"workers_below_expected", "run_orphaned", "worker_unresponsive", "manager_rtt"}
)
# The gathered sections (``failed_sections`` of the snapshot) each rule reads.
# While one of them could not be read, the rule's open alerts neither resolve
# nor count as missing: a failed read looks like a recovery. Sections no rule
# reads (``backlog_counts``, the optional dup/cancelled counts, and
# ``active``, the KPI tiles) hold nothing; rules not listed read no section.
_DEPLOY_SECTIONS = frozenset({"events", "workers"})
RULE_SECTIONS: dict[str, frozenset[str]] = {
    "oldest_pending": frozenset({"backlog"}),
    "stale_docs": frozenset({"stuck"}),
    "provider_paused": frozenset({"pauses"}),
    "run_orphaned": frozenset({"stuck_runs"}) | _DEPLOY_SECTIONS,
    "worker_unresponsive": frozenset({"stuck_runs"}) | _DEPLOY_SECTIONS,
    "run_hung": frozenset({"stuck_runs"}),
    "busy_without_run": frozenset({"stuck_runs"}),
    "reservation_without_owner": frozenset({"stuck_runs"}),
    "email_queue_bad": frozenset({"email_queue"}),
    "email_queue_stalled": frozenset({"email_queue"}),
    "email_ingest_silence": frozenset({"email_ingest"}),
    "failure_spike": frozenset({"failure_spike"}),
    # Whether a deploy is recent comes from the events and the workers.
    "workers_below_expected": _DEPLOY_SECTIONS,
    "manager_rtt": _DEPLOY_SECTIONS,
    "pg_pool_wait": frozenset({"workers"}),
    "log_sink_or_recorder_drops_5m": frozenset({"workers"}),
    "unauth_warning_lines_5m": frozenset({"workers"}),
}
# Scope values an email may carry as text; everything else must be a number
# or a boolean (no workspace lists, no owners, no messages).
# Rules where a lower reading is worse (free disk, days to certificate expiry).
BELOW_RULES = frozenset({"disk_free", "tls_cert_expiry"})
EMAIL_TEXT_KEYS = frozenset({"dependency", "error_class", "basis", "target"})
EMAIL_TEXT_MAX = 64
# Those strings must look like an identifier ("neo4j", "timeout",
# "lightrag-proxy:443"); anything with spaces or odd characters is left out.
_EMAIL_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def alerts_enabled() -> bool:
    """``OPS_ALERTS_ENABLED`` (default on): alert rows are recorded."""
    return _flag(ALERTS_FLAG, True)


def email_enabled() -> bool:
    """``OPS_ALERT_EMAIL`` (default **off**: record-only)."""
    return _flag(EMAIL_FLAG, False)


# -- rule catalogue -------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertRule:
    rule: str
    label: str
    condition: str
    severities: tuple[str, ...]
    thresholds: tuple[str, ...] = ()
    in_app_only: bool = False

    @property
    def runbook(self) -> str:
        return RUNBOOK[self.rule]

    @property
    def emailed(self) -> bool:
        """Whether any severity of the rule can be emailed."""
        return not self.in_app_only and any(
            s in EMAIL_SEVERITIES for s in self.severities
        )


def _rule(
    rule: str,
    label: str,
    condition: str,
    severities: tuple[str, ...],
    thresholds: tuple[str, ...] = (),
) -> AlertRule:
    return AlertRule(
        rule,
        label,
        condition,
        severities,
        thresholds,
        in_app_only=rule in IN_APP_ONLY_RULES,
    )


# §1.7 of docs/ADMIN_OPERATIONS_DASHBOARD.md. Thresholds without a key are
# fixed in health.py / probes.py and shown in ``condition``.
ALERT_RULES: dict[str, AlertRule] = {
    r.rule: r
    for r in (
        _rule(
            "workers_below_expected",
            "Workers below expected",
            "Fewer workers report than expected for 2 evaluations (critical "
            "when none); held for 10 min after a deploy.",
            ("warning", "critical"),
        ),
        _rule(
            "manager_rtt",
            "Shared state slow",
            "Multi-worker only: shared-state round trip over 200 ms for 2 "
            "evaluations (critical when it does not answer).",
            ("warning", "critical"),
        ),
        _rule(
            "pg_probe",
            "Postgres probe",
            "Health check over 100 ms twice in a row (critical when it fails or "
            "takes over 1 s).",
            ("warning", "critical"),
        ),
        _rule(
            "pg_pool_wait",
            "Postgres pool wait",
            "A worker waits over 500 ms for a pooled connection for 2 "
            "evaluations (serious over 5 s or none at all).",
            ("warning", "serious"),
        ),
        _rule(
            "pg_connections",
            "Postgres connections",
            "Connections at 70% of max_connections for 2 readings (serious at 90%).",
            ("warning", "serious"),
        ),
        _rule(
            "pg_blocked",
            "Postgres blocked query",
            "A query blocked over 30 s (serious over 120 s).",
            ("warning", "serious"),
        ),
        _rule(
            "dependency_down",
            "Dependency down",
            "Slow over 500 ms for 3 probes in a row; critical when down for 3 "
            "probes in a row.",
            ("warning", "critical"),
        ),
        _rule(
            "vllm_waiting",
            "vLLM queue",
            "Over 20 requests waiting at vLLM for 2 min (serious over 100).",
            ("warning", "serious"),
        ),
        _rule(
            "llm_error_rate_15m",
            "LLM error rate",
            "LLM error rate over 15 min, with at least the minimum number of calls.",
            ("warning", "serious"),
            ("llm_error_rate", "llm_error_rate_serious", "llm_min_calls"),
        ),
        _rule(
            "llm_p95_15m",
            "LLM latency",
            "LLM p95 over 15 min against the 7-day same-hour baseline, with at "
            "least the minimum number of calls.",
            ("warning", "serious"),
            (
                "llm_p95_ratio_warn",
                "llm_p95_ratio_serious",
                "llm_p95_serious_ms",
                "llm_min_calls",
            ),
        ),
        _rule(
            "provider_paused",
            "Paused by the LLM provider",
            "A workspace's ingestion is paused by a provider rejection. Shown "
            "here only: the provider alert emails it already.",
            ("critical",),
        ),
        _rule(
            "embed_rerank_error_rate_15m",
            "Embed/rerank error rate",
            "Embedding or rerank error rate over 15 min, with at least the "
            "minimum number of calls.",
            ("warning", "serious"),
            ("dep_error_warn", "dep_error_serious", "dep_error_min_calls"),
        ),
        _rule(
            "run_orphaned",
            "Run orphaned",
            "Multi-worker only: a run's worker is confirmed dead and its "
            "workspace stays busy; held for 10 min after a deploy.",
            ("critical",),
            ("orphan_seconds",),
        ),
        _rule(
            "run_hung",
            "Run without progress",
            "A run made no progress (paused runs excluded).",
            ("warning", "serious"),
            ("hung_minutes", "hung_serious_minutes"),
        ),
        _rule(
            "worker_unresponsive",
            "Worker unresponsive",
            "A worker holding a run stopped sending heartbeats but its process "
            "still exists; held for 10 min after a deploy.",
            ("warning",),
        ),
        _rule(
            "busy_without_run",
            "Busy without a run",
            "A workspace is marked busy with no run owning it.",
            ("warning",),
            ("busy_without_run_minutes",),
        ),
        _rule(
            "reservation_without_owner",
            "Reservation without owner",
            "Multi-worker only: a scan or upload reservation outlived its owner.",
            ("serious",),
            ("reservation_minutes",),
        ),
        _rule(
            "stale_docs",
            "Documents stuck in progress",
            "A document has stayed in a processing status.",
            ("warning", "serious"),
            ("stale_doc_minutes", "stale_doc_serious_minutes"),
        ),
        _rule(
            "oldest_pending",
            "Oldest pending document",
            "A document has been pending.",
            ("warning", "serious"),
            ("oldest_pending_minutes", "oldest_pending_serious_minutes"),
        ),
        _rule(
            "failure_spike",
            "Failure spike",
            "Real failures in the last hour (cancellations excluded) above the "
            "factor times the 7-day hourly median and the minimum; serious "
            "above the serious count or share of processed documents.",
            ("warning", "serious"),
            (
                "failure_spike_factor",
                "failure_spike_min",
                "failure_spike_serious",
                "failure_spike_serious_ratio",
            ),
        ),
        _rule(
            "email_queue_stalled",
            "Email re-sync queue stalled",
            "Email re-syncs pending with no drain seen.",
            ("warning", "serious"),
            ("email_waiting_minutes", "email_waiting_serious_minutes"),
        ),
        _rule(
            "email_queue_bad",
            "Unreadable email re-syncs",
            "A queued email re-sync could not be read.",
            ("warning",),
        ),
        _rule(
            "email_ingest_silence",
            "No email ingested",
            "No email ingest call arrived (the connector runs about every 70 min).",
            ("warning", "serious"),
            ("email_silence_minutes", "email_silence_serious_minutes"),
        ),
        _rule(
            "http_5xx_rate_5m",
            "HTTP 5xx rate",
            "Share of 5xx answers over 5 min, with at least the minimum number "
            "of requests.",
            ("warning", "serious"),
            ("http_5xx_warn", "http_5xx_serious", "http_5xx_min_requests"),
        ),
        _rule(
            "signin_failures_15m",
            "Sign-in failures",
            "SSO callback errors or lockouts over 15 min.",
            ("warning", "serious"),
            ("signin_sso_warn", "signin_lockout_warn", "signin_serious"),
        ),
        _rule(
            "tls_cert_expiry",
            "TLS certificate expiry",
            "A certificate expires in under 21 days (critical under 7); "
            "untrusted ones only with OPS_TLS_REQUIRE_TRUSTED=true.",
            ("warning", "critical"),
        ),
        _rule(
            "disk_free",
            "Disk space",
            "Under 15% free on the working or log volume (critical under 5%).",
            ("warning", "critical"),
        ),
        _rule(
            "log_sink_or_recorder_drops_5m",
            "Dropped records",
            "Usage records or log lines dropped in 5 min (serious over 1000, or "
            "no write for over a minute).",
            ("warning", "serious"),
        ),
        _rule(
            "unauth_warning_lines_5m",
            "Warnings from unauthenticated requests",
            "Over 50 WARNING or ERROR lines in 5 min logged while serving "
            "requests that presented no valid credential (open paths and "
            "refused callers). Counted by the structured log sink "
            "(OPS_LOG_SINK) only.",
            ("warning",),
        ),
        _rule(
            "smtp_probe",
            "SMTP check",
            "The SMTP check failed twice in a row: alert email cannot be delivered.",
            ("warning",),
        ),
    )
}

# Editable thresholds: (minimum, maximum, unit). Values outside are refused.
THRESHOLD_BOUNDS: dict[str, tuple[float, float, str]] = {
    "stale_doc_minutes": (5, 10080, "min"),
    "stale_doc_serious_minutes": (5, 10080, "min"),
    "hung_minutes": (2, 1440, "min"),
    "hung_serious_minutes": (2, 1440, "min"),
    "orphan_seconds": (60, 3600, "s"),
    "reservation_minutes": (1, 1440, "min"),
    "busy_without_run_minutes": (1, 1440, "min"),
    "oldest_pending_minutes": (5, 10080, "min"),
    "oldest_pending_serious_minutes": (5, 10080, "min"),
    "email_waiting_minutes": (5, 10080, "min"),
    "email_waiting_serious_minutes": (5, 10080, "min"),
    "email_silence_minutes": (30, 10080, "min"),
    "email_silence_serious_minutes": (30, 10080, "min"),
    "llm_error_rate": (0.001, 1.0, "ratio"),
    "llm_error_rate_serious": (0.001, 1.0, "ratio"),
    "llm_min_calls": (1, 100000, "calls"),
    "llm_p95_ratio_warn": (1.1, 100.0, "x"),
    "llm_p95_ratio_serious": (1.1, 100.0, "x"),
    "llm_p95_serious_ms": (1000, 600000, "ms"),
    "http_5xx_warn": (0.001, 1.0, "ratio"),
    "http_5xx_serious": (0.001, 1.0, "ratio"),
    "http_5xx_min_requests": (1, 1000000, "requests"),
    "dep_error_warn": (0.001, 1.0, "ratio"),
    "dep_error_serious": (0.001, 1.0, "ratio"),
    "dep_error_min_calls": (1, 1000000, "calls"),
    "signin_sso_warn": (1, 100000, "count"),
    "signin_lockout_warn": (1, 100000, "count"),
    "signin_serious": (1, 100000, "count"),
    "failure_spike_factor": (1.1, 100.0, "x"),
    "failure_spike_min": (1, 100000, "docs"),
    "failure_spike_serious": (1, 1000000, "docs"),
    "failure_spike_serious_ratio": (0.01, 1.0, "ratio"),
}
# (warning, serious): the serious threshold may not be below the warning one.
THRESHOLD_PAIRS = (
    ("stale_doc_minutes", "stale_doc_serious_minutes"),
    ("hung_minutes", "hung_serious_minutes"),
    ("oldest_pending_minutes", "oldest_pending_serious_minutes"),
    ("email_waiting_minutes", "email_waiting_serious_minutes"),
    ("email_silence_minutes", "email_silence_serious_minutes"),
    ("llm_error_rate", "llm_error_rate_serious"),
    ("llm_p95_ratio_warn", "llm_p95_ratio_serious"),
    ("http_5xx_warn", "http_5xx_serious"),
    ("dep_error_warn", "dep_error_serious"),
    ("failure_spike_min", "failure_spike_serious"),
)


def runbook_url(rule: str) -> Optional[str]:
    """``OPS_RUNBOOK_BASE_URL`` for a rule: ``{rule}`` is replaced when the
    base has it, else the rule is appended as an anchor."""
    base = (os.getenv(RUNBOOK_BASE_ENV) or "").strip()
    if not base or not base.startswith(("http://", "https://")):
        return None
    if "{rule}" in base:
        return base.replace("{rule}", rule)
    return f"{base.rstrip('#')}#{rule}"


def public_url() -> str:
    """This server's public address for links in emails (may be empty)."""
    value = os.getenv(PUBLIC_URL_ENV) or os.getenv("INSIGHTS_PUBLIC_URL") or ""
    return value.strip().rstrip("/")


def alert_hash(alert_id: Any) -> str:
    """The console link of one alert (FR-52: emails link only to ``oal``)."""
    return "#/?" + urlencode(
        {"ops": "1", "otab": "activity", "osub": "alerts", "oal": str(alert_id)}
    )


# -- KV state ---------------------------------------------------------------------------------


def _iso(epoch: Any) -> Optional[str]:
    if not isinstance(epoch, (int, float)) or isinstance(epoch, bool) or epoch <= 0:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def _as_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        return value
    return None


def _epoch(value: Any) -> Optional[float]:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    return None


def active_mutes(raw: Any, now: float) -> dict[str, dict[str, Any]]:
    """Mutes that have not expired, per rule, as the console shows them."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for rule, value in raw.items():
        if not isinstance(value, dict):
            continue
        until = value.get("until")
        if isinstance(until, (int, float)) and until > now:
            out[str(rule)] = {
                "rule": str(rule),
                "until": _iso(until),
                "until_epoch": float(until),
                "reason": value.get("reason"),
                "actor": value.get("actor"),
                "by": value.get("actor"),
                "at": _iso(value.get("at")),
            }
    return out


def maintenance_view(raw: Any, now: float) -> Optional[dict[str, Any]]:
    """The maintenance window while it is active, else None."""
    if not isinstance(raw, dict):
        return None
    until = raw.get("until")
    if not isinstance(until, (int, float)) or until <= now:
        return None
    return {
        "active": True,
        "until": _iso(until),
        "until_epoch": float(until),
        "started_at": _iso(raw.get("started_at")),
        "at": _iso(raw.get("started_at")),
        "reason": raw.get("reason"),
        "actor": raw.get("actor"),
        "by": raw.get("actor"),
    }


async def _kv_cached(service: OpsService, key: str) -> Any:
    async def load():
        return await service.store.kv_get(key)

    return await service._cached(("alert_state", key), "alert_state", load)


async def maintenance_state(service: OpsService, now: float) -> Optional[dict]:
    if not service.available:
        return None
    try:
        return maintenance_view(await _kv_cached(service, MAINTENANCE_KEY), now)
    except OpsDataUnavailable:
        return None


async def mutes_state(service: OpsService, now: float) -> dict[str, dict]:
    if not service.available:
        return {}
    try:
        return active_mutes(await _kv_cached(service, MUTES_KEY), now)
    except OpsDataUnavailable:
        return {}


def smtp_settings() -> Any:
    from lightrag.api.provider_alerts import SmtpSettings

    return SmtpSettings.from_env()


def alert_recipients(settings: Any) -> list[str]:
    """``OPS_ALERT_EMAIL_TO``, falling back to ``ALERT_EMAIL_TO`` (D1)."""
    own = [a.strip() for a in (os.getenv(EMAIL_TO_ENV) or "").split(",") if a.strip()]
    if own:
        return own
    return list(getattr(settings, "admin_recipients", ()) or ())


def _smtp_failing(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return (
        record.get("status") == "down"
        and int(record.get("consecutive_failures") or 0) >= 2
    )


async def channel_state(
    service: OpsService, smtp_card: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Whether alert email can be delivered (FR-63).

    ``state``: ``ok``, ``record_only`` (``OPS_ALERT_EMAIL`` is off) or
    ``broken`` (email is on but SMTP is not configured, there are no
    recipients, the SMTP check fails, or the last send failed). ``reason``
    is set whatever the state, so record-only mode can say what would fail.
    """
    enabled = email_enabled()
    try:
        settings = smtp_settings()
        configured = bool(settings.enabled)
        recipients = alert_recipients(settings)
    except Exception:
        configured, recipients = False, []
    stored: Any = {}
    probe: Any = smtp_card
    if service.available:
        try:
            stored = await _kv_cached(service, EMAIL_STATE_KEY) or {}
            if probe is None:
                from lightrag.api.ops.probes import KV_PREFIX

                probe = await _kv_cached(service, KV_PREFIX + "smtp")
        except OpsDataUnavailable:
            stored = {}
    stored = stored if isinstance(stored, dict) else {}
    consecutive = int(stored.get("consecutive_failures") or 0)
    if not configured:
        reason = "smtp_not_configured"
    elif not recipients:
        reason = "no_recipients"
    elif consecutive:
        reason = "send_failed"
    elif _smtp_failing(probe):
        reason = "smtp_probe_failing"
    else:
        reason = None
    state = "record_only" if not enabled else ("broken" if reason else "ok")
    return {
        "state": state,
        # None in record-only mode: nothing is meant to be delivered yet.
        "ok": None if state == "record_only" else state == "ok",
        "enabled": enabled,
        "email_enabled": enabled,
        # Failed sends in a row, and since when they fail.
        "failures": consecutive,
        "since": _iso(stored.get("failing_since")) if consecutive else None,
        "reason": reason,
        "configured": configured,
        "recipients": len(recipients),
        "sent_total": int(stored.get("sent_total") or 0),
        "failures_total": int(stored.get("failures_total") or 0),
        "consecutive_failures": consecutive,
        "last_error_class": stored.get("last_error_class"),
        "last_failure_at": _iso(stored.get("last_failure_at")),
        "last_sent_at": _iso(stored.get("last_sent_at")),
        "retry_after": _iso(stored.get("retry_after"))
        if consecutive and (stored.get("retry_after") or 0) > service.clock()
        else None,
    }


def smtp_recovered(state: Any, probe: Any) -> bool:
    """The SMTP check passed after the last failed alert send."""
    if not isinstance(state, dict) or not isinstance(probe, dict):
        return False
    failed_at = state.get("last_failure_at")
    last_ok = probe.get("last_ok_epoch")
    return (
        probe.get("status") in ("ok", "warn")
        and isinstance(failed_at, (int, float))
        and isinstance(last_ok, (int, float))
        and last_ok > failed_at
    )


async def clear_send_failures(
    store: Any, service: OpsService, state: Optional[dict[str, Any]] = None
) -> bool:
    """End the "alert email is not being delivered" state of failed sends
    (the totals stay). True when there was one to clear."""
    if state is None:
        value = await store.kv_get(EMAIL_STATE_KEY)
        state = value if isinstance(value, dict) else {}
    if not int(state.get("consecutive_failures") or 0):
        return False
    try:
        await store.kv_put(
            EMAIL_STATE_KEY,
            {
                **state,
                "consecutive_failures": 0,
                "failing_since": None,
                "retry_after": None,
            },
        )
    except Exception as e:
        logger.warning(
            "Ops alert email state could not be stored (%s)", type(e).__name__
        )
        return False
    service.cache.forget(("alert_state", EMAIL_STATE_KEY))
    logger.info("Ops alert email: the SMTP check passed, delivery state cleared")
    return True


# -- rows and views ---------------------------------------------------------------------------


def fingerprint_of(item: dict[str, Any]) -> str:
    key = str(item.get("id") or item["rule"])
    if len(key) <= FINGERPRINT_MAX:
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return f"{str(item['rule'])[:64]}:{digest}"


def _row_of(item: dict[str, Any]) -> dict[str, Any]:
    """An attention item as an alert row (title redacted, fields capped)."""
    scope = dict(item.get("params") or {})
    if item.get("count") is not None:
        scope.setdefault("count", item["count"])
    if item.get("since"):
        scope["since"] = item["since"]
    return {
        "rule": str(item["rule"])[:64],
        "fingerprint": fingerprint_of(item),
        "severity": item["severity"],
        "title": redact(item.get("title") or item["rule"], 512)[:512],
        "scope": scope,
        "link": (item.get("link") or "")[:512] or None,
        "runbook": (item.get("runbook") or RUNBOOK.get(item["rule"]) or "")[:512]
        or None,
    }


def emailable(rule: str, severity: str) -> bool:
    if rule in IN_APP_ONLY_RULES or severity not in EMAIL_SEVERITIES:
        return False
    definition = ALERT_RULES.get(rule)
    return definition is None or definition.emailed


def email_status(
    row: dict[str, Any],
    mutes: dict[str, dict[str, Any]],
    *,
    email_on: bool,
    maintenance: bool = False,
    failing: bool = False,
) -> str:
    """How an alert reached people: ``sent``, ``in_app_only`` (never
    emailed), ``record_only`` (``OPS_ALERT_EMAIL`` off), ``muted``,
    ``maintenance``, ``suppressed`` (acked before any email, or resolved
    without one), ``failed``
    (sends are failing) or ``pending`` (claimed at the next evaluation)."""
    if row.get("last_emailed_at") is not None:
        return "sent"
    if not emailable(row["rule"], str(row.get("severity"))):
        return "in_app_only"
    if not email_on:
        return "record_only"
    if row.get("resolved_at") is not None:
        # Resolved without ever being emailed (muted, maintenance, acked, or
        # over before the next evaluation): no email will follow.
        return "suppressed"
    if row["rule"] in mutes:
        return "muted"
    if maintenance:
        return "maintenance"
    if row.get("acked_at") is not None:
        return "suppressed"
    if failing:
        return "failed"
    return "pending"


def alert_view(
    row: dict[str, Any],
    mutes: dict[str, dict[str, Any]],
    now: float,
    *,
    email_on: Optional[bool] = None,
    maintenance: bool = False,
    failing: bool = False,
) -> dict[str, Any]:
    """One ``LIGHTRAG_OPS_ALERTS`` row for the console."""
    definition = ALERT_RULES.get(row["rule"])
    first = _epoch(row.get("first_seen"))
    end = _epoch(row.get("resolved_at")) or now
    resolved = row.get("resolved_at") is not None
    mute = mutes.get(row["rule"])
    return {
        "id": int(row["id"]),
        "rule": row["rule"],
        "label": definition.label if definition else row["rule"],
        "fingerprint": row.get("fingerprint"),
        "severity": row.get("severity"),
        "state": "resolved" if resolved else "firing",
        "title": row.get("title"),
        "scope": row.get("scope") or {},
        "link": row.get("link"),
        "alert_link": alert_hash(row["id"]),
        "runbook": row.get("runbook") or RUNBOOK.get(row["rule"]),
        "runbook_url": runbook_url(row["rule"]),
        "first_seen": _as_iso(row.get("first_seen")),
        "last_seen": _as_iso(row.get("last_seen")),
        "resolved_at": _as_iso(row.get("resolved_at")),
        "duration_s": round(end - first, 1) if first is not None else None,
        "acked_by": row.get("acked_by"),
        "acked_at": _as_iso(row.get("acked_at")),
        "ack_note": row.get("ack_note"),
        "last_emailed_at": _as_iso(row.get("last_emailed_at")),
        "notified": row.get("last_emailed_at") is not None,
        "emailable": emailable(row["rule"], str(row.get("severity"))),
        "email": email_status(
            row,
            mutes,
            email_on=email_enabled() if email_on is None else email_on,
            maintenance=maintenance,
            failing=failing,
        ),
        "muted": mute,
        "muted_until": mute["until"] if mute else None,
    }


def rules_view(
    thresholds: dict[str, Any],
    stored: Any,
    mutes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every rule with its runbook, current thresholds and mute."""
    stored = stored if isinstance(stored, dict) else {}
    out = []
    for definition in ALERT_RULES.values():
        mute = mutes.get(definition.rule)
        out.append(
            {
                "rule": definition.rule,
                "label": definition.label,
                "condition": definition.condition,
                "severities": list(definition.severities),
                "emailed": definition.emailed,
                "email": definition.emailed,
                "direction": "below" if definition.rule in BELOW_RULES else "above",
                "in_app_only": definition.in_app_only,
                "runbook": definition.runbook,
                "runbook_url": runbook_url(definition.rule),
                "thresholds": [
                    {
                        "key": key,
                        "value": thresholds.get(key, DEFAULT_THRESHOLDS[key]),
                        "default": DEFAULT_THRESHOLDS[key],
                        "overridden": key in stored,
                        "min": THRESHOLD_BOUNDS[key][0],
                        "max": THRESHOLD_BOUNDS[key][1],
                        "unit": THRESHOLD_BOUNDS[key][2],
                        "integer": isinstance(DEFAULT_THRESHOLDS[key], int),
                    }
                    for key in definition.thresholds
                ],
                "muted": mute,
                "muted_until": mute["until"] if mute else None,
                "mute_reason": mute["reason"] if mute else None,
                "muted_by": mute["actor"] if mute else None,
            }
        )
    return out


class ThresholdError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def flatten_changes(thresholds: Any, rules: Any) -> dict[str, Any]:
    """Threshold changes given flat (``{key: value}``) and/or per rule
    (``{rule: {key: value}}``, keys of that rule only) as one flat map."""
    changes: dict[str, Any] = {}
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            raise ThresholdError("invalid_thresholds")
        changes.update(thresholds)
    if rules is not None:
        if not isinstance(rules, dict):
            raise ThresholdError("invalid_thresholds")
        for rule, values in rules.items():
            definition = ALERT_RULES.get(rule)
            if definition is None:
                raise ThresholdError("unknown_rule")
            if not isinstance(values, dict):
                raise ThresholdError("invalid_thresholds")
            for key, value in values.items():
                if key not in definition.thresholds:
                    raise ThresholdError("unknown_threshold")
                if key in changes and changes[key] != value:
                    raise ThresholdError("conflicting_thresholds")
                changes[key] = value
    return changes


def validate_thresholds(
    changes: Any, stored: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The new ``thresholds`` KV value and what changed.

    ``changes`` maps keys to a number (override) or null (back to the
    default). Unknown keys, non-numbers and values outside their bounds are
    refused, and so is a serious threshold below its warning one.
    """
    if not isinstance(changes, dict) or not changes:
        raise ThresholdError("invalid_thresholds")
    current = {
        k: v
        for k, v in (stored if isinstance(stored, dict) else {}).items()
        if k in THRESHOLD_BOUNDS
    }
    new = dict(current)
    diff: dict[str, dict[str, Any]] = {}
    for key, value in changes.items():
        if key not in THRESHOLD_BOUNDS:
            raise ThresholdError("unknown_threshold")
        default = DEFAULT_THRESHOLDS[key]
        if value is None:
            if key in new:
                diff[key] = {"old": new.pop(key), "new": None}
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ThresholdError("invalid_threshold")
        low, high, _ = THRESHOLD_BOUNDS[key]
        if not (low <= value <= high) or value != value:
            raise ThresholdError("threshold_out_of_bounds")
        if isinstance(default, int) and not float(value).is_integer():
            raise ThresholdError("invalid_threshold")
        value = type(default)(value)
        if new.get(key) != value:
            diff[key] = {"old": new.get(key), "new": value}
        new[key] = value
    for warn, serious in THRESHOLD_PAIRS:
        if new.get(warn, DEFAULT_THRESHOLDS[warn]) > new.get(
            serious, DEFAULT_THRESHOLDS[serious]
        ):
            raise ThresholdError("threshold_order")
    return new, diff


# -- email ----------------------------------------------------------------------------------------


def email_scope(scope: Any) -> list[tuple[str, str]]:
    """The scope values an email may carry: numbers, booleans and a few
    short fixed-vocabulary strings, each redacted. Never lists (workspace
    IDs), objects or free text."""
    out: list[tuple[str, str]] = []
    if not isinstance(scope, dict):
        return out
    for key in sorted(scope):
        value = scope[key]
        if key == "since":
            continue
        if isinstance(value, bool):
            out.append((key, "yes" if value else "no"))
        elif isinstance(value, (int, float)):
            out.append((key, f"{value:g}" if isinstance(value, float) else str(value)))
        elif (
            isinstance(value, str)
            and key in EMAIL_TEXT_KEYS
            and _EMAIL_TOKEN.match(value)
        ):
            text = redact(value, EMAIL_TEXT_MAX)
            if text:
                out.append((key, text))
    return out


_STATE_TEXT = {
    "opened": "Opened",
    "reminder": "Still open",
    "resolved": "Resolved",
}


@dataclass
class EmailEvent:
    row: dict[str, Any]
    kind: str  # opened | reminder | resolved
    claimed: datetime
    previous: Optional[datetime]


def build_message(
    sender: str, recipients: list[str], events: list[EmailEvent], *, link_base: str
) -> EmailMessage:
    """One email for one cycle's alerts. The subject holds rules and
    severities only; the body rule, severity, state, since, safe counts,
    the runbook step and a link to each alert (D9, FR-62)."""
    from lightrag.api.provider_alerts import ALERT_SUBJECT_TAG

    def head(event: EmailEvent) -> str:
        severity = str(event.row.get("severity") or "").upper()
        prefix = "RESOLVED " if event.kind == "resolved" else f"{severity} "
        return f"{prefix}{event.row['rule']}"

    if len(events) == 1:
        subject = f"{ALERT_SUBJECT_TAG} {head(events[0])}"
    else:
        shown = ", ".join(head(e) for e in events[:3])
        more = f" (+{len(events) - 3})" if len(events) > 3 else ""
        subject = f"{ALERT_SUBJECT_TAG} {len(events)} alerts: {shown}{more}"
    blocks = []
    for event in events:
        row = event.row
        rule = str(row["rule"])
        definition = ALERT_RULES.get(rule)
        since = _as_iso(row.get("first_seen")) or "-"
        lines = [
            f"{definition.label if definition else rule} ({rule})",
            f"Severity:   {row.get('severity')}",
            f"State:      {_STATE_TEXT.get(event.kind, event.kind)}",
            f"Since:      {since}",
        ]
        if event.kind == "resolved":
            lines.append(f"Resolved:   {_as_iso(row.get('resolved_at')) or '-'}")
        details = ", ".join(f"{k}={v}" for k, v in email_scope(row.get("scope")))
        if details:
            lines.append(f"Details:    {details}")
        runbook = RUNBOOK.get(rule)
        if runbook and event.kind != "resolved":
            lines.append(f"What to do: {runbook}")
        url = runbook_url(rule)
        if url and event.kind != "resolved":
            lines.append(f"Runbook:    {url}")
        target = (
            f"{link_base}/webui/{alert_hash(row['id'])}"
            if link_base
            else (f"Operations > Activity > Alerts, alert {row['id']}")
        )
        lines.append(f"Open:       {target}")
        blocks.append("\n".join(lines))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(("LightRAG alerts", sender))
    msg["To"] = ", ".join(recipients)
    msg["Message-ID"] = make_msgid(domain=sender.split("@")[-1] or None)
    msg.set_content(
        "\n\n".join(blocks)
        + "\n\nThis email carries no document names, workspace IDs or log text; "
        "open the alert in the Operations console for details.\n"
    )
    return msg


class SendInFlight(Exception):
    """The send budget ran out while the SMTP thread was still running."""


class Mailer:
    """Sends on a dedicated thread (never the default executor), per process.

    ``smtplib``'s timeout applies to each socket operation, so a send can
    outlast its budget and still deliver. A send that times out therefore
    stays *in flight*: its claims are kept (a duplicate email is worse than
    a late one), no new send starts until its thread is done, and its
    outcome is settled at a later cycle (:meth:`take_settled`).
    """

    def __init__(self) -> None:
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pid: Optional[int] = None
        self._inflight: Optional[tuple[concurrent.futures.Future, list[Any]]] = None

    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        pid = os.getpid()
        if self._executor is None or self._pid != pid:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ops-alert-mail"
            )
            self._pid = pid
            self._inflight = None
        return self._executor

    def _current(self) -> Optional[tuple[concurrent.futures.Future, list[Any]]]:
        if self._pid != os.getpid():
            # A forked child never owns its parent's send.
            self._inflight = None
        return self._inflight

    def busy(self) -> bool:
        """A send that timed out is still running."""
        inflight = self._current()
        return inflight is not None and not inflight[0].done()

    def take_settled(self) -> Optional[tuple[Optional[BaseException], list[Any]]]:
        """``(error, events)`` of a timed-out send that has finished since
        (``error`` None: it was delivered after all), once; else None."""
        inflight = self._current()
        if inflight is None or not inflight[0].done():
            return None
        self._inflight = None
        future, events = inflight
        if future.cancelled():
            return asyncio.CancelledError(), events
        return future.exception(), events

    async def send(
        self,
        send: Callable[[Any, list[str], EmailMessage], None],
        settings: Any,
        recipients: list[str],
        msg: EmailMessage,
        timeout_s: float,
        events: Optional[list[Any]] = None,
    ) -> None:
        future = self.executor().submit(send, settings, recipients, msg)
        self._inflight = (future, list(events or ()))
        waiter = asyncio.wrap_future(future)
        # Retrieve a late error, so asyncio never reports it as unhandled.
        waiter.add_done_callback(lambda f: f.cancelled() or f.exception())
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout_s)
        except asyncio.CancelledError:
            raise  # stays in flight; settled at a later cycle
        except BaseException as e:
            if not future.done():
                raise SendInFlight() from e
            # The thread is done (a socket timeout inside the send, or it
            # finished just as the budget ran out): its own outcome counts.
            self._inflight = None
            error = future.exception() if not future.cancelled() else e
            if error is not None:
                raise error from None
        self._inflight = None

    def close(self) -> None:
        executor, self._executor = self._executor, None
        self._inflight = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _send_default(settings: Any, recipients: list[str], msg: EmailMessage) -> None:
    from lightrag.api.provider_alerts import send_email_sync

    send_email_sync(settings, recipients, msg)


def _error_class(exc: BaseException) -> str:
    try:
        from lightrag.api.ops.probes import classify_error

        return classify_error(exc)
    except Exception:
        return "error"


# -- the evaluator ------------------------------------------------------------------------------


@dataclass
class CycleOutcome:
    open: list[dict[str, Any]] = field(default_factory=list)
    resolved: list[dict[str, Any]] = field(default_factory=list)
    fired: list[int] = field(default_factory=list)


class AlertEvaluator:
    """The leased attention snapshot, alert state machine and alert email."""

    def __init__(
        self,
        service: OpsService,
        *,
        process: Callable[[], dict[str, Any]],
        clock: Callable[[], float] = time.time,
        send: Callable[[Any, list[str], EmailMessage], None] = _send_default,
        settings: Callable[[], Any] = smtp_settings,
        mailer: Optional[Mailer] = None,
    ):
        self.service = service
        self.process = process
        self.clock = clock
        self.send = send
        self.settings = settings
        self.mailer = mailer or Mailer()
        self._last_cleanup = 0.0
        self.cycles = 0

    @property
    def store(self) -> Any:
        return self.service.store

    async def cycle(self, loop: Any) -> Optional[CycleOutcome]:
        """One evaluation while the lease is ours (``loop.still_held``)."""
        service = self.service
        now = self.clock()
        try:
            previous = await self.store.kv_get(ATTENTION_KEY)
        except Exception:
            previous = None
        try:
            process = self.process() or {}
        except Exception:
            process = {}
        snapshot = await service.compute_attention_snapshot(process, loop.holder)
        if not await loop.still_held():
            return None
        outcome: Optional[CycleOutcome] = None
        if alerts_enabled():
            try:
                outcome = await self.apply(snapshot, previous)
            except Exception as e:
                # The snapshot is still published: the Overview never depends
                # on the alert table.
                logger.warning("Ops alert state update failed (%s)", type(e).__name__)
        await self.store.kv_put(ATTENTION_KEY, snapshot)
        service.cache.forget(("attention",))
        self.cycles += 1
        if outcome is not None:
            try:
                await self.notify(outcome)
            except Exception as e:
                logger.warning("Ops alert email step failed (%s)", type(e).__name__)
        await self._maybe_cleanup(now)
        return outcome

    async def apply(self, snapshot: dict[str, Any], previous: Any) -> CycleOutcome:
        """Fire or touch a row per item, resolve what stayed away, and mark
        the snapshot's items with their alert rows."""
        store = self.store
        items = snapshot.get("items") or []
        rows = {}
        for item in items:
            row = _row_of(item)
            rows[row["fingerprint"]] = row
        open_before = await store.alerts_open(OPEN_LIMIT)
        deploy = bool(snapshot.get("deploy_recent"))
        if deploy:
            # The deploy window holds the heartbeat and orphan rules: an open
            # row is still touched, but no new one fires (FR-64).
            open_fps = {r["fingerprint"] for r in open_before}
            for fp in [
                fp
                for fp, row in rows.items()
                if row["rule"] in DEPLOY_SUPPRESSED_RULES and fp not in open_fps
            ]:
                del rows[fp]
        upserted = await store.alerts_upsert(list(rows.values()))
        fired = [r["id"] for r in upserted if r.get("inserted")]
        prior_missing = (
            previous.get(MISSING_KEY) if isinstance(previous, dict) else None
        )
        prior_missing = prior_missing if isinstance(prior_missing, dict) else {}
        # A section a rule reads that could not be read looks like a
        # recovery: hold that rule's rows (only those).
        failed = frozenset(snapshot.get("failed_sections") or ())
        missing: dict[str, int] = {}
        resolve: list[str] = []
        for row in open_before:
            fp = row["fingerprint"]
            if fp in rows:
                continue
            count = int(prior_missing.get(fp) or 0)
            if (failed & RULE_SECTIONS.get(row["rule"], frozenset())) or (
                deploy and row["rule"] in DEPLOY_SUPPRESSED_RULES
            ):
                missing[fp] = count
                continue
            count += 1
            if count >= RESOLVE_AFTER_CYCLES:
                resolve.append(fp)
            else:
                missing[fp] = count
        resolved = await store.alerts_resolve(resolve) if resolve else []
        snapshot[MISSING_KEY] = missing
        open_now = await store.alerts_open(OPEN_LIMIT)
        recent = await store.alerts_page("resolved", None, OPEN_LIMIT)
        day_ago = self.clock() - 86400.0
        snapshot["alerts"] = {
            "open": len(open_now),
            "acked": sum(1 for r in open_now if r.get("acked_at") is not None),
            "resolved_24h": sum(
                1 for r in recent if (_epoch(r.get("resolved_at")) or 0) >= day_ago
            ),
        }
        mutes = active_mutes(await store.kv_get(MUTES_KEY), self.clock())
        by_fp = {r["fingerprint"]: r for r in open_now}
        for item in items:
            row = by_fp.get(fingerprint_of(item))
            if row is None:
                continue
            item["alert_id"] = int(row["id"])
            item["acked_by"] = row.get("acked_by")
            item["muted"] = item["rule"] in mutes
        if fired or resolved:
            logger.info(
                "Ops alerts: %d fired, %d resolved, %d open",
                len(fired),
                len(resolved),
                len(open_now),
            )
        return CycleOutcome(open=open_now, resolved=resolved, fired=fired)

    async def _email_state(self) -> dict[str, Any]:
        value = await self.store.kv_get(EMAIL_STATE_KEY)
        return value if isinstance(value, dict) else {}

    async def notify(self, outcome: CycleOutcome) -> int:
        """Claim and send this cycle's emails. Returns how many alerts went out.

        Every read comes before the first claim, and anything that fails
        between a claim and the send undoes the claims, so a row is never
        left marked as emailed for an email that was not sent.
        """
        if not email_enabled():
            return 0
        store = self.store
        now = self.clock()
        if self.mailer.busy():
            # A send that ran out of time may still deliver: nothing new is
            # claimed until its thread is done.
            return 0
        state = await self._settle_inflight(await self._email_state(), now)
        settings = self.settings()
        recipients = alert_recipients(settings)
        if not settings.enabled or not recipients:
            return 0
        if maintenance_view(await store.kv_get(MAINTENANCE_KEY), now) is not None:
            return 0
        retry_after = state.get("retry_after")
        if (
            int(state.get("consecutive_failures") or 0)
            and isinstance(retry_after, (int, float))
            and retry_after > now
        ):
            return 0
        mutes = active_mutes(await store.kv_get(MUTES_KEY), now)
        plans = await self._open_plans(outcome, mutes, now)
        resolved = await self._resolved_to_email(outcome, mutes, now)
        events: list[EmailEvent] = []
        try:
            for row, kind in plans:
                if len(events) >= MAX_ALERTS_PER_EMAIL:
                    break
                claimed = await store.alert_claim_email(
                    row["id"], "open", float(settings.repeat_seconds)
                )
                if claimed is None or kind == "carried":
                    # A carried claim marks a re-fire within the flapping
                    # window as notified; there is nothing to send for it.
                    continue
                kind = "reminder" if row.get("last_emailed_at") else "opened"
                events.append(
                    EmailEvent(row, kind, claimed, row.get("last_emailed_at"))
                )
            for row in resolved:
                if len(events) >= MAX_ALERTS_PER_EMAIL:
                    break
                claimed = await store.alert_claim_email(row["id"], "resolve", 0.0)
                if claimed is not None:
                    events.append(
                        EmailEvent(row, "resolved", claimed, row.get("last_emailed_at"))
                    )
            msg = (
                build_message(
                    settings.sender, recipients, events, link_base=public_url()
                )
                if events
                else None
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._unclaim(events))
            raise
        except BaseException:
            await self._unclaim(events)
            raise
        if msg is None:
            await self._recover_when_idle(state, now)
            return 0
        timeout = min(
            EMAIL_SEND_TIMEOUT_S,
            max(EMAIL_SEND_MIN_TIMEOUT_S, float(settings.timeout) * 3),
        )
        try:
            await self.mailer.send(
                self.send, settings, recipients, msg, timeout, events
            )
        except asyncio.CancelledError:
            # The thread may still deliver: keep the claims; the in-flight
            # send is settled at a later cycle.
            raise
        except SendInFlight as e:
            # Outcome unknown. The claims stay (never a duplicate email); it
            # counts as a failure until the thread reports a delivery.
            await self._record_failure(state, e, len(events), now, uncertain=True)
            return 0
        except Exception as e:
            await self._unclaim(events)
            await self._record_failure(state, e, len(events), now)
            return 0
        await self._record_success(state, len(events), now)
        logger.info(
            "Ops alert email sent for %d alert(s) to %d recipient(s)",
            len(events),
            len(recipients),
        )
        return len(events)

    async def _open_plans(
        self, outcome: CycleOutcome, mutes: dict[str, dict[str, Any]], now: float
    ) -> list[tuple[dict[str, Any], str]]:
        """Open rows to claim, with ``reminder`` (emailed before; the claim
        decides whether one is due), ``opened`` or ``carried``. A row whose
        re-fire waits out the flapping cooldown is left out."""
        plans: list[tuple[dict[str, Any], str]] = []
        lookups = 0
        for row in outcome.open:
            if not emailable(row["rule"], str(row.get("severity"))):
                continue
            if row["rule"] in mutes or row.get("acked_at") is not None:
                continue
            if row.get("last_emailed_at") is not None:
                plans.append((row, "reminder"))
                continue
            if lookups >= FLAP_LOOKUPS_MAX:
                continue
            lookups += 1
            kind = await self._refire_kind(row, now)
            if kind is not None:
                plans.append((row, kind))
        return plans

    async def _refire_kind(self, row: dict[str, Any], now: float) -> Optional[str]:
        """How a never-emailed open row is announced, from the earlier rows
        of its fingerprint: ``opened``, ``carried`` (people were told the
        incident opened and not yet that it resolved, and it came back within
        RESOLVE_EMAIL_QUIET_S) or None (a resolve email for it went out less
        than FLAP_COOLDOWN_S ago: wait)."""
        history = await self.store.alerts_history(row["fingerprint"], FLAP_HISTORY)
        notified = next(
            (
                r
                for r in history
                if int(r["id"]) < int(row["id"])
                and r.get("last_emailed_at") is not None
            ),
            None,
        )
        if notified is None:
            return "opened"
        told = _epoch(notified.get("last_emailed_at"))
        resolved = _epoch(notified.get("resolved_at"))
        if told is None or resolved is None:
            return "opened"
        if told < resolved:
            first = _epoch(row.get("first_seen"))
            if first is not None and first - resolved <= RESOLVE_EMAIL_QUIET_S:
                return "carried"
            return "opened"
        return None if now - told < FLAP_COOLDOWN_S else "opened"

    async def _resolved_to_email(
        self, outcome: CycleOutcome, mutes: dict[str, dict[str, Any]], now: float
    ) -> list[dict[str, Any]]:
        """Resolved rows whose open email went out and whose resolve email did
        not, once they have stayed resolved for RESOLVE_EMAIL_QUIET_S and the
        fingerprint has not come back within that time (then the newer row
        carries the incident and this resolve email is dropped)."""
        rows = {r["id"]: r for r in outcome.resolved}
        for row in await self.store.alerts_page(
            "resolved", None, RESOLVED_EMAIL_LOOKBACK
        ):
            rows.setdefault(row["id"], row)
        out = []
        lookups = 0
        for row in sorted(rows.values(), key=lambda r: int(r["id"])):
            last = _epoch(row.get("last_emailed_at"))
            resolved = _epoch(row.get("resolved_at"))
            if last is None or resolved is None or last >= resolved:
                continue
            if now - resolved > RESOLVED_EMAIL_MAX_AGE_S:
                continue
            if row["rule"] in IN_APP_ONLY_RULES or row["rule"] in mutes:
                continue
            if now - resolved < RESOLVE_EMAIL_QUIET_S or lookups >= FLAP_LOOKUPS_MAX:
                continue
            lookups += 1
            newer = [
                r
                for r in await self.store.alerts_history(
                    row["fingerprint"], FLAP_HISTORY
                )
                if int(r["id"]) > int(row["id"])
            ]
            if any(
                (_epoch(r.get("first_seen")) or 0) - resolved <= RESOLVE_EMAIL_QUIET_S
                and emailable(r["rule"], str(r.get("severity")))
                for r in newer
            ):
                continue
            out.append(row)
        return out

    async def _settle_inflight(self, state: dict[str, Any], now: float) -> dict:
        """The outcome of a send that ran out of time, once its thread is done."""
        settled = self.mailer.take_settled()
        if settled is None:
            return state
        error, events = settled
        if error is None:
            logger.info(
                "Ops alert email for %d alert(s) was delivered after its time budget",
                len(events),
            )
            return await self._record_success(state, len(events), now)
        # Not delivered: the alerts are claimed again later. The failure was
        # counted when the time ran out.
        await self._unclaim(events)
        logger.warning(
            "Ops alert email that ran out of time failed (%s); %d alert(s) "
            "will be retried",
            _error_class(error),
            len(events),
        )
        return state

    async def _recover_when_idle(self, state: dict[str, Any], now: float) -> None:
        """Clear the send-failure state once nothing is left to retry and the
        SMTP check has passed since the last failure (FR-63): a failed alert
        that resolved before its retry must not keep the banner up."""
        if not int(state.get("consecutive_failures") or 0):
            return
        from lightrag.api.ops.probes import KV_PREFIX

        probe = await self.store.kv_get(KV_PREFIX + "smtp")
        if smtp_recovered(state, probe):
            await clear_send_failures(self.store, self.service, state)

    async def _unclaim(self, events: list[EmailEvent]) -> None:
        for event in events:
            try:
                await self.store.alert_unclaim_email(
                    event.row["id"], event.claimed, event.previous
                )
            except Exception as e:
                logger.warning(
                    "Ops alert email claim could not be undone (%s)", type(e).__name__
                )

    async def _record_failure(
        self,
        state: dict[str, Any],
        error: BaseException,
        count: int,
        now: float,
        *,
        uncertain: bool = False,
    ) -> dict[str, Any]:
        consecutive = int(state.get("consecutive_failures") or 0) + 1
        delay = min(EMAIL_BACKOFF_MAX_S, EMAIL_BACKOFF_BASE_S * 2 ** (consecutive - 1))
        error_class = "timeout" if uncertain else _error_class(error)
        if uncertain:
            logger.warning(
                "Ops alert email for %d alert(s) ran out of time; it may still be "
                "delivered and is not sent again",
                count,
            )
        else:
            logger.warning(
                "Ops alert email could not be sent (%s); %d alert(s) will be "
                "retried in %.0f s",
                error_class,
                count,
                delay,
            )
        new = {
            **state,
            "failures_total": int(state.get("failures_total") or 0) + 1,
            "uncertain_total": int(state.get("uncertain_total") or 0)
            + (1 if uncertain else 0),
            "consecutive_failures": consecutive,
            "failing_since": state.get("failing_since")
            if consecutive > 1 and state.get("failing_since")
            else now,
            "last_error_class": error_class,
            "last_failure_at": now,
            "retry_after": now + delay,
        }
        await self._store_email_state(new)
        return new

    async def _record_success(
        self, state: dict[str, Any], count: int, now: float
    ) -> dict[str, Any]:
        new = {
            **state,
            "sent_total": int(state.get("sent_total") or 0) + 1,
            "alerts_sent_total": int(state.get("alerts_sent_total") or 0) + count,
            "consecutive_failures": 0,
            "failing_since": None,
            "last_sent_at": now,
            "retry_after": None,
        }
        await self._store_email_state(new)
        return new

    async def _store_email_state(self, value: dict[str, Any]) -> None:
        try:
            await self.store.kv_put(EMAIL_STATE_KEY, value)
        except Exception as e:
            logger.warning(
                "Ops alert email state could not be stored (%s)", type(e).__name__
            )
        self.service.cache.forget(("alert_state", EMAIL_STATE_KEY))

    async def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < CLEANUP_EVERY_S:
            return
        self._last_cleanup = now
        try:
            deleted = await self.store.alerts_cleanup(RETENTION_DAYS)
            if deleted:
                logger.info("Ops alert retention removed %d rows", deleted)
        except Exception as e:
            logger.warning("Ops alert retention failed (%s)", type(e).__name__)

    def close(self) -> None:
        self.mailer.close()


# -- admin actions (ack, mute, maintenance, thresholds) -------------------------------------


class AlertActionRefused(Exception):
    def __init__(self, status: int, code: str, *, retry_after: Optional[float] = None):
        super().__init__(code)
        self.status = status
        self.code = code
        self.retry_after = retry_after


def clean_reason(reason: Any) -> str:
    text = redact(str(reason or "").strip(), REASON_MAX)
    if len(text) < REASON_MIN:
        raise AlertActionRefused(400, "invalid_reason")
    return text


class AlertActions:
    """Audited alert administration over one worker's :class:`OpsService`."""

    def __init__(self, service: OpsService, *, clock: Callable[[], float] = time.time):
        self.service = service
        self.clock = clock

    @property
    def store(self) -> Any:
        if not self.service.available:
            raise AlertActionRefused(503, "data_unavailable")
        return self.service.store

    async def _rate(self, user: Any) -> None:
        from lightrag.api.ops.actions import ActionRefused, OpsActions

        try:
            await OpsActions(self.service, clock=self.clock).check_rate(user)
        except ActionRefused as e:
            raise AlertActionRefused(e.status, e.code, retry_after=e.retry_after) from e

    async def audited(
        self,
        *,
        user: Any,
        request: Any,
        action: str,
        reason: str,
        params: dict[str, Any],
        target: Optional[str],
        body: Callable[[], Any],
    ) -> dict[str, Any]:
        """Rate limit, intent row (refused without it), ``body``, result row."""
        store = self.store
        await self._rate(user)
        try:
            intent_id = await write_audit(
                store,
                user=user,
                action=action,
                phase="intent",
                request=request,
                target=target,
                params=params,
                reason=reason,
            )
        except Exception as e:
            logger.warning(
                "Ops %s refused: intent audit failed (%s)", action, type(e).__name__
            )
            raise AlertActionRefused(503, "audit_unavailable") from e
        try:
            outcome = await body()
        except AlertActionRefused as e:
            await self._result(
                user, request, action, reason, params, target, intent_id, e.code
            )
            raise
        except Exception:
            await self._result(
                user, request, action, reason, params, target, intent_id, "error"
            )
            raise
        code = str(outcome.get("result") or "ok")
        result_id = await self._result(
            user, request, action, reason, params, target, intent_id, code
        )
        logger.info("Ops %s by %s: %s", action, actor_of(user)["actor"], code)
        return {
            **outcome,
            "ok": True,
            "audit_id": intent_id,
            "result_audit_id": result_id,
        }

    async def _result(
        self, user, request, action, reason, params, target, intent_id, outcome
    ) -> Optional[int]:
        try:
            return await write_audit(
                self.store,
                user=user,
                action=action,
                phase="result",
                request=request,
                target=target,
                params={**params, "intent_id": intent_id},
                reason=reason,
                outcome=outcome[:24],
            )
        except Exception as e:
            logger.warning("Ops %s result audit failed (%s)", action, type(e).__name__)
            return None

    def _forget(self, *keys: str) -> None:
        for key in keys:
            self.service.cache.forget(("alert_state", key))
        self.service.cache.forget(("attention",))

    async def ack(self, *, user: Any, request: Any, alert_id: int, reason: str):
        note = clean_reason(reason)
        actor = actor_of(user)["actor"] or "-"

        async def body() -> dict[str, Any]:
            code, row = await self.service._q(
                lambda: self.store.alert_ack(alert_id, actor, note)
            )
            if code == "not_found":
                raise AlertActionRefused(404, "not_found")
            view = (
                alert_view(
                    row, await mutes_state(self.service, self.clock()), self.clock()
                )
                if row
                else None
            )
            return {"result": code, "alert": view}

        return await self.audited(
            user=user,
            request=request,
            action="alert_ack",
            reason=note,
            params={"alert_id": alert_id},
            target=str(alert_id),
            body=body,
        )

    async def mute(
        self, *, user: Any, request: Any, rule: str, minutes: int, reason: str
    ) -> dict[str, Any]:
        if rule not in ALERT_RULES:
            raise AlertActionRefused(404, "unknown_rule")
        if not isinstance(minutes, int) or not 0 <= minutes <= MUTE_MAX_MINUTES:
            raise AlertActionRefused(400, "invalid_minutes")
        text = clean_reason(reason)
        actor = actor_of(user)["actor"]
        now = self.clock()

        async def body() -> dict[str, Any]:
            value = (
                {"until": now + minutes * 60, "reason": text, "actor": actor, "at": now}
                if minutes
                else None
            )
            # A top-level merge: concurrent mutes of other rules are kept.
            await self.service._q(
                lambda: self.store.kv_merge_many({MUTES_KEY: {rule: value}})
            )
            self._forget(MUTES_KEY)
            mutes = active_mutes(await self.store.kv_get(MUTES_KEY), self.clock())
            return {
                "result": "muted" if minutes else "unmuted",
                "rule": rule,
                "muted": mutes.get(rule),
                "until": (mutes.get(rule) or {}).get("until"),
            }

        return await self.audited(
            user=user,
            request=request,
            action="alert_mute" if minutes else "alert_unmute",
            reason=text,
            params={"rule": rule, "minutes": minutes},
            target=rule,
            body=body,
        )

    async def maintenance(
        self, *, user: Any, request: Any, minutes: int, reason: str
    ) -> dict[str, Any]:
        """Start (``minutes`` > 0) or end (0) maintenance mode."""
        if not isinstance(minutes, int) or not 0 <= minutes <= MAINTENANCE_MAX_MINUTES:
            raise AlertActionRefused(400, "invalid_minutes")
        text = clean_reason(reason)
        actor = actor_of(user)["actor"]
        now = self.clock()

        async def body() -> dict[str, Any]:
            current = maintenance_view(
                await self.service._q(lambda: self.store.kv_get(MAINTENANCE_KEY)), now
            )
            if not minutes and current is None:
                return {"result": "not_active", "maintenance": None}
            value = (
                {
                    "until": now + minutes * 60,
                    "reason": text,
                    "actor": actor,
                    "started_at": now,
                }
                if minutes
                else {"until": 0, "ended_at": now, "ended_by": actor}
            )
            await self.service._q(lambda: self.store.kv_put(MAINTENANCE_KEY, value))
            self._forget(MAINTENANCE_KEY)
            self.service.cache.forget(("overview",))
            view = maintenance_view(value, self.clock())
            return {
                "result": "started" if minutes else "ended",
                "maintenance": view,
                "until": view["until"] if view else None,
            }

        return await self.audited(
            user=user,
            request=request,
            action="maintenance_start" if minutes else "maintenance_end",
            reason=text,
            params={"minutes": minutes},
            target="maintenance",
            body=body,
        )

    async def update_thresholds(
        self, *, user: Any, request: Any, changes: Any, reason: str
    ) -> dict[str, Any]:
        text = clean_reason(reason)
        stored = await self.service._q(lambda: self.store.kv_get("thresholds"))
        try:
            new, diff = validate_thresholds(changes, stored)
        except ThresholdError as e:
            raise AlertActionRefused(400, e.code) from e

        async def body() -> dict[str, Any]:
            if diff:
                await self.service._q(lambda: self.store.kv_put("thresholds", new))
            self.service.cache.forget(("thresholds",))
            self.service.cache.forget(("overview",))
            return {"result": "updated" if diff else "unchanged", "changes": diff}

        return await self.audited(
            user=user,
            request=request,
            action="alert_rules_update",
            reason=text,
            params={"changes": json.loads(json.dumps(diff, default=str))},
            target="thresholds",
            body=body,
        )


# -- incident timeline export ---------------------------------------------------------------------

EXPORT_MAX_DAYS = 31
EXPORT_MAX_LINES = 10000
EXPORT_LIMITS = {"alerts": 2000, "audit": 5000, "runs": 2000}
_RUN_FIELDS = (
    "run_id",
    "workspace",
    "kind",
    "trigger",
    "status",
    "was_orphaned",
    "docs_total",
    "docs_done",
    "docs_failed",
    "stage",
    "error_kind",
)


async def timeline_lines(
    service: OpsService, start: datetime, end: datetime
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Alerts, audit rows, runs and markers in ``[start, end)``, oldest first,
    capped. No log text, run labels or error messages; titles and reasons were
    redacted when written."""
    from lightrag.api.ops.audit import audit_view
    from lightrag.api.ops.service import EVENT_KINDS, RUN_EVENT_KIND
    from lightrag.api.ops.store import AuditFilter, utc_iso

    store = service.store
    now = service.clock()
    lines: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    truncated: list[str] = []
    mutes = await mutes_state(service, now)

    alerts = await service._q(
        lambda: store.alerts_window(start, end, EXPORT_LIMITS["alerts"] + 1)
    )
    if len(alerts) > EXPORT_LIMITS["alerts"]:
        truncated.append("alerts")
    for row in alerts[: EXPORT_LIMITS["alerts"]]:
        view = alert_view(row, mutes, now)
        lines.append({"type": "alert", "ts": view["first_seen"], **view})
    counts["alerts"] = min(len(alerts), EXPORT_LIMITS["alerts"])

    audit = await service._q(
        lambda: store.audit_page(
            AuditFilter(start=start, end=end), None, EXPORT_LIMITS["audit"] + 1
        )
    )
    if len(audit) > EXPORT_LIMITS["audit"]:
        truncated.append("audit")
    for row in audit[: EXPORT_LIMITS["audit"]]:
        view = audit_view(row)
        lines.append({"type": "audit", **view})
    counts["audit"] = min(len(audit), EXPORT_LIMITS["audit"])

    runs: list[dict[str, Any]] = []
    if service.durable_runs:
        cursor: Optional[tuple[datetime, str]] = (end, "f" * 32)
        while len(runs) <= EXPORT_LIMITS["runs"]:
            page = await service._q(
                lambda: store.runs_page(
                    workspace=None, kind=None, status=None, before=cursor, limit=500
                )
            )
            stop = False
            for row in page:
                started = row.get("started_at")
                moment = _epoch(started)
                if moment is not None and moment < start.timestamp():
                    stop = True
                    break
                runs.append(row)
            if stop or len(page) < 500:
                break
            last = page[-1]
            cursor = (last["started_at"], last["run_id"])
        if len(runs) > EXPORT_LIMITS["runs"]:
            truncated.append("runs")
        for row in runs[: EXPORT_LIMITS["runs"]]:
            line = {"type": "run", "ts": utc_iso(row.get("started_at"))}
            line.update({k: row.get(k) for k in _RUN_FIELDS})
            line["started_at"] = utc_iso(row.get("started_at"))
            line["finished_at"] = utc_iso(row.get("finished_at"))
            lines.append(line)
    counts["runs"] = min(len(runs), EXPORT_LIMITS["runs"])

    events = await service._q(
        lambda: service._events(start, end, [*EVENT_KINDS, RUN_EVENT_KIND])
    )
    marker_count = 0
    for event in events["items"]:
        lines.append({"type": "event", **event})
        marker_count += 1
    counts["events"] = marker_count

    lines.sort(key=lambda line: (line.get("ts") or "", line["type"]))
    if len(lines) > EXPORT_MAX_LINES:
        truncated.append("lines")
        lines = lines[-EXPORT_MAX_LINES:]
    meta = {
        "type": "meta",
        "from": start.isoformat(),
        "to": end.isoformat(),
        "generated_at": _iso(now),
        "counts": counts,
        "truncated": truncated,
        "redacted": True,
        "log_text": False,
    }
    return lines, meta


def ndjson(lines: list[dict[str, Any]], meta: dict[str, Any]) -> bytes:
    out = [json.dumps(meta, default=str)]
    out.extend(json.dumps(line, default=str) for line in lines)
    return ("\n".join(out) + "\n").encode("utf-8")


def export_window(
    start: Optional[datetime], end: Optional[datetime], now: datetime
) -> tuple[datetime, datetime]:
    end = end or now
    start = start or end - timedelta(hours=24)
    if end <= start or end - start > timedelta(days=EXPORT_MAX_DAYS):
        raise AlertActionRefused(400, "invalid_range")
    return start, end
