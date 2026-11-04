"""Operations console alerts (PR-8): the leased evaluator that owns the
attention snapshot, the OPS_ALERTS state machine, claim-then-send email,
mutes, acks, maintenance mode, deploy suppression, the SMTP-broken channel,
the new rules (email silence, failure spike), threshold edits, the incident
export and retention.

No test talks to an SMTP server: every send goes to an in-process stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import smtplib
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from lightrag.api.ops import alerts as ops_alerts
from lightrag.api.ops import auth as ops_auth
from lightrag.api.ops.alerts import (
    ALERT_RULES,
    ALERTS_LEASE,
    AlertEvaluator,
    CycleOutcome,
    EmailEvent,
    build_message,
    validate_thresholds,
)
from lightrag.api.ops.auth import require_live_admin
from lightrag.api.ops.leases import LeasedLoop
from lightrag.api.ops.service import (
    ATTENTION_KEY,
    DEFAULT_THRESHOLDS,
    RUNBOOK,
    OpsDataUnavailable,
    OpsService,
    _restarted_recently,
    restart_window,
)
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import MemoryOpsStore, workspace_hash
from tests.ops_fixtures import (
    SECRET_EMAIL,
    SECRET_FILE,
    SECRET_KEY,
    admin,
    build_app,
    ops_routes,
    seeded_store,
)

pytestmark = pytest.mark.offline

CLIENT = {"X-LightRAG-Client": "webui"}
BASE = "/admin/ops"
REASON = {"reason": "investigating the outage"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "OPS_ALERT_EMAIL",
        "OPS_ALERT_EMAIL_TO",
        "OPS_ALERTS_ENABLED",
        "OPS_PUBLIC_URL",
        "INSIGHTS_PUBLIC_URL",
        "OPS_RUNBOOK_BASE_URL",
        "OPS_OPERATOR_ACCOUNTS",
        "ALERT_EMAIL_TO",
        "SMTP_HOST",
        "SMTP_FROM",
    ):
        monkeypatch.delenv(name, raising=False)


class Clock:
    def __init__(self, start: float | None = None):
        self.now = start if start is not None else time.time()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Loop:
    """A lease that is always ours (``LeasedLoop`` itself is tested below)."""

    holder = "w1"

    async def still_held(self) -> bool:
        return True


class Mail:
    """In-process stand-in for ``send_email_sync``: records, or fails."""

    def __init__(self, fail: BaseException | None = None):
        self.sent: list[tuple[list[str], Any]] = []
        self.fail = fail

    def __call__(self, settings, recipients, msg) -> None:
        if self.fail is not None:
            raise self.fail
        self.sent.append((list(recipients), msg))


def smtp(**over: Any) -> SimpleNamespace:
    values = {
        "enabled": True,
        "host": "mail.example.org",
        "port": 25,
        "sender": "lightrag-alerts@example.org",
        "timeout": 5,
        "repeat_seconds": 24 * 3600,
        "admin_recipients": ("ops@example.org",),
    }
    values.update(over)
    return SimpleNamespace(**values)


def item(rule: str, severity: str = "critical", **over: Any) -> dict[str, Any]:
    out = {
        "id": over.pop("id", rule),
        "rule": rule,
        "severity": severity,
        "title": over.pop("title", f"{rule} is happening"),
        "params": over.pop("params", {"count": 1}),
        "since": over.pop("since", None),
        "count": over.pop("count", None),
        "link": "#/?ops=1&otab=health",
        "runbook": RUNBOOK.get(rule),
    }
    out.update(over)
    return out


def snapshot(*items: dict[str, Any], **over: Any) -> dict[str, Any]:
    out = {
        "at_epoch": time.time(),
        "items": [dict(i) for i in items],
        "streaks": {},
        "failed_sections": [],
        "deploy_recent": False,
    }
    out.update(over)
    return out


def service_for(store: MemoryOpsStore, clock: Clock | None = None) -> OpsService:
    service = OpsService(
        store=store,
        reader=NamespaceReader(),
        app_state=SimpleNamespace(),
        clock=clock or time.time,
    )
    service.started_at = time.time() - 3600
    return service


def evaluator(store, clock=None, mail=None, settings=None) -> AlertEvaluator:
    service = service_for(store, clock)
    return AlertEvaluator(
        service,
        process=lambda: {"mode": "uvicorn", "expected": 1, "live": 1},
        clock=clock or time.time,
        send=mail or Mail(),
        settings=lambda: settings or smtp(),
    )


def open_rows(store: MemoryOpsStore) -> list[dict[str, Any]]:
    return [r for r in store.alert_rows if r["resolved_at"] is None]


async def run(ev: AlertEvaluator, *items: dict[str, Any], **over: Any) -> CycleOutcome:
    """One evaluation of a crafted snapshot, as the cycle applies it."""
    previous = await ev.store.kv_get(ATTENTION_KEY)
    snap = snapshot(*items, **over)
    outcome = await ev.apply(snap, previous)
    await ev.store.kv_put(ATTENTION_KEY, snap)
    return outcome


# -- state machine -------------------------------------------------------------------------


async def test_firing_then_resolved_after_two_missing_evaluations():
    store = MemoryOpsStore()
    ev = evaluator(store)
    out = await run(ev, item("dependency_down", id="dependency_down:neo4j"))
    assert len(out.fired) == 1
    [row] = store.alert_rows
    assert row["state"] == "firing" and row["fingerprint"] == "dependency_down:neo4j"
    assert row["runbook"] == RUNBOOK["dependency_down"]
    # Still present: touched, not duplicated.
    out = await run(ev, item("dependency_down", id="dependency_down:neo4j"))
    assert out.fired == [] and len(store.alert_rows) == 1
    # One missed evaluation never resolves (flapping guard)...
    await run(ev)
    assert row["state"] == "firing"
    assert store.kv[ATTENTION_KEY]["alert_missing"] == {"dependency_down:neo4j": 1}
    # ...the second one does.
    out = await run(ev)
    assert row["state"] == "resolved" and row["resolved_at"] is not None
    assert [r["id"] for r in out.resolved] == [row["id"]]


async def test_a_blip_does_not_resolve_and_refire():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("pg_probe"))
    await run(ev)
    await run(ev, item("pg_probe"))
    assert len(store.alert_rows) == 1 and store.alert_rows[0]["state"] == "firing"
    assert store.kv[ATTENTION_KEY]["alert_missing"] == {}


async def test_refire_after_resolve_creates_a_new_row():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("disk_free"))
    await run(ev)
    await run(ev)
    out = await run(ev, item("disk_free"))
    assert len(out.fired) == 1
    assert [r["state"] for r in store.alert_rows] == ["resolved", "firing"]
    assert len(open_rows(store)) == 1


async def test_severity_change_touches_the_open_row():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("disk_free", "warning", title="Disk 12% free"))
    await run(ev, item("disk_free", "critical", title="Disk 3% free"))
    [row] = store.alert_rows
    assert row["severity"] == "critical" and row["title"] == "Disk 3% free"


async def test_nothing_resolves_while_a_section_failed():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("stale_docs", "warning"))
    for _ in range(3):
        await run(ev, failed_sections=["stuck"])
    assert open_rows(store)
    await run(ev)
    await run(ev)
    assert not open_rows(store)


async def test_ack_keeps_the_state_and_adds_no_row():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("run_hung", "serious"))
    [row] = store.alert_rows
    code, acked = await store.alert_ack(row["id"], "khairil", "looking")
    assert code == "acked" and acked["state"] == "firing"
    assert acked["acked_by"] == "khairil" and acked["ack_note"] == "looking"
    await run(ev, item("run_hung", "serious"))
    assert len(store.alert_rows) == 1 and store.alert_rows[0]["acked_by"] == "khairil"
    assert (await store.alert_ack(row["id"], "x", None))[0] == "already_acked"
    assert (await store.alert_ack(999, "x", None))[0] == "not_found"
    snap = store.kv[ATTENTION_KEY]
    assert snap["items"][0]["alert_id"] == row["id"]
    assert snap["items"][0]["acked_by"] == "khairil"


async def test_concurrent_evaluators_keep_one_open_row_per_fingerprint():
    store = MemoryOpsStore()
    store.write_delay_s = 0.01
    first, second = evaluator(store), evaluator(store)
    snap = snapshot(item("pg_probe"), item("disk_free", "warning"))
    await asyncio.gather(
        first.apply(json.loads(json.dumps(snap)), None),
        second.apply(json.loads(json.dumps(snap)), None),
    )
    fingerprints = [r["fingerprint"] for r in open_rows(store)]
    assert sorted(fingerprints) == ["disk_free", "pg_probe"]


async def test_alerts_off_still_publishes_the_snapshot(monkeypatch):
    monkeypatch.setenv("OPS_ALERTS_ENABLED", "false")
    store = MemoryOpsStore()
    ev = evaluator(store)

    async def fake(process, holder):
        return snapshot(item("disk_free"))

    ev.service.compute_attention_snapshot = fake
    assert await ev.cycle(Loop()) is None
    assert store.kv[ATTENTION_KEY]["items"][0]["rule"] == "disk_free"
    assert store.alert_rows == []


# -- the lease ----------------------------------------------------------------------------------


async def test_only_the_lease_holder_evaluates_and_a_takeover_follows_expiry():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    calls: list[str] = []

    def make(holder: str) -> LeasedLoop:
        ev = evaluator(store, clock)

        async def fake(process, h):
            calls.append(h)
            return snapshot(item("disk_free"))

        ev.service.compute_attention_snapshot = fake
        return LeasedLoop(
            store,
            ALERTS_LEASE,
            holder,
            ev.cycle,
            ttl_s=ops_alerts.ALERTS_LEASE_TTL_S,
            interval_s=60,
            cycle_timeout_s=ops_alerts.ALERTS_CYCLE_TIMEOUT_S,
        )

    a, b = make("worker-a"), make("worker-b")
    assert await a.run_once()
    assert not await b.run_once()
    assert calls == ["worker-a"]
    # worker-a stops renewing (it died): after the TTL worker-b takes over.
    clock.advance(ops_alerts.ALERTS_LEASE_TTL_S + 1)
    assert await b.run_once()
    assert calls == ["worker-a", "worker-b"]
    assert len(open_rows(store)) == 1


def test_lease_outlasts_the_cycle():
    assert ops_alerts.ALERTS_LEASE_TTL_S > ops_alerts.ALERTS_CYCLE_TIMEOUT_S
    assert ops_alerts.ALERTS_LEASE == "ops:alerts"


# -- email: claim then send ------------------------------------------------------------------


@pytest.fixture
def email_on(monkeypatch):
    monkeypatch.setenv("OPS_ALERT_EMAIL", "true")


async def test_claim_then_send_sends_once_across_workers(email_on):
    store = MemoryOpsStore()
    store.write_delay_s = 0.005
    mails = [Mail(), Mail(), Mail()]
    workers = [evaluator(store, mail=m) for m in mails]
    outcome = await run(workers[0], item("dependency_down"))
    sent = await asyncio.gather(*(w.notify(outcome) for w in workers))
    assert sorted(sent) == [0, 0, 1]
    assert sum(len(m.sent) for m in mails) == 1
    # Later cycles (any worker) do not send again for the same open alert.
    for w in workers:
        assert await w.notify(await run(w, item("dependency_down"))) == 0
    assert sum(len(m.sent) for m in mails) == 1
    assert store.alert_rows[0]["last_emailed_at"] is not None


async def test_never_sends_when_email_is_off():
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    outcome = await run(ev, item("dependency_down"))
    assert await ev.notify(outcome) == 0
    assert mail.sent == [] and store.alert_rows[0]["last_emailed_at"] is None
    view = ops_alerts.alert_view(store.alert_rows[0], {}, time.time())
    assert view["email"] == "record_only"


async def test_only_critical_and_serious_and_never_provider_paused(email_on):
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    outcome = await run(
        ev,
        item("provider_paused", "critical"),
        item("disk_free", "warning"),
        item("run_hung", "serious"),
    )
    assert await ev.notify(outcome) == 1
    [(_, msg)] = mail.sent
    assert "run_hung" in msg["Subject"]
    assert "provider_paused" not in msg.as_string()
    assert "disk_free" not in msg.as_string()
    paused = next(r for r in store.alert_rows if r["rule"] == "provider_paused")
    assert paused["last_emailed_at"] is None
    assert ops_alerts.alert_view(paused, {}, time.time())["email"] == "in_app_only"


async def test_repeat_after_the_repeat_window_and_one_resolve_email(email_on):
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = Mail()
    ev = evaluator(store, clock, mail, smtp(repeat_seconds=3600))
    assert await ev.notify(await run(ev, item("pg_probe"))) == 1
    clock.advance(1800)
    assert await ev.notify(await run(ev, item("pg_probe"))) == 0
    clock.advance(1900)
    assert await ev.notify(await run(ev, item("pg_probe"))) == 1
    assert "Still open" in mail.sent[-1][1].get_content()
    clock.advance(60)
    await ev.notify(await run(ev))
    # Resolved, but the resolve email waits out the flapping window.
    assert await ev.notify(await run(ev)) == 0
    # One second past the window: stored times round to the microsecond.
    clock.advance(ops_alerts.RESOLVE_EMAIL_QUIET_S + 1)
    assert await ev.notify(await run(ev)) == 1
    assert mail.sent[-1][1]["Subject"].endswith("RESOLVED pg_probe")
    # Never twice.
    assert await ev.notify(await run(ev)) == 0
    assert len(mail.sent) == 3


async def test_no_resolve_email_for_an_alert_never_emailed(email_on, monkeypatch):
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    monkeypatch.setenv("OPS_ALERT_EMAIL", "false")
    await ev.notify(await run(ev, item("pg_probe")))
    monkeypatch.setenv("OPS_ALERT_EMAIL", "true")
    await run(ev)
    out = await run(ev)
    assert out.resolved and await ev.notify(out) == 0
    assert mail.sent == []


async def test_mutes_and_maintenance_suppress_email_but_record(email_on):
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = Mail()
    ev = evaluator(store, clock, mail)
    store.kv[ops_alerts.MUTES_KEY] = {
        "dependency_down": {"until": clock.now + 3600, "reason": "known", "actor": "k"}
    }
    store.kv[ops_alerts.MAINTENANCE_KEY] = {
        "until": clock.now + 600,
        "reason": "deploy",
    }
    out = await run(ev, item("dependency_down"), item("pg_probe"))
    assert len(open_rows(store)) == 2
    assert await ev.notify(out) == 0
    # Maintenance over: the unmuted alert goes out, the muted one does not.
    clock.advance(601)
    assert (
        await ev.notify(await run(ev, item("dependency_down"), item("pg_probe"))) == 1
    )
    body = mail.sent[0][1].get_content()
    assert "pg_probe" in body and "dependency_down" not in body
    view = ops_alerts.alert_view(
        next(r for r in store.alert_rows if r["rule"] == "dependency_down"),
        ops_alerts.active_mutes(store.kv[ops_alerts.MUTES_KEY], clock.now),
        clock.now,
        email_on=True,
    )
    assert view["email"] == "muted" and view["muted_until"]


async def test_acked_alerts_get_no_email(email_on):
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    await run(ev, item("pg_probe"))
    await store.alert_ack(store.alert_rows[0]["id"], "k", "on it")
    assert await ev.notify(await run(ev, item("pg_probe"))) == 0


async def test_smtp_failure_counts_shows_broken_and_retries_with_backoff(email_on):
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = Mail(fail=smtplib.SMTPServerDisconnected("down"))
    ev = evaluator(store, clock, mail)
    outcome = await run(ev, item("pg_probe"))
    assert await ev.notify(outcome) == 0  # no exception
    assert store.alert_rows[0]["last_emailed_at"] is None  # claim undone
    state = store.kv[ops_alerts.EMAIL_STATE_KEY]
    assert state["consecutive_failures"] == 1 and state["failures_total"] == 1
    assert state["last_error_class"] == "smtp_error"
    channel = await ops_alerts.channel_state(ev.service)
    assert channel["state"] == "broken" and channel["reason"] in (
        "send_failed",
        "smtp_not_configured",
    )
    # Within the backoff nothing is attempted.
    mail.fail = None
    assert await ev.notify(await run(ev, item("pg_probe"))) == 0
    clock.advance(ops_alerts.EMAIL_BACKOFF_BASE_S + 1)
    assert await ev.notify(await run(ev, item("pg_probe"))) == 1
    state = store.kv[ops_alerts.EMAIL_STATE_KEY]
    assert state["consecutive_failures"] == 0 and state["sent_total"] == 1


async def test_cycle_survives_a_crashing_send(email_on):
    store = MemoryOpsStore()
    ev = evaluator(store, mail=Mail(fail=RuntimeError("boom")))

    async def fake(process, holder):
        return snapshot(item("pg_probe"))

    ev.service.compute_attention_snapshot = fake
    await ev.cycle(Loop())
    assert store.kv[ATTENTION_KEY]["items"][0]["alert_id"] == 1
    assert store.kv[ops_alerts.EMAIL_STATE_KEY]["consecutive_failures"] == 1


async def test_recipients_fall_back_to_alert_email_to(monkeypatch):
    settings = smtp(admin_recipients=("admins@example.org",))
    assert ops_alerts.alert_recipients(settings) == ["admins@example.org"]
    monkeypatch.setenv("OPS_ALERT_EMAIL_TO", "oncall@example.org, b@example.org")
    assert ops_alerts.alert_recipients(settings) == [
        "oncall@example.org",
        "b@example.org",
    ]


async def test_channel_states(monkeypatch):
    store = MemoryOpsStore()
    service = service_for(store)
    channel = await ops_alerts.channel_state(service)
    assert channel["state"] == "record_only" and channel["ok"] is None
    assert channel["reason"] == "smtp_not_configured"
    monkeypatch.setenv("OPS_ALERT_EMAIL", "true")
    service.cache.clear()
    assert (await ops_alerts.channel_state(service))["state"] == "broken"
    monkeypatch.setenv("SMTP_HOST", "mail.example.org")
    monkeypatch.setenv("SMTP_FROM", "lightrag@example.org")
    monkeypatch.setenv("ALERT_EMAIL_TO", "ops@example.org")
    service.cache.clear()
    channel = await ops_alerts.channel_state(service)
    assert channel["state"] == "ok" and channel["ok"] is True
    assert channel["recipients"] == 1
    store.kv["probe:smtp"] = {"status": "down", "consecutive_failures": 2}
    service.cache.clear()
    channel = await ops_alerts.channel_state(service)
    assert channel["state"] == "broken" and channel["reason"] == "smtp_probe_failing"


# -- email body ---------------------------------------------------------------------------------


def test_email_body_carries_no_names_emails_secrets_or_workspaces(monkeypatch):
    monkeypatch.setenv("OPS_PUBLIC_URL", "https://rag.example.org/")
    row = {
        "id": 42,
        "rule": "run_orphaned",
        "severity": "critical",
        "title": f"Run orphaned in fsktm_unimas_my for {SECRET_FILE}",
        "scope": {
            "count": 2,
            "workspaces": ["fsktm_unimas_my", "a_bin_unimas_my"],
            "owner": SECRET_EMAIL,
            "dependency": f"{SECRET_FILE} {SECRET_KEY}",
            "message": f"error for {SECRET_EMAIL}",
            "by_slot_kind": {"extraction": 2},
            "since": "2026-11-04T01:00:00+00:00",
        },
        "first_seen": datetime(2026, 11, 4, 1, 0, tzinfo=timezone.utc),
        "resolved_at": None,
    }
    msg = build_message(
        "alerts@example.org",
        ["ops@example.org"],
        [EmailEvent(row, "opened", datetime.now(timezone.utc), None)],
        link_base=ops_alerts.public_url(),
    )
    text = msg.as_string()
    for secret in (
        SECRET_FILE,
        "Senate minutes",
        SECRET_EMAIL,
        SECRET_KEY,
        "fsktm_unimas_my",
        "a_bin_unimas_my",
        "Run orphaned in",
    ):
        assert secret not in text, secret
    assert msg["Subject"] == "[LightRAG] CRITICAL run_orphaned"
    body = msg.get_content()
    assert "count=2" in body
    assert RUNBOOK["run_orphaned"][:40] in body
    assert (
        "https://rag.example.org/webui/#/?ops=1&otab=activity&osub=alerts&oal=42"
        in body
    )


def test_subject_of_a_multi_alert_email_names_rules_and_severities_only():
    rows = [
        {"id": i, "rule": r, "severity": s, "scope": {}, "first_seen": None}
        for i, (r, s) in enumerate(
            [
                ("pg_probe", "critical"),
                ("run_hung", "serious"),
                ("disk_free", "critical"),
                ("vllm_waiting", "serious"),
            ],
            start=1,
        )
    ]
    msg = build_message(
        "a@example.org",
        ["b@example.org"],
        [EmailEvent(r, "opened", datetime.now(timezone.utc), None) for r in rows],
        link_base="",
    )
    assert msg["Subject"] == (
        "[LightRAG] 4 alerts: CRITICAL pg_probe, SERIOUS run_hung, CRITICAL disk_free (+1)"
    )
    assert "Operations > Activity > Alerts, alert 4" in msg.get_content()


def test_runbook_link(monkeypatch):
    assert ops_alerts.runbook_url("pg_probe") is None
    monkeypatch.setenv("OPS_RUNBOOK_BASE_URL", "https://wiki.example.org/ops")
    assert ops_alerts.runbook_url("pg_probe") == "https://wiki.example.org/ops#pg_probe"
    monkeypatch.setenv("OPS_RUNBOOK_BASE_URL", "https://wiki.example.org/{rule}.md")
    assert ops_alerts.runbook_url("pg_probe") == "https://wiki.example.org/pg_probe.md"
    monkeypatch.setenv("OPS_RUNBOOK_BASE_URL", "javascript:alert(1)")
    assert ops_alerts.runbook_url("pg_probe") is None


# -- the rule catalogue ---------------------------------------------------------------------------

_OPS = Path(__file__).resolve().parents[1] / "lightrag" / "api" / "ops"


def _emitted_rules() -> set[str]:
    """Every rule name an attention rule can add (from the sources)."""
    found: set[str] = set()
    for name in ("service.py", "health.py", "perf.py"):
        text = (_OPS / name).read_text()
        found |= set(re.findall(r"\badd\(\s*\"([a-z0-9_]+)\"", text))
    return found


def test_every_rule_has_a_runbook_and_a_catalogue_entry():
    emitted = _emitted_rules()
    assert len(emitted) >= 25
    assert emitted <= set(ALERT_RULES), emitted - set(ALERT_RULES)
    for rule in ALERT_RULES.values():
        assert RUNBOOK.get(rule.rule), rule.rule
        assert rule.runbook and len(rule.runbook) <= 512
        assert rule.condition
        for key in rule.thresholds:
            assert key in DEFAULT_THRESHOLDS and key in ops_alerts.THRESHOLD_BOUNDS
            low, high, _ = ops_alerts.THRESHOLD_BOUNDS[key]
            assert low <= DEFAULT_THRESHOLDS[key] <= high, key
    assert not ALERT_RULES["provider_paused"].emailed
    assert ALERT_RULES["dependency_down"].emailed
    assert not ALERT_RULES["smtp_probe"].emailed  # warning only


def test_threshold_validation():
    new, diff = validate_thresholds({"hung_minutes": 20, "llm_error_rate": 0.1}, {})
    assert new == {"hung_minutes": 20, "llm_error_rate": 0.1}
    assert diff["hung_minutes"] == {"old": None, "new": 20}
    assert isinstance(new["hung_minutes"], int)
    new, diff = validate_thresholds({"hung_minutes": None}, {"hung_minutes": 20})
    assert new == {} and diff == {"hung_minutes": {"old": 20, "new": None}}
    for changes, code in (
        ({"nope": 1}, "unknown_threshold"),
        ({"hung_minutes": "20"}, "invalid_threshold"),
        ({"hung_minutes": True}, "invalid_threshold"),
        ({"hung_minutes": 20.5}, "invalid_threshold"),
        ({"hung_minutes": 0}, "threshold_out_of_bounds"),
        ({"llm_error_rate": 2.0}, "threshold_out_of_bounds"),
        ({"hung_minutes": 50}, "threshold_order"),
        ({}, "invalid_thresholds"),
    ):
        with pytest.raises(ops_alerts.ThresholdError) as e:
            validate_thresholds(changes, {})
        assert e.value.code == code, changes
    assert ops_alerts.flatten_changes(None, {"run_hung": {"hung_minutes": 10}}) == {
        "hung_minutes": 10
    }
    with pytest.raises(ops_alerts.ThresholdError):
        ops_alerts.flatten_changes(None, {"run_hung": {"llm_min_calls": 10}})


# -- the new rules, min-sample guards, deploy suppression ------------------------------------


def _inputs(service: OpsService, **over: Any) -> dict[str, Any]:
    inputs = {
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "failed_sections": [],
        "backlog": None,
        "active": None,
        "stale_items": [],
        "stale_total": 0,
        "stuck_runs": None,
        "queue": None,
        "pauses": None,
        "events": [],
        "llm": {"available": False},
        "postgres_ok": True,
        "postgres_down": False,
        "process": {"mode": "uvicorn", "expected": 1, "live": 1},
        "worker_items": None,
        "deps": [],
        "cycle": None,
        "deploy_recent": False,
        "metrics": None,
        "email_ingest": None,
        "failure_spike": None,
    }
    inputs.update(over)
    return inputs


def _rules(service: OpsService, **over: Any) -> dict[str, dict[str, Any]]:
    items, _ = service._attention(_inputs(service, **over), time.time(), streaks={})
    return {i["rule"]: i for i in items}


def test_min_sample_guards_stop_noise():
    service = service_for(MemoryOpsStore())
    few = {"available": True, "calls": 10, "error_rate": 0.9, "p95_ms": 120000}
    assert "llm_error_rate_15m" not in _rules(service, llm=few)
    assert "llm_p95_15m" not in _rules(service, llm=few)
    enough = {"available": True, "calls": 40, "error_rate": 0.5, "p95_ms": 1000}
    assert _rules(service, llm=enough)["llm_error_rate_15m"]["severity"] == "serious"
    spike = {"last_hour": 15, "median": 0.0, "processed": 0}
    assert "failure_spike" not in _rules(service, failure_spike=spike)


async def test_min_sample_guard_records_no_alert():
    store = MemoryOpsStore()
    ev = evaluator(store)
    items, _ = ev.service._attention(
        _inputs(ev.service, llm={"available": True, "calls": 5, "error_rate": 1.0}),
        time.time(),
        streaks={},
    )
    await ev.apply(snapshot(*items), None)
    assert store.alert_rows == []


def test_email_ingest_silence_rule():
    service = service_for(MemoryOpsStore())
    now = time.time()
    assert "email_ingest_silence" not in _rules(service, email_ingest=None)
    assert "email_ingest_silence" not in _rules(service, email_ingest=now - 600)
    warn = _rules(service, email_ingest=now - 150 * 60)["email_ingest_silence"]
    assert warn["severity"] == "warning" and warn["params"]["silent_min"] == 150
    assert "otab=activity" in warn["link"] and warn["runbook"]
    serious = _rules(service, email_ingest=now - 7 * 3600)["email_ingest_silence"]
    assert serious["severity"] == "serious"


def test_failure_spike_rule():
    service = service_for(MemoryOpsStore())
    usual = {"last_hour": 25, "median": 10.0, "processed": 500}
    assert "failure_spike" not in _rules(service, failure_spike=usual)
    spike = {"last_hour": 40, "median": 5.0, "processed": 500}
    assert (
        _rules(service, failure_spike=spike)["failure_spike"]["severity"] == "warning"
    )
    share = {"last_hour": 40, "median": 5.0, "processed": 50}
    assert (
        _rules(service, failure_spike=share)["failure_spike"]["severity"] == "serious"
    )
    many = {"last_hour": 150, "median": 5.0, "processed": 5000}
    assert _rules(service, failure_spike=many)["failure_spike"]["severity"] == "serious"


async def test_failure_spike_inputs_use_the_7_day_hourly_median():
    store = MemoryOpsStore()
    service = service_for(store)
    service.run_tracking = True
    now = time.time()
    base = datetime.fromtimestamp(now, timezone.utc)

    async def throughput(start, end, bucket):
        if end - start <= timedelta(hours=1, seconds=1):
            return {"docs": [{"t": int(now) - 600, "processed": 30, "failed": 42}]}
        hours = [int(start.timestamp()) + 3600 * i for i in range(0, 168, 2)]
        return {"docs": [{"t": t, "processed": 10, "failed": 4} for t in hours]}

    store.throughput = throughput
    spike = await service._failure_spike_inputs(base.timestamp())
    assert spike == {"last_hour": 42, "processed": 30, "median": 2.0}


async def test_deploy_holds_heartbeat_and_orphan_alerts():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("run_orphaned"), item("disk_free"))
    for _ in range(3):
        await run(ev, deploy_recent=True)
    rules = {r["rule"]: r["state"] for r in store.alert_rows}
    assert rules == {"run_orphaned": "firing", "disk_free": "resolved"}
    await run(ev)
    await run(ev)
    assert not open_rows(store)


def test_deploy_suppresses_the_rules_themselves():
    service = service_for(MemoryOpsStore())
    stuck = {
        "orphaned": [{"run_id": "r1", "workspace": "fsktm", "heartbeat_at": None}],
        "hung": [],
    }
    process = {"mode": "gunicorn", "expected": 3, "live": 1}
    workers = [{"alive": True, "pid": 1, "snapshot": {}}]
    after = _rules(service, stuck_runs=stuck, process=process, worker_items=workers)
    assert "run_orphaned" in after
    held = _rules(
        service,
        stuck_runs=stuck,
        process=process,
        worker_items=workers,
        deploy_recent=True,
    )
    assert "run_orphaned" not in held and "workers_below_expected" not in held


def test_a_restart_counts_as_a_deploy():
    now = time.time()

    def worker(started_ago: float, alive: bool = True) -> dict[str, Any]:
        started = datetime.fromtimestamp(now - started_ago, timezone.utc).isoformat()
        return {"alive": alive, "started_at": started}

    assert _restarted_recently([worker(60), worker(120)], now)
    # One worker respawned after a crash: the others started long ago.
    assert not _restarted_recently([worker(60), worker(7200)], now)
    # Dead rows from before the restart do not count.
    assert _restarted_recently([worker(60), worker(7200, alive=False)], now)
    assert not _restarted_recently([], now)
    assert not _restarted_recently(None, now)
    # Workers respawned one by one (a crash loop) are not a deploy.
    assert not _restarted_recently([worker(30), worker(400)], now)


def test_a_crash_loop_gets_one_restart_window_not_suppression_forever():
    now = time.time()

    def boot(started: float) -> list[dict[str, Any]]:
        at = datetime.fromtimestamp(started, timezone.utc).isoformat()
        return [{"alive": True, "started_at": at}] * 4

    # The first restart opens a window and is remembered.
    held, prior = restart_window(boot(now - 60), now, None)
    assert held and prior == pytest.approx(now - 60)
    # The same boot keeps its window, and only for the deploy window.
    assert restart_window(boot(now - 60), now + 240, prior)[0]
    assert not restart_window(boot(now - 60), now + 600, prior)[0]
    # Restarted again 5 minutes later (and again, and again): a crash loop.
    for minutes in (5, 10, 15, 50):
        later = now + minutes * 60
        held, kept = restart_window(boot(later - 30), later, prior)
        assert not held and kept == prior
    # A restart over an hour after the remembered one is a deploy again.
    later = now + 2 * 3600
    held, new = restart_window(boot(later - 30), later, prior)
    assert held and new == pytest.approx(later - 30)


async def test_the_snapshot_remembers_the_restart_boot():
    store = MemoryOpsStore()
    service = service_for(store)
    now = time.time()
    started = datetime.fromtimestamp(now - 60, timezone.utc).isoformat()

    async def workers():
        return [{"alive": True, "started_at": started, "pid": 1, "snapshot": {}}]

    service._worker_items = workers
    process = {"mode": "uvicorn", "expected": 1, "live": 1}
    first = await service.compute_attention_snapshot(process, "w")
    assert first["deploy_recent"] and first["restart_boot"] == pytest.approx(now - 60)
    # A previous restart 20 min earlier: this one is a crash loop.
    store.kv[ATTENTION_KEY] = {**first, "restart_boot": now - 1200}
    second = await service.compute_attention_snapshot(process, "w")
    assert not second["deploy_recent"]
    assert second["restart_boot"] == pytest.approx(now - 1200)


async def test_superseded_worker_rows_do_not_fire():
    store = MemoryOpsStore()
    now = time.time()
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    for worker_id, superseded, age in (("a" * 32, False, 5), ("b" * 32, True, 600)):
        store.worker_rows[worker_id] = {
            "worker_id": worker_id,
            "host": "h",
            "pid": 1 if superseded else 2,
            "pid_create_time": now - 7200,
            "mode": "gunicorn",
            "version": "x",
            "started_at": started,
            "heartbeat_at": datetime.now(timezone.utc) - timedelta(seconds=age),
            "exited_at": None,
            "superseded": superseded,
            "leases_held": [],
            "snapshot": {},
        }
    service = service_for(store)
    process = {"mode": "gunicorn", "expected": 1, "live": None}
    first = await service.compute_attention_snapshot(process, "w")
    service.cache.clear()
    second = await service.compute_attention_snapshot(process, "w")
    assert first["deploy_recent"] is False
    for snap in (first, second):
        assert "workers_below_expected" not in {i["rule"] for i in snap["items"]}


# -- retention and deletion ---------------------------------------------------------------------


async def test_retention_keeps_open_and_recent_alerts():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    ev = evaluator(store, clock)
    await run(ev, item("pg_probe"), item("disk_free"))
    clock.advance(10)
    await run(ev, item("pg_probe"))
    await run(ev, item("pg_probe"))
    old = next(r for r in store.alert_rows if r["rule"] == "disk_free")
    old["resolved_at"] -= timedelta(days=181)
    await ev._maybe_cleanup(clock.now)
    assert [r["rule"] for r in store.alert_rows] == ["pg_probe"]
    # Hourly, not every cycle.
    store.alert_rows.append({**old, "id": 99})
    await ev._maybe_cleanup(clock.now + 60)
    assert len(store.alert_rows) == 2
    await ev._maybe_cleanup(clock.now + ops_alerts.CLEANUP_EVERY_S + 1)
    assert len(store.alert_rows) == 1


async def test_deletion_hashes_workspaces_listed_in_alert_scopes():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(
        ev,
        item("run_orphaned", params={"count": 2, "workspaces": ["fsktm", "other"]}),
    )
    await store.forget_workspace("fsktm")
    assert store.alert_rows[0]["scope"]["workspaces"] == [
        workspace_hash("fsktm"),
        "other",
    ]


# -- routes ---------------------------------------------------------------------------------------


def _app(store=None, *, live_admin=True):
    store = store if store is not None else seeded_store()
    app = build_app(store)
    if live_admin:
        app.dependency_overrides[require_live_admin] = admin
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def _call(app, method: str, path: str, body: Any = None, headers=CLIENT):
    async with _client(app) as client:
        return await client.request(
            method, BASE + path, json=body, headers=headers or {}
        )


async def test_alerts_list_open_history_and_keyset():
    store = seeded_store()
    app = _app(store)
    for i in range(3):
        store.alert_rows.append(
            {
                **store.alert_rows[0],
                "id": 10 + i,
                "fingerprint": f"old{i}",
                "state": "resolved",
                "resolved_at": datetime.now(timezone.utc) - timedelta(hours=i + 1),
            }
        )
    body = (await _call(app, "GET", "/alerts")).json()
    assert [a["id"] for a in body["items"]] == [1]
    first = body["items"][0]
    assert first["state"] == "firing" and first["email"] == "record_only"
    assert first["runbook"] == RUNBOOK["stale_docs"]
    assert first["alert_link"].endswith("oal=1")
    assert body["counts"]["open"] == 1 and body["counts"]["by_severity"]["serious"] == 1
    assert body["email_enabled"] is False and body["channel"]["state"] == "record_only"
    page = (await _call(app, "GET", "/alerts?state=resolved&limit=2")).json()
    assert [a["id"] for a in page["items"]] == [12, 11] and page["next_before"] == 11
    rest = (await _call(app, "GET", "/alerts?state=resolved&before=11")).json()
    assert [a["id"] for a in rest["items"]] == [10] and rest["next_before"] is None
    assert (await _call(app, "GET", "/alerts?state=nope")).status_code == 400
    detail = (await _call(app, "GET", "/alerts/1")).json()
    assert detail["alert"]["id"] == 1 and detail["rule"]["rule"] == "stale_docs"
    assert detail["rule"]["thresholds"][0]["key"] == "stale_doc_minutes"
    assert (await _call(app, "GET", "/alerts/404")).status_code == 404


async def test_ack_is_audited_and_keeps_state():
    store = seeded_store()
    app = _app(store)
    before = len(store.audit_rows)
    response = await _call(app, "POST", "/alerts/1/ack", REASON)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == "acked" and body["ok"] is True and body["audit_id"]
    assert body["alert"]["state"] == "firing" and body["alert"]["acked_by"] == "root"
    rows = store.audit_rows[before:]
    assert [(r["action"], r["phase"]) for r in rows] == [
        ("alert_ack", "intent"),
        ("alert_ack", "result"),
    ]
    assert rows[1]["outcome"] == "acked"
    again = (await _call(app, "POST", "/alerts/1/ack", REASON)).json()
    assert again["result"] == "already_acked"
    assert (await _call(app, "POST", "/alerts/77/ack", REASON)).status_code == 404


async def test_alert_posts_need_the_client_header_and_a_live_admin(monkeypatch):
    app = _app(seeded_store())
    response = await _call(app, "POST", "/alerts/1/ack", REASON, headers={})
    assert response.status_code == 403
    assert response.json()["detail"] == "client_header_required"
    monkeypatch.setattr(ops_auth, "is_live_admin", lambda user: False)
    removed = _app(seeded_store(), live_admin=False)
    for path, body in (
        ("/alerts/1/ack", REASON),
        ("/alerts/rules/pg_probe/mute", {"minutes": 60, **REASON}),
        ("/maintenance", {"minutes": 30, **REASON}),
        ("/maintenance/end", REASON),
    ):
        response = await _call(removed, "POST", path, body)
        assert response.status_code == 403, path
        assert response.json()["detail"] == "admin_not_current"
    response = await _call(
        removed, "PUT", "/alerts/rules", {"thresholds": {"hung_minutes": 20}, **REASON}
    )
    assert response.status_code == 403


async def test_reason_is_required_and_the_audit_must_be_written():
    store = seeded_store()
    app = _app(store)
    response = await _call(app, "POST", "/alerts/1/ack", {"reason": "  x "})
    assert response.status_code in (400, 422)

    async def broken(row):
        raise RuntimeError("audit down")

    store.audit_write = broken
    response = await _call(app, "POST", "/alerts/1/ack", REASON)
    assert response.status_code == 503
    assert response.json()["detail"] == "audit_unavailable"
    assert store.alert_rows[0]["acked_at"] is None


async def test_mute_and_unmute_a_rule():
    store = seeded_store()
    app = _app(store)
    response = await _call(
        app, "POST", "/alerts/rules/dependency_down/mute", {"minutes": 240, **REASON}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == "muted" and body["until"]
    assert store.kv[ops_alerts.MUTES_KEY]["dependency_down"]["actor"] == "root"
    rules = (await _call(app, "GET", "/alerts/rules")).json()
    muted = next(r for r in rules["rules"] if r["rule"] == "dependency_down")
    assert muted["muted"]["reason"] == REASON["reason"] and muted["muted_until"]
    assert [m["rule"] for m in rules["mutes"]] == ["dependency_down"]
    off = await _call(
        app, "POST", "/alerts/rules/dependency_down/mute", {"minutes": 0, **REASON}
    )
    assert off.json()["result"] == "unmuted"
    assert (await _call(app, "GET", "/alerts/rules")).json()["mutes"] == []
    assert [r["action"] for r in store.audit_rows if r["phase"] == "result"][-2:] == [
        "alert_mute",
        "alert_unmute",
    ]
    for path, body, status in (
        ("/alerts/rules/nope/mute", {"minutes": 60, **REASON}, 404),
        ("/alerts/rules/pg_probe/mute", {"minutes": 7 * 24 * 60 + 1, **REASON}, 422),
    ):
        assert (await _call(app, "POST", path, body)).status_code == status


async def test_maintenance_mode_shows_on_the_overview_and_ends():
    store = seeded_store()
    app = _app(store)
    response = await _call(app, "POST", "/maintenance", {"minutes": 30, **REASON})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == "started" and response.json()["until"]
    overview = (await _call(app, "GET", "/overview")).json()
    assert overview["state"] == "maintenance"
    assert overview["maintenance"]["reason"] == REASON["reason"]
    assert overview["alert_channel"]["state"] == "record_only"
    alerts = (await _call(app, "GET", "/alerts")).json()
    assert alerts["maintenance"]["by"] == "root"
    ended = await _call(app, "POST", "/maintenance/end", REASON)
    assert ended.json()["result"] == "ended"
    app.state.ops.service.cache.clear()
    assert (await _call(app, "GET", "/overview")).json()["state"] != "maintenance"
    again = await _call(app, "POST", "/maintenance", {"minutes": 0, **REASON})
    assert again.json()["result"] == "not_active"


async def test_rules_put_validates_audits_and_is_operator_only(monkeypatch):
    store = seeded_store()
    app = _app(store)
    ok = await _call(
        app,
        "PUT",
        "/alerts/rules",
        {"thresholds": {"hung_minutes": 20, "llm_error_rate": 0.1}, **REASON},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["result"] == "updated"
    assert store.kv["thresholds"] == {"hung_minutes": 20, "llm_error_rate": 0.1}
    hung = next(r for r in body["rules"] if r["rule"] == "run_hung")
    assert hung["thresholds"][0] == {
        "key": "hung_minutes",
        "value": 20,
        "default": 15,
        "overridden": True,
        "min": 2,
        "max": 1440,
        "unit": "min",
        "integer": True,
    }
    intent = next(r for r in store.audit_rows if r["action"] == "alert_rules_update")
    assert intent["params"]["changes"]["hung_minutes"] == {"old": None, "new": 20}
    per_rule = await _call(
        app,
        "PUT",
        "/alerts/rules",
        {"rules": {"run_hung": {"hung_minutes": 25}}, **REASON},
    )
    assert per_rule.status_code == 200 and store.kv["thresholds"]["hung_minutes"] == 25
    for change, code in (
        ({"nope": 1}, "unknown_threshold"),
        ({"hung_minutes": 0}, "threshold_out_of_bounds"),
        ({"hung_minutes": 60}, "threshold_order"),
    ):
        response = await _call(
            app, "PUT", "/alerts/rules", {"thresholds": change, **REASON}
        )
        assert response.status_code == 400 and response.json()["detail"] == code
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    refused = await _call(
        app, "PUT", "/alerts/rules", {"thresholds": {"hung_minutes": 30}, **REASON}
    )
    assert refused.status_code == 403 and refused.json()["detail"] == "not_operator"
    assert store.kv["thresholds"]["hung_minutes"] == 25


async def test_rate_limit_applies_to_alert_actions():
    app = _app(seeded_store())
    codes = []
    for _ in range(11):
        response = await _call(
            app, "POST", "/alerts/rules/pg_probe/mute", {"minutes": 60, **REASON}
        )
        codes.append(response.status_code)
    assert codes[:10] == [200] * 10 and codes[10] == 429


async def test_overview_attention_links_to_alerts():
    store = seeded_store()
    app = _app(store)
    service = app.state.ops.service
    ev = AlertEvaluator(service, process=lambda: {"mode": "uvicorn"})

    async def fake(process, holder):
        return snapshot(item("disk_free", "critical"))

    service.compute_attention_snapshot = fake
    await ev.cycle(Loop())
    overview = (await _call(app, "GET", "/overview")).json()
    assert overview["attention_source"] == "snapshot"
    [disk] = [a for a in overview["attention"] if a["rule"] == "disk_free"]
    row = next(r for r in store.alert_rows if r["rule"] == "disk_free")
    assert disk["alert_id"] == row["id"]
    assert overview["alerts"]["open"] == 2


async def test_status_reports_alert_features(monkeypatch):
    app = _app(seeded_store())
    features = (await _call(app, "GET", "/status")).json()["features"]
    assert features["alerts"] is True and features["alert_email"] is False
    monkeypatch.setenv("OPS_ALERT_EMAIL", "true")
    monkeypatch.setenv("OPS_ALERTS_ENABLED", "false")
    features = (await _call(app, "GET", "/status")).json()["features"]
    assert features["alerts"] is False and features["alert_email"] is True


def test_new_routes_have_their_guards():
    names = {}
    for route in ops_routes.router.routes:
        for method in route.methods:
            names[(method, route.path)] = {
                getattr(d.call, "__name__", "") for d in route.dependant.dependencies
            }
    for key in (
        ("POST", BASE + "/alerts/{alert_id}/ack"),
        ("POST", BASE + "/alerts/rules/{rule}/mute"),
        ("POST", BASE + "/maintenance"),
        ("POST", BASE + "/maintenance/end"),
        ("PUT", BASE + "/alerts/rules"),
        ("POST", BASE + "/activity/export"),
        ("POST", BASE + "/logs/capture"),
    ):
        assert {"require_live_admin", "require_ops_client"} <= names[key], key
    for key in (
        ("PUT", BASE + "/alerts/rules"),
        ("POST", BASE + "/activity/export"),
        ("POST", BASE + "/logs/capture"),
    ):
        assert "require_operator" in names[key], key
    for path in ("/alerts", "/alerts/rules", "/alerts/{alert_id}", "/logs/capture"):
        assert ("GET", BASE + path) in names, path


# -- the incident export -------------------------------------------------------------------------


async def test_activity_export_is_redacted_capped_and_audited(monkeypatch):
    store = seeded_store()
    store.alert_rows[0]["title"] = "redacted at write time"
    store.audit_rows.append(
        {
            "id": 2,
            "ts": datetime.now(timezone.utc) - timedelta(minutes=5),
            "actor": "khairil",
            "action": "reveal",
            "workspace": "fsktm",
            "target": "doc-1",
            "phase": "result",
            "outcome": "ok",
            "params": {},
            "reason": "checking",
        }
    )
    app = _app(store)
    monkeypatch.setattr(
        ops_alerts, "EXPORT_LIMITS", {"alerts": 1, "audit": 1, "runs": 1}
    )
    response = await _call(app, "POST", "/activity/export", REASON)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines()]
    meta = lines[0]
    assert meta["type"] == "meta" and meta["log_text"] is False
    assert "audit" in meta["truncated"]
    kinds = {line["type"] for line in lines[1:]}
    assert {"alert", "audit", "event"} <= kinds
    assert sum(1 for line in lines if line["type"] == "audit") == 1
    text = response.text
    for secret in (SECRET_FILE, SECRET_EMAIL, SECRET_KEY, '"secret"'):
        assert secret not in text, secret
    exports = [r for r in store.audit_rows if r["action"] == "activity_export"]
    assert [r["phase"] for r in exports] == ["intent", "result"]
    assert exports[1]["params"]["truncated"] == meta["truncated"]
    assert response.headers["x-ops-audit-id"] == str(exports[0]["id"])


async def test_activity_export_window_and_operator(monkeypatch):
    app = _app(seeded_store())
    now = datetime.now(timezone.utc)
    wide = {
        "from": (now - timedelta(days=40)).isoformat(),
        "to": now.isoformat(),
        **REASON,
    }
    response = await _call(app, "POST", "/activity/export", wide)
    assert response.status_code == 400 and response.json()["detail"] == "invalid_range"
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    response = await _call(app, "POST", "/activity/export", REASON)
    assert response.status_code == 403


def test_activity_export_includes_runs_in_the_window():
    from fastapi.testclient import TestClient

    from tests.ops_fixtures import seed_runs

    store = seeded_store()
    seed_runs(store)
    app = _app(store)
    app.state.ops.service.run_tracking = True
    response = TestClient(app).post(
        BASE + "/activity/export", json=REASON, headers=CLIENT
    )
    assert response.status_code == 200, response.text
    runs = [json.loads(x) for x in response.text.splitlines()]
    runs = [x for x in runs if x["type"] == "run"]
    assert runs and all("label" not in r and "error_summary" not in r for r in runs)
    assert all(r["started_at"] for r in runs)


# -- INFO capture (FR-26) ------------------------------------------------------------------------


def _record(
    module: str, level: int = logging.INFO, msg: str = "step done"
) -> logging.LogRecord:
    record = logging.LogRecord(
        "lightrag", level, f"/app/lightrag/{module}.py", 10, msg, (), None
    )
    record.module = module
    return record


def _sink(store: MemoryOpsStore, clock: Clock, workers: int = 1):
    from lightrag.api.ops.log_sink import OpsLogSink

    return OpsLogSink(store, "a" * 32, clock=clock, workers=workers)


async def test_capture_takes_one_module_and_expires_by_itself():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    sink = _sink(store, clock)
    handler = sink.handler
    assert handler.level == logging.WARNING
    handler.handle(_record("pipeline"))
    assert len(handler.buffer) == 0
    sink.apply_capture(
        {"module": "pipeline", "until": clock.now + 600, "id": "c1", "actor": "root"}
    )
    assert handler.level == logging.INFO
    handler.handle(_record("pipeline", msg=f"parsed {SECRET_FILE} for {SECRET_EMAIL}"))
    handler.handle(_record("operate"))
    handler.handle(_record("pipeline", logging.DEBUG))
    assert len(handler.buffer) == 1
    row = handler.buffer[0]
    assert row["level"] == logging.INFO and row["module"] == "pipeline"
    assert SECRET_FILE not in row["msg"] and SECRET_EMAIL not in row["msg"]
    # WARNING+ of every module is still taken.
    handler.handle(_record("operate", logging.WARNING, "warn"))
    assert len(handler.buffer) == 2
    clock.advance(601)
    handler.handle(_record("pipeline"))
    assert len(handler.buffer) == 2
    assert handler.capture is None and handler.level == logging.WARNING


async def test_capture_is_seen_by_every_worker_through_kv():
    from lightrag.api.ops.log_sink import CAPTURE_KEY

    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    a, b = _sink(store, clock, workers=2), _sink(store, clock, workers=2)
    store.kv[CAPTURE_KEY] = {"module": "operate", "until": clock.now + 300, "id": "x"}
    await a.poll_capture()
    await b.poll_capture()
    for sink in (a, b):
        sink.handler.handle(_record("operate"))
        assert len(sink.handler.buffer) == 1
    store.kv[CAPTURE_KEY] = {"module": None, "until": 0}
    await b.poll_capture()
    b.handler.handle(_record("operate"))
    assert len(b.handler.buffer) == 1 and b.handler.capture is None
    # An unreadable store keeps the capture, which still ends on time.
    store.fail_reads = RuntimeError("db down")
    await a.poll_capture()
    assert a.handler.capture is not None
    clock.advance(301)
    a.handler.handle(_record("operate"))
    assert a.handler.capture is None and len(a.handler.buffer) == 1


async def test_capture_volume_is_capped():
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    sink = _sink(store, clock, workers=4)
    assert sink.handler.capture_cap == 20000 // 4
    sink.handler.capture_cap = 3
    sink.apply_capture({"module": "pipeline", "until": clock.now + 60, "id": "c"})
    for i in range(10):
        sink.handler.handle(_record("pipeline", msg=f"line {i}"))
    assert sink.handler.capture_rows == 3 and sink.handler.capture_dropped == 7
    assert len(sink.handler.buffer) == 3
    assert sink.stats()["capture"]["module"] == "pipeline"
    # A new capture starts a new count.
    sink.apply_capture({"module": "pipeline", "until": clock.now + 60, "id": "d"})
    assert sink.handler.capture_rows == 0


async def test_capture_route_is_operator_only_audited_and_needs_the_sink(monkeypatch):
    from lightrag.api.ops.log_sink import CAPTURE_KEY

    store = seeded_store()
    app = _app(store)
    body = {"module": "pipeline", "minutes": 30, **REASON}
    response = await _call(app, "POST", "/logs/capture", body)
    assert response.status_code == 409 and response.json()["detail"] == "log_sink_off"
    clock = Clock()
    sink = _sink(store, clock)
    app.state.ops.log_sink = sink
    bad = await _call(app, "POST", "/logs/capture", {**body, "module": "httpx"})
    assert bad.status_code == 400 and bad.json()["detail"] == "invalid_module"
    long = await _call(app, "POST", "/logs/capture", {**body, "minutes": 61})
    assert long.status_code == 422
    started = await _call(app, "POST", "/logs/capture", body)
    assert started.status_code == 200, started.text
    out = started.json()
    assert out["result"] == "started" and out["expires_at"] and out["audit_id"]
    assert store.kv[CAPTURE_KEY]["module"] == "pipeline"
    assert sink.handler.capture is not None
    state = (await _call(app, "GET", "/logs/capture")).json()
    assert state["active"]["module"] == "pipeline" and state["active"]["by"] == "root"
    assert "pipeline" in state["modules"] and state["max_minutes"] == 60
    logs = (await _call(app, "GET", "/logs")).json()
    assert logs["sink"]["capture"]["module"] == "pipeline"
    assert logs["sink"]["level"] == "WARNING"
    stopped = await _call(app, "POST", "/logs/capture", {"minutes": 0, **REASON})
    assert stopped.json()["result"] == "stopped" and sink.handler.capture is None
    actions = [
        (r["action"], r["phase"]) for r in store.audit_rows if "capture" in r["action"]
    ]
    assert actions == [
        ("log_capture_start", "intent"),
        ("log_capture_start", "result"),
        ("log_capture_stop", "intent"),
        ("log_capture_stop", "result"),
    ]
    monkeypatch.setenv("OPS_OPERATOR_ACCOUNTS", "someone.else")
    refused = await _call(app, "POST", "/logs/capture", body)
    assert refused.status_code == 403 and refused.json()["detail"] == "not_operator"


# -- review fixes: holds, deploy window, claims, timeouts, flapping, banner ------------


async def test_a_failed_section_holds_only_the_rules_that_read_it():
    store = MemoryOpsStore()
    ev = evaluator(store)
    await run(ev, item("stale_docs", "warning"), item("disk_free"))
    # The optional dup/cancelled counts and the KPI tiles feed no rule.
    for _ in range(2):
        await run(ev, failed_sections=["backlog_counts", "active"])
    assert not open_rows(store)
    await run(
        ev,
        item("stale_docs", "warning"),
        item("oldest_pending", "warning"),
        item("disk_free"),
    )
    for _ in range(3):
        await run(ev, failed_sections=["stuck"])
    assert {r["rule"] for r in open_rows(store)} == {"stale_docs"}


def test_every_rule_section_is_one_the_snapshot_can_report():
    gathered = {
        "backlog",
        "backlog_counts",
        "active",
        "stuck",
        "pauses",
        "stuck_runs",
        "email_queue",
        "events",
        "workers",
        "email_ingest",
        "failure_spike",
    }
    for rule, sections in ops_alerts.RULE_SECTIONS.items():
        assert rule in ALERT_RULES
        assert sections <= gathered - {"backlog_counts", "active"}


async def test_unreadable_email_ingest_and_failure_counts_are_failed_sections():
    store = MemoryOpsStore()
    service = service_for(store)
    service.run_tracking = True

    async def broken(*_args):
        raise OpsDataUnavailable("timeout")

    service._email_ingest = broken
    service._failure_spike_inputs = broken
    snap = await service.compute_attention_snapshot(
        {"mode": "uvicorn", "expected": 1, "live": 1}, "w"
    )
    assert {"email_ingest", "failure_spike"} <= set(snap["failed_sections"])


async def test_the_deploy_window_fires_no_new_heartbeat_alert(email_on):
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    out = await run(
        ev,
        item("manager_rtt"),
        item("worker_unresponsive", "warning"),
        deploy_recent=True,
    )
    assert store.alert_rows == [] and await ev.notify(out) == 0
    # A row opened before the deploy is still touched, not resolved.
    await run(ev, item("run_orphaned"))
    await run(ev, item("run_orphaned"), item("manager_rtt"), deploy_recent=True)
    assert {r["rule"] for r in open_rows(store)} == {"run_orphaned"}
    # After the window the rule fires as usual.
    await run(ev, item("run_orphaned"), item("manager_rtt"))
    assert {r["rule"] for r in open_rows(store)} == {"run_orphaned", "manager_rtt"}
    assert mail.sent == []


async def test_a_failing_claim_undoes_the_claims_already_taken(email_on):
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    outcome = await run(ev, item("disk_free"), item("pg_probe"))
    real = store.alert_claim_email
    calls: list[int] = []

    async def flaky(alert_id, kind, repeat_s):
        calls.append(alert_id)
        if len(calls) == 2:
            raise OpsDataUnavailable("statement timeout")
        return await real(alert_id, kind, repeat_s)

    store.alert_claim_email = flaky
    with pytest.raises(OpsDataUnavailable):
        await ev.notify(outcome)
    assert len(calls) == 2
    assert all(r["last_emailed_at"] is None for r in store.alert_rows)
    store.alert_claim_email = real
    assert await ev.notify(await run(ev, item("disk_free"), item("pg_probe"))) == 2
    assert len(mail.sent) == 1


async def test_reads_come_before_any_claim_and_a_build_error_unclaims(
    email_on, monkeypatch
):
    store = MemoryOpsStore()
    mail = Mail()
    ev = evaluator(store, mail=mail)
    outcome = await run(ev, item("pg_probe"))
    real_page = store.alerts_page

    async def broken_page(*_args):
        raise OpsDataUnavailable("timeout")

    store.alerts_page = broken_page
    with pytest.raises(OpsDataUnavailable):
        await ev.notify(outcome)
    assert store.alert_rows[0]["last_emailed_at"] is None
    store.alerts_page = real_page

    def broken_build(*_args, **_kwargs):
        raise ValueError("bad header")

    monkeypatch.setattr(ops_alerts, "build_message", broken_build)
    with pytest.raises(ValueError):
        await ev.notify(outcome)
    assert store.alert_rows[0]["last_emailed_at"] is None
    assert mail.sent == []


class SlowMail(Mail):
    """A relay that answers slowly: the send outlasts its budget."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def __call__(self, settings, recipients, msg) -> None:
        self.release.wait(5)
        super().__call__(settings, recipients, msg)


async def _until_idle(ev: AlertEvaluator) -> None:
    for _ in range(200):
        if not ev.mailer.busy():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the send thread did not finish")


async def test_a_send_that_outlasts_its_budget_is_never_sent_twice(
    email_on, monkeypatch
):
    monkeypatch.setattr(ops_alerts, "EMAIL_SEND_MIN_TIMEOUT_S", 0.05)
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = SlowMail()
    ev = evaluator(store, clock, mail, smtp(timeout=0.01))
    try:
        assert await ev.notify(await run(ev, item("pg_probe"))) == 0
        # Outcome unknown: the claim stays, the failure is counted.
        assert store.alert_rows[0]["last_emailed_at"] is not None
        state = store.kv[ops_alerts.EMAIL_STATE_KEY]
        assert state["consecutive_failures"] == 1 and state["uncertain_total"] == 1
        assert state["last_error_class"] == "timeout"
        # While the thread runs nothing new is claimed, backoff over or not.
        clock.advance(ops_alerts.EMAIL_BACKOFF_BASE_S + 1)
        out = await run(ev, item("pg_probe"), item("disk_free"))
        assert await ev.notify(out) == 0
        disk = next(r for r in store.alert_rows if r["rule"] == "disk_free")
        assert disk["last_emailed_at"] is None
        # The relay delivers after all: counted as sent, banner cleared, and
        # only the new alert goes out next.
        mail.release.set()
        await _until_idle(ev)
        assert await ev.notify(await run(ev, item("pg_probe"), item("disk_free"))) == 1
        assert len(mail.sent) == 2
        body = mail.sent[1][1].get_content()
        assert "disk_free" in body and "pg_probe" not in body
        state = store.kv[ops_alerts.EMAIL_STATE_KEY]
        assert state["consecutive_failures"] == 0 and state["sent_total"] == 2
    finally:
        mail.release.set()
        ev.close()


async def test_a_timed_out_send_that_then_fails_is_retried(email_on, monkeypatch):
    monkeypatch.setattr(ops_alerts, "EMAIL_SEND_MIN_TIMEOUT_S", 0.05)
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = SlowMail()
    mail.fail = smtplib.SMTPServerDisconnected("gone")
    ev = evaluator(store, clock, mail, smtp(timeout=0.01))
    try:
        assert await ev.notify(await run(ev, item("pg_probe"))) == 0
        mail.release.set()
        await _until_idle(ev)
        mail.fail = None
        clock.advance(ops_alerts.EMAIL_BACKOFF_BASE_S + 1)
        # Settled as not delivered: the claim is undone and the alert sent.
        assert await ev.notify(await run(ev, item("pg_probe"))) == 1
        assert len(mail.sent) == 1
    finally:
        mail.release.set()
        ev.close()


async def test_a_socket_timeout_inside_the_send_is_a_plain_failure(email_on):
    store = MemoryOpsStore()
    ev = evaluator(store, mail=Mail(fail=TimeoutError("timed out")))
    assert await ev.notify(await run(ev, item("pg_probe"))) == 0
    assert store.alert_rows[0]["last_emailed_at"] is None
    assert not ev.mailer.busy()
    state = store.kv[ops_alerts.EMAIL_STATE_KEY]
    assert state["consecutive_failures"] == 1 and not state.get("uncertain_total")


async def test_a_flapping_condition_is_not_emailed_on_every_flap(email_on):
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = Mail()
    ev = evaluator(store, clock, mail)
    dep = item("dependency_down", id="dependency_down:neo4j")
    # Down for 3 probes, up for 2, for an hour.
    for minute in range(60):
        present = minute % 5 < 3
        await ev.notify(await run(ev, *([dep] if present else [])))
        clock.advance(60)
    assert len(store.alert_rows) >= 10  # every flap is still recorded
    assert len(mail.sent) == 1
    assert "Opened" in mail.sent[0][1].get_content()
    # Then it really stops: one resolve email, after the quiet window.
    for _ in range(15):
        await ev.notify(await run(ev))
        clock.advance(60)
    assert len(mail.sent) == 2
    assert mail.sent[1][1]["Subject"].endswith("RESOLVED dependency_down")


async def test_a_refire_after_a_resolve_email_waits_out_the_cooldown(email_on):
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = Mail()
    ev = evaluator(store, clock, mail)
    await ev.notify(await run(ev, item("pg_probe")))
    clock.advance(60)
    await run(ev)
    clock.advance(60)
    await ev.notify(await run(ev))  # resolved
    # One second past the window: stored times round to the microsecond.
    clock.advance(ops_alerts.RESOLVE_EMAIL_QUIET_S + 1)
    assert await ev.notify(await run(ev)) == 1  # the resolve email
    told = clock.now
    clock.advance(180)
    assert await ev.notify(await run(ev, item("pg_probe"))) == 0
    while clock.now - told < ops_alerts.FLAP_COOLDOWN_S - 60:
        clock.advance(60)
        assert await ev.notify(await run(ev, item("pg_probe"))) == 0
    # One second past the cooldown: stored times round to the microsecond.
    clock.advance(61)
    assert await ev.notify(await run(ev, item("pg_probe"))) == 1
    assert len(mail.sent) == 3 and "Opened" in mail.sent[2][1].get_content()


async def test_the_banner_clears_when_nothing_is_left_to_retry(email_on):
    clock = Clock()
    store = MemoryOpsStore(clock=clock)
    mail = Mail(fail=smtplib.SMTPServerDisconnected("blip"))
    ev = evaluator(store, clock, mail)
    await ev.notify(await run(ev, item("pg_probe")))
    assert store.kv[ops_alerts.EMAIL_STATE_KEY]["consecutive_failures"] == 1
    mail.fail = None
    # The alert resolves before any retry; nothing is left to send.
    await run(ev)
    out = await run(ev)
    assert out.resolved
    clock.advance(ops_alerts.EMAIL_BACKOFF_BASE_S + 1)
    assert await ev.notify(out) == 0
    # The SMTP check has not passed since the failure: still failing.
    assert store.kv[ops_alerts.EMAIL_STATE_KEY]["consecutive_failures"] == 1
    store.kv["probe:smtp"] = {"status": "ok", "last_ok_epoch": clock.now}
    assert await ev.notify(await run(ev)) == 0
    state = store.kv[ops_alerts.EMAIL_STATE_KEY]
    assert state["consecutive_failures"] == 0 and state["retry_after"] is None
    assert state["failures_total"] == 1 and mail.sent == []
    channel = await ops_alerts.channel_state(ev.service)
    assert channel["reason"] != "send_failed"


async def test_clear_send_failures_and_smtp_recovered():
    store = MemoryOpsStore()
    service = service_for(store)
    assert not await ops_alerts.clear_send_failures(store, service)
    store.kv[ops_alerts.EMAIL_STATE_KEY] = {
        "consecutive_failures": 3,
        "failures_total": 3,
        "last_failure_at": 100.0,
        "retry_after": 2000.0,
    }
    state = store.kv[ops_alerts.EMAIL_STATE_KEY]
    assert not ops_alerts.smtp_recovered(
        state, {"status": "down", "last_ok_epoch": 200}
    )
    assert not ops_alerts.smtp_recovered(state, {"status": "ok", "last_ok_epoch": 50})
    assert ops_alerts.smtp_recovered(state, {"status": "ok", "last_ok_epoch": 200})
    assert await ops_alerts.clear_send_failures(store, service)
    cleared = store.kv[ops_alerts.EMAIL_STATE_KEY]
    assert cleared["consecutive_failures"] == 0 and cleared["failures_total"] == 3


def test_failure_spike_share_counts_failures_inside_processed():
    service = service_for(MemoryOpsStore())
    # docs_done counts failed documents: 40 of 100 processed is 40 %.
    over = _rules(
        service, failure_spike={"last_hour": 40, "median": 5.0, "processed": 100}
    )["failure_spike"]
    assert over["severity"] == "serious" and over["params"]["share"] == 0.4
    at = _rules(
        service, failure_spike={"last_hour": 30, "median": 5.0, "processed": 100}
    )["failure_spike"]
    assert at["severity"] == "warning"
    # Failures no run counted (processed 0) are all of the hour's documents.
    none = _rules(
        service, failure_spike={"last_hour": 25, "median": 1.0, "processed": 0}
    )["failure_spike"]
    assert none["severity"] == "serious" and none["params"]["share"] == 1.0


# -- unauth_warning_lines_5m (R24) -------------------------------------------------------


async def test_warnings_of_unauthenticated_requests_are_counted():
    from lightrag.api.ops.context import (
        RequestIdMiddleware,
        mark_request_authenticated,
        request_unauthenticated,
    )
    from lightrag.api.ops.log_sink import OpsLogHandler

    handler = OpsLogHandler(level=logging.WARNING)
    log = logging.getLogger("lightrag.test_unauth")
    log.addHandler(handler)
    log.propagate = False

    async def app(scope, receive, send):
        if scope["path"].startswith("/authed"):
            mark_request_authenticated()
        log.warning("something odd on %s", scope["path"])
        log.info("not counted")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestIdMiddleware(app)

    async def call(path: str) -> None:
        async def receive():
            return {"type": "http.request"}

        async def send(_message):
            return None

        scope = {"type": "http", "path": path, "headers": [], "client": None}
        await middleware(scope, receive, send)

    try:
        assert not request_unauthenticated()  # outside a request
        for _ in range(3):
            await call("/health")
        await call("/authed/documents")
        log.warning("background work")
        assert handler.unauth_warnings == 3
    finally:
        log.removeHandler(handler)
        log.propagate = True


def test_unauth_warning_rule_sums_workers_over_its_threshold():
    from lightrag.api.ops.health import UNAUTH_WARNINGS_WARN

    service = service_for(MemoryOpsStore())

    def workers(*counts: int) -> list[dict[str, Any]]:
        return [
            {
                "alive": True,
                "pid": i,
                "snapshot": {"log_sink": {"unauth_warnings_5m": c}},
            }
            for i, c in enumerate(counts)
        ]

    quiet = _rules(service, worker_items=workers(UNAUTH_WARNINGS_WARN))
    assert "unauth_warning_lines_5m" not in quiet
    loud = _rules(service, worker_items=workers(30, 30))["unauth_warning_lines_5m"]
    assert loud["severity"] == "warning" and loud["params"]["lines_5m"] == 60
    assert loud["runbook"] and not ops_alerts.emailable(
        "unauth_warning_lines_5m", "warning"
    )
