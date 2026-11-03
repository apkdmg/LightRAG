"""Health and worker read models: dependency cards, worker rows, their rules.

Pure functions over what the probes and heartbeats stored, shared by the
Health tab routes, the Overview and the leased attention snapshot
(:mod:`lightrag.api.ops.reconcile`). Thresholds follow §1.7 of
``docs/ADMIN_OPERATIONS_DASHBOARD.md``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlencode

from lightrag.api.ops.heartbeat import ALIVE_WITHIN_S
from lightrag.api.ops.probes import (
    PROBE_INTERVAL_S,
    STRIP_SLOT_S,
    STRIP_SLOTS,
    TLS_DOWN_DAYS,
    TLS_WARN_DAYS,
    VLLM_WAITING_WARN,
    ProbeSpec,
)

# §1.7 thresholds for the rules evaluated here.
PG_SLOW_WARN_MS = 100.0
PG_SLOW_CRIT_MS = 1000.0
PG_CONNECTIONS_WARN_PCT = 70.0
PG_CONNECTIONS_SERIOUS_PCT = 90.0
PG_BLOCKED_WARN_S = 30.0
PG_BLOCKED_SERIOUS_S = 120.0
PG_POOL_WAIT_WARN_MS = 500.0
PG_POOL_WAIT_SERIOUS_MS = 5000.0
MANAGER_RTT_WARN_MS = 200.0
DEP_SLOW_MS = 500.0
DEP_CONSECUTIVE = 3
PG_CONSECUTIVE = 2
SMTP_CONSECUTIVE = 2
VLLM_WAITING_SERIOUS = 100
VLLM_WAITING_FOR_S = 120.0
DISK_WARN_FREE_PCT = 15.0
DISK_CRIT_FREE_PCT = 5.0
DROPS_SERIOUS = 1000
RECORDER_FLUSH_STALE_S = 60.0
WORKERS_CONSECUTIVE = 2
# pg_pool_wait, pg_connections and manager_rtt hold for 2 evaluations.
RESOURCE_CONSECUTIVE = 2
DEPLOY_SUPPRESS_S = 600.0
# A fresh process waits for its siblings' first heartbeats before the worker
# count can be judged.
WORKER_RULE_GRACE_S = 90.0
# Dependencies whose outage is the dependency_down rule (the LLM provider has
# its own error-rate rule; Postgres, SMTP, TLS and disk have theirs).
DOWN_RULE_KINDS = frozenset(
    {"graph", "vector", "kv", "embed", "rerank", "parser", "idp"}
)
MAX_WORKER_ITEMS = 50


def _iso(epoch: Optional[float]) -> Optional[str]:
    if not isinstance(epoch, (int, float)) or epoch <= 0:
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
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    return None


def health_link(**params: Any) -> str:
    query = {"ops": "1", "otab": "health"}
    query.update({k: v for k, v in params.items() if v is not None})
    return "#/?" + urlencode(query)


# -- availability strip ------------------------------------------------------------


def strip_cell_status(ok: int, warn: int, down: int, total: int) -> str:
    """One half-hour cell: unknown without results, down when half failed,
    warn on any failure or slow result, else ok."""
    if total <= 0:
        return "unknown"
    if down * 2 >= total:
        return "down"
    if down or warn:
        return "warn"
    return "ok"


def build_strips(
    rows: Iterable[dict[str, Any]],
    now: float,
    *,
    slots: int = STRIP_SLOTS,
    slot_s: int = STRIP_SLOT_S,
) -> dict[str, list[dict[str, Any]]]:
    """48 half-hour cells per dependency, oldest first, ending with the
    current slot."""
    current = int(now // slot_s)
    first = current - slots + 1
    counts: dict[str, dict[int, dict[str, int]]] = {}
    for row in rows:
        slot = int(row.get("slot") or 0)
        if slot < first or slot > current:
            continue
        counts.setdefault(row["dependency"], {})[slot] = {
            k: int(row.get(k) or 0) for k in ("ok", "warn", "down", "total")
        }
    strips: dict[str, list[dict[str, Any]]] = {}
    for dependency, by_slot in counts.items():
        cells = []
        for slot in range(first, current + 1):
            c = by_slot.get(slot) or {"ok": 0, "warn": 0, "down": 0, "total": 0}
            cells.append(
                {
                    "start": slot * slot_s * 1000,
                    "status": strip_cell_status(
                        c["ok"], c["warn"], c["down"], c["total"]
                    ),
                    "total": c["total"],
                    "failed": c["down"],
                    "slow": c["warn"],
                }
            )
        strips[dependency] = cells
    return strips


# -- dependency cards ---------------------------------------------------------------


def dep_card(
    spec: ProbeSpec,
    record: Optional[dict[str, Any]],
    strip: Optional[list[dict[str, Any]]],
    now: float,
) -> dict[str, Any]:
    """The Health tab's card for one dependency (``DepHealth`` in the WebUI)."""
    card: dict[str, Any] = {
        "id": spec.id,
        "name": spec.name,
        "kind": spec.kind,
        "target": spec.target,
        "interval_s": spec.interval_s,
        "passive": spec.passive,
        "probed": spec.probed,
        "strip": strip or [],
    }
    if not spec.probed:
        card.update(
            {
                "status": "unprobed",
                "latency_ms": None,
                "checked_at": None,
                "last_ok_at": None,
                "consecutive_failures": None,
                "consecutive_slow": None,
                "error_class": None,
                "detail": {"reason": spec.unprobed_reason},
                "stale": False,
            }
        )
        return card
    record = record if isinstance(record, dict) else {}
    checked = _epoch(record.get("checked_epoch")) or _epoch(record.get("checked_at"))
    card.update(
        {
            "status": record.get("status") or "unknown",
            "latency_ms": record.get("latency_ms"),
            "checked_at": record.get("checked_at") or _iso(checked),
            "last_ok_at": record.get("last_ok_at"),
            "consecutive_failures": record.get("consecutive_failures"),
            "consecutive_slow": record.get("consecutive_slow"),
            "error_class": record.get("error_class"),
            "detail": record.get("detail") or {},
            # Probes stopped (no lease holder, or probes switched off).
            "stale": checked is None
            or now - checked > max(3 * spec.interval_s, 3 * PROBE_INTERVAL_S),
        }
    )
    return card


def _derived_card(
    dep_id: str, name: str, kind: str, status: str, detail: dict, now: float
) -> dict[str, Any]:
    return {
        "id": dep_id,
        "name": name,
        "kind": kind,
        "target": None,
        "interval_s": None,
        "passive": True,
        "probed": True,
        "strip": [],
        "status": status,
        "latency_ms": detail.get("rtt_ms"),
        "checked_at": _iso(now),
        "last_ok_at": None,
        "consecutive_failures": None,
        "consecutive_slow": None,
        "error_class": None,
        "detail": detail,
        "stale": False,
        "source": "heartbeat",
    }


def derived_deps(
    workers: list[dict[str, Any]], mode: str, now: float
) -> list[dict[str, Any]]:
    """Cards read from heartbeats: the shared-state Manager (multi-worker
    only) and the Insights recorder."""
    alive = [w for w in workers if w.get("alive")]
    cards: list[dict[str, Any]] = []
    if mode == "gunicorn" and alive:
        states = [w["snapshot"].get("manager_state") for w in alive]
        rtts = [
            w["snapshot"].get("manager_rtt_ms")
            for w in alive
            if isinstance(w["snapshot"].get("manager_rtt_ms"), (int, float))
        ]
        worst_rtt = max(rtts, default=None)
        if "unresponsive" in states:
            status = "down"
        elif worst_rtt is not None and worst_rtt > MANAGER_RTT_WARN_MS:
            status = "warn"
        elif rtts:
            status = "ok"
        else:
            status = "unknown"
        cards.append(
            _derived_card(
                "manager",
                "Shared state (Manager)",
                "internal",
                status,
                {"rtt_ms": worst_rtt, "unresponsive": states.count("unresponsive")},
                now,
            )
        )
    recorders = [w["snapshot"].get("recorder") for w in alive]
    recorders = [r for r in recorders if isinstance(r, dict)]
    if recorders:
        drops = sum(
            int(r.get("lost_5m") or 0) + int(r.get("dropped_5m") or 0)
            for r in recorders
        )
        stale = any(
            (r.get("buffer") or 0) > 0
            and isinstance(r.get("last_flush_age_s"), (int, float))
            and r["last_flush_age_s"] > RECORDER_FLUSH_STALE_S
            for r in recorders
        )
        status = "down" if stale or drops > DROPS_SERIOUS else "warn" if drops else "ok"
        cards.append(
            _derived_card(
                "recorder",
                "Insights recorder",
                "internal",
                status,
                {
                    "buffer": sum(int(r.get("buffer") or 0) for r in recorders),
                    "lost_5m": sum(int(r.get("lost_5m") or 0) for r in recorders),
                    "dropped_5m": sum(int(r.get("dropped_5m") or 0) for r in recorders),
                    "last_flush_age_s": max(
                        (
                            r["last_flush_age_s"]
                            for r in recorders
                            if isinstance(r.get("last_flush_age_s"), (int, float))
                        ),
                        default=None,
                    ),
                },
                now,
            )
        )
    return cards


def deps_summary(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """OK count over the dependencies that are checked."""
    checked = [c for c in cards if c.get("probed") and c.get("status") != "unprobed"]
    return {
        "ok": sum(1 for c in checked if c.get("status") == "ok"),
        "total": len(checked),
        "down": sum(1 for c in checked if c.get("status") == "down"),
        "warn": sum(1 for c in checked if c.get("status") == "warn"),
    }


# -- workers ---------------------------------------------------------------------------


def worker_item(row: dict[str, Any], self_id: Optional[str]) -> dict[str, Any]:
    """A ``LIGHTRAG_OPS_WORKERS`` row as the Workers table shows it."""
    age = row.get("heartbeat_age_s")
    age = float(age) if isinstance(age, (int, float)) else None
    superseded = bool(row.get("superseded"))
    exited = row.get("exited_at")
    snapshot = row.get("snapshot") if isinstance(row.get("snapshot"), dict) else {}
    return {
        "worker_id": row.get("worker_id"),
        "pid": row.get("pid"),
        "host": row.get("host"),
        "mode": row.get("mode"),
        "version": row.get("version"),
        "started_at": _as_iso(row.get("started_at")),
        "heartbeat_at": _as_iso(row.get("heartbeat_at")),
        "heartbeat_age_s": None if age is None else round(age, 1),
        "alive": age is not None
        and age < ALIVE_WITHIN_S
        and not superseded
        and exited is None,
        "superseded": superseded,
        "exited_at": _as_iso(exited),
        "leases_held": list(row.get("leases_held") or []),
        "self": bool(self_id) and row.get("worker_id") == self_id,
        "snapshot": snapshot,
    }


def order_workers(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Live workers first, then dead ones, then superseded; newest beat first."""

    def rank(item: dict[str, Any]) -> tuple:
        group = (
            0 if item["alive"] else 2 if item["superseded"] or item["exited_at"] else 1
        )
        return (
            group,
            item["heartbeat_age_s"] if item["heartbeat_age_s"] is not None else 1e12,
        )

    return sorted(items, key=rank)[:MAX_WORKER_ITEMS]


def host_info(items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Container and host numbers from the freshest live worker."""
    for item in items:
        host = item["snapshot"].get("host")
        if item["alive"] and isinstance(host, dict):
            return host
    return None


# -- rules -------------------------------------------------------------------------------


Add = Callable[..., None]


def health_attention(
    add: Add,
    *,
    deps: list[dict[str, Any]],
    workers: Optional[list[dict[str, Any]]],
    process: dict[str, Any],
    now: float,
    streaks: Optional[dict[str, int]],
    deploy_recent: bool,
    uptime_s: Optional[float],
) -> dict[str, int]:
    """Dependency and worker rules. Returns the updated streak counters.

    ``streaks`` (from the previous leased snapshot) makes a rule wait for
    consecutive evaluations; ``None`` (a per-request fallback) evaluates the
    current state only.
    """
    new_streaks: dict[str, int] = {}
    by_id = {d["id"]: d for d in deps}

    def held(rule: str, need: int, marker: Optional[int] = None) -> bool:
        """Count one more evaluation for which ``rule``'s condition holds.

        A rule not counted in an evaluation starts again from zero. With
        ``marker`` (the time of the reading), only a new reading counts, so a
        value carried between slow refreshes is not counted twice.
        """
        previous = streaks or {}
        count = int(previous.get(rule) or 0)
        if marker is None or previous.get(f"{rule}@") != marker:
            count += 1
        count = max(count, 1)
        new_streaks[rule] = count
        if marker is not None:
            new_streaks[f"{rule}@"] = marker
        return streaks is None or count >= need

    require_trusted = os.getenv("OPS_TLS_REQUIRE_TRUSTED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    pg = by_id.get("postgres")
    if pg and not pg.get("stale"):
        failures = int(pg.get("consecutive_failures") or 0)
        slow = int(pg.get("consecutive_slow") or 0)
        latency = pg.get("latency_ms")
        if pg["status"] == "down" and failures >= PG_CONSECUTIVE:
            add(
                "pg_probe",
                "critical",
                "Postgres does not answer its health check",
                health_link(odep="postgres"),
                params={"error_class": pg.get("error_class"), "failures": failures},
                since=pg.get("last_ok_at"),
            )
        elif isinstance(latency, (int, float)) and slow >= PG_CONSECUTIVE:
            if latency > PG_SLOW_CRIT_MS:
                add(
                    "pg_probe",
                    "critical",
                    f"Postgres answers in {latency:.0f} ms",
                    health_link(odep="postgres"),
                    params={"latency_ms": latency},
                )
            elif latency > PG_SLOW_WARN_MS:
                add(
                    "pg_probe",
                    "warning",
                    f"Postgres answers in {latency:.0f} ms",
                    health_link(odep="postgres"),
                    params={"latency_ms": latency},
                )
        detail = pg.get("detail") or {}
        pct = detail.get("connections_pct")
        extras_at = _epoch(detail.get("extras_at"))
        if (
            isinstance(pct, (int, float))
            and pct >= PG_CONNECTIONS_WARN_PCT
            and held(
                "pg_connections",
                RESOURCE_CONSECUTIVE,
                None if extras_at is None else int(extras_at),
            )
        ):
            add(
                "pg_connections",
                "serious" if pct >= PG_CONNECTIONS_SERIOUS_PCT else "warning",
                f"Postgres connections at {pct:.0f}% of the limit",
                health_link(odep="postgres"),
                params={
                    "connections": detail.get("connections"),
                    "max_connections": detail.get("max_connections"),
                    "pct": pct,
                },
            )
        blocked = detail.get("blocked_max_s")
        if isinstance(blocked, (int, float)) and blocked > PG_BLOCKED_WARN_S:
            add(
                "pg_blocked",
                "serious" if blocked > PG_BLOCKED_SERIOUS_S else "warning",
                f"A Postgres query has been blocked for {blocked:.0f} s",
                health_link(odep="postgres"),
                params={"blocked": detail.get("blocked"), "blocked_max_s": blocked},
            )

    for dep in deps:
        if dep.get("kind") not in DOWN_RULE_KINDS or dep.get("stale"):
            continue
        failures = int(dep.get("consecutive_failures") or 0)
        slow = int(dep.get("consecutive_slow") or 0)
        latency = dep.get("latency_ms")
        link = health_link(odep=dep["id"])
        if dep["status"] == "down" and failures >= DEP_CONSECUTIVE:
            add(
                "dependency_down",
                "critical",
                f"{dep['name']} is down",
                link,
                item_id=f"dependency_down:{dep['id']}",
                params={
                    "dependency": dep["id"],
                    "error_class": dep.get("error_class"),
                    "failures": failures,
                },
                since=dep.get("last_ok_at"),
            )
        elif (
            dep["status"] == "warn"
            and slow >= DEP_CONSECUTIVE
            and isinstance(latency, (int, float))
            and latency > DEP_SLOW_MS
        ):
            add(
                "dependency_down",
                "warning",
                f"{dep['name']} is slow ({latency:.0f} ms)",
                link,
                item_id=f"dependency_down:{dep['id']}",
                params={"dependency": dep["id"], "latency_ms": latency},
            )
        if dep["kind"] in ("embed", "rerank"):
            detail = dep.get("detail") or {}
            waiting = detail.get("waiting")
            since = _epoch(detail.get("waiting_since"))
            if (
                isinstance(waiting, int)
                and waiting > VLLM_WAITING_WARN
                and since is not None
                and now - since >= VLLM_WAITING_FOR_S
            ):
                add(
                    "vllm_waiting",
                    "serious" if waiting > VLLM_WAITING_SERIOUS else "warning",
                    f"{waiting} requests waiting at {dep['name']}",
                    link,
                    item_id=f"vllm_waiting:{dep['id']}",
                    params={"dependency": dep["id"], "waiting": waiting},
                    since=_iso(since),
                )

    smtp = by_id.get("smtp")
    if (
        smtp
        and not smtp.get("stale")
        and smtp["status"] == "down"
        and int(smtp.get("consecutive_failures") or 0) >= SMTP_CONSECUTIVE
    ):
        add(
            "smtp_probe",
            "warning",
            "Alert email cannot be delivered: the SMTP check fails",
            health_link(odep="smtp"),
            params={"error_class": smtp.get("error_class")},
            since=smtp.get("last_ok_at"),
        )

    certificates: list[tuple[str, dict[str, Any], str]] = []
    tls = by_id.get("tls")
    if tls:
        for target in (tls.get("detail") or {}).get("targets") or []:
            certificates.append(("tls", target, str(target.get("target") or "")))
    if smtp and isinstance((smtp.get("detail") or {}).get("cert_days_left"), int):
        certificates.append(
            (
                "smtp",
                {"days_left": smtp["detail"]["cert_days_left"], "trusted": True},
                "SMTP",
            )
        )
    for dep_id, target, label in certificates:
        days = target.get("days_left")
        if isinstance(days, int) and days < TLS_WARN_DAYS:
            add(
                "tls_cert_expiry",
                "critical" if days < TLS_DOWN_DAYS else "warning",
                f"TLS certificate of {label} expires in {days} days",
                health_link(odep=dep_id),
                item_id=f"tls_cert_expiry:{label}",
                params={"target": label, "days_left": days},
            )
        elif require_trusted and target.get("trusted") is False:
            add(
                "tls_cert_expiry",
                "warning",
                f"TLS certificate of {label} is not trusted",
                health_link(odep=dep_id),
                item_id=f"tls_cert_expiry:{label}",
                params={"target": label, "self_signed": target.get("self_signed")},
            )

    disk = by_id.get("disk")
    free = (disk.get("detail") or {}).get("free_pct") if disk else None
    if isinstance(free, (int, float)) and free < DISK_WARN_FREE_PCT:
        add(
            "disk_free",
            "critical" if free < DISK_CRIT_FREE_PCT else "warning",
            f"Disk {free:.0f}% free",
            health_link(odep="disk"),
            params={"free_pct": free},
        )

    if workers is None:
        return new_streaks
    alive = [w for w in workers if w.get("alive")]
    mode = process.get("mode")
    expected = process.get("expected")
    live = process.get("live")
    settled = uptime_s is None or uptime_s >= WORKER_RULE_GRACE_S
    if (
        isinstance(live, int)
        and isinstance(expected, int)
        and live < expected
        and settled
        and not deploy_recent
    ):
        streak = (streaks or {}).get("workers_below_expected", 0) + 1
        new_streaks["workers_below_expected"] = streak
        if streaks is None or streak >= WORKERS_CONSECUTIVE:
            add(
                "workers_below_expected",
                "critical" if live <= 0 else "warning",
                f"{live} of {expected} workers reporting",
                health_link(osub="workers"),
                params={"live": live, "expected": expected},
            )
    if mode == "gunicorn":
        unresponsive = [
            w for w in alive if w["snapshot"].get("manager_state") == "unresponsive"
        ]
        rtts = [
            w["snapshot"].get("manager_rtt_ms")
            for w in alive
            if isinstance(w["snapshot"].get("manager_rtt_ms"), (int, float))
        ]
        rtt_high = bool(rtts) and max(rtts) > MANAGER_RTT_WARN_MS
        if (unresponsive or rtt_high) and held("manager_rtt", RESOURCE_CONSECUTIVE):
            if unresponsive:
                add(
                    "manager_rtt",
                    "critical",
                    "Shared pipeline state is not answering",
                    health_link(osub="workers"),
                    params={"workers": [w["pid"] for w in unresponsive]},
                )
            else:
                add(
                    "manager_rtt",
                    "warning",
                    f"Shared state answers in {max(rtts):.0f} ms",
                    health_link(osub="workers"),
                    params={"rtt_ms": max(rtts)},
                )
    waits = [
        (w["pid"], w["snapshot"].get("pool"))
        for w in alive
        if isinstance(w["snapshot"].get("pool"), dict)
    ]
    waits = [
        (pid, pool["wait_ms"], bool(pool.get("saturated")))
        for pid, pool in waits
        if isinstance(pool.get("wait_ms"), (int, float))
    ]
    if waits:
        # A sample that timed out (no connection at all) ranks worst.
        pid, worst, saturated = max(waits, key=lambda pw: (pw[2], pw[1]))
        if (saturated or worst > PG_POOL_WAIT_WARN_MS) and held(
            "pg_pool_wait", RESOURCE_CONSECUTIVE
        ):
            add(
                "pg_pool_wait",
                "serious"
                if saturated or worst > PG_POOL_WAIT_SERIOUS_MS
                else "warning",
                f"Worker {pid} has no database connection after {worst:.0f} ms"
                if saturated
                else f"Worker {pid} waits {worst:.0f} ms for a database connection",
                health_link(osub="workers"),
                params={"pid": pid, "wait_ms": worst, "saturated": saturated},
            )
    recorders = [
        (w["pid"], w["snapshot"].get("recorder"))
        for w in alive
        if isinstance(w["snapshot"].get("recorder"), dict)
    ]
    if recorders:
        drops = sum(
            int(r.get("lost_5m") or 0) + int(r.get("dropped_5m") or 0)
            for _, r in recorders
        )
        stale = [
            pid
            for pid, r in recorders
            if (r.get("buffer") or 0) > 0
            and isinstance(r.get("last_flush_age_s"), (int, float))
            and r["last_flush_age_s"] > RECORDER_FLUSH_STALE_S
        ]
        if drops or stale:
            add(
                "log_sink_or_recorder_drops_5m",
                "serious" if stale or drops > DROPS_SERIOUS else "warning",
                f"Usage records dropped in the last 5 min: {drops}"
                if drops
                else "Usage records have not been written for over a minute",
                health_link(odep="recorder"),
                params={"drops_5m": drops, "stale_workers": stale},
            )
    return new_streaks
