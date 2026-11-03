"""Dependency and worker attention rules (§1.7) and the availability strip."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from lightrag.api.ops.health import (
    build_strips,
    derived_deps,
    health_attention,
    strip_cell_status,
)

pytestmark = pytest.mark.offline

NOW = 1_800_000_000.0


def _dep(dep_id, kind, status="ok", **fields):
    return {
        "id": dep_id,
        "name": dep_id.title(),
        "kind": kind,
        "status": status,
        "latency_ms": fields.pop("latency_ms", 5.0),
        "consecutive_failures": fields.pop("failures", 0),
        "consecutive_slow": fields.pop("slow", 0),
        "detail": fields.pop("detail", {}),
        "stale": fields.pop("stale", False),
        "last_ok_at": None,
        "error_class": fields.pop("error_class", None),
    }


def _worker(pid, alive=True, **snapshot):
    return {"pid": pid, "alive": alive, "snapshot": snapshot}


def _evaluate(
    deps=(), workers=None, process=None, streaks=None, deploy=False, uptime=600
):
    items = []

    def add(
        rule,
        severity,
        title,
        link,
        *,
        params=None,
        since=None,
        count=None,
        item_id=None,
    ):
        items.append(
            {
                "rule": rule,
                "id": item_id or rule,
                "severity": severity,
                "params": params,
            }
        )

    streaks_out = health_attention(
        add,
        deps=list(deps),
        workers=workers,
        process=process or {"mode": "uvicorn", "expected": 1, "live": 1},
        now=NOW,
        streaks=streaks,
        deploy_recent=deploy,
        uptime_s=uptime,
    )
    return {i["id"]: i for i in items}, streaks_out


def test_postgres_rules():
    items, _ = _evaluate([_dep("postgres", "database", "down", failures=1)])
    assert "pg_probe" not in items  # one failure is not enough
    items, _ = _evaluate([_dep("postgres", "database", "down", failures=2)])
    assert items["pg_probe"]["severity"] == "critical"
    items, _ = _evaluate([_dep("postgres", "database", "warn", latency_ms=250, slow=2)])
    assert items["pg_probe"]["severity"] == "warning"
    items, _ = _evaluate(
        [_dep("postgres", "database", "warn", latency_ms=1500, slow=2)]
    )
    assert items["pg_probe"]["severity"] == "critical"
    detail = {"connections_pct": 91.0, "blocked_max_s": 45.0, "blocked": 1}
    items, _ = _evaluate([_dep("postgres", "database", detail=detail)])
    assert items["pg_connections"]["severity"] == "serious"
    assert items["pg_blocked"]["severity"] == "warning"
    detail = {"connections_pct": 72.0, "blocked_max_s": 130.0}
    items, _ = _evaluate([_dep("postgres", "database", detail=detail)])
    assert items["pg_connections"]["severity"] == "warning"
    assert items["pg_blocked"]["severity"] == "serious"


def test_dependency_down_needs_three_failures_and_skips_stale_or_passive():
    items, _ = _evaluate([_dep("milvus", "vector", "down", failures=2)])
    assert not items
    items, _ = _evaluate([_dep("milvus", "vector", "down", failures=3)])
    assert items["dependency_down:milvus"]["severity"] == "critical"
    items, _ = _evaluate([_dep("milvus", "vector", "down", failures=9, stale=True)])
    assert not items
    items, _ = _evaluate([_dep("llm", "llm", "down", failures=9)])
    assert not items  # the LLM has its own error-rate rule
    items, _ = _evaluate([_dep("docling", "parser", "warn", latency_ms=900, slow=3)])
    assert items["dependency_down:docling"]["severity"] == "warning"


def test_vllm_waiting_for_two_minutes():
    since = datetime.fromtimestamp(NOW - 150, timezone.utc).isoformat()
    recent = datetime.fromtimestamp(NOW - 30, timezone.utc).isoformat()
    items, _ = _evaluate(
        [_dep("rerank", "rerank", detail={"waiting": 34, "waiting_since": since})]
    )
    assert items["vllm_waiting:rerank"]["severity"] == "warning"
    items, _ = _evaluate(
        [_dep("rerank", "rerank", detail={"waiting": 140, "waiting_since": since})]
    )
    assert items["vllm_waiting:rerank"]["severity"] == "serious"
    items, _ = _evaluate(
        [_dep("rerank", "rerank", detail={"waiting": 140, "waiting_since": recent})]
    )
    assert not items


def test_smtp_tls_and_disk(monkeypatch):
    monkeypatch.delenv("OPS_TLS_REQUIRE_TRUSTED", raising=False)
    items, _ = _evaluate([_dep("smtp", "mail", "down", failures=2)])
    assert items["smtp_probe"]["severity"] == "warning"
    tls = _dep(
        "tls",
        "tls",
        detail={
            "targets": [
                {"target": "proxy:443", "days_left": 5, "self_signed": True},
                {
                    "target": "rag:443",
                    "days_left": 40,
                    "self_signed": True,
                    "trusted": False,
                },
            ]
        },
    )
    items, _ = _evaluate([tls])
    assert items["tls_cert_expiry:proxy:443"]["severity"] == "critical"
    assert "tls_cert_expiry:rag:443" not in items  # self-signed is not alerted
    monkeypatch.setenv("OPS_TLS_REQUIRE_TRUSTED", "true")
    items, _ = _evaluate([tls])
    assert items["tls_cert_expiry:rag:443"]["severity"] == "warning"
    smtp_cert = _dep("smtp", "mail", detail={"cert_days_left": 15})
    items, _ = _evaluate([smtp_cert])
    assert items["tls_cert_expiry:SMTP"]["severity"] == "warning"
    items, _ = _evaluate([_dep("disk", "disk", detail={"free_pct": 4.0})])
    assert items["disk_free"]["severity"] == "critical"
    items, _ = _evaluate([_dep("disk", "disk", detail={"free_pct": 12.0})])
    assert items["disk_free"]["severity"] == "warning"


def test_workers_below_expected_needs_two_evaluations_and_respects_deploys():
    process = {"mode": "gunicorn", "expected": 3, "live": 2}
    workers = [_worker(1), _worker(2)]
    items, streaks = _evaluate(workers=workers, process=process, streaks={})
    assert not items and streaks == {"workers_below_expected": 1}
    items, streaks = _evaluate(workers=workers, process=process, streaks=streaks)
    assert items["workers_below_expected"]["severity"] == "warning"
    items, _ = _evaluate(
        workers=workers,
        process={**process, "live": 0},
        streaks={"workers_below_expected": 5},
    )
    assert items["workers_below_expected"]["severity"] == "critical"
    items, streaks = _evaluate(
        workers=workers, process=process, streaks={}, deploy=True
    )
    assert not items and streaks == {}
    items, _ = _evaluate(workers=workers, process=process, streaks=None, uptime=20)
    assert not items  # siblings have not reported yet
    # The per-request fallback (no streaks) judges the current state.
    items, _ = _evaluate(workers=workers, process=process, streaks=None)
    assert "workers_below_expected" in items


def test_worker_snapshot_rules():
    workers = [
        _worker(
            1,
            manager_state="ok",
            manager_rtt_ms=250.0,
            pool={"wait_ms": 700.0},
            recorder={"lost_5m": 3, "dropped_5m": 0, "buffer": 0},
        ),
        _worker(2, alive=False, manager_state="unresponsive", pool={"wait_ms": 9000.0}),
    ]
    items, _ = _evaluate(
        workers=workers, process={"mode": "gunicorn", "expected": 1, "live": 1}
    )
    assert items["manager_rtt"]["severity"] == "warning"  # dead workers ignored
    assert items["pg_pool_wait"]["severity"] == "warning"
    assert items["log_sink_or_recorder_drops_5m"]["severity"] == "warning"
    workers[0]["snapshot"]["manager_state"] = "unresponsive"
    workers[0]["snapshot"]["recorder"] = {"buffer": 10, "last_flush_age_s": 120.0}
    items, _ = _evaluate(
        workers=workers, process={"mode": "gunicorn", "expected": 1, "live": 1}
    )
    assert items["manager_rtt"]["severity"] == "critical"
    assert items["log_sink_or_recorder_drops_5m"]["severity"] == "serious"
    # Single process: no Manager rule.
    items, _ = _evaluate(workers=workers)
    assert "manager_rtt" not in items


def test_resource_rules_need_two_consecutive_evaluations():
    process = {"mode": "gunicorn", "expected": 1, "live": 1}
    workers = [_worker(1, manager_rtt_ms=250.0, pool={"wait_ms": 700.0})]
    items, streaks = _evaluate(workers=workers, process=process, streaks={})
    assert "pg_pool_wait" not in items and "manager_rtt" not in items
    assert streaks["pg_pool_wait"] == 1 and streaks["manager_rtt"] == 1
    items, streaks = _evaluate(workers=workers, process=process, streaks=streaks)
    assert items["pg_pool_wait"]["severity"] == "warning"
    assert items["manager_rtt"]["severity"] == "warning"
    # One good evaluation resets the count.
    calm = [_worker(1, manager_rtt_ms=5.0, pool={"wait_ms": 3.0})]
    _, streaks = _evaluate(workers=calm, process=process, streaks=streaks)
    assert "pg_pool_wait" not in streaks and "manager_rtt" not in streaks
    items, _ = _evaluate(workers=workers, process=process, streaks=streaks)
    assert "pg_pool_wait" not in items


def test_pg_connections_needs_two_extras_refreshes():
    def pg(at):
        return _dep(
            "postgres",
            "database",
            detail={
                "connections_pct": 75.0,
                "extras_at": datetime.fromtimestamp(at, timezone.utc).isoformat(),
            },
        )

    items, streaks = _evaluate([pg(NOW - 60)], streaks={})
    assert "pg_connections" not in items
    # The same reading carried into the next evaluation does not count twice.
    items, streaks = _evaluate([pg(NOW - 60)], streaks=streaks)
    assert "pg_connections" not in items and streaks["pg_connections"] == 1
    items, streaks = _evaluate([pg(NOW + 240)], streaks=streaks)
    assert items["pg_connections"]["severity"] == "warning"


def test_a_saturated_pool_is_serious():
    workers = [
        _worker(1, pool={"wait_ms": 600.0, "saturated": False}),
        _worker(2, pool={"wait_ms": 6000.0, "saturated": True}),
    ]
    items, _ = _evaluate(workers=workers, streaks={"pg_pool_wait": 1})
    assert items["pg_pool_wait"]["severity"] == "serious"
    assert items["pg_pool_wait"]["params"]["pid"] == 2
    assert items["pg_pool_wait"]["params"]["saturated"] is True
    # Even a sample that timed out below the threshold (older heartbeats).
    workers = [_worker(1, pool={"wait_ms": 2000.0, "saturated": True})]
    items, _ = _evaluate(workers=workers, streaks={"pg_pool_wait": 1})
    assert items["pg_pool_wait"]["severity"] == "serious"


def test_derived_cards():
    workers = [
        _worker(1, manager_state="ok", manager_rtt_ms=1.2, recorder={"buffer": 0}),
        _worker(2, manager_state="ok", manager_rtt_ms=3.0, recorder={"lost_5m": 2}),
    ]
    cards = {c["id"]: c for c in derived_deps(workers, "gunicorn", NOW)}
    assert cards["manager"]["status"] == "ok" and cards["manager"]["latency_ms"] == 3.0
    assert cards["recorder"]["status"] == "warn"
    assert {c["id"] for c in derived_deps(workers, "uvicorn", NOW)} == {"recorder"}
    assert derived_deps([], "gunicorn", NOW) == []


def test_strip_cells():
    assert strip_cell_status(0, 0, 0, 0) == "unknown"
    assert strip_cell_status(60, 0, 0, 60) == "ok"
    assert strip_cell_status(58, 0, 2, 60) == "warn"
    assert strip_cell_status(0, 0, 2, 2) == "down"
    now = time.time()
    current = int(now // 1800)
    strips = build_strips(
        [
            {
                "dependency": "neo4j",
                "slot": current,
                "ok": 0,
                "warn": 0,
                "down": 2,
                "total": 2,
            },
            {
                "dependency": "neo4j",
                "slot": current - 47,
                "ok": 60,
                "warn": 0,
                "down": 0,
                "total": 60,
            },
            {
                "dependency": "neo4j",
                "slot": current - 48,
                "ok": 1,
                "warn": 0,
                "down": 0,
                "total": 1,
            },
        ],
        now,
    )
    cells = strips["neo4j"]
    assert len(cells) == 48
    assert cells[0]["status"] == "ok" and cells[0]["total"] == 60
    assert cells[-1]["status"] == "down" and cells[-1]["failed"] == 2
    assert cells[1]["status"] == "unknown"
    assert cells[-1]["start"] == current * 1800 * 1000
