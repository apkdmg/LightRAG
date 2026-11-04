"""Optional Prometheus text exposition at ``/admin/ops/metrics`` (PR-9).

Off by default (404), admin-only, valid text format, bounded labels: never a
workspace, user, route, request or worker label, even when the data behind
it carries them.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

from lightrag.api.dependencies import require_admin
from lightrag.api.ops import prom
from lightrag.api.ops.probes import ProbeRunner, ProbeSpec
from tests.ops_fixtures import build_app, seeded_store

pytestmark = pytest.mark.offline

ALLOWED_LABELS = {"role", "series", "severity", "dependency", "gpu"}
_NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
_SAMPLE = re.compile(
    rf"^(?P<name>{_NAME})(?:\{{(?P<labels>[^}}]*)\}})? (?P<value>[-+]?(?:[0-9.]+(?:e[-+]?[0-9]+)?|Inf|NaN))$"
)
_LABEL = re.compile(r'^(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:[^"\\\n]|\\.)*)"$')


def parse_exposition(text: str) -> dict[str, dict]:
    """A strict reader of the text format (0.0.4): HELP and TYPE once per
    family and before its samples, samples grouped under their family, label
    pairs well formed, values numeric, the text ending with a newline."""
    assert text.endswith("\n")
    families: dict[str, dict] = {}
    current = None
    for line in text.splitlines():
        assert line, "empty line"
        if line.startswith("# HELP "):
            name = line.split(" ", 3)[2]
            assert re.fullmatch(_NAME, name)
            assert name not in families, f"family {name} twice"
            families[name] = {"help": line.split(" ", 3)[3], "samples": []}
            current = name
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            assert name == current and "type" not in families[name]
            assert kind in ("counter", "gauge", "histogram", "summary", "untyped")
            families[name]["type"] = kind
            continue
        assert not line.startswith("#"), line
        match = _SAMPLE.match(line)
        assert match, f"bad sample line: {line!r}"
        name = match.group("name")
        assert name == current, f"sample {name} outside its family"
        labels = {}
        if match.group("labels"):
            for pair in match.group("labels").split(","):
                label = _LABEL.match(pair)
                assert label, f"bad label {pair!r}"
                labels[label.group("key")] = label.group("value")
        float(match.group("value"))
        families[name]["samples"].append((labels, float(match.group("value"))))
    for name, family in families.items():
        assert "type" in family and family["samples"], name
        if family["type"] == "counter":
            assert name.endswith("_total"), name
    return families


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


def _app_with_data():
    store = seeded_store()
    now = datetime.now(timezone.utc)
    for worker_id, rss in (("a" * 32, 1000), ("b" * 32, 2000)):
        store.worker_rows[worker_id] = {
            "worker_id": worker_id,
            "host": "h",
            "pid": 1,
            "pid_create_time": 1.0,
            "mode": "gunicorn",
            "version": "x",
            "started_at": now - timedelta(minutes=5),
            "heartbeat_at": now,
            "exited_at": None,
            "superseded": False,
            "leases_held": [],
            "snapshot": {
                "rss": rss,
                "cpu_pct": 12.5,
                "loop_lag_max_ms": 3.0,
                "queues": {"llm": {"running": 2, "in_flight": 3, "queued": 1}},
                "log_sink": {"dropped": 4, "written": 10, "buffer": 2},
                "recorder": {"lost_rows": 1, "buffer": 5},
                # Workspaces appear in heartbeats: they must not become labels.
                "top_instances": [{"workspace": "fsktm", "running": 2, "queued": 1}],
            },
        }
    minute = now.replace(second=0, microsecond=0) - timedelta(minutes=3)
    store.metric_rows.append(
        {
            "minute": minute,
            "worker_id": "a" * 32,
            "series": "http",
            "key": "GET /documents/{doc_id}",
            "count": 10,
            "errors": 1,
            "sum_ms": 500,
            "max_ms": 120,
            "buckets": [0, 5, 5] + [0] * 17,
        }
    )
    app = build_app(store)
    service = app.state.ops.service
    runner = ProbeRunner(
        [
            ProbeSpec("postgres", "PostgreSQL", "database"),
            ProbeSpec("gpu", "GPU", "gpu"),
        ],
        store=store,
        holder="w",
    )
    store.kv["probe:postgres"] = {"status": "ok", "latency_ms": 2.0}
    store.kv["probe:gpu"] = {
        "status": "ok",
        "latency_ms": 4.0,
        "detail": {"gpus": [{"index": 0, "util_pct": 42.0, "temp_c": 61.0}]},
    }
    service.probes = runner
    return app


async def test_off_by_default(monkeypatch):
    monkeypatch.delenv(prom.FLAG, raising=False)
    async with _client(_app_with_data()) as client:
        response = await client.get("/admin/ops/metrics")
        status = (await client.get("/admin/ops/status")).json()
    assert response.status_code == 404
    assert status["features"]["prometheus"] is False


async def test_admin_only(monkeypatch):
    monkeypatch.setenv(prom.FLAG, "true")
    app = _app_with_data()

    def deny():
        raise HTTPException(status_code=403, detail="Admin access required")

    app.dependency_overrides[require_admin] = deny
    async with _client(app) as client:
        response = await client.get("/admin/ops/metrics")
    assert response.status_code == 403


async def test_exposition_parses_with_bounded_labels(monkeypatch):
    monkeypatch.setenv(prom.FLAG, "true")
    async with _client(_app_with_data()) as client:
        response = await client.get("/admin/ops/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert response.headers["cache-control"] == "no-store"
    families = parse_exposition(response.text)
    for name, family in families.items():
        assert name.startswith("lightrag_ops_")
        for labels, _ in family["samples"]:
            assert set(labels) <= ALLOWED_LABELS, (name, labels)
    assert "fsktm" not in response.text and "/documents" not in response.text
    value = {
        name: {tuple(sorted(lb.items())): v for lb, v in f["samples"]}
        for name, f in families.items()
    }
    assert value["lightrag_ops_up"][()] == 1
    assert value["lightrag_ops_workers_live"][()] == 2
    assert value["lightrag_ops_worker_rss_bytes"][()] == 3000
    assert value["lightrag_ops_queue_running"][(("role", "llm"),)] == 4
    assert value["lightrag_ops_log_sink_dropped_total"][()] == 8
    assert families["lightrag_ops_log_sink_dropped_total"]["type"] == "counter"
    assert value["lightrag_ops_alerts_open"][(("severity", "serious"),)] == 1
    assert value["lightrag_ops_dependency_up"][(("dependency", "postgres"),)] == 1
    assert value["lightrag_ops_gpu_utilization_percent"][(("gpu", "0"),)] == 42
    assert value["lightrag_ops_calls_5m"][(("series", "http"),)] == 10
    assert value["lightrag_ops_errors_5m"][(("series", "http"),)] == 1


async def test_without_the_store_only_this_process_answers(monkeypatch):
    monkeypatch.setenv(prom.FLAG, "true")
    app = build_app(None, kv_storage="JsonKVStorage")
    async with _client(app) as client:
        response = await client.get("/admin/ops/metrics")
    assert response.status_code == 200
    families = parse_exposition(response.text)
    assert families["lightrag_ops_store_available"]["samples"] == [({}, 0.0)]


def test_label_values_are_escaped_and_reduced():
    out = prom.Exposition()
    out.add("x", "gauge", "help\nline", 1, {"dependency": 'a"b\\c\nd'})
    out.add("x", "gauge", "ignored", float("nan"), {"dependency": "n"})
    out.add("y_total", "counter", "c", 2.5)
    text = out.render()
    families = parse_exposition(text)
    assert families["lightrag_ops_x"]["samples"] == [({"dependency": "a_b_c_d"}, 1.0)]
    assert families["lightrag_ops_y_total"]["samples"] == [({}, 2.5)]


async def test_rendering_is_cached(monkeypatch):
    monkeypatch.setenv(prom.FLAG, "true")
    app = _app_with_data()
    service = app.state.ops.service
    calls = 0
    original = service._worker_items

    async def counting():
        nonlocal calls
        calls += 1
        return await original()

    monkeypatch.setattr(service, "_worker_items", counting)
    async with _client(app) as client:
        for _ in range(3):
            assert (await client.get("/admin/ops/metrics")).status_code == 200
    assert calls == 1
