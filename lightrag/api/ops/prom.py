"""Optional Prometheus text exposition of the Operations console's numbers.

``GET /admin/ops/metrics`` (admin only, like every console route; 404 unless
``OPS_PROMETHEUS=true``) renders what the console already collects, in the
Prometheus text format (version 0.0.4), for a scraper that can send the
admin's credentials:

- heartbeats (``LIGHTRAG_OPS_WORKERS``): live and expected workers, summed
  RSS and CPU, the worst event-loop lag, and the LLM, embedding and rerank
  queues summed over live workers;
- the latency histograms (``LIGHTRAG_OPS_METRICS_MIN``) of the last five
  complete minutes per series (HTTP, embed, rerank): calls, errors, p95;
- open alerts per severity;
- the structured log sink's, the usage recorder's and run tracking's
  cumulative counters (dropped, rate-limited, failed writes, lost rows),
  summed over live workers;
- the latest dependency probe per dependency (up, latency) and per-GPU
  utilisation, memory, temperature and power from the GPU exporter.

Cardinality is bounded by design: labels are only ``role``, ``series``,
``severity``, ``dependency`` (the configured dependencies) and ``gpu`` (the
GPU index). There is never a workspace, user, route, request or worker
label. Values are host-wide, so N Gunicorn workers answer the same numbers.
Without the ops store only this process's numbers exist and are exported.
Rendering is cached for 15 s, so a scrape costs a few indexed reads.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger("lightrag.api.ops")

FLAG = "OPS_PROMETHEUS"
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
PREFIX = "lightrag_ops_"
CACHE_TTL_S = 15.0
METRIC_WINDOW_S = 300
QUEUE_ROLES = ("llm", "embed", "rerank")
QUEUE_FIELDS = ("running", "in_flight", "queued")
SEVERITIES = ("warning", "serious", "critical")
ALERTS_LIMIT = 500
_LABEL_VALUE = re.compile(r"[^A-Za-z0-9_.:-]")


def prometheus_enabled() -> bool:
    value = os.getenv(FLAG)
    return bool(value) and value.strip().lower() in ("1", "true", "yes", "on")


class Exposition:
    """Collects metric families and renders the text format.

    Every family has one ``# HELP`` and one ``# TYPE`` line followed by its
    samples; metric and label names are fixed in code, label values are
    escaped and reduced to a safe alphabet.
    """

    def __init__(self) -> None:
        self._families: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        kind: str,
        help_text: str,
        value: Any,
        labels: Optional[dict[str, Any]] = None,
    ) -> None:
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(number):
            return
        full = PREFIX + name
        family = self._families.setdefault(
            full, {"kind": kind, "help": help_text, "samples": []}
        )
        family["samples"].append((labels or {}, number))

    @staticmethod
    def _label(value: Any) -> str:
        text = _LABEL_VALUE.sub("_", str(value))[:64]
        return text.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _number(value: float) -> str:
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return repr(value)

    def render(self) -> str:
        lines: list[str] = []
        for name, family in self._families.items():
            help_text = family["help"].replace("\\", "\\\\").replace("\n", " ")
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {family['kind']}")
            for labels, value in family["samples"]:
                if labels:
                    rendered = ",".join(
                        f'{key}="{self._label(val)}"'
                        for key, val in sorted(labels.items())
                    )
                    lines.append(f"{name}{{{rendered}}} {self._number(value)}")
                else:
                    lines.append(f"{name} {self._number(value)}")
        return "\n".join(lines) + "\n"


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sum(items: Iterable[Optional[float]]) -> Optional[float]:
    values = [v for v in items if v is not None]
    return sum(values) if values else None


def _max(items: Iterable[Optional[float]]) -> Optional[float]:
    values = [v for v in items if v is not None]
    return max(values) if values else None


def _workers(out: Exposition, items: list[dict[str, Any]], process: dict) -> None:
    live = [item for item in items if item.get("alive")]
    out.add("workers_live", "gauge", "Workers with a fresh heartbeat.", len(live))
    out.add(
        "workers_expected",
        "gauge",
        "Workers this server is configured to run.",
        process.get("expected"),
    )
    snaps = [item.get("snapshot") or {} for item in live]
    out.add(
        "worker_rss_bytes",
        "gauge",
        "Resident memory summed over live workers.",
        _sum(_num(s.get("rss")) for s in snaps),
    )
    out.add(
        "worker_cpu_percent",
        "gauge",
        "Process CPU summed over live workers (100 = one core).",
        _sum(_num(s.get("cpu_pct")) for s in snaps),
    )
    out.add(
        "event_loop_lag_max_ms",
        "gauge",
        "Worst event-loop lag over live workers in their last beat window.",
        _max(_num(s.get("loop_lag_max_ms")) for s in snaps),
    )
    for role in QUEUE_ROLES:
        for field in QUEUE_FIELDS:
            out.add(
                f"queue_{field}",
                "gauge",
                f"Model calls {field.replace('_', ' ')} per role, summed over live workers.",
                _sum(
                    _num(((s.get("queues") or {}).get(role) or {}).get(field))
                    for s in snaps
                ),
                {"role": role},
            )
    counters = (
        (
            "log_sink",
            "dropped",
            "log_sink_dropped_total",
            "Log lines dropped by the structured sink (buffer full).",
        ),
        (
            "log_sink",
            "rate_limited",
            "log_sink_rate_limited_total",
            "Log lines only counted by the per-call-site limit.",
        ),
        (
            "log_sink",
            "write_failures",
            "log_sink_write_failures_total",
            "Failed structured log writes.",
        ),
        (
            "log_sink",
            "written",
            "log_sink_written_total",
            "Log rows written to LIGHTRAG_OPS_LOGS.",
        ),
        (
            "recorder",
            "lost_rows",
            "usage_lost_rows_total",
            "Usage records lost because the recorder buffer was full.",
        ),
        (
            "recorder",
            "dropped_events",
            "usage_dropped_events_total",
            "System events dropped by the usage recorder.",
        ),
        (
            "recorder",
            "flush_failures",
            "usage_flush_failures_total",
            "Failed usage recorder flushes.",
        ),
        (
            "run_tracking",
            "flush_failures",
            "run_tracking_flush_failures_total",
            "Failed run-tracking flushes.",
        ),
        (
            "run_tracking",
            "dropped_runs",
            "run_tracking_dropped_runs_total",
            "Run rows dropped from the run-tracking buffer.",
        ),
    )
    for part, key, name, help_text in counters:
        values = [_num((s.get(part) or {}).get(key)) for s in snaps]
        if any(v is not None for v in values):
            out.add(name, "counter", help_text, _sum(values))
    gauges = (
        (
            "log_sink",
            "buffer",
            "log_sink_buffer",
            "Log rows waiting in the structured sink buffers.",
        ),
        (
            "recorder",
            "buffer",
            "usage_recorder_buffer",
            "Usage records waiting to be written.",
        ),
    )
    for part, key, name, help_text in gauges:
        values = [_num((s.get(part) or {}).get(key)) for s in snaps]
        if any(v is not None for v in values):
            out.add(name, "gauge", help_text, _sum(values))


async def _latency(out: Exposition, service: Any) -> None:
    from lightrag.api.ops import metrics as ops_metrics
    from lightrag.api.ops.perf import _add_all, complete_end, summarize

    end = complete_end(time.time())
    start = end - METRIC_WINDOW_S

    def utc(epoch: float) -> datetime:
        return datetime.fromtimestamp(epoch, timezone.utc)

    series = (
        ops_metrics.SERIES_HTTP,
        ops_metrics.SERIES_EMBED,
        ops_metrics.SERIES_RERANK,
    )
    rows = await service.store.metrics_read(
        list(series), utc(start), utc(start), utc(end), 60
    )
    for name in series:
        summary = summarize(_add_all([r for r in rows if r["series"] == name]))
        labels = {"series": name}
        out.add(
            "calls_5m",
            "gauge",
            "Calls in the last five complete minutes.",
            summary["count"],
            labels,
        )
        out.add(
            "errors_5m",
            "gauge",
            "Failed calls (HTTP 5xx, model errors) in the last five complete minutes.",
            summary["errors"],
            labels,
        )
        out.add(
            "latency_p95_ms_5m",
            "gauge",
            "95th percentile latency over the last five complete minutes.",
            summary.get("p95"),
            labels,
        )


def _probes(out: Exposition, records: dict[str, Any]) -> None:
    for dep, record in sorted(records.items()):
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        if status in ("ok", "warn", "down"):
            out.add(
                "dependency_up",
                "gauge",
                "1 when the dependency's latest check passed (ok or warn), else 0.",
                1 if status in ("ok", "warn") else 0,
                {"dependency": dep},
            )
        out.add(
            "dependency_latency_ms",
            "gauge",
            "Latency of the dependency's latest check.",
            _num(record.get("latency_ms")),
            {"dependency": dep},
        )
    gpu = records.get("gpu")
    detail = (gpu or {}).get("detail") or {}
    for item in detail.get("gpus") or []:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            continue
        labels = {"gpu": item["index"]}
        for key, name, help_text in (
            ("util_pct", "gpu_utilization_percent", "GPU utilisation (exporter)."),
            ("mem_used_bytes", "gpu_memory_used_bytes", "GPU memory used (exporter)."),
            (
                "mem_total_bytes",
                "gpu_memory_total_bytes",
                "GPU memory total (exporter).",
            ),
            ("temp_c", "gpu_temperature_celsius", "GPU temperature (exporter)."),
            ("power_w", "gpu_power_watts", "GPU power draw (exporter)."),
        ):
            out.add(name, "gauge", help_text, _num(item.get(key)), labels)


async def render(service: Any, process: dict[str, Any]) -> str:
    """The exposition text (cached for :data:`CACHE_TTL_S`)."""

    async def build() -> str:
        out = Exposition()
        out.add("up", "gauge", "The Operations console answered this scrape.", 1)
        out.add(
            "store_available",
            "gauge",
            "1 when the ops store (Postgres) is available.",
            1 if service.available else 0,
        )
        try:
            items = await service._worker_items() or []
        except Exception as e:
            logger.debug("Prometheus: workers unreadable (%s)", type(e).__name__)
            items = []
        _workers(out, items, process)
        if service.available:
            try:
                await _latency(out, service)
            except Exception as e:
                logger.debug("Prometheus: metrics unreadable (%s)", type(e).__name__)
            try:
                rows = await service.store.alerts_open(ALERTS_LIMIT)
                counts = dict.fromkeys(SEVERITIES, 0)
                for row in rows:
                    severity = row.get("severity")
                    if severity in counts:
                        counts[severity] += 1
                for severity, count in counts.items():
                    out.add(
                        "alerts_open",
                        "gauge",
                        "Open (unresolved) alerts per severity.",
                        count,
                        {"severity": severity},
                    )
            except Exception as e:
                logger.debug("Prometheus: alerts unreadable (%s)", type(e).__name__)
        try:
            records, _, _ = await service._probe_records()
            _probes(out, records)
        except Exception as e:
            logger.debug("Prometheus: probes unreadable (%s)", type(e).__name__)
        return out.render()

    return await service.cache.get(("prometheus",), CACHE_TTL_S, build)


__all__ = ["CONTENT_TYPE", "Exposition", "prometheus_enabled", "render"]
