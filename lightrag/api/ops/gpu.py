"""GPU metrics for the Operations console, from an ``nvidia_gpu_exporter`` sidecar.

Decision D5: the API container never gets ``runtime: nvidia`` and never loads
NVML. GPU numbers come from the ``utkuozdemir/nvidia_gpu_exporter`` sidecar
that the setup wizard adds to GPU deployments when asked to
(``OPS_GPU_EXPORTER=true``); the API reads the sidecar's Prometheus
``/metrics`` over the compose network. ``OPS_GPU_EXPORTER_URL`` (e.g.
``http://gpu-exporter:9835``) turns the feature on; unset, nothing here runs
and ``/status`` reports ``features.gpu: false``.

The read is one more dependency probe (``gpu``): it runs on the probes'
own executor with a client deadline like every other blocking check, so a
hung exporter is reported as ``down`` and never blocks the event loop or the
default executor. Each probe row stores per-GPU utilisation, memory,
temperature and power in ``detail.gpus``, which is where the history on the
Performance tab comes from (``LIGHTRAG_OPS_PROBES``, 7 days). vLLM's own
``/metrics`` (running, waiting, KV cache) stays on the embed and rerank
probes.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("lightrag.api.ops")

EXPORTER_URL_ENV = "OPS_GPU_EXPORTER_URL"
DEP_ID = "gpu"
METRICS_MAX_BYTES = 512 * 1024
MAX_GPUS = 16
# A GPU this hot or this full is worth a look (warn); none of it is "down":
# the exporter answered, so the dependency itself is reachable.
TEMP_WARN_C = 85.0
MEM_WARN_PCT = 95.0

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eEInfa]+)(?:\s+-?\d+)?\s*$"
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')
_GPU_NAME = re.compile(r"[^A-Za-z0-9 ._()/-]")

# Exporter metric -> field (value transform applied in _field_value).
_FIELDS = {
    "nvidia_smi_utilization_gpu_ratio": "util_pct",
    "nvidia_smi_memory_used_bytes": "mem_used_bytes",
    "nvidia_smi_memory_total_bytes": "mem_total_bytes",
    "nvidia_smi_temperature_gpu": "temp_c",
    "nvidia_smi_power_draw_watts": "power_w",
    "nvidia_smi_enforced_power_limit_watts": "power_limit_w",
    "nvidia_smi_index": "index",
}

# Windows of the Performance tab's GPU section and their history buckets.
WINDOWS: dict[str, int] = {"15m": 900, "1h": 3600, "24h": 86400, "7d": 7 * 86400}
BUCKET_S: dict[str, int] = {"15m": 60, "1h": 60, "24h": 900, "7d": 3600}
CACHE_TTL_S = 30.0


def exporter_url(value: Optional[str] = None) -> Optional[str]:
    """The exporter's ``/metrics`` URL, or None (feature off).

    ``OPS_GPU_EXPORTER_URL`` may name the base (``http://gpu-exporter:9835``)
    or the metrics path itself; only http(s) URLs are accepted.
    """
    raw = os.getenv(EXPORTER_URL_ENV) if value is None else value
    text = (raw or "").strip()
    if not text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        logger.warning("%s is not a URL; GPU metrics stay off", EXPORTER_URL_ENV)
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        logger.warning(
            "%s must be an http(s) URL; GPU metrics stay off", EXPORTER_URL_ENV
        )
        return None
    path = parts.path.rstrip("/")
    if not path:
        path = "/metrics"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def gpu_enabled() -> bool:
    return exporter_url() is not None


def _labels(raw: Optional[str]) -> dict[str, str]:
    if not raw:
        return {}
    return {
        key: value.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
        for key, value in _LABEL.findall(raw)
    }


def _field_value(field: str, value: float) -> float:
    if field == "util_pct":
        # A ratio (0..1) in the exporter; tolerate a percentage too.
        return value * 100.0 if value <= 1.0 else value
    return value


def parse_gpu_metrics(text: str) -> list[dict[str, Any]]:
    """Per-GPU utilisation, memory, temperature and power from the exporter's
    Prometheus text, ordered by GPU index.

    GPUs are keyed by the ``uuid`` label; the model name comes from
    ``nvidia_smi_gpu_info``. The UUID itself is not returned. Unknown metrics,
    NaN values and malformed lines are skipped.
    """
    gpus: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        match = _SAMPLE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        if name != "nvidia_smi_gpu_info" and name not in _FIELDS:
            continue
        labels = _labels(match.group("labels"))
        key = labels.get("uuid") or labels.get("gpu") or labels.get("index") or "0"
        if key not in gpus and len(gpus) >= MAX_GPUS:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        gpu = gpus.setdefault(key, {})
        if name == "nvidia_smi_gpu_info":
            model = _GPU_NAME.sub("", labels.get("name") or "")[:64].strip()
            if model:
                gpu["name"] = model
            continue
        field = _FIELDS[name]
        gpu[field] = _field_value(field, value)
    out: list[dict[str, Any]] = []
    for position, gpu in enumerate(gpus.values()):
        index = gpu.get("index")
        item: dict[str, Any] = {
            "index": int(index) if isinstance(index, float) else position,
            "name": gpu.get("name"),
        }
        for field in ("util_pct", "temp_c", "power_w", "power_limit_w"):
            if field in gpu:
                item[field] = round(float(gpu[field]), 1)
        used, total = gpu.get("mem_used_bytes"), gpu.get("mem_total_bytes")
        if used is not None:
            item["mem_used_bytes"] = int(used)
        if total is not None:
            item["mem_total_bytes"] = int(total)
        if used is not None and total:
            item["mem_pct"] = round(float(used) * 100.0 / float(total), 1)
        if len(item) > 2:
            out.append(item)
    out.sort(key=lambda g: g["index"])
    return out


def gpu_summary(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Host-level figures over every GPU (maxima, total power)."""

    def values(field: str) -> list[float]:
        return [float(g[field]) for g in gpus if isinstance(g.get(field), (int, float))]

    out: dict[str, Any] = {"gpu_count": len(gpus)}
    for field, key in (
        ("util_pct", "util_max_pct"),
        ("mem_pct", "mem_max_pct"),
        ("temp_c", "temp_max_c"),
    ):
        found = values(field)
        if found:
            out[key] = round(max(found), 1)
    power = values("power_w")
    if power:
        out["power_total_w"] = round(sum(power), 1)
    return out


def gpu_status(gpus: list[dict[str, Any]]) -> tuple[str, Optional[str]]:
    """``(status, reason)`` for parsed GPUs."""
    if not gpus:
        return "warn", "no_gpus"
    if any(float(g.get("temp_c") or 0) >= TEMP_WARN_C for g in gpus):
        return "warn", "hot"
    if any(float(g.get("mem_pct") or 0) >= MEM_WARN_PCT for g in gpus):
        return "warn", "memory_full"
    return "ok", None


def gpu_probe(url: str) -> Callable[[float], Any]:
    """A blocking probe (run on the probe executor) reading the exporter."""
    from lightrag.api.ops.probes import (
        ProbeResult,
        _http_class,
        _http_get,
        display_target,
    )

    def run(timeout: float) -> Any:
        started = time.perf_counter()
        status_code, body = _http_get(url, timeout, max_bytes=METRICS_MAX_BYTES)
        latency = (time.perf_counter() - started) * 1000.0
        detail: dict[str, Any] = {"target": display_target(url)}
        if status_code != 200:
            detail["http_status"] = status_code
            return ProbeResult("down", latency, _http_class(status_code), detail)
        gpus = parse_gpu_metrics(body.decode("utf-8", "replace"))
        status, reason = gpu_status(gpus)
        detail.update(gpu_summary(gpus))
        detail["gpus"] = gpus
        if reason:
            detail["reason"] = reason
        return ProbeResult(status, latency, None, detail)

    return run


def gpu_spec(url: Optional[str] = None) -> Any:
    """The ``gpu`` probe spec, or None when the exporter is not configured."""
    from lightrag.api.ops.probes import ProbeSpec, display_target

    target = url or exporter_url()
    if target is None:
        return None
    return ProbeSpec(
        DEP_ID,
        "GPU",
        "gpu",
        timeout_s=5.0,
        target=display_target(target),
        run=gpu_probe(target),
    )


# -- read model (Performance tab) ------------------------------------------------------


def _iso_ms(epoch_s: Any) -> int:
    return int(float(epoch_s)) * 1000


def latest_view(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The newest probe record as the GPU section shows it."""
    if not isinstance(record, dict):
        return None
    detail = record.get("detail") or {}
    return {
        "status": record.get("status"),
        "checked_at": record.get("checked_at"),
        "error_class": record.get("error_class"),
        "reason": detail.get("reason"),
        "gpus": list(detail.get("gpus") or []),
        "summary": {
            key: detail.get(key)
            for key in (
                "gpu_count",
                "util_max_pct",
                "mem_max_pct",
                "temp_max_c",
                "power_total_w",
            )
            if detail.get(key) is not None
        },
    }


async def gpu_view(service: Any, window: str) -> dict[str, Any]:
    """``GET /performance/gpu``: the latest per-GPU numbers and their history
    (per bucket: average utilisation and power, peak memory and temperature)
    from the probe rows. Works without the ops store (latest only)."""
    from lightrag.api.ops.perf import PerfBadRequest

    if window not in WINDOWS:
        raise PerfBadRequest("invalid_window")
    configured = gpu_enabled()
    probes = getattr(service, "probes", None)
    probed = probes is not None and DEP_ID in getattr(probes, "specs", {})
    body: dict[str, Any] = {
        "window": window,
        "enabled": configured and probed,
        "configured": configured,
        "bucket_s": BUCKET_S[window],
        "latest": None,
        "series": [],
        "durable": bool(getattr(service, "available", False)),
    }
    if not body["enabled"]:
        return body
    try:
        records, _, _ = await service._probe_records()
        body["latest"] = latest_view(records.get(DEP_ID))
    except Exception as e:
        logger.debug("Ops GPU latest unreadable (%s)", type(e).__name__)
    if not service.available:
        return body
    now = time.time()
    bucket = BUCKET_S[window]
    end = (int(now) // bucket + 1) * bucket
    start = end - WINDOWS[window] - bucket

    async def load() -> list[dict[str, Any]]:
        return await service.store.gpu_series(
            datetime.fromtimestamp(start, timezone.utc),
            datetime.fromtimestamp(end, timezone.utc),
            bucket,
        )

    try:
        rows = await service.cache.get(
            ("gpu_series", window), CACHE_TTL_S, lambda: service._q(load)
        )
    except Exception as e:
        # The latest numbers still stand without their history.
        logger.debug("Ops GPU history unreadable (%s)", type(e).__name__)
        rows = []
    body["series"] = [
        {
            "t": _iso_ms(row["t"]),
            "gpu": int(row["gpu"]),
            "util_pct": _round(row.get("util_pct")),
            "mem_pct": _round(row.get("mem_pct")),
            "temp_c": _round(row.get("temp_c")),
            "power_w": _round(row.get("power_w")),
        }
        for row in rows
    ]
    return body


def _round(value: Any) -> Optional[float]:
    return None if value is None else round(float(value), 1)


__all__ = [
    "DEP_ID",
    "EXPORTER_URL_ENV",
    "exporter_url",
    "gpu_enabled",
    "gpu_probe",
    "gpu_spec",
    "gpu_status",
    "gpu_summary",
    "gpu_view",
    "latest_view",
    "parse_gpu_metrics",
]
