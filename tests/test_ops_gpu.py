"""GPU metrics from the nvidia_gpu_exporter sidecar (PR-9, decision D5).

The exporter's Prometheus text is parsed per GPU; the feature is off while
``OPS_GPU_EXPORTER_URL`` is unset; the read is a probe on the probes' own
executor, so a hung exporter is ``down`` within its deadline and never
blocks the event loop; history comes from the probe rows.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from lightrag.api.ops import gpu as ops_gpu
from lightrag.api.ops.probes import ProbeRunner, build_specs, safe_detail
from lightrag.api.ops.store import MemoryOpsStore
from tests.ops_fixtures import build_app

pytestmark = pytest.mark.offline

FIXTURE = """\
# HELP nvidia_smi_gpu_info A metric with a constant '1' value labeled by gpu uuid, name, driver_model_current, driver_model_pending, vbios_version, driver_version.
# TYPE nvidia_smi_gpu_info gauge
nvidia_smi_gpu_info{driver_model_current="N/A",driver_model_pending="N/A",driver_version="550.54.15",name="NVIDIA A30",uuid="GPU-1111aaaa-0000-0000-0000-000000000001",vbios_version="92.00.66.00.01"} 1
nvidia_smi_gpu_info{driver_model_current="N/A",driver_model_pending="N/A",driver_version="550.54.15",name="NVIDIA A30",uuid="GPU-2222bbbb-0000-0000-0000-000000000002",vbios_version="92.00.66.00.01"} 1
# HELP nvidia_smi_index index
# TYPE nvidia_smi_index gauge
nvidia_smi_index{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 0
nvidia_smi_index{uuid="GPU-2222bbbb-0000-0000-0000-000000000002"} 1
# TYPE nvidia_smi_utilization_gpu_ratio gauge
nvidia_smi_utilization_gpu_ratio{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 0.42
nvidia_smi_utilization_gpu_ratio{uuid="GPU-2222bbbb-0000-0000-0000-000000000002"} 0.07
nvidia_smi_utilization_memory_ratio{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 0.11
nvidia_smi_memory_used_bytes{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 1.8253611008e+10
nvidia_smi_memory_total_bytes{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 2.5769803776e+10
nvidia_smi_memory_used_bytes{uuid="GPU-2222bbbb-0000-0000-0000-000000000002"} 1.048576e+09
nvidia_smi_memory_total_bytes{uuid="GPU-2222bbbb-0000-0000-0000-000000000002"} 2.5769803776e+10
nvidia_smi_temperature_gpu{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 61
nvidia_smi_temperature_gpu{uuid="GPU-2222bbbb-0000-0000-0000-000000000002"} 38
nvidia_smi_power_draw_watts{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 112.53
nvidia_smi_power_draw_watts{uuid="GPU-2222bbbb-0000-0000-0000-000000000002"} NaN
nvidia_smi_enforced_power_limit_watts{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 165
nvidia_smi_fan_speed_ratio{uuid="GPU-1111aaaa-0000-0000-0000-000000000001"} 0
garbage line without a value
nvidia_smi_command_exit_code 0
"""


# -- parsing ----------------------------------------------------------------------------


def test_exporter_metrics_are_parsed_per_gpu():
    gpus = ops_gpu.parse_gpu_metrics(FIXTURE)
    assert [g["index"] for g in gpus] == [0, 1]
    first, second = gpus
    assert first["name"] == "NVIDIA A30"
    assert first["util_pct"] == 42.0
    assert first["temp_c"] == 61.0
    assert first["power_w"] == 112.5
    assert first["power_limit_w"] == 165.0
    assert first["mem_used_bytes"] == 18253611008
    assert first["mem_total_bytes"] == 25769803776
    assert first["mem_pct"] == pytest.approx(70.8, abs=0.1)
    # NaN power is skipped, the rest of the GPU stays.
    assert "power_w" not in second and second["util_pct"] == 7.0
    # The UUID never leaves the parser.
    assert not any("GPU-" in str(v) for g in gpus for v in g.values())


def test_summary_and_status():
    gpus = ops_gpu.parse_gpu_metrics(FIXTURE)
    summary = ops_gpu.gpu_summary(gpus)
    assert summary == {
        "gpu_count": 2,
        "util_max_pct": 42.0,
        "mem_max_pct": pytest.approx(70.8, abs=0.1),
        "temp_max_c": 61.0,
        "power_total_w": 112.5,
    }
    assert ops_gpu.gpu_status(gpus) == ("ok", None)
    assert ops_gpu.gpu_status([]) == ("warn", "no_gpus")
    assert ops_gpu.gpu_status([{"index": 0, "temp_c": 90}]) == ("warn", "hot")
    assert ops_gpu.gpu_status([{"index": 0, "mem_pct": 99.0}]) == (
        "warn",
        "memory_full",
    )


def test_parsed_detail_survives_the_detail_whitelist():
    gpus = ops_gpu.parse_gpu_metrics(FIXTURE)
    detail = safe_detail({"gpus": gpus, **ops_gpu.gpu_summary(gpus), "uuid": "x"})
    assert detail["gpus"][0]["util_pct"] == 42.0
    assert detail["gpu_count"] == 2
    assert "uuid" not in detail


def test_exporter_url_normalisation(monkeypatch):
    monkeypatch.delenv(ops_gpu.EXPORTER_URL_ENV, raising=False)
    assert ops_gpu.exporter_url() is None and not ops_gpu.gpu_enabled()
    assert ops_gpu.exporter_url("  ") is None
    assert (
        ops_gpu.exporter_url("http://gpu-exporter:9835")
        == "http://gpu-exporter:9835/metrics"
    )
    assert ops_gpu.exporter_url("http://x:9835/custom/") == "http://x:9835/custom"
    assert ops_gpu.exporter_url("ftp://x/metrics") is None
    assert ops_gpu.exporter_url("not a url") is None


# -- the probe ----------------------------------------------------------------------------


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        embedding_binding_host=None,
        rerank_binding=None,
        llm_binding_host=None,
        llm_binding="openai",
    )


def test_feature_off_when_url_unset(monkeypatch):
    monkeypatch.delenv(ops_gpu.EXPORTER_URL_ENV, raising=False)
    monkeypatch.setenv("OPS_TLS_PROBE_TARGETS", "")
    specs = build_specs(_args(), store=None, app_state=SimpleNamespace())
    assert "gpu" not in {s.id for s in specs}
    monkeypatch.setenv(ops_gpu.EXPORTER_URL_ENV, "http://gpu-exporter:9835")
    specs = {s.id: s for s in build_specs(_args(), store=None, app_state=None)}
    assert specs["gpu"].kind == "gpu" and specs["gpu"].target == "gpu-exporter:9835"
    assert specs["gpu"].run is not None and specs["gpu"].checkable


class _Exporter:
    """A tiny HTTP server answering /metrics (or hanging, or failing)."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(5)
        self.port = self.sock.getsockname()[1]
        self.release = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            try:
                conn.recv(4096)
                if self.mode == "hang":
                    self.release.wait(5)
                    continue
                body = FIXTURE.encode() if self.mode == "ok" else b"err"
                status = "200 OK" if self.mode == "ok" else "500 Error"
                conn.sendall(
                    f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n".encode()
                    + body
                )
            finally:
                conn.close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/metrics"

    def close(self):
        self.release.set()
        self.sock.close()


async def test_probe_reads_the_exporter():
    exporter = _Exporter("ok")
    spec = ops_gpu.gpu_spec(exporter.url)
    runner = ProbeRunner([spec], store=None, holder="w")
    try:
        result = await runner.execute(spec)
    finally:
        runner.close()
        exporter.close()
    assert result.status == "ok"
    assert result.detail["gpu_count"] == 2
    assert result.detail["gpus"][0]["util_pct"] == 42.0
    assert result.detail["target"] == f"127.0.0.1:{exporter.port}"


async def test_exporter_error_is_down():
    exporter = _Exporter("error")
    spec = ops_gpu.gpu_spec(exporter.url)
    runner = ProbeRunner([spec], store=None, holder="w")
    try:
        result = await runner.execute(spec)
    finally:
        runner.close()
        exporter.close()
    assert result.status == "down" and result.error_class == "http_500"


async def test_a_hung_exporter_never_blocks_the_loop():
    exporter = _Exporter("hang")
    spec = ops_gpu.gpu_spec(exporter.url)
    spec.timeout_s = 0.3
    runner = ProbeRunner([spec], store=None, holder="w")
    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker():
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    started = time.perf_counter()
    try:
        result = await runner.execute(spec)
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        await task
        runner.close()
        exporter.close()
    assert result.status == "down"
    assert result.error_class in ("timeout", "probe_hung")
    assert elapsed < 0.3 + 0.1 + 0.2
    assert max(gaps) < 0.1


# -- routes ---------------------------------------------------------------------------------


def _probe_rows(store: MemoryOpsStore) -> None:
    now = datetime.now(timezone.utc)
    for minutes, util0, temp0 in ((30, 40.0, 60.0), (20, 60.0, 64.0), (5, 20.0, 58.0)):
        store.probe_rows.append(
            {
                "ts": now - timedelta(minutes=minutes),
                "dependency": "gpu",
                "status": "ok",
                "latency_ms": 4,
                "error_class": None,
                "detail": {
                    "gpus": [
                        {
                            "index": 0,
                            "util_pct": util0,
                            "mem_pct": 70.0,
                            "temp_c": temp0,
                            "power_w": 100.0,
                        },
                        {"index": 1, "util_pct": 5.0, "mem_pct": 4.0, "temp_c": 38.0},
                    ]
                },
            }
        )
    # Another dependency's rows never leak in.
    store.probe_rows.append(
        {"ts": now, "dependency": "embed", "status": "ok", "detail": {"running": 1}}
    )


async def test_gpu_series_from_probe_rows():
    store = MemoryOpsStore()
    _probe_rows(store)
    now = datetime.now(timezone.utc)
    rows = await store.gpu_series(
        now - timedelta(hours=1), now + timedelta(minutes=1), 3600
    )
    by_gpu = {r["gpu"]: r for r in rows}
    assert set(by_gpu) == {0, 1}
    if len(rows) == 2:  # all three samples in one hourly bucket
        assert by_gpu[0]["util_pct"] == pytest.approx(40.0)
        assert by_gpu[0]["temp_c"] == 64.0
        assert by_gpu[1]["power_w"] is None


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def test_status_and_performance_gpu(monkeypatch):
    store = MemoryOpsStore()
    _probe_rows(store)
    app = build_app(store)
    async with _client(app) as client:
        monkeypatch.delenv(ops_gpu.EXPORTER_URL_ENV, raising=False)
        status = (await client.get("/admin/ops/status")).json()
        assert status["features"]["gpu"] is False
        off = (await client.get("/admin/ops/performance/gpu?window=1h")).json()
        assert off["enabled"] is False and off["series"] == []

        monkeypatch.setenv(ops_gpu.EXPORTER_URL_ENV, "http://gpu-exporter:9835")
        status = (await client.get("/admin/ops/status")).json()
        assert status["features"]["gpu"] is True
        spec = ops_gpu.gpu_spec()
        runner = ProbeRunner([spec], store=store, holder="w")
        runner.latest["gpu"] = {
            "id": "gpu",
            "status": "ok",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "detail": {"gpus": [{"index": 0, "util_pct": 20.0}], "gpu_count": 1},
        }
        store.kv["probe:gpu"] = runner.latest["gpu"]
        app.state.ops.service.probes = runner
        body = (await client.get("/admin/ops/performance/gpu?window=24h")).json()
        assert body["enabled"] is True and body["configured"] is True
        assert body["latest"]["gpus"][0]["util_pct"] == 20.0
        assert body["latest"]["summary"] == {"gpu_count": 1}
        assert {p["gpu"] for p in body["series"]} == {0, 1}
        assert all(isinstance(p["t"], int) for p in body["series"])
        bad = await client.get("/admin/ops/performance/gpu?window=2d")
        assert bad.status_code == 400 and bad.json()["detail"] == "invalid_window"
        health = (await client.get("/admin/ops/health")).json()
        assert "gpu" in {d["id"] for d in health["deps"]}
        runner.close()
