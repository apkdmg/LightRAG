"""Dependency probes: never block the loop, one in flight per dependency,
deadlines, whitelisted detail, parsing, publishing and rate limits."""

from __future__ import annotations

import asyncio
import smtplib
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lightrag.api.ops import probes
from lightrag.api.ops.probes import (
    PostgresProbe,
    ProbeResult,
    ProbeRunner,
    ProbeSpec,
    build_specs,
    certificate_status,
    classify_error,
    parse_certificate,
    parse_tls_targets,
    parse_vllm_metrics,
    safe_detail,
)
from lightrag.api.ops.store import MemoryOpsStore

pytestmark = pytest.mark.offline

VLLM_METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="bge-reranker-v2-m3"} 3.0
vllm:num_requests_running{engine="1",model_name="bge-reranker-v2-m3"} 5.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="bge-reranker-v2-m3"} 34.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="bge-reranker-v2-m3"} 0.71
vllm:kv_cache_usage_perc{engine="1",model_name="bge-reranker-v2-m3"} 0.42
vllm:prompt_tokens_total{model_name="bge-reranker-v2-m3"} 1.2e+06
"""


class Lease:
    def __init__(self, held: bool = True):
        self.held = held

    async def still_held(self) -> bool:
        return self.held


async def _ticker(stop: asyncio.Event, gaps: list[float]) -> None:
    last = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(0.01)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


async def test_a_hanging_probe_is_down_within_its_timeout_and_never_blocks_the_loop():
    release = threading.Event()
    calls = []

    def hang(timeout):
        calls.append(timeout)
        release.wait(5)  # a client that ignores its deadline
        return ProbeResult("ok", 1.0)

    runner = ProbeRunner(
        [ProbeSpec("milvus", "Milvus", "vector", timeout_s=0.3, run=hang)],
        store=None,
        holder="w",
    )
    stop = asyncio.Event()
    gaps: list[float] = []
    ticker = asyncio.create_task(_ticker(stop, gaps))
    started = time.perf_counter()
    try:
        result = await runner.execute(runner.specs["milvus"])
        elapsed = time.perf_counter() - started
        assert result.status == "down" and result.error_class == "timeout"
        assert elapsed < 0.3 + 0.1 + 0.05
        # A second cycle while the first is still stuck: no new thread.
        await asyncio.sleep(0.02)
        again = await runner.execute(runner.specs["milvus"])
        assert again.status == "down" and again.error_class == "probe_hung"
        assert len(calls) == 1
    finally:
        stop.set()
        await ticker
        release.set()
        runner.close()
    # The event loop kept ticking every ~10 ms throughout.
    assert max(gaps) < 0.1


async def test_a_probe_that_sleeps_does_not_delay_other_tasks():
    def slow(timeout):
        time.sleep(timeout)
        return ProbeResult("ok", timeout * 1000)

    runner = ProbeRunner(
        [ProbeSpec("docling", "Docling", "parser", timeout_s=0.5, run=slow)],
        store=None,
        holder="w",
    )
    stop = asyncio.Event()
    gaps: list[float] = []
    ticker = asyncio.create_task(_ticker(stop, gaps))
    try:
        records = await runner.run_cycle()
    finally:
        stop.set()
        await ticker
        runner.close()
    assert records["docling"]["status"] in ("ok", "down")
    assert max(gaps) < 0.1


async def test_the_probe_executor_is_not_the_default_executor():
    names = []

    def where(timeout):
        names.append(threading.current_thread().name)
        return ProbeResult("ok", 1.0)

    runner = ProbeRunner(
        [ProbeSpec("disk", "Disk", "disk", run=where)], store=None, holder="w"
    )
    loop = asyncio.get_running_loop()
    try:
        await runner.execute(runner.specs["disk"])
        assert names and names[0].startswith("ops-probe")
        assert runner.executor() is not getattr(loop, "_default_executor", None)
        assert runner.executor()._max_workers == probes.PROBE_THREADS
    finally:
        runner.close()


async def test_stuck_threads_make_other_probes_unknown_not_down():
    release = threading.Event()

    def hang(timeout):
        release.wait(5)
        return ProbeResult("ok", 1.0)

    runner = ProbeRunner(
        [
            ProbeSpec("neo4j", "Neo4j", "graph", timeout_s=0.1, run=hang),
            ProbeSpec("docling", "Docling", "parser", timeout_s=0.1, run=hang),
        ],
        store=None,
        holder="w",
        threads=1,
    )
    try:
        assert (await runner.execute(runner.specs["neo4j"])).error_class == "timeout"
        await asyncio.sleep(0.05)
        busy = await runner.execute(runner.specs["docling"])
        assert busy.status == "unknown" and busy.error_class == "executor_busy"
    finally:
        release.set()
        runner.close()


async def test_more_checks_than_threads_queue_instead_of_failing():
    def quick(timeout):
        time.sleep(0.05)
        return ProbeResult("ok", 50.0)

    specs = [
        ProbeSpec(f"dep{i}", f"Dep {i}", "parser", timeout_s=0.2, run=quick)
        for i in range(6)
    ]
    runner = ProbeRunner(specs, store=None, holder="w", threads=2)
    try:
        records = await runner.run_cycle()
    finally:
        runner.close()
    # Each check's deadline starts when a thread takes it, not when queued.
    assert {r["status"] for r in records.values()} == {"ok"}


async def test_async_probe_deadline_is_a_timeout():
    async def stuck(timeout):
        await asyncio.sleep(10)

    runner = ProbeRunner(
        [ProbeSpec("postgres", "PostgreSQL", "database", timeout_s=0.1, arun=stuck)],
        store=None,
        holder="w",
    )
    started = time.perf_counter()
    result = await runner.execute(runner.specs["postgres"])
    assert result.error_class == "timeout"
    assert time.perf_counter() - started < 0.1 + 0.1 + 0.05


async def test_exceptions_become_fixed_error_classes_without_text():
    def refuse(timeout):
        raise ConnectionRefusedError("refused by 10.0.0.5 for user secret")

    runner = ProbeRunner(
        [ProbeSpec("neo4j", "Neo4j", "graph", run=refuse)], store=None, holder="w"
    )
    try:
        records = await runner.run_cycle()
    finally:
        runner.close()
    record = records["neo4j"]
    assert record["status"] == "down"
    assert record["error_class"] == "connection_refused"
    assert "secret" not in str(record)


def test_classify_error():
    import socket
    import ssl

    import httpx

    assert classify_error(TimeoutError()) == "timeout"
    assert classify_error(socket.gaierror()) == "dns"
    assert classify_error(ssl.SSLCertVerificationError()) == "tls_verify"
    assert classify_error(httpx.ReadTimeout("x")) == "timeout"
    assert classify_error(smtplib.SMTPServerDisconnected()) == "smtp_error"

    class AuthError(Exception):
        pass

    assert classify_error(AuthError("password=x")) == "auth_failed"
    assert classify_error(ValueError("boom")) == "error"


def test_vllm_metrics_parsing():
    parsed = parse_vllm_metrics(VLLM_METRICS)
    assert parsed == {"running": 8, "waiting": 34, "kv_cache_pct": 71.0}
    older = parse_vllm_metrics("vllm:gpu_cache_usage_perc 0.22\n")
    assert older == {"kv_cache_pct": 22.0}
    assert parse_vllm_metrics("not metrics at all\n") == {}


def _certificate(days: int, *, self_signed: bool = True) -> bytes:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "lightrag-proxy")])
    issuer = (
        subject
        if self_signed
        else x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Example CA")])
    )
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=days, hours=1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.DER)


def test_tls_expiry_parsing():
    parsed = parse_certificate(_certificate(40))
    assert parsed["self_signed"] is True
    assert certificate_status(parsed["not_after"]) == ("ok", 40)
    assert (
        certificate_status(parse_certificate(_certificate(10))["not_after"])[0]
        == "warn"
    )
    assert (
        certificate_status(parse_certificate(_certificate(3))["not_after"])[0] == "down"
    )
    assert (
        parse_certificate(_certificate(40, self_signed=False))["self_signed"] is False
    )
    expired = datetime.now(timezone.utc) - timedelta(days=2)
    assert certificate_status(expired)[0] == "down"


def test_tls_targets():
    assert parse_tls_targets("lightrag-proxy:443, rag.example:8443,bad:port,") == [
        ("lightrag-proxy", 443),
        ("rag.example", 8443),
    ]
    assert parse_tls_targets("") == []


def test_detail_keys_are_whitelisted_and_urls_carry_no_userinfo():
    detail = safe_detail(
        {
            "target": "http://admin:hunter2@vllm-embed:8001/v1?api_key=sk-abc",
            "password": "hunter2",
            "query": "SELECT * FROM users WHERE email = 'a@b.c'",
            "states": {"active": 3, "idle in transaction": 1, "DROP TABLE": 9},
            "blocking": [
                {"pid": 7, "query": "UPDATE secret", "fingerprint": "abcd", "x": 1}
            ],
        }
    )
    assert detail["target"] == "http://vllm-embed:8001/v1"
    assert "password" not in detail and "query" not in detail
    assert detail["states"] == {"active": 3, "idle in transaction": 1}
    assert detail["blocking"] == [{"pid": 7, "fingerprint": "abcd"}]
    assert "hunter2" not in str(detail)


def _args(**overrides):
    base = dict(
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="Neo4JStorage",
        vector_storage="MilvusVectorDBStorage",
        embedding_binding_host="http://emb:s3cret@vllm-embed:8001/v1",
        rerank_binding="cohere",
        rerank_binding_host="http://vllm-rerank:8000/v1/rerank",
        llm_binding="openai",
        llm_binding_host="https://router.example/v1",
        oauth2_issuer="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_specs_follow_the_configuration(monkeypatch, tmp_path):
    for name, value in {
        "NEO4J_URI": "bolt://neo4j:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "pw",
        "MILVUS_URI": "http://root:Milvus@milvus:19530",
        "DOCLING_ENDPOINT": "http://docling:5001/v1/convert/file/async",
        "SMTP_HOST": "mail.example",
        "SMTP_FROM": "rag@example",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("OPS_TLS_PROBE_TARGETS", raising=False)
    specs = {
        s.id: s
        for s in build_specs(
            _args(),
            store=MemoryOpsStore(),
            app_state=SimpleNamespace(),
            disk_paths=[str(tmp_path)],
        )
    }
    assert set(specs) == {
        "postgres",
        "neo4j",
        "milvus",
        "etcd",
        "minio",
        "embed",
        "rerank",
        "llm",
        "docling",
        "smtp",
        "tls",
        "disk",
    }
    assert specs["embed"].target == "vllm-embed:8001"
    assert specs["rerank"].target == "vllm-rerank:8000"
    assert specs["milvus"].target == "milvus:19530"
    assert specs["etcd"].probed is False and specs["minio"].checkable is False
    assert specs["llm"].passive is True
    assert specs["smtp"].interval_s == probes.SMTP_INTERVAL_S
    assert specs["tls"].target == "lightrag-proxy:443"
    assert all("s3cret" not in str(s.target) for s in specs.values())
    assert all("Milvus@" not in str(s.target) for s in specs.values())


def test_specs_without_external_services(monkeypatch, tmp_path):
    for name in ("SMTP_HOST", "DOCLING_ENDPOINT", "REDIS_URI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPS_TLS_PROBE_TARGETS", "")
    specs = build_specs(
        _args(
            kv_storage="JsonKVStorage",
            graph_storage="NetworkXStorage",
            vector_storage="NanoVectorDBStorage",
            embedding_binding_host="DEFAULT_BEDROCK_ENDPOINT",
            rerank_binding="null",
        ),
        store=None,
        app_state=SimpleNamespace(),
        disk_paths=[str(tmp_path)],
    )
    assert [s.id for s in specs] == ["graph", "vector", "llm", "disk"]
    # The local stores still get a card, marked as not probed.
    assert specs[0].probed is False and specs[0].unprobed_reason == "local_storage"
    assert specs[0].name == "NetworkX" and specs[1].name == "NanoVectorDB"


def test_every_graph_and_vector_store_gets_a_card(monkeypatch, tmp_path):
    for name in ("NEO4J_URI", "MILVUS_URI", "QDRANT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPS_TLS_PROBE_TARGETS", "")
    monkeypatch.setattr(
        probes, "_ini_setting", lambda *a, **k: a[3] if len(a) > 3 else None
    )

    def cards(**storages):
        built = build_specs(
            _args(**storages), store=None, app_state=SimpleNamespace(), disk_paths=[]
        )
        return {s.kind: s for s in built if s.kind in ("graph", "vector")}

    pg = cards(graph_storage="PGGraphStorage", vector_storage="PGVectorStorage")
    assert pg["graph"].id == "graph" and pg["vector"].id == "vector"
    assert {c.unprobed_reason for c in pg.values()} == {"covered_by_postgres"}
    assert not any(c.checkable for c in pg.values())
    other = cards(
        graph_storage="MongoGraphStorage", vector_storage="MongoVectorDBStorage"
    )
    assert other["graph"].name == "MongoGraph"
    assert {c.unprobed_reason for c in other.values()} == {"no_probe"}
    # A Neo4j or Milvus whose address cannot be read is shown, not dropped.
    unread = cards(graph_storage="Neo4JStorage", vector_storage="MilvusVectorDBStorage")
    assert unread["graph"].id == "neo4j" and unread["vector"].id == "milvus"
    assert {c.unprobed_reason for c in unread.values()} == {"not_configured"}
    qdrant = cards(vector_storage="QdrantVectorDBStorage")
    assert qdrant["vector"].id == "qdrant" and not qdrant["vector"].probed


def test_an_unprobed_card_says_why():
    from lightrag.api.ops.health import dep_card

    spec = probes._unprobed_store("graph", "PGGraph", "graph", "PGGraphStorage")
    card = dep_card(spec, None, None, time.time())
    assert card["status"] == "unprobed"
    assert card["detail"] == {"reason": "covered_by_postgres"}


async def test_a_cycle_publishes_latest_results_and_history():
    clock_now = [10_000.0]
    store = MemoryOpsStore(clock=lambda: clock_now[0])
    outcomes = iter(["down", "down", "ok"])

    def flaky(timeout):
        status = next(outcomes)
        if status == "down":
            raise ConnectionRefusedError()
        return ProbeResult("ok", 5.0, None, {"target": "neo4j:7687"})

    runner = ProbeRunner(
        [ProbeSpec("neo4j", "Neo4j", "graph", run=flaky)],
        store=store,
        holder="w1",
        clock=lambda: clock_now[0],
    )
    try:
        await runner.run_cycle(Lease())
        record = store.kv["probe:neo4j"]
        assert record["status"] == "down" and record["consecutive_failures"] == 1
        assert record["last_ok_at"] is None
        assert store.kv["probe_cycle"]["holder"] == "w1"
        clock_now[0] += 30
        await runner.run_cycle(Lease())
        assert store.kv["probe:neo4j"]["consecutive_failures"] == 2
        clock_now[0] += 30
        await runner.run_cycle(Lease())
        record = store.kv["probe:neo4j"]
        assert record["status"] == "ok" and record["consecutive_failures"] == 0
        assert record["last_ok_at"] is not None
        assert [r["status"] for r in store.probe_rows] == ["down", "down", "ok"]
    finally:
        runner.close()


async def test_slow_checks_run_only_when_due_and_lost_leases_write_nothing():
    clock_now = [10_000.0]
    store = MemoryOpsStore(clock=lambda: clock_now[0])
    calls = []

    def smtp(timeout):
        calls.append(1)
        return ProbeResult("ok", 100.0)

    runner = ProbeRunner(
        [ProbeSpec("smtp", "SMTP", "mail", interval_s=900, run=smtp)],
        store=store,
        holder="w1",
        clock=lambda: clock_now[0],
    )
    try:
        await runner.run_cycle(Lease())
        clock_now[0] += 30
        assert await runner.run_cycle(Lease()) == {}
        assert len(calls) == 1
        clock_now[0] += 900
        assert await runner.run_cycle(Lease(held=False)) == {}
        assert len(store.probe_rows) == 1  # the lost cycle stored nothing
    finally:
        runner.close()


async def test_rate_limits_are_host_wide_through_the_store():
    clock_now = [5_000.0]
    store = MemoryOpsStore(clock=lambda: clock_now[0])
    a = ProbeRunner([], store=store, holder="worker-a")
    b = ProbeRunner([], store=store, holder="worker-b")
    assert await a.claim("recheck:neo4j", 10) is True
    assert await b.claim("recheck:neo4j", 10) is False  # another worker
    assert await a.claim("recheck:docling", 10) is True  # per dependency
    clock_now[0] += 11
    assert await b.claim("recheck:neo4j", 10) is True
    local = ProbeRunner([], store=None, holder="w")
    assert await local.claim("smtp_test", 60) is True
    assert await local.claim("smtp_test", 60) is False


async def test_postgres_probe_extras_and_blocking_detail():
    store = MemoryOpsStore()
    store.pg_health = {
        "latency_ms": 150.0,
        "connect_ms": 4.0,
        "max_connections": 200,
        "connections": 150,
        "db_size_bytes": 44_000_000_000,
        "server_version": "18.0",
        "states": {"active": 20, "idle": 130},
        "blocking": [
            {
                "pid": 11,
                "blocked_by": [12],
                "duration_s": 95.0,
                "state": "active",
                "statement_type": "UPDATE",
                "fingerprint": "0123456789abcdef",
                "query_length": 512,
            },
            {
                "pid": 12,
                "blocked_by": [],
                "duration_s": 300.0,
                "state": "idle in transaction",
            },
        ],
    }
    clock_now = [1_000.0]
    probe = PostgresProbe(store, clock=lambda: clock_now[0])
    result = await probe(5.0)
    assert result.status == "warn"  # 150 ms > 100 ms
    detail = result.detail
    assert detail["connections_pct"] == 75.0
    assert detail["blocked"] == 1 and detail["blocked_max_s"] == 95.0
    assert "query" not in str(detail).lower().replace("query_length", "")
    clock_now[0] += 30
    again = await probe(5.0)
    assert "connections" not in again.detail  # extras every 5 minutes


async def test_postgres_extras_report_pg_stat_database_deltas():
    store = MemoryOpsStore()
    stats = {
        "xact_commit": 1_000,
        "xact_rollback": 10,
        "deadlocks": 0,
        "temp_bytes": 0,
        "blks_read": 100,
        "blks_hit": 900,
    }
    store.pg_health = {
        "latency_ms": 2.0,
        "connect_ms": 1.0,
        "max_connections": 100,
        "connections": 10,
        "db_stats": dict(stats),
    }
    clock_now = [1_000.0]
    probe = PostgresProbe(store, clock=lambda: clock_now[0])
    first = await probe(5.0)
    assert "db_delta" not in first.detail  # nothing to compare with yet
    clock_now[0] += probes.PG_EXTRAS_INTERVAL_S
    store.pg_health["db_stats"] = {
        **stats,
        "xact_commit": 1_600,
        "xact_rollback": 13,
        "deadlocks": 1,
        "temp_bytes": 4096,
        "blks_read": 110,
        "blks_hit": 1_890,
    }
    second = await probe(5.0)
    assert safe_detail(second.detail)["db_delta"] == {
        "window_s": 300,
        "commits": 600,
        "rollbacks": 3,
        "deadlocks": 1,
        "temp_bytes": 4096,
        "cache_hit_pct": 99.0,
    }
    # A stats reset (counters went down) reports no delta rather than a negative one.
    clock_now[0] += probes.PG_EXTRAS_INTERVAL_S
    store.pg_health["db_stats"] = {**stats, "xact_commit": 5}
    third = await probe(5.0)
    assert "db_delta" not in third.detail


async def test_postgres_extras_are_carried_between_refreshes():
    store = MemoryOpsStore()
    store.pg_health = {
        "latency_ms": 2.0,
        "connect_ms": 1.0,
        "max_connections": 100,
        "connections": 10,
    }
    clock_now = [1_000.0]
    probe = PostgresProbe(store, clock=lambda: clock_now[0])
    runner = ProbeRunner(
        [ProbeSpec("postgres", "PostgreSQL", "database", arun=probe)],
        store=store,
        holder="w",
        clock=lambda: clock_now[0],
    )
    await runner.run_cycle(Lease())
    clock_now[0] += 30
    await runner.run_cycle(Lease())
    assert store.kv["probe:postgres"]["detail"]["connections"] == 10


async def test_vllm_probe_metrics_unavailable_does_not_fail(monkeypatch):
    answers = {"/health": (200, b""), "/metrics": (401, b"")}

    def fake_get(url, timeout, **kwargs):
        return answers["/" + url.rsplit("/", 1)[-1]]

    monkeypatch.setattr(probes, "_http_get", fake_get)
    run = probes.model_endpoint_probe("http://vllm-rerank:8000")
    result = run(1.0)
    assert result.status == "ok" and result.detail["metrics"] == "unavailable"
    answers["/metrics"] = (200, VLLM_METRICS.encode())
    result = run(1.0)
    assert result.detail["waiting"] == 34 and result.detail["metrics"] == "ok"
    answers["/health"] = (404, b"")
    assert run(1.0).detail["health_endpoint"] == "missing"
    answers["/health"] = (503, b"")
    down = run(1.0)
    assert down.status == "down" and down.error_class == "http_503"


async def test_waiting_since_tracks_a_long_vllm_queue():
    clock_now = [1_000.0]
    waiting = [34]

    def rerank(timeout):
        return ProbeResult("ok", 5.0, None, {"waiting": waiting[0]})

    runner = ProbeRunner(
        [ProbeSpec("rerank", "Rerank", "rerank", run=rerank)],
        store=None,
        holder="w",
        clock=lambda: clock_now[0],
    )
    try:
        first = (await runner.run_cycle())["rerank"]["detail"]["waiting_since"]
        clock_now[0] += 30
        assert (await runner.run_cycle())["rerank"]["detail"]["waiting_since"] == first
        waiting[0] = 3
        clock_now[0] += 30
        assert "waiting_since" not in (await runner.run_cycle())["rerank"]["detail"]
    finally:
        runner.close()


def test_smtp_probe_never_logs_in_or_sends(monkeypatch):
    calls = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            calls.append(("connect", host, port))
            self.sock = None

        def ehlo(self):
            calls.append("ehlo")
            return 250, b"ok"

        def starttls(self, context=None):
            calls.append("starttls")

        def noop(self):
            calls.append("noop")
            return 250, b"ok"

        def quit(self):
            calls.append("quit")

        def close(self):
            calls.append("close")

        def login(self, *a):  # pragma: no cover - must not be called
            raise AssertionError("login")

        def send_message(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("send")

        sendmail = send_message

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    settings = SimpleNamespace(host="mail.example", port=587, security="starttls")
    result = probes.smtp_probe(settings)(5.0)
    assert result.status == "ok" and result.detail["starttls"] is True
    assert calls[0] == ("connect", "mail.example", 587)
    assert "noop" in calls and "quit" in calls


async def test_passive_llm_status():
    class Store:
        def __init__(self, data):
            self.data = data

        async def llm_health(self, start, end):
            return self.data

    state = SimpleNamespace(insights=None)
    llm = probes.PassiveLlm(state, "openai", "https://u:p@router.example/v1")
    unknown = await llm(1.0)
    assert unknown.status == "unknown" and unknown.detail["target"] == "router.example"
    state.insights = SimpleNamespace(store=Store({"calls": 5, "errors": 5}))
    assert (await llm(1.0)).detail["low_volume"] is True
    state.insights.store.data = {"calls": 100, "errors": 30, "p95_ms": 4000}
    assert (await llm(1.0)).status == "down"
    state.insights.store.data = {"calls": 100, "errors": 10, "p95_ms": 4000}
    assert (await llm(1.0)).status == "warn"
    state.insights.store.data = {"calls": 100, "errors": 0, "p95_ms": 4000}
    assert (await llm(1.0)).status == "ok"


def test_disk_probe(tmp_path):
    result = probes.disk_probe([str(tmp_path)])(1.0)
    assert result.status in ("ok", "warn", "down")
    assert result.detail["paths"][0]["path"] == str(tmp_path)
    assert probes.disk_probe(["/does/not/exist"])(1.0).status == "unknown"


def test_idp_is_probed_only_when_sso_is_usable(monkeypatch):
    monkeypatch.setenv("OPS_TLS_PROBE_TARGETS", "")
    issuer = "https://id.example/realms/UNIMAS"
    args = _args(
        oauth2_enabled=True,
        oauth2_client_id="lightrag",
        oauth2_client_secret="s",
        oauth2_issuer=issuer,
    )
    specs = {
        s.id: s for s in build_specs(args, store=None, app_state=SimpleNamespace())
    }
    assert specs["idp"].interval_s == probes.IDP_INTERVAL_S
    args.oauth2_client_secret = ""
    assert "idp" not in {
        s.id for s in build_specs(args, store=None, app_state=SimpleNamespace())
    }

    body = b'{"issuer": "https://id.example/realms/UNIMAS"}'
    monkeypatch.setattr(probes, "_http_get", lambda url, timeout, **k: (200, body))
    assert probes.idp_probe(issuer)(1.0).status == "ok"
    monkeypatch.setattr(
        probes, "_http_get", lambda url, timeout, **k: (200, b'{"issuer": "x"}')
    )
    assert probes.idp_probe(issuer)(1.0).error_class == "issuer_mismatch"
    monkeypatch.setattr(probes, "_http_get", lambda url, timeout, **k: (502, b""))
    assert probes.idp_probe(issuer)(1.0).status == "down"


async def test_a_recheck_during_the_cycle_shares_the_running_check():
    calls = []

    def slow(timeout):
        calls.append(1)
        time.sleep(0.2)
        return ProbeResult("ok", 200.0)

    runner = ProbeRunner(
        [ProbeSpec("neo4j", "Neo4j", "graph", timeout_s=1.0, run=slow)],
        store=None,
        holder="w",
    )
    try:
        cycle, recheck = await asyncio.gather(
            runner.run_cycle(), runner.probe_now("neo4j")
        )
    finally:
        runner.close()
    assert cycle["neo4j"]["status"] == "ok" and recheck["status"] == "ok"
    assert len(calls) == 1


async def test_joining_a_queued_check_its_owner_cancels_is_busy_not_cancelled(
    monkeypatch,
):
    # The only thread is busy; a re-check queues a check, the cycle joins it,
    # and the owner then gives up and cancels the queued job. The joiner must
    # get executor_busy, not a CancelledError that would end the probes loop.
    monkeypatch.setattr(probes, "QUEUE_WAIT_S", 0.3)
    release = threading.Event()

    def hang(timeout):
        release.wait(5)
        return ProbeResult("ok", 1.0)

    def quick(timeout):
        return ProbeResult("ok", 1.0)

    runner = ProbeRunner(
        [
            ProbeSpec("milvus", "Milvus", "vector", timeout_s=5.0, run=hang),
            ProbeSpec("neo4j", "Neo4j", "graph", timeout_s=1.0, run=quick),
        ],
        store=None,
        holder="w",
        threads=1,
    )
    try:
        blocker = asyncio.create_task(runner.execute(runner.specs["milvus"]))
        await asyncio.sleep(0.05)
        owner = asyncio.create_task(runner.execute(runner.specs["neo4j"]))
        await asyncio.sleep(0.05)
        joiner = asyncio.create_task(runner.execute(runner.specs["neo4j"]))
        owned, joined = await asyncio.gather(owner, joiner)
        assert owned.error_class == "executor_busy"
        assert joined.status == "unknown" and joined.error_class == "executor_busy"
        assert not joiner.cancelled()
    finally:
        release.set()
        await blocker
        runner.close()


async def test_a_real_cancel_of_a_joining_check_still_propagates(monkeypatch):
    monkeypatch.setattr(probes, "QUEUE_WAIT_S", 5.0)
    release = threading.Event()

    def hang(timeout):
        release.wait(5)
        return ProbeResult("ok", 1.0)

    runner = ProbeRunner(
        [ProbeSpec("neo4j", "Neo4j", "graph", timeout_s=5.0, run=hang)],
        store=None,
        holder="w",
    )
    try:
        owner = asyncio.create_task(runner.execute(runner.specs["neo4j"]))
        await asyncio.sleep(0.05)
        joiner = asyncio.create_task(runner.execute(runner.specs["neo4j"]))
        await asyncio.sleep(0.05)
        joiner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await joiner
    finally:
        release.set()
        await owner
        runner.close()
