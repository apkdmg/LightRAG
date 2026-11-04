"""Dependency probes for the Operations console's Health view.

One worker per host (the holder of the ``ops:probes`` lease) checks every
dependency the server is configured to use, every 30 s; SMTP and TLS every
15 min and the identity provider every 5 min. Postgres extras (connections,
database size, blocked backends) are refreshed every 5 min. Without Postgres
the checks still run, locally, and are kept in memory only.

Never blocking the event loop:

- blocking clients (HTTP, bolt, gRPC, SMTP, TLS handshakes, disk) run on a
  dedicated ``ThreadPoolExecutor`` of 4 threads, never the default executor;
- every probe has a client-side deadline (socket timeouts) and the loop waits
  at most ``timeout + 100 ms`` for it;
- at most one probe per dependency is in flight: when the previous one is
  still stuck in its thread, the cycle records ``down: probe_hung`` and starts
  no new thread; when every thread is stuck, the others record
  ``unknown: executor_busy`` instead of a false ``down``;
- the Postgres probe is async, on a connection of its own (never the pool).

Results: the latest per dependency goes to ``LIGHTRAG_OPS_KV``
(``probe:<id>``, plus ``probe_cycle``), history to ``LIGHTRAG_OPS_PROBES``
(kept 7 days). ``detail`` holds whitelisted keys only; targets are shown as
``host:port`` and URLs never carry userinfo. Nothing here sends mail, logs in
to SMTP, makes a paid LLM call or reads query text.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import configparser
import logging
import os
import re
import smtplib
import socket
import ssl
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from lightrag.api.ops.leases import being_cancelled as _being_cancelled
from lightrag.api.ops.perf import (
    BASELINE_KEY,
    METRIC_THRESHOLDS,
    baseline_for,
    llm_p95_severity,
    merge_thresholds,
)
from lightrag.api.ops.redact import strip_userinfo

logger = logging.getLogger("lightrag.api.ops")

PROBE_LEASE = "ops:probes"
PROBE_INTERVAL_S = 30.0
PROBE_CYCLE_TIMEOUT_S = 45.0
# Outlasts the worst cycle: every probe's timeout runs in parallel waves.
PROBE_LEASE_TTL_S = 60.0
PROBE_THREADS = 4
DEFAULT_TIMEOUT_S = 5.0
WAIT_GRACE_S = 0.1
# Longest wait for a free probe thread; with the longest probe timeout (15 s)
# it stays inside the cycle timeout.
QUEUE_WAIT_S = 20.0
SMTP_INTERVAL_S = 900.0
TLS_INTERVAL_S = 900.0
IDP_INTERVAL_S = 300.0
PG_EXTRAS_INTERVAL_S = 300.0
RECHECK_INTERVAL_S = 10.0
SMTP_TEST_INTERVAL_S = 60.0
# Re-checks of the slow-cadence probes (SMTP, TLS, IdP) touch outside servers
# that normally see one check per 5-15 min: at most one a minute.
SLOW_RECHECK_INTERVAL_S = 60.0
RETENTION_DAYS = 7
CLEANUP_INTERVAL_S = 3600.0
STRIP_SLOTS = 48
STRIP_SLOT_S = 1800

SLOW_MS = 500.0
PG_WARN_MS = 100.0
TLS_WARN_DAYS = 21
TLS_DOWN_DAYS = 7
DISK_WARN_FREE_PCT = 15.0
DISK_DOWN_FREE_PCT = 5.0
VLLM_WAITING_WARN = 20
LLM_MIN_CALLS = 20
LLM_WARN_RATE = 0.05
LLM_DOWN_RATE = 0.20
LLM_SLOW_P95_MS = 60_000.0

KV_PREFIX = "probe:"
CYCLE_KEY = "probe_cycle"
METRICS_MAX_BYTES = 2 * 1024 * 1024
BODY_MAX_BYTES = 64 * 1024
DEFAULT_TLS_TARGETS = "lightrag-proxy:443"
MAX_TLS_TARGETS = 5
STATUSES = ("ok", "warn", "down", "unknown")
_STATUS_RANK = {"ok": 0, "unknown": 1, "warn": 2, "down": 3}

# Every key a probe may put in ``detail``; anything else is dropped.
DETAIL_KEYS = frozenset(
    {
        "target",
        "targets",
        "kind",
        "configured",
        "reason",
        "http_status",
        "health_endpoint",
        "metrics",
        "running",
        "waiting",
        "kv_cache_pct",
        "waiting_since",
        "healthz",
        "version",
        "server_version",
        "connect_ms",
        "connections",
        "max_connections",
        "connections_pct",
        "db_size_bytes",
        "states",
        "blocked",
        "blocked_max_s",
        "blocking",
        "extras_at",
        "db_delta",
        "window_s",
        "commits",
        "rollbacks",
        "deadlocks",
        "temp_bytes",
        "cache_hit_pct",
        "pid",
        "blocked_by",
        "duration_s",
        "state",
        "wait_event",
        "wait_event_type",
        "statement_type",
        "fingerprint",
        "query_length",
        "is_blocking",
        "days_left",
        "not_after",
        "self_signed",
        "trusted",
        "error_class",
        "security",
        "starttls",
        "cert_days_left",
        "cert_not_after",
        "issuer_ok",
        "paths",
        "path",
        "free_pct",
        "used",
        "total",
        "calls",
        "errors",
        "error_rate",
        "p95_ms",
        "baseline_p95_ms",
        "p95_ratio",
        "window",
        "low_volume",
        "model",
        "binding",
        "note",
    }
)
_DETAIL_MAX_STR = 200
_DETAIL_MAX_ITEMS = 50
_STATE_NAME = re.compile(r"^[a-z][a-z ()_]{0,40}$")
_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eEInfa]+)"
)
_LOCAL_STORAGES = frozenset(
    {
        "JsonKVStorage",
        "JsonDocStatusStorage",
        "NetworkXStorage",
        "NanoVectorDBStorage",
        "FaissVectorDBStorage",
    }
)


@dataclass
class ProbeResult:
    status: str
    latency_ms: Optional[float] = None
    error_class: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeSpec:
    """One dependency: how to check it and how often."""

    id: str
    name: str
    kind: str
    interval_s: float = PROBE_INTERVAL_S
    timeout_s: float = DEFAULT_TIMEOUT_S
    target: Optional[str] = None
    # Blocking check, run on the probe executor with a deadline in seconds.
    run: Optional[Callable[[float], ProbeResult]] = None
    # Async check, run on the event loop (it must not block).
    arun: Optional[Callable[[float], Awaitable[ProbeResult]]] = None
    # Health read from real traffic: nothing to re-check.
    passive: bool = False
    # Shown but never checked (etcd and MinIO behind Milvus, a store without
    # a probe); ``unprobed_reason`` says why on the card.
    probed: bool = True
    unprobed_reason: str = "not_probed"

    @property
    def checkable(self) -> bool:
        return self.probed and (self.run is not None or self.arun is not None)


# -- small helpers ---------------------------------------------------------------


def classify_error(exc: BaseException) -> str:
    """A fixed error class for an exception; its message never leaves here."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        return "tls_verify"
    if isinstance(exc, ssl.SSLError):
        return "tls"
    if isinstance(exc, socket.gaierror):
        return "dns"
    if isinstance(exc, ConnectionRefusedError):
        return "connection_refused"
    if isinstance(exc, (TimeoutError, socket.timeout, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(exc, smtplib.SMTPException):
        return "smtp_error"
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.ConnectError):
            cause = exc.__context__ or exc.__cause__
            if cause is not None and cause is not exc:
                inner = classify_error(cause)
                if inner != "error":
                    return inner
            return "connection_error"
        if isinstance(exc, httpx.HTTPError):
            return "http_error"
    except ImportError:  # pragma: no cover - httpx is a dependency
        pass
    if isinstance(exc, ConnectionError):
        return "connection_error"
    if isinstance(exc, OSError):
        return "os_error"
    name = type(exc).__name__
    lowered = name.lower()
    module = type(exc).__module__ or ""
    if module.startswith("pymilvus"):
        return "milvus_error"
    if "auth" in lowered or "unauthor" in lowered or "credential" in lowered:
        return "auth_failed"
    if "timeout" in lowered:
        return "timeout"
    if "unavailable" in lowered or "connection" in lowered:
        return "unavailable"
    return "error"


def display_target(url_or_host: Optional[str]) -> Optional[str]:
    """``host:port`` of a URL (no scheme, userinfo, path or query)."""
    if not url_or_host:
        return None
    text = str(url_or_host).strip()
    if "://" not in text:
        return text.rsplit("@", 1)[-1][:120]
    try:
        parts = urlsplit(text)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return None
    if not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}" if port else host


def safe_detail(value: Any, _depth: int = 0) -> Any:
    """Keep whitelisted keys only, cap strings and lists, strip URL userinfo."""
    if _depth > 4:
        return None
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key == "states" and isinstance(item, dict):
                out[key] = {
                    str(k): int(v)
                    for k, v in item.items()
                    if isinstance(v, (int, float)) and _STATE_NAME.match(str(k))
                }
            elif key in DETAIL_KEYS:
                out[key] = safe_detail(item, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [safe_detail(v, _depth + 1) for v in list(value)[:_DETAIL_MAX_ITEMS]]
    if isinstance(value, str):
        return strip_userinfo(value)[:_DETAIL_MAX_STR]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 3)
    return str(value)[:_DETAIL_MAX_STR]


def _iso(epoch: Optional[float]) -> Optional[str]:
    if not isinstance(epoch, (int, float)) or epoch <= 0:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def _remaining(deadline: float) -> float:
    return max(0.05, deadline - time.monotonic())


def _ini_setting(env: str, section: str, option: str, default: Any = None) -> Any:
    value = os.environ.get(env)
    if value:
        return value
    config = configparser.ConfigParser()
    try:
        config.read("config.ini", "utf-8")
        return config.get(section, option, fallback=default)
    except (configparser.Error, OSError):
        return default


def _http_get(
    url: str,
    timeout: float,
    *,
    headers: Optional[dict[str, str]] = None,
    max_bytes: int = BODY_MAX_BYTES,
) -> tuple[int, bytes]:
    """GET with a hard deadline and a body cap; no redirects, no cookies."""
    import httpx

    with httpx.Client(
        timeout=httpx.Timeout(timeout), follow_redirects=False, trust_env=False
    ) as client:
        with client.stream("GET", url, headers=headers or {}) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) >= max_bytes:
                    break
            return response.status_code, bytes(body[:max_bytes])


def _http_class(status_code: int) -> str:
    return f"http_{status_code}"


def _slow(latency_ms: float, threshold: float = SLOW_MS) -> str:
    return "warn" if latency_ms > threshold else "ok"


# -- parsing (pure, tested) ---------------------------------------------------------


def parse_vllm_metrics(text: str) -> dict[str, Any]:
    """Running and waiting requests and KV-cache use from vLLM ``/metrics``.

    Values are summed over label sets (several engines); KV-cache use is the
    highest one, as a percentage. Newer vLLM names the gauge
    ``vllm:kv_cache_usage_perc``, older ones ``vllm:gpu_cache_usage_perc``.
    """
    running = waiting = None
    kv: Optional[float] = None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if value != value:  # NaN
            continue
        if name == "vllm:num_requests_running":
            running = (running or 0.0) + value
        elif name == "vllm:num_requests_waiting":
            waiting = (waiting or 0.0) + value
        elif name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            kv = value if kv is None else max(kv, value)
    out: dict[str, Any] = {}
    if running is not None:
        out["running"] = int(running)
    if waiting is not None:
        out["waiting"] = int(waiting)
    if kv is not None:
        out["kv_cache_pct"] = round(kv * 100.0 if kv <= 1.0 else kv, 1)
    return out


def parse_certificate(der: bytes) -> dict[str, Any]:
    """``not_after`` (aware UTC) and whether the certificate is self-signed."""
    try:
        from cryptography import x509

        cert = x509.load_der_x509_certificate(der)
        not_after = getattr(cert, "not_valid_after_utc", None)
        if not_after is None:  # cryptography < 42
            not_after = cert.not_valid_after.replace(tzinfo=timezone.utc)
        return {
            "not_after": not_after,
            "self_signed": cert.issuer == cert.subject,
        }
    except ImportError:
        pass
    # Fallback: the stdlib decoder only reads a PEM file.
    pem = ssl.DER_cert_to_PEM_cert(der)
    handle, path = tempfile.mkstemp(suffix=".pem")
    try:
        with os.fdopen(handle, "w", encoding="ascii") as fh:
            fh.write(pem)
        decoded = ssl._ssl._test_decode_cert(path)  # type: ignore[attr-defined]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {
        "not_after": datetime.fromtimestamp(
            ssl.cert_time_to_seconds(decoded["notAfter"]), timezone.utc
        ),
        "self_signed": decoded.get("issuer") == decoded.get("subject"),
    }


def certificate_status(
    not_after: datetime, now: Optional[datetime] = None
) -> tuple[str, int]:
    """Status and whole days left for a certificate's expiry."""
    now = now or datetime.now(timezone.utc)
    days_left = int((not_after - now).total_seconds() // 86400)
    if days_left < TLS_DOWN_DAYS:
        return "down", days_left
    if days_left < TLS_WARN_DAYS:
        return "warn", days_left
    return "ok", days_left


def parse_tls_targets(value: Optional[str]) -> list[tuple[str, int]]:
    targets: list[tuple[str, int]] = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        host, _, port = item.rpartition(":")
        if not host:
            host, port = item, "443"
        try:
            number = int(port)
        except ValueError:
            continue
        if 0 < number < 65536 and host:
            targets.append((host.strip("[]"), number))
    return targets[:MAX_TLS_TARGETS]


# -- blocking checks (run on the probe executor) -----------------------------------------


def http_health_probe(url: str, *, ok_codes: Iterable[int] = (200,)) -> Callable:
    """A probe that GETs ``url`` and expects one of ``ok_codes``."""
    wanted = frozenset(ok_codes)

    def run(timeout: float) -> ProbeResult:
        started = time.perf_counter()
        status_code, _ = _http_get(url, timeout)
        latency = (time.perf_counter() - started) * 1000.0
        detail = {"target": display_target(url), "http_status": status_code}
        if status_code in wanted:
            return ProbeResult(_slow(latency), latency, None, detail)
        return ProbeResult("down", latency, _http_class(status_code), detail)

    return run


def model_endpoint_probe(base_url: str) -> Callable:
    """vLLM-style endpoint: ``/health`` reachability plus ``/metrics`` (V4).

    Any HTTP answer below 500 means the server is up; a 401, 403 or 404 on
    ``/health`` or ``/metrics`` marks that endpoint unavailable in ``detail``
    without failing the dependency.
    """
    base = base_url.rstrip("/")

    def run(timeout: float) -> ProbeResult:
        deadline = time.monotonic() + timeout
        started = time.perf_counter()
        status_code, _ = _http_get(f"{base}/health", _remaining(deadline))
        latency = (time.perf_counter() - started) * 1000.0
        detail: dict[str, Any] = {"target": display_target(base)}
        if status_code >= 500:
            detail["http_status"] = status_code
            return ProbeResult("down", latency, _http_class(status_code), detail)
        detail["health_endpoint"] = (
            "ok"
            if status_code == 200
            else "auth_required"
            if status_code in (401, 403)
            else "missing"
        )
        try:
            code, body = _http_get(
                f"{base}/metrics", _remaining(deadline), max_bytes=METRICS_MAX_BYTES
            )
            if code == 200:
                parsed = parse_vllm_metrics(body.decode("utf-8", "replace"))
                detail["metrics"] = "ok" if parsed else "unavailable"
                detail.update(parsed)
            else:
                detail["metrics"] = "unavailable"
        except Exception:
            detail["metrics"] = "unavailable"
        return ProbeResult(_slow(latency), latency, None, detail)

    return run


def bolt_probe(uri: str, username: Optional[str], password: Optional[str]) -> Callable:
    """Neo4j or Memgraph: ``verify_connectivity()`` on a driver of its own."""

    def run(timeout: float) -> ProbeResult:
        from neo4j import GraphDatabase

        started = time.perf_counter()
        auth = (username, password) if username else None
        driver = GraphDatabase.driver(
            uri,
            auth=auth,
            max_connection_pool_size=1,
            connection_timeout=timeout,
            connection_acquisition_timeout=timeout,
            max_transaction_retry_time=0,
        )
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        latency = (time.perf_counter() - started) * 1000.0
        return ProbeResult(
            _slow(latency), latency, None, {"target": display_target(uri)}
        )

    return run


def milvus_probe(
    uri: str,
    *,
    healthz_url: str,
    user: Optional[str],
    password: Optional[str],
    token: Optional[str],
) -> Callable:
    """Milvus ``/healthz`` on its own HTTP client; when that port is not
    reachable, ``get_server_version`` on a client of its own."""

    def run(timeout: float) -> ProbeResult:
        deadline = time.monotonic() + timeout
        detail: dict[str, Any] = {"target": display_target(uri)}
        started = time.perf_counter()
        try:
            status_code, body = _http_get(healthz_url, _remaining(deadline) / 2)
            latency = (time.perf_counter() - started) * 1000.0
            detail["healthz"] = status_code
            if status_code == 200:
                return ProbeResult(_slow(latency), latency, None, detail)
            if status_code >= 500:
                return ProbeResult("down", latency, _http_class(status_code), detail)
        except Exception as e:
            detail["healthz"] = classify_error(e)
        from pymilvus import MilvusClient

        started = time.perf_counter()
        kwargs: dict[str, Any] = {"uri": uri, "timeout": _remaining(deadline)}
        if token:
            kwargs["token"] = token
        elif user:
            kwargs["user"] = user
            kwargs["password"] = password or ""
        client = MilvusClient(**kwargs)
        try:
            version = client.get_server_version(timeout=_remaining(deadline))
        finally:
            try:
                client.close()
            except Exception:
                pass
        latency = (time.perf_counter() - started) * 1000.0
        detail["version"] = str(version)[:32]
        return ProbeResult(_slow(latency), latency, None, detail)

    return run


def redis_probe(uri: str) -> Callable:
    def run(timeout: float) -> ProbeResult:
        import redis

        started = time.perf_counter()
        client = redis.Redis.from_url(
            uri, socket_timeout=timeout, socket_connect_timeout=timeout
        )
        try:
            client.ping()
        finally:
            client.close()
        latency = (time.perf_counter() - started) * 1000.0
        return ProbeResult(
            _slow(latency), latency, None, {"target": display_target(uri)}
        )

    return run


def _peer_certificate(host: str, port: int, timeout: float) -> tuple[bytes, bool]:
    """The server's certificate (DER) and whether the default trust store accepts it."""
    deadline = time.monotonic() + timeout
    context = ssl.create_default_context()
    try:
        with socket.create_connection(
            (host, port), timeout=_remaining(deadline)
        ) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                return tls.getpeercert(binary_form=True) or b"", True
    except ssl.SSLCertVerificationError:
        pass
    insecure = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=_remaining(deadline)) as sock:
        with insecure.wrap_socket(sock, server_hostname=host) as tls:
            return tls.getpeercert(binary_form=True) or b"", False


def tls_probe(targets: list[tuple[str, int]]) -> Callable:
    """Certificate expiry and trust for each target; the worst one wins.

    A name that does not resolve is ``unknown`` (the target is not part of
    this deployment), not ``down``.
    """

    def run(timeout: float) -> ProbeResult:
        deadline = time.monotonic() + timeout
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        worst = "ok"
        for host, port in targets:
            row: dict[str, Any] = {"target": f"{host}:{port}"}
            try:
                der, trusted = _peer_certificate(
                    host, port, _remaining(deadline) / max(1, len(targets))
                )
                cert = parse_certificate(der)
                status, days_left = certificate_status(cert["not_after"])
                row.update(
                    {
                        "days_left": days_left,
                        "not_after": cert["not_after"],
                        "self_signed": cert["self_signed"],
                        "trusted": trusted,
                        "state": status,
                    }
                )
            except Exception as e:
                error = classify_error(e)
                status = "unknown" if error == "dns" else "down"
                row.update({"error_class": error, "state": status})
            rows.append(row)
            if _STATUS_RANK[status] > _STATUS_RANK[worst]:
                worst = status
        latency = (time.perf_counter() - started) * 1000.0
        errors = [r["error_class"] for r in rows if r.get("error_class")]
        return ProbeResult(
            worst,
            latency,
            errors[0] if worst in ("down", "unknown") and errors else None,
            {"targets": rows},
        )

    return run


def smtp_probe(settings: Any) -> Callable:
    """Connect, EHLO, STARTTLS (when configured), NOOP and QUIT.

    Never logs in and never sends. The certificate's expiry is recorded; the
    server is reached by host name, so the certificate check matches it.
    """

    def run(timeout: float) -> ProbeResult:
        started = time.perf_counter()
        context = ssl.create_default_context()
        if settings.security == "ssl":
            server: smtplib.SMTP = smtplib.SMTP_SSL(
                settings.host, settings.port, timeout=timeout, context=context
            )
        else:
            server = smtplib.SMTP(settings.host, settings.port, timeout=timeout)
        detail: dict[str, Any] = {
            "target": f"{settings.host}:{settings.port}",
            "security": settings.security,
        }
        try:
            code, _ = server.ehlo()
            if code >= 400:
                raise smtplib.SMTPResponseException(code, b"")
            if settings.security == "starttls":
                server.starttls(context=context)
                server.ehlo()
                detail["starttls"] = True
            sock = getattr(server, "sock", None)
            if isinstance(sock, ssl.SSLSocket):
                peer = sock.getpeercert() or {}
                not_after = peer.get("notAfter")
                if not_after:
                    moment = datetime.fromtimestamp(
                        ssl.cert_time_to_seconds(not_after), timezone.utc
                    )
                    _, days_left = certificate_status(moment)
                    detail["cert_days_left"] = days_left
                    detail["cert_not_after"] = moment
            code, _ = server.noop()
            if code >= 400:
                raise smtplib.SMTPResponseException(code, b"")
            try:
                server.quit()
            except smtplib.SMTPException:
                pass
        finally:
            try:
                server.close()
            except Exception:
                pass
        latency = (time.perf_counter() - started) * 1000.0
        days = detail.get("cert_days_left")
        status = "ok"
        if isinstance(days, int) and days < TLS_WARN_DAYS:
            status = "down" if days < TLS_DOWN_DAYS else "warn"
        return ProbeResult(status, latency, None, detail)

    return run


def idp_probe(issuer: str) -> Callable:
    """OIDC discovery document of the identity provider."""
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"

    def run(timeout: float) -> ProbeResult:
        import json

        started = time.perf_counter()
        status_code, body = _http_get(url, timeout)
        latency = (time.perf_counter() - started) * 1000.0
        detail: dict[str, Any] = {
            "target": display_target(url),
            "http_status": status_code,
        }
        if status_code != 200:
            return ProbeResult("down", latency, _http_class(status_code), detail)
        try:
            document = json.loads(body.decode("utf-8", "replace"))
            issuer_ok = str(document.get("issuer", "")).rstrip("/") == issuer.rstrip(
                "/"
            )
        except (ValueError, AttributeError):
            issuer_ok = False
        detail["issuer_ok"] = issuer_ok
        if not issuer_ok:
            return ProbeResult("warn", latency, "issuer_mismatch", detail)
        return ProbeResult(_slow(latency), latency, None, detail)

    return run


def disk_probe(paths: list[str]) -> Callable:
    def run(timeout: float) -> ProbeResult:
        import shutil

        started = time.perf_counter()
        rows = []
        worst = None
        for path in paths:
            try:
                usage = shutil.disk_usage(path)
            except OSError:
                continue
            free_pct = usage.free / usage.total * 100.0 if usage.total else 100.0
            rows.append(
                {
                    "path": path,
                    "free_pct": round(free_pct, 1),
                    "used": int(usage.used),
                    "total": int(usage.total),
                }
            )
            worst = free_pct if worst is None else min(worst, free_pct)
        latency = (time.perf_counter() - started) * 1000.0
        if worst is None:
            return ProbeResult("unknown", latency, "unavailable", {"paths": rows})
        status = (
            "down"
            if worst < DISK_DOWN_FREE_PCT
            else "warn"
            if worst < DISK_WARN_FREE_PCT
            else "ok"
        )
        return ProbeResult(
            status, latency, None, {"paths": rows, "free_pct": round(worst, 1)}
        )

    return run


# -- async checks (on the loop) ------------------------------------------------------


class PostgresProbe:
    """``SELECT 1`` on a connection of its own; extras every 5 minutes."""

    def __init__(self, store: Any, clock: Callable[[], float] = time.time):
        self.store = store
        self.clock = clock
        self.extras_at = 0.0
        # The previous pg_stat_database counters and when they were read.
        self._db_stats: Optional[tuple[float, dict[str, int]]] = None

    def _db_delta(self, stats: Any, now: float) -> Optional[dict[str, Any]]:
        """Counter changes since the previous extras (none after a reset)."""
        if not isinstance(stats, dict):
            return None
        current = {k: int(v) for k, v in stats.items() if isinstance(v, (int, float))}
        previous, self._db_stats = self._db_stats, (now, current)
        if previous is None:
            return None
        at, before = previous
        keys = ("xact_commit", "xact_rollback", "deadlocks", "temp_bytes")
        if any(
            k not in current or k not in before or current[k] < before[k] for k in keys
        ):
            return None  # stats were reset, or a counter is missing
        delta = {
            "window_s": round(now - at),
            "commits": current["xact_commit"] - before["xact_commit"],
            "rollbacks": current["xact_rollback"] - before["xact_rollback"],
            "deadlocks": current["deadlocks"] - before["deadlocks"],
            "temp_bytes": current["temp_bytes"] - before["temp_bytes"],
        }
        reads = current.get("blks_read", 0) - before.get("blks_read", 0)
        hits = current.get("blks_hit", 0) - before.get("blks_hit", 0)
        if reads >= 0 and hits >= 0 and reads + hits > 0:
            delta["cache_hit_pct"] = round(hits / (reads + hits) * 100.0, 1)
        return delta

    async def __call__(self, timeout: float) -> ProbeResult:
        now = self.clock()
        extras = now - self.extras_at >= PG_EXTRAS_INTERVAL_S
        data = await self.store.postgres_health(extras=extras, timeout_s=timeout)
        latency = float(data.get("latency_ms") or 0.0)
        detail: dict[str, Any] = {
            "connect_ms": round(float(data.get("connect_ms") or 0), 1)
        }
        if extras and "max_connections" in data:
            self.extras_at = now
            used = int(data.get("connections") or 0)
            limit = int(data.get("max_connections") or 0)
            blocking = []
            blocked_max = 0.0
            for row in data.get("blocking") or []:
                entry = {
                    "pid": row.get("pid"),
                    "blocked_by": list(row.get("blocked_by") or []),
                    "duration_s": round(float(row.get("duration_s") or 0.0), 1),
                    "state": row.get("state"),
                    "wait_event_type": row.get("wait_event_type"),
                    "wait_event": row.get("wait_event"),
                    "statement_type": row.get("statement_type"),
                    "fingerprint": row.get("fingerprint"),
                    "query_length": row.get("query_length"),
                }
                if entry["blocked_by"]:
                    blocked_max = max(blocked_max, entry["duration_s"])
                blocking.append(entry)
            detail.update(
                {
                    "connections": used,
                    "max_connections": limit,
                    "connections_pct": round(used / limit * 100.0, 1)
                    if limit
                    else None,
                    "db_size_bytes": data.get("db_size_bytes"),
                    "server_version": data.get("server_version"),
                    "states": data.get("states") or {},
                    "blocked": sum(1 for b in blocking if b["blocked_by"]),
                    "blocked_max_s": round(blocked_max, 1),
                    "blocking": blocking,
                    "extras_at": _iso(now),
                }
            )
            delta = self._db_delta(data.get("db_stats"), now)
            if delta is not None:
                detail["db_delta"] = delta
        return ProbeResult(_slow(latency, PG_WARN_MS), latency, None, detail)


class PassiveLlm:
    """LLM provider health from real calls (Insights), never a paid probe.

    The 15-minute p95 is compared with the 7-day same-hour baseline the
    hourly baseline job stores (``llm_p95_baseline``): more than twice the
    usual latency warns, as does any p95 over 60 s.
    """

    BASELINE_TTL_S = 300.0

    def __init__(
        self,
        app_state: Any,
        binding: Optional[str],
        host: Optional[str],
        store: Any = None,
    ):
        self.app_state = app_state
        self.binding = binding
        self.host = host
        self.store = store
        self._baseline: Any = None
        self._baseline_at = 0.0
        # The llm_p95 thresholds with the KV overrides, read with the baseline.
        self.thresholds: dict[str, Any] = dict(METRIC_THRESHOLDS)

    async def _baseline_ms(self, timeout: float) -> Optional[float]:
        if self.store is None:
            return None
        now = time.time()
        if now - self._baseline_at >= self.BASELINE_TTL_S:
            try:
                self._baseline = await asyncio.wait_for(
                    self.store.kv_get(BASELINE_KEY, None), timeout
                )
                stored = await asyncio.wait_for(
                    self.store.kv_get("thresholds", None), timeout
                )
                self.thresholds = merge_thresholds(METRIC_THRESHOLDS, stored)
                self._baseline_at = now
            except Exception as e:
                logger.debug("LLM p95 baseline unreadable (%s)", type(e).__name__)
                return None
        entry = baseline_for(self._baseline, now)
        return entry["p95_ms"] if entry else None

    async def __call__(self, timeout: float) -> ProbeResult:
        detail: dict[str, Any] = {
            "window": "15m",
            "binding": self.binding,
            "target": display_target(self.host),
        }
        insights = getattr(self.app_state, "insights", None)
        if insights is None:
            return ProbeResult("unknown", None, "insights_unavailable", detail)
        end = datetime.now(timezone.utc)
        data = await asyncio.wait_for(
            insights.store.llm_health(end - timedelta(minutes=15), end), timeout
        )
        calls = int(data.get("calls") or 0)
        rate = data.get("error_rate")
        if rate is None:
            errors = int(data.get("errors") or 0)
            rate = errors / calls if calls else 0.0
        p95 = data.get("p95_ms")
        detail.update(
            {
                "calls": calls,
                "errors": data.get("errors"),
                "error_rate": round(float(rate), 4),
                "p95_ms": p95,
            }
        )
        if calls < LLM_MIN_CALLS:
            detail["low_volume"] = True
            return ProbeResult("ok", None, None, detail)
        if rate > LLM_DOWN_RATE:
            return ProbeResult("down", None, "error_rate", detail)
        baseline = await self._baseline_ms(timeout)
        if baseline is not None:
            detail["baseline_p95_ms"] = baseline
            if isinstance(p95, (int, float)):
                detail["p95_ratio"] = round(p95 / baseline, 2)
        slow = isinstance(p95, (int, float)) and (
            p95 > LLM_SLOW_P95_MS
            or llm_p95_severity(p95, baseline, self.thresholds) is not None
        )
        if rate > LLM_WARN_RATE or slow:
            return ProbeResult("warn", None, None, detail)
        return ProbeResult("ok", None, None, detail)


# -- configuration -> specs ----------------------------------------------------------


def _base_url(host: Optional[str], strip_suffixes: Iterable[str] = ()) -> Optional[str]:
    if not host or not isinstance(host, str):
        return None
    text = host.strip()
    if not text.lower().startswith(("http://", "https://")):
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    path = parts.path.rstrip("/")
    changed = True
    while changed:
        changed = False
        for suffix in strip_suffixes:
            if path.endswith(suffix):
                path = path[: -len(suffix)].rstrip("/")
                changed = True
    netloc = parts.netloc.rsplit("@", 1)[-1]
    return f"{parts.scheme}://{netloc}{path}"


def _milvus_healthz(uri: str) -> Optional[str]:
    override = os.environ.get("OPS_MILVUS_HEALTHZ_URL", "").strip()
    if override:
        return override
    try:
        parts = urlsplit(uri)
    except ValueError:
        return None
    if not parts.hostname:
        return None
    scheme = "https" if parts.scheme == "https" else "http"
    return f"{scheme}://{parts.hostname}:9091/healthz"


def _docling_endpoint() -> Optional[str]:
    endpoint = os.environ.get("DOCLING_ENDPOINT", "").strip().rstrip("/")
    suffix = "/v1/convert/file/async"
    if endpoint.endswith(suffix):
        endpoint = endpoint[: -len(suffix)]
    return endpoint or None


def _keycloak_issuer(args: Any) -> Optional[str]:
    """The SSO issuer when sign-in through it is usable (the same test as
    :func:`lightrag.api.oauth2.get_keycloak_client`)."""
    if not (
        getattr(args, "oauth2_enabled", False)
        and getattr(args, "oauth2_client_id", None)
        and getattr(args, "oauth2_client_secret", None)
    ):
        return None
    issuer = getattr(args, "oauth2_issuer", None)
    return issuer if isinstance(issuer, str) and issuer.startswith("http") else None


# Storage backends kept in this server's working directory, and those that
# live in Postgres (whose own card covers them).
_LOCAL_STORES = frozenset(
    {"NetworkXStorage", "NanoVectorDBStorage", "FaissVectorDBStorage"}
)
_POSTGRES_STORES = frozenset({"PGGraphStorage", "PGVectorStorage"})


def _store_name(storage: str) -> str:
    """``MongoGraphStorage`` -> ``MongoGraph``: a readable card name."""
    if storage.endswith("Storage") and len(storage) > len("Storage"):
        return storage[: -len("Storage")]
    return storage


def _unprobed_store(dep_id: str, name: str, kind: str, why: str) -> ProbeSpec:
    """A graph or vector store card without a probe, so the gap is visible.

    ``why`` is a reason code or the storage class name it is derived from.
    """
    if why in _POSTGRES_STORES:
        reason = "covered_by_postgres"
    elif why in _LOCAL_STORES:
        reason = "local_storage"
    elif why == "not_configured":
        reason = why
    else:
        reason = "no_probe"
    return ProbeSpec(dep_id, name, kind, probed=False, unprobed_reason=reason)


def build_specs(
    args: Any,
    *,
    store: Any,
    app_state: Any,
    disk_paths: Iterable[str] = (),
) -> list[ProbeSpec]:
    """Dependencies this server uses, from its storage and endpoint settings."""
    specs: list[ProbeSpec] = []
    storages = {
        getattr(args, name, None)
        for name in (
            "kv_storage",
            "doc_status_storage",
            "graph_storage",
            "vector_storage",
        )
    }
    if store is not None and callable(getattr(store, "postgres_health", None)):
        specs.append(
            ProbeSpec(
                "postgres",
                "PostgreSQL",
                "database",
                timeout_s=5.0,
                target=None,
                arun=PostgresProbe(store),
            )
        )

    graph = getattr(args, "graph_storage", None)
    if graph in ("Neo4JStorage", "MemgraphStorage"):
        neo4j = graph == "Neo4JStorage"
        prefix = "NEO4J" if neo4j else "MEMGRAPH"
        section = "neo4j" if neo4j else "memgraph"
        dep_id, name = ("neo4j", "Neo4j") if neo4j else ("memgraph", "Memgraph")
        uri = _ini_setting(
            f"{prefix}_URI",
            section,
            "uri",
            None if neo4j else "bolt://localhost:7687",
        )
        if uri:
            specs.append(
                ProbeSpec(
                    dep_id,
                    name,
                    "graph",
                    target=display_target(uri),
                    run=bolt_probe(
                        uri,
                        _ini_setting(f"{prefix}_USERNAME", section, "username"),
                        _ini_setting(f"{prefix}_PASSWORD", section, "password"),
                    ),
                )
            )
        else:
            specs.append(_unprobed_store(dep_id, name, "graph", "not_configured"))
    elif isinstance(graph, str) and graph:
        specs.append(_unprobed_store("graph", _store_name(graph), "graph", graph))

    vector = getattr(args, "vector_storage", None)
    if vector == "MilvusVectorDBStorage":
        uri = _ini_setting("MILVUS_URI", "milvus", "uri")
        if uri and str(uri).lower().startswith(
            ("http://", "https://", "tcp://", "grpc://")
        ):
            healthz = _milvus_healthz(uri)
            specs.append(
                ProbeSpec(
                    "milvus",
                    "Milvus",
                    "vector",
                    target=display_target(uri),
                    run=milvus_probe(
                        uri,
                        healthz_url=healthz or "",
                        user=_ini_setting("MILVUS_USER", "milvus", "user"),
                        password=_ini_setting("MILVUS_PASSWORD", "milvus", "password"),
                        token=_ini_setting("MILVUS_TOKEN", "milvus", "token"),
                    ),
                )
            )
            for dep_id, name in (("etcd", "etcd"), ("minio", "MinIO")):
                specs.append(ProbeSpec(dep_id, name, "vector_backend", probed=False))
        else:
            specs.append(
                _unprobed_store("milvus", "Milvus", "vector", "not_configured")
            )
    elif vector == "QdrantVectorDBStorage":
        url = os.environ.get("QDRANT_URL", "").strip()
        if url.lower().startswith(("http://", "https://")):
            api_key = os.environ.get("QDRANT_API_KEY", "")
            healthz = url.rstrip("/") + "/healthz"
            probe = http_health_probe(healthz)

            def qdrant(timeout: float, _probe=probe, _url=healthz, _key=api_key):
                if not _key:
                    return _probe(timeout)
                started = time.perf_counter()
                code, _ = _http_get(_url, timeout, headers={"api-key": _key})
                latency = (time.perf_counter() - started) * 1000.0
                detail = {"target": display_target(_url), "http_status": code}
                if code == 200:
                    return ProbeResult(_slow(latency), latency, None, detail)
                return ProbeResult("down", latency, _http_class(code), detail)

            specs.append(
                ProbeSpec(
                    "qdrant", "Qdrant", "vector", target=display_target(url), run=qdrant
                )
            )
        else:
            specs.append(
                _unprobed_store("qdrant", "Qdrant", "vector", "not_configured")
            )
    elif isinstance(vector, str) and vector:
        specs.append(_unprobed_store("vector", _store_name(vector), "vector", vector))

    if any(isinstance(s, str) and s.startswith("Redis") for s in storages):
        uri = os.environ.get("REDIS_URI", "").strip()
        if uri:
            specs.append(
                ProbeSpec(
                    "redis",
                    "Redis",
                    "kv",
                    target=display_target(uri),
                    run=redis_probe(uri),
                )
            )

    embed = _base_url(getattr(args, "embedding_binding_host", None), ("/v1",))
    if embed:
        specs.append(
            ProbeSpec(
                "embed",
                "Embedding",
                "embed",
                target=display_target(embed),
                run=model_endpoint_probe(embed),
            )
        )
    rerank_binding = getattr(args, "rerank_binding", None)
    if rerank_binding and rerank_binding != "null":
        rerank = _base_url(
            getattr(args, "rerank_binding_host", None), ("/rerank", "/v1")
        )
        if rerank:
            specs.append(
                ProbeSpec(
                    "rerank",
                    "Rerank",
                    "rerank",
                    target=display_target(rerank),
                    run=model_endpoint_probe(rerank),
                )
            )
    llm_host = getattr(args, "llm_binding_host", None)
    specs.append(
        ProbeSpec(
            "llm",
            "LLM provider",
            "llm",
            target=display_target(llm_host) if isinstance(llm_host, str) else None,
            arun=PassiveLlm(
                app_state, getattr(args, "llm_binding", None), llm_host, store=store
            ),
            passive=True,
        )
    )

    docling = _docling_endpoint()
    if docling:
        specs.append(
            ProbeSpec(
                "docling",
                "Docling",
                "parser",
                target=display_target(docling),
                run=http_health_probe(f"{docling}/health"),
            )
        )

    try:
        from lightrag.api.provider_alerts import SmtpSettings

        smtp = SmtpSettings.from_env()
    except Exception:
        smtp = None
    if smtp is not None and smtp.host:
        specs.append(
            ProbeSpec(
                "smtp",
                "SMTP",
                "mail",
                interval_s=SMTP_INTERVAL_S,
                timeout_s=float(min(max(int(smtp.timeout or 10), 3), 15)),
                target=f"{smtp.host}:{smtp.port}",
                run=smtp_probe(smtp),
            )
        )

    issuer = _keycloak_issuer(args)
    if issuer:
        specs.append(
            ProbeSpec(
                "idp",
                "Sign-in (IdP)",
                "idp",
                interval_s=IDP_INTERVAL_S,
                target=display_target(issuer),
                run=idp_probe(issuer),
            )
        )

    targets = parse_tls_targets(
        os.environ.get("OPS_TLS_PROBE_TARGETS", DEFAULT_TLS_TARGETS)
    )
    if targets:
        specs.append(
            ProbeSpec(
                "tls",
                "TLS certificates",
                "tls",
                interval_s=TLS_INTERVAL_S,
                timeout_s=float(min(15, 4 * len(targets))),
                target=", ".join(f"{h}:{p}" for h, p in targets),
                run=tls_probe(targets),
            )
        )

    paths = [p for p in dict.fromkeys(disk_paths) if p]
    if paths:
        specs.append(
            ProbeSpec("disk", "Disk", "disk", target=None, run=disk_probe(paths))
        )
    return specs


def recheck_limit(spec: ProbeSpec) -> tuple[str, float]:
    """The host-wide rate-limit claim (name, interval) for re-checking ``spec``.

    SMTP shares its claim with the SMTP test, so neither route can be used to
    open sessions with the mail relay more than once a minute.
    """
    if spec.id == "smtp":
        return "smtp_test", SMTP_TEST_INTERVAL_S
    if spec.interval_s > PROBE_INTERVAL_S:
        return f"recheck:{spec.id}", max(RECHECK_INTERVAL_S, SLOW_RECHECK_INTERVAL_S)
    return f"recheck:{spec.id}", RECHECK_INTERVAL_S


# -- the runner -----------------------------------------------------------------------


class ProbeRunner:
    """Runs the probes; one per process, cycling only while it holds the lease."""

    def __init__(
        self,
        specs: Iterable[ProbeSpec],
        *,
        store: Any,
        holder: str,
        pid: Optional[int] = None,
        threads: int = PROBE_THREADS,
        clock: Callable[[], float] = time.time,
    ):
        self.specs: dict[str, ProbeSpec] = {s.id: s for s in specs}
        self.store = store
        self.holder = holder
        self.pid = pid or os.getpid()
        self.threads = threads
        self.clock = clock
        self.latest: dict[str, dict[str, Any]] = {}
        self.cycle: Optional[dict[str, Any]] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._executor_pid: Optional[int] = None
        # Per dependency: the last check's future and when a thread took it.
        self._inflight: dict[
            str, tuple[concurrent.futures.Future, dict[str, float]]
        ] = {}
        self._claims: dict[str, float] = {}
        self._claims_lock = threading.Lock()
        self._last_cleanup = 0.0

    @property
    def ids(self) -> list[str]:
        return list(self.specs)

    # -- executor ------------------------------------------------------------
    def executor(self) -> concurrent.futures.ThreadPoolExecutor:
        pid = os.getpid()
        if self._executor is None or self._executor_pid != pid:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads, thread_name_prefix="ops-probe"
            )
            self._executor_pid = pid
            self._inflight.clear()
        return self._executor

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            # A stuck client cannot be interrupted; do not wait for it.
            executor.shutdown(wait=False, cancel_futures=True)

    def _stuck_threads(self) -> int:
        """Threads still busy with a check that ran past its own deadline."""
        now = time.monotonic()
        stuck = 0
        for dep, (future, started) in self._inflight.items():
            spec = self.specs.get(dep)
            limit = (spec.timeout_s if spec else DEFAULT_TIMEOUT_S) + WAIT_GRACE_S
            at = started.get("at")
            if not future.done() and at is not None and now - at > limit:
                stuck += 1
        return stuck

    async def _run_in_thread(self, spec: ProbeSpec) -> ProbeResult:
        """Run a blocking check on the probe executor.

        The deadline counts from when a thread picks the check up, so a
        check queued behind others in the same cycle is not timed out early;
        one that waits longer than ``QUEUE_WAIT_S`` for a thread is reported
        as ``executor_busy``.
        """
        loop = asyncio.get_running_loop()
        picked_up = asyncio.Event()
        started: dict[str, float] = {}

        def job() -> ProbeResult:
            started["at"] = time.monotonic()
            loop.call_soon_threadsafe(picked_up.set)
            return spec.run(spec.timeout_s)

        future = self.executor().submit(job)
        self._inflight[spec.id] = (future, started)
        waiter = asyncio.wrap_future(future)
        try:
            await asyncio.wait_for(picked_up.wait(), QUEUE_WAIT_S)
        except asyncio.TimeoutError:
            if future.cancel():
                self._inflight.pop(spec.id, None)
                waiter.cancel()
                return ProbeResult("unknown", None, "executor_busy", {})
            # Picked up just now: the event is set right after the start time.
            await picked_up.wait()
        return await asyncio.wait_for(waiter, spec.timeout_s + WAIT_GRACE_S)

    # -- one probe -------------------------------------------------------------
    async def execute(self, spec: ProbeSpec) -> ProbeResult:
        """Run one check within its deadline. Never raises."""
        started = time.perf_counter()
        try:
            if spec.arun is not None:
                result = await asyncio.wait_for(
                    spec.arun(spec.timeout_s), spec.timeout_s + WAIT_GRACE_S
                )
            else:
                current = self._inflight.get(spec.id)
                if current is not None and not current[0].done():
                    limit = spec.timeout_s + WAIT_GRACE_S
                    at = current[1].get("at")
                    running_s = time.monotonic() - at if at is not None else 0.0
                    if running_s > limit:
                        return ProbeResult(
                            "down",
                            None,
                            "probe_hung",
                            {"note": f"previous check running for {running_s:.0f}s"},
                        )
                    # The same check is running now (a re-check and the cycle
                    # met): share its result rather than start a second one.
                    wait_s = limit - running_s + (QUEUE_WAIT_S if at is None else 0)
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(asyncio.wrap_future(current[0])),
                            max(0.05, wait_s),
                        )
                    except asyncio.CancelledError:
                        # The owner gave up on a check still queued for a
                        # thread and cancelled it. That is not a cancellation
                        # of this task: report the busy executor instead.
                        if current[0].cancelled() and not _being_cancelled():
                            return ProbeResult("unknown", None, "executor_busy", {})
                        raise
                elif self._stuck_threads() >= self.threads:
                    return ProbeResult("unknown", None, "executor_busy", {})
                else:
                    result = await self._run_in_thread(spec)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return ProbeResult("down", spec.timeout_s * 1000.0, "timeout", {})
        except Exception as e:
            return ProbeResult(
                "down", (time.perf_counter() - started) * 1000.0, classify_error(e), {}
            )
        if not isinstance(result, ProbeResult) or result.status not in STATUSES:
            return ProbeResult("unknown", None, "invalid_result", {})
        return result

    # -- records ---------------------------------------------------------------
    def _record(
        self,
        spec: ProbeSpec,
        result: ProbeResult,
        previous: Optional[dict[str, Any]],
        now: float,
    ) -> dict[str, Any]:
        previous = previous if isinstance(previous, dict) else {}
        healthy = result.status in ("ok", "warn")
        failures = 0 if healthy else int(previous.get("consecutive_failures") or 0) + 1
        slow = (
            int(previous.get("consecutive_slow") or 0) + 1
            if result.status == "warn"
            else 0
        )
        detail = safe_detail(result.detail or {})
        if spec.id == "postgres" and "extras_at" not in detail:
            # Extras are refreshed every 5 minutes; keep the last ones.
            old = previous.get("detail") or {}
            for key in (
                "connections",
                "max_connections",
                "connections_pct",
                "db_size_bytes",
                "server_version",
                "states",
                "blocked",
                "blocked_max_s",
                "blocking",
                "extras_at",
                "db_delta",
            ):
                if key in old:
                    detail[key] = old[key]
        if isinstance(detail.get("waiting"), int):
            since = (previous.get("detail") or {}).get("waiting_since")
            if detail["waiting"] > VLLM_WAITING_WARN:
                detail["waiting_since"] = since or _iso(now)
        last_ok = now if healthy else previous.get("last_ok_epoch")
        return {
            "id": spec.id,
            "status": result.status,
            "latency_ms": None
            if result.latency_ms is None
            else round(float(result.latency_ms), 1),
            "error_class": result.error_class,
            "detail": detail,
            "checked_epoch": now,
            "checked_at": _iso(now),
            "last_ok_epoch": last_ok,
            "last_ok_at": _iso(last_ok),
            "consecutive_failures": failures,
            "consecutive_slow": slow,
        }

    async def read_latest(self) -> tuple[dict[str, dict[str, Any]], Optional[dict]]:
        """Latest record per dependency and the last cycle, host-wide when stored."""
        if self.store is None:
            return dict(self.latest), self.cycle
        keys = [KV_PREFIX + dep for dep in self.specs] + [CYCLE_KEY]
        values = await self.store.kv_get_many(keys)
        records = {
            dep: values[KV_PREFIX + dep]
            for dep in self.specs
            if isinstance(values.get(KV_PREFIX + dep), dict)
        }
        cycle = values.get(CYCLE_KEY)
        return records, cycle if isinstance(cycle, dict) else None

    def _due(self, spec: ProbeSpec, previous: Optional[dict], now: float) -> bool:
        if not spec.checkable:
            return False
        checked = (previous or {}).get("checked_epoch")
        return (
            not isinstance(checked, (int, float))
            or now - checked >= spec.interval_s - 1.0
        )

    async def run_cycle(
        self,
        lease: Any = None,
        *,
        only: Optional[Iterable[str]] = None,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Check what is due (or ``only`` these, now) and publish the results."""
        started = time.perf_counter()
        now = self.clock()
        wanted = set(only) if only is not None else None
        try:
            previous, _ = await self.read_latest()
        except Exception as e:
            logger.warning("Ops probe state could not be read (%s)", type(e).__name__)
            previous = dict(self.latest)
        due = [
            spec
            for spec in self.specs.values()
            if (wanted is None or spec.id in wanted)
            and spec.checkable
            and (force or self._due(spec, previous.get(spec.id), now))
        ]
        if not due:
            return {}
        results = await asyncio.gather(*(self.execute(spec) for spec in due))
        records = {
            spec.id: self._record(spec, result, previous.get(spec.id), now)
            for spec, result in zip(due, results)
        }
        if lease is not None and not await lease.still_held():
            return {}
        self.latest.update(records)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
        full_cycle = wanted is None
        if full_cycle:
            self.cycle = {
                "at": _iso(now),
                "at_epoch": now,
                "duration_ms": duration_ms,
                "holder": self.holder,
                "holder_pid": self.pid,
                "interval_s": PROBE_INTERVAL_S,
            }
        if self.store is not None:
            await self._publish(records, full_cycle)
        return records

    async def _publish(
        self, records: dict[str, dict[str, Any]], full_cycle: bool
    ) -> None:
        patches = {KV_PREFIX + dep: record for dep, record in records.items()}
        if full_cycle and self.cycle is not None:
            patches[CYCLE_KEY] = self.cycle
        try:
            await self.store.kv_merge_many(patches)
            await self.store.probe_write(
                [
                    {
                        "dependency": dep,
                        "status": record["status"],
                        "latency_ms": record["latency_ms"],
                        "error_class": record["error_class"],
                        "detail": record["detail"],
                    }
                    for dep, record in records.items()
                ]
            )
        except Exception as e:
            logger.warning(
                "Ops probe results could not be stored (%s)", type(e).__name__
            )
            return
        now = self.clock()
        if full_cycle and now - self._last_cleanup >= CLEANUP_INTERVAL_S:
            self._last_cleanup = now
            try:
                await self.store.probe_cleanup(RETENTION_DAYS)
            except Exception as e:
                logger.debug("Ops probe cleanup failed (%s)", type(e).__name__)

    # -- on-demand checks ----------------------------------------------------------
    async def claim(self, name: str, interval_s: float) -> bool:
        """Host-wide rate limit: True at most once per ``interval_s`` per name."""
        if self.store is not None:
            try:
                return bool(
                    await self.store.try_lease(f"ops:{name}", uuid4().hex, interval_s)
                )
            except Exception as e:
                logger.warning("Ops rate limit unavailable (%s)", type(e).__name__)
                return False
        now = time.monotonic()
        with self._claims_lock:
            if now - self._claims.get(name, -1e9) < interval_s:
                return False
            self._claims[name] = now
            return True

    async def probe_now(self, dep: str) -> Optional[dict[str, Any]]:
        records = await self.run_cycle(only=[dep], force=True)
        return records.get(dep)
