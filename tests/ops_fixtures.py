"""Shared fixtures for the Operations console route tests (not a test module)."""

from __future__ import annotations

import importlib
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import FastAPI

# The API config parses sys.argv on first use; keep pytest's flags out of it.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
ops_routes = importlib.import_module("lightrag.api.routers.ops_routes")
sys.argv = _original_argv

from lightrag.api.dependencies import UserInfo, require_admin  # noqa: E402
from lightrag.api.ops.context import RequestIdMiddleware  # noqa: E402
from lightrag.api.ops.service import OpsService  # noqa: E402
from lightrag.api.ops.snapshot import NamespaceReader  # noqa: E402
from lightrag.api.ops.store import MemoryOpsStore  # noqa: E402

SECRET_FILE = "Senate minutes 900101-13-5678.pdf"
SECRET_EMAIL = "someone.private@unimas.my"
SECRET_KEY = "sk-test-SECRET1234567890abcdef"
RATE_LIMIT_ERROR = (
    "RateLimitError: Error code: 429 - request {request} for "
    + SECRET_EMAIL
    + " key "
    + SECRET_KEY
)


def utc_naive(minutes_ago: float) -> datetime:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(
        tzinfo=None
    )


def admin() -> UserInfo:
    return UserInfo(username="root", role="admin", workspace_id="w", metadata={})


def seeded_store() -> MemoryOpsStore:
    """A small host: running, stuck, failed, cancelled, dup and paused work."""
    store = MemoryOpsStore()
    add = store.add_doc
    # fsktm: processing now, plus one doc stuck in processing for 3 hours.
    add(
        "fsktm",
        "doc-aaaa0001",
        "processing",
        updated_at=utc_naive(2),
        file_path=SECRET_FILE,
    )
    add(
        "fsktm",
        "doc-aaaa0002",
        "processing",
        updated_at=utc_naive(190),
        file_path=SECRET_FILE,
    )
    add(
        "fsktm",
        "doc-aaaa0003",
        "pending",
        updated_at=utc_naive(150),
        file_path="memo.docx",
    )
    add(
        "fsktm",
        "doc-aaaa0004",
        "failed",
        updated_at=utc_naive(30),
        file_path=SECRET_FILE,
        error_msg=RATE_LIMIT_ERROR.format(request="7f3a9c21"),
    )
    add(
        "fsktm",
        "doc-aaaa0005",
        "failed",
        updated_at=utc_naive(40),
        error_msg="User cancelled during entity extraction",
    )
    add("fsktm", "dup-aaaa0006", "failed", updated_at=utc_naive(5), error_msg="dup")
    add(
        "fsktm",
        "doc-aaaa0007",
        "processed",
        updated_at=utc_naive(300),
        created_at=utc_naive(320),
        file_path=SECRET_FILE,
        metadata={
            "parsing_start_time": int(time.time()) - 19000,
            "processing_start_time": int(time.time()) - 18900,
            "processing_end_time": int(time.time()) - 18000,
        },
    )
    # space1: email batch running.
    add(
        "space1",
        "doc-bbbb0001",
        "parsing",
        updated_at=utc_naive(1),
        file_path="email_1a2b3c4d__master.txt",
    )
    add(
        "space1",
        "doc-bbbb0002",
        "failed",
        updated_at=utc_naive(50),
        error_msg=RATE_LIMIT_ERROR.format(request="0b1c2d3e"),
    )
    # hr_office: paused by the provider, documents waiting.
    add(
        "hr_office",
        "doc-cccc0001",
        "pending",
        updated_at=utc_naive(20),
        file_path="policy.pdf",
    )
    store.owner_rows["fsktm"] = "a.bin@unimas.my"
    store.owner_rows["space1"] = "s.one@unimas.my"
    store.kv["email_queue_summary"] = {
        "scanned_at": time.time() - 10,
        "holder": "w1",
        "total": 3,
        "workspaces": 1,
        "bad": 1,
        "longest_pending_s": 3000.0,
        "items": [
            {
                "workspace": "space1",
                "pending": 3,
                "bad": 1,
                "first_pending_at": time.time() - 3000,
            },
        ],
    }
    store.kv["resync:space1"] = {
        "first_pending_at": time.time() - 3000,
        "attempts": 7,
        "last_result_class": "slot_busy",
    }
    store.usage_rows.append(
        {
            "ts": datetime.now(timezone.utc) - timedelta(minutes=30),
            "kind": "deployment",
            "workspace": None,
            "purpose": None,
            "detail": {"version": "0295", "secret": "x"},
        }
    )
    store.alert_rows.append(
        {
            "id": 1,
            "rule": "stale_docs",
            "fingerprint": "stale_docs",
            "severity": "serious",
            "state": "firing",
            "title": "1 document(s) in progress for over 60 min",
            "scope": {"count": 1, "threshold_min": 60},
            "link": "#/?ops=1&otab=processing&osub=stuck",
            "runbook": None,
            "first_seen": datetime.now(timezone.utc) - timedelta(minutes=40),
            "last_seen": datetime.now(timezone.utc) - timedelta(minutes=1),
            "resolved_at": None,
            "acked_by": None,
            "acked_at": None,
            "ack_note": None,
            "last_emailed_at": None,
        }
    )
    store.audit_rows.append(
        {
            "id": 1,
            "ts": datetime.now(timezone.utc) - timedelta(minutes=10),
            "actor": "khairil",
            "action": "cancel",
            "workspace": "fsktm",
            "target": "fsktm",
            "phase": "result",
            "outcome": "cancellation_requested",
        }
    )
    return store


def shared_state() -> dict[str, Any]:
    now = time.time()
    return {
        "fsktm:pipeline_status": {
            "busy": True,
            "job_name": f"{SECRET_FILE[:20]}...[2 files]",
            "job_start": datetime.now(timezone.utc).isoformat(),
            "docs": 4,
            "cur_batch": 1,
            "latest_message": f"Extracting {SECRET_FILE.replace(' ', '_')} for {SECRET_EMAIL}",
            "history_messages": [f"msg {i} {SECRET_EMAIL}" for i in range(5)],
        },
        "space1:pipeline_status": {
            "busy": True,
            "job_name": "Email batch",
            "history_messages": [],
        },
        "hr_office:pipeline_status": {
            "busy": False,
            "provider_rejection": {
                "role": "extract",
                "kind": "budget",
                "status_code": 402,
                "message": f"Payment required for {SECRET_KEY}",
                "since": now - 600,
                "retry_at": now + 600,
                "probe_doc": None,
            },
            "history_messages": [],
        },
    }


class FakeRegistry:
    def __init__(self, owners: dict[str, str]):
        self.owners = owners

    async def get_by_workspace(self, workspace: str):
        email = self.owners.get(workspace)
        if email is None:
            return None
        return SimpleNamespace(email=email, identity=f"sso:{workspace}")


def build_app(
    store: Optional[MemoryOpsStore],
    *,
    default_rag: bool = True,
    insights: Any = None,
    registry: Any = None,
    manager: Any = None,
    kv_storage: str = "PGKVStorage",
    provider_store: Any = None,
) -> FastAPI:
    from lightrag.api.ops.pauses import OwnProviderCache

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware, trusted_proxies="")
    app.include_router(ops_routes.router)
    rag = SimpleNamespace(
        kv_storage=kv_storage,
        workspace="",
        doc_status=SimpleNamespace(workspace="default"),
    )
    ops_routes.configure_ops(app, rag=rag, workers=None, version="0295")
    if default_rag:
        app.state.default_rag = rag
    app.state.insights = insights
    app.state.workspace_registry = registry
    app.state.workspace_manager = manager
    if store is not None:
        app.state.ops = SimpleNamespace(
            service=OpsService(
                store=store,
                reader=NamespaceReader(),
                app_state=app.state,
                own_providers=OwnProviderCache(provider_store)
                if provider_store
                else None,
            )
        )
    app.dependency_overrides[require_admin] = admin
    return app


WORKER_ID = "0123456789abcdef0123456789abcdef"
OTHER_WORKER_ID = "fedcba9876543210fedcba9876543210"


def health_runner(store: Optional[MemoryOpsStore]):
    """A probe runner with fixed specs: Postgres (async), Neo4j and SMTP
    (threads), the passive LLM card and an unprobed etcd card."""
    from lightrag.api.ops.probes import ProbeResult, ProbeRunner, ProbeSpec

    async def postgres(timeout):
        return ProbeResult("ok", 2.0, None, {"connect_ms": 1.0})

    def neo4j(timeout):
        return ProbeResult("ok", 12.0, None, {"target": "neo4j:7687"})

    def smtp(timeout):
        return ProbeResult(
            "ok", 120.0, None, {"target": "mail.example:587", "starttls": True}
        )

    async def llm(timeout):
        return ProbeResult("ok", None, None, {"calls": 0, "low_volume": True})

    specs = [
        ProbeSpec("postgres", "PostgreSQL", "database", arun=postgres),
        ProbeSpec("neo4j", "Neo4j", "graph", target="neo4j:7687", run=neo4j),
        ProbeSpec("smtp", "SMTP", "mail", interval_s=900, run=smtp),
        ProbeSpec("llm", "LLM provider", "llm", arun=llm, passive=True),
        ProbeSpec("etcd", "etcd", "vector_backend", probed=False),
    ]
    return ProbeRunner(specs, store=store, holder=WORKER_ID)


def seed_health(app: FastAPI, store: MemoryOpsStore) -> None:
    """Probe results, probe history and two worker rows, as if the probe
    lease holder and two heartbeats had run."""
    now = time.time()
    runner = health_runner(store)
    service = app.state.ops.service
    service.probes = runner
    service.worker_id = WORKER_ID
    for dep, status, latency in (
        ("postgres", "ok", 2.0),
        ("neo4j", "down", None),
        ("smtp", "ok", 120.0),
    ):
        store.kv[f"probe:{dep}"] = {
            "id": dep,
            "status": status,
            "latency_ms": latency,
            "error_class": "timeout" if status == "down" else None,
            "detail": {"target": f"{dep}:1"},
            "checked_epoch": now - 5,
            "checked_at": datetime.fromtimestamp(now - 5, timezone.utc).isoformat(),
            "last_ok_epoch": now - 5 if status != "down" else now - 600,
            "last_ok_at": None,
            "consecutive_failures": 4 if status == "down" else 0,
            "consecutive_slow": 0,
        }
    store.kv["probe_cycle"] = {
        "at": datetime.fromtimestamp(now - 5, timezone.utc).isoformat(),
        "at_epoch": now - 5,
        "duration_ms": 140.0,
        "holder": WORKER_ID,
        "holder_pid": 812,
        "interval_s": 30.0,
    }
    for minutes in (1, 31, 61):
        for dep, status in (("postgres", "ok"), ("neo4j", "down")):
            store.probe_rows.append(
                {
                    "ts": datetime.now(timezone.utc) - timedelta(minutes=minutes),
                    "dependency": dep,
                    "status": status,
                    "latency_ms": 3 if status == "ok" else None,
                    "error_class": None if status == "ok" else "timeout",
                    "detail": {},
                }
            )
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    for worker_id, pid, age in ((WORKER_ID, 812, 5), (OTHER_WORKER_ID, 811, 600)):
        store.worker_rows[worker_id] = {
            "worker_id": worker_id,
            "host": "lightrag-1",
            "pid": pid,
            "pid_create_time": now - 7200,
            "mode": "uvicorn",
            "version": "0295",
            "started_at": started,
            "heartbeat_at": datetime.now(timezone.utc) - timedelta(seconds=age),
            "exited_at": None,
            "superseded": False,
            "leases_held": ["ops:probes"] if worker_id == WORKER_ID else [],
            "snapshot": {
                "rss": 2_100_000_000,
                "cpu_pct": 12.0,
                "loop_lag_ms": 3.0,
                "pool": {"size": 12, "idle": 10, "max": 50, "wait_ms": 3.0},
                "host": {"mem_used": 7, "mem_limit": 16, "load": [3.2, 2.9, 2.7]},
            },
        }
        store.sample_rows[(worker_id, int(now // 60 * 60))] = {
            "worker_id": worker_id,
            "minute": int(now // 60 * 60),
            "cpu_pct": 12.0,
            "rss": 2_100_000_000,
            "loop_lag_ms": 3,
            "pool_wait_ms": 1,
        }


RUN_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
ORPHAN_RUN_ID = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
DONE_RUN_ID = "11112222333344445555666677778888"


def seed_runs(store: MemoryOpsStore, worker_id: str = WORKER_ID) -> None:
    """Run tracking rows: a running run in fsktm (owned by ``worker_id``), a
    finished one, an orphaned one in space1, and classified failures."""
    import asyncio

    now = time.time()
    base = {
        "trigger": "upload",
        "worker_id": worker_id,
        "pid": 4242,
        "finished_at": None,
        "docs_total": 4,
        "docs_done": 1,
        "docs_failed": 1,
        "stage": "extract",
        "label": "Senate minutes [2 files]",
        "error_kind": None,
        "error_summary": None,
        "request_id": None,
    }
    rows = [
        {
            **base,
            "run_id": RUN_ID,
            "workspace": "fsktm",
            "kind": "process",
            "status": "running",
            "started_at": now - 600,
            "heartbeat_at": now - 5,
            "progress_at": now - 30,
        },
        {
            **base,
            "run_id": DONE_RUN_ID,
            "workspace": "fsktm",
            "kind": "delete",
            "status": "succeeded",
            "started_at": now - 7200,
            "heartbeat_at": now - 7100,
            "progress_at": now - 7100,
            "finished_at": now - 7100,
        },
        {
            **base,
            "run_id": ORPHAN_RUN_ID,
            "workspace": "space1",
            "kind": "email_batch",
            "trigger": "resync",
            "worker_id": OTHER_WORKER_ID,
            "status": "running",
            "started_at": now - 1800,
            "heartbeat_at": now - 900,
            "progress_at": now - 900,
        },
    ]
    failures = [
        {
            "ts": now - 120,
            "workspace": "fsktm",
            "doc_id": "doc-aaaa0004",
            "track_id": "upload_20261103_101010_aaaa0004",
            "run_id": RUN_ID,
            "stage": "extract",
            "error_kind": "rate_limited",
            "http_status": 429,
            "exc_type": "openai.RateLimitError",
            "message": "RateLimitError: Error code: 429 <redacted>",
            "attempt": 2,
            "model": "gpt-4o-mini",
        },
        {
            "ts": now - 60,
            "workspace": "fsktm",
            "doc_id": "doc-aaaa0005",
            "track_id": None,
            "run_id": RUN_ID,
            "stage": "extract",
            "error_kind": "cancelled",
            "http_status": None,
            "exc_type": "PipelineCancelledException",
            "message": "User cancelled",
            "attempt": 1,
            "model": "gpt-4o-mini",
        },
        {
            "ts": now - 30,
            "workspace": "space1",
            "doc_id": "doc-bbbb0002",
            "track_id": None,
            "run_id": ORPHAN_RUN_ID,
            "stage": "parse",
            "error_kind": "timeout",
            "http_status": None,
            "exc_type": "httpx.ReadTimeout",
            "message": "ReadTimeout docling:5001",
            "attempt": 1,
            "model": None,
        },
    ]

    async def write() -> None:
        await store.runs_write(rows)
        await store.doc_failures_write(failures)
        await store.runs_orphan([ORPHAN_RUN_ID])

    asyncio.run(write())


LOG_REQUEST_ID = "3c9e41d0a1b2c3d4e5f60718293a4b5c"
LOG_TRACK_ID = "upload_20261103_101010_aaaa0004"


def seed_logs(store: MemoryOpsStore, worker_id: str = WORKER_ID) -> list[int]:
    """Stored log rows (already redacted, as the sink writes them): an error
    with a traceback in fsktm for a request, run, track and document, a
    warning in space1 and an uvicorn.error line without a workspace."""
    import asyncio

    now = time.time()
    base = {
        "worker_id": worker_id,
        "pid": 4242,
        "client_request_id": None,
        "operation": None,
        "source": None,
        "exc_type": None,
        "exc": None,
        "repeat_count": 1,
    }
    rows = [
        {
            **base,
            "ts": now - 120,
            "level": 40,
            "logger": "lightrag",
            "module": "pipeline",
            "lineno": 2596,
            "workspace": "fsktm",
            "request_id": LOG_REQUEST_ID,
            "run_id": RUN_ID,
            "track_id": LOG_TRACK_ID,
            "doc_id": "doc-aaaa0004",
            "msg": "Failed to extract document doc-aaaa0004: RateLimitError <redacted>",
            "exc_type": "RateLimitError",
            "exc": "Traceback (most recent call last):\n  ...\nRateLimitError: 429",
            "fingerprint": "a1a1a1a1a1a1a1a1",
        },
        {
            **base,
            "ts": now - 60,
            "level": 30,
            "logger": "lightrag",
            "module": "operate",
            "lineno": 1195,
            "workspace": "space1",
            "request_id": None,
            "run_id": None,
            "track_id": None,
            "doc_id": None,
            "msg": "RateLimitError, retry 2/3 in 8s",
            "fingerprint": "b2b2b2b2b2b2b2b2",
            "repeat_count": 4,
        },
        {
            **base,
            "ts": now - 30,
            "level": 40,
            "logger": "uvicorn.error",
            "module": "httptools_impl",
            "lineno": 414,
            "workspace": None,
            "request_id": "0f8d2c1e9a7b4c3d8e6f5a4b3c2d1e0f",
            "run_id": None,
            "track_id": None,
            "doc_id": None,
            "msg": "Exception in ASGI application",
            "exc_type": "RuntimeError",
            "exc": "Traceback (most recent call last):\nRuntimeError: boom",
            "fingerprint": "c3c3c3c3c3c3c3c3",
        },
    ]
    asyncio.run(store.logs_write(rows))
    return [r["id"] for r in store.log_rows]
