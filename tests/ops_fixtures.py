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
