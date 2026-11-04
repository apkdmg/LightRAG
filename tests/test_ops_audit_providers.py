"""Audit rows of the server-default provider changes (Operations console,
PR-7): written best effort around the existing routes, by slot name only
(never base URLs, models or keys), and never changing their responses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lightrag.api.ops.service import OpsService
from lightrag.api.ops.snapshot import NamespaceReader
from lightrag.api.ops.store import MemoryOpsStore
from tests.test_system_provider import app as provider_app  # noqa: F401
from tests.test_system_provider import store  # noqa: F401

pytestmark = pytest.mark.offline


def _ops_state(app, ops_store) -> None:
    app.state.ops = SimpleNamespace(
        service=OpsService(
            store=ops_store, reader=NamespaceReader(), app_state=app.state
        )
    )


class BrokenAuditStore(MemoryOpsStore):
    async def audit_write(self, row):
        raise ConnectionError("database gone")


def _rows(ops_store: MemoryOpsStore, action: str) -> list[dict[str, Any]]:
    return [r for r in ops_store.audit_rows if r["action"] == action]


def _pair(ops_store: MemoryOpsStore, action: str) -> tuple[dict, dict]:
    intent, result = _rows(ops_store, action)
    assert intent["phase"] == "intent" and result["phase"] == "result"
    assert result["params"]["intent_id"] == intent["id"]
    return intent, result


def test_server_default_changes_are_audited_without_keys(provider_app):  # noqa: F811
    ops_store = MemoryOpsStore()
    _ops_state(provider_app, ops_store)
    client = TestClient(provider_app)
    response = client.put(
        "/admin/system-provider",
        json={
            "query": {
                "base_url": "https://new.example/v1",
                "api_key": "sk-new-secret-9999",
                "model": "m",
            }
        },
    )
    assert response.status_code == 200
    intent, result = _pair(ops_store, "system_provider_set")
    assert intent["actor"] == "alice" and intent["params"] == {"slots": ["query"]}
    assert result["outcome"] == "ok" and "resumed" in result["params"]
    assert "sk-new-secret" not in str(ops_store.audit_rows)
    assert "new.example" not in str(ops_store.audit_rows)
    response = client.delete("/admin/system-provider", params={"slot": "query"})
    assert response.status_code == 200
    intent, result = _pair(ops_store, "system_provider_clear")
    assert intent["params"] == {"slots": ["query"]}


def test_server_default_failures_are_audited_and_unchanged(provider_app):  # noqa: F811
    ops_store = BrokenAuditStore()
    _ops_state(provider_app, ops_store)
    response = TestClient(provider_app).put(
        "/admin/system-provider",
        json={"query": {"base_url": "https://evil.example/v1", "model": "m"}},
    )
    assert response.status_code == 400
    assert "own API key" in response.json()["detail"]
    ops_store = MemoryOpsStore()
    _ops_state(provider_app, ops_store)
    TestClient(provider_app).put(
        "/admin/system-provider",
        json={"query": {"base_url": "https://evil.example/v1", "model": "m"}},
    )
    assert _rows(ops_store, "system_provider_set")[-1]["outcome"] == "http_400"
