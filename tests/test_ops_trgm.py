"""``pg_trgm`` log search beyond 24 h (PR-9, V7).

The extension is detected, never created unless ``OPS_PG_TRGM_CREATE=true``;
the background builder adds the GIN trigram index only while the extension
is installed; a text search may span the full 7 days only while that index
is valid (and only for a ``q`` holding a run of 3 letters or digits).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from lightrag.api.ops import indexes, logs
from lightrag.api.ops.service import OpsBadRequest
from lightrag.api.ops.store import MemoryOpsStore
from tests.ops_fixtures import build_app
from tests.test_ops_indexes import FakeConn

pytestmark = pytest.mark.offline

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _filter(days: float, q: str = "timeout", **kw):
    start = (NOW - timedelta(days=days)).isoformat()
    return logs.parse_filter({"from": start, "to": NOW.isoformat(), "q": q}, NOW, **kw)


# -- the 24 h cap ------------------------------------------------------------------------


def test_q_stays_capped_at_24h_without_the_index():
    assert _filter(1).q == "timeout"
    with pytest.raises(OpsBadRequest) as err:
        _filter(3)
    assert err.value.code == "search_window_too_long"


def test_q_spans_7_days_when_the_index_is_valid():
    f = _filter(7, text_window_s=logs.MAX_WINDOW_S)
    assert f.q == "timeout" and (f.end - f.start) == timedelta(days=7)
    with pytest.raises(OpsBadRequest) as err:
        _filter(8, text_window_s=logs.MAX_WINDOW_S)
    assert err.value.code == "window_too_long"
    # Too short for a trigram: the index cannot serve it, the cap stays.
    with pytest.raises(OpsBadRequest):
        _filter(3, q="ab", text_window_s=logs.MAX_WINDOW_S)
    # 3+ characters but no run of 3 letters or digits: pg_trgm extracts no
    # selective trigram, so the 24 h cap stays.
    for q in ("a b", "1.2", "-->", "a_b", "  ab  "):
        with pytest.raises(OpsBadRequest) as err:
            _filter(3, q=q, text_window_s=logs.MAX_WINDOW_S)
        assert err.value.code == "search_window_too_long", q
        assert _filter(1, q=q, text_window_s=logs.MAX_WINDOW_S).q == q.strip()
    # A run of 3 anywhere in q is enough, in any script.
    for q in ("x db1", "error: abc", "\u5b9e\u4f53\u540d"):
        assert _filter(7, q=q, text_window_s=logs.MAX_WINDOW_S).q == q
    # No q: the window was never capped.
    assert _filter(6, q="").q is None


def test_trgm_searchable_rule():
    assert logs.trgm_searchable("abc") and logs.trgm_searchable("a 123 b")
    assert not logs.trgm_searchable("ab") and not logs.trgm_searchable("")
    assert not logs.trgm_searchable(None) and not logs.trgm_searchable("1.2")
    assert not logs.trgm_searchable("a__b")


async def test_text_search_window_follows_index_validity():
    from lightrag.api.ops.service import OpsService
    from lightrag.api.ops.snapshot import NamespaceReader
    from types import SimpleNamespace

    store = MemoryOpsStore()
    service = OpsService(
        store=store, reader=NamespaceReader(), app_state=SimpleNamespace()
    )
    assert await logs.text_search_window(service) == logs.TEXT_SEARCH_WINDOW_S
    store.trgm_state = {"available": True, "installed": True, "index_valid": False}
    assert await logs.text_search_window(service) == logs.TEXT_SEARCH_WINDOW_S
    store.trgm_state["index_valid"] = True
    assert await logs.text_search_window(service) == logs.MAX_WINDOW_S
    store.fail_reads = RuntimeError("down")
    assert await logs.text_search_window(service) == logs.TEXT_SEARCH_WINDOW_S
    unavailable = OpsService(store=None, reader=NamespaceReader(), app_state=None)
    assert (await logs.trgm_state(unavailable))["index_valid"] is False


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def test_logs_route_lifts_the_cap_only_with_a_valid_index():
    store = MemoryOpsStore()
    app = build_app(store)
    start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    params = {"from": start, "q": "timeout"}
    async with _client(app) as client:
        response = await client.get("/admin/ops/logs", params=params)
        assert response.status_code == 400
        assert response.json()["detail"] == "search_window_too_long"
        status = (await client.get("/admin/ops/status")).json()
        assert status["features"]["log_trgm"] is False

        store.trgm_state = {"available": True, "installed": True, "index_valid": True}
        app.state.ops.service.cache.clear()
        response = await client.get("/admin/ops/logs", params=params)
        assert response.status_code == 200, response.text
        assert response.json()["text_search_window_s"] == logs.MAX_WINDOW_S
        status = (await client.get("/admin/ops/status")).json()
        assert status["features"]["log_trgm"] is True


# -- detection and the index builder -----------------------------------------------------------


async def test_trgm_spec_only_while_the_extension_is_installed(monkeypatch):
    monkeypatch.delenv(indexes.TRGM_CREATE_ENV, raising=False)
    conn = FakeConn(trgm={"available": True, "installed": False})
    assert indexes.LOG_TRGM_SPEC not in await indexes.index_specs(conn)
    assert not [s for s in conn.statements if "EXTENSION" in s]

    conn = FakeConn(trgm={"available": True, "installed": True})
    specs = await indexes.index_specs(conn)
    assert specs[-1] is indexes.LOG_TRGM_SPEC
    assert "USING gin (msg gin_trgm_ops)" in indexes.LOG_TRGM_SPEC.create_sql()
    assert indexes.LOG_TRGM_SPEC.create_sql().startswith(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ops_logs_msg_trgm"
    )


async def test_extension_created_only_when_allowed_and_available(monkeypatch):
    monkeypatch.setenv(indexes.TRGM_CREATE_ENV, "true")
    conn = FakeConn(trgm={"available": True, "installed": False})
    assert indexes.LOG_TRGM_SPEC in await indexes.index_specs(conn)
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in conn.statements

    conn = FakeConn(trgm={"available": False, "installed": False})
    assert indexes.LOG_TRGM_SPEC not in await indexes.index_specs(conn)
    assert not [s for s in conn.statements if "EXTENSION" in s]


async def test_a_failing_create_extension_never_raises(monkeypatch):
    class Refusing(FakeConn):
        async def execute(self, sql, *args):
            if sql.startswith("CREATE EXTENSION"):
                raise PermissionError("permission denied to create extension")
            await super().execute(sql, *args)

    monkeypatch.setenv(indexes.TRGM_CREATE_ENV, "1")
    conn = Refusing(trgm={"available": True, "installed": False})
    assert await indexes.prepare_trgm(conn) is False


async def test_builder_creates_the_trigram_index_concurrently():
    conn = FakeConn(
        tables={"lightrag_ops_logs", "lightrag_doc_status"},
        trgm={"available": True, "installed": True},
    )
    specs = await indexes.index_specs(conn)
    outcome = await indexes.build_indexes(conn, specs)
    assert outcome[indexes.LOG_TRGM_INDEX] == "created"
    assert any(
        s.startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ops_logs_msg_trgm")
        for s in conn.statements
    )


async def test_column_gated_index_waits_for_its_column():
    conn = FakeConn(columns=set())
    outcome = await indexes.build_indexes(conn)
    assert outcome["idx_lightrag_usage_calls_req"] == "skipped"
    assert "idx_lightrag_usage_calls_req" not in {
        s.name for s in await indexes.missing_indexes(conn)
    }
    conn = FakeConn()
    outcome = await indexes.build_indexes(conn)
    assert outcome["idx_lightrag_usage_calls_req"] == "created"


async def test_building_the_trigram_index_drops_the_cached_state():
    conn = FakeConn(
        tables={"lightrag_ops_logs", "lightrag_doc_status"},
        trgm={"available": True, "installed": True},
    )

    class Store:
        _trgm = ({"available": True, "installed": True, "index_valid": False}, 0.0)

        def dedicated_connection(self):
            class Ctx:
                async def __aenter__(self):
                    return conn

                async def __aexit__(self, *exc):
                    return False

            return Ctx()

    store = Store()
    outcome, retry = await indexes._build_once(store, None)
    assert outcome[indexes.LOG_TRGM_INDEX] == "created"
    assert store._trgm is None
