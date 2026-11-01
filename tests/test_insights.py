"""Offline tests for token usage recording and the Insights analytics.

The ``store`` fixture also runs against real Postgres when
LIGHTRAG_RUN_INTEGRATION=true and POSTGRES_* are set, so both stores stay alike.
"""

import asyncio
import importlib
import os
import struct
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_routes = importlib.import_module("lightrag.api.routers.insights_routes")
_email = importlib.import_module("lightrag.api.routers.email_routes")
sys.argv = _original_argv

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from lightrag.api.dependencies import UserInfo, require_admin  # noqa: E402
from lightrag.api.insights import analytics as an  # noqa: E402
from lightrag.api.insights.backfill import run_backfill  # noqa: E402
from lightrag.api.insights.brief import brief_csv, brief_lines, text_pdf  # noqa: E402
from lightrag.api.insights.recorder import UsageRecorder  # noqa: E402
from lightrag.api.insights.scheduler import InsightsScheduler, note_startup  # noqa: E402
from lightrag.api.insights.service import InsightsService  # noqa: E402
from lightrag.api.insights.store import (  # noqa: E402
    BACKFILL_SOURCE,
    MemoryUsageStore,
    PostgresUsageStore,
)
from lightrag.api.provider_alerts import SmtpSettings, is_own_alert_email  # noqa: E402
from lightrag.lightrag import LightRAG  # noqa: E402
from lightrag.llm_roles import RoleLLMConfig  # noqa: E402
from lightrag.usage import (  # noqa: E402
    PURPOSES,
    SystemEvent,
    UsageRecord,
    add_usage_sink,
    current_usage_scope,
    emit_event,
    remove_usage_sink,
    run_in_usage_scope,
    usage_scope,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, handle_cache  # noqa: E402

pytestmark = pytest.mark.offline

TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 11, 1, 15, 30, tzinfo=TZ)
TODAY = NOW.date()
PRICES = {
    "sys-llm": {"prompt": 0.2, "completion": 0.6},
    "sys-query": {"prompt": 1.0, "completion": 3.0},
    "byo-model": {"prompt": 0.5, "completion": 1.0},
}


class _Tok:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8), dtype=float)


async def _build_rag(tmp_path, **overrides):
    async def default_llm(*a, **k):
        return "ok"

    overrides.setdefault("llm_model_func", default_llm)
    rag = LightRAG(
        working_dir=str(tmp_path / uuid4().hex[:8]),
        workspace=f"ins_{uuid4().hex[:8]}",
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=_embed),
        tokenizer=Tokenizer("test", _Tok()),
        **overrides,
    )
    await rag.initialize_storages()
    return rag


@pytest.fixture
def sink():
    records = []
    add_usage_sink(records.append)
    yield records
    remove_usage_sink(records.append)


def _record(day, purpose="query", workspace="alpha", prompt=1000, completion=200, **kw):
    hour = kw.pop("hour", 10)
    ts = datetime(day.year, day.month, day.day, hour, 5, tzinfo=TZ).timestamp()
    kw.setdefault("operation", "extraction" if purpose == "extract" else "mix")
    kw.setdefault(
        "source", "webui" if purpose in ("query", "keyword") else "file_upload"
    )
    kw.setdefault("model", "sys-query" if purpose == "query" else "sys-llm")
    return UsageRecord(
        workspace=workspace,
        purpose=purpose,
        prompt_tokens=prompt,
        completion_tokens=completion,
        ts=ts,
        latency_ms=kw.pop("latency_ms", 100),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Recording at the role wrapper                                               #
# --------------------------------------------------------------------------- #


def _role(func, binding="openai", model="m1"):
    return RoleLLMConfig(
        func=func,
        max_async=2,
        metadata={"binding": binding, "model": model, "host": "https://h"},
    )


async def test_call_is_recorded_with_provider_figures(tmp_path, sink):
    seen = {}

    async def llm(prompt, **kwargs):
        seen.update(kwargs)
        kwargs["token_tracker"].add_usage(
            {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "reasoning_tokens": 2,
                "cached_tokens": 3,
                "cost": 0.5,
                "finish_reason": "stop",
            }
        )
        return "answer"

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        with usage_scope(operation="mix", source="webui", client="c1", doc_id="doc-1"):
            assert await rag.role_llm_funcs["query"]("q") == "answer"
    finally:
        await rag.finalize_storages()
    [r] = sink
    assert (r.purpose, r.operation, r.source, r.client, r.doc_id) == (
        "query",
        "mix",
        "webui",
        "c1",
        "doc-1",
    )
    assert (
        r.prompt_tokens,
        r.completion_tokens,
        r.reasoning_tokens,
        r.cached_tokens,
        r.cost,
    ) == (11, 5, 2, 3, 0.5)
    assert (r.model, r.billing, r.workspace, r.status, r.estimated) == (
        "m1",
        "system",
        rag.workspace,
        "ok",
        False,
    )


async def test_nothing_changes_without_a_sink(tmp_path):
    seen = {}

    async def llm(prompt, **kwargs):
        seen.update(kwargs)
        return "answer"

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        assert await rag.role_llm_funcs["query"]("q") == "answer"
    finally:
        await rag.finalize_storages()
    assert "token_tracker" not in seen


async def test_binding_without_tracking_is_estimated_and_gets_no_tracker(
    tmp_path, sink
):
    seen = {}

    async def llm(prompt, **kwargs):
        seen.update(kwargs)
        return "four"

    rag = await _build_rag(
        tmp_path, role_llm_configs={"query": _role(llm, binding="ollama")}
    )
    try:
        await rag.role_llm_funcs["query"]("abc", system_prompt="xy")
    finally:
        await rag.finalize_storages()
    assert "token_tracker" not in seen
    [r] = sink
    assert (
        r.estimated and r.prompt_tokens == len("abc\nxy") and r.completion_tokens == 4
    )


async def test_vision_without_a_figure_is_not_measured_not_guessed(tmp_path, sink):
    async def llm(prompt, **kwargs):
        return "a chart"

    rag = await _build_rag(
        tmp_path, role_llm_configs={"vlm": _role(llm, binding="ollama")}
    )
    try:
        await rag.role_llm_funcs["vlm"]("describe", image_inputs=["QUJD"])
    finally:
        await rag.finalize_storages()
    [r] = sink
    assert not r.measured and not r.estimated and r.prompt_tokens == 0


async def test_failed_call_is_recorded_and_still_raises(tmp_path, sink):
    async def llm(prompt, **kwargs):
        raise RuntimeError("down")

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        with pytest.raises(RuntimeError):
            await rag.role_llm_funcs["query"]("q")
    finally:
        await rag.finalize_storages()
    assert sink[0].status == "error"


async def test_truncated_answer_is_marked(tmp_path, sink):
    async def llm(prompt, **kwargs):
        kwargs["token_tracker"].add_usage(
            {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": "length"}
        )
        return "cut"

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        await rag.role_llm_funcs["query"]("q")
    finally:
        await rag.finalize_storages()
    assert sink[0].status == "truncated"


async def test_stream_records_when_finished_with_provider_usage(tmp_path, sink):
    async def llm(prompt, **kwargs):
        async def gen():
            yield "a"
            yield "b"
            kwargs["token_tracker"].add_usage(
                {"prompt_tokens": 7, "completion_tokens": 2}
            )

        return gen()

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        with usage_scope(operation="mix"):
            stream = await rag.role_llm_funcs["query"]("q", stream=True)
        assert sink == []
        assert [c async for c in stream] == ["a", "b"]
    finally:
        await rag.finalize_storages()
    [r] = sink
    assert (r.prompt_tokens, r.completion_tokens, r.operation, r.status) == (
        7,
        2,
        "mix",
        "ok",
    )


async def test_abandoned_stream_gets_an_estimate(tmp_path, sink):
    async def llm(prompt, **kwargs):
        async def gen():
            yield "abc"
            yield "def"

        return gen()

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(llm)})
    try:
        stream = await rag.role_llm_funcs["query"]("hello", stream=True)
        async for _ in stream:
            break
        await stream.aclose()
    finally:
        await rag.finalize_storages()
    [r] = sink
    assert r.status == "abandoned" and r.estimated
    assert r.prompt_tokens == len("hello\n") and r.completion_tokens == 3


async def test_scope_survives_the_queue_workers(tmp_path, sink):
    release = asyncio.Event()

    async def llm(prompt, **kwargs):
        await release.wait()
        kwargs["token_tracker"].add_usage({"prompt_tokens": 1, "completion_tokens": 1})
        return prompt

    rag = await _build_rag(
        tmp_path,
        role_llm_configs={
            "extract": RoleLLMConfig(
                func=llm, max_async=1, metadata={"binding": "openai", "model": "m"}
            )
        },
    )
    try:

        async def one(doc):
            return await run_in_usage_scope(
                rag.role_llm_funcs["extract"](doc), doc_id=doc
            )

        tasks = [asyncio.create_task(one(d)) for d in ("d1", "d2", "d3")]
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(*tasks)
    finally:
        await rag.finalize_storages()
    assert sorted(r.doc_id for r in sink) == ["d1", "d2", "d3"]
    assert current_usage_scope().doc_id is None


async def test_byo_override_bills_the_workspace_key(tmp_path, sink):
    from lightrag.api.workspace_providers import _apply_slot, ProviderSlot
    from pydantic import SecretStr

    async def system(prompt, **kwargs):
        return "s"

    async def byo(prompt, **kwargs):
        return "b"

    rag = await _build_rag(tmp_path, role_llm_configs={"query": _role(system)})
    rag.register_role_llm_builder(lambda role, meta: (byo, {}))
    try:
        await _apply_slot(
            rag,
            ("query",),
            ProviderSlot(base_url="https://b", api_key=SecretStr("k"), model="own"),
        )
        assert await rag.role_llm_funcs["query"]("q") == "b"
    finally:
        await rag.finalize_storages()
    assert (sink[0].billing, sink[0].model) == ("byo", "own")


async def test_cache_hit_is_recorded_with_zero_tokens(tmp_path, sink):
    rag = await _build_rag(tmp_path)
    try:
        key = "default:keywords:abc"
        await rag.llm_response_cache.upsert(
            {key: {"return": "x", "cache_type": "keywords"}}
        )
        with usage_scope(operation="retrieval"):
            hit = await handle_cache(
                rag.llm_response_cache,
                "abc",
                "p",
                mode="default",
                cache_type="keywords",
            )
    finally:
        await rag.finalize_storages()
    assert hit is not None
    [r] = sink
    assert (r.purpose, r.cache_hit, r.prompt_tokens, r.operation, r.workspace) == (
        "keyword",
        True,
        0,
        "retrieval",
        rag.workspace,
    )


async def test_email_image_is_booked_to_its_workspace_on_the_system_key(tmp_path, sink):
    async def vision(prompt, **kwargs):
        kwargs["token_tracker"].add_usage({"prompt_tokens": 9, "completion_tokens": 3})
        return "a chart"

    system = await _build_rag(
        tmp_path, role_llm_configs={"vlm": _role(vision, model="sys-vlm")}
    )
    workspace = await _build_rag(tmp_path)
    image = _email.ParsedAttachment(
        filename="c.png",
        content_type="image/png",
        content=b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", 200, 100),
        is_inline=True,
    )
    email = _email.ParsedEmail(
        message_id="<a@b>",
        from_address="a@b",
        to_addresses=[],
        cc_addresses=[],
        subject="s",
        date=None,
        body_text="b",
        body_html=None,
        inline_images=[image],
        attachments=[],
    )
    try:
        service = _email.EmailIngestionService(
            workspace, _email._system_vision_func(system)
        )
        await service._process_inline_image(image, "b", email, 0)
        await service._process_inline_image(image, "b", email, 0)
    finally:
        await workspace.finalize_storages()
        await system.finalize_storages()
    [call, hit] = sink
    assert (
        call.workspace,
        call.purpose,
        call.operation,
        call.source,
        call.billing,
    ) == (workspace.workspace, "vlm", "email_inline", "email", "system")
    assert (call.prompt_tokens, call.model) == (9, "sys-vlm")
    assert (hit.workspace, hit.purpose, hit.cache_hit) == (
        workspace.workspace,
        "vlm",
        True,
    )


async def test_stream_usage_requested_and_dropped_if_rejected(monkeypatch):
    openai_mod = importlib.import_module("lightrag.llm.openai")
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(dict(kwargs))
            if "stream_options" in kwargs:
                request = SimpleNamespace(method="POST", url="u", headers={})
                raise openai_mod.BadRequestError(
                    "Unknown parameter: stream_options",
                    response=SimpleNamespace(
                        request=request, status_code=400, headers={}
                    ),
                    body=None,
                )

            async def gen():
                yield SimpleNamespace(
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(content="hi", reasoning_content=""),
                        )
                    ],
                )

            return gen()

    class Client:
        chat = SimpleNamespace(completions=Completions())

        async def close(self):
            pass

    monkeypatch.setattr(openai_mod, "create_openai_async_client", lambda **k: Client())
    stream = await openai_mod.openai_complete_if_cache(
        "m",
        "p",
        api_key="k",
        stream=True,
        token_tracker=SimpleNamespace(add_usage=lambda c: None),
    )
    assert [c async for c in stream] == ["hi"]
    assert "stream_options" in calls[0] and "stream_options" not in calls[1]


# --------------------------------------------------------------------------- #
# Masking                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,masked",
    [
        ("Senate minutes, September.pdf", "Sen•••er.pdf"),
        ("Annual report 2025.pdf", "Ann•••.pdf"),
        ("880101135555.pdf", "•••.pdf"),
        ("CV.PDF", "C•••.pdf"),
        ("folder/inner/Budget_review.xlsx", "Bud•••ew.xlsx"),
        ("noextension", "noe•••on"),
    ],
)
def test_mask_name(name, masked):
    assert an.mask_name(name) == masked


def test_document_labels():
    assert an.document_label("doc-a3f1aa", "Senate minutes.pdf") == {
        "kind": "document",
        "label": "Sen•••es.pdf",
        "code": "a3f1",
    }
    assert an.document_label("doc-a3f1aa", None)["kind"] == "deleted"
    assert an.document_label("doc-1", "email_8dfc468c74c7__master.txt") == {
        "kind": "email",
        "label": "",
        "code": "8dfc",
    }
    assert an.document_label(
        "doc-1", "email_8dfc468c74c7__attachment_000_Budget_2026.xlsx.txt"
    ) == {"kind": "email_attachment", "label": "Bud•••.xlsx", "code": "8dfc"}
    assert (
        an.document_label("doc-1", "email_8dfc468c74c7__inline_001_image002.png.txt")[
            "kind"
        ]
        == "email_image"
    )


# --------------------------------------------------------------------------- #
# Plain calculations                                                          #
# --------------------------------------------------------------------------- #


def test_usual_is_the_same_weekday_median():
    day = date(2026, 10, 26)
    history = {
        day - timedelta(days=7 * n): v
        for n, v in zip(range(1, 9), [10, 20, 30, 40, 50, 60, 70, 1000])
    }
    history[day - timedelta(days=1)] = 99999
    assert an.usual_value(history, day) == 45
    assert an.usual_value({day - timedelta(days=7): 5}, day) is None


@pytest.mark.parametrize(
    "args,expected",
    [
        ((100, 100, 10, 10), "normal"),
        ((200, 100, 20, 10), "more_work"),
        ((200, 100, 10, 10), "more_per_unit"),
        ((200, 100, 11, 10, 0.1, 0.4), "cache_misses"),
        ((50, 100, 5, 10), "less_work"),
        ((50, 100, 10, 10), "less_per_unit"),
        ((50, None, 10, 10), "unknown"),
    ],
)
def test_verdict(args, expected):
    assert an.verdict(*args) == expected


def test_forecast_range_widens_and_growth_applies():
    daily = [1000.0] * 70
    plain = an.forecast_series(daily)
    assert len(plain["points"]) == 13
    assert plain["points"][0] == pytest.approx(7000)
    assert plain["low"][12] < plain["low"][0] and plain["high"][12] > plain["high"][0]
    grown = an.forecast_series(daily, growth=0.5)
    assert grown["points"][12] == pytest.approx(10500)
    bulk = an.forecast_series(daily, bulk_tokens=5000)
    assert bulk["points"][3] == pytest.approx(12000) and bulk["points"][
        4
    ] == pytest.approx(7000)


def test_trend_is_clamped():
    daily = [10.0] * 28 + [1000.0] * 28
    assert an.forecast_series(daily)["trend"] == 0.03


def test_budget_days_and_purchase():
    assert an.days_until_spent(70, [70, 70]) == 7
    assert an.days_until_spent(105, [70, 70]) == 10.5
    assert an.days_until_spent(1000, [70, 70]) is None
    assert an.suggested_purchase(120, 100) == 50
    assert an.suggested_purchase(90, 100) == 0


# --------------------------------------------------------------------------- #
# Store and service                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    if request.param == "memory":
        yield MemoryUsageStore(TZ)
        return
    if os.getenv("LIGHTRAG_RUN_INTEGRATION", "").lower() != "true" or not os.getenv(
        "POSTGRES_HOST"
    ):
        pytest.skip("Needs LIGHTRAG_RUN_INTEGRATION=true and POSTGRES_*")
    pg = PostgresUsageStore(None, TZ)
    await pg.initialize()
    conn_factory = pg._db
    for table in ("CALLS", "HOURLY", "DAILY", "EVENTS", "KV"):
        await conn_factory.execute(f"TRUNCATE TABLE LIGHTRAG_USAGE_{table}")
    yield pg
    await pg.finalize()


def _seed(store_records, day_list, **kw):
    return [_record(d, **kw) for d in day_list]


def _history(days=70):
    """Steady weekday usage in two workspaces, with BYO on one purpose in one."""
    records = []
    for i in range(days, 0, -1):
        d = TODAY - timedelta(days=i)
        records.append(_record(d, "extract", "alpha", 7000, 1000))
        records.append(_record(d, "keyword", "alpha", 300, 50, model="sys-llm"))
        records.append(_record(d, "query", "alpha", 4000, 1000))
        records.append(
            _record(
                d,
                "query",
                "beta",
                2000,
                500,
                model="byo-model",
                billing="byo",
                client="moodle",
            )
        )
        records.append(
            _record(
                d,
                "vlm",
                "beta",
                800,
                200,
                operation="email_inline",
                source="email",
                model="sys-llm",
            )
        )
        records.append(_record(d, "query", "alpha", 0, 0, cache_hit=True))
    records.append(_record(TODAY, "query", "alpha", 400, 100, hour=9))
    return records


async def _service(store, records=None, settings=True):
    await store.write(_history() if records is None else records, [])
    service = InsightsService(store, now=lambda: NOW)
    if settings:
        await service.save_settings({"prices": PRICES})
    return service


async def test_rollups_cost_and_work(store):
    service = await _service(store)
    day = TODAY - timedelta(days=1)
    rows = {r["purpose"]: r for r in await service._rows(day, day, ["purpose"])}
    q = rows["query"]
    assert q["tokens"] == 7500
    assert q["cost"] == pytest.approx(
        4000 / 1e6 * 1 + 1000 / 1e6 * 3 + 2000 / 1e6 * 0.5 + 500 / 1e6 * 1
    )
    assert q["byo_cost"] == pytest.approx(2000 / 1e6 * 0.5 + 500 / 1e6 * 1)
    assert q["work"] == 3  # two calls and one cache hit
    assert rows["extract"]["work"] == 1


async def test_unpriced_model_is_counted_not_free(store):
    service = await _service(store, [_record(TODAY, "query", model="mystery")])
    [row] = await service._rows(TODAY, TODAY, [])
    assert row["cost"] == 0 and row["unpriced_tokens"] == 1200


async def test_provider_reported_cost_wins_over_the_table(store):
    service = await _service(store, [_record(TODAY, "query", cost=0.25)])
    [row] = await service._rows(TODAY, TODAY, [])
    assert row["cost"] == pytest.approx(0.25) and row["unpriced_tokens"] == 0


async def test_hourly_and_daily_buckets_use_local_time(store):
    late = datetime(2026, 10, 30, 23, 30, tzinfo=TZ).timestamp()
    await store.write(
        [
            UsageRecord(
                workspace="a",
                purpose="query",
                operation="mix",
                prompt_tokens=5,
                ts=late,
            )
        ],
        [],
    )
    rows = await store.rollup(
        "day", date(2026, 10, 30), date(2026, 10, 31), group_by=["bucket"]
    )
    assert [r["bucket"] for r in rows] == [date(2026, 10, 30)]
    hours = await store.rollup(
        "hour",
        datetime(2026, 10, 30, 0, tzinfo=TZ),
        datetime(2026, 10, 31, 0, tzinfo=TZ),
        group_by=["bucket"],
    )
    assert hours[0]["bucket"].astimezone(TZ).hour == 23


async def test_second_write_adds_to_the_rollup(store):
    await store.write([_record(TODAY)], [])
    await store.write([_record(TODAY)], [])
    [row] = await store.rollup("day", TODAY, TODAY)
    assert row["calls"] == 2 and row["prompt_tokens"] == 2000


async def test_overview_shape_and_totals(store):
    service = await _service(store)
    out = await service.overview(30)
    assert out["learning"]["days_collected"] == 71
    by = {p["id"]: p for p in out["purposes"]}
    assert set(by) == set(PURPOSES)
    assert by["query"]["billing"] == "system"
    assert out["totals"]["byo_cost"] > 0
    assert len(out["chart"]["points"]) == 30
    assert out["chart"]["points"][-1]["partial"] is True
    assert out["models"][0]["id"] == "sys-llm"
    assert {w["id"] for w in out["workspaces"]} == {"alpha", "beta"}
    today = await service.overview(1)
    assert today["chart"]["grain"] == "hour" and len(today["chart"]["points"]) == 24
    assert today["chart"]["points"][9]["tokens"]["query"] == 500
    assert today["chart"]["points"][20]["future"] is True


async def test_overview_scoped_to_a_workspace_compares_with_others(store):
    service = await _service(store)
    out = await service.overview(30, "alpha")
    query = next(p for p in out["purposes"] if p["id"] == "query")
    assert query["others_per_unit"] is not None
    assert "models" not in out


async def test_usual_by_now_is_not_understated_for_a_quiet_workspace(store):
    # Each past day is active in a different hour, so every hour's median is zero.
    # The usual total by this hour is still about 12000 tokens a day.
    records = []
    for week in range(1, 9):
        day = TODAY - timedelta(days=7 * week)
        records.append(_record(day, "query", "quiet", 10000, 2000, hour=week % 3 + 1))
    records.append(_record(TODAY, "query", "quiet", 10000, 2000, hour=1))
    service = await _service(store, records)
    out = await service.overview(7, "quiet")
    query = next(p for p in out["purposes"] if p["id"] == "query")
    assert query["status"]["code"] == "normal", query["status"]


async def test_status_compares_with_usual_by_this_hour(store):
    records = _history()
    # Usual by 15:00 is 5000 tokens of Query in alpha; today is five times that.
    records += [_record(TODAY, "query", "alpha", 25000, 5000, hour=9)]
    service = await _service(store, records)
    out = await service.overview(7, "alpha")
    query = next(p for p in out["purposes"] if p["id"] == "query")
    assert query["status"]["code"] == "above"
    assert out["attention"]["above"] == ["query"]


async def test_first_weeks_say_learning(store):
    service = await _service(
        store, [_record(TODAY - timedelta(days=i)) for i in range(5)]
    )
    out = await service.overview(7)
    assert (
        out["learning"]["anomalies_ready"] is False
        and out["learning"]["anomalies_in_days"] == 9
    )
    assert out["learning"]["forecast_in_days"] == 23
    assert all(p["status"]["code"] == "learning" for p in out["purposes"])
    assert out["chart"]["points"][0]["gap"] is True
    assert (await service.anomalies())["items"] == []
    assert (await service.forecast())["ready"] is False


async def test_purpose_page(store):
    service = await _service(store)
    await store.write(
        [],
        [
            SystemEvent(
                kind="model_change",
                purpose="query",
                detail={"old": "a", "new": "b"},
                ts=(NOW - timedelta(days=10)).timestamp(),
            )
        ],
    )
    page = await service.purpose_page("query", 30)
    assert page["byo_share"] > 0 and page["byo_workspaces"] == ["beta"]
    assert [m["id"] for m in page["models"]] == ["sys-query", "byo-model"]
    assert page["operations"][0]["id"] == "mix"
    assert page["model_change"]["ready"] is True and page["model_change"]["to"] == "b"
    assert page["split"]["prompt"] > 0


async def test_model_change_waits_seven_days(store):
    service = await _service(store)
    await store.write(
        [],
        [
            SystemEvent(
                kind="model_change",
                purpose="extract",
                detail={"old": "a", "new": "b"},
                ts=(NOW - timedelta(days=2)).timestamp(),
            )
        ],
    )
    change = await service.model_change_check("extract")
    assert (
        change["ready"] is False
        and change["ready_on"] == (TODAY + timedelta(days=5)).isoformat()
    )


async def test_workspaces_and_page(store):
    service = await _service(store)
    if isinstance(store, MemoryUsageStore):
        store.status_counts = {"alpha": {"processed": 10, "pending": 2, "failed": 1}}
        store.document_rows = [
            {
                "day": TODAY - timedelta(days=2),
                "workspace": "alpha",
                "source": "file_upload",
                "documents": 4,
                "chunks": 40,
            }
        ]
        store.paths = {("alpha", "doc-abcd1234"): "Annual report 2025.pdf"}
        await store.write(
            [
                _record(
                    TODAY - timedelta(days=1), "extract", "alpha", doc_id="doc-abcd1234"
                )
            ],
            [],
        )
        service._cache.clear()
    rows = (await service.workspaces(30))["rows"]
    assert {r["id"] for r in rows} >= {"alpha", "beta"}
    assert rows[0]["cost"] >= rows[-1]["cost"]
    page = await service.workspace_page("alpha", 30)
    assert page["rank"] == 1 and page["share_cost"] > 0
    if isinstance(store, MemoryUsageStore):
        alpha = next(r for r in rows if r["id"] == "alpha")
        assert alpha["needs_look"] is True and alpha["failed_documents"] == 1
        assert (
            page["documents"] == 4
            and page["live"]["waiting"] == 2
            and page["live"]["failed"] == 1
        )
        labels = [i["display"] for i in page["items"]]
        assert {"kind": "document", "label": "Ann•••.pdf", "code": "abcd"} in labels


async def test_full_document_name_never_appears(store):
    service = await _service(store, [_record(TODAY, "extract", doc_id="doc-abcd1234")])
    if isinstance(store, MemoryUsageStore):
        store.paths = {("alpha", "doc-abcd1234"): "Secret salary review.pdf"}
    out = await service.breakdown("document", 7, {"workspace": "alpha"})
    assert "Secret salary" not in str(out)
    assert out["rows"][0]["key"] == "doc-abcd1234"


async def test_breakdown_groups_and_filters(store):
    service = await _service(store)
    by_purpose = await service.breakdown("purpose", 30, {})
    assert by_purpose["rows"][0]["key"] == "extract"
    filtered = await service.breakdown("workspace", 30, {"purpose": "vlm"})
    assert [r["key"] for r in filtered["rows"]] == ["beta"]
    clients = await service.breakdown("client", 30, {"workspace": "beta"})
    assert clients["rows"][0]["key"] == "moodle"
    one_day = await service.breakdown(
        "purpose", 30, {}, day=(TODAY - timedelta(days=3)).isoformat()
    )
    assert sum(r["tokens"] for r in one_day["rows"]) < sum(
        r["tokens"] for r in by_purpose["rows"]
    )
    with pytest.raises(ValueError):
        await service.breakdown("nonsense", 30, {})


async def test_calls_carry_no_text(store):
    service = await _service(store, [_record(TODAY, "query", doc_id="d")])
    [row] = await service.calls(7, {"workspace": "alpha"})
    assert row["prompt_tokens"] == 1000 and row["purpose"] == "query"
    assert not any("text" in key or key == "prompt" for key in row)


async def _spike_service(store):
    records = _history()
    spike = TODAY - timedelta(days=3)
    records += [_record(spike, "extract", "alpha", 60000, 6000) for _ in range(1)]
    records += [
        _record(spike, "extract", "alpha", 0, 0, cache_hit=True) for _ in range(9)
    ]
    return await _service(store, records), spike


async def test_anomaly_found_with_its_workspace_and_state_is_kept(store):
    service, spike = await _spike_service(store)
    items = (await service.anomalies())["items"]
    match = [
        a for a in items if a["day"] == spike.isoformat() and a["workspace"] == "alpha"
    ]
    assert match, items
    a = match[0]
    assert a["purposes"][0]["id"] == "extract" and a["extra_cost"] > 0
    assert a["kind"] == "more_work" and a["state"] == "open"
    await service.set_anomaly_state(a["id"], "expected", "root")
    assert (
        next(x for x in (await service.anomalies())["items"] if x["id"] == a["id"])[
            "state"
        ]
        == "expected"
    )
    assert not [x for x in await service.anomalies_open(None) if x["id"] == a["id"]]


async def test_lasting_change_is_one_anomaly_with_a_stable_id(store):
    start = TODAY - timedelta(days=70)
    days = [start + timedelta(days=i) for i in range(71)]
    lasting = days[-6:-1]
    records = [
        _record(
            d,
            "extract",
            "alpha",
            1000 * (3 if d in lasting else 1),
            0,
            operation="extraction",
        )
        for d in days
        for _ in range(10)
    ]
    service = await _service(store, records)
    run = [
        a
        for a in (await service.anomalies())["items"]
        if a["kind"] == "more_per_unit" and lasting[-1].isoformat() in a["run_days"]
    ]
    assert len(run) == 1, run
    assert run[0]["run_days"] == [d.isoformat() for d in lasting]
    assert run[0]["day"] == lasting[-1].isoformat()
    assert run[0]["id"].startswith(lasting[0].isoformat())
    # Every day of the run reaches the calendar and the day explanation.
    calendar = {c["day"]: c["severity"] for c in await service.calendar()}
    assert all(calendar[d.isoformat()] for d in lasting)
    day = await service.explain_day(lasting[1])
    assert [a["id"] for a in day["anomalies"]] == [run[0]["id"]]


async def test_anomalies_follow_the_workspace_filter(store):
    service, spike = await _spike_service(store)
    assert not [
        a
        for a in (await service.anomalies(workspace="beta"))["items"]
        if a["workspace"] == "alpha"
    ]
    assert (await service.anomalies(workspace="alpha"))["items"]
    assert not (await service.anomalies(purpose="vlm"))["items"]


async def test_backfill_estimates_stay_out_of_anomalies(store):
    records = _history()
    spike = TODAY - timedelta(days=3)
    records.append(
        _record(
            spike, "extract", "alpha", 900000, 0, source=BACKFILL_SOURCE, estimated=True
        )
    )
    service = await _service(store, records)
    assert not [
        a for a in (await service.anomalies())["items"] if a["day"] == spike.isoformat()
    ]


async def test_provider_rejection_is_critical(store):
    service = await _service(store)
    ts = (NOW - timedelta(days=2)).timestamp()
    await store.write(
        [],
        [
            SystemEvent(
                kind="provider_rejection",
                workspace="alpha",
                purpose="extract",
                detail={"status": 402, "byo": False},
                ts=ts,
            )
        ],
    )
    items = (await service.anomalies())["items"]
    assert [
        a
        for a in items
        if a["kind"] == "provider_rejected" and a["severity"] == "critical"
    ]


async def test_explain_day(store):
    service, spike = await _spike_service(store)
    out = await service.explain_day(spike)
    extract = next(p for p in out["purposes"] if p["id"] == "extract")
    assert extract["verdict"] == "more_work" and extract["tokens"] == 74000
    assert out["workspaces"][0]["id"] == "alpha"
    assert out["anomalies"]
    assert (await service.explain_day(spike, "beta"))["workspaces"] == []


async def test_work_view_keeps_ingestion_and_questions_apart(store):
    service = await _service(store)
    ingest = await service.work("ingest", 30)
    assert set(ingest["days"][0]["tokens"]) <= {"extract", "vlm"}
    questions = await service.work("questions", 30)
    assert set(questions["days"][0]["tokens"]) <= {"keyword", "query"}
    assert {s["id"] for s in questions["sources"]} == {"webui"}
    assert questions["days"][0]["units"] > 0


async def test_forecast_counts_byo_and_shows_each_share(store):
    service = await _service(store)
    await service.save_settings(
        {"budget": {"amount": 0.5, "start": (TODAY - timedelta(days=10)).isoformat()}}
    )
    out = await service.forecast()
    assert out["ready"] is True
    assert len(out["points"]) == 13 and len(out["actual"]) == 10
    assert out["next90"]["byo_cost"] > 0
    assert out["next90"]["cost"] > out["next90"]["byo_cost"]
    assert all(p["system"] < p["value"] for p in out["points"])
    budget = out["budget"]
    assert budget["runs_out"] <= budget["runs_out_without_byo"]
    assert budget["left"] == pytest.approx(0.5 - budget["spent"], abs=1e-3)
    query = next(r for r in out["rows"] if r["id"] == "query")
    assert 0 < query["byo_share"] < 1


async def test_forecast_what_if(store):
    service = await _service(store)
    base = await service.forecast()
    more = await service.forecast(question_change=1.0)
    extract_base = next(r for r in base["rows"] if r["id"] == "extract")
    extract_more = next(r for r in more["rows"] if r["id"] == "extract")
    query_base = next(r for r in base["rows"] if r["id"] == "query")
    query_more = next(r for r in more["rows"] if r["id"] == "query")
    assert extract_more["tokens"] == pytest.approx(extract_base["tokens"])
    assert query_more["tokens"] > query_base["tokens"]
    cheaper = await service.forecast(swap="query")
    assert (
        next(r for r in cheaper["rows"] if r["id"] == "query")["cost"]
        < query_base["cost"]
    )


async def test_bill_check_leaves_byo_out(store):
    service = await _service(store)
    await service.save_settings({"invoices": {"2026-10": 0.5}})
    [bill] = (await service.forecast())["bill"]
    first, last = date(2026, 10, 1), date(2026, 10, 31)
    all_rows = await service._rows(first, last, [])
    assert bill["system_cost"] == pytest.approx(
        all_rows[0]["cost"] - all_rows[0]["byo_cost"], abs=1e-3
    )
    assert bill["difference"] is not None


async def test_retention_and_anonymize(store):
    old = _record(TODAY - timedelta(days=100), doc_id="doc-1", client="c")
    fresh = _record(TODAY, doc_id="doc-2", client="c")
    await store.write(
        [old, fresh],
        [
            SystemEvent(
                kind="byo_change",
                workspace="alpha",
                detail={"slot": "query"},
                ts=fresh.ts,
            )
        ],
    )
    await store.purge()
    start, end = NOW - timedelta(days=400), NOW + timedelta(days=1)
    assert len(await store.calls(start, end)) == 1
    assert (await store.rollup("day", TODAY - timedelta(days=200), TODAY))[0][
        "calls"
    ] == 2
    await store.anonymize_workspace("alpha", "deleted-abc")
    assert await store.calls(start, end, filters={"workspace": "alpha"}) == []
    [call] = await store.calls(start, end, filters={"workspace": "deleted-abc"})
    assert call["doc_id"] is None and call.get("client") is None
    assert (
        await store.rollup("day", TODAY, TODAY, filters={"workspace": "deleted-abc"})
    )[0]["calls"] == 1
    assert (await store.events(start, end))[0]["workspace"] == "deleted-abc"


async def test_lease_has_one_holder(store):
    key = f"t{uuid4().hex[:6]}"
    assert await store.try_lease(key, 60) is True
    assert await store.try_lease(key, 60) is False


async def test_settings_survive(store):
    service = await _service(store, [])
    saved = await service.save_settings({"rate_limit_tpm": 5000})
    assert saved["rate_limit_tpm"] == 5000 and saved["prices"] == PRICES
    assert (await InsightsService(store, now=lambda: NOW).settings())[
        "rate_limit_tpm"
    ] == 5000


# --------------------------------------------------------------------------- #
# Recorder                                                                    #
# --------------------------------------------------------------------------- #


async def test_recorder_flushes_records_and_events():
    store = MemoryUsageStore(TZ)
    recorder = UsageRecorder(store, flush_seconds=0.01)
    recorder.start()
    from lightrag.usage import emit_usage

    emit_usage(_record(TODAY))
    emit_event("cache_cleared", workspace="alpha")
    await recorder.stop()
    assert len(store.call_rows) == 1 and store.event_rows[0].kind == "cache_cleared"


async def test_failed_write_is_retried_without_losing_rows():
    store = MemoryUsageStore(TZ)
    real = store.write
    attempts = []

    async def flaky(records, events):
        attempts.append(len(records))
        if len(attempts) == 1:
            raise RuntimeError("db down")
        await real(records, events)

    store.write = flaky
    recorder = UsageRecorder(store)
    recorder.record(_record(TODAY))
    await recorder.flush()
    assert store.call_rows == [] and len(recorder._records) == 1
    await recorder.flush()
    assert len(store.call_rows) == 1 and recorder._records == []


async def test_full_buffer_counts_lost_rows():
    store = MemoryUsageStore(TZ)
    recorder = UsageRecorder(store, max_buffer=1)
    recorder.record(_record(TODAY))
    recorder.record(_record(TODAY))
    await recorder.flush()
    assert await store.kv_get("lost_rows") == 1


async def test_deleted_workspace_keeps_numbers_under_a_hash():
    store = MemoryUsageStore(TZ)
    recorder = UsageRecorder(store)
    recorder.record(_record(TODAY, workspace="secret_person"))
    recorder.event(SystemEvent(kind="workspace_deleted", workspace="secret_person"))
    await recorder.flush()
    workspaces = await store.known_workspaces()
    assert "secret_person" not in workspaces and workspaces[0].startswith("deleted-")
    assert store.event_rows[0].workspace.startswith("deleted-")


# --------------------------------------------------------------------------- #
# Backfill, startup, alerts, brief                                            #
# --------------------------------------------------------------------------- #


async def test_backfill_estimates_old_text_only_once():
    store = MemoryUsageStore(TZ)
    await store.write([_record(TODAY - timedelta(days=5))], [])
    before = NOW - timedelta(days=30)
    store.cache_rows = [
        {
            "workspace": "a",
            "id": "1",
            "cache_type": "extract",
            "original_prompt": "abcd",
            "return_value": "xy",
            "create_time": before,
        },
        {
            "workspace": "a",
            "id": "2",
            "cache_type": "analysis",
            "original_prompt": "x<vlm_images>[]</vlm_images>",
            "return_value": "y",
            "create_time": before,
        },
        {
            "workspace": "a",
            "id": "3",
            "cache_type": "query",
            "original_prompt": "new",
            "return_value": "z",
            "create_time": NOW,
        },
    ]
    state = await run_backfill(store, _Tok())
    assert state["status"] == "done" and state["rows"] == 1
    rows = await store.rollup(
        "day", TODAY - timedelta(days=60), TODAY, filters={"source": BACKFILL_SOURCE}
    )
    assert rows[0]["prompt_tokens"] == 4 and rows[0]["estimated_calls"] == 1
    assert (await store.first_recorded_day()) == TODAY - timedelta(days=5)
    assert (await run_backfill(store, _Tok()))["rows"] == 1
    rows = await store.rollup(
        "day", TODAY - timedelta(days=60), TODAY, filters={"source": BACKFILL_SOURCE}
    )
    assert rows[0]["prompt_tokens"] == 4


async def test_model_change_is_noticed_at_startup():
    store = MemoryUsageStore(TZ)
    recorder = UsageRecorder(store)
    from lightrag.usage import add_event_sink, remove_event_sink

    add_event_sink(recorder.event)
    try:
        rag = SimpleNamespace(get_llm_role_config=lambda p: {"model": "m1"})
        await note_startup(
            InsightsService(store, default_rag=rag, now=lambda: NOW), "1.0"
        )
        store.leases.clear()
        rag2 = SimpleNamespace(
            get_llm_role_config=lambda p: {"model": "m2" if p == "query" else "m1"}
        )
        await note_startup(
            InsightsService(store, default_rag=rag2, now=lambda: NOW), "1.1"
        )
    finally:
        remove_event_sink(recorder.event)
    await recorder.flush()
    kinds = {(e.kind, e.purpose) for e in store.event_rows}
    assert kinds == {("model_change", "query"), ("deployment", None)}


def _smtp():
    return SmtpSettings(
        host="smtp",
        port=25,
        security="none",
        username="",
        password="",
        sender="a@b.c",
        timeout=1,
        admin_recipients=("admin@b.c",),
        repeat_seconds=3600,
    )


async def test_anomaly_alert_is_one_email_and_not_repeated():
    store = MemoryUsageStore(TZ)
    service, spike = await _spike_service(store)
    sent = []
    scheduler = InsightsScheduler(
        service, settings=_smtp(), send=lambda s, to, msg: sent.append(msg)
    )
    ids = await scheduler.send_anomaly_alerts()
    assert ids and len(sent) == 1
    assert "Extraction" in sent[0].get_content()
    assert is_own_alert_email(sent[0]["From"], sent[0]["Subject"], "a@b.c")
    assert await scheduler.send_anomaly_alerts() == [] and len(sent) == 1


async def test_budget_warning_has_both_dates_and_fires_once_per_threshold():
    store = MemoryUsageStore(TZ)
    service = await _service(store)
    await service.save_settings({"budget": {"amount": 0.1, "start": TODAY.isoformat()}})
    sent = []
    scheduler = InsightsScheduler(
        service, settings=_smtp(), send=lambda s, to, msg: sent.append(msg)
    )
    assert await scheduler.send_budget_warning() is not None
    body = sent[0].get_content()
    assert "Without BYO" in body and "Requesty" in body
    assert is_own_alert_email(sent[0]["From"], sent[0]["Subject"], "a@b.c")
    assert await scheduler.send_budget_warning() is None and len(sent) == 1


async def test_weekly_digest_goes_once_a_week():
    store = MemoryUsageStore(TZ)
    service = await _service(store)
    sent = []
    scheduler = InsightsScheduler(
        service, settings=_smtp(), send=lambda s, to, msg: sent.append(msg)
    )
    assert await scheduler.send_digest() is True
    assert "Query" in sent[0].get_content()
    assert is_own_alert_email(sent[0]["From"], sent[0]["Subject"], "a@b.c")
    assert await scheduler.send_digest() is False


async def test_no_smtp_sends_nothing():
    store = MemoryUsageStore(TZ)
    service, _ = await _spike_service(store)
    off = SmtpSettings(
        host="",
        port=25,
        security="none",
        username="",
        password="",
        sender="",
        timeout=1,
        admin_recipients=(),
        repeat_seconds=3600,
    )
    scheduler = InsightsScheduler(
        service, settings=off, send=lambda *a: pytest.fail("sent")
    )
    assert await scheduler.send_anomaly_alerts() == []
    assert await scheduler.send_digest() is False


async def test_brief_csv_and_pdf():
    store = MemoryUsageStore(TZ)
    service = await _service(store)
    await service.save_settings(
        {"budget": {"amount": 5.0, "start": TODAY.isoformat()}, "myr_rate": 4.2}
    )
    forecast = await service.forecast()
    actuals = await service._rows(TODAY - timedelta(days=89), TODAY, ["purpose"])
    csv_text = brief_csv(forecast, actuals)
    assert "next_90_days,query" in csv_text and "suggested_purchase" in csv_text
    lines = brief_lines(forecast, actuals)
    assert any("MYR" in line for line in lines) and any("BYO" in line for line in lines)
    pdf = text_pdf(lines)
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    import pypdf
    from io import BytesIO

    assert "procurement brief" in pypdf.PdfReader(BytesIO(pdf)).pages[0].extract_text()


# --------------------------------------------------------------------------- #
# Routes                                                                      #
# --------------------------------------------------------------------------- #


def _client(service, admin=True):
    app = FastAPI()
    app.include_router(_routes.router)
    app.state.insights = service
    app.state.insights_tasks = set()
    if admin:
        app.dependency_overrides[require_admin] = lambda: UserInfo(
            username="root", role="admin", workspace_id="w", metadata={}
        )
    return TestClient(app)


async def test_routes_answer_and_validate():
    store = MemoryUsageStore(TZ)
    service = await _service(store)
    client = _client(service)
    assert client.get("/admin/insights/status").json()["available"] is True
    assert (
        client.get("/admin/insights/overview", params={"days": 7}).json()["scope"][
            "range"
        ]
        == 7
    )
    start = (TODAY - timedelta(days=9)).isoformat()
    end = (TODAY - timedelta(days=5)).isoformat()
    custom = client.get(
        "/admin/insights/overview", params={"start": start, "end": end}
    ).json()
    assert (
        custom["scope"]["start"],
        custom["scope"]["end"],
        custom["scope"]["range"],
    ) == (start, end, 5)
    assert (
        client.get("/admin/insights/overview", params={"purpose": "nope"}).status_code
        == 400
    )
    assert client.get("/admin/insights/purposes/query").json()["id"] == "query"
    assert client.get("/admin/insights/workspaces").json()["rows"]
    assert client.get("/admin/insights/workspaces/alpha").json()["id"] == "alpha"
    assert client.get("/admin/insights/breakdown", params={"group": "model"}).json()[
        "rows"
    ]
    assert (
        client.get("/admin/insights/breakdown", params={"group": "bad"}).status_code
        == 400
    )
    assert client.get("/admin/insights/days/2026-10-20").json()["day"] == "2026-10-20"
    assert client.get("/admin/insights/days/garbage").status_code == 400
    assert "calendar" in client.get("/admin/insights/anomalies").json()
    assert (
        client.post(
            "/admin/insights/anomalies/2026-10-20:alpha", json={"state": "bad"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/admin/insights/anomalies/2026-10-20:alpha", json={"state": "expected"}
        ).status_code
        == 200
    )
    assert (
        client.get("/admin/insights/work", params={"side": "questions"}).json()["side"]
        == "questions"
    )
    assert (
        client.get(
            "/admin/insights/forecast", params={"documents": 50, "swap": "query"}
        ).json()["ready"]
        is True
    )
    assert (
        client.get("/admin/insights/live").json()["purposes"]["query"]["running"] == 0
    )


async def test_settings_route_and_exports():
    store = MemoryUsageStore(TZ)
    service = await _service(store)
    client = _client(service)
    assert (
        client.put(
            "/admin/insights/settings", json={"invoices": {"October": 1}}
        ).status_code
        == 400
    )
    assert (
        client.put(
            "/admin/insights/settings",
            json={"prices": {"m": {"prompt": -1, "completion": 1}}},
        ).status_code
        == 422
    )
    saved = client.put(
        "/admin/insights/settings",
        json={"rate_limit_tpm": 100, "budget": {"amount": 9, "start": "2026-10-01"}},
    ).json()
    assert (
        saved["rate_limit_tpm"] == 100
        and saved["prices"] == PRICES
        and saved["budget"]["start"] == "2026-10-01"
    )
    exported = client.get("/admin/insights/breakdown.csv", params={"group": "purpose"})
    assert (
        exported.headers["content-type"].startswith("text/csv")
        and "query" in exported.text
    )
    assert client.get("/admin/insights/brief.csv").status_code == 200
    assert client.get("/admin/insights/brief.pdf").content.startswith(b"%PDF")


async def test_export_masks_document_names():
    store = MemoryUsageStore(TZ)
    service = await _service(store, [_record(TODAY, "extract", doc_id="doc-abcd1234")])
    store.paths = {("alpha", "doc-abcd1234"): "Secret salary review.pdf"}
    text = (
        _client(service)
        .get("/admin/insights/breakdown.csv", params={"group": "document"})
        .text
    )
    assert "Secret" not in text and "Sec•••ew.pdf #abcd" in text


async def test_every_route_needs_admin():
    store = MemoryUsageStore(TZ)
    service = await _service(store, [])
    app = FastAPI()
    app.include_router(_routes.router)
    app.state.insights = service
    app.dependency_overrides[require_admin] = lambda: (_ for _ in ()).throw(
        _routes.HTTPException(status_code=403, detail="no")
    )
    client = TestClient(app, raise_server_exceptions=False)
    for route in _routes.router.routes:
        method = sorted(route.methods)[0]
        path = (
            route.path.replace("{purpose}", "query")
            .replace("{workspace}", "a")
            .replace("{day}", "2026-10-20")
            .replace("{anomaly_id:path}", "x")
        )
        assert (
            client.request(method, path, json={"state": "open"}).status_code == 403
        ), path


def test_without_postgres_the_page_says_so():
    app = FastAPI()
    app.include_router(_routes.router)
    app.state.insights = None
    app.dependency_overrides[require_admin] = lambda: UserInfo(
        username="root", role="admin", workspace_id="w", metadata={}
    )
    client = TestClient(app)
    assert client.get("/admin/insights/status").json() == {"available": False}
    assert client.get("/admin/insights/overview").status_code == 503


async def test_middleware_tags_the_source():
    from lightrag.api.insights.middleware import UsageContextMiddleware

    seen = {}

    async def app(scope, receive, send):
        seen["scope"] = current_usage_scope()

    async def call(path, headers=()):
        await UsageContextMiddleware(app)(
            {"type": "http", "path": path, "headers": list(headers)}, None, None
        )
        return seen["scope"]

    assert (await call("/query", [(b"x-lightrag-client", b"webui")])).source == "webui"
    assert (
        await call("/query/stream", [(b"user-agent", b"curl/8")])
    ).source == "rest_api"
    assert (await call("/query/stream", [(b"user-agent", b"curl/8")])).client == "curl"
    assert (await call("/query/data")).source == "retrieval"
    assert (await call("/api/chat")).source == "ollama"
    assert (await call("/v1/chat/completions")).source == "openai_api"
