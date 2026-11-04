"""Structured log sink: redaction before buffering, rate limits, coalescing,
non-blocking emit, no recursion, the flusher's budget, deletion, retention
and the runtime switch (OPS_LOG_SINK)."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from lightrag.api.ops import context, log_sink, runtime
from lightrag.api.ops.log_sink import OpsLogHandler, OpsLogSink
from lightrag.api.ops.store import MemoryOpsStore, PostgresOpsStore, deleted_key
from lightrag.usage import usage_scope
from tests.ops_fixtures import ops_routes  # noqa: F401  (config parses argv once)

pytestmark = pytest.mark.offline

JWT = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJvcHNvQHVuaW1hcy5teSJ9."
    "Zx3cQm9yqLk2R8sVt0uWw5yXz1aBcDeFgHiJkLmNoPq"
)
WORKER = "0123456789abcdef0123456789abcdef"
RUN = "a1b2c3d4e5f60718293a4b5c6d7e8f90"

# FR-72 audit corpus: (format, args, substrings that must never be stored).
CORPUS = [
    ("student IC %s rejected", ("900101-13-5678",), ["900101-13-5678"]),
    ("IC without dashes 900101135678", (), ["900101135678"]),
    ("call %s or %s", ("+60 12-345 6789", "019 876 5432"), ["12-345 6789", "876 5432"]),
    ("mail from %s bounced", ("alice.tan@unimas.my",), ["alice.tan@unimas.my"]),
    (
        "Failed to read /app/data/inputs/fsktm_unimas_my/Minit Mesyuarat Jabatan.pdf",
        (),
        ["Minit", "Mesyuarat", "Jabatan", "fsktm_unimas_my/"],
    ),
    ("OSError: '%s'", ("/app/inputs/ws/Gaji Ahmad bin Ali.xlsx",), ["Gaji", "Ahmad"]),
    ("attachment %s skipped", ("Slip_Gaji_Mac_2026.pdf",), ["Slip_Gaji_Mac"]),
    ("provider key %s", ("sk-proj-AbCdEf1234567890xyz",), ["AbCdEf1234567890xyz"]),
    ("config api_key=%s loaded", ("abc123secretvalue",), ["abc123secretvalue"]),
    ("token %s expired", (JWT,), [JWT, JWT[:30]]),
    ("Authorization: Bearer %s", ("abcdef1234567890XYZ",), ["abcdef1234567890XYZ"]),
    ("calling with bearer %s", ("Zm9vYmFyYmF6cXV4",), ["Zm9vYmFyYmF6cXV4"]),
    (
        "connect %s failed",
        ("postgres://lightrag:Sup3rS3cret@postgres:5432/db",),
        ["Sup3rS3cret", "lightrag:Sup3r"],
    ),
    ("https://user:pa55@example.org/x unreachable", (), ["pa55", "user:"]),
    ("Subject: %s", ("Re: Gaji Disember Ahmad",), ["Gaji Disember", "Ahmad"]),
    ("subject='%s' queued", ("Salary slip for Siti",), ["Salary slip", "Siti"]),
    (
        "PostgreSQL database,\nsql:INSERT ...,\ndata:%s",
        ({"content": "Dear bob@unimas.my", "password": "hunter2"},),
        ["bob@unimas.my", "hunter2"],
    ),
    # File names with spaces and no quotes or directory (pipeline.py's
    # "File processing error" line, an upload failure), a directory with
    # spaces, a Windows path, browser duplicate suffixes, brackets, commas,
    # apostrophes and long names.
    (
        "File processing error: - ID: doc-0123 %s",
        ("Surat Tawaran Ali Bin Abu.docx",),
        ["Surat", "Tawaran", "Ali Bin"],
    ),
    (
        "Upload failed for %s",
        ("Laporan Prestasi Siti Aminah.pdf",),
        ["Prestasi", "Siti"],
    ),
    (
        "Upload failed for %s",
        ("/app/inputs/Ali Bin Abu/Laporan Prestasi.pdf",),
        ["Ali Bin Abu", "Prestasi", "/app/inputs"],
    ),
    (
        "open %s failed",
        ("C:\\Users\\Ali Bin Abu\\Docs\\Laporan Kewangan 2024.docx",),
        ["Ali Bin", "Kewangan", "C:\\Users"],
    ),
    ("Docling failed on %s", ("/app/inputs/ws/Gaji Ali (1).pdf",), ["Gaji Ali", "(1)"]),
    (
        "Docling failed on %s",
        ("/app/inputs/ws/Gaji Ali bin Abu [final].docx",),
        ["Gaji Ali", "bin Abu"],
    ),
    (
        "OSError: %r",
        ("/x/inputs/ws/Ali's Gaji Oktober.pdf",),
        ["Ali's", "Gaji", "Oktober"],
    ),
    (
        "parse error %s",
        ("/app/inputs/ws/Minit Mesyuarat Jabatan Kewangan Bil 3 Tahun 2024.pdf",),
        ["Mesyuarat", "Jabatan", "Kewangan", "Tahun"],
    ),
    (
        "parse error %s",
        ("/app/inputs/ws/Gaji, Ali bin Abu.docx",),
        ["Gaji,", "Ali bin"],
    ),
    # document_routes.py deletion lines, any extension (masked whole).
    (
        "File deletion skipped, missing or unsafe file: %s",
        ("Minit Mesyuarat Jabatan Gaji.pdf",),
        ["Mesyuarat", "Jabatan", "Gaji"],
    ),
    (
        "Error deleting document 1/2: doc-0a1b[%s] - boom",
        ("senarai_pelajar_ahmad_bin_ali.json",),
        ["pelajar", "ahmad"],
    ),
    ("Failed to delete file %s: denied", ("Gaji Ahmad.json",), ["Gaji", "Ahmad"]),
    # IC and phone numbers next to "_" (slugged names, labels).
    ("nric_%s bad", ("900101-13-5678",), ["900101-13-5678"]),
    ("kp %s_baru failed", ("900101-13-5678",), ["900101-13-5678"]),
    ("[Borang_%s] Pipeline busy", ("900101-13-5678",), ["900101-13-5678"]),
    ("phone_%s", ("0123456789",), ["0123456789"]),
    # URL userinfo whose password holds "/" or "@".
    (
        "connect %s failed",
        ("postgresql://lightrag:pa/ss0rd@db:5432/x",),
        ["pa/ss0rd", "lightrag:pa"],
    ),
    ("unreachable %s", ("neo4j://neo4j:p@ss0rd@neo4j:7687",), ["ss0rd", "neo4j:p"]),
]


def _store() -> MemoryOpsStore:
    return MemoryOpsStore()


def _sink(store=None, **kwargs) -> OpsLogSink:
    kwargs.setdefault("rows_per_s", 1e9)
    return OpsLogSink(store or _store(), WORKER, **kwargs)


def _record(
    msg,
    *args,
    name="lightrag",
    level=logging.WARNING,
    lineno=10,
    module_path="/app/lightrag/pipeline.py",
    exc_info=None,
):
    return logging.LogRecord(name, level, module_path, lineno, msg, args, exc_info)


def _stored_text(store: MemoryOpsStore) -> str:
    return "\n".join(f"{r['msg']}\n{r.get('exc') or ''}" for r in store.log_rows)


async def test_redaction_corpus_leaves_nothing_in_stored_rows():
    store = _store()
    sink = _sink(store)
    for index, (fmt, args, _) in enumerate(CORPUS):
        sink.handler.handle(_record(fmt, *args, lineno=100 + index))
    # A traceback whose exception text carries secrets.
    try:
        raise RuntimeError(
            "connect failed for carol@unimas.my with sk-test-SECRET1234567890 "
            "IC 900101-13-5678 file /app/inputs/ws/Laporan Kewangan 2024.docx"
        )
    except RuntimeError:
        import sys

        sink.handler.handle(_record("pipeline failed", exc_info=sys.exc_info()))
    # logger.error(traceback.format_exc()) style.
    sink.handler.handle(
        _record(
            'Traceback (most recent call last):\n  File "x.py", line 1\n'
            "ValueError: bad value dave@unimas.my 012-3456789",
            lineno=900,
        )
    )
    assert await sink.flush_once()
    assert len(store.log_rows) == len(CORPUS) + 2
    stored = _stored_text(store)
    for _, _, forbidden in CORPUS:
        for needle in forbidden:
            assert needle not in stored, needle
    for needle in (
        "carol@unimas.my",
        "SECRET1234567890",
        "900101-13-5678",
        "Laporan",
        "Kewangan",
        "dave@unimas.my",
        "012-3456789",
    ):
        assert needle not in stored, needle
    # The audit regexes (weekly job) agree: nothing looks unredacted.
    counts = await store.logs_redaction_audit(7)
    assert counts["rows"] == len(store.log_rows)
    assert {k: v for k, v in counts.items() if k != "rows" and v} == {}
    # Exceptions keep their type; logged tracebacks become the exc field.
    by_line = {r["lineno"]: r for r in store.log_rows}
    assert by_line[900]["exc_type"] == "ValueError"
    assert by_line[900]["msg"].startswith("ValueError")
    assert by_line[900]["exc"].startswith("Traceback")
    errors = [r for r in store.log_rows if r["exc_type"] == "RuntimeError"]
    assert errors and errors[0]["exc"].startswith("Traceback")


async def test_messages_and_tracebacks_are_capped():
    store = _store()
    sink = _sink(store)
    sink.handler.handle(_record("x" * 10_000))
    try:
        raise ValueError("y" * 50_000)
    except ValueError:
        import sys

        sink.handler.handle(_record("boom", lineno=11, exc_info=sys.exc_info()))
    await sink.flush_once()
    rows = {r["lineno"]: r for r in store.log_rows}
    assert len(rows[10]["msg"]) <= 2048
    assert len(rows[11]["exc"]) <= 8192


async def test_context_fields_come_from_scope_request_and_run():
    store = _store()
    sink = _sink(store)
    rid = context._request_id.set("req0123456789abcdef")
    run = context.set_run_id(RUN)
    try:
        with usage_scope(workspace="fsktm", track_id="upload_x", source="email"):
            sink.handler.handle(_record("extract failed"))
        sink.handler.handle(
            _record("[space1] failed doc-33e75ff09dd601bbe69f351039152189", lineno=12)
        )
    finally:
        context.reset_run_id(run)
        context._request_id.reset(rid)
    await sink.flush_once()
    first, second = sorted(store.log_rows, key=lambda r: r["lineno"])
    assert first["workspace"] == "fsktm"
    assert first["request_id"] == "req0123456789abcdef"
    assert first["run_id"] == RUN
    assert first["track_id"] == "upload_x" and first["source"] == "email"
    assert first["worker_id"] == WORKER
    assert first["module"] == "pipeline" and first["logger"] == "lightrag"
    # A leading [workspace] prefix and a doc ID in the text are picked up.
    assert second["workspace"] == "space1"
    assert second["doc_id"] == "doc-33e75ff09dd601bbe69f351039152189"


async def test_token_bucket_limits_a_storm_and_counts_the_rest():
    now = [1000.0]
    store = _store()
    sink = _sink(store, clock=lambda: now[0])
    for i in range(100):
        sink.handler.handle(_record("retry %d failed", i))
    assert len(sink.handler.buffer) == 20
    assert sink.handler.rate_limited == 80
    # Three seconds refill one token: the next row carries the 80.
    now[0] += 3.1
    sink.handler.handle(_record("retry %d failed", 101))
    assert sink.handler.buffer[-1]["repeat_count"] == 81
    # Another call site has its own bucket.
    sink.handler.handle(_record("other", lineno=99))
    assert sink.handler.buffer[-1]["repeat_count"] == 1
    await sink.flush_once()
    assert sum(r["repeat_count"] for r in store.log_rows) == 102


async def test_suppressed_counts_are_kept_when_the_storm_stops():
    now = [1000.0]
    store = _store()
    sink = _sink(store, clock=lambda: now[0])
    for _ in range(25):
        sink.handler.handle(_record("same line"))
    await sink.flush_once()
    # 20 identical rows coalesced into one; 5 rate-limited and pending.
    assert [r["repeat_count"] for r in store.log_rows] == [20]
    await sink.flush_once()
    assert len(store.log_rows) == 1  # no token yet: the storm may go on
    now[0] += 10
    await sink.flush_once()
    assert [r["repeat_count"] for r in store.log_rows] == [20, 5]
    assert store.log_rows[1]["msg"] == "same line"


async def test_carry_over_rows_have_no_per_record_context():
    """Suppressed lines may come from other requests, documents and
    workspaces: the carry-over row names none of them."""
    now = [1000.0]
    store = _store()
    sink = _sink(store, clock=lambda: now[0])
    rid = context._request_id.set("req-first")
    try:
        for i in range(25):
            with usage_scope(workspace=f"ws_{i}"):
                sink.handler.handle(
                    _record("embedding failed for doc-%032x", i, lineno=7)
                )
    finally:
        context._request_id.reset(rid)
    await sink.flush_once()
    now[0] += 10
    await sink.flush_once()
    carried = store.log_rows[-1]
    assert carried["repeat_count"] == 5
    assert carried["operation"] == log_sink.SUPPRESSED_OPERATION
    for field in ("workspace", "request_id", "run_id", "doc_id", "track_id"):
        assert carried[field] is None, field
    assert carried["fingerprint"] == store.log_rows[0]["fingerprint"]
    assert "doc-" in carried["msg"] and "%032x" not in carried["msg"]
    assert "0000000000000013" not in carried["msg"]


async def test_carry_over_counts_beyond_the_batch_room_stay_pending():
    now = [1000.0]
    store = _store()
    sink = _sink(store, clock=lambda: now[0], rate_per_min=1.0)
    for site in range(10):
        for _ in range(3):
            sink.handler.handle(_record("site %d", site, lineno=100 + site))
    await sink.flush_once()
    assert len(store.log_rows) == 10
    now[0] += 120  # every storm is over
    rows = sink.handler.take_suppressed(now[0], 3)
    assert [r["repeat_count"] for r in rows] == [2, 2, 2]
    # The other seven still hold their counts.
    rest = sink.handler.take_suppressed(now[0], 100)
    assert len(rest) == 7 and sum(r["repeat_count"] for r in rest) == 14
    assert sink.handler.take_suppressed(now[0], 100) == []


async def test_quiet_flushes_keep_the_sink_fresh():
    """A quiet minute, then one line: not 'last write long ago'."""
    from lightrag.api.ops.health import _flush_stale

    now = [1000.0]
    store = _store()
    sink = _sink(store, clock=lambda: now[0])
    sink.handler.handle(_record("first"))
    await sink.flush_once()
    for _ in range(60):
        now[0] += 2.0
        await sink.flush_once()
    sink.handler.handle(_record("second", lineno=2))
    stats = sink.stats()
    assert stats["buffer"] == 1 and stats["last_flush_age_s"] <= 2.0
    assert _flush_stale(stats) is False
    # A sink that really cannot write still turns stale.
    store.fail_writes = ConnectionError("down")
    now[0] += 2.0
    assert await sink.flush_once() is False
    now[0] += 90.0
    assert _flush_stale(sink.stats()) is True


def test_budgets_are_shared_by_gunicorn_workers():
    sink = OpsLogSink(_store(), WORKER, workers=4)
    assert sink.rows_per_s == log_sink.ROWS_PER_S / 4
    assert sink.batch_rows == log_sink.BATCH_ROWS // 4
    assert sink.handler.rate_per_min == log_sink.RATE_PER_MIN / 4
    assert sink.stats()["workers"] == 4
    one = OpsLogSink(_store(), WORKER, workers=0)
    assert one.rows_per_s == log_sink.ROWS_PER_S and one.workers == 1


async def test_identical_records_in_a_batch_are_coalesced_but_contexts_kept():
    store = _store()
    sink = _sink(store)
    for _ in range(5):
        sink.handler.handle(_record("embedding call failed"))
    with usage_scope(workspace="fsktm"):
        sink.handler.handle(_record("embedding call failed"))
    await sink.flush_once()
    counts = sorted((r["workspace"] or "", r["repeat_count"]) for r in store.log_rows)
    assert counts == [("", 5), ("fsktm", 1)]


def test_full_buffer_drops_and_counts_without_blocking():
    sink = _sink(max_buffer=50, rate_per_min=1e9)
    started = time.perf_counter()
    for i in range(2000):
        sink.handler.handle(_record("line %d", i, lineno=i + 1))
    elapsed = time.perf_counter() - started
    assert len(sink.handler.buffer) == 50
    assert sink.handler.dropped == 1950
    # Pure in-memory work: far below a millisecond per record.
    assert elapsed < 2.0, elapsed


async def test_store_down_keeps_rows_backs_off_and_never_blocks_emit():
    store = _store()
    store.fail_writes = ConnectionError("down")
    now = [1000.0]
    sink = _sink(store, clock=lambda: now[0], max_retry_rows=30)
    for i in range(40):
        sink.handler.handle(_record("line %d", i, lineno=i + 1))
    assert not await sink.flush_once()
    assert sink.write_failures == 1
    stats = sink.stats()
    # The retry list is capped: the oldest rows are dropped and counted.
    assert stats["buffer"] == 30 and stats["dropped"] == 10
    assert stats["last_flush_error"] == "ConnectionError"
    assert sink._backoff_until > now[0]
    started = time.perf_counter()
    for i in range(200):
        sink.handler.handle(_record("more %d", i, lineno=1000 + i))
    assert time.perf_counter() - started < 1.0
    store.fail_writes = None
    now[0] += 120
    assert await sink.flush_once()
    assert len(store.log_rows) == 230 and sink.written == 230


async def test_rejected_rows_are_dropped_one_by_one():
    class Picky(MemoryOpsStore):
        async def logs_write(self, rows):
            if any("bad" in r["msg"] for r in rows):
                raise ValueError("invalid byte sequence")
            await super().logs_write(rows)

    store = Picky()
    sink = _sink(store)
    for i, text in enumerate(("good one", "bad one", "good two")):
        sink.handler.handle(_record(text, lineno=i + 1))
    assert await sink.flush_once()
    assert sorted(r["msg"] for r in store.log_rows) == ["good one", "good two"]
    assert sink.bad_rows == 1


async def test_flusher_never_records_its_own_failures():
    """A write that fails and logs (driver, pool, store) must not feed itself."""

    class Noisy(MemoryOpsStore):
        async def logs_write(self, rows):
            logging.getLogger("lightrag").error("PostgreSQL error while writing")
            logging.getLogger("asyncpg.pool").warning("connection lost")
            raise ConnectionError("down")

    store = Noisy()
    sink = _sink(store)
    sink.attach()
    try:
        logging.getLogger("lightrag").setLevel(logging.INFO)
        logging.getLogger("lightrag.ops_test").warning("a real warning")
        assert len(sink.handler.buffer) == 1
        token = log_sink._in_flush.set(True)
        try:
            await sink.flush_once()
        finally:
            log_sink._in_flush.reset(token)
        # The batch is kept for retry; nothing new arrived from the failure.
        assert len(sink.handler.buffer) == 0 and len(sink._retry) == 1
        assert sink.handler.self_suppressed >= 2
        # The sink's own logger is never taken either.
        logging.getLogger(log_sink.SINK_LOGGER_NAME).error("sink trouble")
        assert len(sink.handler.buffer) == 0
    finally:
        sink.detach()


def test_emit_does_not_recurse_through_its_own_thread():
    sink = _sink()
    handler = sink.handler
    handler._local.active = True
    handler.handle(_record("nested"))
    handler._local.active = False
    assert handler.self_suppressed == 1 and not handler.buffer


def test_emit_never_raises():
    sink = _sink()

    class Broken:
        def __str__(self):
            raise RuntimeError("cannot format")

    sink.handler.handle(_record("value %s", Broken()))
    sink.handler.handle(_record("ok", lineno=2))
    assert len(sink.handler.buffer) >= 1


async def test_row_budget_spreads_a_burst():
    now = [1000.0]
    store = _store()
    sink = OpsLogSink(
        store,
        WORKER,
        clock=lambda: now[0],
        batch_rows=100,
        rows_per_s=50.0,
        rate_per_min=1e9,
    )
    for i in range(400):
        sink.handler.handle(_record("line %d", i, lineno=i + 1))
    await sink.flush_once()
    assert len(store.log_rows) == 100  # the burst allowance
    await sink.flush_once()
    assert len(store.log_rows) == 100  # no time passed, no budget
    now[0] += 2.0
    await sink.flush_once()
    assert len(store.log_rows) == 200  # 50 rows/s
    # A long quiet spell refills at most one batch.
    now[0] += 60
    await sink.flush_once()
    assert len(store.log_rows) == 300
    now[0] += 2.0
    await sink.flush_once()
    assert len(store.log_rows) == 400


async def test_flush_loop_writes_in_the_background():
    store = _store()
    sink = OpsLogSink(store, WORKER, interval_s=0.05, poll_s=0.01)
    sink.start()
    try:
        sink.handler.handle(_record("background line"))
        for _ in range(100):
            if store.log_rows:
                break
            await asyncio.sleep(0.01)
        assert [r["msg"] for r in store.log_rows] == ["background line"]
    finally:
        await sink.stop()


async def test_deleted_workspace_rows_are_never_written():
    store = _store()
    sink = _sink(store)
    with usage_scope(workspace="gone_ws"):
        sink.handler.handle(_record("about gone"))
    with usage_scope(workspace="other_ws"):
        sink.handler.handle(_record("about other", lineno=2))
    await sink.forget_flushed("gone_ws")
    await sink.flush_once()
    assert [r["workspace"] for r in store.log_rows] == ["other_ws"]
    # Deleted by another worker: its tombstone drops rows buffered here.
    with usage_scope(workspace="far_ws"):
        sink.handler.handle(_record("about far", lineno=3))
    store.kv[deleted_key("far_ws")] = {"at": time.time()}
    await sink.flush_once()
    assert [r["workspace"] for r in store.log_rows] == ["other_ws"]
    assert sink.dropped_deleted == 2


async def test_attached_loggers_and_root_allowlist(monkeypatch):
    store = _store()
    sink = _sink(store)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    sink.attach()
    try:
        # With the sink on the root logger, a stand-in keeps unhandled
        # third-party warnings on stderr (logging.lastResort no longer fires).
        assert any(isinstance(h, log_sink._LastResortStandIn) for h in root.handlers)
        for name in ("lightrag", "uvicorn.error", "gunicorn.error"):
            assert sink.handler in logging.getLogger(name).handlers
        logging.getLogger("lightrag.api.oauth2").setLevel(logging.NOTSET)
        logging.getLogger("lightrag").setLevel(logging.INFO)
        logging.getLogger("lightrag.api.oauth2").warning("sso callback failed")
        logging.getLogger("lightrag.api.email").error("email ingest failed")
        uvicorn_error = logging.getLogger("uvicorn.error")
        monkeypatch.setattr(uvicorn_error, "propagate", True)
        uvicorn_error.error("Exception in ASGI application")
        logging.getLogger("httpx").warning("HTTP Request failed")
        logging.getLogger("somelib").warning("not on the allowlist")
        logging.getLogger("lightrag").info("info is below the level")
        names = sorted(r["logger"] for r in sink.handler.buffer)
        # uvicorn.error propagates to the root (Gunicorn): stored once only.
        assert names == [
            "httpx",
            "lightrag.api.email",
            "lightrag.api.oauth2",
            "uvicorn.error",
        ]
    finally:
        sink.detach()
    assert sink.handler not in logging.getLogger("lightrag").handlers
    assert not any(isinstance(h, log_sink._LastResortStandIn) for h in root.handlers)


def test_level_parsing():
    assert log_sink.level_from(None) == logging.WARNING
    assert log_sink.level_from("info") == logging.INFO
    assert log_sink.level_from("warn") == logging.WARNING
    assert log_sink.level_from("40") == logging.ERROR
    assert log_sink.level_from("nonsense") == logging.WARNING
    handler = OpsLogHandler(level=logging.ERROR)
    assert handler.level == logging.ERROR


async def test_retention_and_weekly_audit():
    store = _store()
    now = time.time()
    await store.logs_write(
        [
            {
                "ts": now - 20 * 86400,
                "level": 30,
                "logger": "lightrag",
                "msg": "old line",
                "fingerprint": "f" * 16,
            },
            {
                "ts": now - 60,
                "level": 40,
                "logger": "lightrag",
                "msg": "leaked eve@unimas.my and Laporan.pdf",
                "fingerprint": "e" * 16,
            },
            {
                "ts": now - 30,
                "level": 40,
                "logger": "lightrag",
                "msg": "masked Lap•••an.pdf and Sen•••.docx",
                "fingerprint": "d" * 16,
            },
            {
                "ts": now - 20,
                "level": 40,
                "logger": "lightrag",
                "msg": "masked Lap•••an.pdf next to Senarai.docx",
                "fingerprint": "c" * 16,
            },
        ]
    )
    loop = SimpleNamespace(still_held=lambda: _true())
    out = await log_sink.retention_cycle(loop, store=store, days=14)
    assert out["deleted"] == 1 and len(store.log_rows) == 3
    audit = store.kv[log_sink.AUDIT_KEY]
    assert audit["rows"] == 3 and audit["clean"] is False
    # A masked name no longer hides an unmasked one in the same row.
    assert audit["matches"] == {"email": 1, "file_name": 2}
    # The audit runs weekly; the next hourly cycle only deletes.
    out = await log_sink.retention_cycle(loop, store=store, days=14)
    assert "audit" not in out


async def _true():
    return True


def test_retention_days_setting(monkeypatch):
    monkeypatch.delenv("OPS_LOG_RETENTION_DAYS", raising=False)
    assert log_sink.retention_days() == 14
    monkeypatch.setenv("OPS_LOG_RETENTION_DAYS", "30")
    assert log_sink.retention_days() == 30
    monkeypatch.setenv("OPS_LOG_RETENTION_DAYS", "x")
    assert log_sink.retention_days() == 14


def _pg_stubs(monkeypatch):
    async def ready(self):
        self.ready = self.doc_status_ready = True

    async def nothing(self, *args, **kwargs):
        return None

    async def lease(self, name, holder, ttl_s):
        return False

    for name, fn in (
        ("initialize", ready),
        ("finalize", nothing),
        ("try_lease", lease),
        ("release_lease", nothing),
    ):
        monkeypatch.setattr(PostgresOpsStore, name, fn)
    for flag in ("OPS_INDEX_BUILD", "OPS_QUEUE_SCAN", "OPS_PROBES_ENABLED"):
        monkeypatch.setenv(flag, "false")
    monkeypatch.setenv("OPS_RUN_TRACKING", "false")


def _sink_handlers() -> list[logging.Handler]:
    return [
        h
        for h in logging.getLogger("lightrag").handlers
        if isinstance(h, OpsLogHandler)
    ]


async def test_flag_off_attaches_no_handler(monkeypatch):
    _pg_stubs(monkeypatch)
    monkeypatch.delenv("OPS_LOG_SINK", raising=False)
    app = SimpleNamespace(state=SimpleNamespace())
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="PGKVStorage"))
    try:
        assert rt.log_sink is None
        assert _sink_handlers() == []
        assert not any(
            isinstance(h, OpsLogHandler) for h in logging.getLogger().handlers
        )
        # Retention still runs over what is stored.
        assert "ops:log-retention" in [loop.name for loop in rt.loops]
    finally:
        await runtime.stop_ops(app)


async def test_flag_on_attaches_per_worker_and_stop_detaches(monkeypatch):
    _pg_stubs(monkeypatch)
    monkeypatch.setenv("OPS_LOG_SINK", "true")
    monkeypatch.setenv("OPS_LOG_LEVEL", "ERROR")
    app = SimpleNamespace(state=SimpleNamespace())
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="PGKVStorage"))
    try:
        assert rt.log_sink is not None and rt.log_sink.attached
        assert _sink_handlers() == [rt.log_sink.handler]
        assert rt.log_sink.handler.level == logging.ERROR
        assert rt.log_sink.worker_id == rt.worker_id
        assert rt.heartbeat.log_sink is rt.log_sink
    finally:
        await runtime.stop_ops(app)
    assert _sink_handlers() == []


async def test_without_postgres_no_sink_even_when_flag_is_on(monkeypatch):
    monkeypatch.setenv("OPS_LOG_SINK", "true")
    app = SimpleNamespace(state=SimpleNamespace())
    rt = await runtime.start_ops(app, None, SimpleNamespace(kv_storage="JsonKVStorage"))
    try:
        assert rt.log_sink is None and _sink_handlers() == []
    finally:
        await runtime.stop_ops(app)


async def test_heartbeat_exports_sink_health():
    from lightrag.api.ops.heartbeat import Heartbeat

    sink = _sink(max_buffer=1)
    sink.handler.handle(_record("one"))
    sink.handler.handle(_record("two", lineno=2))
    beat = Heartbeat(
        worker_id=WORKER,
        mode="uvicorn",
        version="x",
        store=None,
        reader=SimpleNamespace(state="local"),
        app_state=SimpleNamespace(),
        leases=lambda: [],
        data_rag=lambda: None,
    )
    beat.log_sink = sink
    health = beat._log_sink(time.time())
    assert health["buffer"] == 1 and health["dropped"] == 1
    assert health["dropped_5m"] == 0  # first sample of the window
    assert set(health) >= {"buffer", "dropped", "rate_limited", "write_failures"}
