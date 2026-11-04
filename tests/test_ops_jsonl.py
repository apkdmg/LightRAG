"""Per-PID JSONL log files and the live INFO tail (PR-9, §3.8.6).

The handler writes redacted JSON lines, 0600, rotated by size, never through
a symlink; the tail k-way merges the newest lines of every live (or just
exited) process by timestamp, reads only the files' ends, accepts only its
own file names, redacts again on read, and is admin-only.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from types import SimpleNamespace

import httpx
import pytest

from lightrag.api.ops import jsonl
from tests.ops_fixtures import SECRET_EMAIL, SECRET_KEY, build_app, ops_routes

pytestmark = pytest.mark.offline


def _line(ts: float, msg: str, *, level: int = 20, logger="lightrag", module="x"):
    return json.dumps(
        {
            "ts": ts,
            "level": logging.getLevelName(level),
            "levelno": level,
            "logger": logger,
            "module": module,
            "lineno": 1,
            "pid": 1,
            "msg": msg,
        }
    )


def _write(path, lines) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(msg: str, *, name="lightrag", level=logging.INFO, exc_info=None):
    return logging.LogRecord(name, level, "/app/lightrag/x.py", 7, msg, None, exc_info)


# -- the handler ------------------------------------------------------------------------------


def test_handler_writes_redacted_json_lines_owner_only(tmp_path):
    path = tmp_path / jsonl.file_name(4242)
    handler = jsonl.JsonlFileHandler(str(path))
    try:
        handler.handle(_record(f"call failed for {SECRET_EMAIL} with key {SECRET_KEY}"))
        try:
            raise RuntimeError(f"Bearer {SECRET_KEY}")
        except RuntimeError:
            import sys

            handler.handle(
                _record("boom", level=logging.ERROR, exc_info=sys.exc_info())
            )
        # A logger outside the console's set is not written.
        handler.handle(_record("third party", name="urllib3"))
        handler.flush()
    finally:
        handler.close()
    text = path.read_text(encoding="utf-8")
    assert SECRET_EMAIL not in text and SECRET_KEY not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 2
    assert rows[0]["level"] == "INFO" and rows[0]["levelno"] == 20
    assert rows[0]["logger"] == "lightrag" and rows[0]["lineno"] == 7
    assert rows[1]["exc_type"] == "RuntimeError" and "Traceback" in rows[1]["exc"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_handler_rotates_and_keeps_permissions(tmp_path):
    path = tmp_path / jsonl.file_name(77)
    handler = jsonl.JsonlFileHandler(str(path), max_bytes=2048, backup_count=2)
    try:
        for i in range(200):
            handler.handle(_record(f"line {i} " + "x" * 40))
    finally:
        handler.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["lightrag-77.jsonl", "lightrag-77.jsonl.1", "lightrag-77.jsonl.2"]
    for name in names:
        info = os.stat(tmp_path / name)
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_size <= 2048 + 512


def test_handler_refuses_a_planted_symlink(tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep\n")
    path = tmp_path / jsonl.file_name(5)
    os.symlink(target, path)
    handler = jsonl.JsonlFileHandler(str(path))
    try:
        handler.handle(_record("must not land in the target"))
    finally:
        handler.close()
    assert target.read_text() == "keep\n"
    assert handler.errors >= 1


def test_start_and_detach_attach_to_the_console_loggers(tmp_path):
    logs = jsonl.start_jsonl(str(tmp_path), pid=99)
    try:
        assert logs.handler in logging.getLogger("lightrag").handlers
        assert logs.handler in logging.getLogger().handlers
        logging.getLogger("lightrag.api.test").warning("hello %s", SECRET_EMAIL)
    finally:
        logs.detach()
    assert logs.handler not in logging.getLogger("lightrag").handlers
    text = (tmp_path / "lightrag-99.jsonl").read_text()
    assert "hello" in text and SECRET_EMAIL not in text


def test_prune_deletes_only_old_files_of_dead_processes(tmp_path):
    for pid in (11, 12, 13):
        _write(tmp_path / jsonl.file_name(pid), [_line(1.0, "x")])
    old = time.time() - jsonl.DEAD_KEEP_S - 60
    os.utime(tmp_path / jsonl.file_name(11), (old, old))
    os.utime(tmp_path / jsonl.file_name(12), (old, old))
    removed = jsonl.prune_dead_files(str(tmp_path), alive=lambda pid, last: pid == 12)
    assert removed == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "lightrag-12.jsonl",
        "lightrag-13.jsonl",
    ]


# -- the tail ----------------------------------------------------------------------------------


def test_tail_merges_processes_newest_first(tmp_path):
    base = time.time() - 100
    _write(
        tmp_path / jsonl.file_name(101),
        [_line(base + i * 2, f"a{i}") for i in range(10)],
    )
    _write(
        tmp_path / jsonl.file_name(102),
        [_line(base + i * 2 + 1, f"b{i}") for i in range(10)],
    )
    out = jsonl.tail(str(tmp_path), lines=6, alive=lambda pid, last: True)
    assert [e["msg"] for e in out["items"]] == ["b9", "a9", "b8", "a8", "b7", "a7"]
    stamps = [e["ts"] for e in out["items"]]
    assert stamps == sorted(stamps, reverse=True)
    assert {s["pid"] for s in out["sources"]} == {101, 102}
    assert out["newest_ts"] == stamps[0]


def test_tail_level_and_module_filters_and_cap(tmp_path):
    now = time.time()
    _write(
        tmp_path / jsonl.file_name(7),
        [
            _line(now - 5, "debug", level=10),
            _line(now - 4, "info a", module="pipeline"),
            _line(now - 3, "warn b", level=30, logger="lightrag.api.email"),
            _line(now - 2, "error c", level=40, module="operate"),
        ],
    )
    warn = jsonl.tail(str(tmp_path), level=30, alive=lambda pid, last: True)
    assert [e["msg"] for e in warn["items"]] == ["error c", "warn b"]
    by_module = jsonl.tail(
        str(tmp_path), module="pipeline", alive=lambda pid, last: True
    )
    assert [e["msg"] for e in by_module["items"]] == ["info a"]
    by_logger = jsonl.tail(
        str(tmp_path), module="lightrag.api", alive=lambda pid, last: True
    )
    assert [e["msg"] for e in by_logger["items"]] == ["warn b"]
    assert jsonl.tail(str(tmp_path), lines=10**6, alive=lambda p, last: True)[
        "lines"
    ] == (jsonl.TAIL_MAX_LINES)


def test_tail_reads_only_the_end_of_large_files(tmp_path):
    now = time.time()
    lines = [_line(now - 50000 + i, f"m{i:06d} " + "y" * 100) for i in range(40000)]
    _write(tmp_path / jsonl.file_name(8), lines)
    size = os.path.getsize(tmp_path / jsonl.file_name(8))
    assert size > 4 * jsonl.MAX_READ_BYTES
    out = jsonl.tail(str(tmp_path), lines=5, alive=lambda pid, last: True)
    source = out["sources"][0]
    assert source["bytes_read"] <= jsonl.MAX_READ_BYTES
    assert source["truncated"] is True
    assert out["items"][0]["msg"].startswith("m039999")
    # A cut first line is dropped, never parsed half.
    assert all(e["msg"].startswith("m") for e in out["items"])


def test_tail_reads_the_rotated_file_when_the_current_one_is_short(tmp_path):
    now = time.time()
    _write(tmp_path / "lightrag-9.jsonl.1", [_line(now - 10, "older")])
    _write(tmp_path / "lightrag-9.jsonl", [_line(now - 1, "newer")])
    out = jsonl.tail(str(tmp_path), alive=lambda pid, last: True)
    assert [e["msg"] for e in out["items"]] == ["newer", "older"]


def test_tail_skips_dead_processes_unless_recent(tmp_path):
    now = time.time()
    _write(tmp_path / jsonl.file_name(201), [_line(now - 5, "recent exit")])
    _write(tmp_path / jsonl.file_name(202), [_line(now - 5, "long gone")])
    old = now - jsonl.RECENT_EXIT_S - 60
    os.utime(tmp_path / jsonl.file_name(202), (old, old))
    out = jsonl.tail(str(tmp_path), alive=lambda pid, last: False)
    assert [e["msg"] for e in out["items"]] == ["recent exit"]
    assert out["sources"][0]["alive"] is False


def test_tail_accepts_only_its_own_regular_files(tmp_path):
    now = time.time()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.jsonl"
    _write(secret, [_line(now, "outside the log dir")])
    logs = tmp_path / "logs"
    logs.mkdir()
    # A symlink with a valid name, odd names and a directory are all ignored.
    os.symlink(secret, logs / jsonl.file_name(300))
    _write(logs / "lightrag-301.jsonl.bak", [_line(now, "backup")])
    _write(logs / "other.jsonl", [_line(now, "other")])
    (logs / "lightrag-302.jsonl").mkdir()
    _write(logs / jsonl.file_name(303), [_line(now - 1, "mine")])
    out = jsonl.tail(str(logs), alive=lambda pid, last: True)
    assert [e["msg"] for e in out["items"]] == ["mine"]
    assert jsonl._open_regular(str(logs), "../outside/secret.jsonl") is None
    assert jsonl._open_regular(str(logs), jsonl.file_name(300)) is None


def test_tail_redacts_again_on_read(tmp_path):
    _write(
        tmp_path / jsonl.file_name(5),
        [
            _line(time.time(), f"raw {SECRET_KEY} {SECRET_EMAIL}"),
            "not json",
            json.dumps({"msg": "no ts"}),
        ],
    )
    out = jsonl.tail(str(tmp_path), alive=lambda pid, last: True)
    assert len(out["items"]) == 1
    text = json.dumps(out)
    assert SECRET_KEY not in text and SECRET_EMAIL not in text


def test_tail_keeps_the_longest_lines_the_writer_can_produce(tmp_path):
    # An ERROR with a non-ASCII message and traceback at their character caps
    # is far over 16 KiB of UTF-8; control characters escape to 6 bytes.
    from lightrag.api.ops.redact import EXC_MAX_CHARS, MSG_MAX_CHARS

    path = tmp_path / jsonl.file_name(4343)
    handler = jsonl.JsonlFileHandler(str(path))
    try:
        try:
            raise RuntimeError("\u5b9e\u4f53" * EXC_MAX_CHARS)
        except RuntimeError:
            import sys

            record = _record("\u5b9e\u4f53\x01" * MSG_MAX_CHARS, level=40)
            record.exc_info = sys.exc_info()
            handler.handle(record)
    finally:
        handler.close()
    raw = path.read_bytes().rstrip(b"\n")
    assert len(raw) > 16 * 1024 and len(raw) <= jsonl.LINE_MAX_BYTES
    out = jsonl.tail(str(tmp_path), alive=lambda pid, last: True)
    assert len(out["items"]) == 1
    item = out["items"][0]
    assert item["level"] == 40 and item["msg"].startswith("\u5b9e\u4f53")
    assert item["exc"] and "RuntimeError" in item["exc"]


def test_a_reused_pid_does_not_revive_an_old_file(tmp_path, monkeypatch):
    now = time.time()
    _write(
        tmp_path / jsonl.file_name(9), [_line(now - 3600, "before restart", level=40)]
    )
    _write(tmp_path / jsonl.file_name(10), [_line(now - 1, "live writer", level=40)])
    last_9 = now - 3600
    os.utime(tmp_path / jsonl.file_name(9), (last_9, last_9))
    # PID 9 now belongs to a process that started after the file's last
    # write (a new container's master); PID 10 started before its write.
    started = {9: now - 60, 10: now - 600}
    monkeypatch.setattr(jsonl, "pid_started", lambda pid: started.get(pid))
    assert not jsonl.writer_alive(9, last_9)
    assert jsonl.writer_alive(10, now - 1)
    assert not jsonl.writer_alive(11, now)
    out = jsonl.tail(str(tmp_path), level=40)
    assert [e["msg"] for e in out["items"]] == ["live writer"]
    assert [(s["pid"], s["alive"]) for s in out["sources"]] == [(10, True)]
    # Once old enough, the stale file is pruned although PID 9 runs.
    old = now - jsonl.DEAD_KEEP_S - 60
    os.utime(tmp_path / jsonl.file_name(9), (old, old))
    assert jsonl.prune_dead_files(str(tmp_path)) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lightrag-10.jsonl"]


def test_writer_alive_for_this_process(tmp_path):
    assert jsonl.pid_started(os.getpid()) is not None
    assert jsonl.writer_alive(os.getpid(), time.time())
    assert not jsonl.writer_alive(os.getpid(), 0.0)
    assert jsonl.pid_started(0) is None


def test_read_tail_lines_is_bounded(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"".join(b"%05d\n" % i for i in range(10000)))
    fd = os.open(path, os.O_RDONLY)
    try:
        lines, read, truncated = jsonl.read_tail_lines(fd, 600)
    finally:
        os.close(fd)
    assert read == 600 and truncated
    assert lines[-1] == b"09999" and len(lines) == 99


# -- the route ----------------------------------------------------------------------------------


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def test_route_off_without_the_flag_and_without_postgres():
    app = build_app(None, kv_storage="JsonKVStorage")
    async with _client(app) as client:
        response = await client.get("/admin/ops/logs/tail")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False and body["items"] == []
    assert response.headers["cache-control"] == "no-store"


async def test_route_tails_the_files(tmp_path):
    os.getpid()
    _write(
        tmp_path / jsonl.file_name(os.getpid()),
        [_line(time.time() - 1, "first"), _line(time.time(), "second", level=30)],
    )
    app = build_app(None, kv_storage="JsonKVStorage")
    handler = SimpleNamespace(baseFilename=str(tmp_path / jsonl.file_name(os.getpid())))
    app.state.ops = SimpleNamespace(jsonl=SimpleNamespace(handler=handler))
    async with _client(app) as client:
        body = (await client.get("/admin/ops/logs/tail?lines=10")).json()
        assert body["enabled"] is True
        assert [e["msg"] for e in body["items"]] == ["second", "first"]
        warn = (await client.get("/admin/ops/logs/tail?level=WARNING")).json()
        assert [e["msg"] for e in warn["items"]] == ["second"]
        bad = await client.get("/admin/ops/logs/tail?module=../../etc")
        assert bad.status_code == 400 and bad.json()["detail"] == "invalid_module"
        too_many = await client.get("/admin/ops/logs/tail?lines=5000")
        assert too_many.status_code == 422


async def test_route_is_admin_only():
    from fastapi import HTTPException

    from lightrag.api.dependencies import require_admin

    app = build_app(None)

    def deny():
        raise HTTPException(status_code=403, detail="Admin access required")

    app.dependency_overrides[require_admin] = deny
    async with _client(app) as client:
        response = await client.get("/admin/ops/logs/tail")
    assert response.status_code == 403
    assert "/admin/ops/logs/tail" in {r.path for r in ops_routes.router.routes}


async def test_route_is_audited_like_a_log_search(tmp_path):
    from lightrag.api.ops.store import MemoryOpsStore

    _write(tmp_path / jsonl.file_name(os.getpid()), [_line(time.time(), "x")])
    store = MemoryOpsStore()
    app = build_app(store)
    handler = SimpleNamespace(baseFilename=str(tmp_path / jsonl.file_name(os.getpid())))
    app.state.ops.jsonl = SimpleNamespace(handler=handler)
    async with _client(app) as client:
        for _ in range(3):
            assert (await client.get("/admin/ops/logs/tail")).status_code == 200
    rows = [r for r in store.audit_rows if r["action"] == "log_tail"]
    # Once per actor and filters within the dedupe window.
    assert len(rows) == 1 and rows[0]["actor"] == "root"


def test_a_propagating_record_is_written_once(tmp_path):
    logs = jsonl.start_jsonl(str(tmp_path), pid=98)
    uvicorn_error = logging.getLogger("uvicorn.error")
    propagate = uvicorn_error.propagate
    uvicorn_error.propagate = True
    try:
        uvicorn_error.error("server hiccup")
    finally:
        uvicorn_error.propagate = propagate
        logs.detach()
    text = (tmp_path / "lightrag-98.jsonl").read_text()
    assert text.count("server hiccup") == 1
