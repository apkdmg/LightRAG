"""Server logging setup: redacted handlers, one handler pair, LOG_LEVEL, uvicorn.error."""

from __future__ import annotations

import importlib
import logging
import logging.config
import os
import sys

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def restore_logging():
    """Snapshot every logger the setup functions touch and restore it after."""
    names = ["lightrag", "uvicorn", "uvicorn.access", "uvicorn.error"] + [
        n for n in logging.root.manager.loggerDict if n.startswith("lightrag.")
    ]
    saved = {}
    for name in names:
        lg = logging.getLogger(name)
        saved[name] = (list(lg.handlers), list(lg.filters), lg.level, lg.propagate)
    yield
    for name in [
        n
        for n in logging.root.manager.loggerDict
        if n in saved or n.startswith("lightrag") or n.startswith("uvicorn")
    ]:
        lg = logging.getLogger(name)
        for handler in lg.handlers:
            if name in saved and handler in saved[name][0]:
                continue
            try:
                handler.close()
            except Exception:
                pass
        if name in saved:
            handlers, filters, level, propagate = saved[name]
            lg.handlers = handlers
            lg.filters = filters
            lg.setLevel(level)
            lg.propagate = propagate


def _flush(*names: str) -> None:
    for name in names:
        for handler in logging.getLogger(name).handlers:
            handler.flush()


def _server_module():
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        return importlib.import_module("lightrag.api.lightrag_server")
    finally:
        sys.argv = original_argv


def test_uvicorn_logging_redacts_file_output(tmp_path, monkeypatch, restore_logging):
    server = _server_module()
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    server.configure_logging()
    assert oct(log_dir.stat().st_mode & 0o777) == "0o700"

    logging.getLogger("lightrag.api.email").warning(
        "[opso_unimas_my] Vision model failed for Gaji_Ahmad.pdf: key sk-live-ABCDEFGH1234"
    )
    logging.getLogger("lightrag").error("contact bob@unimas.my or 012-345 6789")
    _flush("lightrag")
    written = (log_dir / "lightrag.log").read_text(encoding="utf-8")
    for secret in (
        "Gaji_Ahmad",
        "sk-live-ABCDEFGH1234",
        "bob@unimas.my",
        "012-345 6789",
    ):
        assert secret not in written
    # Context fields: pid, request id ("-" outside a request), workspace.
    assert f"[{os.getpid()}] [-] [opso_unimas_my] - [opso_unimas_my] Vision" in written
    # A child logger's record is written once, through the "lightrag" handlers.
    assert written.count("Vision model failed") == 1


def test_gunicorn_logconfig_honours_level(monkeypatch):
    gunicorn_config = importlib.import_module("lightrag.api.gunicorn_config")
    config = gunicorn_config.build_logconfig_dict("debug")
    for name in ("lightrag", "gunicorn", "gunicorn.error", "gunicorn.access"):
        assert config["loggers"][name]["level"] == "DEBUG"
    for handler in config["handlers"].values():
        assert handler["filters"] == ["redact", "context"]
    monkeypatch.setenv("LOG_LEVEL", "warning")
    assert gunicorn_config.build_logconfig_dict(None)["loggers"]["lightrag"][
        "level"
    ] == ("WARNING")
    assert gunicorn_config.resolve_log_level("nonsense") == "INFO"
    assert gunicorn_config.resolve_log_level("warn") == "WARNING"


def test_gunicorn_logconfig_applies(tmp_path, monkeypatch, restore_logging):
    """The dictConfig Gunicorn applies in the master is valid and redacts."""
    gunicorn_config = importlib.import_module("lightrag.api.gunicorn_config")
    log_file = tmp_path / "gunicorn.log"
    monkeypatch.setattr(gunicorn_config, "log_file_path", str(log_file))
    names = ("gunicorn", "gunicorn.error", "gunicorn.access")
    saved = {
        n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level)
        for n in names
    }
    try:
        logging.config.dictConfig(gunicorn_config.build_logconfig_dict("info"))
        logging.getLogger("gunicorn.error").error("worker died for carol@unimas.my")
        _flush("gunicorn.error")
        written = log_file.read_text(encoding="utf-8")
        assert "carol@unimas.my" not in written and "<email>" in written
    finally:
        for name, (handlers, level) in saved.items():
            lg = logging.getLogger(name)
            for handler in lg.handlers:
                if handler not in handlers:
                    handler.close()
            lg.handlers = handlers
            lg.setLevel(level)


def test_gunicorn_worker_logging_one_handler_pair(
    tmp_path, monkeypatch, restore_logging
):
    gunicorn_config = importlib.import_module("lightrag.api.gunicorn_config")
    log_file = tmp_path / "worker.log"
    monkeypatch.setattr(gunicorn_config, "log_file_path", str(log_file))

    child = logging.getLogger("lightrag.api.ops_test_child")
    child.addHandler(logging.NullHandler())
    child.propagate = False

    gunicorn_config.configure_worker_logging("warning")

    root = logging.getLogger("lightrag")
    assert len(root.handlers) == 2
    assert root.level == logging.WARNING
    for name in ("uvicorn", "uvicorn.access"):
        assert logging.getLogger(name).handlers == root.handlers
    assert child.handlers == [] and child.propagate is True
    assert child.level == logging.NOTSET

    uvicorn_error = logging.getLogger("uvicorn.error")
    assert uvicorn_error.handlers == [] and uvicorn_error.propagate is True
    assert uvicorn_error.level == logging.ERROR

    child.warning("child line for dave@unimas.my")
    uvicorn_error.error("Exception in ASGI application")
    uvicorn_error.info("Started server process")
    root.info("below the configured level")
    _flush("lightrag")
    written = log_file.read_text(encoding="utf-8")
    assert written.count("child line for <email>") == 1
    assert "dave@unimas.my" not in written
    assert "Exception in ASGI application" in written
    assert "Started server process" not in written
    assert "below the configured level" not in written
