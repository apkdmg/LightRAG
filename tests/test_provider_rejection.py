"""Provider rejection handling: classification, ingestion pause, admin alerts.

A rejected key / exhausted budget must pause the workspace's ingestion after
the first failure (no provider call per remaining document), keep the other
documents pending, report exactly one event, and resume once the key works.
"""

from uuid import uuid4

import httpx
import numpy as np
import openai
import pytest

from lightrag.api.provider_alerts import ProviderRejectionNotifier, SmtpSettings
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.lightrag import LightRAG
from lightrag.pipeline import PROVIDER_PAUSE_KEY
from lightrag.provider_rejection import (
    ProviderRejection,
    ProviderRejectionEvent,
    add_provider_rejection_listener,
    classify_provider_rejection,
    remove_provider_rejection_listener,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

_REQUEST = httpx.Request("POST", "https://router.example.test/v1/chat/completions")


def _status_error(cls, code: int, message: str):
    return cls(
        f"Error code: {code}",
        response=httpx.Response(code, request=_REQUEST),
        body={"error": {"origin": "router", "message": message}},
    )


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_classifies_wrapped_permission_denied():
    original = _status_error(
        openai.PermissionDeniedError, 403, "Invalid authorization token"
    )
    try:
        try:
            raise original
        except Exception as e:
            raise RuntimeError("chunk-000: PermissionDeniedError") from e
    except RuntimeError as outer:
        wrapped = outer

    rejection = classify_provider_rejection(wrapped)
    assert rejection == ProviderRejection(
        kind="invalid_key", status_code=403, message="Invalid authorization token"
    )


def test_classifies_spend_limit_and_ignores_transient_errors():
    budget = _status_error(openai.APIStatusError, 412, "monthly spend limit reached")
    assert classify_provider_rejection(budget).kind == "budget_exhausted"
    assert (
        classify_provider_rejection(
            _status_error(openai.AuthenticationError, 401, "bad key")
        ).kind
        == "invalid_key"
    )

    assert (
        classify_provider_rejection(
            _status_error(openai.RateLimitError, 429, "slow down")
        )
        is None
    )
    assert (
        classify_provider_rejection(
            _status_error(openai.BadRequestError, 400, "file is corrupted")
        )
        is None
    )
    assert classify_provider_rejection(RuntimeError("boom")) is None
    assert classify_provider_rejection(None) is None


def test_provider_message_masks_key_material():
    err = _status_error(
        openai.AuthenticationError, 401, "Incorrect API key sk-abcdef1234567890XYZ"
    )
    message = classify_provider_rejection(err).message
    assert "1234567890XYZ" not in message
    assert message.startswith("Incorrect API key sk-abcdef")


# --------------------------------------------------------------------------- #
# Pipeline pause
# --------------------------------------------------------------------------- #


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


def _no_op_chunking(tokenizer, content, *_args, **_kwargs) -> list[dict]:
    return [{"tokens": 1, "content": content, "chunk_order_index": 0}]


class _Provider:
    """LLM stub whose key is rejected until ``accept`` is set."""

    def __init__(self):
        self.accept = False
        self.calls = 0

    async def __call__(self, *args, **kwargs) -> str:
        self.calls += 1
        if not self.accept:
            raise _status_error(
                openai.PermissionDeniedError, 403, "Invalid authorization token"
            )
        return "ok"


async def _build_rag(tmp_path, provider: _Provider) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "provider_pause"),
        workspace=f"pausetest_{uuid4().hex[:8]}",
        llm_model_func=provider,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_no_op_chunking,
        max_parallel_insert=1,
        enable_llm_cache_for_entity_extract=False,
    )
    await rag.initialize_storages()
    return rag


async def _statuses(rag) -> dict[str, str]:
    docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    return {
        did: str(getattr(st.status, "value", st.status)) for did, st in docs.items()
    }


async def _pause(rag):
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    return status.get(PROVIDER_PAUSE_KEY)


async def _expire_pause(rag) -> None:
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
        status[PROVIDER_PAUSE_KEY] = {**status[PROVIDER_PAUSE_KEY], "retry_at": 0}


async def _clear_pause(rag) -> None:
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
        status.pop(PROVIDER_PAUSE_KEY, None)


async def _cleanup(rag, listener) -> None:
    remove_provider_rejection_listener(listener)
    task = getattr(rag, "_provider_resume_task", None)
    if task is not None:
        task.cancel()
    await rag.finalize_storages()


async def _enqueue(rag, n: int) -> None:
    await rag.apipeline_enqueue_documents(
        [f"document number {i} about topic {i}" for i in range(n)],
        file_paths=[f"doc_{i}.txt" for i in range(n)],
    )


async def test_rejection_pauses_ingestion_and_resumes_after_key_fix(tmp_path):
    provider = _Provider()
    events: list[ProviderRejectionEvent] = []

    async def listener(event):
        events.append(event)

    add_provider_rejection_listener(listener)
    rag = await _build_rag(tmp_path, provider)
    try:
        await _enqueue(rag, 4)
        await rag.apipeline_process_enqueue_documents()

        # One provider call, one event; the rest of the queue waits.
        assert provider.calls == 1
        assert len(events) == 1
        assert events[0].workspace == rag.workspace
        assert events[0].role == "extract"
        assert events[0].rejection.status_code == 403
        assert events[0].paused is True
        statuses = sorted((await _statuses(rag)).values())
        assert statuses == ["failed", "pending", "pending", "pending"]
        assert (await _pause(rag))["kind"] == "invalid_key"

        # New uploads while paused do not reach the provider.
        await rag.apipeline_process_enqueue_documents()
        assert provider.calls == 1

        # Key fixed + pause lifted (what saving the Providers page does).
        provider.accept = True
        await _clear_pause(rag)
        await rag.apipeline_process_enqueue_documents()
        assert set((await _statuses(rag)).values()) == {"processed"}
        assert len(events) == 1
    finally:
        await _cleanup(rag, listener)


async def test_probe_after_retry_window(tmp_path):
    provider = _Provider()
    events: list[ProviderRejectionEvent] = []

    async def listener(event):
        events.append(event)

    add_provider_rejection_listener(listener)
    rag = await _build_rag(tmp_path, provider)
    try:
        await _enqueue(rag, 3)
        await rag.apipeline_process_enqueue_documents()
        assert provider.calls == 1

        # Window elapsed, key still bad: exactly one probe call, re-armed pause.
        await _expire_pause(rag)
        await rag.apipeline_process_enqueue_documents()
        assert provider.calls == 2
        assert len(events) == 2
        pause = await _pause(rag)
        assert pause is not None and pause["retry_at"] > 0

        # Window elapsed, key now good: the probe succeeds, the pause clears
        # and the documents set aside in the same run are picked back up.
        provider.accept = True
        await _expire_pause(rag)
        await rag.apipeline_process_enqueue_documents()
        assert await _pause(rag) is None
        assert set((await _statuses(rag)).values()) == {"processed"}
    finally:
        await _cleanup(rag, listener)


# --------------------------------------------------------------------------- #
# Admin email notifier
# --------------------------------------------------------------------------- #


def _settings(**overrides) -> SmtpSettings:
    base = dict(
        host="smtp.example.test",
        port=587,
        security="starttls",
        username="",
        password="",
        sender="lightrag@example.test",
        timeout=5,
        admin_recipients=("admin@example.test",),
        repeat_seconds=3600,
    )
    base.update(overrides)
    return SmtpSettings(**base)


def _event(workspace="ws_alice", role="extract", code=403) -> ProviderRejectionEvent:
    return ProviderRejectionEvent(
        workspace=workspace,
        role=role,
        rejection=ProviderRejection(
            kind="invalid_key" if code in (401, 403) else "budget_exhausted",
            status_code=code,
            message="Invalid authorization token",
        ),
        host="https://router.example.test/v1",
        model="openai/gpt-x",
        paused=True,
        retry_after_minutes=30,
    )


def _notifier(sent, *, byo=True, owner="alice@example.test", **settings):
    async def owner_email(_workspace):
        return owner

    return ProviderRejectionNotifier(
        _settings(**settings),
        is_byo_slot=lambda _ws, _slot: byo,
        owner_email=owner_email,
        send=lambda _s, to, msg: sent.append((to, msg)),
    )


async def test_byo_key_alert_goes_to_admins_and_owner():
    sent = []
    await _notifier(sent)(_event())
    assert len(sent) == 1
    to, msg = sent[0]
    assert to == ["admin@example.test", "alice@example.test"]
    body = msg.get_content()
    assert "ws_alice" in msg["Subject"] and "extraction" in msg["Subject"]
    assert "HTTP 403: Invalid authorization token" in body
    assert "Providers page" in body
    assert "paused" in body


async def test_system_key_alert_goes_to_admins_only():
    sent = []
    await _notifier(sent, byo=False)(_event())
    to, msg = sent[0]
    assert to == ["admin@example.test"]
    assert ".env" in msg.get_content()


async def test_alert_is_rate_limited_per_problem():
    sent = []
    notifier = _notifier(sent)
    await notifier(_event())
    await notifier(_event())  # same problem: suppressed
    await notifier(_event(code=412))  # different problem: sent
    await notifier(_event(role="vlm"))  # different slot: sent
    assert len(sent) == 3


async def test_alert_skipped_without_smtp_config():
    sent = []
    await _notifier(sent, host="")(_event())
    await _notifier(sent, byo=False, admin_recipients=())(_event())
    assert sent == []


async def test_alert_send_failure_does_not_raise():
    def boom(*_args):
        raise OSError("connection refused")

    async def owner_email(_ws):
        return None

    notifier = ProviderRejectionNotifier(
        _settings(), is_byo_slot=lambda *_: False, owner_email=owner_email, send=boom
    )
    await notifier(_event())  # logged, not raised


# --------------------------------------------------------------------------- #
# Saving provider credentials lifts the pause
# --------------------------------------------------------------------------- #


async def test_provider_save_lifts_pause_and_restarts_pipeline():
    import asyncio
    import importlib
    import sys

    from lightrag.kg.shared_storage import (
        initialize_pipeline_status,
        initialize_share_data,
    )

    argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    provider_routes = importlib.import_module("lightrag.api.routers.provider_routes")
    sys.argv = argv

    workspace = f"saveresume_{uuid4().hex[:8]}"
    initialize_share_data()
    await initialize_pipeline_status(workspace=workspace)
    status = await get_namespace_data("pipeline_status", workspace=workspace)
    status[PROVIDER_PAUSE_KEY] = {"kind": "invalid_key", "retry_at": 9e18}

    started = asyncio.Event()

    class _Rag:
        async def apipeline_process_enqueue_documents(self):
            started.set()

    class _Manager:
        async def get_lightrag_instance(self, ws):
            assert ws == workspace
            return _Rag()

    await provider_routes._resume_paused_ingestion(_Manager(), workspace)
    await asyncio.wait_for(started.wait(), timeout=5)
    assert PROVIDER_PAUSE_KEY not in status

    # No pause: nothing to resume, no pipeline kick.
    started.clear()
    await provider_routes._resume_paused_ingestion(_Manager(), workspace)
    await asyncio.sleep(0)
    assert not started.is_set()


async def test_query_time_rejection_is_reported_and_reraised(tmp_path):
    provider = _Provider()
    events: list[ProviderRejectionEvent] = []

    async def listener(event):
        events.append(event)

    add_provider_rejection_listener(listener)
    rag = await _build_rag(tmp_path, provider)
    try:
        query_func = rag._role_llm_states["query"].wrapped
        assert callable(getattr(query_func, "shutdown", None))
        with pytest.raises(openai.PermissionDeniedError):
            await query_func("what is in my mailbox?")
        assert [(e.role, e.paused) for e in events] == [("query", False)]

        # Ingestion roles are reported by the pipeline instead (no double event).
        with pytest.raises(openai.PermissionDeniedError):
            await rag._role_llm_states["extract"].wrapped("extract this")
        assert len(events) == 1
    finally:
        await _cleanup(rag, listener)


async def test_explicit_system_key_skips_owner_lookup():
    sent = []
    event = ProviderRejectionEvent(
        workspace="ws_alice",
        role="vlm",
        rejection=ProviderRejection("invalid_key", 403, "bad key"),
        byo_key=False,
    )
    await _notifier(sent, byo=True)(event)  # store says BYO; event says system
    to, msg = sent[0]
    assert to == ["admin@example.test"]
    assert ".env" in msg.get_content()


def test_send_email_sync_uses_starttls_login_and_ssl(monkeypatch):
    from email.message import EmailMessage

    from lightrag.api import provider_alerts

    calls = []

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None, context=None):
            calls.append(("connect", type(self).__name__, host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.append(("quit",))

        def starttls(self, context=None):
            calls.append(("starttls",))

        def login(self, user, password):
            calls.append(("login", user))

        def send_message(self, msg, to_addrs=None):
            calls.append(("send", tuple(to_addrs)))

    class _FakeSMTP_SSL(_FakeSMTP):
        pass

    monkeypatch.setattr(provider_alerts.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(provider_alerts.smtplib, "SMTP_SSL", _FakeSMTP_SSL)
    msg = EmailMessage()

    provider_alerts.send_email_sync(
        _settings(username="relay", password="pw"), ["a@example.test"], msg
    )
    assert calls == [
        ("connect", "_FakeSMTP", "smtp.example.test", 587),
        ("starttls",),
        ("login", "relay"),
        ("send", ("a@example.test",)),
        ("quit",),
    ]

    calls.clear()
    provider_alerts.send_email_sync(
        _settings(security="ssl", port=465), ["a@example.test"], msg
    )
    assert calls[0] == ("connect", "_FakeSMTP_SSL", "smtp.example.test", 465)
    assert ("starttls",) not in calls and not any(c[0] == "login" for c in calls)
