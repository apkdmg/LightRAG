"""LightRAG's own alert emails must not be ingested when a connector returns them."""

import importlib
import json
import sys
from email.message import EmailMessage
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Importing the routers package triggers import-time arg parsing (auth handler);
# neutralize pytest's argv during the import, mirroring the other API tests.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_email_routes = importlib.import_module("lightrag.api.routers.email_routes")
_alerts = importlib.import_module("lightrag.api.provider_alerts")
_rejection = importlib.import_module("lightrag.provider_rejection")
sys.argv = _original_argv

ALERT_SUBJECT_TAG = _alerts.ALERT_SUBJECT_TAG
ProviderRejection = _rejection.ProviderRejection
ProviderRejectionEvent = _rejection.ProviderRejectionEvent
ProviderRejectionNotifier = _alerts.ProviderRejectionNotifier
SmtpSettings = _alerts.SmtpSettings
is_own_alert_email = _alerts.is_own_alert_email

pytestmark = pytest.mark.offline

SENDER = "rag@mailer.example.test"
ALERT_SUBJECT = f"{ALERT_SUBJECT_TAG} LLM provider refused - workspace a (extraction)"


def _settings() -> SmtpSettings:
    return SmtpSettings(
        host="mail.example.test",
        port=25,
        security="none",
        username="",
        password="",
        sender=SENDER,
        timeout=5,
        admin_recipients=("admin@example.test",),
        repeat_seconds=3600,
    )


@pytest.mark.parametrize(
    "from_address, subject, expected",
    [
        (f"LightRAG alerts <{SENDER}>", ALERT_SUBJECT, True),
        (SENDER, ALERT_SUBJECT, True),
        (SENDER.upper(), ALERT_SUBJECT, True),
        # A person's reply or forward carries the tag but not the sender.
        ("Pat <pat@example.test>", f"Fwd: {ALERT_SUBJECT}", False),
        # Other mail from a shared sender address is not ours to drop.
        (SENDER, "Timetable for next week", False),
        # A display name alone proves nothing.
        (f"{SENDER} <pat@example.test>", ALERT_SUBJECT, False),
        ("", ALERT_SUBJECT, False),
        (None, None, False),
    ],
)
def test_recognises_only_own_alerts(from_address, subject, expected):
    assert is_own_alert_email(from_address, subject, SENDER) is expected


def test_nothing_matches_without_smtp_from():
    assert not is_own_alert_email("", ALERT_SUBJECT, "")
    assert not is_own_alert_email(SENDER, ALERT_SUBJECT, "")


def test_reads_smtp_from_from_the_environment(monkeypatch):
    monkeypatch.setenv("SMTP_FROM", SENDER)
    assert is_own_alert_email(SENDER, ALERT_SUBJECT)
    monkeypatch.delenv("SMTP_FROM")
    assert not is_own_alert_email(SENDER, ALERT_SUBJECT)


async def test_provider_alert_as_sent_is_recognised():
    sent: list[EmailMessage] = []
    notifier = ProviderRejectionNotifier(
        _settings(),
        is_byo_slot=lambda *_: False,
        owner_email=lambda _ws: None,
        send=lambda _s, _to, msg: sent.append(msg),
    )
    await notifier(
        ProviderRejectionEvent(
            workspace="a",
            role="extract",
            rejection=ProviderRejection(
                kind="budget_exhausted", status_code=412, message="limit reached"
            ),
            host="https://router.example.test/v1",
            model="m",
            paused=True,
            retry_after_minutes=30,
        )
    )
    assert sent
    assert is_own_alert_email(sent[0]["From"], sent[0]["Subject"], SENDER)


def _client(monkeypatch, ingested: list[str]) -> TestClient:
    monkeypatch.setenv("SMTP_FROM", SENDER)

    async def fake_ingest(self, parsed, **_kwargs):
        ingested.append(parsed.subject)
        return {
            "documents_created": 1,
            "attachments_processed": 0,
            "inline_images_processed": 0,
        }

    async def fake_rag(_request, default):
        return default

    monkeypatch.setattr(
        _email_routes.EmailIngestionService, "ingest_email", fake_ingest
    )
    monkeypatch.setattr(_email_routes, "get_lightrag_for_request", fake_rag)
    app = FastAPI()
    app.include_router(_email_routes.create_email_routes(SimpleNamespace()))
    return TestClient(app)


def _eml(from_address: str, subject: str) -> bytes:
    message = EmailMessage()
    message["From"] = from_address
    message["To"] = "admin@example.test"
    message["Subject"] = subject
    message["Message-ID"] = "<id-1@example.test>"
    message.set_content("The LLM provider refused the request.")
    return message.as_bytes()


def test_eml_upload_of_own_alert_is_skipped(monkeypatch):
    ingested: list[str] = []
    response = _client(monkeypatch, ingested).post(
        "/documents/email",
        files={
            "email_file": (
                "alert.eml",
                _eml(f"LightRAG alerts <{SENDER}>", ALERT_SUBJECT),
                "message/rfc822",
            )
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert response.json()["track_id"] == ""
    assert ingested == []


def test_structured_upload_of_own_alert_is_skipped(monkeypatch):
    ingested: list[str] = []
    response = _client(monkeypatch, ingested).post(
        "/documents/email",
        data={
            "metadata": json.dumps(
                {"from": SENDER, "to": ["a@example.test"], "subject": ALERT_SUBJECT}
            ),
            "body_text": "The LLM provider refused the request.",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    assert ingested == []


def test_forward_of_an_alert_is_still_ingested(monkeypatch):
    ingested: list[str] = []
    subject = f"Fwd: {ALERT_SUBJECT}"
    response = _client(monkeypatch, ingested).post(
        "/documents/email",
        files={
            "email_file": (
                "fwd.eml",
                _eml("Pat <pat@example.test>", subject),
                "message/rfc822",
            )
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert ingested == [subject]
