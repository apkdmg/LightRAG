"""Offline tests for per-workspace BYO LLM / Vision-LLM provider credentials.

Covers the encrypted store (round-trip, at-rest secrecy, masking, fail-closed,
slot deletion, legacy migration) and the apply helper that maps the three
owner-facing slots onto LightRAG roles with reasoning control and correct
fallback behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from lightrag.api.workspace_providers import (
    EXTRACTION_ROLES,
    QUERY_ROLE,
    VISION_LLM_ROLE,
    ProviderSlot,
    WorkspaceProviderConfig,
    WorkspaceProviderError,
    WorkspaceProviderStore,
    apply_workspace_provider_overrides,
    generate_secret,
    slot_effective_view,
)

pytestmark = pytest.mark.offline


def _store(tmp_path, secret="unit-test-secret"):
    return WorkspaceProviderStore(tmp_path, get_secret=lambda: secret)


def _extraction_cfg():
    return WorkspaceProviderConfig(
        extraction=ProviderSlot(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr("sk-ext-abcd"),
            model="meta-llama/llama-3.3-70b-instruct",
            preset_id="openrouter",
        )
    )


def _full_cfg():
    return WorkspaceProviderConfig(
        extraction=ProviderSlot(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr("sk-ext-abcd"),
            model="gemini-2.0-flash",
            preset_id="openrouter",
        ),
        query=ProviderSlot(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=SecretStr("sk-qry-9999"),
            model="gemini-3.5-flash",
            reasoning_effort="medium",
            preset_id="gemini",
        ),
        vision=ProviderSlot(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=SecretStr("sk-vis-7777"),
            model="gemini-2.0-flash",
            preset_id="gemini",
        ),
    )


# ----------------------------- store -----------------------------
def test_encrypt_decrypt_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _full_cfg())
    got = store.get("ws1")
    assert got.extraction.is_active() and got.query.is_active() and got.vision.is_active()
    assert got.extraction.api_key.get_secret_value() == "sk-ext-abcd"
    assert got.query.api_key.get_secret_value() == "sk-qry-9999"
    assert got.query.reasoning_effort == "medium"
    assert got.extraction.reasoning_effort is None


def test_api_key_not_stored_in_plaintext(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _full_cfg())
    on_disk = (Path(tmp_path) / ".workspace_providers" / "ws1.json").read_text()
    for secret in ("sk-ext-abcd", "sk-qry-9999", "sk-vis-7777"):
        assert secret not in on_disk
    payload = json.loads(on_disk)
    assert payload["extraction"]["api_key_enc"]
    assert "api_key" not in payload["query"]


def test_masked_view_hides_secret(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _full_cfg())
    masked = store.get_masked("ws1")
    assert masked["extraction"]["api_key_preview"] == "...abcd"
    assert masked["query"]["reasoning_effort"] == "medium"
    assert masked["query"]["active"] is True
    for slot in ("extraction", "query", "vision"):
        assert "api_key" not in masked[slot]


def test_get_missing_returns_none(tmp_path):
    assert _store(tmp_path).get("nope") is None
    masked = _store(tmp_path).get_masked("nope")
    assert masked["extraction"]["active"] is False
    assert masked["query"]["active"] is False
    assert masked["vision"]["active"] is False


def test_fail_closed_without_secret(tmp_path):
    store = WorkspaceProviderStore(tmp_path, get_secret=lambda: None)
    assert store.has_secret() is False
    with pytest.raises(WorkspaceProviderError):
        store.set("ws1", _extraction_cfg())


def test_get_without_secret_falls_back_to_none(tmp_path):
    _store(tmp_path, secret="s1").set("ws1", _extraction_cfg())
    no_secret = WorkspaceProviderStore(tmp_path, get_secret=lambda: None)
    assert no_secret.get("ws1") is None


def test_wrong_secret_does_not_decrypt(tmp_path):
    _store(tmp_path, secret="right").set("ws1", _extraction_cfg())
    wrong = WorkspaceProviderStore(tmp_path, get_secret=lambda: "wrong")
    assert wrong.get("ws1") is None


def test_arbitrary_passphrase_is_accepted(tmp_path):
    store = WorkspaceProviderStore(tmp_path, get_secret=lambda: "a short phrase")
    store.set("ws1", _extraction_cfg())
    assert store.get("ws1").extraction.api_key.get_secret_value() == "sk-ext-abcd"


def test_generate_secret_is_usable(tmp_path):
    secret = generate_secret()
    store = WorkspaceProviderStore(tmp_path, get_secret=lambda: secret)
    store.set("ws1", _extraction_cfg())
    assert store.get("ws1").extraction.is_active()


def test_delete_slot_then_all(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _full_cfg())
    assert store.delete("ws1", "extraction") is True
    after = store.get("ws1")
    assert not after.extraction.is_active()
    assert after.query.is_active() and after.vision.is_active()
    assert store.delete("ws1", "query") is True
    assert store.delete("ws1", "vision") is True
    assert store.get("ws1") is None  # last slot removed -> file gone


def test_delete_invalid_slot_raises(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _extraction_cfg())
    with pytest.raises(ValueError):
        store.delete("ws1", "bogus")


def test_legacy_llm_config_migrates(tmp_path):
    """An old single-`llm` file maps onto extraction + query on read."""
    store = _store(tmp_path)
    fernet = store._fernet()
    enc = store._encode_slot(
        ProviderSlot(api_key=SecretStr("sk-legacy")), fernet
    )["api_key_enc"]
    legacy = {
        "llm": {"base_url": "https://legacy/v1", "model": "old-model", "api_key_enc": enc},
        "vision": {},
        "updated_by": "old",
    }
    (Path(tmp_path) / ".workspace_providers").mkdir(parents=True, exist_ok=True)
    (Path(tmp_path) / ".workspace_providers" / "wsl.json").write_text(json.dumps(legacy))

    cfg = store.get("wsl")
    assert cfg.extraction.base_url == "https://legacy/v1"
    assert cfg.query.base_url == "https://legacy/v1"
    assert cfg.extraction.api_key.get_secret_value() == "sk-legacy"
    assert not cfg.vision.is_active()


# ----------------------------- apply helper -----------------------------
class _FakeRag:
    def __init__(self, default_opts=None):
        self.calls = []
        self._default_opts = default_opts or {}

    def get_llm_role_config(self, role=None):
        return {
            "binding": "openai",
            "model": "default",
            "host": "https://api.openai.com/v1",
            "metadata": {"provider_options": dict(self._default_opts)},
        }

    async def aupdate_llm_role_config(self, role, **kwargs):
        self.calls.append((role, kwargs))


async def test_apply_maps_slots_to_roles(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _full_cfg())
    rag = _FakeRag(default_opts={"temperature": 0.0})
    await apply_workspace_provider_overrides(rag, "ws1", store)

    by_role = {r: kw for r, kw in rag.calls}
    assert set(by_role) == set(EXTRACTION_ROLES) | {QUERY_ROLE, VISION_LLM_ROLE}
    # extraction → extract + keyword, fast model, no reasoning override
    assert by_role["extract"]["model"] == "gemini-2.0-flash"
    assert by_role["extract"]["provider_options"] is None
    assert by_role["keyword"]["model"] == "gemini-2.0-flash"
    # query → reasoning merged onto existing options (temperature preserved)
    assert by_role["query"]["model"] == "gemini-3.5-flash"
    assert by_role["query"]["provider_options"] == {
        "temperature": 0.0,
        "reasoning_effort": "medium",
    }
    # vision → vlm
    assert by_role["vlm"]["model"] == "gemini-2.0-flash"


async def test_apply_partial_leaves_other_roles_default(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _extraction_cfg())  # only extraction
    rag = _FakeRag()
    await apply_workspace_provider_overrides(rag, "ws1", store)
    assert {r for r, _ in rag.calls} == set(EXTRACTION_ROLES)  # query/vlm untouched


async def test_apply_no_override_is_noop(tmp_path):
    rag = _FakeRag()
    await apply_workspace_provider_overrides(rag, "ws1", _store(tmp_path))
    assert rag.calls == []


async def test_apply_store_none_is_noop(tmp_path):
    rag = _FakeRag()
    await apply_workspace_provider_overrides(rag, "ws1", None)
    assert rag.calls == []


# ----------------------------- effective view -----------------------------
def test_slot_effective_labels_source_and_reasoning():
    role_cfg = {
        "binding": "openai",
        "model": "gemini-3.5-flash",
        "host": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "metadata": {"provider_options": {"reasoning_effort": "medium"}},
    }
    custom = slot_effective_view(role_cfg, active=True)
    assert custom["source"] == "custom"
    assert custom["reasoning_effort"] == "medium"
    assert custom["model"] == "gemini-3.5-flash"
    default = slot_effective_view(role_cfg, active=False)
    assert default["source"] == "system_default"
    assert set(default.keys()) == {"binding", "model", "host", "reasoning_effort", "source"}
