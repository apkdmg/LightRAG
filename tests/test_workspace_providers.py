"""Offline tests for per-workspace BYO LLM / Vision-LLM provider credentials.

Covers the encrypted store (round-trip, at-rest secrecy, masking, fail-closed,
slot deletion, forgiving secret derivation) and the apply helper that maps the
two owner-facing slots onto LightRAG roles with correct fallback behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from lightrag.api.workspace_providers import (
    OVERRIDE_BINDING,
    TEXT_LLM_ROLES,
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


def _llm_cfg():
    return WorkspaceProviderConfig(
        llm=ProviderSlot(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr("sk-secret-abcd"),
            model="meta-llama/llama-3.3-70b-instruct",
            preset_id="openrouter",
        )
    )


# ----------------------------- store -----------------------------
def test_encrypt_decrypt_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _llm_cfg())

    got = store.get("ws1")
    assert got is not None
    assert got.llm.is_active()
    assert got.llm.api_key.get_secret_value() == "sk-secret-abcd"
    assert got.llm.base_url == "https://openrouter.ai/api/v1"
    assert got.llm.preset_id == "openrouter"
    assert not got.vision.is_active()


def test_api_key_not_stored_in_plaintext(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _llm_cfg())

    on_disk = (Path(tmp_path) / ".workspace_providers" / "ws1.json").read_text()
    assert "sk-secret-abcd" not in on_disk
    payload = json.loads(on_disk)
    assert payload["llm"]["api_key_enc"]  # encrypted blob present
    assert "api_key" not in payload["llm"]


def test_masked_view_hides_secret(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _llm_cfg())

    masked = store.get_masked("ws1")
    assert masked["llm"]["api_key_set"] is True
    assert masked["llm"]["api_key_preview"] == "...abcd"
    assert masked["llm"]["active"] is True
    assert "api_key" not in masked["llm"]
    # vision slot empty
    assert masked["vision"]["api_key_set"] is False
    assert masked["vision"]["active"] is False


def test_get_missing_returns_none(tmp_path):
    assert _store(tmp_path).get("nope") is None
    assert _store(tmp_path).get_masked("nope")["llm"]["active"] is False


def test_fail_closed_without_secret(tmp_path):
    store = WorkspaceProviderStore(tmp_path, get_secret=lambda: None)
    assert store.has_secret() is False
    with pytest.raises(WorkspaceProviderError):
        store.set("ws1", _llm_cfg())


def test_get_without_secret_falls_back_to_none(tmp_path):
    # Persist with a secret, then attempt to read with none configured: the
    # store must degrade to "no override" rather than raise.
    _store(tmp_path, secret="s1").set("ws1", _llm_cfg())
    no_secret = WorkspaceProviderStore(tmp_path, get_secret=lambda: None)
    assert no_secret.get("ws1") is None


def test_wrong_secret_does_not_decrypt(tmp_path):
    _store(tmp_path, secret="right").set("ws1", _llm_cfg())
    wrong = WorkspaceProviderStore(tmp_path, get_secret=lambda: "wrong")
    assert wrong.get("ws1") is None


def test_arbitrary_passphrase_is_accepted(tmp_path):
    # Non-Fernet passphrase is stretched via SHA-256; round-trip must still work.
    store = WorkspaceProviderStore(tmp_path, get_secret=lambda: "a short phrase")
    store.set("ws1", _llm_cfg())
    assert store.get("ws1").llm.api_key.get_secret_value() == "sk-secret-abcd"


def test_generate_secret_is_usable(tmp_path):
    secret = generate_secret()
    store = WorkspaceProviderStore(tmp_path, get_secret=lambda: secret)
    store.set("ws1", _llm_cfg())
    assert store.get("ws1").llm.is_active()


def test_delete_slot_then_all(tmp_path):
    store = _store(tmp_path)
    cfg = _llm_cfg()
    cfg.vision = ProviderSlot(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=SecretStr("sk-vision-9999"),
        model="gemini-2.5-flash",
        preset_id="gemini",
    )
    store.set("ws1", cfg)

    # Clear only the llm slot; vision survives.
    assert store.delete("ws1", "llm") is True
    after = store.get("ws1")
    assert after is not None
    assert not after.llm.is_active()
    assert after.vision.is_active()

    # Clearing the remaining slot removes the whole record.
    assert store.delete("ws1", "vision") is True
    assert store.get("ws1") is None


def test_delete_invalid_slot_raises(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _llm_cfg())
    with pytest.raises(ValueError):
        store.delete("ws1", "bogus")


# ----------------------------- apply helper -----------------------------
class _FakeRag:
    def __init__(self):
        self.calls = []

    async def aupdate_llm_role_config(self, role, **kwargs):
        self.calls.append((role, kwargs))


async def test_apply_text_llm_maps_to_text_roles(tmp_path):
    store = _store(tmp_path)
    store.set("ws1", _llm_cfg())
    rag = _FakeRag()

    await apply_workspace_provider_overrides(rag, "ws1", store)

    roles = {role for role, _ in rag.calls}
    assert roles == set(TEXT_LLM_ROLES)
    for _, kwargs in rag.calls:
        assert kwargs["binding"] == OVERRIDE_BINDING
        assert kwargs["host"] == "https://openrouter.ai/api/v1"
        assert kwargs["api_key"] == "sk-secret-abcd"
        assert kwargs["model"] == "meta-llama/llama-3.3-70b-instruct"


async def test_apply_vision_maps_to_vlm_role(tmp_path):
    store = _store(tmp_path)
    store.set(
        "ws1",
        WorkspaceProviderConfig(
            vision=ProviderSlot(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=SecretStr("sk-vision-9999"),
                model="gemini-2.5-flash",
            )
        ),
    )
    rag = _FakeRag()

    await apply_workspace_provider_overrides(rag, "ws1", store)

    assert [role for role, _ in rag.calls] == [VISION_LLM_ROLE]
    _, kwargs = rag.calls[0]
    assert kwargs["host"].endswith("/v1beta/openai/")
    assert kwargs["api_key"] == "sk-vision-9999"
    assert kwargs["model"] == "gemini-2.5-flash"


async def test_apply_no_override_is_noop(tmp_path):
    rag = _FakeRag()
    # No stored config → fallback (no role updates).
    await apply_workspace_provider_overrides(rag, "ws1", _store(tmp_path))
    assert rag.calls == []


async def test_apply_store_none_is_noop(tmp_path):
    rag = _FakeRag()
    await apply_workspace_provider_overrides(rag, "ws1", None)
    assert rag.calls == []


# ----------------------------- effective view -----------------------------
def test_slot_effective_labels_source():
    role_cfg = {
        "binding": "openai",
        "model": "gpt-4o-mini",
        "host": "https://api.openai.com/v1",
        "is_cross_provider": False,
    }
    custom = slot_effective_view(role_cfg, active=True)
    assert custom == {
        "binding": "openai",
        "model": "gpt-4o-mini",
        "host": "https://api.openai.com/v1",
        "source": "custom",
    }
    default = slot_effective_view(role_cfg, active=False)
    assert default["source"] == "system_default"
    # never leaks anything beyond the allow-listed keys
    assert set(default.keys()) == {"binding", "model", "host", "source"}
