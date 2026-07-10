"""
Per-workspace bring-your-own (BYO) LLM / Vision-LLM provider credentials.

A workspace owner may optionally supply their own OpenAI-compatible provider
(base URL + API key + model, plus an optional reasoning effort) for each role
group:

- ``extraction`` → applied to the ``extract`` and ``keyword`` roles
  (ingestion + query-keyword extraction; a fast, non-thinking model is ideal).
- ``query``      → applied to the ``query`` role (answer generation; a reasoning
  model is usually preferred).
- ``vision``     → applied to the ``vlm`` role (image/infographic description).

When a slot has no override, the system default applies — that fallback is
handled by the role-config resolution in ``lightrag_server.py``; this module
only owns *storage* of the overrides and *applying* them to a LightRAG instance.

Design notes (fork-only, self-contained to keep upstream merges trivial):

- API keys are encrypted at rest with ``cryptography.fernet.Fernet``. The key is
  derived from the ``WORKSPACE_PROVIDER_SECRET`` server secret.
- Persistence is one JSON file per workspace under
  ``working_dir/.workspace_providers/<workspace>.json``.
- The persistence detail sits behind :class:`WorkspaceProviderStore` so a
  DB-backed implementation can replace the file store later without touching
  any caller.
- Overrides are applied purely via the public ``rag.aupdate_llm_role_config``
  API (no core LightRAG code is modified).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger("lightrag.api.workspace_providers")

# Role groups the three owner-facing slots map onto. Imported by the routes.
EXTRACTION_ROLES: tuple[str, ...] = ("extract", "keyword")
QUERY_ROLE: str = "query"
VISION_LLM_ROLE: str = "vlm"

# Slot name -> the roles it drives.
SLOT_ROLES: dict[str, tuple[str, ...]] = {
    "extraction": EXTRACTION_ROLES,
    "query": (QUERY_ROLE,),
    "vision": (VISION_LLM_ROLE,),
}

# All BYO overrides are OpenAI-compatible by contract.
OVERRIDE_BINDING = "openai"

# Accepted reasoning-effort values (OpenAI-compatible; Gemini maps these to its
# thinking budget). None/"" means "leave the provider default".
REASONING_EFFORTS = ("none", "low", "medium", "high")

_SUBDIR = ".workspace_providers"
_ENV_SECRET = "WORKSPACE_PROVIDER_SECRET"


class WorkspaceProviderError(Exception):
    """Raised when an operation cannot proceed (e.g. missing encryption secret)."""


class ProviderSlot(BaseModel):
    """One OpenAI-compatible provider override for a role group."""

    base_url: Optional[str] = None
    api_key: Optional[SecretStr] = None
    model: Optional[str] = None
    # Optional thinking/reasoning control (none|low|medium|high). None = default.
    reasoning_effort: Optional[str] = None
    # Opaque UI hint (which preset the owner picked). Stored verbatim.
    preset_id: Optional[str] = None

    def is_active(self) -> bool:
        """An override only takes effect when both endpoint and key are present."""
        return bool(self.base_url and self.api_key and self.api_key.get_secret_value())


class WorkspaceProviderConfig(BaseModel):
    """Full per-workspace provider configuration (three role-group slots)."""

    extraction: ProviderSlot = Field(default_factory=ProviderSlot)
    query: ProviderSlot = Field(default_factory=ProviderSlot)
    vision: ProviderSlot = Field(default_factory=ProviderSlot)
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

    def slot(self, name: str) -> ProviderSlot:
        return getattr(self, name)

    def is_empty(self) -> bool:
        return not (
            self.extraction.is_active()
            or self.query.is_active()
            or self.vision.is_active()
        )


def generate_secret() -> str:
    """Return a fresh Fernet-compatible secret for WORKSPACE_PROVIDER_SECRET."""
    return Fernet.generate_key().decode("utf-8")


def _build_fernet(secret: str) -> Fernet:
    """Build a Fernet from an operator-supplied secret.

    Accepts either a canonical Fernet key (urlsafe-base64, 32 bytes) or any
    arbitrary passphrase, which is deterministically stretched to a 32-byte
    urlsafe-base64 key via SHA-256.
    """
    secret = secret.strip()
    try:
        return Fernet(secret.encode("utf-8"))
    except (ValueError, TypeError):
        derived = base64.urlsafe_b64encode(
            hashlib.sha256(secret.encode("utf-8")).digest()
        )
        return Fernet(derived)


def _mask_key(plain: str) -> str:
    """Return a non-reversible preview of a secret (last 4 chars only)."""
    if not plain:
        return ""
    tail = plain[-4:] if len(plain) >= 4 else plain
    return f"...{tail}"


class WorkspaceProviderStore:
    """File-backed, encrypted store for per-workspace provider overrides.

    The storage detail is intentionally hidden behind this class; a DB-backed
    implementation can be substituted later without changing callers.
    """

    def __init__(
        self,
        working_dir: str | os.PathLike[str],
        get_secret: Callable[[], Optional[str]] | None = None,
    ) -> None:
        self._base_dir = Path(working_dir) / _SUBDIR
        self._get_secret = get_secret or (lambda: os.getenv(_ENV_SECRET))

    # -- secret handling -------------------------------------------------
    def has_secret(self) -> bool:
        secret = self._get_secret()
        return bool(secret and secret.strip())

    def _fernet(self) -> Fernet:
        secret = self._get_secret()
        if not secret or not secret.strip():
            raise WorkspaceProviderError(
                f"{_ENV_SECRET} is not configured; cannot encrypt/decrypt "
                "workspace provider credentials. Set it to a stable secret "
                "(generate one via WorkspaceProviderStore.generate_secret())."
            )
        return _build_fernet(secret)

    # -- paths -----------------------------------------------------------
    def _path(self, workspace_id: str) -> Path:
        return self._base_dir / f"{workspace_id}.json"

    # -- read ------------------------------------------------------------
    def get(self, workspace_id: str) -> Optional[WorkspaceProviderConfig]:
        """Load and decrypt a workspace's provider config.

        Returns ``None`` when no override is stored. On a decryption failure
        (e.g. the secret was rotated/lost) this logs an error and returns
        ``None`` so the instance falls back to the system default rather than
        failing to start.

        Backward compatibility: an older config used a single ``llm`` slot
        (applied to extract/keyword/query). It is migrated on read by mapping
        ``llm`` onto both ``extraction`` and ``query``.
        """
        path = self._path(workspace_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read provider config for '{workspace_id}': {e}")
            return None

        try:
            fernet = self._fernet()
        except WorkspaceProviderError as e:
            logger.error(f"Cannot decrypt provider config for '{workspace_id}': {e}")
            return None

        legacy = raw.get("llm")  # pre role-aware format
        try:
            return WorkspaceProviderConfig(
                extraction=self._decode_slot(raw.get("extraction") or legacy, fernet),
                query=self._decode_slot(raw.get("query") or legacy, fernet),
                vision=self._decode_slot(raw.get("vision"), fernet),
                updated_at=raw.get("updated_at"),
                updated_by=raw.get("updated_by"),
            )
        except InvalidToken:
            logger.error(
                f"Provider config for '{workspace_id}' could not be decrypted "
                f"(wrong {_ENV_SECRET}?); ignoring override."
            )
            return None

    def _decode_slot(self, data: Optional[dict], fernet: Fernet) -> ProviderSlot:
        if not data:
            return ProviderSlot()
        api_key: Optional[SecretStr] = None
        enc = data.get("api_key_enc")
        if enc:
            api_key = SecretStr(fernet.decrypt(enc.encode("utf-8")).decode("utf-8"))
        return ProviderSlot(
            base_url=data.get("base_url"),
            api_key=api_key,
            model=data.get("model"),
            reasoning_effort=data.get("reasoning_effort"),
            preset_id=data.get("preset_id"),
        )

    # -- write -----------------------------------------------------------
    def set(self, workspace_id: str, config: WorkspaceProviderConfig) -> None:
        """Encrypt secrets and persist a workspace's provider config."""
        fernet = self._fernet()  # fail closed if no secret
        payload: dict[str, Any] = {
            "extraction": self._encode_slot(config.extraction, fernet),
            "query": self._encode_slot(config.query, fernet),
            "vision": self._encode_slot(config.vision, fernet),
            "updated_at": config.updated_at or datetime.now(timezone.utc).isoformat(),
            "updated_by": config.updated_by,
        }
        path = self._path(workspace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        logger.info(f"Stored provider config for workspace '{workspace_id}'")

    def _encode_slot(self, slot: ProviderSlot, fernet: Fernet) -> dict[str, Any]:
        out: dict[str, Any] = {
            "base_url": slot.base_url,
            "model": slot.model,
            "reasoning_effort": slot.reasoning_effort,
            "preset_id": slot.preset_id,
        }
        if slot.api_key and slot.api_key.get_secret_value():
            out["api_key_enc"] = fernet.encrypt(
                slot.api_key.get_secret_value().encode("utf-8")
            ).decode("utf-8")
        return out

    # -- delete ----------------------------------------------------------
    def delete(self, workspace_id: str, which: str = "all") -> bool:
        """Remove an override.

        ``which`` may be ``"all"`` (delete the whole record) or a slot name
        (``"extraction"`` / ``"query"`` / ``"vision"``) to clear one slot,
        keeping the others. Returns True if anything was removed.
        """
        path = self._path(workspace_id)
        if not path.exists():
            return False
        if which == "all":
            try:
                path.unlink()
                logger.info(f"Deleted provider config for workspace '{workspace_id}'")
                return True
            except OSError as e:
                logger.error(
                    f"Failed to delete provider config for '{workspace_id}': {e}"
                )
                return False

        if which not in SLOT_ROLES:
            raise ValueError(
                f"Invalid slot '{which}'; expected 'all' or one of {list(SLOT_ROLES)}"
            )

        config = self.get(workspace_id)
        if config is None:
            return False
        setattr(config, which, ProviderSlot())
        if config.is_empty():
            return self.delete(workspace_id, "all")
        self.set(workspace_id, config)
        return True

    # -- masked view (safe for API responses) ----------------------------
    def get_masked(self, workspace_id: str) -> dict[str, Any]:
        """Return a secret-free view of the stored config for API responses."""
        config = self.get(workspace_id) or WorkspaceProviderConfig()
        return {
            "extraction": _masked_slot(config.extraction),
            "query": _masked_slot(config.query),
            "vision": _masked_slot(config.vision),
            "updated_at": config.updated_at,
            "updated_by": config.updated_by,
        }


async def _apply_slot(rag: Any, roles: tuple[str, ...], slot: ProviderSlot) -> None:
    """Apply one slot's override to its role(s) via the public role API."""
    api_key = slot.api_key.get_secret_value()
    for role in roles:
        provider_options = None
        if slot.reasoning_effort:
            # Merge onto the role's existing (scrubbed) provider options so we
            # don't drop temperature/etc.; get_llm_role_config never leaks keys.
            current = rag.get_llm_role_config(role)
            provider_options = dict(
                (current.get("metadata") or {}).get("provider_options") or {}
            )
            provider_options["reasoning_effort"] = slot.reasoning_effort
        await rag.aupdate_llm_role_config(
            role,
            binding=OVERRIDE_BINDING,
            host=slot.base_url,
            api_key=api_key,
            model=slot.model,
            provider_options=provider_options,
        )


async def apply_workspace_provider_overrides(
    rag: Any,
    workspace_id: str,
    store: Optional["WorkspaceProviderStore"],
) -> None:
    """Apply a workspace's stored provider overrides to its LightRAG roles.

    ``extraction`` → ``extract``/``keyword``; ``query`` → ``query``;
    ``vision`` → ``vlm``. Roles without an override keep their env/default
    configuration (the system-default fallback). Safe to call on every instance
    build; a no-op when nothing is stored.
    """
    if store is None:
        return
    config = store.get(workspace_id)
    if config is None or config.is_empty():
        return

    if config.extraction.is_active():
        await _apply_slot(rag, EXTRACTION_ROLES, config.extraction)
        logger.info(f"Applied BYO extraction provider for workspace '{workspace_id}'")
    if config.query.is_active():
        await _apply_slot(rag, (QUERY_ROLE,), config.query)
        logger.info(f"Applied BYO query provider for workspace '{workspace_id}'")
    if config.vision.is_active():
        await _apply_slot(rag, (VISION_LLM_ROLE,), config.vision)
        logger.info(f"Applied BYO vision provider for workspace '{workspace_id}'")


def slot_effective_view(role_cfg: dict[str, Any], active: bool) -> dict[str, Any]:
    """Build the effective-provider view for one owner-facing slot.

    ``role_cfg`` is a credential-scrubbed entry from
    ``rag.get_llm_role_config(role)``. ``source`` tells the owner whether the
    value in effect comes from their own override (``custom``) or the system
    default fallback (``system_default``).
    """
    meta = role_cfg.get("metadata") or {}
    options = meta.get("provider_options") or {}
    return {
        "binding": role_cfg.get("binding"),
        "model": role_cfg.get("model"),
        "host": role_cfg.get("host"),
        "reasoning_effort": options.get("reasoning_effort"),
        "source": "custom" if active else "system_default",
    }


def _masked_slot(slot: ProviderSlot) -> dict[str, Any]:
    plain = slot.api_key.get_secret_value() if slot.api_key else ""
    return {
        "base_url": slot.base_url,
        "model": slot.model,
        "reasoning_effort": slot.reasoning_effort,
        "preset_id": slot.preset_id,
        "api_key_set": bool(plain),
        "api_key_preview": _mask_key(plain),
        "active": slot.is_active(),
    }
