"""
Per-workspace BYO LLM / Vision-LLM provider routes (role-aware).

A workspace **owner** (self-service) — or an admin acting on behalf of a
workspace via ``X-Target-Workspace`` — can register, view, update and clear
their own OpenAI-compatible provider per role group:

- ``extraction`` → ``extract`` + ``keyword`` (ingestion; fast / non-thinking)
- ``query``      → ``query`` (answer generation; reasoning model preferred)
- ``vision``     → ``vlm`` (image description)

Each slot carries ``base_url`` + ``api_key`` + ``model`` and an optional
``reasoning_effort`` (none/low/medium/high). The system default is the fallback
when a slot has no override.

Changes take effect with **no server restart**: after persisting, the cached
LightRAG instance is invalidated, so the next request rebuilds it and the
build-time hook re-applies the (new) overrides.

Registered only when both multi-tenancy and ``ENABLE_WORKSPACE_PROVIDERS`` are
enabled. Fork-only and self-contained.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, SecretStr

from lightrag.api.dependencies import (
    UserInfo,
    get_current_user,
    get_current_workspace,
    get_workspace_manager,
)
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.workspace_providers import (
    EXTRACTION_ROLES,
    QUERY_ROLE,
    REASONING_EFFORTS,
    SLOT_ROLES,
    VISION_LLM_ROLE,
    ProviderSlot,
    WorkspaceProviderConfig,
    WorkspaceProviderError,
    WorkspaceProviderStore,
    slot_effective_view,
)

logger = logging.getLogger("lightrag.api.provider_routes")

# Representative role used to surface each slot's effective config.
_SLOT_REPRESENTATIVE_ROLE = {
    "extraction": "extract",
    "query": QUERY_ROLE,
    "vision": VISION_LLM_ROLE,
}


# ----------------------------- request/response models -----------------------------
class ProviderSlotInput(BaseModel):
    """One provider slot as submitted by the client (plaintext key over TLS)."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None  # omit to keep the existing stored key
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None  # none|low|medium|high; "" = default
    preset_id: Optional[str] = None


class UpdateProviderConfigRequest(BaseModel):
    """Update request. Omitted slots are left unchanged; use DELETE to clear."""

    extraction: Optional[ProviderSlotInput] = None
    query: Optional[ProviderSlotInput] = None
    vision: Optional[ProviderSlotInput] = None


def _get_store(request: Request) -> WorkspaceProviderStore:
    store = getattr(request.app.state, "workspace_provider_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Per-workspace providers are not enabled on this server",
        )
    return store


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base URL '{base_url}'; must be an http(s) URL",
        )


def _normalize_reasoning(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    if value == "":
        return None
    if value not in REASONING_EFFORTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"reasoning_effort must be one of {list(REASONING_EFFORTS)} or empty",
        )
    return value


def _merge_slot(existing: ProviderSlot, incoming: ProviderSlotInput) -> ProviderSlot:
    """Merge a submitted slot onto the existing stored slot.

    A missing ``api_key`` keeps the previously stored key (lets owners edit the
    base URL / model / reasoning without re-entering their secret). The merged
    result is validated to be a complete, active override.
    """
    base_url = incoming.base_url if incoming.base_url is not None else existing.base_url
    model = incoming.model if incoming.model is not None else existing.model
    preset_id = (
        incoming.preset_id if incoming.preset_id is not None else existing.preset_id
    )
    reasoning_effort = (
        _normalize_reasoning(incoming.reasoning_effort)
        if incoming.reasoning_effort is not None
        else existing.reasoning_effort
    )
    if incoming.api_key:
        api_key: Optional[SecretStr] = SecretStr(incoming.api_key)
    else:
        api_key = existing.api_key

    if not base_url:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "base_url is required")
    _validate_base_url(base_url)
    if not (api_key and api_key.get_secret_value()):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "api_key is required")
    if not model:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "model is required")
    return ProviderSlot(
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        preset_id=preset_id,
    )


async def _probe_provider(slot: ProviderSlot) -> None:
    """Best-effort connectivity/auth probe against the provider's /models."""
    url = slot.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {slot.api_key.get_secret_value()}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not reach provider at {slot.base_url}: {e}",
        )
    if resp.status_code in (401, 403):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider rejected the API key (authentication failed)",
        )


def _role_source(role_name: str, masked: dict) -> str:
    if role_name in EXTRACTION_ROLES:
        active = masked["extraction"]["active"]
    elif role_name == QUERY_ROLE:
        active = masked["query"]["active"]
    elif role_name == VISION_LLM_ROLE:
        active = masked["vision"]["active"]
    else:
        active = False
    return "custom" if active else "system_default"


def create_provider_routes(api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(prefix="/workspace", tags=["providers"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/provider-config", dependencies=[Depends(combined_auth)])
    async def get_provider_config(
        request: Request,
        workspace: str = Depends(get_current_workspace),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Masked config plus the **effective** provider in use for each slot."""
        store = _get_store(request)
        masked = store.get_masked(workspace)
        rag = await workspace_manager.get_instance(workspace)
        roles = rag.get_llm_role_config()
        for slot, role in _SLOT_REPRESENTATIVE_ROLE.items():
            masked[slot]["effective"] = slot_effective_view(
                roles[role], masked[slot]["active"]
            )
        return masked

    @router.get("/provider-config/effective", dependencies=[Depends(combined_auth)])
    async def get_effective_role_config(
        request: Request,
        workspace: str = Depends(get_current_workspace),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Live, credential-scrubbed config of every LLM role + its source."""
        store = _get_store(request)
        masked = store.get_masked(workspace)
        rag = await workspace_manager.get_instance(workspace)
        roles = rag.get_llm_role_config()
        out: dict[str, dict] = {}
        for role_name, cfg in roles.items():
            meta = cfg.get("metadata") or {}
            out[role_name] = {
                "binding": cfg.get("binding"),
                "model": cfg.get("model"),
                "host": cfg.get("host"),
                "reasoning_effort": (meta.get("provider_options") or {}).get(
                    "reasoning_effort"
                ),
                "is_cross_provider": cfg.get("is_cross_provider", False),
                "source": _role_source(role_name, masked),
            }
        return {"roles": out}

    @router.put("/provider-config", dependencies=[Depends(combined_auth)])
    async def update_provider_config(
        request: Request,
        body: UpdateProviderConfigRequest,
        test: bool = False,
        workspace: str = Depends(get_current_workspace),
        user: UserInfo = Depends(get_current_user),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Set / update provider credentials per slot, then apply live."""
        store = _get_store(request)
        if not store.has_secret():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server is missing WORKSPACE_PROVIDER_SECRET; cannot store "
                "provider credentials securely.",
            )
        provided = {
            name: getattr(body, name)
            for name in SLOT_ROLES
            if getattr(body, name) is not None
        }
        if not provided:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provide at least one of {list(SLOT_ROLES)}",
            )

        config = store.get(workspace) or WorkspaceProviderConfig()
        for name, incoming in provided.items():
            merged = _merge_slot(config.slot(name), incoming)
            setattr(config, name, merged)
            if test:
                await _probe_provider(merged)

        config.updated_at = datetime.now(timezone.utc).isoformat()
        config.updated_by = user.username
        try:
            store.set(workspace, config)
        except WorkspaceProviderError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
            )

        await workspace_manager.invalidate(workspace)
        logger.info(
            f"Provider config updated for workspace '{workspace}' by "
            f"'{user.username}' (slots: {list(provided)})"
        )
        return store.get_masked(workspace)

    @router.delete("/provider-config", dependencies=[Depends(combined_auth)])
    async def delete_provider_config(
        request: Request,
        slot: str = "all",
        workspace: str = Depends(get_current_workspace),
        user: UserInfo = Depends(get_current_user),
        workspace_manager=Depends(get_workspace_manager),
    ):
        """Clear a provider override (``slot`` = all | extraction | query | vision)."""
        if slot != "all" and slot not in SLOT_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"slot must be 'all' or one of {list(SLOT_ROLES)}",
            )
        store = _get_store(request)
        removed = store.delete(workspace, which=slot)
        await workspace_manager.invalidate(workspace)
        logger.info(
            f"Provider config slot '{slot}' cleared for workspace '{workspace}' "
            f"by '{user.username}' (removed={removed})"
        )
        return store.get_masked(workspace)

    return router
