"""Starts and stops usage recording with the API server."""

from __future__ import annotations

from typing import Any

from lightrag.provider_rejection import (
    ProviderRejectionEvent,
    add_provider_rejection_listener,
    remove_provider_rejection_listener,
)
from lightrag.usage import emit_event
from lightrag.utils import logger

from .recorder import UsageRecorder
from .scheduler import InsightsScheduler, note_startup
from .service import InsightsService
from .store import create_usage_store


async def _record_rejection(event: ProviderRejectionEvent) -> None:
    emit_event(
        "provider_rejection",
        workspace=event.workspace,
        purpose=event.role,
        status=event.rejection.status_code,
        byo=event.byo_key,
        paused=event.paused,
    )


async def start_insights(app: Any, rag: Any, args: Any) -> None:
    """Begin recording. Version 1 needs Postgres; otherwise the page says so."""
    app.state.insights = None
    app.state.insights_tasks = set()
    store = create_usage_store(args)
    if store is None:
        logger.info("Token Insights is off: it needs LIGHTRAG_KV_STORAGE=PGKVStorage")
        return
    try:
        await store.initialize()
    except Exception as e:
        # Usage recording must never stop the server from starting.
        logger.error(f"Token Insights could not start: {e}")
        return
    service = InsightsService(store, default_rag=rag, app_state=app.state)
    recorder = UsageRecorder(store)
    recorder.start()
    scheduler = InsightsScheduler(service)
    scheduler.start()
    add_provider_rejection_listener(_record_rejection)
    app.state.insights = service
    app.state.insights_recorder = recorder
    app.state.insights_scheduler = scheduler
    try:
        from lightrag.api import __api_version__

        await note_startup(service, str(__api_version__))
    except Exception as e:
        logger.warning(f"Could not note startup for Token Insights: {e}")
    logger.info("Token Insights: recording LLM usage")


async def stop_insights(app: Any) -> None:
    service = getattr(app.state, "insights", None)
    if service is None:
        return
    remove_provider_rejection_listener(_record_rejection)
    await app.state.insights_scheduler.stop()
    await app.state.insights_recorder.stop()
    await service.store.finalize()
    app.state.insights = None
