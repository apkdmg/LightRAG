"""Admin-only API behind the Insights tab. Every route needs the admin role.

The page never receives a full document name, a prompt or an answer. Settings
here are the dashboard's own (prices, budget, invoices); changing the system's
models or keys stays on the Providers page.
"""

from __future__ import annotations

import asyncio
import csv
import io
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from lightrag.api.dependencies import UserInfo, require_admin
from lightrag.api.insights.backfill import run_backfill
from lightrag.api.insights.brief import brief_csv, brief_lines, text_pdf
from lightrag.api.insights.service import (
    WORKSPACE_OPTION_LIMIT,
    WORKSPACE_PAGE_SIZE,
    InsightsService,
)
from lightrag.usage import PURPOSES

router = APIRouter(
    prefix="/admin/insights", tags=["insights"], dependencies=[Depends(require_admin)]
)

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_ANOMALY_STATES = ("open", "expected", "acknowledged")


class ModelPrice(BaseModel):
    prompt: float = Field(ge=0)
    completion: float = Field(ge=0)


class BudgetSetting(BaseModel):
    amount: Optional[float] = Field(default=None, ge=0)
    start: Optional[date] = None


class SettingsPatch(BaseModel):
    prices: Optional[Dict[str, ModelPrice]] = None
    budget: Optional[BudgetSetting] = None
    rate_limit_tpm: Optional[int] = Field(default=None, ge=0)
    myr_rate: Optional[float] = Field(default=None, ge=0)
    invoices: Optional[Dict[str, float]] = None


class AnomalyState(BaseModel):
    state: str


def _service(request: Request) -> InsightsService:
    service = getattr(request.app.state, "insights", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Token Insights needs PostgreSQL storage (LIGHTRAG_KV_STORAGE=PGKVStorage)",
        )
    return service


def _purpose(value: Optional[str]) -> Optional[str]:
    if value and value not in PURPOSES:
        raise HTTPException(status_code=400, detail=f"Unknown purpose: {value}")
    return value or None


def _day(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date: {value}")


def _range(
    service: InsightsService, days: int, start: Optional[str], end: Optional[str]
) -> tuple[int, Optional[date]]:
    """Resolve a preset length or a custom start and end date."""
    first, last = _day(start), _day(end)
    if first and last:
        if last < first or last > service.today():
            raise HTTPException(status_code=400, detail="Invalid date range")
        return min((last - first).days + 1, 730), last
    return days, None


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "insights", None)
    if service is None:
        return {"available": False}
    return {"available": True, "learning": await service.learning()}


@router.get("/overview")
async def overview(
    request: Request,
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
    purpose: Optional[str] = None,
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    service = _service(request)
    length, last = _range(service, days, start, end)
    return await service.overview(
        length, workspace or None, _purpose(purpose), end=last
    )


@router.get("/purposes/{purpose}")
async def purpose_page(
    request: Request,
    purpose: str,
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    service = _service(request)
    length, last = _range(service, days, start, end)
    return await service.purpose_page(
        _purpose(purpose), length, workspace or None, end=last
    )


@router.get("/workspaces")
async def workspaces(
    request: Request,
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
    purpose: Optional[str] = None,
    q: str = Query("", max_length=255),
    show: str = Query("all", pattern="^(all|byo|look)$"),
    sort: str = Query("cost", pattern="^(cost|tokens|change|documents|questions)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(WORKSPACE_PAGE_SIZE, ge=1, le=200),
) -> dict[str, Any]:
    """One page of the table. Search, filter and sort cover every workspace."""
    service = _service(request)
    length, last = _range(service, days, start, end)
    return await service.workspaces(
        length,
        _purpose(purpose),
        end=last,
        query=q,
        show=show,
        sort=sort,
        page=page,
        page_size=page_size,
    )


@router.get("/workspace-options")
async def workspace_options(
    request: Request,
    q: str = Query("", max_length=255),
    limit: int = Query(WORKSPACE_OPTION_LIMIT, ge=1, le=100),
) -> dict[str, Any]:
    """Names for the workspace picker, without the cost of the full table."""
    return await _service(request).workspace_options(q, limit)


@router.get("/workspaces/{workspace}")
async def workspace_page(
    request: Request,
    workspace: str,
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, Any]:
    service = _service(request)
    length, last = _range(service, days, start, end)
    return await service.workspace_page(workspace, length, end=last)


def _filters(
    purpose: Optional[str],
    workspace: Optional[str],
    operation: Optional[str],
    model: Optional[str],
    source: Optional[str],
    billing: Optional[str],
) -> dict[str, str]:
    values = {
        "purpose": _purpose(purpose),
        "workspace": workspace,
        "operation": operation,
        "model": model,
        "source": source,
        "billing": billing,
    }
    return {k: v for k, v in values.items() if v}


@router.get("/breakdown")
async def breakdown(
    request: Request,
    group: str = "purpose",
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
    day: Optional[str] = None,
    purpose: Optional[str] = None,
    workspace: Optional[str] = None,
    operation: Optional[str] = None,
    model: Optional[str] = None,
    source: Optional[str] = None,
    billing: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """The largest rows, and how many there are. The CSV export has them all."""
    service = _service(request)
    length, last = _range(service, days, start, end)
    try:
        result = await service.breakdown(
            group,
            length,
            _filters(purpose, workspace, operation, model, source, billing),
            day=day,
            end=last,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {**result, "rows": result["rows"][:limit], "total": len(result["rows"])}


@router.get("/breakdown.csv")
async def breakdown_csv(
    request: Request,
    group: str = "purpose",
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
    day: Optional[str] = None,
    purpose: Optional[str] = None,
    workspace: Optional[str] = None,
    operation: Optional[str] = None,
    model: Optional[str] = None,
    source: Optional[str] = None,
    billing: Optional[str] = None,
) -> Response:
    """The same rows as the table. Document names are masked here too."""
    service = _service(request)
    length, last = _range(service, days, start, end)
    try:
        result = await service.breakdown(
            group,
            length,
            _filters(purpose, workspace, operation, model, source, billing),
            day=day,
            end=last,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([group, "tokens", "calls", "cache_hits", "cost_usd"])
    for row in result["rows"]:
        display = row.get("display") or {}
        name = display.get("label") or row["key"]
        if display.get("code"):
            name = f"{name} #{display['code']}".strip()
        writer.writerow(
            [name, row["tokens"], row["calls"], row["cache_hits"], row["cost"]]
        )
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="usage-by-{group}.csv"'},
    )


@router.get("/calls")
async def calls(
    request: Request,
    days: int = Query(30, ge=1, le=90),
    day: Optional[str] = None,
    purpose: Optional[str] = None,
    workspace: Optional[str] = None,
    operation: Optional[str] = None,
    model: Optional[str] = None,
    source: Optional[str] = None,
    billing: Optional[str] = None,
    doc_id: Optional[str] = None,
    client: Optional[str] = None,
    request_id: Optional[str] = Query(None, max_length=64),
) -> dict[str, Any]:
    service = _service(request)
    return {
        "rows": await service.calls(
            days,
            _filters(purpose, workspace, operation, model, source, billing),
            doc_id=doc_id,
            client=client,
            day=day,
            request_id=request_id or None,
        )
    }


@router.get("/days/{day}")
async def explain_day(
    request: Request, day: str, workspace: Optional[str] = None
) -> dict[str, Any]:
    return await _service(request).explain_day(_day(day), workspace or None)


@router.get("/anomalies")
async def anomalies(
    request: Request, workspace: Optional[str] = None, purpose: Optional[str] = None
) -> dict[str, Any]:
    service = _service(request)
    result = await service.anomalies(workspace or None, _purpose(purpose))
    result["calendar"] = await service.calendar(workspace or None, purpose or None)
    return result


@router.post("/anomalies/{anomaly_id:path}")
async def set_anomaly_state(
    request: Request,
    anomaly_id: str,
    body: AnomalyState,
    admin: UserInfo = Depends(require_admin),
) -> dict[str, str]:
    if body.state not in _ANOMALY_STATES:
        raise HTTPException(status_code=400, detail="Unknown state")
    await _service(request).set_anomaly_state(anomaly_id, body.state, admin.username)
    return {"status": "ok"}


@router.get("/work")
async def work(
    request: Request,
    side: str = Query("ingest", pattern="^(ingest|questions)$"),
    days: int = Query(30, ge=1, le=730),
    start: Optional[str] = None,
    end: Optional[str] = None,
    purpose: Optional[str] = None,
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    service = _service(request)
    length, last = _range(service, days, start, end)
    return await service.work(
        side, length, workspace or None, _purpose(purpose), end=last
    )


@router.get("/forecast")
async def forecast(
    request: Request,
    documents: float = Query(0, ge=-100, le=500),
    questions: float = Query(0, ge=-100, le=500),
    bulk: int = Query(0, ge=0),
    swap: Optional[str] = None,
    purpose: Optional[str] = None,
) -> dict[str, Any]:
    return await _service(request).forecast(
        documents / 100, questions / 100, bulk, _purpose(swap), _purpose(purpose)
    )


async def _brief_data(
    service: InsightsService,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    today = service.today()
    actuals = await service._rows(today - timedelta(days=89), today, ["purpose"])
    return await service.forecast(), actuals


@router.get("/brief.csv")
async def procurement_csv(request: Request) -> Response:
    forecast_data, actuals = await _brief_data(_service(request))
    return Response(
        brief_csv(forecast_data, actuals),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="procurement-brief.csv"'},
    )


@router.get("/brief.pdf")
async def procurement_pdf(request: Request) -> Response:
    forecast_data, actuals = await _brief_data(_service(request))
    return Response(
        text_pdf(brief_lines(forecast_data, actuals)),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="procurement-brief.pdf"'},
    )


@router.get("/live")
async def live(request: Request) -> dict[str, Any]:
    return await _service(request).live()


@router.get("/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    return await _service(request).settings()


@router.put("/settings")
async def put_settings(request: Request, body: SettingsPatch) -> dict[str, Any]:
    patch = body.model_dump(exclude_unset=True, mode="json")
    for month in patch.get("invoices") or {}:
        if not _MONTH.match(month):
            raise HTTPException(status_code=400, detail=f"Invalid month: {month}")
    return await _service(request).save_settings(patch)


@router.post("/backfill")
async def backfill(request: Request) -> dict[str, Any]:
    """Estimate usage from before recording began. Runs once, in the background."""
    service = _service(request)
    current = await service.store.kv_get("backfill")
    if current and current.get("status") in ("running", "done"):
        return current
    tokenizer = getattr(service.default_rag, "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="No tokenizer available")
    task = asyncio.create_task(run_backfill(service.store, tokenizer))
    request.app.state.insights_tasks.add(task)
    task.add_done_callback(request.app.state.insights_tasks.discard)
    return {"status": "running", "at": datetime.now().isoformat()}
