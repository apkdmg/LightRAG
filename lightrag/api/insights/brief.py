"""The procurement brief, as CSV and as a small dependency-free PDF."""

from __future__ import annotations

import csv
import io
from typing import Any

PURPOSE_NAMES = {
    "extract": "Extraction",
    "vlm": "Vision",
    "keyword": "Keyword",
    "query": "Query",
}


def _money(value: Any, rate: Any = None) -> str:
    if value is None:
        return "-"
    text = f"${float(value):,.2f}"
    return f"{text} (MYR {float(value) * float(rate):,.2f})" if rate else text


def _tokens(value: Any) -> str:
    return f"{float(value or 0) / 1e6:,.1f}M"


def brief_lines(forecast: dict[str, Any], actuals: list[dict[str, Any]]) -> list[str]:
    rate = forecast.get("myr_rate")
    budget = forecast.get("budget") or {}
    next90 = forecast.get("next90") or {}
    lines = [
        "Token usage: procurement brief",
        "",
        "Last 90 days, by purpose",
    ]
    lines += [
        f"  {PURPOSE_NAMES.get(r['purpose'], r['purpose']):<12}"
        f"{_tokens(r['tokens']):>10}  {_money(r['cost'], rate)}"
        f"  of which BYO {_money(r['byo_cost'], rate)}"
        for r in actuals
    ]
    lines += ["", "Next 90 days, by purpose"]
    lines += [
        f"  {PURPOSE_NAMES.get(r['id'], r['id']):<12}"
        f"{_tokens(r['tokens']):>10}  range {_tokens(r['low'])} to {_tokens(r['high'])}"
        f"  {_money(r['cost'], rate)}  BYO {r['byo_share'] * 100:.0f}%"
        for r in forecast.get("rows") or []
    ]
    lines += [
        "",
        f"Expected cost, all keys: {_money(next90.get('cost'), rate)}",
        f"  of which on BYO keys:  {_money(next90.get('byo_cost'), rate)}",
        f"High end of the range:   {_money(next90.get('high_cost'), rate)}",
    ]
    if budget.get("set"):
        lines += [
            "",
            f"Budget left:             {_money(budget.get('left'), rate)}",
            f"Runs out, BYO included:  {budget.get('runs_out') or 'after the forecast'}",
            f"Runs out, without BYO:   {budget.get('runs_out_without_byo') or 'after the forecast'}",
            f"Suggested purchase:      {_money(budget.get('suggested_purchase'), rate)}",
        ]
    lines += [
        "",
        "Assumptions",
        "  Documents drive Extraction and Vision. Questions drive Keyword and Query.",
        "  Totals include usage on workspace (BYO) keys, which the owners pay for.",
        "  Costs use the admin price table, and provider-reported costs where given.",
    ]
    if forecast.get("learning", {}).get("forecast_low_confidence"):
        lines.append("  Low confidence: there are fewer than 8 weeks of history.")
    return lines


def brief_csv(forecast: dict[str, Any], actuals: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["section", "purpose", "tokens", "low", "high", "cost_usd", "byo_cost_usd"]
    )
    for r in actuals:
        writer.writerow(
            [
                "last_90_days",
                r["purpose"],
                r["tokens"],
                "",
                "",
                r["cost"],
                r["byo_cost"],
            ]
        )
    for r in forecast.get("rows") or []:
        writer.writerow(
            [
                "next_90_days",
                r["id"],
                round(r["tokens"]),
                round(r["low"]),
                round(r["high"]),
                r["cost"],
                round(r["cost"] * r["byo_share"], 4),
            ]
        )
    budget = forecast.get("budget") or {}
    if budget.get("set"):
        writer.writerow(["budget_left", "", "", "", "", budget.get("left"), ""])
        writer.writerow(
            ["suggested_purchase", "", "", "", "", budget.get("suggested_purchase"), ""]
        )
    return out.getvalue()


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(lines: list[str], title_lines: int = 1) -> bytes:
    """A plain one-font PDF: enough for a brief, with no new dependency."""
    per_page = 58
    pages = [lines[i : i + per_page] for i in range(0, len(lines), per_page)] or [[]]
    objects: list[bytes] = []
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    font_id = 3 + len(pages) * 2
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1")
    )
    for index, page in enumerate(pages):
        stream = ["BT", "/F1 9.5 Tf", "12 TL", "48 790 Td"]
        for number, line in enumerate(page):
            if index == 0 and number < title_lines:
                stream += ["/F1 15 Tf", f"({_pdf_escape(line)}) Tj", "T*", "/F1 9.5 Tf"]
            else:
                stream += [f"({_pdf_escape(line)}) Tj", "T*"]
        stream.append("ET")
        content = "\n".join(stream).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {4 + index * 2} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode("latin-1")
        )
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode("latin-1")
            + content
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
