"""PDF action report generation for Scenario Lab."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from textwrap import wrap
from typing import Any

try:
    from reportlab.platypus import Flowable
except ImportError:  # pragma: no cover - exercised only when reportlab is unavailable
    class Flowable:  # type: ignore[no-redef]
        pass


DISCLAIMER = "Decision support only. This report is not an official emergency warning or evacuation order."


class ActionReportService:
    def build_pdf(
        self,
        *,
        scenario: dict[str, Any],
        monitoring: dict[str, Any],
        feedback: dict[str, Any],
        drift: dict[str, Any],
        citations: list[dict[str, Any]] | None = None,
    ) -> bytes:
        try:
            return _build_reportlab_pdf(
                scenario=scenario,
                monitoring=monitoring,
                feedback=feedback,
                drift=drift,
                citations=citations or [],
            )
        except ImportError:
            return _build_basic_pdf(
                _report_lines(
                    scenario=scenario,
                    monitoring=monitoring,
                    feedback=feedback,
                    drift=drift,
                    citations=citations or [],
                )
            )


def _report_lines(
    *,
    scenario: dict[str, Any],
    monitoring: dict[str, Any],
    feedback: dict[str, Any],
    drift: dict[str, Any],
    citations: list[dict[str, Any]],
) -> list[str]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines = [
        "FloodLens Action Brief",
        f"Generated: {generated_at}",
        f"Location: {scenario.get('place_name')} / {scenario.get('district')}",
        f"Record: {scenario.get('record_id')}",
        f"Coordinates: {scenario.get('latitude')}, {scenario.get('longitude')}",
        f"Model version: {scenario.get('model_version')}",
        "",
        "Decision Summary",
        f"Baseline risk: {scenario.get('baseline_risk_score')} ({scenario.get('baseline_risk_level')})",
        f"Scenario risk: {scenario.get('scenario_risk_score')} ({scenario.get('scenario_risk_level')})",
        f"Score delta: {scenario.get('score_delta')}",
        f"Risk level delta: {scenario.get('risk_level_delta')}",
        f"Operational priority: {scenario.get('operational_priority')}",
        "",
        "Risk Reasons And Assumptions",
        f"Top drivers: {', '.join(scenario.get('risk_drivers') or []) or '-'}",
        f"Changed fields: {', '.join(scenario.get('changed_fields') or []) or '-'}",
        f"Recommended action: {scenario.get('recommended_action')}",
        "",
        "Evidence Snapshot",
        f"Total predictions: {monitoring.get('total_predictions')}",
        f"Latest prediction: {monitoring.get('latest_prediction_at')}",
        f"Feedback events: {feedback.get('total_feedback')}",
        f"Feedback disagreement rate: {feedback.get('disagreement_rate')}",
        f"Drift status: {drift.get('status')}",
        f"Drift recommendation: {drift.get('recommendation')}",
    ]
    if citations:
        lines.extend(["", "Document Evidence"])
        for citation in citations[:5]:
            title = citation.get("title") or citation.get("document_title") or "Document"
            page = citation.get("page") or citation.get("page_number") or "-"
            lines.append(f"- {title}, page {page}: {citation.get('snippet') or citation.get('text') or ''}")
    lines.extend(["", DISCLAIMER])
    return lines


def _build_reportlab_pdf(
    *,
    scenario: dict[str, Any],
    monitoring: dict[str, Any],
    feedback: dict[str, Any],
    drift: dict[str, Any],
    citations: list[dict[str, Any]],
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Flowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="FloodLens Action Brief",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#475569"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyClean",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
        )
    )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    place = _clean(scenario.get("place_name"), "Selected place")
    district = _clean(scenario.get("district"), "Unknown district")
    latitude = _as_float(scenario.get("latitude"))
    longitude = _as_float(scenario.get("longitude"))
    story: list[Any] = [
        Paragraph("FloodLens Action Brief", styles["ReportTitle"]),
        Paragraph(
            _escape_xml(f"{place}, {district} - generated {generated_at}"),
            styles["Small"],
        ),
        Spacer(1, 8),
        _summary_table(scenario),
        Spacer(1, 10),
        KeepTogether(
            [
                Paragraph("Location Map", styles["SectionTitle"]),
                LocationMapFlowable(latitude=latitude, longitude=longitude, width=172 * mm, height=72 * mm),
            ]
        ),
        Paragraph("Decision Summary", styles["SectionTitle"]),
        _key_value_table(
            [
                ("Baseline risk", _risk_text(scenario.get("baseline_risk_score"), scenario.get("baseline_risk_level"))),
                ("Scenario risk", _risk_text(scenario.get("scenario_risk_score"), scenario.get("scenario_risk_level"))),
                ("Risk change", _signed(scenario.get("score_delta"))),
                ("Priority", _clean(scenario.get("operational_priority"), "Not assigned")),
                ("Risk movement", _clean(scenario.get("risk_level_delta"), "No level change")),
                ("Who should review", "District response desk, field operations lead, local authority liaison"),
            ]
        ),
        Paragraph("Recommended Action", styles["SectionTitle"]),
        Paragraph(_escape_xml(_clean(scenario.get("recommended_action"), "Review the location and prepare a local response checklist.")), styles["BodyClean"]),
        Paragraph("Risk Reasons", styles["SectionTitle"]),
        _bullet_table(scenario.get("risk_drivers") or ["No major driver listed"]),
        Paragraph("Scenario Assumptions Changed", styles["SectionTitle"]),
        _bullet_table(_friendly_changed_fields(scenario.get("changed_fields") or [])),
        Paragraph("Operational Evidence Snapshot", styles["SectionTitle"]),
        _key_value_table(
            [
                ("Predictions logged", monitoring.get("total_predictions")),
                ("Latest activity", monitoring.get("latest_prediction_at") or "No activity logged"),
                ("Feedback received", feedback.get("total_feedback")),
                ("Disagreement rate", feedback.get("disagreement_rate")),
                ("Monitoring status", drift.get("status")),
                ("Monitoring recommendation", drift.get("recommendation")),
            ]
        ),
    ]

    if citations:
        story.append(Paragraph("Document Evidence", styles["SectionTitle"]))
        citation_rows = []
        for citation in citations[:5]:
            title = citation.get("title") or citation.get("document_title") or "Document"
            page = citation.get("page") or citation.get("page_number") or "-"
            snippet = citation.get("snippet") or citation.get("text") or ""
            citation_rows.append([Paragraph(_escape_xml(f"{title}, page {page}"), styles["Small"]), Paragraph(_escape_xml(snippet), styles["Small"])])
        story.append(_styled_table(citation_rows, col_widths=[46 * mm, 126 * mm]))

    story.extend(
        [
            Spacer(1, 10),
            Paragraph("Decision Limit", styles["SectionTitle"]),
            Paragraph(_escape_xml(DISCLAIMER), styles["Small"]),
        ]
    )

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def _summary_table(scenario: dict[str, Any]) -> Any:
    from reportlab.lib.units import mm

    rows = [
        ["Place", _clean(scenario.get("place_name"), "Selected place"), "District", _clean(scenario.get("district"), "-")],
        ["Scenario risk", _risk_text(scenario.get("scenario_risk_score"), scenario.get("scenario_risk_level")), "Priority", _clean(scenario.get("operational_priority"), "-")],
    ]
    return _styled_table(rows, col_widths=[30 * mm, 58 * mm, 30 * mm, 54 * mm], header=False)


def _key_value_table(rows: list[tuple[str, Any]]) -> Any:
    from reportlab.lib.units import mm

    return _styled_table([[label, _clean(value, "-")] for label, value in rows], col_widths=[48 * mm, 124 * mm], header=False)


def _bullet_table(items: list[Any]) -> Any:
    from reportlab.lib.units import mm

    rows = [["-", _clean(item, "-")] for item in items]
    return _styled_table(rows, col_widths=[8 * mm, 164 * mm], header=False)


def _styled_table(rows: list[list[Any]], *, col_widths: list[Any], header: bool = False) -> Any:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from reportlab.platypus import Table, TableStyle

    cell_style = ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1f2937"),
    )
    header_style = ParagraphStyle(
        name="TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"),
    )

    prepared_rows = []
    for row_index, row in enumerate(rows):
        prepared_row = []
        for cell in row:
            if hasattr(cell, "wrap"):
                prepared_row.append(cell)
            else:
                prepared_row.append(
                    Paragraph(_escape_xml(_clean(cell, "-")), header_style if header and row_index == 0 else cell_style)
                )
        prepared_rows.append(prepared_row)

    table = Table(prepared_rows, colWidths=col_widths, hAlign="LEFT")
    commands = [
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(commands))
    return table


class LocationMapFlowable(Flowable):
    """Small static Sri Lanka context map for the generated action brief."""

    boundary = [
        (79.55, 5.85),
        (80.05, 5.78),
        (80.75, 5.88),
        (81.45, 6.15),
        (81.88, 6.55),
        (82.05, 7.25),
        (81.92, 8.05),
        (81.55, 8.75),
        (80.95, 9.55),
        (80.25, 9.90),
        (79.65, 9.80),
        (79.35, 9.25),
        (79.55, 8.35),
        (79.65, 7.45),
        (79.45, 6.75),
    ]

    def __init__(self, *, latitude: float | None, longitude: float | None, width: float, height: float) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.width = width
        self.height = height

    def draw(self) -> None:
        from reportlab.lib import colors

        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#eef6f8"))
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=1)

        min_lon, max_lon = 79.2, 82.2
        min_lat, max_lat = 5.7, 10.0

        def project(lon: float, lat: float) -> tuple[float, float]:
            x = 16 + ((lon - min_lon) / (max_lon - min_lon)) * (self.width - 32)
            y = 10 + ((lat - min_lat) / (max_lat - min_lat)) * (self.height - 20)
            return x, y

        points = [project(lon, lat) for lon, lat in self.boundary]
        path = canvas.beginPath()
        path.moveTo(*points[0])
        for x, y in points[1:]:
            path.lineTo(x, y)
        path.close()
        canvas.setFillColor(colors.HexColor("#dbeafe"))
        canvas.setStrokeColor(colors.HexColor("#0891b2"))
        canvas.setLineWidth(1.2)
        canvas.drawPath(path, fill=1, stroke=1)

        if self.latitude is not None and self.longitude is not None:
            x, y = project(self.longitude, self.latitude)
            canvas.setFillColor(colors.HexColor("#dc2626"))
            canvas.circle(x, y, 4.2, fill=1, stroke=0)
            canvas.setStrokeColor(colors.HexColor("#ffffff"))
            canvas.setLineWidth(1)
            canvas.circle(x, y, 5.4, fill=0, stroke=1)
            canvas.setFillColor(colors.HexColor("#0f172a"))
            canvas.setFont("Helvetica-Bold", 7)
            canvas.drawString(min(x + 8, self.width - 72), min(y + 2, self.height - 12), "assessment point")

        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(10, 8, "Static context map - marker is approximate")
        canvas.restoreState()


def _footer(canvas: Any, doc: Any) -> None:
    from reportlab.lib import colors

    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 24, "FloodLens - decision support only")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 24, f"Page {doc.page}")
    canvas.restoreState()


def _build_basic_pdf(lines: list[str]) -> bytes:
    wrapped_lines: list[str] = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(wrap(str(line), width=88) or [""])

    pages = [wrapped_lines[index : index + 45] for index in range(0, len(wrapped_lines), 45)]
    objects: list[bytes] = []
    content_object_ids: list[int] = []
    page_object_ids: list[int] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"")
    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    assert catalog_id == 1 and pages_id == 2 and font_id == 3

    for page_lines in pages:
        commands = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in page_lines:
            commands.append(f"({_escape_pdf(line)}) Tj")
            commands.append("T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        content_id = add_object(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page_id = add_object(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        content_object_ids.append(content_id)
        page_object_ids.append(page_id)

    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{page_id} 0 R' for page_id in page_object_ids)}] /Count {len(page_object_ids)} >>".encode()

    output = BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{index} 0 obj\n".encode())
        output.write(payload)
        output.write(b"\nendobj\n")
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n".encode())
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return output.getvalue()


def _escape_pdf(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _escape_xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _clean(value: Any, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_text(score: Any, level: Any) -> str:
    number = _as_float(score)
    score_text = f"{number:.4f}" if number is not None else "-"
    level_text = _clean(level, "Unknown")
    return f"{score_text} ({level_text})"


def _signed(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "-"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:.4f}"


def _friendly_changed_fields(fields: list[Any]) -> list[str]:
    labels = {
        "rainfall_7d_mm": "Rain expected this week",
        "monthly_rainfall_mm": "Wet month pressure",
        "elevation_m": "Ground height",
        "distance_to_river_m": "Distance from river",
        "nearest_evac_km": "Access to safe point",
        "population_density_per_km2": "People exposed nearby",
        "historical_flood_count": "Known past flood events",
        "infrastructure_score": "Drainage and road condition",
    }
    if not fields:
        return ["No scenario assumption was changed."]
    return [labels.get(str(field), str(field)) for field in fields]
