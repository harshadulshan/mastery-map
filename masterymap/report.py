"""Printable practice sheets (PDF: one page per student, then the teacher's answer key) and an Excel copy."""

from __future__ import annotations

import io
import os
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_REG = False


def _fonts():
    global _REG
    if not _REG:
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
        _REG = True


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _math(t):
    """Escaped text with powers written a^b or a^(...) raised: 2^5 -> 2<super>5</super>."""
    t = re.sub(r"\^\(([^)]*)\)", r"<super>\1</super>", _esc(t))
    return re.sub(r"\^(-?\d+)", r"<super>\1</super>", t)


def worksheets(sets: dict, names: dict, class_name: str, day: pd.Timestamp) -> bytes:
    """sets: {student id: practice set table (plan.practice_set)}; names: {student id: name}."""
    _fonts()
    st = {
        "h": ParagraphStyle("h", fontName="DejaVu-Bold", fontSize=14, leading=18, spaceAfter=2),
        "sub": ParagraphStyle("sub", fontName="DejaVu", fontSize=9, leading=12, textColor=colors.HexColor("#555555")),
        "q": ParagraphStyle("q", fontName="DejaVu", fontSize=11, leading=15, alignment=TA_LEFT, leftIndent=18,
                            firstLineIndent=-18),
        "tag": ParagraphStyle("tag", fontName="DejaVu", fontSize=7.5, leading=10, leftIndent=18,
                              textColor=colors.HexColor("#777777")),
        "small": ParagraphStyle("small", fontName="DejaVu", fontSize=8, leading=10),
    }
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"Practice sets, {class_name}", author="Mastery Map")
    story = []
    for sid, df in sets.items():
        focus = ", ".join(dict.fromkeys(df.loc[df["Kind"] == "focus", "Skill"]))
        part = [Paragraph(f"Practice set · {_esc(names.get(sid, sid))}", st["h"]),
                Paragraph(f"{_esc(class_name)} · {sid} · {day:%d %B %Y} · {len(df)} questions"
                          + (f" · focus: {_esc(focus)}" if focus else ""), st["sub"]), Spacer(1, 5 * mm)]
        gap = max(6.0, min(30.0, (265 - 24 - 11 * max(len(df), 1)) / max(len(df), 1) - 1.5)) * mm   # one page
        for j, r in enumerate(df.itertuples(index=False), 1):
            q = r.Question if isinstance(r.Question, str) and r.Question.strip() else f"Question {r[0]}"
            part.append(KeepTogether([Paragraph(f"<b>{j}.</b>&nbsp;&nbsp;{_math(q)}", st["q"]),
                                      Paragraph(f"{_esc(r.Skill)} · {r[0]}", st["tag"]), Spacer(1, gap)]))
        story += part + [PageBreak()]
    # the teacher's answer key
    story.append(Paragraph("Answer key", st["h"]))
    story.append(Paragraph(f"{_esc(class_name)} · {day:%d %B %Y}", st["sub"]))
    story.append(Spacer(1, 4 * mm))
    for sid, df in sets.items():
        cells = [[Paragraph(f"<b>{_esc(names.get(sid, sid))}</b> ({sid})", st["small"]), ""]]
        for j, r in enumerate(df.itertuples(index=False), 1):
            ans = r.Answer if isinstance(r.Answer, str) and r.Answer.strip() else "–"
            cells.append([Paragraph(f"{j}. {r[0]}", st["small"]), Paragraph(_math(ans), st["small"])])
        t = Table(cells, colWidths=[40 * mm, 130 * mm])
        t.setStyle(TableStyle([("SPAN", (0, 0), (1, 0)), ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#eef3fb")),
                               ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
        story += [KeepTogether([t]), Spacer(1, 3 * mm)]
    doc.build(story)
    return buf.getvalue()


def sets_excel(sets: dict, names: dict) -> bytes:
    rows = []
    for sid, df in sets.items():
        for j, r in enumerate(df.to_dict("records"), 1):
            rows.append({"Student ID": sid, "Name": names.get(sid, sid), "#": j, **{k: r[k] for k in
                         ("Question ID", "Skill", "Kind", "Question", "Answer", "Why")},
                         "Chance": round(r["Chance"], 2)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Practice sets", index=False)
        ws = xw.book["Practice sets"]
        for col, w in zip("ABCDEFGHIJ", (11, 22, 4, 10, 26, 8, 60, 22, 50, 8)):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
    return buf.getvalue()
