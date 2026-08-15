
from __future__ import annotations

from html import escape
from typing import Any

ORANGE = "#FF7A00"
BRIGHT_ORANGE = "#FF9A2E"
BLACK = "#05070A"
PANEL = "#111820"
TEXT = "#F7FAFC"
MUTED = "#8B98A5"
GREEN = "#20C997"
RED = "#FF4D4F"
AMBER = "#F5A623"
INFO = "#4DA3FF"


def esc(value: Any) -> str:
    return escape("" if value is None else str(value))


def fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        import pandas as pd
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "n/a"


def fmt_pct_points(value: Any, digits: int = 2) -> str:
    try:
        import pandas as pd
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "n/a"


def fmt_money(value: Any) -> str:
    try:
        import pandas as pd
        if pd.isna(value):
            return "n/a"
        return f"${float(value):,.2f}"
    except Exception:
        return "n/a"


def fmt_num(value: Any, digits: int = 3) -> str:
    try:
        import pandas as pd
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def simple_svg_line(series: list[float], width: int = 760, height: int = 190, color: str = ORANGE, label: str = "") -> str:
    clean = [float(x) for x in series if x is not None]
    if len(clean) < 2:
        return "<div class='chart-empty'>Chart unavailable: insufficient data</div>"
    mn, mx = min(clean), max(clean)
    span = mx - mn or 1.0
    step = width / max(len(clean) - 1, 1)
    points = []
    for i, value in enumerate(clean):
        x = i * step
        y = height - ((value - mn) / span) * (height - 24) - 12
        points.append(f"{x:.2f},{y:.2f}")
    return f"""
    <svg viewBox='0 0 {width} {height}' class='line-chart' role='img' aria-label='{esc(label)}'>
      <rect x='0' y='0' width='{width}' height='{height}' rx='12' fill='rgba(255,255,255,0.025)' stroke='rgba(255,255,255,0.08)'/>
      <polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}'/>
      <text x='14' y='24' fill='{MUTED}' font-size='12'>{esc(label)}</text>
      <text x='14' y='{height-12}' fill='{MUTED}' font-size='11'>min {mn:.2f} · max {mx:.2f}</text>
    </svg>
    """


def table_html(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "<div class='note'>No rows available.</div>"
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{esc(row.get(c, ''))}</td>" for c in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
