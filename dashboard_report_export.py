
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from dashboard_report_assets import BLACK, BRIGHT_ORANGE, MUTED, ORANGE, esc, fmt_num, fmt_pct, fmt_pct_points, simple_svg_line, table_html
from dashboard_report_templates import ReportBundle



def _df_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "No rows available."
    view = df.head(max_rows).fillna("").copy()
    cols = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        vals = [str(row.get(c, "")).replace("|", "/") for c in view.columns]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Trimmed to first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)

def _kv_md(title: str, items: dict) -> str:
    lines = [f"### {title}", ""]
    for k, v in items.items():
        lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
    return "\n".join(lines)


def markdown_report(bundle: ReportBundle) -> str:
    ex = bundle.sections["executive"]
    perf = bundle.sections["performance"]
    stats = perf["stats"]
    alerts = bundle.sections["alerts"]["active"]
    holdings = bundle.sections["portfolio"]["holdings"]
    actions = bundle.sections["execution"]["actions"]
    lines = [
        f"# {bundle.title}",
        "",
        f"Generated: {bundle.generated_at}",
        f"Period: {bundle.start_date} to {bundle.end_date}",
        "",
        "## Table of Contents",
        "1. Executive Summary",
        "2. Performance",
        "3. Portfolio",
        "4. Risk",
        "5. Execution",
        "6. Research",
        "7. Governance",
        "8. Alerts",
        "9. Appendix",
        "",
        "## 1. Executive Summary",
        _kv_md("Key Metrics", ex),
        "## 2. Performance",
        _kv_md("Performance Statistics", {k: fmt_pct(v) if k in {"CAGR", "Max Drawdown", "Volatility"} else fmt_num(v) for k, v in stats.items()}),
        "## 3. Portfolio",
        _df_to_markdown(holdings) if not holdings.empty else "No official holdings available.",
        "## 4. Risk",
        _kv_md("Risk Metrics", {k: fmt_pct(v) if k not in {"HHI"} else fmt_num(v) for k, v in bundle.sections["risk"].items()}),
        "## 5. Execution",
        _df_to_markdown(actions) if not actions.empty else "No official actions in period.",
        "## 6. Research",
        _df_to_markdown(bundle.sections["research"]) if not bundle.sections["research"].empty else "No research governance rows available.",
        "## 7. Governance",
        "Official namespace only. Real capital remains governed by current lifecycle status.",
        "## 8. Alerts",
        _df_to_markdown(alerts) if not alerts.empty else "No active alerts.",
        "## 9. Appendix",
        _df_to_markdown(bundle.source_audit),
        "",
        "Read-only report. No model, optimizer, scheduler, execution, paper, accounting, governance, parameter, or order logic changed.",
    ]
    return "\n\n".join(lines)


def html_report(bundle: ReportBundle) -> str:
    ex = bundle.sections["executive"]
    perf = bundle.sections["performance"]
    returns = []
    rows = perf["rows"]
    if not rows.empty and "gross_portfolio_value" in rows.columns:
        returns = pd.to_numeric(rows["gross_portfolio_value"], errors="coerce").dropna().tolist()
    elif not rows.empty and "gross_daily_return" in rows.columns:
        returns = ((1 + pd.to_numeric(rows["gross_daily_return"], errors="coerce").fillna(0)).cumprod() * 100000).tolist()
    holdings = bundle.sections["portfolio"]["holdings"]
    actions = bundle.sections["execution"]["actions"]
    alerts = bundle.sections["alerts"]["active"]
    stat_rows = [{"metric": k, "value": fmt_pct(v) if k in {"CAGR", "Max Drawdown", "Volatility"} else fmt_num(v)} for k, v in perf["stats"].items()]
    kpis = "".join(f"<div class='card'><div class='label'>{esc(k.replace('_',' ').title())}</div><div class='value'>{esc(v)}</div></div>" for k, v in ex.items())
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{esc(bundle.title)}</title>
<style>
body {{ margin:0; background:{BLACK}; color:#F7FAFC; font-family:Inter,Arial,sans-serif; }}
.page {{ max-width:1180px; margin:0 auto; padding:36px; }}
.hero {{ border:1px solid rgba(255,255,255,.09); border-radius:22px; padding:28px; background:linear-gradient(135deg, rgba(255,122,0,.18), rgba(17,24,32,.92)); }}
h1 {{ margin:0; font-size:34px; }} h2 {{ margin-top:34px; border-bottom:1px solid rgba(255,255,255,.10); padding-bottom:8px; }}
.subtitle,.label,.caption {{ color:{MUTED}; }} .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:22px 0; }}
.card {{ border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; background:#111820; }}
.value {{ font-size:20px; font-weight:800; margin-top:8px; color:white; }}
table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }} th,td {{ border-bottom:1px solid rgba(255,255,255,.08); padding:8px; text-align:left; }} th {{ color:{BRIGHT_ORANGE}; }}
.line-chart {{ width:100%; height:auto; margin:12px 0; }} .toc li {{ margin:5px 0; }} .disclaimer {{ color:{MUTED}; margin-top:34px; font-size:12px; }}
</style></head><body><div class='page'>
<div class='hero'><h1>{esc(bundle.title)}</h1><div class='subtitle'>Generated {esc(bundle.generated_at)} · Period {esc(bundle.start_date)} to {esc(bundle.end_date)} · Official namespace only</div></div>
<h2>Table of Contents</h2><ol class='toc'><li>Executive Summary</li><li>Performance</li><li>Portfolio</li><li>Risk</li><li>Execution</li><li>Research</li><li>Governance</li><li>Alerts</li><li>Appendix</li></ol>
<h2>1. Executive Summary</h2><div class='grid'>{kpis}</div>
<h2>2. Performance</h2>{simple_svg_line(returns, label='Figure 1. Official Gross Equity Curve')} {simple_svg_line(perf['drawdown'], color='#FF4D4F', label='Figure 2. Official Drawdown Curve')} {table_html(stat_rows, ['metric','value'])}
<h2>3. Portfolio</h2>{table_html(holdings.fillna('').to_dict('records'), list(holdings.columns)) if not holdings.empty else '<div class=note>No holdings.</div>'}
<h2>4. Risk</h2>{table_html([{ 'metric': k, 'value': fmt_pct(v) if k not in {'HHI'} else fmt_num(v)} for k, v in bundle.sections['risk'].items()], ['metric','value'])}
<h2>5. Execution</h2>{table_html(actions.fillna('').to_dict('records'), list(actions.columns)) if not actions.empty else '<div class=note>No actions in period.</div>'}
<h2>6. Research</h2>{table_html(bundle.sections['research'].fillna('').to_dict('records'), list(bundle.sections['research'].columns)) if not bundle.sections['research'].empty else '<div class=note>No research rows.</div>'}
<h2>7. Governance</h2><p>Pipeline, market data, accounting, lifecycle and promotion state are read from official governance sources only.</p>
<h2>8. Alerts</h2>{table_html(alerts.fillna('').to_dict('records'), list(alerts.columns)) if not alerts.empty else '<div class=note>No active alerts.</div>'}
<h2>9. Appendix</h2>{table_html(bundle.source_audit.fillna('').to_dict('records'), list(bundle.source_audit.columns))}
<div class='disclaimer'>Read-only institutional report. No model, optimizer, scheduler, execution, paper, accounting, governance, parameter or order logic changed.</div>
</div></body></html>"""


def _pdf_escape(text: str) -> str:
    return str(text).replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def _wrap(text: str, width: int = 92) -> list[str]:
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def pdf_report(bundle: ReportBundle, path: str) -> None:
    md = markdown_report(bundle)
    lines = []
    for raw in md.splitlines():
        if raw.startswith('#'):
            lines.append(raw.replace('#','').strip().upper())
        else:
            lines.extend(_wrap(raw, 96))
    pages = [lines[i:i+48] for i in range(0, len(lines), 48)] or [[]]
    objects = []
    page_ids = []
    font_id = 3
    for page in pages:
        content = ["BT", "/F1 9 Tf", "50 790 Td"]
        for line in page:
            content.append(f"({_pdf_escape(line)}) Tj")
            content.append("0 -14 Td")
        content.append("ET")
        stream = "\n".join(content).encode('latin-1', errors='replace')
        content_id = len(objects) + 4
        page_id = len(objects) + 5
        objects.append(f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream\nendobj\n")
        objects.append(f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>\nendobj\n".encode())
        page_ids.append(page_id)
    header_objs = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(str(i)+' 0 R' for i in page_ids)}] /Count {len(page_ids)} >>\nendobj\n".encode(),
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    all_objs = header_objs + objects
    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in all_objs:
        offsets.append(len(buf)); buf.extend(obj)
    xref = len(buf)
    buf.extend(f"xref\n0 {len(all_objs)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode())
    buf.extend(f"trailer\n<< /Size {len(all_objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    Path(path).write_bytes(buf)


def export_report(bundle: ReportBundle, basename: str) -> dict[str, str]:
    paths = {"markdown": basename + ".md", "html": basename + ".html", "pdf": basename + ".pdf"}
    Path(paths["markdown"]).write_text(markdown_report(bundle), encoding='utf-8')
    Path(paths["html"]).write_text(html_report(bundle), encoding='utf-8')
    pdf_report(bundle, paths["pdf"])
    return paths
