from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

PRIMARY_BLACK = "#05070A"
SECONDARY_BLACK = "#0A0F14"
PANEL = "#0E151C"
ELEVATED_PANEL = "#131C24"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#F7FAFC"
MUTED = "#8B98A5"
ORANGE = "#FF7A00"
BRIGHT_ORANGE = "#FF9A2E"
MUTED_ORANGE = "#B85A00"
AMBER = "#F5A623"
GREEN = "#20C997"
RED = "#FF4D4F"
INFO = "#4DA3FF"
DISABLED = "#4B5563"
CYAN = "#35D0C8"
PURPLE = "#A78BFA"
GRID = "rgba(255,255,255,0.075)"

CHART_COLORS = {
    "growth": ORANGE,
    "growth_net": BRIGHT_ORANGE,
    "spy": INFO,
    "qqq": PURPLE,
    "positive": GREEN,
    "negative": RED,
    "warning": AMBER,
    "neutral": MUTED,
    "cash": DISABLED,
}

PLOTLY_TEMPLATE = "growth_terminal_dark"

pio.templates[PLOTLY_TEMPLATE] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, IBM Plex Sans, system-ui, sans-serif", "color": TEXT, "size": 12},
        colorway=[ORANGE, INFO, PURPLE, GREEN, RED, AMBER, CYAN, MUTED],
        xaxis={
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": BORDER,
            "tickfont": {"color": MUTED},
            "title": {"font": {"color": MUTED}},
        },
        yaxis={
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "linecolor": BORDER,
            "tickfont": {"color": MUTED},
            "title": {"font": {"color": MUTED}},
        },
        legend={"font": {"color": MUTED}, "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 28, "r": 28, "t": 58, "b": 42},
    )
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;650;750;850&display=swap');
:root {{
  --bg: {PRIMARY_BLACK}; --bg2: {SECONDARY_BLACK}; --panel: {PANEL}; --panel-elevated: {ELEVATED_PANEL};
  --border: {BORDER}; --text: {TEXT}; --muted: {MUTED}; --orange: {ORANGE}; --orange-bright: {BRIGHT_ORANGE};
  --orange-muted: {MUTED_ORANGE}; --amber: {AMBER}; --green: {GREEN}; --red: {RED}; --info: {INFO}; --disabled: {DISABLED};
  --radius-card: 16px; --radius-small: 10px; --space-1: 8px; --space-2: 16px; --space-3: 24px; --space-4: 32px;
}}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid='stHeader'] {{ background: rgba(5,7,10,.42); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(255,255,255,0.04); }}
header[data-testid='stHeader'] button {{ color: var(--orange-bright) !important; }}
.stApp {{
  background:
    radial-gradient(circle at 10% 0%, rgba(255,122,0,.16) 0%, rgba(255,122,0,.055) 21%, rgba(5,7,10,0) 43%),
    linear-gradient(135deg, #05070A 0%, #070A0E 45%, #020304 100%);
  color: var(--text);
  font-family: Inter, IBM Plex Sans, system-ui, sans-serif;
}}
.block-container {{ padding: 20px 28px 42px 28px; max-width: 1560px; }}
[data-testid='stSidebar'] {{
  background: linear-gradient(180deg, #070A0E 0%, #05070A 100%);
  border-right: 1px solid var(--border);
  box-shadow: 18px 0 40px rgba(0,0,0,.24);
}}
[data-testid='stSidebar'] * {{ font-family: Inter, IBM Plex Sans, system-ui, sans-serif; }}
[data-testid='stSidebar'] .stRadio label, [data-testid='stSidebar'] .stSelectbox label {{ color: var(--muted) !important; font-size: .78rem; }}
[data-testid='stSidebar'] [role='radiogroup'] label {{
  border-radius: 10px; padding: 8px 10px; margin: 3px 0; border-left: 3px solid transparent;
}}
[data-testid='stSidebar'] [role='radiogroup'] label:hover {{ background: rgba(255,122,0,.08); border-left-color: rgba(255,122,0,.42); }}
.stSelectbox div[data-baseweb='select'] > div {{ background: #0E151C; border: 1px solid var(--border); border-radius: 10px; }}
button, input, textarea {{ font-family: Inter, IBM Plex Sans, system-ui, sans-serif !important; }}
.hero {{
  padding: 22px 24px; border: 1px solid var(--border); border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,122,0,.13), rgba(19,28,36,.82) 42%, rgba(10,15,20,.92));
  box-shadow: 0 18px 55px rgba(0,0,0,.30); margin-bottom: 24px;
}}
.hero h1 {{ margin: 0; color: var(--text); font-size: 2rem; font-weight: 850; letter-spacing: -.045em; line-height: 1.05; }}
.hero p {{ color: var(--muted); margin: 8px 0 0 0; font-size: .92rem; }}
.page-head {{
  display: flex; justify-content: space-between; gap: 18px; align-items: flex-start;
  padding: 18px 20px; margin-bottom: 24px; border: 1px solid var(--border); border-radius: 18px;
  background: rgba(14,21,28,.88);
}}
.page-title {{ font-size: 1.55rem; font-weight: 780; letter-spacing: -.035em; margin: 0; }}
.page-subtitle {{ color: var(--muted); font-size: .88rem; margin-top: 6px; }}
.kpi {{
  min-height: 108px; padding: 16px; border: 1px solid var(--border); border-radius: var(--radius-card);
  background: linear-gradient(180deg, rgba(19,28,36,.95), rgba(14,21,28,.96));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.035), 0 10px 28px rgba(0,0,0,.18);
}}
.kpi.positive {{ border-color: rgba(32,201,151,.28); }}
.kpi.negative {{ border-color: rgba(255,77,79,.32); }}
.kpi.warning {{ border-color: rgba(245,166,35,.34); }}
.kpi.info {{ border-color: rgba(77,163,255,.30); }}
.kpi.official {{ box-shadow: inset 3px 0 0 rgba(255,122,0,.88), 0 10px 28px rgba(0,0,0,.18); }}
.kpi-label {{ color: var(--muted); font-size: .74rem; font-weight: 650; letter-spacing: .015em; }}
.kpi-value {{ color: var(--text); font-size: 1.48rem; font-weight: 850; margin-top: 9px; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }}
.kpi-note {{ color: var(--muted); font-size: .78rem; margin-top: 8px; min-height: 18px; }}
.badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: 999px; font-size: .72rem; font-weight: 750; border: 1px solid var(--border); background: rgba(255,255,255,.035); color: var(--muted); }}
.badge.pass {{ color: var(--green); border-color: rgba(32,201,151,.30); background: rgba(32,201,151,.08); }}
.badge.warmup {{ color: var(--info); border-color: rgba(77,163,255,.30); background: rgba(77,163,255,.08); }}
.badge.warning {{ color: var(--amber); border-color: rgba(245,166,35,.32); background: rgba(245,166,35,.08); }}
.badge.blocked, .badge.failed {{ color: var(--red); border-color: rgba(255,77,79,.34); background: rgba(255,77,79,.08); }}
.badge.official {{ color: var(--orange-bright); border-color: rgba(255,122,0,.40); background: rgba(255,122,0,.10); }}
.badge.diagnostic, .badge.reconstructed {{ color: var(--muted); border-color: var(--border); background: rgba(139,152,165,.08); }}
.badge.single_source_fresh {{ color: var(--info); border-color: rgba(77,163,255,.30); background: rgba(77,163,255,.08); }}
.holding-card {{ padding: 16px; border-radius: var(--radius-card); background: linear-gradient(180deg, rgba(19,28,36,.96), rgba(9,13,18,.96)); border: 1px solid var(--border); margin-bottom: 12px; box-shadow: inset 3px 0 0 rgba(255,122,0,.72); }}
.holding-ticker {{ font-size: 1.42rem; font-weight: 850; color: var(--text); }}
.small-muted, .section-note {{ color: var(--muted); font-size: .84rem; }}
.status-strip {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding:10px 14px; border:1px solid var(--border); border-radius:16px; background:linear-gradient(90deg, rgba(255,122,0,.08), rgba(15,22,30,.86)); margin:10px 0 18px 0; }}
.status-strip span {{ color: var(--muted); font-size:.86rem; }}
.status-strip b {{ color: var(--text); }}
.alert-box {{ border: 1px solid var(--border); border-radius: var(--radius-card); padding: 14px 16px; background: rgba(14,21,28,.92); color: var(--muted); margin: 12px 0; }}
.alert-box.warning {{ border-color: rgba(245,166,35,.32); color: #ffd796; }}
.alert-box.negative {{ border-color: rgba(255,77,79,.34); color: #ffb0b1; }}
.alert-box.info {{ border-color: rgba(77,163,255,.30); color: #b9dcff; }}
.stDataFrame {{ border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
.stDataFrame [data-testid='stTable'] {{ font-size: .84rem; }}
div[data-testid='stDataFrame'] div[role='columnheader'] {{ background: #131C24 !important; color: var(--orange-bright) !important; }}
hr {{ border-color: var(--border); }}
.stTabs [data-baseweb='tab-list'] {{ gap: 10px; }}
.stTabs [data-baseweb='tab'] {{ background: rgba(14,21,28,.86); border-radius: 10px; border: 1px solid var(--border); color: var(--muted); }}
.stTabs [aria-selected='true'] {{ color: var(--orange-bright) !important; border-color: rgba(255,122,0,.42); background: rgba(255,122,0,.08); }}
</style>
"""


def apply_plotly_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": title, "font": {"size": 17, "color": TEXT}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter, IBM Plex Sans, system-ui, sans-serif"},
        margin={"l": 28, "r": 28, "t": 60 if title else 30, "b": 42},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickfont={"color": MUTED}, title_font={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickfont={"color": MUTED}, title_font={"color": MUTED})
    return fig
