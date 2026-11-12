"""Shared look & feel: colours, cards, table styles, chart layout."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# colours (validated reference palette: blue/red diverging, gray neutral)
# ---------------------------------------------------------------------------
UP, DOWN, NEUTRAL, MUTED = "#2a78d6", "#e34948", "#898781", "#c3c2b7"
UP_LIGHT, DOWN_LIGHT = "#86b6ef", "#f0a3a2"
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"
STATE_COLOR = {"UPTREND": UP, "DOWNTREND": DOWN, "MIXED": NEUTRAL,
               "Up · Low-vol": UP, "Up · High-vol": UP_LIGHT, "Down · Low-vol": DOWN_LIGHT, "Down · High-vol": DOWN,
               "ABOVE": UP, "BELOW": DOWN}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

def inject_css():
    st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
.sub {color:#52514e; font-size:0.95rem; margin-bottom: 1rem;}
.card {border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:14px 16px; background:#fcfcfb;
       height:100%; }
.card .lbl {font-size:0.78rem; text-transform:uppercase; letter-spacing:.04em; color:#52514e;}
.card .val {font-size:1.15rem; font-weight:650; margin:4px 0 2px 0; line-height:1.25}
.card .note {font-size:0.85rem; color:#52514e; line-height:1.35}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600;
       color:#fff; margin-right:6px}
.read {border-left:4px solid #2a78d6; background:#fcfcfb; padding:14px 18px; border-radius:6px;
       font-size:1rem; line-height:1.55}
.explain {background:#f0efec; border-radius:8px; padding:12px 16px; font-size:0.9rem; line-height:1.5;
          color:#2b2a28}
</style>
""", unsafe_allow_html=True)



def pill(text: str, color: str | None = None) -> str:
    color = color or STATE_COLOR.get(text, NEUTRAL)
    fg = "#0b0b0b" if color in (UP_LIGHT, DOWN_LIGHT, WARN) else "#fff"
    return f'<span class="pill" style="background:{color};color:{fg}">{text}</span>'


def card(label: str, value: str, note: str = "", color: str | None = None) -> str:
    bar = f"border-top:4px solid {color};" if color else ""
    return (f'<div class="card" style="{bar}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="note">{note}</div></div>')


def explain(md: str, title: str = "How this works"):
    with st.expander(title):
        st.markdown(md)


def signed_color(v: float) -> str:
    if pd.isna(v):
        return ""
    return f"color: {UP if v > 0 else DOWN if v < 0 else NEUTRAL}"


def z_color(v: float) -> str:
    if pd.isna(v) or abs(v) < 1.5:
        return ""
    a = min(abs(v) / 3, 1) * 0.35 + 0.12
    rgb = "42,120,214" if v > 0 else "227,73,72"
    return f"background-color: rgba({rgb},{a:.2f}); font-weight:600"


def state_style(v) -> str:
    c = STATE_COLOR.get(v)
    return f"color:{c}; font-weight:600" if c else ""


def base_layout(fig: go.Figure, h: int = 340, title: str | None = None, legend=True) -> go.Figure:
    if legend == "right":
        fig.update_layout(height=h, margin=dict(l=10, r=10, t=40 if title else 8, b=10),
                          title=dict(text=title, font=dict(size=14), y=1, yanchor="top", yref="container",
                                     pad=dict(t=8)) if title else None, hovermode="closest",
                          legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top"))
        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#e1e0d9", zeroline=False)
        return fig
    top = (34 if title else 8) + (30 if legend else 0)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=top, b=10),
                      title=dict(text=title, font=dict(size=14), y=1, yanchor="top", yref="container",
                                 pad=dict(t=8)) if title else None,
                      hovermode="x unified", showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e1e0d9", zeroline=False)
    return fig


def last_n_years(df, years):
    return df[df.index >= df.index[-1] - pd.DateOffset(years=years)]


