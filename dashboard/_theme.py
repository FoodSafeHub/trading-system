"""Shared visual layer for the Streamlit dashboard.

Every page calls `apply_theme()` once near the top to install the CSS overrides,
then composes pages out of the helpers below (`section`, `kpi_row`, `pill`,
`money`, `pct`). The goal is one consistent visual language across pages so the
UI feels less like a stack of unrelated dashboards.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import streamlit as st


# ── Market sessions ──────────────────────────────────────────────────────────
# (label, open, close, tz). Kept in sync with app/config.py defaults — the
# backend risk engine is the authority that actually gates orders; this is the
# at-a-glance display so you know whether a market is tradeable right now.
_SESSIONS = [
    ("US",    time(9, 30), time(16, 0), "America/New_York"),
    ("India", time(9, 15), time(15, 30), "Asia/Kolkata"),
]


def _session_status(open_t: time, close_t: time, tz_name: str) -> tuple[bool, str]:
    now = datetime.now(tz=ZoneInfo(tz_name))
    is_weekday = now.weekday() < 5
    is_open = is_weekday and (open_t <= now.time() < close_t)
    return is_open, now.strftime("%H:%M")


def market_status_bar() -> None:
    """Render a compact OPEN/CLOSED line for the US and India sessions.

    Surfaces both clocks so a Zerodha (₹, IST) order isn't a surprise when the
    NSE is shut. Display-only — order gating lives in the risk engine.
    """
    bits = []
    for label, open_t, close_t, tz_name in _SESSIONS:
        is_open, clock = _session_status(open_t, close_t, tz_name)
        dot = "🟢" if is_open else "🔴"
        state = "OPEN" if is_open else "CLOSED"
        bits.append(f"{dot} **{label}** {state} · {clock}")
    st.caption("&nbsp;&nbsp;|&nbsp;&nbsp;".join(bits), unsafe_allow_html=True)


# ── CSS overrides ────────────────────────────────────────────────────────────
# Reasoning behind specific rules:
#  - Streamlit's default metric padding is too generous → tightens scan density.
#  - Section dividers are full-bleed black bars; replaced with hairline grey.
#  - Monospace numeric values in metric/table cells make column alignment obvious.
#  - Tabs get a quieter underline + readable letter-spacing so a 4-tab page
#    doesn't read like browser chrome.
_CSS = """
<style>
/* ── Layout ──────────────────────────────────────────────────────────── */
section.main > div.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

/* ── Headings ────────────────────────────────────────────────────────── */
h1 { font-weight: 600; letter-spacing: -0.01em; margin-bottom: 0.25rem; }
h2 { font-weight: 600; letter-spacing: -0.005em; margin-top: 1.5rem; }
h3 { font-weight: 500; color: rgba(250,250,250,0.85); margin-top: 1rem; }

/* ── Metrics ─────────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: rgba(250,250,250,0.55);
}
div[data-testid="stMetricValue"] {
    font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 1.35rem;
    font-weight: 500;
}
div[data-testid="stMetricDelta"] {
    font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 0.78rem;
}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr {
    margin: 1.5rem 0 1rem 0;
    border: 0;
    border-top: 1px solid rgba(255,255,255,0.08);
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.92rem;
    letter-spacing: 0.01em;
    padding: 0.4rem 1rem;
}

/* ── Tables ──────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] th {
    font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    font-size: 0.85rem;
}

/* ── Pills (status badges built via st.markdown) ─────────────────────── */
.tx-pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    line-height: 1.4;
    border: 1px solid transparent;
}
.tx-pill.green { background: rgba(0, 200, 120, 0.12); color: #4ade80; border-color: rgba(0, 200, 120, 0.35); }
.tx-pill.red   { background: rgba(240, 65, 65, 0.12); color: #f87171; border-color: rgba(240, 65, 65, 0.35); }
.tx-pill.amber { background: rgba(240, 170, 50, 0.14); color: #fbbf24; border-color: rgba(240, 170, 50, 0.40); }
.tx-pill.grey  { background: rgba(160, 160, 160, 0.10); color: #a3a3a3; border-color: rgba(160, 160, 160, 0.30); }
.tx-pill.blue  { background: rgba(60, 130, 240, 0.12); color: #60a5fa; border-color: rgba(60, 130, 240, 0.35); }

/* ── Section subtitles (caption-sized helper text under section()) ───── */
.tx-subtle {
    color: rgba(250,250,250,0.55);
    font-size: 0.85rem;
    margin: -0.5rem 0 0.75rem 0;
}
</style>
"""


def apply_theme(page_title: str, *, page_icon: str | None = None) -> None:
    """Set page config and inject the dashboard's CSS.

    Call once per page, before any other Streamlit output. `page_icon` is
    optional — when omitted, the browser tab gets no emoji.
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Layout helpers ───────────────────────────────────────────────────────────


def section(title: str, subtitle: str | None = None, *, level: int = 2) -> None:
    """Render a section header with optional subtitle.

    Replaces the `st.subheader(...) + st.caption(...)` pair that's repeated
    across every page, and gives subtitles a quieter look than st.caption.
    """
    if level == 2:
        st.markdown(f"## {title}")
    elif level == 3:
        st.markdown(f"### {title}")
    else:
        st.markdown(f"#### {title}")
    if subtitle:
        st.markdown(f"<div class='tx-subtle'>{subtitle}</div>", unsafe_allow_html=True)


def divider() -> None:
    """Quiet hairline divider — replaces st.divider() which is too heavy."""
    st.markdown("<hr/>", unsafe_allow_html=True)


def kpi_row(items: Sequence[tuple[str, str]] | Sequence[tuple[str, str, str | None]]) -> None:
    """Render a row of KPI metrics from `(label, value)` or `(label, value, delta)` tuples.

    Picks `len(items)` columns automatically. Use this instead of hand-rolled
    `c1, c2, c3 = st.columns(3); c1.metric(...)` blocks.
    """
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        if len(item) == 2:
            label, value = item
            col.metric(label, value)
        else:
            label, value, delta = item
            col.metric(label, value, delta=delta)


def pill(text: str, color: str = "grey") -> str:
    """Return an HTML pill — use with `st.markdown(..., unsafe_allow_html=True)`.

    Colors: green, red, amber, grey, blue. Use these consistently:
        green = healthy / OK / on
        red   = error / kill switch active / loss
        amber = warning / approaching limit
        blue  = informational / inactive-but-not-bad
        grey  = neutral / off / disabled
    """
    return f"<span class='tx-pill {color}'>{text}</span>"


def status_row(items: Iterable[tuple[str, str, str]]) -> None:
    """Render a horizontal row of `(label, pill_text, color)` status indicators.

    Use for the top strip on Home / status bars: API up, market open, etc.
    """
    parts = []
    for label, text, color in items:
        parts.append(f"<span style='margin-right:1.5rem;'><span style='color:rgba(250,250,250,0.55);font-size:0.78rem;'>{label}</span> {pill(text, color)}</span>")
    st.markdown("<div style='margin-bottom:0.5rem;'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


# ── Formatters ───────────────────────────────────────────────────────────────


# Broker → currency symbol. Zerodha trades NSE/BSE in rupees; everything else
# is a US-dollar broker. Used so India positions don't render as "$".
_BROKER_CURRENCY = {
    "zerodha": "₹",
    "schwab": "$",
    "webull": "$",
    "paper": "$",
    "default": "$",
}


def currency_symbol(broker: str | None) -> str:
    """Currency glyph for a broker name. Unknown/None → '$'."""
    if not broker:
        return "$"
    return _BROKER_CURRENCY.get(broker.lower().strip(), "$")


def money(
    value: float | None,
    *,
    decimals: int = 2,
    dash: str = "—",
    currency: str = "$",
) -> str:
    """Format a monetary value. `None` or NaN → em-dash.

    `currency` is the glyph to prefix (default '$'). Pass currency_symbol(broker)
    for India (₹) vs US ($) values.
    """
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if v != v:  # NaN
        return dash
    return f"{currency}{v:,.{decimals}f}"


def pct(value: float | None, *, decimals: int = 1, dash: str = "—") -> str:
    """Format a percentage. Accepts either fractional (0.15) or whole (15.0).

    Heuristic: if abs(value) <= 1.5, treat as fraction; otherwise treat as
    already-percentage. Most of our APIs return fractions, but some return %.
    """
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if v != v:
        return dash
    if abs(v) <= 1.5:
        v = v * 100
    return f"{v:+.{decimals}f}%" if v else f"0.{decimals * '0'}%"


def num(value: float | int | None, *, decimals: int = 2, dash: str = "—") -> str:
    """Plain numeric formatter. `None` or NaN → em-dash."""
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if v != v:
        return dash
    if decimals == 0:
        return f"{int(v):,}"
    return f"{v:,.{decimals}f}"
