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


# ── CSS overrides — "institutional terminal" design system ─────────────────────
# One cohesive premium dark theme. Strategy:
#  - All colour decisions live in CSS custom properties (:root) so the palette is
#    tunable in one place and reused by the pill/section helpers below.
#  - Palette: deep graphite background, layered slate panels, soft off-white text,
#    muted warm-grey secondary text, restrained brushed-teal primary accent, and a
#    very subtle brushed-gold reserved for the section accent rule only.
#  - Positive = elegant teal-green, Negative = muted (not saturated) red.
#  - Native Streamlit widgets (buttons, selects, sidebar, tabs, banners, tables,
#    progress) are retargeted by their data-testid / baseweb hooks so nothing
#    reads as a default admin dashboard. Functionality is untouched — this is
#    purely presentational.
#
# Token reference (mirrored in .streamlit/config.toml for native widgets):
#   --bg        #0e1117  app background (graphite, not black)
#   --panel     #161a22  card / surface
#   --panel-2   #1c212c  raised surface (hover, headers)
#   --line      hairline borders
#   --text      #e6e8ec  primary
#   --text-2    #9aa3b2  secondary / labels
#   --text-3    #6b7382  tertiary / helper
#   --teal      #3fb6a8  primary accent
#   --gold      #c2a878  reserved highlight
#   --pos / --neg        gains / losses
_CSS = """
<style>
/* Inter — professional fintech sans. Falls back to the system stack offline. */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:       #0e1117;
    --panel:    #161a22;
    --panel-2:  #1c212c;
    --line:     rgba(255,255,255,0.07);
    --line-2:   rgba(255,255,255,0.12);
    --text:     #e6e8ec;
    --text-2:   #9aa3b2;
    --text-3:   #6b7382;
    --teal:     #3fb6a8;
    --teal-dim: rgba(63,182,168,0.14);
    --gold:     #c2a878;
    --pos:      #5ec8a0;
    --pos-bg:   rgba(94,200,160,0.12);
    --neg:      #d98a8a;
    --neg-bg:   rgba(217,138,138,0.12);
    --font:     "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --mono:     "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;
    --radius:   10px;
    --shadow:   0 1px 2px rgba(0,0,0,0.35), 0 8px 24px rgba(0,0,0,0.22);

    /* Spacing scale — 4px base ramp, used for all vertical rhythm. */
    --sp-1: 4px;  --sp-2: 8px;  --sp-3: 12px; --sp-4: 16px;
    --sp-5: 24px; --sp-6: 32px; --sp-7: 48px;
}

/* Tabular-figures numerics: aligns columns without a code/mono look. */
.stApp { font-feature-settings: "tnum" 0; }

/* ── App canvas ──────────────────────────────────────────────────────── */
.stApp { background: var(--bg); font-family: var(--font); }
section.main > div.block-container {
    padding-top: var(--sp-6);
    padding-bottom: var(--sp-7);
    max-width: 1520px;
}
body, .stApp, .stMarkdown, p, span, label, div { color: var(--text); font-family: var(--font); }
/* Body copy: generous line-height for readability. */
.stMarkdown p, .stApp p { font-size: 0.9rem; line-height: 1.6; }

/* ── Sidebar — slim premium nav rail ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0b0e14;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
/* Nav links: quiet by default, teal accent + raised surface on hover/active. */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
    border-radius: 8px;
    padding: var(--sp-1) var(--sp-3);
    margin: 1px 0;
    color: var(--text-2);
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.005em;
    transition: background 0.12s ease, color 0.12s ease;
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.04);
    color: var(--text);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: var(--teal-dim);
    color: var(--text);
    box-shadow: inset 2px 0 0 var(--teal);
}

/* ── Headings — strict 3-step hierarchy (H1 ≫ H2 > H3) ───────────────── */
h1 {
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 1.75rem;
    line-height: 1.15;
    margin: 0 0 var(--sp-1) 0;
    color: #f3f5f8;
}
h2 {
    font-weight: 600;
    letter-spacing: -0.005em;
    font-size: 1.15rem;
    line-height: 1.25;
    margin: var(--sp-6) 0 var(--sp-3) 0;
    color: var(--text);
    padding-left: var(--sp-2);
    border-left: 2px solid var(--gold);   /* subtle brushed-gold section rule */
}
h3 {
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.3;
    letter-spacing: 0;
    color: var(--text);
    margin: var(--sp-4) 0 var(--sp-2) 0;
}

/* ── Metric / KPI cards — illuminated dark surfaces ──────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-4);
    box-shadow: var(--shadow);
    transition: border-color 0.15s ease;
}
div[data-testid="stMetric"]:hover { border-color: var(--line-2); }
div[data-testid="stMetricLabel"] {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-2);
    font-weight: 600;
    margin-bottom: var(--sp-1);
}
/* Tabular figures keep multi-card rows vertically aligned without a mono look. */
div[data-testid="stMetricValue"] {
    font-family: var(--font);
    font-variant-numeric: tabular-nums;
    font-size: 1.45rem;
    font-weight: 650;
    line-height: 1.1;
    letter-spacing: -0.01em;
    color: #f3f5f8;
}
div[data-testid="stMetricDelta"] {
    font-family: var(--font);
    font-variant-numeric: tabular-nums;
    font-size: 0.78rem;
}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr { margin: var(--sp-5) 0 var(--sp-4) 0; border: 0; border-top: 1px solid var(--line); }

/* ── Tabs — quiet, terminal-like ─────────────────────────────────────── */
div[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--line);
    gap: 0.25rem;
}
button[data-baseweb="tab"] {
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    padding: 0.45rem 1rem;
    color: var(--text-2);
}
button[data-baseweb="tab"]:hover { color: var(--text); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); font-weight: 550; }
div[data-baseweb="tab-highlight"] { background: var(--teal); height: 2px; }

/* ── Buttons — restrained, precise, consistent height ────────────────── */
div[data-testid="stButton"] > button,
div[data-testid="stLinkButton"] > a,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 8px;
    border: 1px solid var(--line-2);
    background: var(--panel-2);
    color: var(--text);
    font-weight: 500;
    font-size: 0.86rem;
    letter-spacing: 0.01em;
    min-height: 38px;                 /* aligns with selects/inputs in filter rows */
    transition: background 0.12s ease, border-color 0.12s ease;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stLinkButton"] > a:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #232936;
    border-color: var(--teal);
    color: #fff;
}
/* Primary buttons: a calm filled teal, no glow. */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: var(--teal);
    border-color: var(--teal);
    color: #07110f;
    font-weight: 600;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    background: #4cc7b8; border-color: #4cc7b8; color: #06100e;
}

/* ── Inputs / selects — consistent height for scannable filter rows ──── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: var(--panel) !important;
    border-color: var(--line-2) !important;
    border-radius: 8px !important;
    min-height: 38px;
    font-size: 0.86rem !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within { border-color: var(--teal) !important; }
/* Widget labels: quiet secondary text, tight to their control. */
div[data-testid="stWidgetLabel"] label, label[data-testid="stWidgetLabel"] {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-2);
    margin-bottom: var(--sp-1);
}

/* ── Radio / horizontal toggles ──────────────────────────────────────── */
div[role="radiogroup"] label { color: var(--text-2); font-size: 0.86rem; }

/* ── Tables — aligned numerics, header hierarchy, subtle hover ───────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
    margin: var(--sp-4) 0 var(--sp-5) 0;   /* breathing room above & below */
}
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] th {
    font-family: var(--font);
    font-variant-numeric: tabular-nums;     /* numeric columns stay aligned */
    font-size: 0.84rem;
}
div[data-testid="stDataFrame"] th {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem;
    color: var(--text-2);
    background: var(--panel-2);
    font-weight: 600;
}
div[data-testid="stDataFrame"] [role="row"]:hover { background: rgba(255,255,255,0.03); }

/* ── Progress bars (risk gauges) ─────────────────────────────────────── */
div[data-testid="stProgress"] > div > div > div { background: var(--teal); }

/* ── Banners / notices — also the polished empty-state surface ───────── */
div[data-testid="stAlert"] {
    border-radius: var(--radius);
    border: 1px solid var(--line-2);
    background: var(--panel);
    padding: var(--sp-3) var(--sp-4);
}
div[data-testid="stAlert"] p { font-size: 0.85rem; line-height: 1.5; color: var(--text-2); }

/* ── Pills (status badges built via st.markdown) ─────────────────────── */
.tx-pill {
    display: inline-block;
    padding: var(--sp-1) var(--sp-2);
    border-radius: 6px;          /* squared, terminal-credible — not bubbly */
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1.4;
    border: 1px solid transparent;
    font-family: var(--font);
}
.tx-pill.green { background: var(--pos-bg);             color: var(--pos);  border-color: rgba(94,200,160,0.3); }
.tx-pill.red   { background: var(--neg-bg);             color: var(--neg);  border-color: rgba(217,138,138,0.3); }
.tx-pill.amber { background: rgba(194,168,120,0.14);    color: var(--gold); border-color: rgba(194,168,120,0.34); }
.tx-pill.grey  { background: rgba(154,163,178,0.10);    color: var(--text-2); border-color: rgba(154,163,178,0.24); }
.tx-pill.blue  { background: var(--teal-dim);           color: var(--teal); border-color: rgba(63,182,168,0.32); }

/* ── Status strip (top command bar) ──────────────────────────────────── */
/* Sits between the page title and the first section — token margins set the
   title→bar→content rhythm. */
.tx-statusbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--sp-5);
    padding: var(--sp-3) var(--sp-4);
    margin: var(--sp-4) 0 var(--sp-2) 0;
    background: linear-gradient(180deg, var(--panel-2), var(--panel));
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}
.tx-stat { display: inline-flex; align-items: center; gap: var(--sp-2); }
.tx-stat-label { color: var(--text-3); font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }

/* ── Section subtitles (helper text under section()) ─────────────────── */
.tx-subtle {
    color: var(--text-2);
    font-size: 0.8rem;
    line-height: 1.5;
    margin: calc(-1 * var(--sp-1)) 0 var(--sp-3) var(--sp-2);
}

/* ── Captions / help text — the micro/meta tier ──────────────────────── */
div[data-testid="stCaptionContainer"], .stCaption {
    color: var(--text-3);
    font-size: 0.78rem;
    line-height: 1.5;
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
    """Render the top command/status bar from `(label, pill_text, color)` tuples.

    Use for the top strip on Home / status bars: API up, market open, etc.
    Renders as a single elevated panel so it reads as a command bar rather than
    a loose line of text.
    """
    parts = []
    for label, text, color in items:
        parts.append(
            f"<span class='tx-stat'>"
            f"<span class='tx-stat-label'>{label}</span>{pill(text, color)}"
            f"</span>"
        )
    st.markdown(
        "<div class='tx-statusbar'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


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
