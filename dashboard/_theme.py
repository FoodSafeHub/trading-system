"""Shared visual layer for the Streamlit dashboard.

Every page calls `apply_theme()` once near the top to install the CSS overrides,
then composes pages out of the helpers below (`section`, `kpi_row`, `pill`,
`money`, `pct`). The goal is one consistent visual language across pages so the
UI feels less like a stack of unrelated dashboards and more like a professional
trading terminal.

Design system token reference (mirrored in .streamlit/config.toml):
  --bg        #0a0d13   deep graphite canvas
  --panel     #111620   primary surface (cards, containers)
  --panel-2   #181d28   raised surface (hover, headers, nav active)
  --panel-3   #1e2433   highest elevation (dropdowns, tooltips)
  --line      rgba(255,255,255,0.06)   hairline border
  --line-2    rgba(255,255,255,0.11)   medium border (hover, focus)
  --line-3    rgba(255,255,255,0.18)   strong border (active)
  --text      #eaecf0   primary off-white
  --text-2    #8e97a8   secondary / labels
  --text-3    #5a6373   tertiary / helpers / metadata
  --teal      #37b8aa   primary accent (restrained, not neon)
  --teal-dim  rgba(55,184,170,0.12)   teal background tint
  --gold      #b89c6b   section accent rule only
  --pos       #4db896   gains (teal-green, elegant)
  --pos-bg    rgba(77,184,150,0.10)
  --neg       #d07a7a   losses (muted terracotta, not alarming)
  --neg-bg    rgba(208,122,122,0.10)
  --warn      #c29445   warnings / amber
  --warn-bg   rgba(194,148,69,0.10)
  --info      #5b92d1   informational / blue
  --info-bg   rgba(91,146,209,0.10)
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import streamlit as st


# ── Market sessions ──────────────────────────────────────────────────────────
_SESSIONS = [
    ("US",    time(9, 30), time(16, 0),  "America/New_York"),
    ("India", time(9, 15), time(15, 30), "Asia/Kolkata"),
]


def _session_status(open_t: time, close_t: time, tz_name: str) -> tuple[bool, str]:
    now = datetime.now(tz=ZoneInfo(tz_name))
    is_weekday = now.weekday() < 5
    is_open = is_weekday and (open_t <= now.time() < close_t)
    return is_open, now.strftime("%H:%M")


def market_status_bar() -> None:
    """Compact OPEN/CLOSED clock bar for US and India sessions."""
    bits = []
    for label, open_t, close_t, tz_name in _SESSIONS:
        is_open, clock = _session_status(open_t, close_t, tz_name)
        dot = "🟢" if is_open else "🔴"
        state = "OPEN" if is_open else "CLOSED"
        bits.append(f"{dot} **{label}** {state} · {clock}")
    st.caption("&nbsp;&nbsp;|&nbsp;&nbsp;".join(bits), unsafe_allow_html=True)


# ── CSS — "institutional terminal" design system ──────────────────────────────
_CSS = """
<style>
/* ── Typeface ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design tokens ────────────────────────────────────────────────────── */
:root {
    /* Canvas */
    --bg:       #0a0d13;
    --panel:    #111620;
    --panel-2:  #181d28;
    --panel-3:  #1e2433;

    /* Borders */
    --line:     rgba(255,255,255,0.06);
    --line-2:   rgba(255,255,255,0.11);
    --line-3:   rgba(255,255,255,0.20);

    /* Text */
    --text:     #eaecf0;
    --text-2:   #8e97a8;
    --text-3:   #5a6373;

    /* Brand / accent */
    --teal:     #37b8aa;
    --teal-dim: rgba(55,184,170,0.12);
    --teal-glow:rgba(55,184,170,0.22);
    --gold:     #b89c6b;

    /* Semantic */
    --pos:      #4db896;
    --pos-bg:   rgba(77,184,150,0.10);
    --pos-bd:   rgba(77,184,150,0.28);
    --neg:      #d07a7a;
    --neg-bg:   rgba(208,122,122,0.10);
    --neg-bd:   rgba(208,122,122,0.28);
    --warn:     #c29445;
    --warn-bg:  rgba(194,148,69,0.10);
    --warn-bd:  rgba(194,148,69,0.28);
    --info:     #5b92d1;
    --info-bg:  rgba(91,146,209,0.10);
    --info-bd:  rgba(91,146,209,0.28);

    /* Typography */
    --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono: "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;

    /* Shape */
    --radius:   8px;
    --radius-lg:12px;
    --radius-sm:5px;

    /* Shadow / elevation */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.40);
    --shadow:    0 2px 8px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.30);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.30);

    /* Spacing scale — 4 px base */
    --sp-1: 4px;  --sp-2: 8px;   --sp-3: 12px;  --sp-4: 16px;
    --sp-5: 24px; --sp-6: 32px;  --sp-7: 48px;  --sp-8: 64px;
}

/* ── App canvas ────────────────────────────────────────────────────────── */
.stApp {
    background: var(--bg);
    font-family: var(--font);
    font-feature-settings: "tnum" 1;   /* tabular numerics everywhere */
}
section.main > div.block-container {
    padding-top: var(--sp-5);
    padding-bottom: var(--sp-8);
    max-width: 1560px;
}
body, .stApp, .stMarkdown, p, span, label, div {
    color: var(--text);
    font-family: var(--font);
}
.stMarkdown p, .stApp p {
    font-size: 0.875rem;
    line-height: 1.65;
    color: var(--text-2);
}

/* ── Sidebar — premium dark nav rail ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #080b10;
    border-right: 1px solid var(--line);
    min-width: 220px !important;
    max-width: 244px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 0.75rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}
/* App name at top */
section[data-testid="stSidebar"] h1 {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    color: var(--teal) !important;
    margin-bottom: var(--sp-3) !important;
    padding-bottom: var(--sp-3);
    border-bottom: 1px solid var(--line);
}
/* Nav group labels injected via .tx-nav-group markdown divs */
.tx-nav-group {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: var(--text-3);
    padding: var(--sp-4) var(--sp-3) var(--sp-1) var(--sp-3);
    margin-top: var(--sp-2);
}
/* Visual breaks between nav groups via nth-child spacing.
   The auto-generated nav list places each page as a sibling <a> element.
   We can't inject labels between them (Streamlit owns the DOM) so we use
   generous top-margin on specific link positions to simulate group breaks.
   Pages order: 0=Home, 1=Strategy, 2=Orders, 3=Charts, 4=Backtest,
   5=Perplexity, 6=Risk, 7=DayTrading, 8=Notifications, 9=Scanner,
   10=PnL, 11=India, 12=Schwab, 13=Webull */
section[data-testid="stSidebar"] ul li:nth-child(2) a,
section[data-testid="stSidebar"] ul li:nth-child(7) a,
section[data-testid="stSidebar"] ul li:nth-child(9) a,
section[data-testid="stSidebar"] ul li:nth-child(11) a {
    margin-top: var(--sp-4) !important;
    border-top: 1px solid var(--line);
    padding-top: var(--sp-3) !important;
}
/* Nav links */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
    border-radius: var(--radius);
    padding: 6px var(--sp-3);
    margin: 1px 0;
    color: var(--text-2);
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: background 0.10s ease, color 0.10s ease;
    display: flex;
    align-items: center;
    gap: var(--sp-2);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.04);
    color: var(--text);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: var(--teal-dim);
    color: var(--teal);
    font-weight: 600;
    box-shadow: inset 3px 0 0 var(--teal);
}

/* ── Trading workflow — risk / alert emphasis ─────────────────────────── */
/* Kill switch active: full-width danger banner */
.tx-kill-active {
    background: var(--neg-bg);
    border: 1px solid var(--neg-bd);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-4);
    color: var(--neg);
    font-weight: 600;
    font-size: 0.88rem;
    margin-bottom: var(--sp-3);
}
/* Open position card */
.tx-position-card {
    background: var(--panel-2);
    border: 1px solid var(--line-2);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-4);
    margin-bottom: var(--sp-2);
}
.tx-position-card.long  { border-left: 3px solid var(--pos); }
.tx-position-card.short { border-left: 3px solid var(--neg); }
/* Risk gauge label row */
.tx-gauge-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.8rem;
    color: var(--text-2);
    margin-bottom: 4px;
}
.tx-gauge-val { font-weight: 600; font-variant-numeric: tabular-nums; }
.tx-gauge-val.ok     { color: var(--teal); }
.tx-gauge-val.warn   { color: var(--warn); }
.tx-gauge-val.danger { color: var(--neg); }
/* Eligibility / signal state chips */
.tx-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: nowrap;
    border: 1px solid transparent;
}
.tx-chip.ready   { background: var(--pos-bg);  color: var(--pos);  border-color: var(--pos-bd); }
.tx-chip.watch   { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-bd); }
.tx-chip.blocked { background: var(--neg-bg);  color: var(--neg);  border-color: var(--neg-bd); }
.tx-chip.idle    { background: rgba(90,99,115,0.14); color: var(--text-3); border-color: rgba(90,99,115,0.28); }
/* Regime / direction display chip (wider, not a pill) */
.tx-regime {
    display: inline-block;
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.tx-regime.bull  { background: var(--pos-bg);  color: var(--pos);  border: 1px solid var(--pos-bd); }
.tx-regime.bear  { background: var(--neg-bg);  color: var(--neg);  border: 1px solid var(--neg-bd); }
.tx-regime.chop  { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-bd); }
.tx-regime.pre   { background: var(--info-bg); color: var(--info); border: 1px solid var(--info-bd); }
.tx-regime.unkn  { background: rgba(90,99,115,0.12); color: var(--text-3); border: 1px solid rgba(90,99,115,0.24); }

/* ── Headings — strict 3-step hierarchy ──────────────────────────────── */
h1 {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin: 0 0 var(--sp-1) 0;
    color: var(--text);
}
h2 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    line-height: 1.3;
    margin: var(--sp-6) 0 var(--sp-3) 0;
    color: var(--text);
    padding-left: var(--sp-2);
    border-left: 2px solid var(--gold);
}
h3 {
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.3;
    letter-spacing: 0.005em;
    color: var(--text);
    margin: var(--sp-4) 0 var(--sp-2) 0;
}
h4 {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-3);
    margin: var(--sp-3) 0 var(--sp-2) 0;
}

/* ── Page header shell (rendered by page_header() helper) ─────────────── */
.tx-page-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--sp-4);
    margin-bottom: var(--sp-2);
    padding-bottom: var(--sp-3);
    border-bottom: 1px solid var(--line);
}
.tx-page-title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--text);
    line-height: 1.15;
}
.tx-page-sub {
    font-size: 0.8rem;
    color: var(--text-3);
    margin-top: 3px;
    line-height: 1.4;
}

/* ── Metric / KPI cards ──────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: var(--sp-3) var(--sp-4);
    box-shadow: var(--shadow-sm);
    transition: border-color 0.12s ease, box-shadow 0.12s ease;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
}
div[data-testid="stMetric"]:hover {
    border-color: var(--line-2);
    box-shadow: var(--shadow);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--text-3);
    font-weight: 600;
    margin-bottom: var(--sp-1);
}
div[data-testid="stMetricValue"] {
    font-family: var(--font);
    font-variant-numeric: tabular-nums;
    font-size: 1.5rem;
    font-weight: 650;
    line-height: 1.1;
    letter-spacing: -0.015em;
    color: var(--text);
}
div[data-testid="stMetricDelta"] {
    font-variant-numeric: tabular-nums;
    font-size: 0.75rem;
    margin-top: var(--sp-1);
}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr {
    margin: var(--sp-5) 0 var(--sp-4) 0;
    border: 0;
    border-top: 1px solid var(--line);
}

/* ── Tabs — polished terminal style ──────────────────────────────────── */
div[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--line);
    gap: 0;
    background: transparent;
}
button[data-baseweb="tab"] {
    font-family: var(--font);
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.015em;
    padding: 0.55rem 1.1rem;
    color: var(--text-3);
    background: transparent;
    border-bottom: 2px solid transparent;
    transition: color 0.12s ease;
}
button[data-baseweb="tab"]:hover { color: var(--text-2); }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--text);
    font-weight: 600;
}
div[data-baseweb="tab-highlight"] {
    background: var(--teal);
    height: 2px;
    border-radius: 2px 2px 0 0;
}
div[data-baseweb="tab-panel"] { padding-top: var(--sp-4); }

/* ── Buttons ─────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button,
div[data-testid="stLinkButton"] > a,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: var(--radius);
    border: 1px solid var(--line-2);
    background: var(--panel-2);
    color: var(--text-2);
    font-family: var(--font);
    font-weight: 500;
    font-size: 0.83rem;
    letter-spacing: 0.015em;
    min-height: 36px;
    padding: 0 var(--sp-4);
    transition: background 0.10s ease, border-color 0.10s ease, color 0.10s ease;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stLinkButton"] > a:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--panel-3);
    border-color: var(--line-3);
    color: var(--text);
}
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: var(--teal);
    border-color: var(--teal);
    color: #061210;
    font-weight: 600;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    background: #43ccbc;
    border-color: #43ccbc;
}
/* Destructive / danger button — opt-in via class on the markdown container */
.tx-btn-danger div[data-testid="stButton"] > button {
    border-color: var(--neg-bd);
    color: var(--neg);
}
.tx-btn-danger div[data-testid="stButton"] > button:hover {
    background: var(--neg-bg);
}

/* ── Inputs / selects ────────────────────────────────────────────────── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: var(--panel) !important;
    border-color: var(--line-2) !important;
    border-radius: var(--radius) !important;
    min-height: 36px;
    font-family: var(--font) !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
    transition: border-color 0.10s ease;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 2px var(--teal-glow) !important;
    outline: none;
}
/* Widget labels */
div[data-testid="stWidgetLabel"] label,
label[data-testid="stWidgetLabel"] {
    font-size: 0.77rem;
    font-weight: 500;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: var(--sp-1);
}
/* Select dropdown options */
ul[data-testid="stSelectboxVirtualDropdown"] li,
div[data-baseweb="popover"] li {
    font-size: 0.85rem;
    color: var(--text);
}

/* ── Radio / toggles ─────────────────────────────────────────────────── */
div[role="radiogroup"] label {
    color: var(--text-2);
    font-size: 0.84rem;
    font-weight: 500;
}
div[data-testid="stRadio"] > div > div:hover label { color: var(--text); }

/* ── Tables — premium data grid ──────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    overflow: hidden;
    margin: var(--sp-3) 0 var(--sp-5) 0;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] th {
    font-family: var(--font) !important;
    font-variant-numeric: tabular-nums;
    font-size: 0.82rem;
    line-height: 1.4;
}
div[data-testid="stDataFrame"] th {
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-size: 0.67rem;
    color: var(--text-3) !important;
    background: var(--panel-2) !important;
    font-weight: 700;
    border-bottom: 1px solid var(--line-2) !important;
}
div[data-testid="stDataFrame"] td { color: var(--text); }
div[data-testid="stDataFrame"] [role="row"]:hover { background: rgba(255,255,255,0.025) !important; }
/* Zebra striping: subtle alternate rows */
div[data-testid="stDataFrame"] [role="row"]:nth-child(even) {
    background: rgba(255,255,255,0.012);
}

/* ── Progress bars (risk gauges) ─────────────────────────────────────── */
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--teal), #2aa89c);
    border-radius: 4px;
}
div[data-testid="stProgress"] > div > div {
    background: var(--panel-2);
    border-radius: 4px;
}

/* ── Alert / banner / notice panels ──────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: var(--radius);
    border-width: 1px;
    border-style: solid;
    padding: var(--sp-3) var(--sp-4);
    font-size: 0.84rem;
}
/* Info */
div[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stAlert"] .stAlert-info {
    background: var(--info-bg) !important;
    border-color: var(--info-bd) !important;
    color: var(--info) !important;
}
/* Warning */
div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
div[data-testid="stAlert"] .stAlert-warning {
    background: var(--warn-bg) !important;
    border-color: var(--warn-bd) !important;
}
/* Error */
div[data-testid="stAlert"][data-baseweb="notification"][kind="error"],
div[data-testid="stAlert"] .stAlert-error {
    background: var(--neg-bg) !important;
    border-color: var(--neg-bd) !important;
}
/* Success */
div[data-testid="stAlert"][data-baseweb="notification"][kind="success"],
div[data-testid="stAlert"] .stAlert-success {
    background: var(--pos-bg) !important;
    border-color: var(--pos-bd) !important;
}
div[data-testid="stAlert"] p {
    font-size: 0.84rem;
    line-height: 1.5;
}

/* ── Expanders ───────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    background: var(--panel) !important;
    margin-bottom: var(--sp-2);
}
div[data-testid="stExpander"] summary {
    font-size: 0.84rem;
    font-weight: 500;
    color: var(--text-2);
    padding: var(--sp-3) var(--sp-4);
}
div[data-testid="stExpander"] summary:hover { color: var(--text); }
div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    padding: 0 var(--sp-4) var(--sp-3) var(--sp-4);
    border-top: 1px solid var(--line);
}

/* ── Containers with border ──────────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    background: var(--panel) !important;
    padding: var(--sp-4) !important;
    box-shadow: var(--shadow-sm);
}

/* ── Checkboxes / toggles ─────────────────────────────────────────────  */
div[data-testid="stCheckbox"] label,
div[data-testid="stToggle"] label {
    font-size: 0.84rem;
    color: var(--text-2);
    font-weight: 500;
}

/* ── Slider ───────────────────────────────────────────────────────────── */
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--teal) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
div[data-testid="stSpinner"] > div { border-top-color: var(--teal) !important; }

/* ── Pills (status badges) ───────────────────────────────────────────── */
.tx-pill {
    display: inline-block;
    padding: 2px var(--sp-2);
    border-radius: var(--radius-sm);
    font-family: var(--font);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    line-height: 1.5;
    border: 1px solid transparent;
    white-space: nowrap;
}
.tx-pill.green  { background: var(--pos-bg);  color: var(--pos);  border-color: var(--pos-bd); }
.tx-pill.red    { background: var(--neg-bg);  color: var(--neg);  border-color: var(--neg-bd); }
.tx-pill.amber  { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-bd); }
.tx-pill.grey   { background: rgba(90,99,115,0.15); color: var(--text-2); border-color: rgba(90,99,115,0.3); }
.tx-pill.blue   { background: var(--info-bg); color: var(--info); border-color: var(--info-bd); }
.tx-pill.teal   { background: var(--teal-dim); color: var(--teal); border-color: rgba(55,184,170,0.3); }

/* ── Stat band (top command bar) ─────────────────────────────────────── */
.tx-statband {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--sp-5);
    padding: var(--sp-3) var(--sp-5);
    margin: var(--sp-3) 0 var(--sp-4) 0;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}
.tx-stat { display: inline-flex; align-items: center; gap: var(--sp-2); }
.tx-stat-label {
    color: var(--text-3);
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
}

/* Legacy: keep tx-statusbar as alias */
.tx-statusbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--sp-5);
    padding: var(--sp-3) var(--sp-5);
    margin: var(--sp-3) 0 var(--sp-4) 0;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}

/* ── Section subtitles ───────────────────────────────────────────────── */
.tx-subtle {
    color: var(--text-3);
    font-size: 0.78rem;
    line-height: 1.5;
    margin: calc(-1 * var(--sp-1)) 0 var(--sp-3) var(--sp-2);
}

/* ── Numeric display class — larger tabular numbers ─────────────────── */
.tx-num {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum" 1;
    letter-spacing: -0.01em;
}

/* ── KPI mini-label ──────────────────────────────────────────────────── */
.tx-kpi-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
    color: var(--text-3);
    margin-bottom: 2px;
}
.tx-kpi-value {
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    line-height: 1.1;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.tx-kpi-delta {
    font-size: 0.73rem;
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
}
.tx-kpi-delta.pos { color: var(--pos); }
.tx-kpi-delta.neg { color: var(--neg); }
.tx-kpi-delta.muted { color: var(--text-3); }

/* ── Result / info card ──────────────────────────────────────────────── */
.tx-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: var(--sp-4) var(--sp-5);
    box-shadow: var(--shadow-sm);
    margin-bottom: var(--sp-4);
}
.tx-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-3);
    margin-bottom: var(--sp-2);
    padding-bottom: var(--sp-2);
    border-bottom: 1px solid var(--line);
}

/* ── Empty state ─────────────────────────────────────────────────────── */
.tx-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--sp-8) var(--sp-5);
    text-align: center;
    background: var(--panel);
    border: 1px dashed var(--line-2);
    border-radius: var(--radius-lg);
    margin: var(--sp-4) 0;
}
.tx-empty-icon { font-size: 2rem; margin-bottom: var(--sp-3); opacity: 0.5; }
.tx-empty-title { font-size: 0.92rem; font-weight: 600; color: var(--text-2); margin-bottom: var(--sp-2); }
.tx-empty-body  { font-size: 0.8rem; color: var(--text-3); max-width: 340px; line-height: 1.55; }

/* ── Filter toolbar ─────────────────────────────────────────────────── */
.tx-filterbar {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    padding: var(--sp-3) var(--sp-4);
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    margin-bottom: var(--sp-4);
    flex-wrap: wrap;
}

/* ── Captions / meta text ────────────────────────────────────────────── */
div[data-testid="stCaptionContainer"],
.stCaption,
.element-container small {
    color: var(--text-3);
    font-size: 0.76rem;
    line-height: 1.5;
}

/* ── Code blocks ─────────────────────────────────────────────────────── */
code, pre {
    font-family: var(--mono) !important;
    font-size: 0.82rem;
    background: var(--panel-2);
    border-radius: var(--radius-sm);
}

/* ── Inline colored text helpers ─────────────────────────────────────── */
.tx-pos  { color: var(--pos)  !important; }
.tx-neg  { color: var(--neg)  !important; }
.tx-warn { color: var(--warn) !important; }
.tx-info { color: var(--info) !important; }
.tx-muted{ color: var(--text-3) !important; }
.tx-accent{ color: var(--teal) !important; }

/* ── Market clock bar (compact status line) ──────────────────────────── */
.tx-clock {
    display: inline-flex;
    gap: var(--sp-4);
    font-size: 0.76rem;
    color: var(--text-3);
    padding: var(--sp-1) 0 var(--sp-3) 0;
}
</style>
"""


def nav_group(label: str) -> None:
    """Render a nav section label inside the sidebar (call from any page top-level).

    Usage (inside ``with st.sidebar:`` or at top of page before content)::

        nav_group("TRADING")
    """
    st.sidebar.markdown(
        f"<div class='tx-nav-group'>{label}</div>",
        unsafe_allow_html=True,
    )


def apply_theme(page_title: str, *, page_icon: str | None = None) -> None:
    """Set page config and inject the dashboard CSS.

    Call once per page, before any other Streamlit output.
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
    """Render a section header with optional subtitle."""
    tag = {2: "##", 3: "###", 4: "####"}.get(level, "##")
    st.markdown(f"{tag} {title}")
    if subtitle:
        st.markdown(f"<div class='tx-subtle'>{subtitle}</div>", unsafe_allow_html=True)


def divider() -> None:
    """Quiet hairline divider."""
    st.markdown("<hr/>", unsafe_allow_html=True)


def kpi_row(
    items: Sequence[tuple[str, str]] | Sequence[tuple[str, str, str | None]],
    *,
    gap: str = "small",
) -> None:
    """Render a row of KPI metric cards.

    Accepts ``(label, value)`` or ``(label, value, delta)`` tuples.
    """
    if not items:
        return
    cols = st.columns(len(items), gap=gap)
    for col, item in zip(cols, items):
        if len(item) == 2:
            label, value = item
            col.metric(label, value)
        else:
            label, value, delta = item
            col.metric(label, value, delta=delta)


def pill(text: str, color: str = "grey") -> str:
    """Return an HTML status pill for use with ``st.markdown(..., unsafe_allow_html=True)``.

    Colors: green, red, amber, grey, blue, teal.
    """
    return f"<span class='tx-pill {color}'>{text}</span>"


def status_row(items: Iterable[tuple[str, str, str]]) -> None:
    """Render the top stat/status band from ``(label, text, color)`` tuples."""
    parts = []
    for label, text, color in items:
        parts.append(
            f"<span class='tx-stat'>"
            f"<span class='tx-stat-label'>{label}</span>{pill(text, color)}"
            f"</span>"
        )
    st.markdown(
        "<div class='tx-statband'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def empty_state(
    title: str,
    body: str = "",
    icon: str = "📭",
) -> None:
    """Render a centered empty-state panel."""
    st.markdown(
        f"<div class='tx-empty'>"
        f"<div class='tx-empty-icon'>{icon}</div>"
        f"<div class='tx-empty-title'>{title}</div>"
        f"<div class='tx-empty-body'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def card(title: str = "", *, border: bool = True) -> "st.delta_generator.DeltaGenerator":  # type: ignore[name-defined]
    """Return a bordered container with an optional card-title label.

    Usage::
        with card("Open Positions"):
            st.dataframe(df)
    """
    if title:
        st.markdown(
            f"<div class='tx-card-title'>{title}</div>",
            unsafe_allow_html=True,
        )
    return st.container(border=border)


# ── Formatters ───────────────────────────────────────────────────────────────

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
    """Format a monetary value. ``None`` or NaN → em-dash."""
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dash
    if v != v:
        return dash
    return f"{currency}{v:,.{decimals}f}"


def pct(value: float | None, *, decimals: int = 1, dash: str = "—") -> str:
    """Format a percentage. Accepts fractional (0.15) or whole (15.0)."""
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
    """Plain numeric formatter. ``None`` or NaN → em-dash."""
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
