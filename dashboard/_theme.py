"""Shared visual layer for the Streamlit dashboard.

Every page calls `apply_theme()` once near the top to install the CSS overrides,
then composes pages out of the helpers below (`section`, `kpi_row`, `pill`,
`money`, `pct`). The goal is one consistent visual language across pages so the
UI feels less like a stack of unrelated dashboards and more like a professional
trading terminal.

Design language: "Midnight" — a modern fintech dark theme on a blue-black
canvas with an electric violet→cyan accent gradient, glassy elevated surfaces,
and soft accent glows. Token NAMES are kept stable (e.g. ``--teal`` is still the
primary-accent token) so all pages reskin automatically; only the VALUES and
component styling change.

Design system token reference (mirrored in .streamlit/config.toml):
  --bg        #0b0c14   blue-black canvas
  --panel     #14162280 primary surface (glassy cards, containers)
  --panel-2   #191c2e   raised surface (hover, headers, nav active)
  --panel-3   #21253a   highest elevation (dropdowns, tooltips)
  --line      rgba(255,255,255,0.07)   hairline border
  --line-2    rgba(255,255,255,0.12)   medium border (hover, focus)
  --line-3    rgba(255,255,255,0.20)   strong border (active)
  --text      #eef0f6   primary near-white
  --text-2    #9aa3bd   secondary / labels
  --text-3    #626a86   tertiary / helpers / metadata
  --teal      #7c5cff   PRIMARY accent (electric violet) — name kept for compat
  --cyan      #22d3ee   secondary accent (cyan, end of gradient)
  --accent-grad linear violet→cyan   buttons, highlights, active bars
  --teal-dim  rgba(124,92,255,0.14)   accent background tint
  --gold      #f0b429   section accent rule only (warm amber)
  --pos       #34d399   gains (emerald)
  --pos-bg    rgba(52,211,153,0.11)
  --neg       #fb7185   losses (rose)
  --neg-bg    rgba(251,113,133,0.11)
  --warn      #fbbf24   warnings / amber
  --warn-bg   rgba(251,191,36,0.11)
  --info      #38bdf8   informational / sky
  --info-bg   rgba(56,189,248,0.11)
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Design tokens — "Midnight" violet/cyan system ───────────────────── */
:root {
    /* Canvas */
    --bg:       #0b0c14;
    --panel:    rgba(20,22,34,0.72);
    --panel-solid: #141622;
    --panel-2:  #191c2e;
    --panel-3:  #21253a;

    /* Borders */
    --line:     rgba(255,255,255,0.07);
    --line-2:   rgba(255,255,255,0.12);
    --line-3:   rgba(255,255,255,0.20);

    /* Text */
    --text:     #eef0f6;
    --text-2:   #9aa3bd;
    --text-3:   #626a86;

    /* Brand / accent — violet primary, cyan secondary, gradient highlight */
    --teal:      #7c5cff;   /* name kept for back-compat; now electric violet */
    --teal-2:    #9d83ff;
    --cyan:      #22d3ee;
    --teal-dim:  rgba(124,92,255,0.14);
    --teal-glow: rgba(124,92,255,0.30);
    --cyan-glow: rgba(34,211,238,0.28);
    --accent-grad: linear-gradient(135deg, #7c5cff 0%, #22d3ee 100%);
    --accent-grad-soft: linear-gradient(135deg, rgba(124,92,255,0.18), rgba(34,211,238,0.14));
    --gold:      #f0b429;

    /* Semantic */
    --pos:      #34d399;
    --pos-bg:   rgba(52,211,153,0.11);
    --pos-bd:   rgba(52,211,153,0.30);
    --neg:      #fb7185;
    --neg-bg:   rgba(251,113,133,0.11);
    --neg-bd:   rgba(251,113,133,0.30);
    --warn:     #fbbf24;
    --warn-bg:  rgba(251,191,36,0.11);
    --warn-bd:  rgba(251,191,36,0.30);
    --info:     #38bdf8;
    --info-bg:  rgba(56,189,248,0.11);
    --info-bd:  rgba(56,189,248,0.30);

    /* Typography */
    --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono: "JetBrains Mono", "SF Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;

    /* Shape — slightly rounder for the modern feel */
    --radius:   10px;
    --radius-lg:16px;
    --radius-sm:6px;

    /* Shadow / elevation */
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.45);
    --shadow:    0 4px 16px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35);
    --shadow-lg: 0 12px 40px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.35);
    --shadow-glow: 0 0 0 1px rgba(124,92,255,0.20), 0 8px 28px rgba(124,92,255,0.14);

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
/* Ambient aurora — two soft radial accent washes fixed behind the content,
   gives the flat dark canvas depth without distracting from data. */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(900px 520px at 12% -8%,  rgba(124,92,255,0.16), transparent 60%),
        radial-gradient(820px 480px at 92% 4%,   rgba(34,211,238,0.10), transparent 62%),
        radial-gradient(700px 600px at 70% 108%, rgba(124,92,255,0.07), transparent 60%);
}
section.main, section[data-testid="stSidebar"] { position: relative; z-index: 1; }
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

/* ── Sidebar — premium dark nav rail (st.navigation) ─────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0e1a 0%, #090a12 100%);
    border-right: 1px solid var(--line);
    min-width: 236px !important;
    max-width: 262px !important;
    box-shadow: inset -1px 0 0 rgba(124,92,255,0.06);
}
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] > div > div {
    padding-top: var(--sp-3);
    padding-left: var(--sp-3);
    padding-right: var(--sp-3);
}
/* Collapse Streamlit's own top padding so the brand sits flush */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { padding-bottom: 0; }

/* Brand block ------------------------------------------------------------ */
.tx-brand {
    padding: var(--sp-2) var(--sp-2) var(--sp-3) var(--sp-2);
    margin-bottom: var(--sp-2);
    border-bottom: 1px solid var(--line);
}
.tx-brand-row { display: flex; align-items: center; gap: var(--sp-2); }
.tx-brand-mark {
    font-size: 1.05rem;
    line-height: 1;
    background: var(--accent-grad);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 8px rgba(124,92,255,0.45));
}
.tx-brand-name {
    font-size: 0.98rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    color: var(--text);
}
.tx-brand-name-2 {
    background: var(--accent-grad);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.tx-brand-clock {
    display: flex; gap: var(--sp-3);
    margin-top: var(--sp-2);
    font-size: 0.64rem;
    font-variant-numeric: tabular-nums;
    color: var(--text-3);
}
.tx-brand-clock-item { display: inline-flex; align-items: center; gap: 4px; }
.tx-brand-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.tx-brand-dot.open   { background: var(--pos); box-shadow: 0 0 6px var(--pos); }
.tx-brand-dot.closed { background: var(--text-3); }

/* Section headers (st.navigation group titles) -------------------------- */
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {
    font-size: 0.6rem !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: var(--text-3) !important;
    padding: var(--sp-4) var(--sp-2) var(--sp-1) var(--sp-2);
    margin: 0;
}
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"]:first-of-type {
    padding-top: var(--sp-2);
}

/* Nav links ------------------------------------------------------------- */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
    border-radius: var(--radius);
    padding: 7px 10px;
    margin: 2px 0;
    color: var(--text-2);
    font-size: 0.84rem;
    font-weight: 500;
    letter-spacing: 0.005em;
    transition: background 0.12s ease, color 0.12s ease, box-shadow 0.12s ease;
    display: flex;
    align-items: center;
    gap: 10px;
    position: relative;
}
/* Icon sizing/tint */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] span[data-testid="stIconMaterial"] {
    font-size: 1.05rem !important;
    color: var(--text-3);
    transition: color 0.12s ease;
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {
    background: rgba(255,255,255,0.045);
    color: var(--text);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover span[data-testid="stIconMaterial"] {
    color: var(--text-2);
}
/* Active link — gradient tint, accent bar, glowing icon */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: var(--accent-grad-soft);
    color: #d6cdff;
    font-weight: 600;
    box-shadow: inset 0 0 0 1px rgba(124,92,255,0.18);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"]::before {
    content: "";
    position: absolute;
    left: -3px; top: 7px; bottom: 7px;
    width: 3px;
    border-radius: 3px;
    background: var(--accent-grad);
    box-shadow: 0 0 8px rgba(124,92,255,0.6);
}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] span[data-testid="stIconMaterial"] {
    color: var(--cyan);
}

/* Sidebar footer / utilities -------------------------------------------- */
.tx-sb-divider {
    height: 1px;
    background: var(--line);
    margin: var(--sp-4) 0 var(--sp-3) 0;
}
.tx-sb-foot {
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-3);
    text-align: center;
    margin-top: var(--sp-3);
    opacity: 0.7;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    width: 100%;
    background: rgba(255,255,255,0.03);
    border-color: var(--line);
    font-size: 0.78rem;
    min-height: 32px;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    border-color: rgba(124,92,255,0.4);
    color: var(--text);
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
    padding-left: var(--sp-3);
    border-left: 3px solid transparent;
    border-image: var(--accent-grad) 1;
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
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #ffffff 0%, #c8bcff 55%, #8fe7f5 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
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
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: var(--sp-4) var(--sp-4);
    box-shadow: var(--shadow-sm);
    transition: border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-grad);
    opacity: 0.55;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--line-2);
    box-shadow: var(--shadow-glow);
    transform: translateY(-1px);
}
div[data-testid="stMetric"]:hover::before { opacity: 1; }
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
    background: var(--accent-grad);
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
    background: var(--accent-grad);
    border: 1px solid transparent;
    color: #0a0b14;
    font-weight: 700;
    box-shadow: 0 4px 16px rgba(124,92,255,0.30);
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 6px 22px rgba(124,92,255,0.42), 0 0 0 1px rgba(34,211,238,0.30);
    transform: translateY(-1px);
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
    background: linear-gradient(90deg, var(--teal), var(--cyan));
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
    border-radius: var(--radius-lg) !important;
    background: var(--panel) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
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
.tx-pill.teal   { background: var(--teal-dim); color: var(--teal-2); border-color: rgba(124,92,255,0.34); }

/* ── Stat band (top command bar) ─────────────────────────────────────── */
.tx-statband {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--sp-5);
    padding: var(--sp-3) var(--sp-5);
    margin: var(--sp-3) 0 var(--sp-4) 0;
    background: var(--panel);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
}
.tx-statband::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent-grad);
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

/* ── Portfolio hero band (Home cockpit) ──────────────────────────────── */
.tx-hero {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--line);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin: var(--sp-2) 0 var(--sp-5) 0;
    box-shadow: var(--shadow-sm);
    position: relative;
}
.tx-hero::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-grad);
    z-index: 1;
}
.tx-hero-cell {
    background: var(--panel-solid);
    padding: var(--sp-4) var(--sp-5);
    transition: background 0.15s ease;
}
.tx-hero-cell:hover { background: var(--panel-2); }
.tx-hero-label {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    color: var(--text-3);
    margin-bottom: var(--sp-2);
}
.tx-hero-value {
    font-size: 1.85rem;
    font-weight: 750;
    letter-spacing: -0.02em;
    line-height: 1.05;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.tx-hero-sub {
    font-size: 0.72rem;
    color: var(--text-3);
    margin-top: var(--sp-1);
    font-variant-numeric: tabular-nums;
}
.tx-hero-sub.tx-pos { color: var(--pos); }
.tx-hero-sub.tx-neg { color: var(--neg); }
.tx-hero-sub.tx-muted { color: var(--text-3); }
@media (max-width: 1100px) {
    .tx-hero { grid-template-columns: repeat(2, 1fr); }
}

/* ── Result / info card ──────────────────────────────────────────────── */
.tx-card {
    background: var(--panel);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
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


def brand_header() -> None:
    """Render the brand block at the very top of the sidebar (above the nav).

    Called by the ``streamlit_app.py`` entrypoint. Shows a gradient logo mark,
    the product name, and a live market-session clock.
    """
    bits = []
    for label, open_t, close_t, tz_name in _SESSIONS:
        is_open, clock = _session_status(open_t, close_t, tz_name)
        cls = "open" if is_open else "closed"
        bits.append(
            f"<span class='tx-brand-clock-item'>"
            f"<span class='tx-brand-dot {cls}'></span>{label} {clock}"
            f"</span>"
        )
    st.markdown(
        "<div class='tx-brand'>"
        "  <div class='tx-brand-row'>"
        "    <span class='tx-brand-mark'>◆</span>"
        "    <span class='tx-brand-name'>TRADING<span class='tx-brand-name-2'>SYS</span></span>"
        "  </div>"
        "  <div class='tx-brand-clock'>" + "".join(bits) + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    """Render utility controls pinned below the nav (restart button + version)."""
    st.markdown("<div class='tx-sb-divider'></div>", unsafe_allow_html=True)
    try:
        from _server_controls import render_restart_button
        render_restart_button(key="sidebar_restart_main")
    except Exception:
        pass
    st.markdown(
        "<div class='tx-sb-foot'>v2 · Midnight UI</div>",
        unsafe_allow_html=True,
    )


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
    """Inject the dashboard CSS (and set page config if not already set).

    Call once per page, before any other Streamlit output. When the app is
    driven by the ``streamlit_app.py`` entrypoint (``st.navigation``), page
    config is owned by the entrypoint and ``set_page_config`` here is a no-op —
    the second call would raise, so it's guarded.
    """
    try:
        st.set_page_config(
            page_title=page_title,
            page_icon=page_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Already configured by the st.navigation entrypoint — fine.
        pass
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
