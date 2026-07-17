"""Dashboard entrypoint — owns navigation and page config.

Run with::

    streamlit run dashboard/streamlit_app.py

This replaces Streamlit's auto-generated multipage nav (which can't group pages)
with a real ``st.navigation`` rail: titled sections, per-page icons, a brand
header, market clock, and utility controls. Each page script still calls
``apply_theme()`` / ``render_sidebar()`` near its top; those are now guarded /
no-ops so the 14 page files did not need editing.
"""
from __future__ import annotations

import os
import sys

import streamlit as st

# Pages import sibling modules (api, _theme, …) by bare name.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _theme import _CSS, brand_header, sidebar_footer  # noqa: E402

# ── Page config (owned here, once) ───────────────────────────────────────────
st.set_page_config(
    page_title="Trading System",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_CSS, unsafe_allow_html=True)


def _p(path: str, title: str, icon: str, *, default: bool = False) -> "st.Page":
    return st.Page(path, title=title, icon=icon, default=default)


# ── Navigation: real grouped sections with icons ─────────────────────────────
nav = {
    "Overview": [
        _p("Home.py",            "Cockpit",      ":material/dashboard:", default=True),
        _p("pages/10_PnL.py",    "P&L",          ":material/payments:"),
    ],
    "Research": [
        _p("pages/1_Strategy.py",   "Strategy",   ":material/insights:"),
        _p("pages/3_Charts.py",     "Charts",     ":material/candlestick_chart:"),
        _p("pages/4_Backtest.py",   "Backtest",   ":material/science:"),
        _p("pages/5_Perplexity.py", "India Swing", ":material/auto_awesome:"),
        _p("pages/15_Ratings.py",   "Analyst Ratings", ":material/reviews:"),
    ],
    "Trading": [
        _p("pages/7_DayTrading.py",    "Day Trading", ":material/bolt:"),
        _p("pages/9_Scanner.py",       "Scanner",     ":material/radar:"),
        _p("pages/11_India.py",        "India",       ":material/public:"),
        _p("pages/14_M1_Portfolio.py", "M1 Portfolio",":material/account_balance:"),
    ],
    "Risk & Ops": [
        _p("pages/6_Risk.py",          "Risk",          ":material/shield:"),
        _p("pages/2_Orders.py",        "Orders",        ":material/receipt_long:"),
        _p("pages/8_Notifications.py", "Notifications", ":material/notifications:"),
        _p("pages/12_Schwab.py",       "Schwab",        ":material/account_balance_wallet:"),
        _p("pages/13_Webull.py",       "Webull",        ":material/account_balance_wallet:"),
    ],
}

# ── Sidebar chrome: brand header above the nav, utilities below ──────────────
with st.sidebar:
    brand_header()

page = st.navigation(nav, position="sidebar")

with st.sidebar:
    sidebar_footer()

page.run()
