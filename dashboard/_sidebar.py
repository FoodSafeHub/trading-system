"""Shared sidebar rendered on every page.

Call render_sidebar() at the top of each page (after apply_theme) to get
consistent navigation labels, the Restart API button, and the data-source
status pill on every page — not just Home.
"""
from __future__ import annotations

import streamlit as st


def render_sidebar() -> None:
    """Inject nav-group labels and utilities into the sidebar."""
    from _theme import nav_group

    with st.sidebar:
        nav_group("Overview")
        # Home, PnL
        nav_group("Research")
        # Strategy, Charts, Backtest, Perplexity
        nav_group("Trading")
        # DayTrading, Scanner, India
        nav_group("Risk & Ops")
        # Risk, Notifications, Schwab, Webull, Orders

        st.markdown("---")
        try:
            from _server_controls import render_restart_button
            render_restart_button(key=f"sidebar_restart_{id(render_sidebar)}")
        except Exception:
            pass
