"""Notification log — every BUY/SELL signal fired on an assigned symbol."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, divider, empty_state, pill
from _components import page_header, stat_band

import streamlit as st

apply_theme("Notifications")

from _sidebar import render_sidebar
render_sidebar()

# ── Load counts for the stat band ─────────────────────────────────────────────
try:
    _cnt = api.notifications_unread_count() or {}
    _unread = _cnt.get("unread", 0)
except Exception:
    _unread = 0

page_header(
    "Notifications",
    subtitle="Alerts fired on active assignment signals — scheduler, scanner, and other sources.",
    badge=f"{_unread} unread" if _unread else "all read",
    badge_color="red" if _unread else "green",
)

# ── Controls ───────────────────────────────────────────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
unread_only = ctrl1.checkbox("Unread only", value=False)
limit       = ctrl2.selectbox("Show", [25, 50, 100, 200], index=1)
ctrl3.write("")
if ctrl3.button("Mark all read", use_container_width=True):
    try:
        res = api.notifications_mark_all_read()
        st.success(f"Marked {res.get('marked_read', 0)} notifications as read.")
        st.rerun()
    except Exception as exc:
        st.error(f"Failed: {exc}")

divider()

# ── List ───────────────────────────────────────────────────────────────────────
# Pull a wider page so each tab still has enough rows after filtering — buys far
# outnumber sells, so a 50-row page would otherwise bury every sell.
try:
    items = api.notifications_list(limit=max(limit * 4, 200), unread_only=unread_only)
except Exception as exc:
    st.error(f"Cannot load notifications: {exc}", icon="⚠️")
    st.stop()

if not items:
    empty_state(
        "No notifications yet",
        "They appear here once an assigned symbol fires a BUY or SELL signal from the scheduler or scanner.",
        icon="🔔",
    )
    st.stop()


def _render(rows):
    """Render a list of notification cards (already filtered for the tab)."""
    rows = rows[:limit]
    if not rows:
        st.caption("No notifications in this category yet.")
        return
    st.caption(f"{len(rows)} notification(s)")
    for n in rows:
        unread    = n.get("read_at") is None
        direction = (n.get("direction") or "").upper()
        dir_icon  = {"BUY": "🟢", "SELL": "🔴"}.get(direction, "🔵")
        new_badge = f"&nbsp;{pill('NEW', 'amber')}" if unread else ""
        src       = n.get("source") or "—"
        strat     = n.get("strategy") or "—"
        ts        = (n.get("created_at") or "")[:19].replace("T", " ")

        with st.container(border=True):
            left, right = st.columns([9, 2])
            with left:
                st.markdown(
                    f"**{dir_icon} {n['title']}**{new_badge}",
                    unsafe_allow_html=True,
                )
                st.caption(f"{ts} · {src} · {strat}")
                if n.get("body"):
                    st.markdown(n["body"])
            with right:
                if unread and st.button("Mark read", key=f"mr_{n['id']}", use_container_width=True):
                    try:
                        api.notifications_mark_read(n["id"])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
                if st.button("Delete", key=f"del_{n['id']}", use_container_width=True):
                    try:
                        api.notifications_delete(n["id"])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")


# Split by source/direction. Buy & Sell are scheduler/scanner auto-trade signals;
# M1 is the (un-gated) advisory feed, source="m1".
def _is_m1(n):
    return (n.get("source") or "").lower() == "m1"

m1_rows  = [n for n in items if _is_m1(n)]
buy_rows  = [n for n in items if not _is_m1(n) and (n.get("direction") or "").upper() == "BUY"]
sell_rows = [n for n in items if not _is_m1(n) and (n.get("direction") or "").upper() == "SELL"]

tab_buy, tab_sell, tab_m1 = st.tabs([
    f"🟢 Buy ({len(buy_rows)})",
    f"🔴 Sell ({len(sell_rows)})",
    f"📊 M1 ({len(m1_rows)})",
])
with tab_buy:
    _render(buy_rows)
with tab_sell:
    _render(sell_rows)
with tab_m1:
    _render(m1_rows)
