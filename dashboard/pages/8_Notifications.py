"""Notification log — every BUY/SELL signal fired on an assigned symbol."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme

import pandas as pd
import streamlit as st

apply_theme("Notifications")
st.title("Notifications")
st.caption(
    "Alerts fired whenever an active assignment produces a BUY/SELL signal — from the scheduler, "
    "scanner, or any future source. Only symbols in your assignments table generate notifications."
)

# ── Controls ────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    unread_only = st.checkbox("Unread only", value=False)
with c2:
    limit = st.selectbox("Show", [25, 50, 100, 200], index=1)
with c3:
    st.write("")
    if st.button("Mark all read", use_container_width=True):
        try:
            res = api.notifications_mark_all_read()
            st.success(f"Marked {res.get('marked_read', 0)} notifications as read.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed: {exc}")

# ── List ────────────────────────────────────────────────────────────────
try:
    items = api.notifications_list(limit=limit, unread_only=unread_only)
except Exception as exc:
    st.error(f"Cannot load notifications: {exc}")
    st.stop()

if not items:
    st.info("No notifications yet. They'll show up here once an assigned symbol fires a signal.")
    st.stop()

st.markdown(f"**{len(items)} notification(s)**")

for n in items:
    unread = n.get("read_at") is None
    direction = (n.get("direction") or "").upper()
    color = {"BUY": "🟢", "SELL": "🔴"}.get(direction, "🔵")
    badge = " · **NEW**" if unread else ""

    with st.container(border=True):
        top = st.columns([8, 2])
        with top[0]:
            st.markdown(f"### {color} {n['title']}{badge}")
            st.caption(
                f"{n.get('created_at') or '—'} · source: {n.get('source') or '—'} · "
                f"strategy: {n.get('strategy') or '—'}"
            )
            if n.get("body"):
                st.markdown(n["body"])
        with top[1]:
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
