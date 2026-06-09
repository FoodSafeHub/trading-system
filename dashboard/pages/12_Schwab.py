from __future__ import annotations

"""Schwab connection cockpit — connect / re-authorize and see token health."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import api
from _theme import apply_theme, section, kpi_row, pill, money, divider, empty_state
from _components import page_header, stat_band


def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


def _usd(v) -> str:
    return money(v, currency="$") if v is not None else "—"


apply_theme("Schwab")

from _sidebar import render_sidebar
render_sidebar()

# ── Status data ────────────────────────────────────────────────────────────────
status   = _safe(api.schwab_status, {"state": "error"})
state    = status.get("state", "error")
accts    = _safe(lambda: api.broker_account_summary("schwab"), []) or []
positions = _safe(lambda: api.broker_positions("schwab"), []) or []

_PILL = {
    "connected":      ("CONNECTED",              "green"),
    "expiring":       ("EXPIRING — auto-refresh", "blue"),
    "expired":        ("EXPIRED",                 "red"),
    "disconnected":   ("NOT CONNECTED",           "grey"),
    "not_configured": ("NOT CONFIGURED",          "red"),
    "error":          ("UNAVAILABLE",             "red"),
}
conn_label, conn_color = _PILL.get(state, ("UNKNOWN", "grey"))

# Token expiry detail
_expiry_str = "—"
expires_at   = status.get("expires_at")
seconds_left = status.get("seconds_left")
if expires_at:
    try:
        exp_dt = datetime.fromisoformat(expires_at)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        local = exp_dt.astimezone()
        _expiry_str = local.strftime("%H:%M %Z")
        if seconds_left is not None and seconds_left > 0:
            mins = int(seconds_left // 60)
            _expiry_str += f" (~{mins}m left)" if mins < 120 else f" (~{mins // 60}h left)"
    except Exception:
        _expiry_str = str(expires_at)[:16]

page_header(
    "Charles Schwab",
    subtitle="OAuth brokerage connection — access token ~30 min, refresh token ~7 days.",
    badge=conn_label,
    badge_color=conn_color,
)

stat_band([
    ("Connection",  conn_label,                                      conn_color),
    ("Token expiry", _expiry_str,                                    "amber" if state == "expiring" else "grey"),
    ("Account",      status.get("account_number", "—") or "—",      "grey"),
    ("Positions",    str(len(positions)),                            "teal" if positions else "grey"),
])

# ── State-specific guidance ────────────────────────────────────────────────────
if state == "not_configured":
    st.error(
        "**SCHWAB_CLIENT_ID is not set in `.env`.**  "
        "Add your Schwab app credentials (SCHWAB_CLIENT_ID, SCHWAB_CLIENT_SECRET, "
        "SCHWAB_REDIRECT_URI) and restart the API.",
        icon="🔧",
    )
elif state == "expired":
    msg = (
        "The access token expired and the refresh token is no longer valid — "
        "Schwab refresh tokens last ~7 days. Re-authorize below."
        if not status.get("has_refresh_token") else
        "The access token expired and there is no refresh token. Re-authorize below."
    )
    st.warning(msg, icon="⏰")
elif state == "disconnected":
    st.info("Schwab has never been authorized on this machine. Connect below.", icon="🔌")
elif state == "expiring":
    st.info("Token is within the auto-refresh window — the next API call should renew it.", icon="♻️")
elif state == "error":
    st.error(f"Couldn't read Schwab status from the API at {api.BASE}. Is the backend running?", icon="⚠️")

divider()

# ── Connect / re-authorize ─────────────────────────────────────────────────────
section("Connect / Re-authorize")
st.caption(
    "Step 1: get the authorization link. "
    "Step 2: open it, log in to Schwab, approve — Schwab redirects to the callback which stores fresh tokens. "
    "Step 3: come back and click Refresh status."
)

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🔗 Get authorization link", type="primary", use_container_width=True,
                 disabled=(state == "not_configured")):
        try:
            resp = api.schwab_auth_url()
            st.session_state["schwab_auth_url"] = resp.get("authorization_url", "")
        except Exception as e:
            st.error(f"Could not start Schwab auth: {e}")
            st.session_state.pop("schwab_auth_url", None)
with c2:
    if st.button("🔄 Refresh status", use_container_width=True):
        st.session_state.pop("schwab_auth_url", None)
        st.rerun()

auth_url = st.session_state.get("schwab_auth_url")
if auth_url:
    st.link_button("Open Schwab login →", auth_url, use_container_width=True)
    st.caption(
        "Opens Schwab's login in a new tab. After approving you'll see a small JSON 'success' "
        "message — that's expected. Return here and click **Refresh status**."
    )

divider()

# ── Account snapshot ───────────────────────────────────────────────────────────
section("Account Snapshot")

if accts:
    eq   = sum(a.get("equity")       or 0 for a in accts) or None
    cash = sum(a.get("cash")         or 0 for a in accts) or None
    bp   = sum(a.get("buying_power") or 0 for a in accts) or None
    kpi_row([
        ("Equity",         _usd(eq)),
        ("Cash",           _usd(cash)),
        ("Buying power",   _usd(bp)),
        ("Open positions", str(len(positions))),
    ])
    acct_ids = ", ".join(str(a.get("account_id") or "") for a in accts if a.get("account_id"))
    if acct_ids:
        st.caption(f"Account {acct_ids}")
else:
    empty_state(
        "No account data",
        "Schwab is unauthenticated or the token expired. Re-authorize above, then click Refresh status.",
        icon="🔌",
    )

if positions:
    df = pd.DataFrame(positions)
    keep = [c for c in ["symbol", "quantity", "average_cost", "current_price",
                         "market_value", "unrealized_pnl"] if c in df.columns]
    if keep:
        df = df[keep].rename(columns={
            "symbol": "Symbol", "quantity": "Qty", "average_cost": "Avg Cost",
            "current_price": "Last", "market_value": "Value",
            "unrealized_pnl": "Unrealised P&L",
        })
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol":        st.column_config.TextColumn("Symbol",       width="small"),
            "Qty":           st.column_config.NumberColumn("Qty",        format="%.2f", width="small"),
            "Avg Cost":      st.column_config.TextColumn("Avg Cost",     width="small"),
            "Last":          st.column_config.TextColumn("Last",         width="small"),
            "Value":         st.column_config.TextColumn("Value",        width="small"),
            "Unrealised P&L":st.column_config.TextColumn("Unrealised P&L", width="medium"),
        },
    )
elif state == "connected":
    st.caption("No open Schwab positions.")
