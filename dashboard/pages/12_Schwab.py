from __future__ import annotations

"""Schwab connection cockpit — connect / re-authorize and see token health.

Why this page exists: the Home dashboard builds broker tabs *dynamically* from
whatever account data the API returns. When Schwab's OAuth token expires (the
access token lasts ~30 min and the refresh token ~7 days), get_accounts() raises
and the Home page's _safe() wrapper swallows it — so Schwab silently disappears
with no explanation. This page makes the connection state visible and gives a
one-click re-authorize so getting Schwab back doesn't require curling the API.

OAuth flow (differs from Zerodha's redirect-based login):
  1. Click "Connect / Re-authorize" → calls /schwab/auth, which returns an
     authorization_url.
  2. Open that URL, log in to Schwab, approve. Schwab redirects to the
     configured redirect_uri (default https://127.0.0.1:8182/schwab/callback),
     which exchanges the code and stores fresh tokens in the DB.
  3. Return here and click "Refresh status" — Schwab should now read "connected"
     and reappear on the Home dashboard.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import api
from _theme import apply_theme, section, kpi_row, pill, money

apply_theme("Schwab")
st.title("Charles Schwab")
st.caption(
    "Connect and monitor the Schwab brokerage link. Schwab OAuth tokens expire "
    "(access ~30 min, refresh ~7 days), so this is where you re-authorize when "
    "Schwab drops off the Home dashboard."
)


def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


def _usd(v) -> str:
    return money(v, currency="$") if v is not None else "—"


# ── Connection status ─────────────────────────────────────────────────────────
section("Connection status")

status = _safe(api.schwab_status, {"state": "error"})
state = status.get("state", "error")

_PILL = {
    "connected":      ("CONNECTED", "green"),
    "expiring":       ("EXPIRING — auto-refresh", "blue"),
    "expired":        ("EXPIRED — reconnect", "red"),
    "disconnected":   ("NOT CONNECTED", "grey"),
    "not_configured": ("NOT CONFIGURED", "red"),
    "error":          ("STATUS UNAVAILABLE", "red"),
}
label, color = _PILL.get(state, ("UNKNOWN", "grey"))
st.markdown(pill(label, color), unsafe_allow_html=True)

# Human-readable expiry + time-left so you can tell *when* it'll break.
expires_at = status.get("expires_at")
seconds_left = status.get("seconds_left")
if expires_at:
    try:
        exp_dt = datetime.fromisoformat(expires_at)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        local = exp_dt.astimezone()
        when = local.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        when = expires_at
    if seconds_left is not None and seconds_left > 0:
        mins = int(seconds_left // 60)
        rel = f"{mins} min" if mins < 120 else f"{mins // 60} hr"
        st.caption(f"Access token expires at **{when}** (~{rel} left).")
    elif seconds_left is not None:
        st.caption(f"Access token expired at **{when}**.")
    else:
        st.caption(f"Token expiry: **{when}**.")

if status.get("account_number"):
    st.caption(f"Configured account: {status['account_number']}")

# State-specific guidance.
if state == "not_configured":
    st.error(
        "SCHWAB_CLIENT_ID is not set in `.env`. Add your Schwab app credentials "
        "(SCHWAB_CLIENT_ID, SCHWAB_CLIENT_SECRET, SCHWAB_REDIRECT_URI) and restart "
        "the API before connecting."
    )
elif state == "expired":
    if status.get("has_refresh_token"):
        st.warning(
            "The access token expired and the refresh token is no longer valid "
            "(Schwab refresh tokens last ~7 days). Re-authorize below to restore "
            "Schwab on the Home dashboard."
        )
    else:
        st.warning(
            "The access token expired and there is no refresh token to renew it. "
            "Re-authorize below."
        )
elif state == "disconnected":
    st.info("Schwab has never been authorized on this machine. Connect below to begin.")
elif state == "expiring":
    st.info("Token is within the refresh window — the next API call should auto-refresh it.")
elif state == "error":
    st.error(
        "Couldn't read Schwab status from the API. Is the backend running on "
        f"{api.BASE}? Restart it if you just added the /schwab/status route."
    )


# ── Connect / re-authorize ─────────────────────────────────────────────────────
section("Connect / Re-authorize")
st.caption(
    "Step 1: get the authorization link. Step 2: open it, log in to Schwab, and "
    "approve — Schwab redirects to the callback which stores fresh tokens. "
    "Step 3: come back and refresh status."
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
        "Opens Schwab's login in a new tab. After approving you'll land on the "
        "callback page (a small JSON 'success' message) — that's expected. Return "
        "here and click **Refresh status**."
    )


# ── Account snapshot ────────────────────────────────────────────────────────────
# Query Schwab directly (not the global-routing endpoints) so its account shows
# regardless of where the Home routing toggle points. Both degrade to [] on
# failure, so an unauthenticated broker reads as 'no data' rather than a 500.
section("Account snapshot")
accts = _safe(lambda: api.broker_account_summary("schwab"), []) or []
positions = _safe(lambda: api.broker_positions("schwab"), []) or []

if accts:
    eq = sum(a.get("equity") or 0 for a in accts) or None
    cash = sum(a.get("cash") or 0 for a in accts) or None
    bp = sum(a.get("buying_power") or 0 for a in accts) or None
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
    st.info(
        "No Schwab account data — the broker is unauthenticated or the token "
        "expired. Re-authorize above, then click Refresh status."
    )

if positions:
    df = pd.DataFrame(positions)
    keep = [c for c in ["symbol", "quantity", "average_cost", "current_price",
                        "market_value", "unrealized_pnl"] if c in df.columns]
    if keep:
        df = df[keep]
        for col in ("average_cost", "current_price", "market_value", "unrealized_pnl"):
            if col in df.columns:
                df[col] = df[col].apply(_usd)
        df = df.rename(columns={
            "symbol": "Symbol", "quantity": "Qty", "average_cost": "Avg cost",
            "current_price": "Last", "market_value": "Value",
            "unrealized_pnl": "Unrealised P&L",
        })
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No open Schwab positions.")
