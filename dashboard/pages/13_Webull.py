from __future__ import annotations

"""Webull connection cockpit — credential status and account snapshot.

Webull uses HMAC-SHA1 request signing (app_key + app_secret) rather than OAuth,
so there is no authorization URL. Connection = credentials present in .env and
account ping succeeds.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")


def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = val.strip().strip('"').strip("'")
    except OSError:
        pass


_load_dotenv()

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


apply_theme("Webull")

from _sidebar import render_sidebar
render_sidebar()

# ── Status data ────────────────────────────────────────────────────────────────
status    = _safe(api.webull_status, {"state": "error"})
state     = status.get("state", "error")
accts     = _safe(lambda: api.broker_account_summary("webull"), []) or []
positions = _safe(lambda: api.broker_positions("webull"), []) or []

_PILL = {
    "connected":      ("CONNECTED",      "green"),
    "not_configured": ("NOT CONFIGURED", "red"),
    "error":          ("CONN ERROR",     "red"),
}
conn_label, conn_color = _PILL.get(state, ("UNKNOWN", "grey"))

page_header(
    "Webull",
    subtitle="HMAC-signed API key connection — no OAuth. Credentials set in .env.",
    badge=conn_label,
    badge_color=conn_color,
)

stat_band([
    ("Connection", conn_label,                                  conn_color),
    ("Account ID", status.get("account_id", "—") or "—",       "grey"),
    ("Positions",  str(len(positions)),                         "teal" if positions else "grey"),
])

# ── State-specific guidance ────────────────────────────────────────────────────
detail = status.get("detail", "")
if detail:
    st.caption(detail)

if state == "not_configured":
    st.error(
        "**WEBULL_APP_KEY and/or WEBULL_APP_SECRET are not set in `.env`.**  "
        "Add your Webull Open API credentials and restart the API server.",
        icon="🔧",
    )
    with st.expander("How to get Webull API credentials"):
        st.markdown("""
1. Log in to [Webull](https://www.webull.com) and go to **Account → Open API**.
2. Create an application to receive your **App Key** and **App Secret**.
3. Note your **Account ID** from the account overview.
4. Add to your `.env`:
   ```
   WEBULL_APP_KEY=your_key_here
   WEBULL_APP_SECRET=your_secret_here
   WEBULL_ACCOUNT_ID=your_account_id_here
   ```
5. Restart the API server and refresh this page.
""")
elif state == "error":
    st.warning(
        "Credentials are configured but the account ping failed. "
        "Check that your App Key and Secret are correct and that your Webull account has API trading enabled.",
        icon="⚠️",
    )
    with st.expander("Troubleshooting"):
        st.markdown("""
- Verify `WEBULL_APP_KEY` and `WEBULL_APP_SECRET` in `.env` are correct.
- Confirm the Webull Open API is enabled on your account.
- Check that `WEBULL_ACCOUNT_ID` matches your actual account ID.
- Restart the API server after any `.env` change.
""")

refresh_col, _ = st.columns([2, 8])
if refresh_col.button("🔄 Refresh status", use_container_width=True):
    st.rerun()

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
    msg = (
        "Configure credentials above and restart the API."
        if state != "connected" else
        "The account ping succeeded but no account data was returned."
    )
    empty_state("No Webull account data", msg, icon="🔌")

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
            "Symbol":         st.column_config.TextColumn("Symbol",       width="small"),
            "Qty":            st.column_config.NumberColumn("Qty",        format="%.2f", width="small"),
            "Avg Cost":       st.column_config.TextColumn("Avg Cost",     width="small"),
            "Last":           st.column_config.TextColumn("Last",         width="small"),
            "Value":          st.column_config.TextColumn("Value",        width="small"),
            "Unrealised P&L": st.column_config.TextColumn("Unrealised P&L", width="medium"),
        },
    )
elif state == "connected":
    st.caption("No open Webull positions.")

divider()

# ── Configuration reference ────────────────────────────────────────────────────
section("Configuration", "Environment variables that control this connection.")

settings_env = {
    "WEBULL_APP_KEY":    "App Key from Webull Open API",
    "WEBULL_APP_SECRET": "App Secret from Webull Open API",
    "WEBULL_ACCOUNT_ID": "Your numeric Webull account ID",
}
rows = []
for key, desc in settings_env.items():
    val = os.getenv(key, "")
    rows.append({"Variable": key, "Description": desc, "Set": "✅ Yes" if val else "❌ No"})

st.dataframe(
    pd.DataFrame(rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Variable":    st.column_config.TextColumn("Variable",    width="medium"),
        "Description": st.column_config.TextColumn("Description", width="large"),
        "Set":         st.column_config.TextColumn("Set",         width="small"),
    },
)
st.caption("Edit `.env` in the project root and restart the API server to apply changes. Never commit `.env` to version control.")
