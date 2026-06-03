from __future__ import annotations

"""Webull connection cockpit — credential status and account snapshot.

Webull uses HMAC-SHA1 request signing (app_key + app_secret) rather than
OAuth, so there is no authorization URL to visit. Connection = credentials
present in .env and account ping succeeds.

To connect:
  1. Add WEBULL_APP_KEY, WEBULL_APP_SECRET, and WEBULL_ACCOUNT_ID to .env
  2. Restart the API server.
  3. Return here — status should read CONNECTED.
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

# Streamlit doesn't auto-load .env — parse it manually so os.getenv works below.
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
from _theme import apply_theme, section, kpi_row, pill, money

apply_theme("Webull")
st.title("Webull")
st.caption(
    "Monitor the Webull brokerage link. Webull uses HMAC-signed API keys "
    "(no OAuth), so connection is maintained as long as the keys in .env are valid."
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

status = _safe(api.webull_status, {"state": "error"})
state = status.get("state", "error")

_PILL = {
    "connected":      ("CONNECTED", "green"),
    "not_configured": ("NOT CONFIGURED", "red"),
    "error":          ("CONNECTION ERROR", "red"),
}
label, color = _PILL.get(state, ("UNKNOWN", "grey"))
st.markdown(pill(label, color), unsafe_allow_html=True)

detail = status.get("detail", "")
if detail:
    st.caption(detail)

if status.get("account_id"):
    st.caption(f"Account ID: {status['account_id']}")

if state == "not_configured":
    st.error(
        "WEBULL_APP_KEY and/or WEBULL_APP_SECRET are not set in `.env`. "
        "Add your Webull Open API credentials and restart the API server."
    )
    with st.expander("How to get Webull API credentials"):
        st.markdown(
            """
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
"""
        )
elif state == "error":
    st.warning(
        "Credentials are configured but the account ping failed. "
        "Check that your App Key and Secret are correct and that your "
        "Webull account has API trading enabled."
    )
    with st.expander("Troubleshooting"):
        st.markdown(
            """
- Verify `WEBULL_APP_KEY` and `WEBULL_APP_SECRET` in `.env` are correct.
- Confirm the Webull Open API is enabled on your account.
- Check that `WEBULL_ACCOUNT_ID` matches your actual account ID.
- Restart the API server after any `.env` change.
"""
        )

c1, _ = st.columns([1, 3])
with c1:
    if st.button("🔄 Refresh status", use_container_width=True):
        st.rerun()


# ── Account snapshot ──────────────────────────────────────────────────────────
section("Account snapshot")
accts = _safe(lambda: api.broker_account_summary("webull"), []) or []
positions = _safe(lambda: api.broker_positions("webull"), []) or []

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
        "No Webull account data. "
        + ("Configure credentials above and restart the API." if state != "connected"
           else "The account ping succeeded but no account data was returned.")
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
elif state == "connected":
    st.caption("No open Webull positions.")


# ── Configuration reference ───────────────────────────────────────────────────
section("Configuration")
settings_env = {
    "WEBULL_APP_KEY":     "App Key from Webull Open API",
    "WEBULL_APP_SECRET":  "App Secret from Webull Open API",
    "WEBULL_ACCOUNT_ID":  "Your numeric Webull account ID",
}
rows = []
for key, desc in settings_env.items():
    val = os.getenv(key, "")
    rows.append({
        "Variable": key,
        "Description": desc,
        "Set": "✅ Yes" if val else "❌ No",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "Edit `.env` in the project root and restart the API server to apply changes. "
    "Never commit `.env` to version control."
)
