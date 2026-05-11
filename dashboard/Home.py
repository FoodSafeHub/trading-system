from __future__ import annotations

import streamlit as st
import pandas as pd
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import api

st.set_page_config(page_title="Trading System", page_icon="📈", layout="wide")
st.title("📈 Trading System — Dashboard")

# ── Health ────────────────────────────────────────────────────
try:
    h = api.health()
    col1, col2, col3 = st.columns(3)
    col1.metric("Status", h["status"].upper())
    col2.metric("Broker", h["broker"].upper())
    col3.metric("Mode", h["mode"])
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

st.divider()

# ── Account ───────────────────────────────────────────────────
st.subheader("Account")
try:
    accounts = api.account_summary()
    if accounts:
        acct = accounts[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Equity", f"${acct['equity']:,.2f}")
        c2.metric("Cash", f"${acct['cash']:,.2f}")
        c3.metric("Buying Power", f"${acct['buying_power']:,.2f}")
        c4.metric("Account", acct["account_id"])
except Exception as e:
    st.warning(f"Account data unavailable: {e}")

st.divider()

# ── Risk Status ───────────────────────────────────────────────
st.subheader("Risk Engine")
try:
    risk = api.risk_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Hours", "OPEN" if risk["market_hours_active"] else "CLOSED")
    c2.metric("Orders Today", f"{risk['orders_today']} / {risk['max_orders_per_day']}")
    c3.metric("Daily Loss", f"${risk['daily_loss_usd']:,.2f}", delta_color="inverse",
              delta=f"limit ${risk['max_daily_loss_usd']:,.2f}")
    kill_label = "🔴 ACTIVE" if risk["kill_switch_active"] else "🟢 OFF"
    c4.metric("Kill Switch", kill_label)

    st.divider()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if risk["kill_switch_active"]:
            if st.button("✅ Deactivate Kill Switch", type="primary"):
                api.set_kill_switch(False)
                st.rerun()
        else:
            if st.button("🛑 Activate Kill Switch", type="secondary"):
                api.set_kill_switch(True)
                st.rerun()
except Exception as e:
    st.warning(f"Risk data unavailable: {e}")

st.divider()

# ── P&L Summary ───────────────────────────────────────────────
st.subheader("Paper Trading P&L")
try:
    orders = api.orders()
    filled = [o for o in orders if o.get("status") == "filled"]

    if filled:
        import json
        total_spent   = sum(o.get("fill_price", 0) * o.get("quantity", 0)
                            for o in filled if o.get("side") == "BUY")
        total_received = sum(o.get("fill_price", 0) * o.get("quantity", 0)
                             for o in filled if o.get("side") == "SELL")
        realised_pnl  = total_received - total_spent
        total_orders  = len(filled)
        buy_orders    = len([o for o in filled if o.get("side") == "BUY"])
        sell_orders   = len([o for o in filled if o.get("side") == "SELL"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Realised P&L",   f"${realised_pnl:,.2f}",
                  delta_color="normal" if realised_pnl >= 0 else "inverse")
        c2.metric("Filled Orders",  total_orders)
        c3.metric("BUY fills",      buy_orders)
        c4.metric("SELL fills",     sell_orders)

        # Order history table
        df = pd.DataFrame(filled)
        keep = [c for c in ["created_at","symbol","side","quantity","fill_price","status","strategy_name"] if c in df.columns]
        if keep:
            df = df[keep]
            if "fill_price" in df.columns:
                df["fill_price"] = df["fill_price"].apply(lambda v: f"${v:,.2f}" if v else "—")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No filled orders yet — P&L will appear here once the bot places trades during market hours.")
except Exception as e:
    st.warning(f"P&L data unavailable: {e}")

st.divider()

# ── Positions ─────────────────────────────────────────────────
st.subheader("Open Positions")
try:
    pos = api.positions()
    if pos:
        st.dataframe(pd.DataFrame(pos), use_container_width=True)
    else:
        st.info("No open positions.")
except Exception as e:
    st.warning(f"Positions unavailable: {e}")
