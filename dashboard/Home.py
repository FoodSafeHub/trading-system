"""Operator dashboard — the single page you open when the market opens.

Layout (3 zones):
  TOP STRIP   ─ API / market / kill-switch / auto-trader status pills
  LEFT (2/3)  ─ Today's P&L + open positions
  RIGHT (1/3) ─ Risk budget gauge + kill switch + recent fills
"""
from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import api
from _theme import apply_theme, section, divider, kpi_row, pill, status_row, money

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ET = ZoneInfo("America/New_York")

apply_theme("Trading System")
st.title("Trading System")


# ── Load everything once (each call is cheap & cached at the API layer) ──
def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


health = _safe(api.health, None)
if health is None:
    st.error("Cannot reach the API. Is the backend running on 127.0.0.1:8001?")
    st.stop()

risk      = _safe(api.risk_status, {})
accounts  = _safe(api.account_summary, [])
positions = _safe(api.positions, [])
orders    = _safe(api.orders, [])
autot     = _safe(api.autotrader_status, {"running": False, "traders": {}})


# ── TOP STRIP — at-a-glance state ───────────────────────────────────────
now_et = datetime.now(tz=ET).strftime("%H:%M ET · %a %b %d")

api_color    = "green" if health.get("status") == "ok" else "red"
market_color = "green" if risk.get("market_hours_active") else "grey"
ks_color     = "red" if risk.get("kill_switch_active") else "green"
auto_color   = "blue" if autot.get("running") else "grey"

mode_text = "LIVE" if risk.get("is_live") else "Paper"
mode_color = "red" if risk.get("is_live") else "blue"

status_row([
    ("API",          health.get("status", "?").upper(),                                   api_color),
    ("Market",       "OPEN" if risk.get("market_hours_active") else "CLOSED",             market_color),
    ("Mode",         mode_text,                                                            mode_color),
    ("Kill switch",  "ACTIVE" if risk.get("kill_switch_active") else "OFF",               ks_color),
    ("Auto-trader",  f"ON ({len(autot.get('traders', {}))} symbols)" if autot.get("running") else "OFF", auto_color),
    ("Clock",        now_et,                                                               "grey"),
])

divider()


# ── ZONE 1 (LEFT 2/3) + ZONE 2 (RIGHT 1/3) ──────────────────────────────
left, right = st.columns([2, 1])


# ─── LEFT — P&L + positions ─────────────────────────────────────────────
with left:
    section("Today's P&L")

    filled = [o for o in orders if o.get("status") == "filled"]
    today_str = datetime.now(tz=ET).strftime("%Y-%m-%d")

    def _is_today(o):
        ts = o.get("created_at") or ""
        return ts.startswith(today_str)

    today_fills = [o for o in filled if _is_today(o)]

    spent    = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in today_fills if o.get("side") == "BUY")
    received = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in today_fills if o.get("side") == "SELL")
    realised = received - spent

    kpi_row([
        ("Realised P&L (today)", money(realised)),
        ("Fills today",          str(len(today_fills))),
        ("Buys",                 str(sum(1 for o in today_fills if o.get("side") == "BUY"))),
        ("Sells",                str(sum(1 for o in today_fills if o.get("side") == "SELL"))),
    ])

    # Account equity row
    if accounts:
        a = accounts[0]
        kpi_row([
            ("Equity",        money(a.get("equity"))),
            ("Cash",          money(a.get("cash"))),
            ("Buying power",  money(a.get("buying_power"))),
            ("Account",       str(a.get("account_id", "—"))),
        ])

    divider()

    section("Open Positions", f"{len(positions)} held" if positions else None)
    if positions:
        # Tidy column selection so the table is readable
        df = pd.DataFrame(positions)
        keep = [c for c in ["symbol", "quantity", "avg_price", "current_price",
                            "market_value", "unrealized_pl", "unrealized_pl_pct"]
                if c in df.columns]
        if keep:
            df = df[keep]
            for col in ("avg_price", "current_price", "market_value", "unrealized_pl"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: money(v) if v is not None else "—")
            if "unrealized_pl_pct" in df.columns:
                df["unrealized_pl_pct"] = df["unrealized_pl_pct"].apply(
                    lambda v: f"{v:+.2f}%" if v is not None else "—"
                )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")


# ─── RIGHT — risk + kill switch + auto-trader controls ──────────────────
with right:
    section("Risk Budget")

    if risk:
        orders_used = risk.get("orders_today", 0)
        orders_max  = risk.get("max_orders_per_day", 1)
        loss_used   = risk.get("daily_loss_usd", 0)
        loss_max    = risk.get("max_daily_loss_usd", 1)

        order_pct = min(orders_used / max(orders_max, 1), 1.0)
        loss_pct  = min(loss_used / max(loss_max, 1), 1.0)

        st.markdown(f"**Orders today** &nbsp; {orders_used} / {orders_max}")
        st.progress(order_pct)

        st.markdown(f"**Daily loss** &nbsp; {money(loss_used)} / {money(loss_max)}")
        st.progress(loss_pct)

        # Kill switch button — surfaced here so it's one click from home
        if risk.get("kill_switch_active"):
            if st.button("Deactivate kill switch", type="primary", use_container_width=True, key="home_ks_off"):
                api.set_kill_switch(False)
                st.rerun()
        else:
            if st.button("Activate kill switch", type="secondary", use_container_width=True, key="home_ks_on"):
                api.set_kill_switch(True)
                st.rerun()
    else:
        st.warning("Risk data unavailable")

    divider()

    section("Auto-trader")
    if autot.get("running"):
        traders = autot.get("traders", {})
        symbols = ", ".join(traders.keys()) or "—"
        st.markdown(pill(f"ON — {len(traders)} symbol(s)", "green"), unsafe_allow_html=True)
        st.caption(symbols)
        if st.button("Stop (keep positions)", use_container_width=True, key="home_auto_stop"):
            api.autotrader_stop(flatten=False)
            st.rerun()
    else:
        st.markdown(pill("OFF", "grey"), unsafe_allow_html=True)
        st.caption("Start on the Day Trading page.")


divider()


# ── BOTTOM — recent fills (compact) ──────────────────────────────────────
section("Recent Fills", "Last 10 across all strategies")

if filled:
    recent = sorted(filled, key=lambda o: o.get("created_at") or "", reverse=True)[:10]
    rows = []
    for o in recent:
        rows.append({
            "Time":       (o.get("created_at") or "")[11:19],
            "Date":       (o.get("created_at") or "")[:10],
            "Symbol":     o.get("symbol", "—"),
            "Side":       o.get("side", "—"),
            "Qty":        o.get("quantity", "—"),
            "Fill":       money(o.get("fill_price")),
            "Strategy":   o.get("strategy_name") or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No fills yet.")
