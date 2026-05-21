"""Operator dashboard — the single page you open when the market opens.

Layout:
  TOP STRIP   ─ API / market / kill-switch / auto-trader status pills
  RIGHT RAIL  ─ Risk budget gauge + kill switch + auto-trader controls
  PER BROKER  ─ One tab per broker. Each tab shows that broker's P&L,
                equity / cash / buying-power KPIs, and open positions in a
                roomy 2-up layout so labels and values don't truncate.
  BOTTOM      ─ Recent fills across brokers, with broker + strategy columns
"""
from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import api
from _theme import apply_theme, section, divider, kpi_row, pill, status_row, money

from collections import defaultdict
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


# ── Helpers to group state by broker ────────────────────────────────────
def _broker_of_order(o: dict) -> str:
    """Map raw broker tag to a presentation label.

    Multi-broker mode stamps Order.broker as "multi:schwab+webull" because the
    fan-out adapter owns the call. The dashboard wants per-leg attribution, so
    we keep "multi:..." rows in a single bucket labelled by the prefix.
    """
    b = (o.get("broker") or "").lower()
    if not b:
        return "unknown"
    if b.startswith("multi:"):
        return b
    return b


today_str = datetime.now(tz=ET).strftime("%Y-%m-%d")
filled = [o for o in orders if o.get("status") == "filled"]


def _is_today(o):
    ts = o.get("created_at") or ""
    return ts.startswith(today_str)


today_fills = [o for o in filled if _is_today(o)]

# Group accounts and positions by broker tag.
accounts_by_broker: dict[str, list[dict]] = defaultdict(list)
for a in accounts:
    accounts_by_broker[(a.get("broker") or "unknown").lower()].append(a)

positions_by_broker: dict[str, list[dict]] = defaultdict(list)
for p in positions:
    positions_by_broker[(p.get("broker") or "unknown").lower()].append(p)

# Order the broker columns deterministically: Schwab first, then Webull, then
# anything else (paper / unknown). New brokers slot in automatically because
# we union the keys from accounts and positions.
KNOWN_ORDER = ["schwab", "webull", "paper"]
seen = set(accounts_by_broker) | set(positions_by_broker)
broker_keys = [b for b in KNOWN_ORDER if b in seen] + sorted(seen - set(KNOWN_ORDER))


# ── BROKER ZONE (left, full width) + RIGHT RAIL ─────────────────────────
broker_area, right = st.columns([2, 1])


PRETTY_BROKER = {
    "schwab": "Schwab",
    "webull": "Webull",
    "paper":  "Paper",
}


def _broker_label(b: str) -> str:
    if b in PRETTY_BROKER:
        return PRETTY_BROKER[b]
    if b.startswith("multi:"):
        return "Multi (" + b[len("multi:"):] + ")"
    return b.capitalize() if b else "Unknown"


def _render_broker_block(broker: str) -> None:
    accts_here = accounts_by_broker.get(broker, [])
    positions_here = positions_by_broker.get(broker, [])
    fills_here = [o for o in today_fills if _broker_of_order(o) == broker]

    spent    = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in fills_here if o.get("side") == "BUY")
    received = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in fills_here if o.get("side") == "SELL")
    realised = received - spent

    # ── Row 1: account snapshot (Equity / Cash / Buying power / Account) ──
    if accts_here:
        eq   = sum(a.get("equity")       or 0 for a in accts_here) or None
        cash = sum(a.get("cash")         or 0 for a in accts_here) or None
        bp   = sum(a.get("buying_power") or 0 for a in accts_here) or None
        acct_ids = [str(a.get("account_id") or "") for a in accts_here if a.get("account_id")]
        acct_label = ", ".join(acct_ids) or "—"
        # Wide layout: 4 columns in the broker_area (which is 2/3 of the page)
        # gives ~200px per cell, enough that values like "$2,659.11" and
        # account ids don't truncate. Account id is rendered separately below
        # so the 4 KPI cells stay roomy.
        kpi_row([
            ("Equity",        money(eq)),
            ("Cash",          money(cash)),
            ("Buying power",  money(bp)),
            ("Realised P&L",  money(realised)),
        ])
        st.caption(f"Account {acct_label} · {len(fills_here)} fills today "
                   f"({sum(1 for o in fills_here if o.get('side')=='BUY')} buys, "
                   f"{sum(1 for o in fills_here if o.get('side')=='SELL')} sells)")
    else:
        st.warning(
            f"No account data for **{_broker_label(broker)}** — broker may be "
            "unauthenticated, or the API hasn't been restarted since the "
            "multi-broker changes. Restart with `start.bat` to refresh."
        )
        # Still surface today's fills even if accounts are missing.
        kpi_row([
            ("Realised P&L (today)", money(realised)),
            ("Fills",                str(len(fills_here))),
            ("Buys",                 str(sum(1 for o in fills_here if o.get("side") == "BUY"))),
            ("Sells",                str(sum(1 for o in fills_here if o.get("side") == "SELL"))),
        ])

    # ── Row 2: open positions table ───────────────────────────────────────
    if positions_here:
        df = pd.DataFrame(positions_here)
        keep = [c for c in ["symbol", "quantity", "average_cost", "current_price",
                            "market_value", "unrealized_pnl"]
                if c in df.columns]
        if keep:
            df = df[keep]
            rename = {
                "symbol":         "Symbol",
                "quantity":       "Qty",
                "average_cost":   "Avg cost",
                "current_price":  "Last",
                "market_value":   "Value",
                "unrealized_pnl": "Unrealised P&L",
            }
            for col in ("average_cost", "current_price", "market_value", "unrealized_pnl"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: money(v) if v is not None else "—")
            df = df.rename(columns=rename)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions.")


with broker_area:
    section("By broker")
    if not broker_keys:
        st.info("No broker data yet — once accounts authenticate they'll appear here.")
    else:
        # Tabs keep each broker on the full broker_area width so KPI labels
        # and values never overlap, no matter how many brokers are wired up.
        tabs = st.tabs([_broker_label(b) for b in broker_keys])
        for tab, broker in zip(tabs, broker_keys):
            with tab:
                _render_broker_block(broker)


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


# ── BOTTOM — recent fills across brokers (unified, with Broker column) ───
section("Recent Fills", "Last 15 across brokers and strategies")

if filled:
    recent = sorted(filled, key=lambda o: o.get("created_at") or "", reverse=True)[:15]
    rows = []
    for o in recent:
        rows.append({
            "Time":       (o.get("created_at") or "")[11:19],
            "Date":       (o.get("created_at") or "")[:10],
            "Broker":     _broker_label((o.get("broker") or "").lower()),
            "Symbol":     o.get("symbol", "—"),
            "Side":       o.get("side", "—"),
            "Qty":        o.get("quantity", "—"),
            "Fill":       money(o.get("fill_price")),
            "Strategy":   o.get("strategy_name") or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No fills yet.")
