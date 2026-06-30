"""Operator dashboard — the single page you open when the market opens.

Layout:
  HEADER     — page title + market clock
  STAT BAND  — API / market / mode / kill-switch / auto-trader status
  MAIN AREA  — [broker tabs : 2/3 width] | [risk + controls : 1/3 width]
  BOTTOM     — recent fills across all brokers
"""
from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import api
from _theme import (
    apply_theme, section, divider, kpi_row, pill,
    money, currency_symbol, empty_state,
)
from _components import page_header, stat_band, metric_band, risk_gauge_html

from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ET = ZoneInfo("America/New_York")

apply_theme("Trading System")

from _sidebar import render_sidebar
render_sidebar()

# ── Load everything once ─────────────────────────────────────────────────────
def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


# Broker-backed reads fan out to every broker; the Webull leg alone can take
# ~15s. Two problems were stacked: (1) the calls ran serially on every render
# (account_summary ~2s + positions ~6s ≈ 8s), and (2) nothing cached them, so
# EVERY Streamlit rerun — every button click — re-paid that ~8s, making the page
# feel frozen. Fix: fetch all the independent reads concurrently in ONE cached
# snapshot with a short TTL. The TTL bounds staleness (≤10s on a cockpit you
# actively watch is fine); the parallel fetch bounds the cold-render cost by the
# slowest call instead of the sum. The whole gather is inside the cached function
# so the thread pool never touches Streamlit's (thread-local) cache machinery.
import concurrent.futures as _futures


@st.cache_data(ttl=10, show_spinner=False)
def _dashboard_snapshot():
    jobs = {
        "risk":      (api.risk_status, {}),
        "accounts":  (api.account_summary, []),
        "positions": (api.positions, []),
        "orders":    (api.orders, []),
        "autot":     (api.autotrader_status, {"running": False, "traders": {}}),
        "notif":     (api.notifications_unread_count, {"unread": 0}),
    }
    out = {}
    with _futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {ex.submit(_safe, fn, default): key for key, (fn, default) in jobs.items()}
        for fut in _futures.as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def _clear_dashboard_cache():
    """Drop the cached snapshot so the next render shows fresh state. Call after
    any action that mutates server state (kill switch, auto-trader)."""
    _dashboard_snapshot.clear()


# A manual refresh that bypasses the TTL for when you want state *now*.
_hdr_l, _hdr_r = st.columns([5, 1])
with _hdr_r:
    if st.button("↻ Refresh", use_container_width=True, help="Force-refresh broker & risk data"):
        _clear_dashboard_cache()
        st.rerun()

health    = _safe(api.health, None)
if health is None:
    st.error("Cannot reach the API — is the backend running on 127.0.0.1:8001?")
    st.stop()

_data     = _dashboard_snapshot()
risk      = _data["risk"]
accounts  = _data["accounts"]
positions = _data["positions"]
orders    = _data["orders"]
autot     = _data["autot"]
notif_cnt = _data["notif"].get("unread", 0)

# ── Page header ──────────────────────────────────────────────────────────────
now_et = datetime.now(tz=ET).strftime("%H:%M ET · %a %b %d")
page_header(
    "Trading System",
    subtitle="Operator cockpit — account, risk, fills, auto-trader",
    actions=f"<span style='font-size:0.78rem;color:var(--text-3)'>{now_et}</span>",
)

# ── Stat band ────────────────────────────────────────────────────────────────
api_color    = "green" if health.get("status") == "ok" else "red"
market_color = "green" if risk.get("market_hours_active") else "grey"
ks_color     = "red"   if risk.get("kill_switch_active") else "green"
auto_color   = "teal"  if autot.get("running") else "grey"
mode_color   = "red"   if risk.get("is_live") else "blue"
notif_color  = "red"   if notif_cnt > 0 else "grey"

stat_band([
    ("API",          health.get("status", "?").upper(),                                           api_color),
    ("Market",       "OPEN" if risk.get("market_hours_active") else "CLOSED",                    market_color),
    ("Mode",         "LIVE" if risk.get("is_live") else "PAPER",                                 mode_color),
    ("Kill switch",  "ACTIVE" if risk.get("kill_switch_active") else "OFF",                      ks_color),
    ("Auto-trader",  f"ON · {len(autot.get('traders', {}))} sym" if autot.get("running") else "OFF", auto_color),
    ("Alerts",       f"{notif_cnt} unread" if notif_cnt else "clear",                            notif_color),
])

if notif_cnt > 0:
    st.warning(f"**{notif_cnt} unread alert(s)** — open the **Notifications** page to review.", icon="🔔")

# ── Helpers ──────────────────────────────────────────────────────────────────
def _broker_of_order(o: dict) -> str:
    b = (o.get("broker") or "").lower()
    return b if b else "unknown"


today_str  = datetime.now(tz=ET).strftime("%Y-%m-%d")
filled     = [o for o in orders if o.get("status") == "filled"]
today_fills = [o for o in filled if (o.get("created_at") or "").startswith(today_str)]

accounts_by_broker:  dict[str, list[dict]] = defaultdict(list)
positions_by_broker: dict[str, list[dict]] = defaultdict(list)
for a in accounts:
    accounts_by_broker[(a.get("broker") or "unknown").lower()].append(a)
for p in positions:
    positions_by_broker[(p.get("broker") or "unknown").lower()].append(p)

KNOWN_ORDER = ["schwab", "webull", "zerodha", "paper"]
seen = set(accounts_by_broker) | set(positions_by_broker)
broker_keys = [b for b in KNOWN_ORDER if b in seen] + sorted(seen - set(KNOWN_ORDER))

PRETTY_BROKER = {"schwab": "Schwab", "webull": "Webull", "zerodha": "Zerodha", "paper": "Paper"}

def _broker_label(b: str) -> str:
    if b in PRETTY_BROKER:
        return PRETTY_BROKER[b]
    if b.startswith("multi:"):
        return "Multi (" + b[len("multi:"):] + ")"
    return b.capitalize() if b else "Unknown"


# ── Broker block renderer ─────────────────────────────────────────────────────
def _render_broker_block(broker: str) -> None:
    accts_here     = accounts_by_broker.get(broker, [])
    positions_here = positions_by_broker.get(broker, [])
    fills_here     = [o for o in today_fills if _broker_of_order(o) == broker]
    cur            = currency_symbol(broker)

    spent    = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in fills_here if o.get("side") == "BUY")
    received = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in fills_here if o.get("side") == "SELL")
    realised = received - spent

    if accts_here:
        eq   = sum(a.get("equity")       or 0 for a in accts_here) or None
        cash = sum(a.get("cash")         or 0 for a in accts_here) or None
        bp   = sum(a.get("buying_power") or 0 for a in accts_here) or None
        acct_ids = [str(a.get("account_id") or "") for a in accts_here if a.get("account_id")]
        kpi_row([
            ("Equity",       money(eq,       currency=cur)),
            ("Cash",         money(cash,     currency=cur)),
            ("Buying power", money(bp,       currency=cur)),
            ("Realised P&L today", money(realised, currency=cur)),
        ])
        b_cnt  = sum(1 for o in fills_here if o.get("side") == "BUY")
        s_cnt  = sum(1 for o in fills_here if o.get("side") == "SELL")
        st.caption(
            f"Account {', '.join(acct_ids) or '—'} · "
            f"{len(fills_here)} fills today ({b_cnt} buys, {s_cnt} sells)"
        )
    else:
        st.warning(
            f"No account data for **{_broker_label(broker)}** — broker may be unauthenticated.",
            icon="⚠️",
        )
        kpi_row([
            ("Realised P&L today", money(realised, currency=cur)),
            ("Fills",              str(len(fills_here))),
        ])

    if positions_here:
        df = pd.DataFrame(positions_here)
        keep = [c for c in ["symbol", "quantity", "average_cost", "current_price",
                             "market_value", "unrealized_pnl"] if c in df.columns]
        if keep:
            df = df[keep]
            for col in ("average_cost", "current_price", "market_value", "unrealized_pnl"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: money(v, currency=cur) if v is not None else "—")
            df = df.rename(columns={
                "symbol": "Symbol", "quantity": "Qty",
                "average_cost": "Avg Cost", "current_price": "Last",
                "market_value": "Value", "unrealized_pnl": "Unrealised P&L",
            })
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions.")


# ── Portfolio hero band — aggregate across all brokers ───────────────────────
def _hero_metric(label: str, value: str, sub: str = "", *, tone: str = "") -> str:
    tone_cls = f" tx-{tone}" if tone else ""
    sub_html = f"<div class='tx-hero-sub{tone_cls}'>{sub}</div>" if sub else ""
    return (
        f"<div class='tx-hero-cell'>"
        f"<div class='tx-hero-label'>{label}</div>"
        f"<div class='tx-hero-value'>{value}</div>"
        f"{sub_html}"
        f"</div>"
    )

total_equity = sum((a.get("equity") or 0) for a in accounts) or None
total_cash   = sum((a.get("cash") or 0) for a in accounts) or None
total_upnl   = sum((p.get("unrealized_pnl") or 0) for p in positions)
open_pos     = len([p for p in positions if (p.get("quantity") or 0) != 0])
day_buys     = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in today_fills if o.get("side") == "BUY")
day_sells    = sum((o.get("fill_price") or 0) * (o.get("quantity") or 0) for o in today_fills if o.get("side") == "SELL")
day_realised = day_sells - day_buys
upnl_tone    = "pos" if total_upnl > 0 else "neg" if total_upnl < 0 else "muted"
real_tone    = "pos" if day_realised > 0 else "neg" if day_realised < 0 else "muted"

st.markdown(
    "<div class='tx-hero'>"
    + _hero_metric("Total Equity", money(total_equity), f"{len(broker_keys)} broker(s)")
    + _hero_metric("Cash", money(total_cash))
    + _hero_metric("Unrealised P&L", money(total_upnl), f"{open_pos} open position(s)", tone=upnl_tone)
    + _hero_metric("Realised P&L Today", money(day_realised), f"{len(today_fills)} fill(s)", tone=real_tone)
    + "</div>",
    unsafe_allow_html=True,
)

# ── Main layout: broker area (left) + risk rail (right) ──────────────────────
broker_area, right = st.columns([11, 5], gap="medium")

with broker_area:
    section("Broker Accounts")
    if not broker_keys:
        empty_state(
            "No broker data",
            "Authenticate at least one broker (Schwab, Webull, or Zerodha) to see account data here.",
            icon="🔌",
        )
    else:
        tabs = st.tabs([_broker_label(b) for b in broker_keys])
        for tab, broker in zip(tabs, broker_keys):
            with tab:
                _render_broker_block(broker)


# ── Risk rail ─────────────────────────────────────────────────────────────────
with right:
    section("Risk Budget")

    if risk:
        orders_used = risk.get("orders_today", 0)
        orders_max  = risk.get("max_orders_per_day", 1)
        loss_used   = risk.get("daily_loss_usd", 0)
        loss_max    = risk.get("max_daily_loss_usd", 1)
        order_pct   = min(orders_used / max(orders_max, 1), 1.0)
        loss_pct    = min(abs(loss_used) / max(abs(loss_max), 1), 1.0)

        # Orders gauge
        st.markdown(risk_gauge_html("Orders today", orders_used, orders_max, prefix=""), unsafe_allow_html=True)
        st.progress(order_pct)

        # Loss gauge
        st.markdown("<div style='margin-top:var(--sp-3)'></div>", unsafe_allow_html=True)
        st.markdown(risk_gauge_html("Daily loss used", abs(loss_used), abs(loss_max)), unsafe_allow_html=True)
        st.progress(loss_pct)

        st.markdown("") # spacing
        if risk.get("kill_switch_active"):
            st.error("**Kill switch is ACTIVE** — all new orders are blocked.", icon="🛑")
            if st.button("Deactivate kill switch", type="primary", use_container_width=True, key="home_ks_off"):
                api.set_kill_switch(False)
                _clear_dashboard_cache()
                st.rerun()
        else:
            if st.button("⚡ Activate kill switch", use_container_width=True, key="home_ks_on"):
                api.set_kill_switch(True)
                _clear_dashboard_cache()
                st.rerun()
    else:
        st.warning("Risk data unavailable.", icon="⚠️")

    divider()
    section("Auto-trader", level=3)

    if autot.get("running"):
        traders = autot.get("traders", {})
        symbols = ", ".join(traders.keys()) or "—"
        st.markdown(pill(f"RUNNING — {len(traders)} symbol(s)", "green"), unsafe_allow_html=True)
        st.caption(f"Active: {symbols}")
        if st.button("Stop (keep positions)", use_container_width=True, key="home_auto_stop"):
            api.autotrader_stop(flatten=False)
            _clear_dashboard_cache()
            st.rerun()
        if st.button("Stop & flatten all", use_container_width=True, key="home_auto_flatten"):
            api.autotrader_stop(flatten=True)
            _clear_dashboard_cache()
            st.rerun()
    else:
        st.markdown(pill("OFFLINE", "grey"), unsafe_allow_html=True)
        st.caption("Start via the Day Trading → Scanner tab.")


# ── Recent fills ──────────────────────────────────────────────────────────────
divider()
section("Recent Fills", "Last 20 fills across all brokers")

if filled:
    recent = sorted(filled, key=lambda o: o.get("created_at") or "", reverse=True)[:20]
    rows = []
    for o in recent:
        _side = o.get("side", "—")
        rows.append({
            "Time":     (o.get("created_at") or "")[11:19],
            "Date":     (o.get("created_at") or "")[:10],
            "Broker":   _broker_label((o.get("broker") or "").lower()),
            "Symbol":   o.get("symbol", "—"),
            "Side":     _side,
            "Qty":      o.get("quantity", "—"),
            "Fill $":   money(o.get("fill_price")),
            "Strategy": o.get("strategy_name") or "—",
        })
    df_fills = pd.DataFrame(rows)

    def _fill_row_style(row):
        if row["Side"] == "BUY":
            return ["background-color: rgba(52,211,153,0.07)"] * len(row)
        elif row["Side"] == "SELL":
            return ["background-color: rgba(251,113,133,0.07)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_fills.style.apply(_fill_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    empty_state("No fills yet", "Completed orders will appear here.", icon="📋")
