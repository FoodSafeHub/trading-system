from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, divider, kpi_row, pill, money, empty_state
from _components import page_header, stat_band, eligibility_chip, blocker_chip, risk_gauge_html, filter_cols

import pandas as pd
import streamlit as st

apply_theme("Risk & Safety")

from _sidebar import render_sidebar
render_sidebar()

# ── Load status once ──────────────────────────────────────────────────────────
try:
    s = api.risk_status()
except Exception as e:
    st.error(f"Cannot reach risk API: {e}", icon="⚠️")
    st.stop()

ks_active  = s["kill_switch_active"]
is_live    = s.get("is_live", False)
mkt_open   = s.get("market_hours_active", False)
broker     = (s.get("active_broker") or "—").upper()
orders_t   = s.get("orders_today", 0)
orders_max = s.get("max_orders_per_day", 1)
loss_usd   = s.get("daily_loss_usd", 0)
loss_max   = s.get("max_daily_loss_usd", 1)

# Derive overall system readiness
checks_data = [
    ("Kill switch is OFF",         not ks_active),
    ("Broker is Schwab (live)",    broker == "SCHWAB"),
    ("LIVE_TRADING_ENABLED=true",  s.get("live_trading_enabled", False)),
    ("LIVE_TRADING_CONFIRMED=true",s.get("live_trading_confirmed", False)),
    ("Daily order limit not hit",  orders_t < orders_max),
    ("Daily loss limit not hit",   loss_usd < loss_max),
]
all_pass = all(ok for _, ok in checks_data)
fail_count = sum(1 for _, ok in checks_data if not ok)

page_header(
    "Risk & Safety",
    subtitle="Kill switch, daily limits, and live-trading gates — check before going live.",
    badge="ALL CLEAR" if all_pass else f"{fail_count} FAIL{'S' if fail_count > 1 else ''}",
    badge_color="green" if all_pass else "red",
)

stat_band([
    ("Kill switch",   "ACTIVE" if ks_active else "OFF",                          "red"   if ks_active else "green"),
    ("Mode",          "LIVE"   if is_live   else "PAPER",                         "red"   if is_live   else "blue"),
    ("Market",        "OPEN"   if mkt_open  else "CLOSED",                        "green" if mkt_open  else "grey"),
    ("Broker",        broker,                                                      "grey"),
    ("Orders today",  f"{orders_t} / {orders_max}",                               "amber" if orders_t >= orders_max * 0.75 else "grey"),
    ("Daily loss",    f"{money(abs(loss_usd))} / {money(abs(loss_max))}",         "amber" if abs(loss_usd) >= abs(loss_max) * 0.6 else "grey"),
])

# ── 1. Kill switch ─────────────────────────────────────────────────────────────
section("Emergency Kill Switch", "Halts all automated trading instantly.")

ks_left, ks_right = st.columns([4, 1])
with ks_left:
    if ks_active:
        st.markdown(
            "<div class='tx-kill-active'>🛑 KILL SWITCH ACTIVE — all automated trading halted</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"&nbsp;{eligibility_chip('ready', 'Order paths open — trading allowed')} "
            f"&nbsp; Kill switch is off.",
            unsafe_allow_html=True,
        )
with ks_right:
    if ks_active:
        if st.button("Deactivate", type="primary", key="ks_off", use_container_width=True):
            api.set_kill_switch(False)
            st.rerun()
    else:
        if st.button("Activate kill switch", type="secondary", key="ks_on", use_container_width=True):
            api.set_kill_switch(True)
            st.rerun()

divider()

# ── 2. Risk gauges ─────────────────────────────────────────────────────────────
section("Daily Usage")

g1, g2 = st.columns(2)
with g1:
    st.markdown(risk_gauge_html("Orders today", orders_t, orders_max, prefix=""), unsafe_allow_html=True)
    st.progress(min(orders_t / max(orders_max, 1), 1.0))
with g2:
    st.markdown(risk_gauge_html("Daily loss used", abs(loss_usd), abs(loss_max)), unsafe_allow_html=True)
    st.progress(min(abs(loss_usd) / max(abs(loss_max), 1), 1.0))

kpi_row([
    ("Broker",                   broker),
    ("Mode",                     "LIVE" if is_live else "Paper"),
    ("Live Trading Enabled",     "Yes" if s.get("live_trading_enabled") else "No"),
    ("Live Confirmed in .env",   "Yes" if s.get("live_trading_confirmed") else "No"),
])

divider()

# ── 3. Pre-live checklist ──────────────────────────────────────────────────────
section("Pre-live Checklist", "All six must pass before live trading on Schwab is safe.")

# Map each check to an internal blocker key so we can use blocker_chip
_CHECK_BLOCKER_KEYS = [
    "KILL_SWITCH",
    "NOT_CONFIGURED",    # broker not Schwab
    "NOT_CONFIGURED",    # LIVE_TRADING_ENABLED
    "NOT_CONFIGURED",    # LIVE_TRADING_CONFIRMED
    "ORDER_LIMIT",
    "DAILY_LOSS_LIMIT",
]
for (check_label, ok), blocker_key in zip(checks_data, _CHECK_BLOCKER_KEYS):
    if ok:
        chip = eligibility_chip("ready")
    else:
        chip = blocker_chip(blocker_key)
    st.markdown(f"{chip} &nbsp; {check_label}", unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
if all_pass:
    st.success("All checks passed — system is ready for live trading.", icon="✅")
else:
    st.warning(
        f"{fail_count} check(s) failed. Fix them before enabling live trading on Schwab.",
        icon="⚠️",
    )

divider()

# ── 4. Per-symbol caps ─────────────────────────────────────────────────────────
section("Per-Symbol Capital Caps",
        "Limit how much a single signal can deploy. 0 = fall back to global settings.")

try:
    assignments = api.list_assignments()
except Exception as e:
    st.error(f"Cannot load assignments: {e}")
    assignments = []

if not assignments:
    empty_state("No assignments yet", "Add them on the Strategy page first.", icon="📋")
else:
    rows = []
    for a in assignments:
        cap = a.get("max_capital_usd")
        rows.append({
            "Symbol":   a["symbol"],
            "Strategy": a["strategy_name"].replace("_", " "),
            "System":   a["system"].title(),
            "Cap":      f"${cap:,.0f}" if cap else "— (global)",
            "Status":   "Active" if a["enabled"] else "Paused",
        })

    uncapped = [a for a in assignments if not a.get("max_capital_usd")]
    if uncapped:
        st.warning(
            f"{len(uncapped)} symbol(s) have no cap: "
            + ", ".join(a["symbol"] for a in uncapped)
            + " — they fall back to the global max position size from .env",
            icon="⚠️",
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol":   st.column_config.TextColumn("Symbol",   width="small"),
            "Strategy": st.column_config.TextColumn("Strategy", width="large"),
            "System":   st.column_config.TextColumn("System",   width="small"),
            "Cap":      st.column_config.TextColumn("$ Cap",    width="small"),
            "Status":   st.column_config.TextColumn("Status",   width="small"),
        },
    )

    st.markdown("**Set or update a cap**")
    cap_c1, cap_c2, cap_c3 = filter_cols(2, 2, 1)
    cap_sym = cap_c1.selectbox("Symbol", [a["symbol"] for a in assignments], key="cap_sym")
    current_cap = next((a.get("max_capital_usd") or 0 for a in assignments if a["symbol"] == cap_sym), 0)
    new_cap = cap_c2.number_input(
        "Max capital ($)", min_value=0, value=int(current_cap), step=100, key="cap_val",
        help="0 = remove cap and fall back to global settings",
    )
    cap_c3.write("")
    cap_c3.write("")
    if cap_c3.button("Save cap", type="primary", key="cap_save", use_container_width=True):
        try:
            api.set_assignment_cap(cap_sym, float(new_cap) if new_cap > 0 else None)
            st.success(f"Cap for {cap_sym} updated.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

divider()

# ── 5. Self-test ───────────────────────────────────────────────────────────────
section("Kill Switch Verification", "Confirms the scheduler reads the flag before placing orders.")

if st.button("Run kill switch test", key="ks_test"):
    with st.spinner("Testing…"):
        try:
            api.set_kill_switch(True)
            ks_on = api.risk_status()["kill_switch_active"]
            api.set_kill_switch(False)
            ks_off = api.risk_status()["kill_switch_active"]
            if ks_on and not ks_off:
                st.success(
                    "Test passed — activate set the flag ON, deactivate cleared it. "
                    "The scheduler reads this flag before every cycle.",
                    icon="✅",
                )
            else:
                st.error(f"Test failed — activate={ks_on}, deactivate={not ks_off}. Check /risk/status manually.")
        except Exception as e:
            st.error(f"Test error: {e}")
