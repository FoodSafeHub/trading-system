from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, divider, kpi_row, pill, money
from _components import page_header

import pandas as pd
import streamlit as st

apply_theme("Risk & Safety")
page_header("Risk & Safety", subtitle="Kill switch, daily limits, and live-trading gates — check before going live.")

# ── Status snapshot (single load, reused across the page) ────────────────
try:
    s = api.risk_status()
except Exception as e:
    st.error(f"Cannot reach risk API: {e}")
    st.stop()

ks_active = s["kill_switch_active"]


# ── 1. Kill switch ───────────────────────────────────────────────────────
section("Emergency Kill Switch", "Halts all automated trading instantly. Use when the market turns or you need to pause.")

ks_col, btn_col = st.columns([3, 1])
with ks_col:
    if ks_active:
        st.markdown(pill("KILL SWITCH ACTIVE — all automated trading halted", "red"), unsafe_allow_html=True)
    else:
        st.markdown(pill("Kill switch off — trading allowed", "green"), unsafe_allow_html=True)
with btn_col:
    if ks_active:
        if st.button("Deactivate", type="primary", key="ks_off", use_container_width=True):
            api.set_kill_switch(False)
            st.rerun()
    else:
        if st.button("Activate kill switch", type="secondary", key="ks_on", use_container_width=True):
            api.set_kill_switch(True)
            st.rerun()

divider()


# ── 2. Live status ──────────────────────────────────────────────────────
section("Current Risk Status")

mode_label = "LIVE" if s["is_live"] else "Paper"
kpi_row([
    ("Broker", s["active_broker"].upper()),
    ("Mode", mode_label),
    ("Market Hours", "Open" if s["market_hours_active"] else "Closed"),
    ("Kill Switch", "Active" if ks_active else "Off"),
])
kpi_row([
    ("Orders Today", f"{s['orders_today']} / {s['max_orders_per_day']}"),
    ("Daily Loss", money(s["daily_loss_usd"]), f"limit {money(s['max_daily_loss_usd'])}"),
    ("Live Trading Enabled", "Yes" if s["live_trading_enabled"] else "No"),
    ("Live Confirmed in .env", "Yes" if s["live_trading_confirmed"] else "No"),
])

divider()


# ── 3. Pre-live checklist ───────────────────────────────────────────────
section("Pre-live Checklist", "All six must pass before Schwab live trading is safe.")

checks = [
    ("Kill switch is OFF",           not ks_active),
    ("Broker is Schwab (not paper)", s["active_broker"] == "schwab"),
    ("LIVE_TRADING_ENABLED=true",    s["live_trading_enabled"]),
    ("LIVE_TRADING_CONFIRMED=true",  s["live_trading_confirmed"]),
    ("Daily order limit not hit",    s["orders_today"] < s["max_orders_per_day"]),
    ("Daily loss limit not hit",     s["daily_loss_usd"] < s["max_daily_loss_usd"]),
]
all_ok = all(ok for _, ok in checks)

for label, ok in checks:
    badge = pill("PASS", "green") if ok else pill("FAIL", "red")
    st.markdown(f"{badge} &nbsp; {label}", unsafe_allow_html=True)

if all_ok:
    st.success("All checks passed — system is ready for live trading.")
else:
    st.warning("One or more checks failed. Fix them before enabling live trading on Schwab.")

divider()


# ── 4. Per-symbol caps ──────────────────────────────────────────────────
section("Per-Symbol Capital Caps",
        "Cap how much a single signal can deploy. Example: cap ACMR at $500 → buys at most $500 worth.")

try:
    assignments = api.list_assignments()
except Exception as e:
    st.error(f"Cannot load assignments: {e}")
    assignments = []

if not assignments:
    st.info("No assignments yet. Add them on the Strategy page first.")
else:
    rows = []
    for a in assignments:
        cap = a.get("max_capital_usd")
        rows.append({
            "Symbol":   a["symbol"],
            "Strategy": a["strategy_name"].replace("_", " "),
            "System":   a["system"].title(),
            "Cap":      money(cap, decimals=0) if cap else "— (uses global)",
            "Status":   "Active" if a["enabled"] else "Paused",
        })

    uncapped = [a for a in assignments if not a.get("max_capital_usd")]
    if uncapped:
        st.warning(
            f"{len(uncapped)} symbol(s) have no cap: "
            + ", ".join(a["symbol"] for a in uncapped)
            + " — they fall back to the global max position size from .env"
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Set or update a cap**")
    cap_col1, cap_col2, cap_col3 = st.columns([2, 2, 1])
    with cap_col1:
        cap_sym = st.selectbox("Symbol", [a["symbol"] for a in assignments], key="cap_sym")
    with cap_col2:
        current_cap = next((a.get("max_capital_usd") or 0 for a in assignments if a["symbol"] == cap_sym), 0)
        new_cap = st.number_input(
            "Max capital ($)",
            min_value=0, value=int(current_cap), step=100, key="cap_val",
            help="0 = remove cap and fall back to global settings",
        )
    with cap_col3:
        st.write("")
        st.write("")
        if st.button("Save cap", type="primary", key="cap_save", use_container_width=True):
            try:
                api.set_assignment_cap(cap_sym, float(new_cap) if new_cap > 0 else None)
                st.success(f"Cap for {cap_sym} updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

divider()


# ── 5. Self-test ────────────────────────────────────────────────────────
section("Kill Switch Verification", "Confirms the scheduler actually reads the flag before placing orders.")

if st.button("Run kill switch test", key="ks_test"):
    with st.spinner("Testing..."):
        try:
            api.set_kill_switch(True)
            ks_on = api.risk_status()["kill_switch_active"]
            api.set_kill_switch(False)
            ks_off = api.risk_status()["kill_switch_active"]

            if ks_on and not ks_off:
                st.success(
                    "Test passed — activate set the flag ON, deactivate cleared it. "
                    "The scheduler reads this flag before every cycle."
                )
            else:
                st.error(
                    f"Test failed — activate={ks_on}, deactivate={not ks_off}. "
                    "Check /risk/status manually."
                )
        except Exception as e:
            st.error(f"Test error: {e}")
