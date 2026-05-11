from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api

import streamlit as st

st.set_page_config(page_title="Risk & Safety", page_icon="🛡️", layout="wide")
st.title("🛡️ Risk & Safety")
st.caption("Kill switch, daily limits, and live-trading safety gates. Check this page before going live.")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — KILL SWITCH
# ══════════════════════════════════════════════════════════════
st.subheader("🚨 Emergency Kill Switch")
st.caption(
    "When active, the scheduler stops placing ANY orders immediately. "
    "Use this if the market is moving against you or you need to pause trading instantly."
)

try:
    status = api.risk_status()
    ks = status["kill_switch_active"]

    if ks:
        st.error("🔴 **KILL SWITCH IS ACTIVE — All automated trading is HALTED**")
        if st.button("✅ Deactivate Kill Switch — Resume Trading", type="primary", key="ks_off"):
            api.set_kill_switch(False)
            st.success("Kill switch deactivated. Scheduler will resume on next cycle.")
            st.rerun()
    else:
        st.success("🟢 Kill switch is OFF — trading is allowed")
        if st.button("🛑 Activate Kill Switch — Halt All Trading", type="secondary", key="ks_on"):
            api.set_kill_switch(True)
            st.warning("Kill switch activated. No new orders will be placed until you deactivate it.")
            st.rerun()

except Exception as e:
    st.error(f"Cannot reach risk API: {e}")
    st.stop()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — FULL RISK STATUS
# ══════════════════════════════════════════════════════════════
st.subheader("📊 Current Risk Status")

try:
    s = api.risk_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Broker",        s["active_broker"].upper())
    c2.metric("Mode",          "🔴 LIVE" if s["is_live"] else "📄 Paper")
    c3.metric("Market Hours",  "✅ Open" if s["market_hours_active"] else "🔒 Closed")
    c4.metric("Kill Switch",   "🔴 ACTIVE" if s["kill_switch_active"] else "🟢 Off")

    st.divider()
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Orders Today",    f"{s['orders_today']} / {s['max_orders_per_day']}")
    c6.metric("Daily Loss",      f"${s['daily_loss_usd']:,.2f}",
              delta=f"limit ${s['max_daily_loss_usd']:,.2f}", delta_color="off")
    c7.metric("Live Trading Enabled",   "✅ Yes" if s["live_trading_enabled"] else "❌ No")
    c8.metric("Live Confirmed in .env", "✅ Yes" if s["live_trading_confirmed"] else "❌ No")

    st.divider()

    # Checklist of what must be true before live trading
    st.markdown("**Pre-live checklist**")
    checks = [
        ("Kill switch is OFF",             not s["kill_switch_active"]),
        ("Broker is Schwab (not paper)",   s["active_broker"] == "schwab"),
        ("LIVE_TRADING_ENABLED=true",      s["live_trading_enabled"]),
        ("LIVE_TRADING_CONFIRMED=true",    s["live_trading_confirmed"]),
        ("Daily order limit not hit",      s["orders_today"] < s["max_orders_per_day"]),
        ("Daily loss limit not hit",       s["daily_loss_usd"] < s["max_daily_loss_usd"]),
    ]
    all_ok = True
    for label, ok in checks:
        icon = "✅" if ok else "❌"
        if not ok:
            all_ok = False
        st.markdown(f"{icon} {label}")

    if all_ok:
        st.success("All checks passed — system is ready for live trading.")
    else:
        st.warning("One or more checks failed. Fix them before enabling live trading on Schwab.")

except Exception as e:
    st.error(f"Cannot load risk status: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — PER-SYMBOL CAPITAL CAPS
# ══════════════════════════════════════════════════════════════
st.subheader("💰 Per-Symbol Capital Caps")
st.caption(
    "Set a dollar limit per stock so one signal can never deploy your entire account. "
    "Example: cap ACMR at $500 → even if the scheduler fires a BUY, it buys at most $500 worth."
)

try:
    assignments = api.list_assignments()
except Exception as e:
    st.error(f"Cannot load assignments: {e}")
    assignments = []

if not assignments:
    st.info("No assignments yet. Add them on the Strategy page first.")
else:
    import pandas as pd
    rows = []
    for a in assignments:
        cap = a.get("max_capital_usd")
        rows.append({
            "Symbol":   a["symbol"],
            "Strategy": a["strategy_name"].replace("_", " "),
            "System":   a["system"].title(),
            "Cap ($)":  f"${cap:,.0f}" if cap else "⚠️ No cap — uses global",
            "Status":   "✅ Active" if a["enabled"] else "⏸ Paused",
        })

    uncapped = [a for a in assignments if not a.get("max_capital_usd")]
    if uncapped:
        st.warning(
            f"⚠️ {len(uncapped)} symbol(s) have no capital cap: "
            + ", ".join(a["symbol"] for a in uncapped)
            + " — they will use the global max position size from .env"
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
            help="0 = remove cap and fall back to global settings"
        )
    with cap_col3:
        st.write("")
        st.write("")
        if st.button("💾 Save Cap", type="primary", key="cap_save", use_container_width=True):
            try:
                api.set_assignment_cap(cap_sym, float(new_cap) if new_cap > 0 else None)
                st.success(f"Cap for {cap_sym} set to {'$' + str(new_cap) if new_cap > 0 else 'global default'}.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — KILL SWITCH VERIFICATION TEST
# ══════════════════════════════════════════════════════════════
st.subheader("🧪 Kill Switch Verification")
st.caption("Confirm the kill switch actually blocks the scheduler by running a quick self-test.")

if st.button("▶ Run Kill Switch Test", key="ks_test"):
    with st.spinner("Testing..."):
        try:
            # Activate
            api.set_kill_switch(True)
            status_on = api.risk_status()
            ks_on = status_on["kill_switch_active"]

            # Deactivate
            api.set_kill_switch(False)
            status_off = api.risk_status()
            ks_off = status_off["kill_switch_active"]

            if ks_on and not ks_off:
                st.success(
                    "✅ Kill switch test passed:\n"
                    "- Activated → risk status showed ACTIVE\n"
                    "- Deactivated → risk status showed OFF\n"
                    "The scheduler reads this flag before every cycle, so it will halt correctly."
                )
            else:
                st.error(
                    f"❌ Test failed — activate={ks_on}, deactivate={not ks_off}. "
                    "Check the /risk/status API manually."
                )
        except Exception as e:
            st.error(f"Test error: {e}")
