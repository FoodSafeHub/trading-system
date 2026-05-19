from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

import pandas as pd
import streamlit as st
import api

st.set_page_config(page_title="Strategy & Signals", page_icon="⚙️", layout="wide")
st.title("⚙️ Strategy & Signals")

def _load_perplexity_strategy_names() -> list[str]:
    try:
        data = api.perplexity_strategies()
        return [s["name"] for s in data if isinstance(s, dict) and "name" in s]
    except Exception:
        pass
    return [
        "EMA_Mean_Reversion", "MA_Crossover_RSI", "Breakout_Consolidation",
        "BB_Mean_Reversion", "Fib_Pullback_Support",
        "RSI_Swing_Reversal", "Supertrend_Swing", "BB_Breakout",
    ]

PERPLEXITY_STRATEGIES = _load_perplexity_strategy_names()

ALL_SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL", "JPM"]

# ══════════════════════════════════════════════════════════════
# SECTION 1 — SCHEDULER STATUS
# ══════════════════════════════════════════════════════════════
st.subheader("Auto-Scheduler")
try:
    sched = api.scheduler_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scheduler", "🟢 Running" if sched["running"] else "🔴 Stopped")
    c2.metric("Interval", f"{sched['interval_seconds']}s")
    c3.metric("Cycle Active", "Yes" if sched["cycle_active"] else "No")
    c4.metric("Next Run", sched["next_run"][:19].replace("T", " ") if sched["next_run"] else "—")

    if not sched["running"]:
        st.warning("Scheduler is not running. Restart the server to fix this.")
    else:
        b_on = sched.get("run_bollinger", True)
        p_on = sched.get("run_perplexity", True)
        active = []
        if b_on: active.append("Bollinger")
        if p_on: active.append("Perplexity")
        systems = " + ".join(active) if active else "none"
        st.success(f"Scheduler active — evaluating every {sched['interval_seconds']}s during market hours. "
                   f"Active systems: **{systems}**")
except Exception as e:
    st.warning(f"Cannot reach scheduler: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — STRATEGY ASSIGNMENTS (main feature)
# ══════════════════════════════════════════════════════════════
st.subheader("Symbol → Strategy Assignments")
st.caption(
    "Assign a specific strategy to each stock. When the scheduler runs, it uses ONLY that strategy "
    "for that stock — no consensus needed. Symbols without an assignment use the general consensus pool."
)

try:
    assignments = api.list_assignments()
except Exception as e:
    st.error(f"Cannot load assignments: {e}")
    assignments = []

assigned_symbols = {a["symbol"] for a in assignments}

# ── Current assignments table ─────────────────────────────────
if assignments:
    st.markdown("**Current Assignments**")
    rows = []
    for a in assignments:
        cap = a.get("max_capital_usd")
        rows.append({
            "Symbol":      a["symbol"],
            "System":      a["system"].title(),
            "Strategy":    a["strategy_name"].replace("_", " "),
            "Auto-Trade":  "✅ Active" if a["enabled"] else "⏸ Paused",
            "Cap ($)":     f"${cap:,.0f}" if cap else "Global",
            "Notes":       a.get("notes") or "—",
            "Set":         a["assigned_at"][:10] if a.get("assigned_at") else "—",
        })
    df_asgn = pd.DataFrame(rows)
    st.dataframe(df_asgn, use_container_width=True, hide_index=True)

    # Per-row actions
    st.markdown("**Manage assignments**")
    sel_sym = st.selectbox("Select symbol to manage", [a["symbol"] for a in assignments], key="mgmt_sym")
    sel_asgn = next((a for a in assignments if a["symbol"] == sel_sym), None)
    if sel_asgn:
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if sel_asgn["enabled"]:
                if st.button("⏸ Pause auto-trade", key="pause_btn"):
                    api.toggle_assignment(sel_sym, enabled=False)
                    st.rerun()
            else:
                if st.button("▶ Resume auto-trade", key="resume_btn", type="primary"):
                    api.toggle_assignment(sel_sym, enabled=True)
                    st.rerun()
        with mc2:
            confirm_del = st.checkbox(
                f"Confirm remove {sel_sym}",
                key=f"del_confirm_{sel_sym}",
            )
            if st.button("🗑 Remove assignment", key="del_btn", disabled=not confirm_del):
                api.delete_assignment(sel_sym)
                st.success(f"Removed assignment for {sel_sym}")
                st.rerun()
        with mc3:
            st.caption(f"Current: **{sel_asgn['strategy_name'].replace('_',' ')}** ({sel_asgn['system']})")
else:
    st.info("No assignments yet. Add one below to start auto-trading a symbol with a specific strategy.")

st.divider()

# ── Add / update assignment ────────────────────────────────────
st.markdown("**Add or Update Assignment**")
st.caption("Pick a symbol, choose Perplexity or Bollinger, then select the strategy. "
           "Run Compare All on the Perplexity page first to find the best strategy for that symbol.")

ac1, ac2, ac3 = st.columns([2, 2, 3])
with ac1:
    new_sym_input = st.text_input(
        "Symbol",
        placeholder="e.g. NVDA, AMD, NFLX — any US stock",
        key="new_sym_text",
        help="Type any stock ticker available on your broker. Not limited to the default list."
    ).upper().strip()
    new_sym = new_sym_input if new_sym_input else ""
with ac2:
    new_system = st.selectbox("System", ["perplexity", "bollinger"], key="new_system")
with ac3:
    if new_system == "perplexity":
        new_strat = st.selectbox("Strategy", PERPLEXITY_STRATEGIES, key="new_strat_p")
    else:
        try:
            boll_configs = api.strategy_configs()
            boll_names = [c["name"] for c in boll_configs if c.get("symbol", "").upper() == new_sym]
            if not boll_names:
                boll_names = [c["name"] for c in boll_configs]
        except Exception:
            boll_names = []
        new_strat = st.selectbox("Strategy", boll_names if boll_names else ["—"], key="new_strat_b")

cap_col, notes_col = st.columns([1, 2])
with cap_col:
    new_cap = st.number_input(
        "Max capital for this stock ($)",
        min_value=0, value=0, step=100,
        key="new_cap",
        help="Dollar limit for this stock. 0 = use global account settings. "
             "Example: set 500 so only $500 is ever deployed on this symbol."
    )
with notes_col:
    new_notes = st.text_input("Notes (optional)", placeholder="e.g. Best on 5y backtest, PF=3.26", key="new_notes")

max_capital_usd = float(new_cap) if new_cap and new_cap > 0 else None

if st.button("💾 Save Assignment", type="primary", key="save_asgn"):
    if not new_sym:
        st.warning("Enter a stock symbol first.")
    elif new_strat and new_strat != "—":
        try:
            api.upsert_assignment(new_sym, new_system, new_strat, enabled=True, notes=new_notes,
                                  max_capital_usd=max_capital_usd)
            st.success(f"Assigned **{new_strat.replace('_',' ')}** to **{new_sym}**. It will auto-trade on the next scheduler cycle.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")
    else:
        st.warning("Select a strategy first.")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — SCHEDULER SYSTEM TOGGLES
# ══════════════════════════════════════════════════════════════
st.subheader("Scheduler Config")
st.caption("Toggle which strategy systems participate in the general pool "
           "(applies to symbols without an assignment).")

try:
    sched_cfg = api.scheduler_status()
    col_b, col_p = st.columns(2)

    with col_b:
        bollinger_on = sched_cfg.get("run_bollinger", True)
        st.markdown("**Bollinger Strategies** (consensus pool)")
        st.caption("Bollinger Band strategies from strategies.json — requires 2+ to agree")
        new_bollinger = st.toggle("Enable Bollinger in scheduler", value=bollinger_on, key="tog_bollinger")

    with col_p:
        perplexity_on = sched_cfg.get("run_perplexity", True)
        st.markdown("**Perplexity Strategies** (consensus pool)")
        st.caption("All 8 advanced swing strategies — requires 2+ to agree on same symbol+direction")
        new_perplexity = st.toggle("Enable Perplexity in scheduler", value=perplexity_on, key="tog_perplexity")

    if new_bollinger != bollinger_on or new_perplexity != perplexity_on:
        try:
            api.update_scheduler_config(
                run_bollinger=new_bollinger if new_bollinger != bollinger_on else None,
                run_perplexity=new_perplexity if new_perplexity != perplexity_on else None,
            )
            st.success("Scheduler config updated.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to update: {e}")

    if not new_bollinger and not new_perplexity:
        st.warning("Both general pools disabled — only assigned symbols will trade.")

except Exception as e:
    st.warning(f"Could not load scheduler config: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — MANUAL CYCLE
# ══════════════════════════════════════════════════════════════
st.subheader("Manual Strategy Cycle")
st.caption("Runs immediately, regardless of market hours — useful for testing.")
if st.button("▶ Run Now", type="primary"):
    with st.spinner("Running strategy cycle..."):
        try:
            result = api.run_strategy()
            rows = result.get("results", [])
            min_agree = result.get("min_signal_agreement", 2)
            orders_placed = result.get("orders_placed", 0)

            if orders_placed > 0:
                st.success(f"✅ {orders_placed} order(s) placed.")
            else:
                st.warning(f"No orders placed — no symbol reached consensus ({min_agree} needed).")

            if rows:
                df = pd.DataFrame(rows)
                def _dir(v):
                    v = str(v).upper()
                    return "🟢 BUY" if v == "BUY" else ("🔴 SELL" if v == "SELL" else "⬜ HOLD")
                def _consensus(row):
                    if row.get("direction") == "HOLD": return "—"
                    met = row.get("consensus_met", False)
                    agree = row.get("agreement", "")
                    return f"✅ {agree}" if met else f"⏳ {agree}"
                if "direction" in df.columns:
                    df["direction"] = df["direction"].apply(_dir)
                if "consensus_met" in df.columns:
                    df["consensus"] = df.apply(_consensus, axis=1)
                show_cols = [c for c in ["strategy", "symbol", "direction", "price", "consensus", "order_status"]
                             if c in df.columns]
                st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Cycle failed: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — ALL CONFIGURED STRATEGIES (read-only reference)
# ══════════════════════════════════════════════════════════════
with st.expander("View all configured strategies", expanded=False):
    st.markdown("**Bollinger Strategies** (from strategies.json)")
    try:
        configs = api.strategy_configs()
        if configs:
            df_b = pd.DataFrame(configs)
            df_b["enabled"] = df_b["enabled"].map({True: "✅", False: "❌"})
            st.dataframe(df_b, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load Bollinger configs: {e}")

    st.markdown("**Perplexity Strategies**")
    try:
        perp = api._get("/perplexity/strategies")
        if perp:
            df_p = pd.DataFrame(perp)
            if "enabled" in df_p.columns:
                df_p["enabled"] = df_p["enabled"].map({True: "✅", False: "❌"})
            st.dataframe(df_p, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Could not load Perplexity strategies: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 6 — RECENT SIGNALS
# ══════════════════════════════════════════════════════════════
st.subheader("Recent Signals")
try:
    sigs = api.signals()
    if sigs:
        df = pd.DataFrame(sigs)
        if "direction" in df.columns:
            df["direction"] = df["direction"].apply(
                lambda v: "🟢 BUY" if str(v).upper() == "BUY" else ("🔴 SELL" if str(v).upper() == "SELL" else "⬜ HOLD")
            )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No signals yet. Run a strategy cycle or wait for the scheduler to fire during market hours.")
except Exception as e:
    st.warning(f"Could not load signals: {e}")
