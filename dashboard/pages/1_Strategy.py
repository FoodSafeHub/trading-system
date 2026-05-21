from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

import pandas as pd
import streamlit as st
import api
from _theme import apply_theme
from _broker_routing import render_broker_routing_toggle

apply_theme("Strategy & Signals")
st.title("Strategy & Signals")

# Broker-routing toggle: where live orders get sent. Mirrors the Day Trading
# page — autoscheduled assignments fire through whichever broker is selected.
render_broker_routing_toggle(key_suffix="strategy")


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def _load_perplexity_strategy_names() -> list[str]:
    try:
        data = api.perplexity_strategies()
        return [s["name"] for s in data if isinstance(s, dict) and "name" in s]
    except Exception:
        return [
            "EMA_Mean_Reversion", "MA_Crossover_RSI", "Breakout_Consolidation",
            "BB_Mean_Reversion", "Fib_Pullback_Support",
            "RSI_Swing_Reversal", "Supertrend_Swing", "BB_Breakout",
        ]


def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


PERPLEXITY_STRATEGIES = _load_perplexity_strategy_names()


# ════════════════════════════════════════════════════════════════
# SECTION 1 — SAFETY BANNER (kill switch + what the scheduler can do)
# ════════════════════════════════════════════════════════════════
st.subheader("Trading safety")
st.caption(
    "Snapshot of every gate that controls live order firing. Read this from top to "
    "bottom before changing assignments."
)

risk = _safe(api.risk_status, {})
sched = _safe(api.scheduler_status, {})
assignments = _safe(api.list_assignments, [])
positions = _safe(api.positions, [])

# Kill switch row
kill_active = bool(risk.get("kill_switch_active", False))
ksc1, ksc2 = st.columns([3, 1])
with ksc1:
    if kill_active:
        st.error(
            "🛑 **KILL SWITCH ON** — every order path is blocked. Manual orders, "
            "scheduler, scanner, autotrader all rejected at the risk gate."
        )
    else:
        st.success(
            "✅ Kill switch OFF — order paths follow their normal gates. "
            "Turn it on if you see anything unexpected below."
        )
with ksc2:
    if kill_active:
        if st.button("Deactivate kill switch", key="kill_off", help="Re-enable order firing."):
            api.set_kill_switch(False)
            st.rerun()
    else:
        if st.button("🛑 Activate kill switch", key="kill_on", type="primary",
                     help="Immediately blocks every order path at the risk gate."):
            api.set_kill_switch(True)
            st.rerun()

# Live-firing-path summary — the most important line on this page.
sched_running = bool(sched.get("running", False))
bollinger_on = bool(sched.get("run_bollinger", False))
perplexity_on = bool(sched.get("run_perplexity", False))
n_assigned_active = sum(1 for a in assignments if a.get("enabled"))
n_assigned_paused = sum(1 for a in assignments if not a.get("enabled"))
consensus_on = bollinger_on or perplexity_on

# Build a plain-English description of what the next cycle can fire.
firing_bits = []
if n_assigned_active:
    firing_bits.append(f"{n_assigned_active} assigned symbol(s)")
if consensus_on:
    systems = []
    if bollinger_on: systems.append("Bollinger")
    if perplexity_on: systems.append("Perplexity")
    firing_bits.append(f"consensus pool ({' + '.join(systems)}, 2 must agree)")

if kill_active:
    firing_summary = "Nothing — kill switch is ON."
    firing_color = "info"
elif not sched_running:
    firing_summary = "Nothing — scheduler is not running."
    firing_color = "warning"
elif not firing_bits:
    firing_summary = "Nothing — no active assignments and consensus pool is OFF."
    firing_color = "info"
else:
    firing_summary = "Can fire on: " + " + ".join(firing_bits) + "."
    firing_color = "warning" if consensus_on else "info"

st.markdown(f"**Next scheduler cycle ({sched.get('interval_seconds', '?')}s):**")
{"info": st.info, "warning": st.warning}.get(firing_color, st.info)(firing_summary)

# Compact status row underneath
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Scheduler", "🟢 Running" if sched_running else "🔴 Stopped")
sc2.metric("Interval", f"{sched.get('interval_seconds', '?')}s")
sc3.metric("Assignments", f"{n_assigned_active} active / {n_assigned_paused} paused")
sc4.metric(
    "Consensus pool",
    "ON" if consensus_on else "OFF",
    delta=("⚠ unbounded symbols" if consensus_on else "only assigned"),
    delta_color=("inverse" if consensus_on else "normal"),
)

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 2 — WHAT IS AUTO-TRADING (assignments + live exposure)
# ════════════════════════════════════════════════════════════════
st.subheader("What is auto-trading")
st.caption(
    "Each row is a symbol the scheduler will evaluate with one specific strategy. "
    "Live position and exposure come from the broker, not the DB — so you can see "
    "real dollars at risk next to the cap you set."
)

# Index live positions by symbol so we can join them onto assignments.
pos_by_symbol: dict[str, dict] = {}
for p in positions or []:
    sym = (p.get("symbol") or "").upper()
    if sym:
        pos_by_symbol[sym] = p

if assignments:
    rows = []
    for a in assignments:
        sym = a["symbol"]
        cap = a.get("max_capital_usd")
        pos = pos_by_symbol.get(sym, {})
        qty = pos.get("quantity") or 0
        mkt_val = pos.get("market_value") or 0
        rows.append({
            "Symbol":         sym,
            "Strategy":       a["strategy_name"].replace("_", " "),
            "System":         a["system"].title(),
            "Auto-trade":     "✅ Active" if a["enabled"] else "⏸ Paused",
            "Cap":            f"${cap:,.0f}" if cap else "(global)",
            "Held":           f"{qty:g}" if qty else "—",
            "Exposure":       f"${mkt_val:,.0f}" if mkt_val else "—",
            "Notes":          a.get("notes") or "",
        })
    df_asgn = pd.DataFrame(rows)
    st.dataframe(df_asgn, use_container_width=True, hide_index=True)

    # Bulk safety action — easier than walking every row.
    if n_assigned_active > 0:
        bcol1, bcol2 = st.columns([3, 1])
        with bcol1:
            st.caption(
                f"**Bulk action:** pause every active assignment in one click. "
                f"Use this if you're stepping away from the desk."
            )
        with bcol2:
            confirm_pause_all = st.checkbox("Confirm pause all", key="pause_all_confirm")
            if st.button("⏸ Pause all", disabled=not confirm_pause_all, key="pause_all_btn"):
                paused = 0
                for a in assignments:
                    if a.get("enabled"):
                        try:
                            api.toggle_assignment(a["symbol"], enabled=False)
                            paused += 1
                        except Exception as e:
                            st.error(f"Failed to pause {a['symbol']}: {e}")
                st.success(f"Paused {paused} assignment(s).")
                st.rerun()

    # Per-symbol management
    st.markdown("**Manage a single assignment**")
    sel_sym = st.selectbox(
        "Symbol",
        [a["symbol"] for a in assignments],
        key="mgmt_sym",
        help="Pause, resume, or remove a single assignment.",
    )
    sel_asgn = next((a for a in assignments if a["symbol"] == sel_sym), None)
    if sel_asgn:
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if sel_asgn["enabled"]:
                if st.button("⏸ Pause", key="pause_btn", use_container_width=True):
                    api.toggle_assignment(sel_sym, enabled=False)
                    st.rerun()
            else:
                if st.button("▶ Resume", key="resume_btn", type="primary",
                             use_container_width=True):
                    api.toggle_assignment(sel_sym, enabled=True)
                    st.rerun()
        with mc2:
            confirm_del = st.checkbox(f"Confirm remove {sel_sym}",
                                      key=f"del_confirm_{sel_sym}")
            if st.button("🗑 Remove", key="del_btn",
                         disabled=not confirm_del, use_container_width=True):
                api.delete_assignment(sel_sym)
                st.success(f"Removed assignment for {sel_sym}")
                st.rerun()
        with mc3:
            st.caption(
                f"Current: **{sel_asgn['strategy_name'].replace('_',' ')}** "
                f"({sel_asgn['system']})  ·  "
                f"Cap: {('$' + format(sel_asgn['max_capital_usd'], ',.0f')) if sel_asgn.get('max_capital_usd') else 'global'}"
            )
else:
    st.info(
        "No assignments yet. With the consensus pool **OFF**, the scheduler has "
        "nothing to fire on — add an assignment below to start auto-trading a "
        "specific symbol."
    )

# ── Add / update assignment ─────────────────────────────────────
with st.expander("Add or update an assignment", expanded=not assignments):
    st.caption(
        "Pick the symbol you want to auto-trade, choose Perplexity or Bollinger, "
        "then pick the strategy. Set a cap to limit dollars deployed on this "
        "symbol — 0 means use the global account settings."
    )

    ac1, ac2, ac3 = st.columns([2, 2, 3])
    with ac1:
        new_sym_input = st.text_input(
            "Symbol",
            placeholder="e.g. NVDA",
            key="new_sym_text",
            help="Any US stock available on your broker.",
        ).upper().strip()
        new_sym = new_sym_input
    with ac2:
        new_system = st.selectbox(
            "System",
            ["perplexity", "bollinger"],
            key="new_system",
            help="Perplexity = the 8 advanced swing strategies. "
                 "Bollinger = entries from strategies.json.",
        )
    with ac3:
        if new_system == "perplexity":
            new_strat = st.selectbox("Strategy", PERPLEXITY_STRATEGIES, key="new_strat_p")
        else:
            try:
                boll_configs = api.strategy_configs()
                boll_names = [c["name"] for c in boll_configs
                              if c.get("symbol", "").upper() == new_sym]
                if not boll_names:
                    boll_names = [c["name"] for c in boll_configs]
            except Exception:
                boll_names = []
            new_strat = st.selectbox(
                "Strategy",
                boll_names if boll_names else ["—"],
                key="new_strat_b",
                help="Bollinger strategies are filtered to those tagged for this symbol "
                     "when possible.",
            )

    cap_col, notes_col = st.columns([1, 2])
    with cap_col:
        new_cap = st.number_input(
            "Max capital ($)", min_value=0, value=0, step=100, key="new_cap",
            help="Dollar limit for this symbol. 0 = use global settings.",
        )
    with notes_col:
        new_notes = st.text_input(
            "Notes", placeholder="e.g. Best on 5y backtest, PF=3.26", key="new_notes",
        )

    max_capital_usd = float(new_cap) if new_cap and new_cap > 0 else None
    save_disabled = (not new_sym) or (not new_strat) or new_strat == "—"
    if st.button("💾 Save assignment", type="primary", key="save_asgn",
                 disabled=save_disabled):
        try:
            api.upsert_assignment(new_sym, new_system, new_strat, enabled=True,
                                  notes=new_notes, max_capital_usd=max_capital_usd)
            st.success(
                f"Assigned **{new_strat.replace('_',' ')}** to **{new_sym}**. "
                f"It will be evaluated on the next scheduler cycle."
            )
            st.rerun()
        except Exception as e:
            st.error(f"Failed: {e}")

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 3 — CONSENSUS POOL (collapsed — danger zone)
# ════════════════════════════════════════════════════════════════
with st.expander(
    "⚠ Consensus pool — fires on UNASSIGNED symbols (advanced)",
    expanded=consensus_on,  # only auto-expand if it's on
):
    st.caption(
        "When ON, the scheduler also evaluates a built-in pool of symbols "
        "(AAPL, MSFT, SPY, GOOGL, NVDA, TSLA, AMZN, META, AMD, JPM) using every "
        "Bollinger and/or Perplexity strategy, and fires when ≥2 strategies agree "
        "on the same symbol + direction. **This bypasses your assignments table.** "
        "Leave OFF unless you specifically want that behavior."
    )

    col_b, col_p = st.columns(2)
    with col_b:
        st.markdown("**Bollinger pool**")
        new_bollinger = st.toggle(
            "Enable Bollinger in consensus pool",
            value=bollinger_on, key="tog_bollinger",
            help="Runs every Bollinger config in strategies.json against the built-in symbol list.",
        )
    with col_p:
        st.markdown("**Perplexity pool**")
        new_perplexity = st.toggle(
            "Enable Perplexity in consensus pool",
            value=perplexity_on, key="tog_perplexity",
            help="Runs all 8 advanced swing strategies against the built-in symbol list.",
        )

    if new_bollinger != bollinger_on or new_perplexity != perplexity_on:
        try:
            api.update_scheduler_config(
                run_bollinger=new_bollinger if new_bollinger != bollinger_on else None,
                run_perplexity=new_perplexity if new_perplexity != perplexity_on else None,
            )
            st.success("Consensus pool updated. Reload the page to see the new state.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to update: {e}")

    if not new_bollinger and not new_perplexity:
        st.info("Both pools OFF — only assigned symbols will trade. Recommended default.")
    else:
        st.warning(
            "Consensus pool is ON. The scheduler can fire orders on symbols you "
            "did not assign as long as 2+ strategies agree."
        )

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 4 — MANUAL CYCLE (with dry-run + confirm)
# ════════════════════════════════════════════════════════════════
st.subheader("Run a cycle now")
st.caption(
    "Forces a single scheduler cycle right now, regardless of market hours. "
    "Use this to preview what would fire before enabling auto-trading."
)

rc1, rc2, rc3 = st.columns([2, 2, 2])
with rc1:
    dry_run = st.checkbox(
        "Dry run (recommended)", value=True, key="dry_run",
        help="If checked, the cycle still produces signals but **does not place orders**. "
             "Currently this only fully bypasses orders if the kill switch is ON or all "
             "firing paths are off — see the safety banner above.",
    )
with rc2:
    confirm_live = st.checkbox(
        "I understand this places live orders",
        value=False, key="confirm_live",
        disabled=dry_run,
        help="Required to run a cycle without dry-run.",
    )
with rc3:
    run_label = "▶ Run cycle (dry run)" if dry_run else "▶ Run cycle (LIVE)"
    run_disabled = not dry_run and not confirm_live
    run_clicked = st.button(
        run_label, type="primary", disabled=run_disabled,
        key="run_cycle_btn", use_container_width=True,
    )

if run_clicked:
    # Dry-run guard: temporarily activate the kill switch around the call.
    restored_kill = False
    if dry_run and not kill_active:
        try:
            api.set_kill_switch(True)
            restored_kill = True
        except Exception as e:
            st.error(f"Could not engage kill switch for dry run — aborting: {e}")
            st.stop()

    with st.spinner("Running strategy cycle..."):
        try:
            result = api.run_strategy()
        except Exception as e:
            result = {"error": str(e)}
        finally:
            if restored_kill:
                try:
                    api.set_kill_switch(False)
                except Exception as e:
                    st.error(
                        f"⚠ Could not deactivate kill switch after dry run: {e}. "
                        f"Restore manually in the safety banner above."
                    )

    if "error" in result:
        st.error(f"Cycle failed: {result['error']}")
    else:
        rows = result.get("results", [])
        min_agree = result.get("min_signal_agreement", 2)
        orders_placed = result.get("orders_placed", 0)

        if dry_run:
            st.info(
                f"Dry run complete. {len(rows)} strategy result(s) evaluated; "
                f"orders blocked by kill switch (none placed)."
            )
        elif orders_placed > 0:
            st.success(f"{orders_placed} order(s) placed.")
        else:
            st.warning(
                f"No orders placed — no symbol reached consensus "
                f"({min_agree} strategies must agree)."
            )

        if rows:
            df = pd.DataFrame(rows)

            def _dir(v):
                v = str(v).upper()
                return "🟢 BUY" if v == "BUY" else ("🔴 SELL" if v == "SELL" else "⬜ HOLD")

            def _consensus(row):
                if row.get("direction") == "HOLD":
                    return "—"
                met = row.get("consensus_met", False)
                agree = row.get("agreement", "")
                return f"✅ {agree}" if met else f"⏳ {agree}"

            if "direction" in df.columns:
                df["direction"] = df["direction"].apply(_dir)
            if "consensus_met" in df.columns:
                df["consensus"] = df.apply(_consensus, axis=1)
            show_cols = [c for c in ["strategy", "symbol", "direction", "price",
                                     "consensus", "order_status"] if c in df.columns]
            st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 5 — RECENT SIGNALS
# ════════════════════════════════════════════════════════════════
st.subheader("Recent signals")
st.caption("Most recent strategy outputs. These are observations, not orders.")

try:
    sigs = api.signals()
    if sigs:
        df = pd.DataFrame(sigs)
        if "direction" in df.columns:
            df["direction"] = df["direction"].apply(
                lambda v: "🟢 BUY" if str(v).upper() == "BUY"
                else ("🔴 SELL" if str(v).upper() == "SELL" else "⬜ HOLD")
            )

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sym_options = ["All"] + sorted(df["symbol"].dropna().unique().tolist()) \
                if "symbol" in df.columns else ["All"]
            sym_filter = st.selectbox("Symbol", sym_options, key="sig_sym")
        with fc2:
            dir_options = ["All", "🟢 BUY", "🔴 SELL", "⬜ HOLD"]
            dir_filter = st.selectbox("Direction", dir_options, key="sig_dir")
        with fc3:
            top_n = st.number_input("Show last N", min_value=10, max_value=500,
                                    value=50, step=10, key="sig_n")

        if sym_filter != "All" and "symbol" in df.columns:
            df = df[df["symbol"] == sym_filter]
        if dir_filter != "All" and "direction" in df.columns:
            df = df[df["direction"] == dir_filter]
        df = df.head(int(top_n))

        # Put the timestamp first so it's never missed.
        priority = ["created_at", "symbol", "direction", "price", "strategy_name",
                    "confidence"]
        show_cols = [c for c in priority if c in df.columns] + \
                    [c for c in df.columns if c not in priority]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info(
            "No signals yet. Run a cycle above (dry run is fine) or wait for the "
            "scheduler during market hours."
        )
except Exception as e:
    st.warning(f"Could not load signals: {e}")

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 6 — STRATEGY CATALOG (collapsed reference)
# ════════════════════════════════════════════════════════════════
with st.expander("Strategy catalog (read-only)", expanded=False):
    st.caption(
        "Every strategy this system knows about. Use this to look up names before "
        "creating an assignment above."
    )

    tab_b, tab_p = st.tabs(["Bollinger", "Perplexity"])

    with tab_b:
        try:
            configs = api.strategy_configs()
            if configs:
                df_b = pd.DataFrame(configs)
                if "enabled" in df_b.columns:
                    df_b["enabled"] = df_b["enabled"].map({True: "✅", False: "❌"})
                # Search box so the ~57-row list is usable.
                q = st.text_input("Filter Bollinger configs",
                                  placeholder="e.g. NVDA, BB_, mean_reversion",
                                  key="catalog_b_q").strip().lower()
                if q:
                    mask = df_b.apply(
                        lambda row: q in " ".join(str(v).lower() for v in row.values),
                        axis=1,
                    )
                    df_b = df_b[mask]
                st.dataframe(df_b, use_container_width=True, hide_index=True)
            else:
                st.info("No Bollinger configs loaded.")
        except Exception as e:
            st.warning(f"Could not load Bollinger configs: {e}")

    with tab_p:
        try:
            perp = api._get("/perplexity/strategies")
            if perp:
                df_p = pd.DataFrame(perp)
                if "enabled" in df_p.columns:
                    df_p["enabled"] = df_p["enabled"].map({True: "✅", False: "❌"})
                st.dataframe(df_p, use_container_width=True, hide_index=True)
            else:
                st.info("No Perplexity strategies loaded.")
        except Exception as e:
            st.warning(f"Could not load Perplexity strategies: {e}")
