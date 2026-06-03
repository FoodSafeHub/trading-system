from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

import pandas as pd
import streamlit as st
import api
from _theme import apply_theme, market_status_bar
from _broker_routing import render_broker_routing_toggle

apply_theme("Strategy & Signals")
st.title("Strategy & Signals")

# Live US + India session clocks. A Zerodha (₹) symbol only trades when the
# India market is open; this makes that obvious at a glance.
market_status_bar()

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


# The 7 generic scanner strategies — 5 regime-aware + 2 legacy.
# Names are templated per symbol to match what _make_generic_configs_full(symbol)
# produces, so the scheduler can resolve them to the right StrategyConfig.
SCANNER_GENERIC_LABELS: list[tuple[str, str]] = [
    ("{sym}_RSI2_Mean_Reversion",      "RSI-2 Mean Reversion"),
    ("{sym}_EMA_MACD_Crossover",       "EMA + MACD Crossover"),
    ("{sym}_BB_Squeeze_Breakout",      "Bollinger Squeeze Breakout"),
    ("{sym}_Pullback_EMA50",           "Pullback to EMA(50)"),
    ("{sym}_VIX_Spike_Reversal",       "VIX Spike Reversal"),
    ("Legacy_{sym}_BB_Mean_Reversion", "Legacy: Bollinger Mean Reversion"),
    ("Legacy_{sym}_Fib_Pullback",      "Legacy: Fibonacci Pullback"),
]


def _generic_strategy_names_for(symbol: str) -> list[tuple[str, str]]:
    """Return [(name, display_label), ...] for the 7 generic strategies on this symbol."""
    sym = (symbol or "SYMBOL").upper().strip() or "SYMBOL"
    return [(tmpl.format(sym=sym), label) for tmpl, label in SCANNER_GENERIC_LABELS]


# Broker route options for the per-assignment override. "default" defers to
# the global active_broker / trade_routing toggle; the others pin orders for
# the symbol to a specific broker adapter.
BROKER_OPTIONS: list[tuple[str, str]] = [
    ("default", "Default (use global toggle)"),
    ("schwab",  "Schwab"),
    ("webull",  "Webull"),
    ("zerodha", "Zerodha (India)"),
    ("paper",   "Paper"),
]
BROKER_LABEL = {k: v for k, v in BROKER_OPTIONS}
BROKER_VALUES = [k for k, _ in BROKER_OPTIONS]


# Small Nifty-50 hint set for the UI badge only (not the routing authority —
# app.services.markets is the backend authority for actual order routing).
_NIFTY50_HINT = {
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "HCLTECH", "BAJAJFINSV", "TECHM", "GRASIM",
    "INDUSINDBK", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "DIVISLAB", "HINDALCO", "BPCL", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN",
}


def _is_india(symbol: str) -> bool:
    """India-symbol check for UI hints. Mirrors app.services.markets but kept
    local so the dashboard doesn't import backend modules."""
    s = (symbol or "").upper().strip()
    return s.startswith(("NSE:", "BSE:")) or s.endswith((".NS", ".BO")) or s in _NIFTY50_HINT


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
        shares_cap = a.get("max_shares")
        pos = pos_by_symbol.get(sym, {})
        qty = pos.get("quantity") or 0
        mkt_val = pos.get("market_value") or 0
        broker_route = (a.get("broker") or "default")
        rows.append({
            "Symbol":         sym,
            "Strategy":       a["strategy_name"].replace("_", " "),
            "System":         a["system"].title(),
            "Auto-trade":     "✅ Active" if a["enabled"] else "⏸ Paused",
            "Broker":         broker_route.title(),
            "$ Cap":          f"${cap:,.0f}" if cap else "(global)",
            "Shares Cap":     f"{shares_cap:g}" if shares_cap else "—",
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

    # ── Bulk broker routing ──────────────────────────────────────────────
    # "Select all → set broker" plus a one-click auto-route by market, so the
    # whole table can be pointed at the right broker without editing each row.
    st.markdown("**Bulk broker routing**")
    st.caption(
        "Route many symbols at once. **Auto-route by market** pins India (NSE) "
        "symbols to Zerodha and leaves every US symbol untouched — so your "
        "existing Schwab/Webull pins are preserved. Or pick specific symbols "
        "below and set them to one broker."
    )
    auto_col, auto_msg = st.columns([1, 3])
    with auto_col:
        if st.button("🌐 Auto-route by market", key="auto_route_btn",
                     use_container_width=True,
                     help="India (NSE) → Zerodha. US symbols left untouched."):
            try:
                res = api.bulk_set_assignment_broker(auto_by_market=True)
                st.success(f"Auto-routed {res.get('count', 0)} assignment(s) by market.")
                st.rerun()
            except Exception as e:
                st.error(f"Auto-route failed: {e}")
    with auto_msg:
        india_syms = [a["symbol"] for a in assignments
                      if _is_india(a["symbol"])]
        if india_syms:
            st.caption(f"India symbols detected: {', '.join(india_syms)} → Zerodha")
        else:
            st.caption("No India (NSE) symbols in your assignments yet.")

    all_syms = [a["symbol"] for a in assignments]
    sel_syms = st.multiselect(
        "Symbols to retag (leave empty + 'Apply' to set ALL)",
        all_syms,
        key="bulk_broker_syms",
    )
    bb1, bb2 = st.columns([2, 1])
    with bb1:
        bulk_broker = st.selectbox(
            "Set selected symbols to broker",
            BROKER_VALUES,
            format_func=lambda v: BROKER_LABEL.get(v, v),
            key="bulk_broker_pick",
        )
    with bb2:
        if st.button("Apply broker", key="bulk_broker_apply", use_container_width=True):
            try:
                res = api.bulk_set_assignment_broker(
                    symbols=sel_syms or all_syms, broker=bulk_broker)
                st.success(f"Set {res.get('count', 0)} assignment(s) → {BROKER_LABEL.get(bulk_broker, bulk_broker)}.")
                st.rerun()
            except Exception as e:
                st.error(f"Bulk set failed: {e}")

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
            cur_cap = sel_asgn.get("max_capital_usd")
            cur_shares = sel_asgn.get("max_shares")
            cap_bits = []
            if cur_cap:
                cap_bits.append(f"${cur_cap:,.0f}")
            if cur_shares:
                cap_bits.append(f"{cur_shares:g} shs")
            cap_label = " / ".join(cap_bits) if cap_bits else "global"
            st.caption(
                f"Current: **{sel_asgn['strategy_name'].replace('_',' ')}** "
                f"({sel_asgn['system']})  ·  Cap: {cap_label}"
            )

        # Edit caps row — dollar cap and shares cap. Dollar cap wins when both
        # are set; shares cap is the fallback used when the dollar cap is empty.
        ec1, ec2, ec3 = st.columns([2, 2, 1])
        with ec1:
            edit_cap = st.number_input(
                "Max capital ($)",
                min_value=0.0,
                value=float(cur_cap) if cur_cap else 0.0,
                step=100.0,
                key=f"edit_cap_{sel_sym}",
                help="0 = clear dollar cap. Wins over shares cap when both are set.",
            )
        with ec2:
            edit_shares = st.number_input(
                "Max shares (qty)",
                min_value=0.0,
                value=float(cur_shares) if cur_shares else 0.0,
                step=1.0,
                key=f"edit_shares_{sel_sym}",
                help="0 = clear shares cap. Used only when dollar cap is empty.",
            )
        with ec3:
            st.write("")
            st.write("")
            if st.button("💾 Update caps", key=f"update_caps_{sel_sym}",
                         use_container_width=True):
                try:
                    api.set_assignment_cap(
                        sel_sym,
                        float(edit_cap) if edit_cap and edit_cap > 0 else None,
                    )
                    api.set_assignment_shares(
                        sel_sym,
                        float(edit_shares) if edit_shares and edit_shares > 0 else None,
                    )
                    st.success(f"Caps updated for {sel_sym}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Update failed: {exc}")

        # Per-assignment broker override. "default" follows the global toggle;
        # anything else pins this symbol's orders to a specific broker so two
        # assignments can fire to different brokers in the same cycle.
        cur_broker = (sel_asgn.get("broker") or "default")
        br1, br2 = st.columns([4, 1])
        with br1:
            edit_broker = st.selectbox(
                "Broker route",
                BROKER_VALUES,
                index=BROKER_VALUES.index(cur_broker) if cur_broker in BROKER_VALUES else 0,
                format_func=lambda v: BROKER_LABEL.get(v, v.title()),
                key=f"edit_broker_{sel_sym}",
                help="Default = global toggle. Otherwise this symbol's orders "
                     "are pinned to the selected broker even if the global "
                     "toggle points somewhere else.",
            )
        with br2:
            st.write("")
            st.write("")
            if st.button("💾 Update broker", key=f"update_broker_{sel_sym}",
                         use_container_width=True,
                         disabled=(edit_broker == cur_broker)):
                try:
                    api.set_assignment_broker(sel_sym, edit_broker)
                    st.success(f"{sel_sym} → {BROKER_LABEL[edit_broker]}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Update failed: {exc}")
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
            ["perplexity", "bollinger", "scanner"],
            key="new_system",
            help="Perplexity = the 8 advanced swing strategies. "
                 "Bollinger / Scanner = the 7 generic strategies (RSI-2, EMA+MACD, "
                 "BB Squeeze, Pullback EMA50, VIX Spike + 2 legacy) plus anything "
                 "defined in strategies.json.",
        )
    with ac3:
        if new_system == "perplexity":
            new_strat = st.selectbox("Strategy", PERPLEXITY_STRATEGIES, key="new_strat_p")
        else:
            # Build the dropdown with the 7 generic strategies on top, then any
            # predefined strategies.json entries that match (or all of them as a
            # fallback when none are tagged for this symbol).
            generic_pairs = _generic_strategy_names_for(new_sym)
            generic_names = [n for n, _ in generic_pairs]
            try:
                boll_configs = api.strategy_configs()
                predefined = [c["name"] for c in boll_configs
                              if c.get("symbol", "").upper() == new_sym]
                if not predefined:
                    predefined = [c["name"] for c in boll_configs]
            except Exception:
                predefined = []
            # De-dup while preserving order: generics first, then predefined.
            seen: set[str] = set()
            combined: list[str] = []
            for n in generic_names + predefined:
                if n not in seen:
                    seen.add(n)
                    combined.append(n)

            # Display label maps name -> friendly display (only for the generics).
            label_for = {n: lbl for n, lbl in generic_pairs}
            new_strat = st.selectbox(
                "Strategy",
                combined if combined else ["—"],
                key="new_strat_b",
                format_func=lambda n: label_for.get(n, n.replace("_", " ")),
                help="Top 7 = generic strategies (RSI-2, EMA+MACD, BB Squeeze, "
                     "Pullback EMA50, VIX Spike + 2 legacy). Below = strategies.json "
                     "entries tagged for this symbol.",
            )

    cap_col, shares_col, broker_col = st.columns([1, 1, 2])
    with cap_col:
        new_cap = st.number_input(
            "Max capital ($)", min_value=0, value=0, step=100, key="new_cap",
            help="Dollar limit for this symbol. Wins over shares cap when both "
                 "are set. 0 = no dollar cap (falls back to shares cap or global).",
        )
    with shares_col:
        new_shares = st.number_input(
            "Max shares (qty)", min_value=0.0, value=0.0, step=1.0, key="new_shares",
            help="Shares cap. Used only when the dollar cap is empty. 0 = no shares cap.",
        )
    with broker_col:
        new_broker = st.selectbox(
            "Broker route",
            BROKER_VALUES,
            index=0,
            format_func=lambda v: BROKER_LABEL.get(v, v.title()),
            key="new_broker",
            help="Default = global toggle. Otherwise this symbol's orders are "
                 "pinned to the selected broker even if the global toggle "
                 "points somewhere else.",
        )

    new_notes = st.text_input(
        "Notes", placeholder="e.g. Best on 5y backtest, PF=3.26", key="new_notes",
    )

    max_capital_usd = float(new_cap) if new_cap and new_cap > 0 else None
    max_shares = float(new_shares) if new_shares and new_shares > 0 else None
    save_disabled = (not new_sym) or (not new_strat) or new_strat == "—"
    if st.button("💾 Save assignment", type="primary", key="save_asgn",
                 disabled=save_disabled):
        try:
            api.upsert_assignment(new_sym, new_system, new_strat, enabled=True,
                                  notes=new_notes,
                                  max_capital_usd=max_capital_usd,
                                  max_shares=max_shares,
                                  broker=new_broker)
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
    "Forces a single full scheduler cycle right now — runs every assigned symbol "
    "with its assigned strategy, plus the consensus pool. Dry run activates the "
    "kill switch so signals are evaluated but no orders are placed."
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
    with st.spinner("Running full scheduler cycle (all assigned symbols + consensus pool)…"):
        try:
            result = api.run_scheduler_now(dry_run=dry_run)
            cycle_error = None
        except Exception as e:
            result = {}
            cycle_error = str(e)

    if cycle_error:
        st.error(f"Cycle failed: {cycle_error}")
    else:
        if dry_run:
            st.info(
                "Dry run complete — all assigned symbols evaluated, no orders placed. "
                "Check Recent Signals below for what would fire."
            )
        else:
            st.success(
                "Cycle complete — orders placed for all actionable signals. "
                "Check Recent Signals and Recent Fills for results."
            )

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 5 — RECENT SIGNALS
# ════════════════════════════════════════════════════════════════
st.subheader("Recent signals")
st.caption(
    "Most recent strategy outputs. **Would fire** = this signal matches a live "
    "assignment, so the scheduler would actually trade it; everything else is "
    "just an observation. Defaults to actionable BUY/SELL only — HOLDs are noise."
)


def _signal_reason(direction: str, ind: dict) -> str:
    """Human one-liner from indicators_json — the *why*, not a raw blob."""
    if not ind:
        return "—"
    bits = []
    if "rsi2" in ind:
        bits.append(f"RSI2 {ind['rsi2']:.0f}")
    elif "rsi" in ind:
        bits.append(f"RSI {ind['rsi']:.0f}")
    if "dist_pct" in ind:
        sign = "+" if ind["dist_pct"] >= 0 else ""
        bits.append(f"{sign}{ind['dist_pct']:.1f}% vs EMA")
    if ind.get("squeeze") is True:
        bits.append("BB squeeze")
    if "macd" in ind and "macd_signal" in ind:
        bits.append("MACD>sig" if ind["macd"] > ind["macd_signal"] else "MACD<sig")
    if ind.get("atr_pct") not in (None, 0, 0.0):
        bits.append(f"ATR {ind['atr_pct']:.1f}%")
    return " · ".join(bits) if bits else "—"


# Build the set of (symbol, strategy_name) pairs the scheduler would actually act
# on — an enabled assignment. Drives the "would fire" flag. Source of truth is the
# live assignments table loaded at the top of the page; no hardcoded symbols.
_live_pairs = {
    (str(a.get("symbol", "")).upper(), str(a.get("strategy_name", "")))
    for a in assignments if a.get("enabled")
}

try:
    sigs = api.signals()
    if sigs:
        df = pd.DataFrame(sigs)

        # Parse indicators_json once into a readable reason, then drop the blob.
        if "indicators_json" in df.columns:
            import json as _json

            def _parse(v):
                if isinstance(v, dict):
                    return v
                try:
                    return _json.loads(v) if v else {}
                except Exception:
                    return {}
            _ind = df["indicators_json"].apply(_parse)
            df["why"] = [
                _signal_reason(str(d).upper(), i)
                for d, i in zip(df.get("direction", ""), _ind)
            ]

        # Would-fire flag: matches an enabled assignment AND is actionable.
        # Scheduler prefixes scanner/perplexity labels ("scanner:KO_RSI2…") but
        # the assignment stores the bare name — strip the prefix for comparison.
        def _would_fire(row) -> str:
            d = str(row.get("direction", "")).upper()
            if d not in ("BUY", "SELL"):
                return "—"
            sym = str(row.get("symbol", "")).upper()
            raw_name = str(row.get("strategy_name", ""))
            bare_name = raw_name.split(":", 1)[-1] if ":" in raw_name else raw_name
            return "🎯 yes" if (sym, bare_name) in _live_pairs or (sym, raw_name) in _live_pairs else "no"
        df["would_fire"] = df.apply(_would_fire, axis=1)

        # Acted-on / order link, made legible.
        if "acted_on" in df.columns:
            df["acted_on"] = df.apply(
                lambda r: f"✅ #{r['order_id']}" if r.get("acted_on") and r.get("order_id")
                else ("✅" if r.get("acted_on") else "—"),
                axis=1,
            )

        if "direction" in df.columns:
            df["direction"] = df["direction"].apply(
                lambda v: "🟢 BUY" if str(v).upper() == "BUY"
                else ("🔴 SELL" if str(v).upper() == "SELL" else "⬜ HOLD")
            )

        # Filters — symbol list is built from whatever actually came back.
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
        with fc1:
            sym_options = ["All"] + sorted(df["symbol"].dropna().unique().tolist()) \
                if "symbol" in df.columns else ["All"]
            sym_filter = st.selectbox("Symbol", sym_options, key="sig_sym")
        with fc2:
            dir_options = ["Actionable (BUY/SELL)", "All", "🟢 BUY", "🔴 SELL", "⬜ HOLD"]
            dir_filter = st.selectbox("Direction", dir_options, key="sig_dir")
        with fc3:
            only_fire = st.checkbox("Would-fire only", value=False, key="sig_fire",
                                    help="Only signals that match a live assignment.")
        with fc4:
            top_n = st.number_input("Show last N", min_value=10, max_value=500,
                                    value=50, step=10, key="sig_n")

        if sym_filter != "All" and "symbol" in df.columns:
            df = df[df["symbol"] == sym_filter]
        if dir_filter == "Actionable (BUY/SELL)":
            df = df[df["direction"].isin(["🟢 BUY", "🔴 SELL"])]
        elif dir_filter != "All" and "direction" in df.columns:
            df = df[df["direction"] == dir_filter]
        if only_fire:
            df = df[df["would_fire"] == "🎯 yes"]
        df = df.head(int(top_n))

        if df.empty:
            st.info("No signals match this filter. Switch Direction to 'All' to see HOLDs.")
        else:
            # Trader-first column order: when, what, would it trade, why, did it.
            priority = ["created_at", "symbol", "direction", "would_fire",
                        "strategy_name", "price_at_signal", "why", "acted_on"]
            drop = {"indicators_json", "strength", "order_id", "id"}
            show_cols = [c for c in priority if c in df.columns] + \
                        [c for c in df.columns if c not in priority and c not in drop]
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
