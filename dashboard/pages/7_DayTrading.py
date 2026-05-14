from __future__ import annotations

import sys
import os

dashboard_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_root = os.path.abspath(os.path.join(dashboard_root, ".."))
sys.path.insert(0, workspace_root)
sys.path.insert(0, dashboard_root)

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from zoneinfo import ZoneInfo

from app.services.strategy.daytrading.market_open import (
    ET,
    is_market_open,
    is_pre_market,
    market_status,
    get_spy_regime,
)
from app.services.strategy.daytrading.brain.symbol_profiles import SymbolAnalyzer, _KNOWN_PROFILES
from app.services.strategy.daytrading.brain.strategy_selector import StrategySelector
from app.services.strategy.daytrading.runner import (
    fetch_intraday,
    profile_symbol,
    run_backtest,
    run_backtest_all,
    run_backtest_with_brain,
    run_scan,
    run_signals,
)
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP

st.set_page_config(page_title="Day Trading", page_icon="⚡", layout="wide")
st.title("⚡ Day Trading Signals")
st.caption(
    "5 intraday strategies on 5m and 15m bars. "
    "All signals expire at market close. No overnight holds."
)

STRATEGY_DESCRIPTIONS = {
    "ORBBreakout": (
        "Opening Range Breakout — price breaks above/below the first 15m range "
        "with volume confirmation. Best in BULL_OPEN regime."
    ),
    "VWAPMeanReversion": (
        "VWAP Mean Reversion — fades moves >0.5% below VWAP with RSI oversold + "
        "reversal wick. BULL_OPEN only. Target = VWAP."
    ),
    "EMAMomentum": (
        "EMA 9/21 Momentum — crossover on 15m bars confirmed by MACD. "
        "Adapts direction for BULL and BEAR regimes."
    ),
    "OpeningGapFade": (
        "Opening Gap Fade — fades 0.75–3% gaps that lack news volume. "
        "Statistical edge: >60% of such gaps partially fill within 90 minutes."
    ),
    "VolumeSpikeReversal": (
        "Volume Spike Reversal — catches capitulation moves driven by 3× volume spikes "
        "at RSI extremes. Works in all regimes."
    ),
}

DEFAULT_SYMBOLS = "SPY,QQQ,AAPL,TSLA,NVDA,MSFT,AMZN,META"

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_signals, tab_backtest, tab_sizer, tab_compare, tab_scanner, tab_config = st.tabs([
    "📡 Live Signals",
    "🔬 Backtest",
    "📐 Position Sizer",
    "📊 Compare All",
    "🔍 Scanner",
    "⚙️ Config",
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _regime_badge(regime: str) -> str:
    colors = {
        "BULL_OPEN": "🟢",
        "BEAR_OPEN": "🔴",
        "CHOPPY": "🟡",
        "PRE_MARKET": "🔵",
    }
    return f"{colors.get(regime, '⚪')} {regime}"


def _direction_color(direction: str) -> str:
    return {"BUY": "#00d4aa", "SELL": "#ff4b4b", "HOLD": "#888888"}.get(direction, "#888888")


def _render_signal_card(sig: dict, idx: int) -> None:
    direction = sig.get("direction", "HOLD")
    confidence = sig.get("confidence", 0.0)
    label = (
        f"**{sig['strategy']}** · {direction} · "
        f"conf {confidence:.0%} · {sig.get('timeframe', '')} · "
        f"{_regime_badge(sig.get('regime', ''))}"
    )
    with st.expander(label, expanded=idx == 0):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry", f"${sig['entry_price']:.2f}")
        c2.metric("Stop", f"${sig['stop_price']:.2f}")
        c3.metric("Target", f"${sig['target_price']:.2f}")
        rr = sig.get("r_multiple", 0)
        c4.metric("R:R", f"{rr:.1f}:1")

        st.progress(min(confidence, 1.0))
        st.caption(f"Reason: {sig.get('reason', '')}")
        st.info("⏱ **DAY ORDER — expires at close (3:45 PM ET)**")

        indicators = sig.get("indicators", {})
        if indicators:
            with st.expander("Indicator values"):
                ind_df = pd.DataFrame(
                    [{"Indicator": k, "Value": v} for k, v in indicators.items()]
                )
                st.dataframe(ind_df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=300)
def _cached_signals(symbol: str) -> dict:
    return run_signals(symbol, apply_brain=True)


@st.cache_data(ttl=600)
def _cached_backtest_brain(symbol: str, strategy: str, period: str, capital: float) -> dict:
    return run_backtest_with_brain(symbol, strategy, period, capital)


@st.cache_data(ttl=600)
def _cached_profile(symbol: str, period: str) -> dict:
    return profile_symbol(symbol, period)


@st.cache_data(ttl=300)
def _cached_backtest(symbol: str, strategy: str, period: str, capital: float) -> dict:
    return run_backtest(symbol, strategy, period, capital)


@st.cache_data(ttl=300)
def _cached_backtest_all(symbol: str, period: str, capital: float) -> list:
    return run_backtest_all(symbol, period, capital)


@st.cache_data(ttl=300)
def _cached_scan(symbols_str: str) -> list:
    syms = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
    return run_scan(syms)


def _equity_chart(equity_curve: list, trades: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, mode="lines",
        line=dict(color="#00d4aa", width=2),
        name="Equity",
    ))
    buy_x = [i for i, t in enumerate(trades) if t["direction"] == "BUY" and t["pnl"] > 0]
    buy_y = [equity_curve[i] for i in buy_x]
    sell_x = [i for i, t in enumerate(trades) if t["pnl"] <= 0]
    sell_y = [equity_curve[i] for i in sell_x]
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers",
                                  marker=dict(color="#00d4aa", size=8, symbol="triangle-up"),
                                  name="Win"))
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers",
                                  marker=dict(color="#ff4b4b", size=8, symbol="triangle-down"),
                                  name="Loss"))
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        title="Equity Curve",
    )
    return fig


def _metrics_row(m: dict) -> None:
    cols = st.columns(8)
    cols[0].metric("Net P&L", f"${m.get('net_pnl', m.get('total_pnl', 0)):,.0f}")
    cols[1].metric("Return", f"{m.get('total_return_pct', 0):.1f}%")
    cols[2].metric("Win Rate", f"{m.get('win_rate', 0):.1f}%")
    cols[3].metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
    cols[4].metric("Max Drawdown", f"{m.get('max_drawdown_pct', 0):.1f}%")
    cols[5].metric("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}")
    cols[6].metric("Trades", str(m.get("total_trades", 0)))
    cols[7].metric("Avg Hold", f"{m.get('avg_hold_bars', 0):.1f} bars")

    # P&L cost breakdown — only shown when commission/slippage data is present
    gross = m.get("gross_pnl")
    comm = m.get("total_commission", 0.0)
    slippage = m.get("total_slippage", 0.0)
    if gross is not None and (comm > 0 or slippage > 0):
        net = m.get("net_pnl", m.get("total_pnl", 0))
        st.caption(
            f"Gross P&L: **${gross:,.2f}** &nbsp;|&nbsp; "
            f"Commission: **-${comm:,.2f}** &nbsp;|&nbsp; "
            f"Slippage: **-${slippage:,.2f}** &nbsp;|&nbsp; "
            f"Net P&L: **${net:,.2f}**"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Signals
# ─────────────────────────────────────────────────────────────────────────────
with tab_signals:
    col_sym, col_btn = st.columns([3, 1])
    symbol_input = col_sym.text_input("Symbol", value="SPY", key="signals_symbol").upper()
    get_btn = col_btn.button("▶ Get Signals", use_container_width=True)

    # Market status banner
    status = market_status()
    now_str = status.get("time_et", "")
    if status["is_open"]:
        st.success(f"🟢 Market OPEN — {now_str}")
    elif is_pre_market():
        st.info(f"🔵 PRE-MARKET — {now_str}. Live signals only generated 9:30–3:45 PM ET.")
    else:
        st.warning(f"🔴 Market CLOSED — {now_str}. Showing most recent data.")

    st.info(
        "Signals based on last completed 5m bar. "
        "Refresh every 5 minutes during market hours."
    )

    if get_btn or st.session_state.get("_dt_signals_loaded"):
        st.session_state["_dt_signals_loaded"] = True
        with st.spinner("Fetching intraday data and running strategies…"):
            result = _cached_signals(symbol_input)

        regime = result.get("regime", "CHOPPY")
        signals = result.get("signals", [])
        rejected = result.get("rejected_signals", [])
        brain = result.get("brain", {})

        # ── Brain Status Panel ────────────────────────────────────────────────
        if brain:
            st.markdown("---")
            st.markdown("### 🧠 Brain Status")
            ms_state = brain.get("market_state", "UNKNOWN")
            ms_conf = brain.get("state_confidence", 0)
            kill = brain.get("kill_switch", False)
            size_mult = brain.get("size_multiplier", 1.0)
            trades_today = brain.get("trades_today", 0)
            losses_row = brain.get("losses_in_a_row", 0)
            daily_pnl = brain.get("daily_pnl_pct", 0.0)

            _STATE_EMOJI = {
                "TREND_UP": "🟢", "TREND_DOWN": "🔴",
                "CHOPPY": "🟡", "HIGH_VOL": "🟠", "NEWS_RISK": "🔴", "UNKNOWN": "⚪",
            }
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Market State", f"{_STATE_EMOJI.get(ms_state, '⚪')} {ms_state}", f"conf {ms_conf:.0%}")
            b2.metric("Kill Switch", "🔴 ON" if kill else "🟢 OFF")
            b3.metric("Size Multiplier", f"{size_mult:.0%}")
            b4.metric("Trades Today", str(trades_today))
            b5.metric("Losses in a Row", str(losses_row))

            if kill:
                st.error(f"**KILL SWITCH ACTIVE**: {brain.get('kill_switch_reason', '')}")

            enabled = brain.get("enabled_strategies", [])
            disabled = brain.get("disabled_strategies", [])
            if enabled or disabled:
                brow1, brow2 = st.columns(2)
                with brow1:
                    st.markdown("**Allowed strategies:**")
                    for s in enabled:
                        st.success(f"✅ {s}")
                with brow2:
                    st.markdown("**Blocked strategies:**")
                    for s in disabled:
                        st.error(f"❌ {s}")

            reasons = brain.get("state_reasons", [])
            if reasons:
                with st.expander("Why this market state?"):
                    for r in reasons:
                        st.caption(f"• {r}")

            st.caption(brain.get("routing_summary", ""))
            st.markdown("---")

        # ── Signal metrics ────────────────────────────────────────────────────
        buys = [s for s in signals if s["direction"] == "BUY"]
        sells = [s for s in signals if s["direction"] == "SELL"]
        holds = [s for s in signals if s["direction"] == "HOLD"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("BUY Signals", len(buys))
        m2.metric("SELL Signals", len(sells))
        m3.metric("HOLD", len(holds))
        m4.metric("Regime", _regime_badge(regime))
        m5.metric("Rejected by Brain", len(rejected))

        if not signals:
            st.info("No signals passed brain filters for current bars.")
        else:
            st.markdown("### ✅ Accepted Signals")
            for i, sig in enumerate(signals):
                _render_signal_card(sig, i)
                brain_reason = sig.get("brain_reason", "")
                brain_size = sig.get("brain_size_multiplier", 1.0)
                if brain_reason:
                    st.caption(f"Brain: {brain_reason}  |  Size: {brain_size:.0%}")

        if rejected:
            with st.expander(f"❌ {len(rejected)} signal(s) rejected by brain"):
                for sig in rejected:
                    st.markdown(
                        f"**{sig.get('strategy')}** — {sig.get('direction')} @ "
                        f"${sig.get('entry_price', 0):.2f}  |  "
                        f"Reason: {sig.get('brain_reason', 'filtered')}"
                    )

        col_ref, _ = st.columns([2, 8])
        if col_ref.button("🔄 Refresh Now"):
            _cached_signals.clear()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Backtest
# ─────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.caption(
        "5m backtest: max 60 days (yfinance limit). "
        "15m backtest: up to 2 years."
    )
    bc1, bc2, bc3, bc4 = st.columns(4)
    bt_symbol = bc1.text_input("Symbol", value="SPY", key="bt_symbol").upper()
    bt_period = bc2.selectbox(
        "Period", ["30d", "60d", "90d", "180d", "730d"], index=1, key="bt_period"
    )
    bt_capital = bc3.number_input("Capital ($)", value=10_000, step=1_000, key="bt_capital")

    # Profile button — placed next to symbol
    profile_btn = bc4.button("📊 Profile Symbol", key="profile_btn", use_container_width=True)
    if profile_btn:
        with st.spinner(f"Profiling {bt_symbol}…"):
            prof = _cached_profile(bt_symbol, bt_period)
            st.session_state["bt_profile"] = prof

    # ── Symbol Profile Card ───────────────────────────────────────────────────
    _prof = st.session_state.get("bt_profile")
    if _prof and not _prof.get("error"):
        _VOL_ICON = "🔴" if _prof["volatility_pct"] >= 1.8 else ("🟡" if _prof["volatility_pct"] >= 1.0 else "🟢")
        _LIQ_ICON = "✅" if _prof["liquidity_score"] >= 0.9 else ("🟡" if _prof["liquidity_score"] >= 0.7 else "⚠️")
        _LIQ_LABEL = "Excellent" if _prof["liquidity_score"] >= 0.9 else ("Good" if _prof["liquidity_score"] >= 0.7 else "Low")
        _TREND_ICON = "📈" if _prof["trend_strength"] >= 0.6 else ("↔️" if _prof["trend_strength"] >= 0.4 else "🔄")
        _TREND_LABEL = "Strong" if _prof["trend_strength"] >= 0.6 else ("Moderate" if _prof["trend_strength"] >= 0.4 else "Weak / Mean-reverting")

        with st.container(border=True):
            st.markdown(f"#### 📊 {bt_symbol} — Symbol Profile")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric(f"{_VOL_ICON} Volatility (ATR%)", f"{_prof['volatility_pct']:.2f}%")
            p2.metric(f"{_LIQ_ICON} Liquidity", _LIQ_LABEL)
            p3.metric(f"{_TREND_ICON} Trend Strength", _TREND_LABEL)
            p4.metric("Gap Days", f"{_prof['gap_frequency_pct']:.0f}%")

            st.caption(f"**Market type:** {_prof['market_type']}")
            st.caption(f"**Hours pattern:** {_prof['trading_hours_pattern']}")

            best = _prof.get("best_strategies", [])
            avoid = _prof.get("avoid_strategies", [])
            if best:
                st.success(f"✅ **Recommended:** {' · '.join(best)}")
            if avoid:
                st.error(f"❌ **Not recommended:** {' · '.join(avoid)}")

            with st.expander("Full analysis notes"):
                st.info(_prof.get("notes", ""))

    # ── Smart strategy selector (populated from profile if available) ─────────
    st.markdown("---")
    _all_strategy_names = [s.name for s in ALL_STRATEGIES]

    if _prof and not _prof.get("error"):
        best_strategies = _prof.get("best_strategies", _all_strategy_names)
        avoid_strategies = _prof.get("avoid_strategies", [])

        # Reorder: recommended first, then others, avoid last
        ordered = (
            [s for s in best_strategies if s in _all_strategy_names]
            + [s for s in _all_strategy_names if s not in best_strategies and s not in avoid_strategies]
            + [s for s in avoid_strategies if s in _all_strategy_names]
        )
        # Label each option with a fit indicator
        def _strat_label(name: str) -> str:
            if name in best_strategies[:1]:
                return f"⭐ {name} (best fit)"
            if name in best_strategies:
                return f"✅ {name} (good fit)"
            if name in avoid_strategies:
                return f"⚠️ {name} (not recommended)"
            return name

        labeled = [_strat_label(n) for n in ordered]
        _sel_idx = st.radio(
            "Strategy (ordered by fit for this symbol)",
            range(len(ordered)),
            format_func=lambda i: labeled[i],
            horizontal=False,
            key="bt_strategy_idx",
        )
        bt_strategy = ordered[_sel_idx]

        # Warn if user picks an avoid strategy
        if bt_strategy in avoid_strategies:
            st.warning(
                f"⚠️ **{bt_strategy}** is not recommended for **{bt_symbol}**. "
                f"{_prof.get('notes', '')[:200]}…"
            )
    else:
        bt_strategy = st.selectbox(
            "Strategy", _all_strategy_names, key="bt_strategy"
        )

    sz1, sz2 = st.columns(2)
    bt_sizing = sz1.radio(
        "Position Sizing", ["Risk-Based (% of equity)", "Fixed %"], key="bt_sizing"
    )
    bt_pos_pct = sz2.slider(
        "Position Size % of equity", 0.1, 1.0, 0.95, 0.05, key="bt_pos_pct"
    )

    bt_mode = st.radio(
        "Mode", ["Raw Strategy", "Brain-Filtered", "Side-by-Side Comparison"],
        horizontal=True, key="bt_mode",
    )

    if st.button("▶ Run Backtest", key="run_bt"):
        with st.spinner("Running bar-by-bar simulation…"):
            if bt_mode == "Side-by-Side Comparison":
                cmp_result = _cached_backtest_brain(bt_symbol, bt_strategy, bt_period, float(bt_capital))
                st.session_state["bt_cmp_result"] = cmp_result
                st.session_state["bt_single_result"] = None
                st.session_state["bt_mode_stored"] = "Side-by-Side Comparison"
            elif bt_mode == "Brain-Filtered":
                cmp_result = _cached_backtest_brain(bt_symbol, bt_strategy, bt_period, float(bt_capital))
                st.session_state["bt_single_result"] = cmp_result.get("brain_filtered", {})
                st.session_state["bt_cmp_result"] = None
                st.session_state["bt_mode_stored"] = "Brain-Filtered"
            else:
                st.session_state["bt_single_result"] = _cached_backtest(bt_symbol, bt_strategy, bt_period, float(bt_capital))
                st.session_state["bt_cmp_result"] = None
                st.session_state["bt_mode_stored"] = "Raw Strategy"

    # ── Render results (persisted in session_state) ───────────────────────────
    _stored_mode = st.session_state.get("bt_mode_stored", "")

    def _render_single_result(bt_result: dict) -> None:
        if not bt_result:
            return
        if "error" in bt_result:
            st.error(bt_result["error"])
            return

        # ── Config auto-tuning banner ─────────────────────────────────────────
        adj = bt_result.get("config_adjustment", {})
        if adj.get("has_changes"):
            with st.expander("⚙️ Parameters auto-tuned for this symbol", expanded=True):
                st.caption(adj.get("reason_summary", ""))
                for change in adj.get("changes", []):
                    st.markdown(f"• {change}")

        # ── Zero-trades: rich explanation ─────────────────────────────────────
        if bt_result.get("metrics", {}).get("total_trades", 0) == 0:
            exp = bt_result.get("explanation", {})
            sel = bt_result.get("selection", {})
            alt = bt_result.get("recommended_alternative", "")

            if exp:
                urgency = exp.get("urgency", "low")
                urgency_fn = st.error if urgency == "high" else (st.warning if urgency == "medium" else st.info)
                urgency_fn(
                    f"**0 trades generated** — {exp.get('root_cause_label', 'No setups found')}. "
                    f"Fit score: {exp.get('fit_score', 0):.0%}"
                )

                with st.expander("🔍 Why no trades fired? (click to diagnose)", expanded=True):
                    st.markdown(exp.get("why_no_trades", ""))

                    st.markdown("#### Recommendations")
                    for rec in exp.get("recommendations", []):
                        st.markdown(f"• {rec}")

                    if alt:
                        st.markdown(f"#### 🔄 Recommended alternative: **{alt}**")
                        alt_period = exp.get("alternative_period", "90d")
                        if st.button(
                            f"▶ Try {alt} on {bt_symbol} ({alt_period})",
                            key=f"try_alt_{alt}",
                        ):
                            with st.spinner(f"Running {alt}…"):
                                alt_result = run_backtest(bt_symbol, alt, alt_period, 10_000.0)
                                st.session_state["bt_single_result"] = alt_result
                                st.session_state["bt_mode_stored"] = "Raw Strategy"
                            st.rerun()

                if sel:
                    st.markdown("#### Strategy fit scores for this symbol")
                    scores = sel.get("symbol_fit_scores", {})
                    score_rows = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    for strat_name, score in score_rows:
                        bar_color = "#00d4aa" if score >= 0.6 else ("#FFD700" if score >= 0.35 else "#ff4b4b")
                        st.markdown(
                            f"**{strat_name}** — {score:.0%} "
                            f"<span style='color:{bar_color}'>{'█' * int(score * 10)}</span>",
                            unsafe_allow_html=True,
                        )
            else:
                st.warning(
                    "No trades generated. Try a longer period or a more volatile symbol (TSLA, NVDA)."
                )
            return

        m = bt_result.get("metrics", {})
        _metrics_row(m)

        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Best Hour", str(m.get("best_hour", "N/A")))
        ex2.metric("Best DoW", str(m.get("best_day_of_week", "N/A")))
        ex3.metric("Avg P&L/Trade", f"${m.get('avg_pnl_per_trade', 0):.2f}")
        ex4.metric("Max Consec. Losses", str(m.get("max_consecutive_losses", 0)))

        eq_curve = bt_result.get("equity_curve", [])
        trades_list = bt_result.get("trades", [])
        if eq_curve and len(eq_curve) > 1:
            st.plotly_chart(_equity_chart(eq_curve, trades_list), use_container_width=True)

        analysis = bt_result.get("analysis", {})
        if analysis:
            st.markdown("---")
            st.markdown("### 🧠 Trade Analyzer")

            strengths = analysis.get("strengths", [])
            weaknesses = analysis.get("weaknesses", [])
            diagnosis = analysis.get("diagnosis", [])

            if strengths:
                with st.expander("✅ Strengths", expanded=True):
                    for s in strengths:
                        st.success(s)
            if weaknesses:
                with st.expander("⚠️ Weaknesses", expanded=True):
                    for w in weaknesses:
                        st.warning(w)
            if diagnosis:
                with st.expander("🔍 Observations"):
                    for d in diagnosis:
                        st.info(d)

            ra1, ra2, ra3, ra4 = st.columns(4)
            ra1.metric("Expectancy/trade", f"{analysis.get('expectancy', 0):+.3f}%")
            ra2.metric("Payoff Ratio", f"{analysis.get('payoff_ratio', 0):.2f}:1")
            ra3.metric("Avg Win", f"{analysis.get('avg_win_pct', 0):.3f}%")
            ra4.metric("Avg Loss", f"-{analysis.get('avg_loss_pct', 0):.3f}%")

            hold1, hold2 = st.columns(2)
            hold1.metric("Avg Win Hold (bars)", f"{analysis.get('avg_win_hold_bars', 0):.1f}")
            hold2.metric("Avg Loss Hold (bars)", f"{analysis.get('avg_loss_hold_bars', 0):.1f}")

            regime_data = analysis.get("regime_breakdown", {})
            if regime_data:
                st.markdown("#### Performance by Regime")
                df_regime = pd.DataFrame([
                    {"Regime": k, "Trades": v["trades"], "Win%": v["win_rate"],
                     "Avg P&L%": v["avg_pnl"], "Total P&L ($)": v["total_pnl"]}
                    for k, v in regime_data.items()
                ])
                st.dataframe(df_regime, use_container_width=True, hide_index=True)

            hourly_data = analysis.get("hourly_performance", {})
            if hourly_data:
                st.markdown("#### Win Rate by Hour of Day")
                hours = sorted(hourly_data.keys())
                wr_vals = [hourly_data[h]["win_rate"] for h in hours]
                trade_counts = [hourly_data[h]["trades"] for h in hours]
                colors = ["#00d4aa" if w >= 55 else ("#FFD700" if w >= 45 else "#ff4b4b") for w in wr_vals]
                fig_h = go.Figure(go.Bar(
                    x=[f"{h}:00" for h in hours], y=wr_vals,
                    marker_color=colors,
                    text=[f"{c}T" for c in trade_counts],
                    textposition="outside",
                ))
                fig_h.update_layout(
                    template="plotly_dark", height=260,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(title="Win%", range=[0, 105]),
                    xaxis=dict(title="Entry Hour (ET)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_h, use_container_width=True)

            outcome_data = analysis.get("outcome_breakdown", {})
            if outcome_data:
                st.markdown("#### Outcome Breakdown")
                df_out = pd.DataFrame([
                    {"Outcome": k, "Count": v["count"], "Win%": v["win_rate"], "Avg P&L%": v["avg_pnl_pct"]}
                    for k, v in outcome_data.items()
                ])
                st.dataframe(df_out, use_container_width=True, hide_index=True)

        if trades_list:
            st.markdown("---")
            st.markdown("### Trade Log")
            df_trades = pd.DataFrame(trades_list)
            display_cols = [
                "date", "direction", "entry_price", "exit_price",
                "entry_time", "exit_time", "hold_bars",
                "pnl", "pnl_pct", "outcome", "regime", "confidence",
            ]
            display_cols = [c for c in display_cols if c in df_trades.columns]
            st.dataframe(
                df_trades[display_cols].style.map(
                    lambda v: "color: #00d4aa" if isinstance(v, (int, float)) and v > 0
                    else ("color: #ff4b4b" if isinstance(v, (int, float)) and v < 0 else ""),
                    subset=["pnl", "pnl_pct"],
                ),
                use_container_width=True,
            )

    if _stored_mode == "Side-by-Side Comparison":
        cmp_data = st.session_state.get("bt_cmp_result", {})
        if cmp_data:
            raw_result = cmp_data.get("raw", {})
            brain_result = cmp_data.get("brain_filtered", {})
            comparison = cmp_data.get("comparison", {})

            # ── Comparison summary table ──────────────────────────────────────
            st.markdown("### 🧠 Brain Filter — Side-by-Side Comparison")
            st.caption("Same period, same symbol. Brain applies market state routing, risk limits, and execution checks.")

            if comparison:
                label_map = {
                    "total_trades": "Total Trades",
                    "win_rate": "Win Rate %",
                    "profit_factor": "Profit Factor",
                    "total_pnl": "Total P&L ($)",
                    "max_drawdown_pct": "Max Drawdown %",
                    "sharpe_ratio": "Sharpe Ratio",
                }
                cmp_rows = []
                for key, label in label_map.items():
                    if key in comparison:
                        raw_val = comparison[key].get("raw", 0)
                        brain_val = comparison[key].get("brain", 0)
                        delta = round(brain_val - raw_val, 3) if isinstance(raw_val, (int, float)) else 0
                        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
                        cmp_rows.append({
                            "Metric": label,
                            "Raw": raw_val,
                            "Brain Filtered": brain_val,
                            "Δ": f"{arrow} {abs(delta)}",
                        })
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

            # ── Raw vs Brain equity curves side by side ───────────────────────
            col_raw, col_brain = st.columns(2)
            with col_raw:
                st.markdown("#### Raw Strategy")
                _render_single_result(raw_result)
            with col_brain:
                st.markdown("#### Brain-Filtered")
                bfm = brain_result.get("metrics", {})
                if bfm.get("total_trades", 0) == 0:
                    st.info("Brain filter removed all trades in this period — strategy not aligned with market state rules.")
                else:
                    _render_single_result(brain_result)

    elif _stored_mode in ("Raw Strategy", "Brain-Filtered"):
        result = st.session_state.get("bt_single_result", {})
        if result is not None:
            label = "🧠 Brain-Filtered Results" if _stored_mode == "Brain-Filtered" else "Raw Strategy Results"
            st.markdown(f"### {label}")
            _render_single_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Position Sizer
# ─────────────────────────────────────────────────────────────────────────────
with tab_sizer:
    st.markdown("### Day Trade Position Sizer")
    st.info(
        "Day traders should risk **max 0.5–1% per trade** to survive "
        "losing streaks. PDT rule: max 3 day trades per 5 rolling days "
        "unless account > $25,000."
    )

    ps1, ps2 = st.columns(2)
    ps_account = ps1.number_input("Account Size ($)", value=25_000, step=1_000)
    ps_risk_pct = ps2.slider(
        "Max Risk per Trade (%)", 0.1, 2.0, 0.5, 0.1,
        help="Recommended: 0.5–1.0% for day trading"
    )

    ps3, ps4, ps5 = st.columns(3)
    ps_entry = ps3.number_input("Entry Price ($)", value=100.0, step=0.01)
    ps_stop = ps4.number_input("Stop Price ($)", value=99.0, step=0.01)
    ps_commission = ps5.number_input(
        "Commission per trade ($)", value=0.0, step=0.01,
        help="$0 for most retail brokers"
    )

    if ps_entry > 0 and ps_stop > 0 and ps_entry != ps_stop:
        risk_per_share = abs(ps_entry - ps_stop)
        max_dollar_risk = ps_account * (ps_risk_pct / 100)
        shares = int(max_dollar_risk / risk_per_share)
        position_value = shares * ps_entry
        actual_risk = shares * risk_per_share + ps_commission

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Shares", f"{shares:,}")
        r2.metric("Position Value", f"${position_value:,.0f}")
        r3.metric("Dollar Risk", f"${actual_risk:,.2f}")
        r4.metric("% of Account", f"{position_value / ps_account * 100:.1f}%")

        if ps_commission > 0:
            breakeven_pct = ps_commission / position_value * 100 if position_value > 0 else 0
            st.caption(
                f"Commission breakeven: price must move {breakeven_pct:.3f}% "
                f"(${ps_commission / shares:.4f}/share) before profiting."
            )

        # R multiple targets
        st.markdown("#### R-Multiple Targets")
        for r_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
            direction = "BUY" if ps_entry > ps_stop else "SELL"
            if direction == "BUY":
                target = ps_entry + risk_per_share * r_mult
                profit = (target - ps_entry) * shares - ps_commission
            else:
                target = ps_entry - risk_per_share * r_mult
                profit = (ps_entry - target) * shares - ps_commission
            st.write(f"**{r_mult}R** → ${target:.2f}  |  Profit: **${profit:,.0f}**")
    else:
        st.warning("Enter valid entry and stop prices.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Compare All
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    ca1, ca2, ca3 = st.columns(3)
    ca_symbol = ca1.text_input("Symbol", value="SPY", key="ca_symbol").upper()
    ca_period = ca2.selectbox("Period", ["30d", "60d", "90d"], index=1, key="ca_period")
    ca_capital = ca3.number_input("Capital ($)", value=10_000, step=1_000, key="ca_capital")

    if st.button("▶ Compare All Strategies", key="run_compare"):
        with st.spinner("Running all 5 strategies…"):
            results = _cached_backtest_all(ca_symbol, ca_period, float(ca_capital))

        if results:
            df_compare = pd.DataFrame(results)
            df_compare.insert(0, "Rank", range(1, len(df_compare) + 1))

            def _status(row):
                if row["profit_factor"] >= 1.5 and row["win_rate"] >= 50:
                    return "✅ Strong"
                elif row["profit_factor"] >= 1.0 and row["win_rate"] >= 45:
                    return "🟡 Marginal"
                else:
                    return "🔴 Weak"

            df_compare["Status"] = df_compare.apply(_status, axis=1)
            rename = {
                "strategy": "Strategy",
                "trades": "Trades",
                "win_rate": "Win%",
                "profit_factor": "Profit Factor",
                "total_pnl": "Total P&L ($)",
                "avg_pnl_pct": "Avg P&L%",
                "sharpe_ratio": "Sharpe",
                "max_drawdown_pct": "Max DD%",
                "best_hour": "Best Hour",
                "best_day_of_week": "Best DoW",
            }
            df_compare = df_compare.rename(columns=rename)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
        else:
            st.info("No results returned.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Scanner
# ─────────────────────────────────────────────────────────────────────────────
with tab_scanner:
    sc_input = st.text_input(
        "Symbols (comma-separated)", value=DEFAULT_SYMBOLS, key="scanner_symbols"
    )
    sc_filter = st.radio(
        "Filter", ["Show all", "BUY only", "SELL only"], horizontal=True, key="sc_filter"
    )

    if st.button("🔍 Scan All", key="run_scan_btn"):
        with st.spinner(f"Scanning {len(sc_input.split(','))} symbols across 5 strategies…"):
            scan_results = _cached_scan(sc_input)

        if sc_filter == "BUY only":
            scan_results = [s for s in scan_results if s["direction"] == "BUY"]
        elif sc_filter == "SELL only":
            scan_results = [s for s in scan_results if s["direction"] == "SELL"]

        if not scan_results:
            st.info("No signals found for current bars.")
        else:
            df_scan = pd.DataFrame(scan_results)
            show_cols = [
                "symbol", "strategy", "direction", "confidence",
                "entry_price", "stop_price", "target_price",
                "r_multiple", "regime", "signal_time",
            ]
            show_cols = [c for c in show_cols if c in df_scan.columns]
            df_scan = df_scan[show_cols].sort_values("confidence", ascending=False)

            def _row_color(row):
                if row["direction"] == "BUY":
                    return ["background-color: rgba(0,212,170,0.15)"] * len(row)
                elif row["direction"] == "SELL":
                    return ["background-color: rgba(255,75,75,0.15)"] * len(row)
                return ["background-color: rgba(128,128,128,0.1)"] * len(row)

            st.dataframe(
                df_scan.style.apply(_row_color, axis=1),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"{len(df_scan)} signal(s) found across {sc_input}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Config
# ─────────────────────────────────────────────────────────────────────────────
with tab_config:
    st.markdown("### Day Trading Notes")
    st.warning(
        "**PDT Rule**: Pattern Day Trader — if you make 4+ day trades in 5 "
        "rolling business days, your broker may flag you as a PDT and require "
        "a $25,000 minimum account balance. Max 3 day trades per 5 days for "
        "accounts under $25k."
    )
    st.error(
        "**NEVER hold overnight.** All positions generated by this module must "
        "be closed by 3:45 PM ET. The system marks all signals as DAY orders."
    )
    st.info("**Recommended minimum account for day trading: $25,000+**")

    st.markdown("---")
    st.markdown("### Strategy Configuration")

    for strategy in ALL_STRATEGIES:
        with st.expander(f"⚙️ {strategy.name}  —  {STRATEGY_DESCRIPTIONS.get(strategy.name, '')}"):
            enabled_key = f"cfg_enabled_{strategy.name}"
            st.toggle(f"Enable {strategy.name}", value=True, key=enabled_key)

            cfg = strategy.default_config
            st.markdown("**Parameters (defaults — editing not persisted in this view):**")
            cfg_df = pd.DataFrame(
                [{"Parameter": k, "Default Value": v} for k, v in cfg.items()]
            )
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)
