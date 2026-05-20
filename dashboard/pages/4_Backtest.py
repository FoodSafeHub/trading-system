from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme
import _charts as charts

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

apply_theme("Backtest")
st.title("Strategy Backtester")
st.caption("Simulates how a strategy would have performed on historical data — no real money involved.")

# ── New strategy metadata ──────────────────────────────────────────────────────

NEW_STRATEGY_TYPES = {
    "rsi2_mean_reversion",
    "ema_macd_crossover",
    "bb_squeeze_breakout",
    "pullback_ema50",
    "vix_spike_reversal",
}

NEW_STRATEGY_DESCRIPTIONS = {
    "rsi2_mean_reversion": (
        "RSI-2 Mean Reversion (Connors-style). Buys deep short-term oversold dips "
        "(RSI(2) < 10) inside a long-term uptrend. Historically 85%+ win rate on SPY. "
        "BULL only. Hold 1–10 bars."
    ),
    "ema_macd_crossover": (
        "EMA Crossover + MACD Confirmation. Three-filter system: EMA(9) crosses EMA(21) "
        "+ MACD above signal + RSI 45–65 + volume spike. BULL only. Hold 5–20 bars."
    ),
    "bb_squeeze_breakout": (
        "Bollinger Band Squeeze + Breakout. Detects consolidation (5+ bars of contracting "
        "BB width) then buys the breakout above the upper band. BULL only. Hold 3–15 bars."
    ),
    "pullback_ema50": (
        "Pullback to Rising EMA(50). Buys dips to a rising 50-EMA with RSI 35–55 and a "
        "bullish wick. Works in mild bear too (not extreme bear). Highest signal frequency."
    ),
    "vix_spike_reversal": (
        "VIX Spike Reversal. Catches reversals after panic sell-offs using ATR% as VIX proxy. "
        "Works in BOTH bull and bear regimes. Tight 8-bar hold, 4% stop."
    ),
}

SYMBOL_RECOMMENDATIONS = {
    "AAPL": {
        "primary":   "AAPL_Pullback_EMA50",
        "backup":    "AAPL_RSI2_Mean_Reversion",
        "key_filters": [
            "RSI 35–55 at EMA50 touch",
            "Bullish wick ratio > 0.4",
            "EMA50 must be rising",
            "SPY above SMA200 (BULL regime)",
        ],
    },
    "MSFT": {
        "primary":   "MSFT_EMA_MACD_Crossover",
        "backup":    "MSFT_Pullback_EMA50",
        "key_filters": [
            "EMA(9) crosses EMA(21)",
            "MACD above signal line",
            "RSI 45–65 at crossover",
            "Volume > 1.1x 20-day avg",
        ],
    },
    "SPY": {
        "primary":   "SPY_RSI2_Mean_Reversion",
        "backup":    "SPY_Pullback_EMA50",
        "key_filters": [
            "RSI(2) < 10 (deeply oversold)",
            "SPY itself above SMA200",
            "ATR% not in extreme spike",
            "Exit when RSI(2) > 70 or above SMA(5)",
        ],
    },
    "GOOGL": {
        "primary":   "GOOGL_BB_Squeeze_Breakout",
        "backup":    "GOOGL_EMA_MACD_Crossover",
        "key_filters": [
            "BB width contracting 5+ bars",
            "Close above upper BB on breakout",
            "Volume > 1.3x 20-day avg",
            "RSI > 50 confirms momentum",
        ],
    },
}

# Legacy recommendations still supported
_LEGACY_RECOMMENDATIONS = {
    "AAPL_BB_Mean_Reversion":      "Legacy_AAPL_BB_Mean_Reversion",
    "AAPL_Fib_Pullback":           "Legacy_AAPL_Fib_Pullback",
    "MSFT_EMA_Trend_Continuation": "Legacy_MSFT_EMA_Trend_Continuation",
    "MSFT_Fib_Pullback":           "Legacy_MSFT_Fib_Pullback",
    "SPY_EMA_Swing":               "Legacy_SPY_EMA_Swing",
    "SPY_Fib_Pullback":            "Legacy_SPY_Fib_Pullback",
    "GOOGL_Breakout":              "Legacy_GOOGL_Breakout",
    "GOOGL_BB_Mean_Reversion":     "Legacy_GOOGL_BB_Mean_Reversion",
}


def get_symbol_recommendation(symbol: str) -> dict:
    return SYMBOL_RECOMMENDATIONS.get(symbol, {})


def _is_new_strategy(strategy_name: str) -> bool:
    return any(t in strategy_name.lower().replace("-", "_") for t in NEW_STRATEGY_TYPES)


def _resolve_recommended_strategy(symbol: str, available: list[str], preferred: str) -> str | None:
    if preferred in available:
        return preferred
    # Try legacy alias
    legacy = _LEGACY_RECOMMENDATIONS.get(preferred)
    if legacy and legacy in available:
        return legacy
    return None


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _render_recommendation_overview() -> None:
    rows = []
    for symbol, rec in SYMBOL_RECOMMENDATIONS.items():
        rows.append({
            "Symbol":   symbol,
            "Primary":  rec["primary"],
            "Backup":   rec["backup"],
            "Key filters": "; ".join(rec["key_filters"]),
        })
    with st.expander("Symbol Strategy Recommendations (v2 — Regime-Aware)", expanded=True):
        st.markdown(
            "All 5 new strategies use **SPY SMA(200) as a market regime filter**.  "
            "Strategies 1–3 are BULL-only; Strategies 4–5 also fire in mild bear conditions."
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_symbol_recommendation_card(symbol: str) -> None:
    rec = get_symbol_recommendation(symbol)
    if not rec:
        return
    st.info(
        f"**Recommended for {symbol}** (regime-aware v2)\n\n"
        f"- Primary strategy: **{rec['primary']}**\n"
        f"- Backup strategy: **{rec['backup']}**\n"
        f"- Suggested filters: {', '.join(rec['key_filters'])}\n"
    )


def _render_filter_summary(symbol: str) -> None:
    rec = get_symbol_recommendation(symbol)
    if not rec:
        return
    st.subheader("Recommended Filters for This Symbol")
    for item in rec["key_filters"]:
        st.markdown(f"- {item}")


def _render_strategy_description(strategy_name: str) -> None:
    for key, desc in NEW_STRATEGY_DESCRIPTIONS.items():
        if key in strategy_name.lower().replace("-", "_"):
            st.caption(f"**Strategy notes:** {desc}")
            return


def _render_zero_trade_debug_single(symbol: str, strategy: str, period: str) -> None:
    rec = get_symbol_recommendation(symbol)
    st.error("No trades were generated for this backtest.")
    st.markdown(
        f"**Backtest details:**\n"
        f"- Symbol: **{symbol or 'unknown'}**\n"
        f"- Strategy: **{strategy or 'unknown'}**\n"
        f"- Period: **{period}**\n"
    )

    # Per-strategy guidance
    strategy_guidance = {
        "rsi2_mean_reversion": (
            "RSI-2 needs very deep oversold dips (RSI(2) < 10). "
            "Try a longer period (2y) to capture more market corrections. "
            "Most active during pullbacks in bull markets."
        ),
        "ema_macd_crossover": (
            "EMA crossover fires only when EMA(9) crosses EMA(21) with MACD and volume confirmation. "
            "Try 2y for more crossover events, or loosen vol_ratio_min in the strategy config."
        ),
        "bb_squeeze_breakout": (
            "BB squeeze requires 5+ consecutive bars of contracting bandwidth then a breakout. "
            "These setups are relatively rare — try 2y to see more. "
            "Reduce squeeze_bars to 3 for more signals."
        ),
        "pullback_ema50": (
            "Pullback EMA50 needs price within ±1% of a rising EMA50 with RSI 35–55. "
            "This is the highest-frequency strategy — if no signals appear in 1y, "
            "loosen price_ema_proximity_pct to 2.0."
        ),
        "vix_spike_reversal": (
            "VIX Spike Reversal needs ATR% > 3.0 (panic conditions). "
            "These happen rarely — 2020 COVID crash, 2022 rate shock, etc. "
            "Try a 2y+ period that includes a market correction."
        ),
    }
    for key, guidance in strategy_guidance.items():
        if key in (strategy or "").lower().replace("-", "_"):
            st.warning(f"**Strategy-specific guidance:** {guidance}")
            break

    if rec:
        st.markdown(
            f"**Recommended primary strategy:** **{rec['primary']}**\n"
            f"**Recommended backup strategy:** **{rec['backup']}**\n"
        )
    st.markdown(
        "**General next actions:**\n"
        "- Try a longer period (**2y**)\n"
        "- Switch to the recommended primary strategy\n"
        "- Lower min_agreement in Consensus Mode\n"
    )


def _render_zero_trade_debug_consensus(symbol: str, min_agreement: int) -> None:
    guidance = {
        "AAPL":  "Try AAPL_Pullback_EMA50 (highest frequency) or lower min_agreement to 1.",
        "MSFT":  "Try MSFT_EMA_MACD_Crossover with min_agreement=1 first, then raise.",
        "SPY":   "SPY_RSI2_Mean_Reversion fires most often. Try 2y period.",
        "GOOGL": "GOOGL_BB_Squeeze_Breakout needs 2y+ for enough squeeze setups.",
    }
    st.error("No trades were placed with the current consensus settings.")
    if symbol in guidance:
        st.info(f"Symbol-specific guidance for {symbol}: {guidance[symbol]}")
    st.markdown(
        "**Suggested next actions:**\n"
        "- Lower the minimum agreement threshold to **1**\n"
        "- Increase period to **2y**\n"
        "- Ensure SPY data is available (regime filter requires it)\n"
    )


# ── Market Regime Panel ────────────────────────────────────────────────────────

def _render_regime_panel(trades: list, period: str) -> None:
    """Show BULL vs BEAR trade breakdown when trade dicts contain a 'regime' field."""
    regime_trades = [t for t in trades if t.get("regime")]
    if not regime_trades:
        return

    bull_trades = [t for t in regime_trades if t.get("regime", "").upper() == "BULL"]
    bear_trades = [t for t in regime_trades if t.get("regime", "").upper() in ("BEAR", "DEEP_BEAR")]

    st.divider()
    st.subheader("Market Regime Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("BULL Regime Trades", len(bull_trades))
    c2.metric("BEAR Regime Trades", len(bear_trades))
    c3.metric("Total Trades", len(regime_trades))


# ── Equity + price-action charts ───────────────────────────────────────────────

def _resolve_symbol(r: dict) -> str | None:
    """Best-effort: pull the symbol from the result dict or trades."""
    sym = r.get("symbol") or r.get("ticker")
    if sym:
        return str(sym).upper()
    trades = r.get("trades") or []
    for t in trades:
        if t.get("symbol"):
            return str(t["symbol"]).upper()
    return None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_chart(symbol: str, period: str) -> dict | None:
    try:
        return api.chart_data(symbol, period=period)
    except Exception:
        return None


def _equity_chart(r: dict, *, symbol: str | None = None, period: str = "1y") -> None:
    if not r.get("equity_curve"):
        return
    st.divider()
    st.subheader("Equity Curve — OHLC view")
    st.caption(
        "Equity track resampled into weekly OHLC candles with the daily mark-to-market line "
        "behind it. Green/red triangles mark BUY/SELL fills; the lower ribbon shows running drawdown."
    )

    bucket_label = st.radio(
        "Candle aggregation",
        ["Daily", "Weekly", "Monthly"],
        index=1, horizontal=True, key=f"eq_bucket_{r.get('strategy_name', 'x')}",
    )
    bucket = {"Daily": "D", "Weekly": "W", "Monthly": "M"}[bucket_label]

    charts.render_equity_chart(
        r["equity_curve"],
        trades=r.get("trades", []),
        initial_capital=r.get("initial_capital"),
        title=f"Equity — {r.get('strategy_name', '')}",
        bucket=bucket,
        height=480,
    )


def _price_action_chart(r: dict, symbol: str | None, period: str) -> None:
    """Render the underlying price as candlesticks with trade markers + indicators."""
    if not symbol:
        return
    payload = _fetch_chart(symbol, period)
    if not payload or not payload.get("dates"):
        st.caption(f"No OHLC data available for {symbol} — skipping price-action chart.")
        return

    st.divider()
    st.subheader(f"Price Action — {symbol} with Trade Markers")
    st.caption(
        "Candlesticks show daily OHLC for the backtest window. BUY triangles sit at the fill price; "
        "the lower panes show RSI(14) and MACD so you can read each entry in context."
    )

    overlay_choices = st.multiselect(
        "Indicator layers",
        options=["ema9", "ema21", "ema50", "ema200", "vwap", "bb_upper", "bb_lower", "supertrend"],
        default=["ema21", "ema50", "vwap"],
        format_func=lambda k: {
            "ema9": "EMA 9", "ema21": "EMA 21", "ema50": "EMA 50", "ema200": "EMA 200",
            "vwap": "VWAP", "bb_upper": "Bollinger ↑", "bb_lower": "Bollinger ↓",
            "supertrend": "Supertrend",
        }[k],
        key=f"price_overlays_{symbol}",
    )

    charts.render_price_chart(
        payload,
        trades=r.get("trades", []),
        overlays=tuple(overlay_choices),
        include_volume=True,
        include_rsi=True,
        include_macd=True,
        title=f"{symbol} — {period}",
    )


def _side_tag(v: str) -> str:
    if str(v) == "BUY":   return "🟢 BUY"
    if "SELL" in str(v):  return "🔴 SELL"
    return str(v)


def _pnl_tag(v) -> str:
    if v is None: return "—"
    return f"🟢 +${v:,.2f}" if v >= 0 else f"🔴 -${abs(v):,.2f}"


def _render_backtest_metrics(r: dict) -> None:
    pnl = r["total_pnl"]
    ret = r["total_return_pct"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Starting Capital", f"${r['initial_capital']:,.0f}")
    c2.metric("Final Capital",    f"${r['final_capital']:,.0f}")
    c3.metric("Total P&L",        f"${pnl:,.2f}", delta=f"{ret:+.2f}%",
              delta_color="normal" if pnl >= 0 else "inverse")
    c4.metric("Win Rate",         f"{r['win_rate_pct']:.1f}%",
              delta=f"{r['winning_trades']}W / {r['losing_trades']}L")
    c5.metric("Max Drawdown",     f"{r['max_drawdown_pct']:.1f}%", delta_color="inverse")
    c6.metric("Sharpe Ratio",     r["sharpe_ratio"] if r["sharpe_ratio"] else "—")


def _single_trades_table(trades: list) -> None:
    st.divider()
    st.subheader(f"All Trades ({len(trades)} total)")
    if not trades:
        st.info("No trades were generated in this period.")
        st.caption("Try a longer period (2y) or a different strategy.")
        return

    rows = []
    pending_buy = None
    for t in trades:
        if t["side"] == "BUY":
            pending_buy = t
            rows.append({**t, "pnl": None})
        elif "SELL" in t["side"]:
            pnl = round(t["value"] - pending_buy["value"], 2) if pending_buy else None
            rows.append({**t, "pnl": pnl})
            pending_buy = None

    df = pd.DataFrame(rows)
    df["side"]  = df["side"].apply(_side_tag)
    df["value"] = df["value"].apply(lambda v: f"${v:,.2f}")
    df["price"] = df["price"].apply(lambda v: f"${v:,.2f}")
    df["pnl"]   = df["pnl"].apply(_pnl_tag)
    df = df.rename(columns={"pnl": "profit / loss"})
    # Drop internal fields not useful to display
    for col in ["signal_from", "regime"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _consensus_trades_table(trades: list) -> None:
    st.divider()
    st.subheader(f"All Trades ({len(trades)} total)")
    if not trades:
        st.info("No trades were generated.")
        return

    rows = [{
        "date":         t.get("date"),
        "side":         _side_tag(t.get("side", "")),
        "price":        f"${t['price']:,.2f}" if t.get("price") is not None else "—",
        "quantity":     t.get("quantity"),
        "value":        f"${t['value']:,.2f}" if t.get("value") is not None else "—",
        "agreed by":    ", ".join(t.get("agreeing", [])) or "—",
        "profit / loss": _pnl_tag(t.get("pnl")),
    } for t in trades]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# MODE TOGGLE
# ══════════════════════════════════════════════════════════════
_render_recommendation_overview()
mode = st.radio(
    "Backtest Mode",
    ["Single Strategy", "Consensus Mode"],
    horizontal=True,
    help="Single: test one strategy alone.\n"
         "Consensus: test a symbol using multiple strategies with agreement filtering.",
)

st.divider()

# ══════════════════════════════════════════════════════════════
# SINGLE STRATEGY MODE
# ══════════════════════════════════════════════════════════════
if mode == "Single Strategy":
    try:
        strategies = api._get("/backtest/strategies")
        strategy_names = [s["name"] for s in strategies]
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    # Sort: new strategies first, then legacy
    new_names    = [n for n in strategy_names if not n.startswith("Legacy_")]
    legacy_names = [n for n in strategy_names if n.startswith("Legacy_")]
    strategy_names = new_names + legacy_names

    symbol_list = []
    symbol_info = {}
    try:
        symbols_data = api._get("/backtest/consensus-symbols")
        symbol_list = [s["symbol"] for s in symbols_data]
        symbol_info = {s["symbol"]: s for s in symbols_data}
    except Exception:
        symbol_list = []
        symbol_info = {}

    strategy_by_symbol: dict[str, list[str]] = {}
    for s in strategies:
        strategy_by_symbol.setdefault(s["symbol"], []).append(s["name"])

    col1, col2, col3, col4, col5 = st.columns([2, 2.5, 1.5, 2, 1])
    with col1:
        if symbol_list:
            chosen_sym = st.selectbox("Symbol", symbol_list)
        else:
            chosen_sym = st.text_input("Symbol", value="AAPL")

    sym_strats = strategy_by_symbol.get(chosen_sym, strategy_names)
    if not sym_strats:
        sym_strats = strategy_names
    # New strategies first within symbol
    sym_new    = [n for n in sym_strats if not n.startswith("Legacy_")]
    sym_legacy = [n for n in sym_strats if n.startswith("Legacy_")]
    sym_strats = sym_new + sym_legacy

    with col2:
        use_recommended = st.checkbox(
            "Use recommended strategy for selected symbol",
            value=True,
        )
        rec = get_symbol_recommendation(chosen_sym)
        recommended_choice = None
        if rec:
            recommended_choice = _resolve_recommended_strategy(
                chosen_sym, sym_strats, rec.get("primary", "")
            )

        use_rec_picker = use_recommended and bool(recommended_choice)
        default_idx = sym_strats.index(recommended_choice) if recommended_choice in sym_strats else 0
        chosen = st.selectbox(
            "Strategy", sym_strats,
            index=default_idx,
            disabled=use_rec_picker,
        )
        if use_rec_picker and recommended_choice:
            chosen = recommended_choice
        elif use_recommended and rec and not recommended_choice:
            st.warning("Recommended strategy not available — choose manually.")

    with col3:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y"], index=1)
    with col4:
        capital = st.number_input("Starting Capital ($)", value=100000, min_value=1000, step=10000)
    with col5:
        st.write("")
        st.write("")
        run = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    _render_symbol_recommendation_card(chosen_sym)
    _render_strategy_description(chosen or "")

    if not run and "bt_result" not in st.session_state:
        st.info("Select a symbol and strategy, then click Run Backtest.")
        st.stop()

    if run:
        with st.spinner(f"Running backtest for {chosen_sym} / {chosen} over {period}..."):
            try:
                params = {"period": period, "initial_capital": capital, "quantity": 0}
                if chosen_sym:
                    params["symbol"] = chosen_sym
                result = api._get(f"/backtest/run/{chosen}", timeout=120, params=params)
                st.session_state["bt_result"] = result
                st.session_state.pop("bt_consensus", None)
            except Exception as e:
                # Fallback: retry without symbol param for older API compatibility
                if chosen_sym:
                    try:
                        result = api._get(
                            f"/backtest/run/{chosen}?period={period}&initial_capital={capital}&quantity=0",
                            timeout=120,
                        )
                        st.session_state["bt_result"] = result
                        st.session_state.pop("bt_consensus", None)
                    except Exception:
                        st.error(f"Backtest failed: {e}")
                        st.stop()
                else:
                    st.error(f"Backtest failed: {e}")
                    st.stop()

    r = st.session_state.get("bt_result")
    if not r:
        st.stop()

    st.subheader(f"Results — {r['strategy_name']} ({r['start_date']} → {r['end_date']})")

    _render_backtest_metrics(r)

    if not r.get("trades"):
        _render_zero_trade_debug_single(chosen_sym, chosen, period)

    _render_regime_panel(r.get("trades", []), period)
    _render_filter_summary(chosen_sym)
    _equity_chart(r, symbol=chosen_sym, period=period)
    _price_action_chart(r, chosen_sym, period)
    _single_trades_table(r["trades"])


# ══════════════════════════════════════════════════════════════
# CONSENSUS MODE
# ══════════════════════════════════════════════════════════════
else:
    try:
        symbols_data = api._get("/backtest/consensus-symbols")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    symbol_list = [s["symbol"] for s in symbols_data]
    symbol_info = {s["symbol"]: s for s in symbols_data}

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        chosen_sym = st.selectbox("Symbol", symbol_list)
    with col2:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y"], index=1)
    with col3:
        capital = st.number_input("Starting Capital ($)", value=100000, min_value=1000, step=10000)

    _render_symbol_recommendation_card(chosen_sym)

    info = symbol_info.get(chosen_sym, {})
    strat_count = info.get("strategy_count", 0)
    strat_names = info.get("strategies", [])

    new_strats   = [n for n in strat_names if not n.startswith("Legacy_")]
    legacy_strats = [n for n in strat_names if n.startswith("Legacy_")]

    st.caption(
        f"**{chosen_sym}** — "
        f"**{len(new_strats)} new strategies**: {', '.join(new_strats[:5])}  |  "
        f"**{len(legacy_strats)} legacy**: {', '.join(legacy_strats)}"
    )

    # Regime info
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        use_new_only = st.checkbox(
            "Use 5 new strategies only (recommended)",
            value=True,
            help="New strategies are regime-aware and optimised for 2015–2025. "
                 "Uncheck to include legacy strategies.",
        )
    with col_r2:
        effective_count = len(new_strats) if use_new_only else strat_count
        min_agreement = st.slider(
            "Minimum strategies that must agree",
            min_value=1,
            max_value=max(effective_count, 1),
            value=min(2, effective_count),
            help="1 = any signal fires (most trades). 2 = two must agree (balanced). 3+ = high confidence.",
        )

    agree_labels = {1: "any signal fires (high noise)", 2: "two must agree (balanced)", 3: "all must agree (high confidence)"}
    st.info(
        f"**min_agreement = {min_agreement}** — {agree_labels.get(min_agreement, '')}. "
        f"Using {'new strategies only' if use_new_only else 'all strategies'}."
    )

    run_c = st.button("▶ Run Consensus Backtest", type="primary")

    if not run_c and "bt_consensus" not in st.session_state:
        st.stop()

    if run_c:
        with st.spinner(f"Running consensus backtest for {chosen_sym}..."):
            try:
                result = api._get(
                    f"/backtest/consensus/{chosen_sym}"
                    f"?min_agreement={min_agreement}&period={period}"
                    f"&initial_capital={capital}&new_only={'true' if use_new_only else 'false'}",
                    timeout=120,
                )
                st.session_state["bt_consensus"] = result
                st.session_state.pop("bt_result", None)
            except Exception as e:
                st.error(f"Consensus backtest failed: {e}")
                st.stop()

    r = st.session_state.get("bt_consensus")
    if not r:
        st.stop()

    st.divider()
    st.subheader(
        f"Consensus Results — {r['symbol']}  |  "
        f"Min Agreement: {r['min_agreement']}  |  "
        f"{r['start_date']} → {r['end_date']}"
    )
    st.caption(f"Strategies used: {', '.join(r['strategies_used'])}")

    _render_backtest_metrics(r)

    if r["total_trades"] == 0:
        _render_zero_trade_debug_consensus(chosen_sym, r.get("min_agreement", 0))

    _render_regime_panel(r.get("trades", []), period)
    _render_filter_summary(chosen_sym)
    _equity_chart(r, symbol=chosen_sym, period=period)
    _price_action_chart(r, chosen_sym, period)
    _consensus_trades_table(r["trades"])
