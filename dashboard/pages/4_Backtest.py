from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme
import _charts as charts

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# Shared trader-style ranking — the SAME service the scanner badge and the
# /recommendations API use, so the banner here can never disagree with them.
from app.services.recommendations.winner import score_strategies

apply_theme("Backtest")
st.title("Strategy Backtester")
st.caption("Simulates how a strategy would have performed on historical data — no real money involved.")

# ── Broker route options (per-assignment override at Promote time) ─────────────
# "default" defers to the global active_broker / trade_routing toggle; the
# others pin orders for the promoted symbol to a specific broker adapter.
BROKER_OPTIONS: list[tuple[str, str]] = [
    ("default", "Default (use global toggle)"),
    ("schwab",  "Schwab"),
    ("webull",  "Webull"),
    ("paper",   "Paper"),
]
BROKER_LABEL = {k: v for k, v in BROKER_OPTIONS}
BROKER_VALUES = [k for k, _ in BROKER_OPTIONS]


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


def _price_action_chart(r: dict, symbol: str | None, period: str,
                        key_suffix: str = "") -> None:
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
        key=f"price_overlays_{symbol}_{key_suffix}" if key_suffix else f"price_overlays_{symbol}",
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
# LIVE SIGNALS — what would fire RIGHT NOW (read-only snapshot)
# ══════════════════════════════════════════════════════════════
with st.expander("📡 Live Signals — what would fire right now", expanded=False):
    st.caption(
        "Runs all 7 strategies on the latest market data — no trade is placed, "
        "just analysis. Use this before running a backtest to see what's actively "
        "signalling on a symbol."
    )

    ls_c1, ls_c2 = st.columns([3, 1])
    with ls_c1:
        ls_symbol = st.text_input(
            "Symbol", value="AAPL", key="bt_live_sym",
            help="Type any US stock ticker — BRK-B, NVDA, TQQQ, etc.",
        ).upper().strip()
    with ls_c2:
        st.write("")
        st.write("")
        ls_run = st.button("▶ Get Signals", type="primary",
                           use_container_width=True, key="bt_live_run")

    if ls_run:
        if not ls_symbol:
            st.error("Enter a symbol.")
        else:
            with st.spinner(f"Fetching signals for {ls_symbol}..."):
                try:
                    st.session_state["bt_live_result"] = api.backtest_live_signals(ls_symbol)
                except Exception as exc:
                    st.error(f"Failed: {exc}")

    live = st.session_state.get("bt_live_result")
    if live and live.get("signals"):
        sigs = live["signals"]
        buy_n  = sum(1 for s in sigs if s.get("direction") == "BUY")
        sell_n = sum(1 for s in sigs if s.get("direction") == "SELL")
        hold_n = sum(1 for s in sigs if s.get("direction") == "HOLD")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY signals",  buy_n)
        c2.metric("🔴 SELL signals", sell_n)
        c3.metric("⬜ HOLD",         hold_n)
        c4.metric("Last close", f"${live.get('last_close'):,.2f}" if live.get("last_close") else "—")
        st.caption(f"As of: **{live.get('as_of', '—')}** for **{live.get('symbol', '—')}**")
        st.divider()

        for s in sigs:
            direction = s.get("direction", "HOLD")
            name = s.get("strategy_name", "?")
            if s.get("error"):
                with st.expander(f"❌ {name} — error", expanded=False):
                    st.error(s["error"])
                continue
            icon = "🟢" if direction == "BUY" else ("🔴" if direction == "SELL" else "⬜")
            strength = s.get("strength") or 0
            strength_str = f"  ·  strength {strength:.0%}" if strength else ""
            with st.expander(
                f"{icon} **{name}** — {direction}{strength_str}",
                expanded=(direction in ("BUY", "SELL")),
            ):
                if s.get("reason"):
                    st.info(f"**Reason:** {s['reason']}")

                if direction in ("BUY", "SELL"):
                    mc1, mc2, mc3 = st.columns(3)
                    if s.get("entry_price"):
                        mc1.metric("Entry", f"${s['entry_price']:,.2f}")
                    if s.get("stop_price") and s.get("entry_price"):
                        stop_pct = (s["entry_price"] - s["stop_price"]) / s["entry_price"] * 100
                        mc2.metric("Stop", f"${s['stop_price']:,.2f}",
                                   delta=f"-{stop_pct:.1f}% from entry",
                                   delta_color="inverse")
                    if s.get("target_price") and s.get("entry_price"):
                        tgt_pct = (s["target_price"] - s["entry_price"]) / s["entry_price"] * 100
                        mc3.metric("Target", f"${s['target_price']:,.2f}",
                                   delta=f"+{tgt_pct:.1f}% from entry")

                if s.get("indicators"):
                    with st.expander("Indicator values", expanded=False):
                        st.json(s["indicators"])
    elif live:
        st.warning("No signals returned for this symbol.")
    else:
        st.caption("Enter a symbol and click Get Signals.")


# ══════════════════════════════════════════════════════════════
# MODE TOGGLE
# ══════════════════════════════════════════════════════════════
_render_recommendation_overview()
mode = st.radio(
    "Backtest Mode",
    ["Single Strategy", "Consensus Mode", "Custom Symbol"],
    horizontal=True,
    help="Single: test one strategy alone.\n"
         "Consensus: test a symbol using its configured strategies with agreement filtering.\n"
         "Custom Symbol: test ANY ticker (e.g. BRK-B) using the 5 scanner strategies — "
         "useful for reproducing scanner candidates.",
)

st.divider()

# ══════════════════════════════════════════════════════════════
# SINGLE STRATEGY MODE
# ══════════════════════════════════════════════════════════════
if mode == "Single Strategy":
    # The 7 strategy types available via _make_generic_configs_full.
    # Ordered: 5 regime-aware first, then the 2 legacy types.
    SINGLE_STRATEGY_CHOICES: list[tuple[str, str]] = [
        ("rsi2_mean_reversion",  "RSI-2 Mean Reversion"),
        ("ema_macd_crossover",   "EMA + MACD Crossover"),
        ("bb_squeeze_breakout",  "Bollinger Squeeze Breakout"),
        ("pullback_ema50",       "Pullback to EMA(50)"),
        ("vix_spike_reversal",   "VIX Spike Reversal"),
        ("bollinger",            "Legacy: Bollinger Mean Reversion"),
        ("fib_pullback",         "Legacy: Fibonacci Pullback"),
    ]
    _label_by_type = {t: lbl for t, lbl in SINGLE_STRATEGY_CHOICES}
    _types = [t for t, _ in SINGLE_STRATEGY_CHOICES]

    col1, col2, col3, col4, col5 = st.columns([2, 2.5, 1.5, 2, 1])
    with col1:
        chosen_sym = st.text_input(
            "Symbol", value="AAPL",
            help="Any Yahoo-Finance-compatible ticker (AAPL, BRK-B, NVDA, TQQQ, ...).",
        ).strip().upper()
    with col2:
        chosen_type = st.selectbox(
            "Strategy", _types,
            format_func=lambda t: _label_by_type[t],
            index=3,  # Pullback EMA50 — the highest-frequency strategy
        )
    with col3:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=1)
    with col4:
        capital = st.number_input("Starting Capital ($)", value=100000, min_value=1000, step=10000)
    with col5:
        st.write("")
        st.write("")
        run = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    # Strategy notes — re-use the description card if it matches.
    # _render_strategy_description expects a strategy NAME, but matches on
    # type substrings, so the bare type works.
    _render_strategy_description(chosen_type)

    if not run and "bt_result" not in st.session_state:
        st.info("Type a symbol, pick a strategy, then click Run Backtest.")
        st.stop()

    if run:
        if not chosen_sym:
            st.error("Enter a symbol.")
            st.stop()
        with st.spinner(f"Running backtest for {chosen_sym} / {_label_by_type[chosen_type]} over {period}..."):
            try:
                result = api.backtest_run_generic(
                    chosen_sym, chosen_type,
                    period=period, initial_capital=capital, timeout=120,
                )
                st.session_state["bt_result"] = result
                st.session_state.pop("bt_consensus", None)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

    # Keep `chosen` populated so the Promote block below still works.
    chosen = st.session_state.get("bt_result", {}).get("strategy_name", "")

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

    # ── Promote this exact strategy/symbol pair to auto-trade ──────────
    # Lives after the trades table so the user has full context (equity
    # curve, drawdown, trade list) before committing the strategy to the
    # live scheduler.
    st.divider()
    st.markdown("##### Promote this strategy to auto-trade")
    st.caption(
        f"Assigns **{r['strategy_name']}** to **{chosen_sym}** in the live scheduler. "
        "BUY/SELL signals on this symbol will create notifications once promoted."
    )
    p1, p2 = st.columns([1, 1])
    with p1:
        single_cap_str = st.text_input(
            "Max capital ($)", value="", placeholder="optional",
            key=f"single_promote_cap_{chosen_sym}_{chosen}",
            help="Dollar cap. Wins over shares cap when both are set. "
                 "Leave blank to use the global account cap.",
        )
    with p2:
        single_shares_str = st.text_input(
            "Max shares (qty)", value="", placeholder="optional",
            key=f"single_promote_shares_{chosen_sym}_{chosen}",
            help="Shares cap. Used only when the dollar cap is empty.",
        )
    p3, p4, p5 = st.columns([3, 1, 2])
    with p3:
        single_broker = st.selectbox(
            "Broker route",
            BROKER_VALUES,
            index=0,
            format_func=lambda v: BROKER_LABEL.get(v, v.title()),
            key=f"single_promote_broker_{chosen_sym}_{chosen}",
            help="Default = global toggle. Otherwise pins this symbol's orders "
                 "to the selected broker.",
        )
    with p4:
        single_enabled = st.checkbox(
            "Enabled", value=True,
            key=f"single_promote_en_{chosen_sym}_{chosen}",
        )
    with p5:
        st.write("")
        st.write("")
        single_go = st.button(
            "Promote", type="primary", use_container_width=True,
            key=f"single_promote_btn_{chosen_sym}_{chosen}",
        )
    if single_go:
        try:
            single_cap_val: float | None = None
            if single_cap_str.strip():
                single_cap_val = float(single_cap_str.strip())
            single_shares_val: float | None = None
            if single_shares_str.strip():
                single_shares_val = float(single_shares_str.strip())
            api.upsert_assignment(
                symbol=chosen_sym,
                system="scanner",
                strategy_name=r["strategy_name"],
                enabled=single_enabled,
                notes=f"Promoted from Single Strategy backtest ({period})",
                max_capital_usd=single_cap_val,
                max_shares=single_shares_val,
                broker=single_broker,
            )
            st.success(
                f"Assigned **{r['strategy_name']}** to **{chosen_sym}** "
                f"(scanner, enabled={single_enabled}). "
                "BUY/SELL signals on this symbol will now create notifications."
            )
        except Exception as exc:
            st.error(f"Promote failed: {exc}")


# ══════════════════════════════════════════════════════════════
# CONSENSUS MODE
# ══════════════════════════════════════════════════════════════
elif mode == "Consensus Mode":
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


# ══════════════════════════════════════════════════════════════
# CUSTOM SYMBOL MODE — any ticker, 5 scanner strategies consensus
# ══════════════════════════════════════════════════════════════
else:  # mode == "Custom Symbol"
    st.info(
        "Type any ticker (e.g. **BRK-B**, **NVDA**, **TQQQ**) and the backtester will "
        "run each of the 7 strategies INDEPENDENTLY — RSI2 Mean Reversion, "
        "EMA+MACD Crossover, BB Squeeze Breakout, Pullback to EMA(50), VIX Spike "
        "Reversal, plus the 2 legacy strategies (Bollinger Mean Reversion, "
        "Fibonacci Pullback) — so you see the same coverage as predefined symbols."
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        custom_sym = st.text_input(
            "Symbol", value="BRK-B",
            help="Any Yahoo-Finance-compatible ticker. Use the same form the scanner used "
                 "(e.g. BRK-B with a hyphen, not BRK.B).",
            key="custom_bt_sym",
        ).strip().upper()
    with c2:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=1,
                              key="custom_bt_period")
    with c3:
        capital = st.number_input("Starting Capital ($)", value=10000, min_value=1000,
                                  step=1000, key="custom_bt_capital")

    run_cs = st.button("▶ Compare All 7 Strategies", type="primary", key="custom_bt_run")

    if run_cs:
        if not custom_sym:
            st.error("Enter a symbol.")
            st.stop()
        with st.spinner(f"Running all 7 strategies on {custom_sym} over {period}..."):
            try:
                results = api.backtest_custom_compare_all(
                    custom_sym, period=period, initial_capital=capital, timeout=240,
                )
                st.session_state["bt_custom_compare"] = {"symbol": custom_sym,
                                                          "period": period,
                                                          "rows": results}
            except Exception as e:
                st.error(f"Custom compare failed: {e}")
                st.stop()

    state = st.session_state.get("bt_custom_compare")
    if not state:
        st.caption("Enter a symbol and click Compare.")
        st.stop()

    cmp = state["rows"]
    cmp_symbol = state["symbol"]
    cmp_period = state["period"]

    st.divider()

    # Require at least 3 trades for the comparison; same statistical guardrail
    # the Perplexity Compare All tab uses (set lower here because some scanner
    # strategies only fire 3–4 times in 1y on a single name).
    valid = [r for r in cmp if not r.get("error") and (r.get("total_trades") or 0) >= 3]

    # ── Best strategy recommendation (trader-style weighted scoring) ───────────
    # Delegates to the shared winner service: a weighted score (expectancy,
    # profit factor, drawdown, Sharpe) multiplied by a steep sample-size
    # confidence factor — so a tiny "perfect" backtest can't win on a
    # technicality, but a dramatically superior rare setup still can.
    best = None
    ranked = score_strategies(valid) if valid else []
    if ranked:
        best = ranked[0]
        b_pf  = best.get("profit_factor")
        b_ret = best.get("total_return_pct") or 0
        b_wr  = best.get("win_rate_pct") or 0
        b_dd  = best.get("max_drawdown_pct") or 0
        pf_str = f"{b_pf:.2f}" if b_pf is not None else "—"

        st.success(
            f"**Recommended strategy for {cmp_symbol}: {best['strategy_name'].replace('_', ' ')}**  \n"
            f"Score: **{best.get('_score', 0):.3f}** "
            f"({best.get('_confidence_label', 'unrated')})  \n"
            f"Win Rate: **{b_wr:.1f}%** | Profit Factor: **{pf_str}** | "
            f"Return over {cmp_period}: **{b_ret:+.2f}%** | "
            f"Max Drawdown: **{b_dd:.1f}%** | Trades: **{best.get('total_trades', 0)}**  \n"
            f"_{best.get('_reason', '')}_"
        )
        warns = best.get("_warnings") or []
        if warns:
            st.warning("  \n".join(f"⚠️ {w}" for w in warns))

        # Runners-up, so the user sees the next-best picks and why the leader
        # beat them (the spec asks for transparency, not just a single name).
        if len(ranked) > 1:
            with st.expander("Why this one? — full ranking", expanded=False):
                for rank, r in enumerate(ranked, start=1):
                    marker = "⭐ " if rank == 1 else f"{rank}. "
                    rpf = r.get("profit_factor")
                    rpf_str = f"{rpf:.2f}" if rpf is not None else "—"
                    st.markdown(
                        f"**{marker}{r['strategy_name'].replace('_', ' ')}** — "
                        f"score **{r.get('_score', 0):.3f}** · {r.get('_confidence_label', '')}  \n"
                        f"return {(r.get('total_return_pct') or 0):+.1f}% · "
                        f"PF {rpf_str} · win {(r.get('win_rate_pct') or 0):.1f}% · "
                        f"{r.get('total_trades', 0)} trades  \n"
                        f"{r.get('_reason', '')}"
                    )
                    for w in (r.get("_warnings") or []):
                        st.caption(f"⚠️ {w}")
        st.caption("Use the Promote control below to assign this to auto-trade.")
        st.divider()
    else:
        st.warning(
            "None of the 5 strategies generated enough trades on this symbol/period "
            "to compare reliably (need at least 3). Try a longer period (2y or 5y)."
        )

    # ── Comparison table ───────────────────────────────────────────────
    rows = []
    for r in cmp:
        if r.get("error"):
            rows.append({
                "Strategy":      r["strategy_name"],
                "Trades":        "—",
                "Win Rate":      "—",
                "Profit Factor": "—",
                "Avg Win":       "—",
                "Expectancy":    "—",
                "Total Return":  "ERROR",
                "CAGR":          "—",
                "Total P&L":     r["error"][:60],
                "Max Drawdown":  "—",
                "Sharpe":        "—",
            })
            continue
        pnl = r.get("total_pnl") or 0
        is_best = best is not None and r["strategy_name"] == best["strategy_name"]
        pf = r.get("profit_factor")
        sh = r.get("sharpe_ratio")
        rows.append({
            "Strategy":      ("⭐ " if is_best else "") + r["strategy_name"],
            "Trades":        r.get("total_trades", 0),
            "Win Rate":      f"{(r.get('win_rate_pct') or 0):.1f}%",
            "Profit Factor": f"{pf:.2f}" if pf is not None else "—",
            "Avg Win":       f"{(r.get('avg_win_pct') or 0):+.2f}%",
            "Expectancy":    f"{(r.get('expectancy_pct') or 0):+.2f}%",
            "Total Return":  f"{(r.get('total_return_pct') or 0):+.2f}%",
            "CAGR":          f"{(r.get('cagr') or 0):+.2f}%",
            "Total P&L":     f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}",
            "Max Drawdown":  f"{(r.get('max_drawdown_pct') or 0):.1f}%",
            "Sharpe":        f"{sh:.2f}" if sh is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Per-strategy detail expanders ──────────────────────────────────
    # Click any strategy below to see its equity curve, trade markers on
    # the price chart, and full trade list — same view as Single Strategy
    # mode, but for any ticker. Recommended winner is expanded by default.
    st.markdown("##### Strategy details — click to expand")
    best_name = best["strategy_name"] if best else None
    for row in cmp:
        sname = row.get("strategy_name", "?")
        is_winner = (sname == best_name)
        if row.get("error"):
            with st.expander(f"❌ {sname} — ERROR", expanded=False):
                st.error(row["error"])
            continue
        label = ("⭐ " if is_winner else "") + sname
        ret = row.get("total_return_pct") or 0
        wr = row.get("win_rate_pct") or 0
        n = row.get("total_trades") or 0
        with st.expander(
            f"{label}  ·  {ret:+.2f}% return  ·  {wr:.1f}% win rate  ·  {n} trades",
            expanded=is_winner,
        ):
            _render_backtest_metrics(row)
            if not row.get("trades"):
                st.info(
                    "No trades fired for this strategy on this symbol/period. "
                    "Try a longer period (2y or 5y)."
                )
            else:
                _equity_chart(row, symbol=cmp_symbol, period=cmp_period)
                _price_action_chart(row, cmp_symbol, cmp_period, key_suffix=sname)
                _single_trades_table(row["trades"])

    # ── Bar chart of returns (gold = winner) ───────────────────────────
    if valid:
        best_name = best["strategy_name"]
        fig = go.Figure(go.Bar(
            x=[r["strategy_name"].replace("_", " ") for r in valid],
            y=[r.get("total_return_pct") or 0 for r in valid],
            marker_color=[
                "#FFD700" if r["strategy_name"] == best_name
                else ("#00d4aa" if (r.get("total_return_pct") or 0) >= 0 else "#ff4b4b")
                for r in valid
            ],
            text=[("⭐ " if r["strategy_name"] == best_name else "") +
                  f"{(r.get('total_return_pct') or 0):+.1f}%" for r in valid],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Total Return — {cmp_symbol} ({cmp_period})  |  Gold = Recommended",
            height=350, template="plotly_dark",
            yaxis_title="Total Return (%)", xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Promote to auto-trade ──────────────────────────────────────────
    promotable = [r for r in cmp if not r.get("error")]
    if promotable:
        st.divider()
        st.markdown("##### Promote a strategy to auto-trade")
        st.caption(
            "Picks the strategy/symbol pair the live scheduler should use for this ticker. "
            "Notifications fire on every BUY/SELL signal once promoted."
        )
        p1, p2, p3 = st.columns([3, 2, 2])
        with p1:
            names = [r["strategy_name"] for r in promotable]
            default_name = best["strategy_name"] if best else names[0]
            pick = st.selectbox(
                "Strategy to promote",
                names,
                index=names.index(default_name) if default_name in names else 0,
                key=f"custom_promote_pick_{cmp_symbol}",
                help="Defaults to the recommended winner. Override if you want a different one.",
            )
        with p2:
            cap_str = st.text_input(
                "Max capital ($)", value="", placeholder="optional",
                key=f"custom_promote_cap_{cmp_symbol}",
                help="Dollar cap. Wins over shares cap when both are set. "
                     "Leave blank to use the global account cap.",
            )
        with p3:
            shares_str = st.text_input(
                "Max shares (qty)", value="", placeholder="optional",
                key=f"custom_promote_shares_{cmp_symbol}",
                help="Shares cap. Used only when the dollar cap is empty.",
            )
        p4, p5, p6 = st.columns([3, 1, 2])
        with p4:
            custom_broker = st.selectbox(
                "Broker route",
                BROKER_VALUES,
                index=0,
                format_func=lambda v: BROKER_LABEL.get(v, v.title()),
                key=f"custom_promote_broker_{cmp_symbol}",
                help="Default = global toggle. Otherwise pins this symbol's "
                     "orders to the selected broker.",
            )
        with p5:
            enabled = st.checkbox("Enabled", value=True, key=f"custom_promote_en_{cmp_symbol}")
        with p6:
            st.write("")
            st.write("")
            go_btn = st.button("Promote", type="primary",
                               use_container_width=True,
                               key=f"custom_promote_btn_{cmp_symbol}")
        if go_btn:
            try:
                cap_val: float | None = None
                if cap_str.strip():
                    cap_val = float(cap_str.strip())
                shares_val: float | None = None
                if shares_str.strip():
                    shares_val = float(shares_str.strip())
                api.upsert_assignment(
                    symbol=cmp_symbol,
                    system="scanner",
                    strategy_name=pick,
                    enabled=enabled,
                    notes=f"Promoted from Custom Symbol compare ({cmp_period})",
                    max_capital_usd=cap_val,
                    max_shares=shares_val,
                    broker=custom_broker,
                )
                st.success(
                    f"Assigned **{pick}** to **{cmp_symbol}** "
                    f"(scanner, enabled={enabled}). "
                    "BUY/SELL signals on this symbol will now create notifications."
                )
            except Exception as exc:
                st.error(f"Promote failed: {exc}")
