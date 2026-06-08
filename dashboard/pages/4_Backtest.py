from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, divider as _divider
from _components import page_header, stat_band, empty_state, regime_chip
import _charts as charts

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# Shared trader-style ranking — the SAME service the scanner badge and the
# /recommendations API use, so the banner here can never disagree with them.
from app.services.recommendations.winner import score_strategies

apply_theme("Backtest")
page_header(
    "Strategy Backtester",
    subtitle="Simulates how a strategy would have performed on historical data — no real money involved.",
)

# ── Broker route options (per-assignment override at Promote time) ─────────────
# "default" defers to the global active_broker / trade_routing toggle; the
# others pin orders for the promoted symbol to a specific broker adapter.
BROKER_OPTIONS: list[tuple[str, str]] = [
    ("default", "Default (use global toggle)"),
    ("schwab",  "Schwab"),
    ("webull",  "Webull"),
    ("zerodha", "Zerodha (India)"),
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
    section("Recommended Filters for This Symbol")
    for item in rec["key_filters"]:
        st.markdown(f"- {item}")


# Scanner strategy types that support per-symbol calibration (Optimize Filters).
_CALIBRATABLE_TYPES = {
    "rsi2_mean_reversion", "ema_macd_crossover", "bb_squeeze_breakout",
    "pullback_ema50", "vix_spike_reversal",
}

# Human labels for the param overrides shown in the "applied filters" table.
_PARAM_LABELS = {
    "rsi_entry_threshold": "RSI(2) entry ceiling",
    "atr_skip_threshold":  "Max ATR% (skip if calmer needed)",
    "vol_ratio_min":       "Min volume vs 20d avg",
    "rsi_min":             "Min RSI at entry",
    "rsi_entry_min":       "Min RSI at breakout",
    "rsi_entry_max":       "Max RSI at entry (panic depth)",
    "wick_ratio_min":      "Min reclaim wick ratio",
    "price_ema_proximity_pct": "Max distance from EMA50 (%)",
    "atr_spike_threshold": "Min ATR% panic spike",
}


def _render_optimize_filters(symbol: str, strategy_type: str, period: str) -> None:
    """
    Perplexity-style Optimize-Filters-and-save for scanner strategies.
    Analyzes winning vs losing entries, proposes tighter per-symbol params,
    re-runs, and saves only if it improves win-rate OR expectancy OR return
    with >=40% trade survival. Saved params flow straight into the live
    scheduler/scanner via _make_generic_configs.
    """
    if strategy_type not in _CALIBRATABLE_TYPES:
        return

    st.divider()
    st.markdown("##### 🔬 Analyze & Optimize Filters")
    st.caption(
        "Grid-searches this strategy's own entry params for **this symbol** "
        "(pullback proximity, RSI window, reclaim wick, volume/ATR thresholds — "
        "whatever the rule actually gates on), maximising per-trade expectancy with "
        "≥40% of trades surviving. Saves only if it materially beats the factory "
        "defaults on win rate, expectancy, or total return. Saved params take effect "
        "immediately in the live scanner and scheduler — no restart. (~30–60s.)"
    )

    # Show the currently-saved profile (if any) with a revert button.
    try:
        existing = api.scanner_profile_get(symbol, strategy_type)
    except Exception:
        existing = {}
    if existing and existing.get("param_overrides"):
        ov = existing["param_overrides"]
        st.info(
            f"📌 Saved profile active for **{symbol} / {strategy_type}** "
            f"(calibrated {existing.get('calibrated_at','?')}, "
            f"{existing.get('win_rate_pct','?')}% win rate). "
            f"Overrides: " + ", ".join(f"{_PARAM_LABELS.get(k,k)} = {v}" for k, v in ov.items())
        )
        if st.button("🗑 Revert to factory defaults", key=f"revert_{symbol}_{strategy_type}"):
            try:
                api.scanner_profile_delete(symbol, strategy_type)
                st.success("Reverted. Re-run the backtest to see factory-default results.")
                st.rerun()
            except Exception as e:
                st.error(f"Revert failed: {e}")

    cal_period = st.selectbox(
        "Calibration period (more history = more trades to learn from)",
        ["2y", "5y", "10y"], index=1, key=f"cal_period_{symbol}_{strategy_type}",
    )
    if not st.button("⚡ Optimize Filters → Save Profile", type="primary",
                     key=f"optimize_{symbol}_{strategy_type}"):
        return

    with st.spinner(f"Analyzing {symbol} / {strategy_type} over {cal_period}..."):
        try:
            cal = api.scanner_calibrate(symbol, strategy_type, period=cal_period)
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            return

    if cal.get("saved"):
        cmp = cal.get("comparison", {})
        b, f = cmp.get("baseline", {}), cmp.get("filtered", {})
        bwr, fwr = b.get("win_rate_pct", 0), f.get("win_rate_pct", 0)
        st.success(
            f"✅ Profile saved for **{symbol} / {strategy_type}** — "
            f"Win rate **{bwr:.1f}% → {fwr:.1f}%**  |  "
            f"Trades kept: {f.get('round_trips','?')} "
            f"({cmp.get('survival_rate_pct','?')}% survival)"
        )
        st.caption("These params are now live in the scanner and scheduler for this symbol.")
    else:
        st.warning(cal.get("skip_reason") or "No improvement — factory defaults kept.")

    # Proposed param changes (what the optimizer decided to tighten).
    overrides = cal.get("param_overrides") or cal.get("proposed_overrides") or {}
    if overrides:
        import pandas as _pd
        st.markdown("**Param changes proposed**" + ("" if cal.get("saved") else " (not saved)"))
        prows = [{"Param": _PARAM_LABELS.get(k, k), "New value": v} for k, v in overrides.items()]
        st.dataframe(_pd.DataFrame(prows), use_container_width=True, hide_index=True)

    # With/without comparison table — always shown so the user sees the effect.
    cmp = cal.get("comparison") or {}
    if cmp.get("baseline") and cmp.get("filtered"):
        b, f = cmp["baseline"], cmp["filtered"]
        improved = set(cmp.get("improved_by") or [])
        st.markdown("##### With vs without optimization")
        if improved:
            st.caption("Improved on: " + ", ".join(
                {"win_rate": "win rate", "expectancy": "expectancy",
                 "total_return": "total return"}.get(m, m) for m in improved))

        def _d(key, suffix=""):
            bv, fv = b.get(key), f.get(key)
            if bv is None or fv is None:
                return "—", "—", ""
            delta = fv - bv
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
            return f"{bv:g}{suffix}", f"{fv:g}{suffix}", f"{arrow} {delta:+.2f}{suffix}"

        import pandas as _pd
        rows = []
        for label, key, suf in [
            ("Round trips",   "round_trips",      ""),
            ("Win rate",      "win_rate_pct",     "%"),
            ("Avg win",       "avg_win_pct",      "%"),
            ("Avg loss",      "avg_loss_pct",     "%"),
            ("Expectancy",    "expectancy_pct",   "%"),
            ("Total return",  "total_return_pct", "%"),
            ("Profit factor", "profit_factor",    ""),
            ("Max drawdown",  "max_drawdown_pct", "%"),
        ]:
            bs, fs, dl = _d(key, suf)
            rows.append({"Metric": label, "Factory defaults": bs,
                         "Optimized": fs, "Δ": dl})
        st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Optimized run kept {cmp.get('survival_rate_pct','—')}% of the baseline trades. "
            "Δ is optimized minus factory; for max drawdown, closer to zero is better."
        )


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
    section("Market Regime Analysis", level=3)

    # Regime chips side by side then metrics
    chip_html = (
        f"{regime_chip('BULL_OPEN')} &nbsp; {len(bull_trades)} trades &nbsp;&nbsp; "
        f"{regime_chip('BEAR_OPEN')} &nbsp; {len(bear_trades)} trades"
    )
    st.markdown(chip_html, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("BULL Regime Trades", len(bull_trades))
    c2.metric("BEAR Regime Trades", len(bear_trades))
    c3.metric("Total Trades",       len(regime_trades))


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
    section("Equity Curve — OHLC view")
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
    section(f"Price Action — {symbol} with Trade Markers")
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
    # Entry (BUY) rows carry no realised P/L. In a DataFrame the None is
    # upcast to NaN, so guard for both — otherwise NaN slips to the negative
    # branch and renders as "-$nan".
    if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
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

    # Separate Approach C signal markers from actual fills
    signal_markers = [t for t in trades if t.get("side") == "SELL_SIGNAL"]
    real_trades    = [t for t in trades if t.get("side") != "SELL_SIGNAL"]

    section(
        f"All Trades — {len(real_trades)} total"
        + (f"  ·  {len(signal_markers)} Approach C signal(s)" if signal_markers else ""),
        level=3,
    )
    if not real_trades:
        empty_state("No trades generated", "Try a longer period (2y) or a different strategy.", icon="📉")
        return

    rows = []
    pending_buy = None
    for t in real_trades:
        if t["side"] == "BUY":
            pending_buy = t
            rows.append({**t, "pnl": None})
        elif t["side"] in ("SELL", "SELL (close)") or \
             (t["side"].startswith("SELL") and t.get("quantity", 0) > 0):
            pnl = round(t["value"] - pending_buy["value"], 2) if pending_buy else None
            rows.append({**t, "pnl": pnl})
            pending_buy = None

    df = pd.DataFrame(rows)
    df["side"]  = df["side"].apply(_side_tag)
    df["value"] = df["value"].apply(lambda v: f"${v:,.2f}")
    df["price"] = df["price"].apply(lambda v: f"${v:,.2f}")
    df["pnl"]   = df["pnl"].apply(_pnl_tag)
    df = df.rename(columns={"pnl": "profit / loss"})
    for col in ["signal_from", "regime"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date":          st.column_config.TextColumn("Date",     width="small"),
            "side":          st.column_config.TextColumn("Side",     width="small"),
            "price":         st.column_config.TextColumn("Price",    width="small"),
            "quantity":      st.column_config.NumberColumn("Qty",    format="%.4f", width="small"),
            "value":         st.column_config.TextColumn("Value",    width="small"),
            "profit / loss": st.column_config.TextColumn("P&L",      width="small"),
        },
    )

    # Show Approach C signal markers as a separate informational table
    if signal_markers:
        with st.expander(f"Approach C — SELL signal markers ({len(signal_markers)})", expanded=False):
            st.caption(
                "These are the points where the assigned strategy fired a SELL signal "
                "and the 2% tight trailing stop was placed. The position stayed open "
                "until the trail hit — the actual exit is the SELL row above this marker."
            )
            sig_df = pd.DataFrame([{
                "Signal date": t.get("date"),
                "Signal price": f"${t['price']:,.2f}",
                "Strategy": (t.get("signal_from") or "").replace(":approach_c_signal", ""),
            } for t in signal_markers])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)


def _consensus_trades_table(trades: list) -> None:
    st.divider()
    section(f"All Trades — {len(trades)} total", level=3)
    if not trades:
        empty_state("No trades generated", "Lower min_agreement or try a longer period.", icon="📉")
        return

    rows = [{
        "date":          t.get("date"),
        "side":          _side_tag(t.get("side", "")),
        "price":         t["price"] if t.get("price") is not None else None,
        "quantity":      t.get("quantity"),
        "value":         t["value"] if t.get("value") is not None else None,
        "agreed by":     ", ".join(t.get("agreeing", [])) or "—",
        "profit / loss": _pnl_tag(t.get("pnl")),
    } for t in trades]

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "date":          st.column_config.TextColumn("Date",      width="small"),
            "side":          st.column_config.TextColumn("Side",      width="small"),
            "price":         st.column_config.NumberColumn("Price",   format="$%.2f", width="small"),
            "quantity":      st.column_config.NumberColumn("Qty",     format="%.4f",  width="small"),
            "value":         st.column_config.NumberColumn("Value",   format="$%.2f", width="small"),
            "agreed by":     st.column_config.TextColumn("Agreed by", width="large"),
            "profit / loss": st.column_config.TextColumn("P&L",       width="small"),
        },
    )


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
# MARKET SELECTOR — US vs India
# ══════════════════════════════════════════════════════════════
# The backtest backend already routes India (NSE) symbols to the right data
# path (Upstox → yfinance .NS) via the exchange-aware provider, so any of the
# modes below works on NSE tickers. This selector just makes India explicit:
# it surfaces the Nifty 50 quick-pick and reminds you what to type.
_NIFTY50_QUICKPICK = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "HCLTECH", "BAJAJFINSV", "TECHM", "GRASIM",
    "INDUSINDBK", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "DIVISLAB", "HINDALCO", "BPCL", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN",
]
bt_market = st.radio(
    "Market", ["🇺🇸 US", "🇮🇳 India (NSE)"], horizontal=True, key="bt_market",
    help="US = Schwab/Webull universe (data via yfinance). India = NSE symbols "
         "(orders route to Zerodha; data via Upstox → yfinance .NS). The backend "
         "auto-detects the market from the symbol, so this is mainly a helper.",
)
if bt_market.startswith("🇮🇳"):
    st.info(
        "**India mode.** Type any NSE symbol into the modes below (e.g. "
        "**RELIANCE**, **TCS**, **INFY**) — it will auto-fetch India data and "
        "trade/price in ₹. For a deep India cockpit (positions, Nifty 50 scan, "
        "Perplexity-vs-scanner compare), use the dedicated **India** page."
    )
    with st.expander("Nifty 50 quick reference (copy a symbol)", expanded=False):
        st.dataframe(
            pd.DataFrame({"NSE symbol": _NIFTY50_QUICKPICK}),
            use_container_width=True, hide_index=True, height=240,
        )

# ══════════════════════════════════════════════════════════════
# MODE TOGGLE
# ══════════════════════════════════════════════════════════════
_render_recommendation_overview()
mode = st.radio(
    "Backtest Mode",
    ["Single Strategy", "Consensus Mode", "Custom Symbol", "Walk-Forward OOS"],
    horizontal=True,
    help="Single: test one strategy alone.\n"
         "Consensus: test a symbol using its configured strategies with agreement filtering.\n"
         "Custom Symbol: test ANY ticker (e.g. BRK-B) using the 5 scanner strategies — "
         "useful for reproducing scanner candidates.\n"
         "Walk-Forward OOS: split history into in-sample/out-of-sample windows and "
         "measure how much of the in-sample edge survives out-of-sample (overfit check).",
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

    _india = bt_market.startswith("🇮🇳")
    col1, col2, col3, col4, col5 = st.columns([2, 2.5, 1.5, 2, 1])
    with col1:
        chosen_sym = st.text_input(
            "Symbol", value="RELIANCE" if _india else "AAPL",
            help="Any Yahoo-Finance-compatible ticker. US: AAPL, NVDA, TQQQ. "
                 "India: RELIANCE, TCS, INFY (NSE symbols auto-fetch India data).",
        ).strip().upper()
    with col2:
        chosen_type = st.selectbox(
            "Strategy", _types,
            format_func=lambda t: _label_by_type[t],
            index=3,  # Pullback EMA50 — the highest-frequency strategy
        )
    with col3:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y", "8y", "10y"], index=1)
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

    # ── Exit-policy override (per-run, doesn't touch saved profiles) ──────
    with st.expander("Exit-policy override (compare runs with layers off)", expanded=False):
        st.caption(
            "These toggles only affect *this* backtest run — they strip the "
            "selected exit layer(s) from the params before the engine starts. "
            "Saved scanner profiles, strategies.json, and live trading are "
            "untouched."
        )
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            disable_trail = st.checkbox(
                "Disable Layer 2 — Chandelier trailing overlay",
                value=False, key="bt_disable_trail",
                help="Removes trail_enabled / trail_trigger_pct / atr_trail_mult / "
                     "atr_trail_period from params. Useful to A/B the trail's "
                     "contribution for symbols that have it on (NVDA, COP, TOST, "
                     "AVPT, ARLO).",
            )
        with oc2:
            disable_exit_policy = st.checkbox(
                "Disable Layer 3 — exit_policy dict",
                value=False, key="bt_disable_exit_policy",
                help="Removes the Phase-1 exit_policy. Only relevant for the "
                     "unified strategy types which carry one; legacy types "
                     "don't, so this toggle is inert for them.",
            )
        with oc3:
            bt_stop_loss_pct = st.number_input(
                "Hard stop-loss % (Layer 0)",
                min_value=0.0, max_value=50.0, value=8.0, step=0.5,
                key="bt_stop_loss_pct",
                help="Exit immediately if unrealized loss reaches this % from entry. "
                     "Applied before any strategy signal — cannot be overridden. "
                     "Set to 0 to disable. Default 8%.",
            )

    with st.expander("RSI sell threshold — A/B test old vs new", expanded=False):
        st.caption(
            "RSI sell thresholds were raised to **72** for US market conditions "
            "(was 65–70). Use this slider to compare a specific value against the "
            "new default. Set to 0 to use the strategy default."
        )
        _rc1, _rc2 = st.columns([2, 3])
        with _rc1:
            bt_exit_rsi = st.slider(
                "RSI sell threshold", min_value=0, max_value=95, value=0, step=1,
                key="bt_exit_rsi",
                help="0 = strategy default (72). Try 65 or 70 to compare old behaviour.",
            )
        with _rc2:
            if bt_exit_rsi > 0:
                st.info(f"Overriding sell RSI to **{bt_exit_rsi}** for this run.")
            else:
                st.info("Using strategy default — 72 for US momentum strategies.")

    with st.expander("Approach C — SELL signal → tight trailing stop", expanded=False):
        st.caption(
            "Simulates the live Approach C behaviour: when the assigned strategy fires "
            "a SELL signal, instead of exiting immediately the backtest holds the position "
            "and places a tight trailing stop from the signal price. The position only "
            "closes when the stock drops `trail %` from its post-signal high. "
            "**Step 1:** Slide the trail % to the value you want to test. "
            "**Step 2:** Check Enable Approach C. "
            "**Step 3:** Run Backtest. Repeat with different trail % values to compare."
        )
        _ac1, _ac2 = st.columns([2, 5])
        with _ac1:
            bt_approach_c = st.checkbox(
                "Enable Approach C", value=False, key="bt_approach_c",
                help="SELL signal → tight trailing stop instead of immediate exit.",
            )
        with _ac2:
            # Always slidable — the slider sets which trail % to test.
            # Enable Approach C to activate it in the backtest run.
            bt_tight_trail = st.slider(
                "Tight trail % to test",
                min_value=1.0, max_value=10.0,
                value=2.0, step=0.5, key="bt_tight_trail",
                help="Slide to the trail % you want to test. Works independently of the checkbox — "
                     "set the value first, then enable Approach C and run.",
            )
        if bt_approach_c:
            st.info(
                f"Approach C **ON** — SELL signal activates a **{bt_tight_trail:.1f}% trailing stop** "
                f"from signal price. Run the backtest and compare the result to the default (Approach C off). "
                f"Try 1.5%, 2%, 3% to find the best value for this symbol."
            )
        else:
            st.info(
                f"Approach C **OFF** — SELL signal exits immediately (default). "
                f"Trail % set to **{bt_tight_trail:.1f}%** — enable Approach C and run to test it."
            )

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
                    period=period, initial_capital=capital,
                    disable_trail=disable_trail,
                    disable_exit_policy=disable_exit_policy,
                    stop_loss_pct=float(bt_stop_loss_pct),
                    exit_rsi=float(bt_exit_rsi),
                    approach_c=bt_approach_c,
                    tight_trail_pct=float(bt_tight_trail),
                    timeout=120,
                )
                st.session_state["bt_result"] = result
                st.session_state.pop("bt_consensus", None)

                # When Approach C is ON, also run default and all trail % variants
                # in one shot so the comparison table is available immediately.
                if bt_approach_c:
                    trail_variants = {}
                    # Default (no Approach C)
                    r_def = api.backtest_run_generic(
                        chosen_sym, chosen_type,
                        period=period, initial_capital=capital,
                        disable_trail=disable_trail,
                        disable_exit_policy=disable_exit_policy,
                        stop_loss_pct=float(bt_stop_loss_pct),
                        exit_rsi=float(bt_exit_rsi),
                        approach_c=False,
                        timeout=60,
                    )
                    trail_variants["default"] = r_def
                    # Test a range of trail % around the chosen value
                    tested_pcts = sorted({1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, float(bt_tight_trail)})
                    for tp in tested_pcts:
                        try:
                            rv = api.backtest_run_generic(
                                chosen_sym, chosen_type,
                                period=period, initial_capital=capital,
                                disable_trail=disable_trail,
                                disable_exit_policy=disable_exit_policy,
                                stop_loss_pct=float(bt_stop_loss_pct),
                                exit_rsi=float(bt_exit_rsi),
                                approach_c=True,
                                tight_trail_pct=tp,
                                timeout=60,
                            )
                            trail_variants[tp] = rv
                        except Exception:
                            pass
                    st.session_state["bt_trail_variants"] = trail_variants
                else:
                    st.session_state.pop("bt_trail_variants", None)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

    # Keep `chosen` populated so the Promote block below still works.
    chosen = st.session_state.get("bt_result", {}).get("strategy_name", "")

    r = st.session_state.get("bt_result")
    if not r:
        st.stop()

    section(f"Results — {r['strategy_name']} ({r['start_date']} → {r['end_date']})")

    _render_backtest_metrics(r)

    # ── Approach C comparison table ───────────────────────────────────────────
    _trail_variants = st.session_state.get("bt_trail_variants")
    if _trail_variants:
        st.divider()
        st.markdown("#### Approach C — Trail % Comparison")
        st.caption(
            "All trail % values tested in one run. **Highlighted row** = the trail % "
            "you selected. Use this to pick the best value before promoting."
        )
        _selected_trail = float(st.session_state.get("bt_tight_trail", 2.0))
        _cmp_rows = []
        # Default row
        _d = _trail_variants.get("default", {})
        _cmp_rows.append({
            "Mode": "Default (immediate SELL)",
            "Trail %": "—",
            "Trades": _d.get("total_trades", 0),
            "Win Rate": f"{_d.get('win_rate_pct', 0):.1f}%",
            "Return": f"{_d.get('total_return_pct', 0):+.2f}%",
            "Total P/L": f"${_d.get('total_pnl', 0):+,.0f}",
            "Max DD": f"{_d.get('max_drawdown_pct', 0):.1f}%",
            "Sharpe": f"{_d.get('sharpe_ratio') or '—'}",
            "_ret": _d.get("total_return_pct", 0),
            "_selected": False,
        })
        # Approach C rows
        for tp, rv in sorted((k, v) for k, v in _trail_variants.items() if k != "default"):
            _cmp_rows.append({
                "Mode": f"Approach C",
                "Trail %": f"{tp:.1f}%",
                "Trades": rv.get("total_trades", 0),
                "Win Rate": f"{rv.get('win_rate_pct', 0):.1f}%",
                "Return": f"{rv.get('total_return_pct', 0):+.2f}%",
                "Total P/L": f"${rv.get('total_pnl', 0):+,.0f}",
                "Max DD": f"{rv.get('max_drawdown_pct', 0):.1f}%",
                "Sharpe": f"{rv.get('sharpe_ratio') or '—'}",
                "_ret": rv.get("total_return_pct", 0),
                "_selected": abs(tp - _selected_trail) < 0.01,
            })

        # Find best return row
        best_ret = max(row["_ret"] for row in _cmp_rows)

        import pandas as pd
        _df_cmp = pd.DataFrame([
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in _cmp_rows
        ])

        def _style_row(row):
            idx = _df_cmp.index[_df_cmp["Trail %"] == row["Trail %"]].tolist()
            if not idx: return [""] * len(row)
            orig = _cmp_rows[idx[0]]
            if orig["_selected"]:
                return ["background-color: #1a3a4a; font-weight: bold"] * len(row)
            if abs(orig["_ret"] - best_ret) < 0.01:
                return ["background-color: #0f3b1a"] * len(row)
            return [""] * len(row)

        st.dataframe(
            _df_cmp.style.apply(_style_row, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Dark blue = your selected trail %. Dark green = best return. "
            "Pick the trail % that balances highest return with acceptable drawdown, "
            "then set the **Promote** slider to that value below."
        )

    # Effective exit policy: shows which layers actually ran for THIS result
    # (after any override toggles were applied). Compare two runs side-by-side
    # by toggling Disable Layer 2/3 and re-running.
    _eff_params = r.get("params")
    _eff_stype = r.get("strategy_type") or chosen_type
    if _eff_params is not None:
        import _exit_policy as exit_policy
        ov = r.get("overrides") or {}
        override_notes = []
        if ov.get("disable_trail"): override_notes.append("chandelier trail disabled")
        if ov.get("disable_exit_policy"): override_notes.append("exit policy disabled")
        if ov.get("exit_rsi"): override_notes.append(f"RSI sell threshold = {ov['exit_rsi']}")
        if ov.get("approach_c"): override_notes.append(f"Approach C — {ov.get('tight_trail_pct',2)}% tight trail on SELL signal")
        if override_notes:
            st.caption("Overrides for this run: " + " · ".join(override_notes))
        exit_policy.render(_eff_stype, _eff_params)

    if not r.get("trades"):
        _render_zero_trade_debug_single(chosen_sym, chosen, period)

    _render_regime_panel(r.get("trades", []), period)
    _render_filter_summary(chosen_sym)
    _equity_chart(r, symbol=chosen_sym, period=period)
    _price_action_chart(r, chosen_sym, period)
    _single_trades_table(r["trades"])

    # Analyze winners/losers and auto-tune this symbol's entry params.
    _render_optimize_filters(chosen_sym, chosen_type, period)

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

    # Tight trail % — pre-filled from the backtest Approach C slider if it was run
    _bt_trail_default = float(st.session_state.get("bt_tight_trail", 2.0))
    _approach_c_was_run = bool(st.session_state.get("bt_approach_c", False))
    p_trail1, p_trail2 = st.columns([2, 3])
    with p_trail1:
        single_tight_trail = st.slider(
            "Approach C tight trail %",
            min_value=1.0, max_value=10.0,
            value=_bt_trail_default,
            step=0.5,
            key=f"single_promote_trail_{chosen_sym}_{chosen}",
            help="Tight trailing stop % placed when the assigned strategy fires a SELL signal. "
                 "Pre-filled from your backtest Approach C slider. "
                 "Low-vol stocks (KO, SO): 2–3%. High-vol (NVDA, TSLA): 3–5%.",
        )
    with p_trail2:
        if _approach_c_was_run:
            st.info(
                f"Pre-filled from your backtest Approach C run ({_bt_trail_default:.1f}%). "
                f"Adjust if you tested multiple trail values."
            )
        else:
            st.info(
                "Run the backtest with **Approach C enabled** first to find the best trail % "
                "for this symbol, then promote with that value."
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
                notes=f"Promoted from Single Strategy backtest ({period}), trail={single_tight_trail:.1f}%",
                max_capital_usd=single_cap_val,
                max_shares=single_shares_val,
                broker=single_broker,
                tight_trail_pct=single_tight_trail,
            )
            st.success(
                f"Assigned **{r['strategy_name']}** to **{chosen_sym}** "
                f"with **{single_tight_trail:.1f}% tight trail** "
                f"(enabled={single_enabled}). "
                "SELL signals will place a tight trailing stop at this distance."
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
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y", "8y", "10y"], index=1)
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
    section(
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
elif mode == "Custom Symbol":
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
            "Symbol", value="RELIANCE" if bt_market.startswith("🇮🇳") else "BRK-B",
            help="Any Yahoo-Finance-compatible ticker. US: BRK-B (hyphen, not BRK.B). "
                 "India: RELIANCE, TCS, INFY (NSE symbols auto-fetch India data).",
            key="custom_bt_sym",
        ).strip().upper()
    with c2:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y", "8y", "10y"], index=1,
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
            # Effective exit policy for this row (Layer 1 always; Layers 2/3
            # only when present in the params used). Lets the trader see at a
            # glance which strategies have the trail on vs off for this symbol.
            row_params = row.get("params")
            row_stype = row.get("strategy_type")
            if row_params is not None and row_stype:
                import _exit_policy as exit_policy
                exit_policy.render(row_stype, row_params)
            if not row.get("trades"):
                st.info(
                    "No trades fired for this strategy on this symbol/period. "
                    "Try a longer period (2y, 5y, 8y, or 10y)."
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
        ctrail1, ctrail2 = st.columns([2, 3])
        with ctrail1:
            custom_tight_trail = st.slider(
                "Approach C tight trail %",
                min_value=1.0, max_value=10.0, value=2.0, step=0.5,
                key=f"custom_promote_trail_{cmp_symbol}",
                help="Tight trailing stop % when SELL signal fires. "
                     "Run Approach C in the compare table to find the best value for this symbol.",
            )
        with ctrail2:
            st.info(
                f"SELL signal → **{custom_tight_trail:.1f}% tight trailing stop** placed from signal price. "
                "Enable Approach C in the backtest expanders above to test different values first."
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
                    notes=f"Promoted from Custom Symbol compare ({cmp_period}), trail={custom_tight_trail:.1f}%",
                    max_capital_usd=cap_val,
                    max_shares=shares_val,
                    broker=custom_broker,
                    tight_trail_pct=custom_tight_trail,
                )
                st.success(
                    f"Assigned **{pick}** to **{cmp_symbol}** "
                    f"with **{custom_tight_trail:.1f}% tight trail** "
                    f"(enabled={enabled}). "
                    "SELL signals will place a tight trailing stop at this distance."
                )
            except Exception as exc:
                st.error(f"Promote failed: {exc}")

# ══════════════════════════════════════════════════════════════
# WALK-FORWARD OOS MODE
# ══════════════════════════════════════════════════════════════
elif mode == "Walk-Forward OOS":
    st.info(
        "**Walk-forward out-of-sample validation.** History is split into an "
        "in-sample (IS) window the strategy is 'fit' to and an out-of-sample "
        "(OOS) window it never saw. The key number is **Walk-Forward Efficiency "
        "(WFE) = OOS CAGR ÷ IS CAGR** — how much of the in-sample edge survives "
        "out of sample. ~1.0 is excellent; below ~0.5 suggests overfitting or a "
        "regime change. Drives the same no-lookahead engine as a normal backtest, "
        "so the numbers are directly comparable."
    )

    WF_CHOICES: list[tuple[str, str]] = [
        ("rsi2_mean_reversion",  "RSI-2 Mean Reversion"),
        ("ema_macd_crossover",   "EMA + MACD Crossover"),
        ("bb_squeeze_breakout",  "Bollinger Squeeze Breakout"),
        ("pullback_ema50",       "Pullback to EMA(50)"),
        ("vix_spike_reversal",   "VIX Spike Reversal"),
        ("bollinger",            "Legacy: Bollinger Mean Reversion"),
        ("fib_pullback",         "Legacy: Fibonacci Pullback"),
    ]
    _wf_label_by_type = {t: lbl for t, lbl in WF_CHOICES}
    _wf_india = bt_market.startswith("🇮🇳")

    wc1, wc2, wc3 = st.columns([2, 2.5, 2])
    with wc1:
        wf_sym = st.text_input(
            "Symbol", value="RELIANCE" if _wf_india else "NVDA",
            key="wf_symbol",
            help="Any ticker. US: NVDA, AAPL. India: RELIANCE, TCS (NSE).",
        ).strip().upper()
    with wc3:
        wf_mode = st.radio(
            "Method", ["Simple split", "Rolling windows", "Scan all strategies"],
            horizontal=True, key="wf_mode",
            help="Simple: one IS/OOS split for the chosen strategy. "
                 "Rolling: many sliding windows stitched into a composite OOS "
                 "curve (needs long history). "
                 "Scan all: runs the simple split for EVERY strategy in parallel "
                 "and ranks them by WFE so you can see which strategy actually "
                 "fits this symbol.",
        )
    with wc2:
        if wf_mode == "Scan all strategies":
            st.caption(
                f"Scanning **{len(WF_CHOICES)} strategies** in parallel on the "
                "same symbol / period / split. WFE-ranked summary below."
            )
            wf_type = None
        else:
            wf_type = st.selectbox(
                "Strategy", [t for t, _ in WF_CHOICES],
                format_func=lambda t: _wf_label_by_type[t],
                index=0, key="wf_type",
            )

    if wf_mode == "Simple split":
        sc1, sc2, sc3 = st.columns([2, 2, 2])
        with sc1:
            wf_period = st.selectbox("Period", ["2y", "5y", "10y"], index=2, key="wf_period_s")
        with sc2:
            wf_train_pct = st.slider("In-sample %", 50, 85, 70, 5, key="wf_train_pct") / 100.0
        with sc3:
            wf_cap = st.number_input("Capital ($)", value=100000, min_value=1000,
                                     step=10000, key="wf_cap_s")
        wf_kwargs = dict(mode="simple", period=wf_period, train_pct=wf_train_pct,
                         initial_capital=wf_cap)
    elif wf_mode == "Scan all strategies":
        sc1, sc2, sc3 = st.columns([2, 2, 2])
        with sc1:
            wf_period = st.selectbox("Period", ["2y", "5y", "10y"], index=2, key="wf_period_scan")
        with sc2:
            wf_train_pct = st.slider("In-sample %", 50, 85, 70, 5, key="wf_train_pct_scan") / 100.0
        with sc3:
            wf_cap = st.number_input("Capital ($)", value=100000, min_value=1000,
                                     step=10000, key="wf_cap_scan")
        wf_kwargs = dict(mode="simple", period=wf_period, train_pct=wf_train_pct,
                         initial_capital=wf_cap)
    else:
        rc1, rc2, rc3, rc4 = st.columns([1.5, 1.5, 1.5, 2])
        with rc1:
            wf_period = st.selectbox("Period", ["5y", "10y"], index=1, key="wf_period_r")
        with rc2:
            wf_train_y = st.number_input("Train yrs", value=3.0, min_value=1.0,
                                         max_value=6.0, step=0.5, key="wf_train_y")
        with rc3:
            wf_test_y = st.number_input("Test yrs", value=1.0, min_value=0.5,
                                        max_value=3.0, step=0.5, key="wf_test_y")
        with rc4:
            wf_cap = st.number_input("Capital ($)", value=100000, min_value=1000,
                                     step=10000, key="wf_cap_r")
        wf_kwargs = dict(mode="rolling", period=wf_period, train_years=wf_train_y,
                         test_years=wf_test_y, step_years=wf_test_y, initial_capital=wf_cap)

    if wf_mode == "Scan all strategies":
        if st.button("▶ Scan all strategies", type="primary", key="wf_scan_run"):
            if not wf_sym:
                st.warning("Enter a symbol.")
                st.stop()
            import concurrent.futures
            results: list[dict] = []
            errors: list[tuple[str, str]] = []
            progress = st.progress(0.0, text=f"Walk-forward scan: 0 / {len(WF_CHOICES)}")

            def _run_one(item: tuple[str, str]) -> tuple[str, str, dict | Exception]:
                stype, label = item
                try:
                    r = api.backtest_walkforward(wf_sym, stype, **wf_kwargs)
                    return (stype, label, r)
                except Exception as exc:  # noqa: BLE001
                    return (stype, label, exc)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(WF_CHOICES))
            ) as ex:
                futures = [ex.submit(_run_one, item) for item in WF_CHOICES]
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    stype, label, r = fut.result()
                    done += 1
                    progress.progress(
                        done / len(WF_CHOICES),
                        text=f"Walk-forward scan: {done} / {len(WF_CHOICES)}",
                    )
                    if isinstance(r, Exception):
                        errors.append((label, str(r)))
                        continue
                    iss, oos = r.get("is_segment", {}), r.get("oos_segment", {})
                    results.append({
                        "strategy_type": stype,
                        "strategy_name": r.get("strategy_name", label),
                        "wfe": r.get("wfe"),
                        "wfe_label": r.get("wfe_label", ""),
                        "is_cagr": iss.get("cagr", 0.0),
                        "oos_cagr": oos.get("cagr", 0.0),
                        "is_return_pct": iss.get("total_return_pct", 0.0),
                        "oos_return_pct": oos.get("total_return_pct", 0.0),
                        "oos_win_pct": oos.get("win_rate_pct", 0.0),
                        "oos_trades": oos.get("trades", 0),
                        "oos_max_dd_pct": oos.get("max_drawdown_pct", 0.0),
                    })
            progress.empty()
            st.session_state["wf_scan_result"] = {
                "symbol": wf_sym, "kwargs": wf_kwargs,
                "results": results, "errors": errors,
            }
    elif st.button("▶ Run Walk-Forward", type="primary", key="wf_run"):
        if not wf_sym:
            st.warning("Enter a symbol.")
            st.stop()
        with st.spinner(f"Walk-forward {wf_sym} / {_wf_label_by_type[wf_type]}…"):
            try:
                wf = api.backtest_walkforward(wf_sym, wf_type, **wf_kwargs)
            except Exception as exc:
                st.error(f"Walk-forward failed: {exc}")
                st.stop()
        st.session_state["wf_result"] = wf

    # ── Scan-all results table ──────────────────────────────────────────
    scan = st.session_state.get("wf_scan_result")
    if wf_mode == "Scan all strategies" and scan:
        rows = scan.get("results", [])
        if not rows:
            st.error("Scan returned no results. See errors below.")
        else:
            section(f"Scan results — {scan['symbol']}")
            # WFE-rank: highest WFE first; None (no IS edge) sinks to bottom.
            def _wfe_key(r):
                w = r.get("wfe")
                return (w is None, -(w if w is not None else 0.0))
            rows_sorted = sorted(rows, key=_wfe_key)

            # Best fit + simple verdict bucket.
            best = rows_sorted[0]
            excellent = [r for r in rows_sorted
                         if r.get("wfe") is not None and r["wfe"] >= 1.0
                         and r["oos_cagr"] > 0]
            acceptable = [r for r in rows_sorted
                          if r.get("wfe") is not None and 0.7 <= r["wfe"] < 1.0
                          and r["oos_cagr"] > 0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best fit",
                      best.get("strategy_name", "—"),
                      f"WFE {best['wfe']:.2f}" if best.get("wfe") is not None else "WFE N/A")
            m2.metric("Excellent (WFE ≥ 1.0)", len(excellent))
            m3.metric("Acceptable (0.7–1.0)", len(acceptable))
            m4.metric("Strategies scanned", f"{len(rows)} / {len(WF_CHOICES)}")

            df_scan = pd.DataFrame([
                {
                    "Rank": i + 1,
                    "Strategy": r["strategy_name"],
                    "WFE": (f"{r['wfe']:.2f}" if r.get("wfe") is not None else "N/A"),
                    "Verdict": r.get("wfe_label", ""),
                    "IS CAGR %": round(r["is_cagr"], 2),
                    "OOS CAGR %": round(r["oos_cagr"], 2),
                    "OOS Return %": round(r["oos_return_pct"], 2),
                    "OOS Win %": round(r["oos_win_pct"], 2),
                    "OOS Trades": r["oos_trades"],
                    "OOS Max DD %": round(r["oos_max_dd_pct"], 2),
                }
                for i, r in enumerate(rows_sorted)
            ])

            def _wfe_color(val: object) -> str:
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return ""
                if v >= 1.0:
                    return "color: #2ec4b6; font-weight: 600"
                if v >= 0.7:
                    return "color: #f4d35e"
                if v >= 0.3:
                    return "color: #f4a261"
                return "color: #e84545"

            st.dataframe(
                # Styler.applymap was removed in pandas 2.1+; .map has the
                # identical signature.
                df_scan.style.map(_wfe_color, subset=["WFE"]),
                use_container_width=True,
                hide_index=True,
            )
            if scan.get("errors"):
                with st.expander(f"⚠ {len(scan['errors'])} strategy run(s) failed"):
                    for label, msg in scan["errors"]:
                        st.write(f"**{label}** — {msg}")

    wf = st.session_state.get("wf_result")
    if wf_mode != "Scan all strategies" and wf:
        if wf.get("mode") == "simple":
            iss, oos = wf["is_segment"], wf["oos_segment"]
            wfe = wf.get("wfe")
            section(f"{wf.get('strategy_name', wf_type)} — {wf['symbol']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("WFE (OOS/IS)", f"{wfe:.2f}" if wfe is not None else "N/A",
                      help="OOS CAGR ÷ IS CAGR. ~1.0 excellent, <0.5 likely overfit.")
            m2.metric("IS CAGR", f"{iss['cagr']:.1f}%")
            m3.metric("OOS CAGR", f"{oos['cagr']:.1f}%")
            st.caption(f"Verdict: **{wf.get('wfe_label', '')}**")

            st.dataframe(pd.DataFrame([
                {"Window": "In-Sample", "Start": iss["start"], "End": iss["end"],
                 "Bars": iss["bars"], "Return %": iss["total_return_pct"],
                 "CAGR %": iss["cagr"], "Win %": iss["win_rate_pct"],
                 "Trades": iss["trades"], "Max DD %": iss["max_drawdown_pct"]},
                {"Window": "Out-of-Sample", "Start": oos["start"], "End": oos["end"],
                 "Bars": oos["bars"], "Return %": oos["total_return_pct"],
                 "CAGR %": oos["cagr"], "Win %": oos["win_rate_pct"],
                 "Trades": oos["trades"], "Max DD %": oos["max_drawdown_pct"]},
            ]), use_container_width=True, hide_index=True)

            if oos.get("equity_curve"):
                section("Out-of-Sample Equity Curve")
                charts.render_equity_chart(
                    oos["equity_curve"], trades=[],
                    initial_capital=wf_cap,
                    title=f"OOS Equity — {wf['symbol']}", bucket="W", height=420,
                )
        else:  # rolling
            gwfe = wf.get("global_wfe")
            section(f"{wf.get('strategy_name', wf_type)} — {wf['symbol']} (rolling)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Global WFE", f"{gwfe:.2f}" if gwfe is not None else "N/A")
            m2.metric("OOS CAGR (composite)", f"{wf['global_oos_cagr']:.1f}%")
            m3.metric("Avg IS CAGR", f"{wf['avg_is_cagr']:.1f}%")
            m4.metric("Windows", len(wf.get("segments", [])))
            st.caption(f"Verdict: **{wf.get('global_wfe_label', '')}** · "
                       f"composite OOS return {wf['global_oos_total_return_pct']:.1f}%")

            segs = wf.get("segments", [])
            if segs:
                st.dataframe(pd.DataFrame([
                    {"#": s["window"], "IS end": s["is_end"], "OOS end": s["oos_end"],
                     "IS CAGR %": s["is_cagr"], "OOS CAGR %": s["oos_cagr"],
                     "OOS Win %": s["oos_win_rate_pct"], "OOS Trades": s["oos_trades"],
                     "WFE": s["wfe"], "Verdict": s["wfe_label"]}
                    for s in segs
                ]), use_container_width=True, hide_index=True)

            if wf.get("oos_composite_curve"):
                section("Composite Out-of-Sample Equity Curve")
                charts.render_equity_chart(
                    wf["oos_composite_curve"], trades=[],
                    initial_capital=wf_cap,
                    title=f"Composite OOS — {wf['symbol']}", bucket="W", height=420,
                )
