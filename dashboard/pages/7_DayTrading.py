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

sys.path.insert(0, dashboard_root)
from _theme import apply_theme  # noqa: E402
import _charts as charts  # noqa: E402
import api  # noqa: E402
from _broker_routing import render_broker_routing_toggle  # noqa: E402

apply_theme("Day Trading")
st.title("Day Trading")
st.caption(
    f"{len(ALL_STRATEGIES)} intraday strategies on 5m and 15m bars. "
    "All signals expire at market close. No overnight holds."
)

# Provider-health badge: shows which market-data provider has served the most
# bar requests in this API session. Helps spot when TD has been exhausted and
# the fallback chain (Webull -> yfinance) is doing the work.
try:
    _ds = api.daytrading_data_source_status() or {}
    _status = _ds.get("status", "idle")
    _label = _ds.get("label", "unknown")
    _counters = _ds.get("counters", {}) or {}
    _color = {"ok": "green", "degraded": "orange", "idle": "gray"}.get(_status, "gray")
    _detail = (
        f"TD: {_counters.get('twelvedata', 0)} · "
        f"Webull: {_counters.get('webull', 0)} · "
        f"yfinance: {_counters.get('yfinance', 0)}"
    )
    st.markdown(
        f":{_color}-background[**Data: {_label}**] &nbsp; "
        f"<span style='color:#888;font-size:0.85em;'>{_detail}</span>",
        unsafe_allow_html=True,
    )
except Exception:
    pass

# Broker-routing toggle: where live orders get sent. "Both" fans out to
# Schwab + Webull. Webull execution is scaffolded (not yet implemented).
render_broker_routing_toggle(key_suffix="daytrading")

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
    "BollingerMomentum": (
        "Bollinger Momentum Breakout — BB squeeze (band width in lowest 20%) "
        "followed by close above/below the band with EMA + RSI + volume confirmation."
    ),
    "SupertrendTrend": (
        "Supertrend Trend-Following — 15m Supertrend sets macro bias; 5m pullback "
        "to ST/EMA20 with reclaim candle. Stop trails the 5m ST line."
    ),
}

DEFAULT_SYMBOLS = "SPY,QQQ,AAPL,TSLA,NVDA,MSFT,AMZN,META"

# ── Tabs (4 top-level groups → sub-tabs within) ──────────────────────────────
#   Trade    : what an operator wants live during market hours (signals + bot)
#   Research : tools you run before/after the session (backtests, sizing, compare)
#   Scanner  : its own surface — pre-market + per-symbol candidate ranking
#   Settings : configuration that rarely changes (strategy config, symbol policy)
tab_trade, tab_research, tab_scanner, tab_settings = st.tabs([
    "Trade",
    "Research",
    "Scanner",
    "Settings",
])

with tab_trade:
    tab_signals, tab_autotrader = st.tabs(["Live Signals", "Auto Trader"])

with tab_research:
    tab_backtest, tab_compare, tab_sizer, tab_watchlist = st.tabs([
        "Backtest", "Compare All", "Position Sizer", "🔍 Watchlist Analyzer",
    ])

with tab_settings:
    tab_config, tab_policy = st.tabs(["Strategy Config", "Symbol Policy"])


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
    """Equity track for day-trading backtests as OHLC candles with WIN/LOSS markers.

    The runner returns a flat list of floats (one per trade close); we synthesise
    a date axis (one trading day per step) and resample into 5-trade buckets so
    the chart reads as proper candlesticks rather than a noisy step line.
    """
    if not equity_curve or len(equity_curve) < 2:
        fig = go.Figure()
        fig.add_annotation(text="No equity data", showarrow=False)
        fig.update_layout(template="plotly_dark", height=320,
                          margin=dict(l=0, r=0, t=30, b=0), title="Equity Curve")
        return fig

    # Build a synthetic date axis (1 bar per equity step). Bucket every 5 steps
    # into an OHLC candle — the underlying daily line stays visible behind it.
    base = pd.Timestamp.today().normalize() - pd.Timedelta(days=len(equity_curve))
    dates = pd.date_range(base, periods=len(equity_curve), freq="D")
    eq_series = pd.Series(equity_curve, index=dates, name="equity")
    bucket = max(1, len(equity_curve) // 30)  # ~30 candles regardless of length
    resampled = eq_series.resample(f"{bucket}D").ohlc().dropna()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=resampled.index,
        open=resampled["open"], high=resampled["high"],
        low=resampled["low"],   close=resampled["close"],
        name="Equity",
        increasing=dict(line=dict(color=charts.GREEN, width=1), fillcolor=charts.GREEN),
        decreasing=dict(line=dict(color=charts.RED,   width=1), fillcolor=charts.RED),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=equity_curve, mode="lines",
        line=dict(color="rgba(38,166,154,0.55)", width=1.2),
        name="Equity (per-trade)",
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ))

    win_idx  = [i for i, t in enumerate(trades) if t.get("pnl", 0) > 0]
    loss_idx = [i for i, t in enumerate(trades) if t.get("pnl", 0) <= 0]
    # The equity track records mark-to-market AFTER each trade closes, so the
    # marker sits at equity_curve[i+1] (post-trade equity).
    def _eq_at(i):
        j = min(i + 1, len(equity_curve) - 1)
        return equity_curve[j]

    if win_idx:
        fig.add_trace(go.Scatter(
            x=[dates[min(i + 1, len(dates) - 1)] for i in win_idx],
            y=[_eq_at(i) for i in win_idx],
            mode="markers",
            marker=dict(symbol="triangle-up", size=11, color=charts.GREEN,
                        line=dict(color="rgba(0,0,0,0.6)", width=1)),
            name="Winner",
            text=[f"{trades[i].get('direction','')} +${trades[i].get('pnl',0):.2f}" for i in win_idx],
            hovertemplate="%{text}<br>Equity $%{y:,.2f}<extra></extra>",
        ))
    if loss_idx:
        fig.add_trace(go.Scatter(
            x=[dates[min(i + 1, len(dates) - 1)] for i in loss_idx],
            y=[_eq_at(i) for i in loss_idx],
            mode="markers",
            marker=dict(symbol="triangle-down", size=11, color=charts.RED,
                        line=dict(color="rgba(0,0,0,0.6)", width=1)),
            name="Loser",
            text=[f"{trades[i].get('direction','')} ${trades[i].get('pnl',0):.2f}" for i in loss_idx],
            hovertemplate="%{text}<br>Equity $%{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=charts.BG, plot_bgcolor=charts.BG,
        height=420,
        margin=dict(l=8, r=8, t=40, b=8),
        title=dict(text="Equity Curve — OHLC", x=0.01, y=0.97, font=dict(size=14)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                     bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=charts.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=charts.GRID, zeroline=False, tickprefix="$")
    return fig


def _render_pipeline_diagnostics(diag: dict, expanded: bool = False) -> None:
    """Render a structured pipeline diagnostics panel."""
    if not diag:
        return

    data    = diag.get("data", {})
    regime  = diag.get("regime", {})
    signals = diag.get("signals", {})
    brain   = diag.get("brain_filter", {})
    exec_   = diag.get("execution", {})
    dx      = diag.get("diagnosis", {})

    root_cause = dx.get("root_cause", "")
    steps      = dx.get("steps", [])

    raw   = signals.get("raw_signals_generated", 0)
    acc   = brain.get("accepted_signals", raw)  # backtest has no brain filter
    rej   = brain.get("rejected_by_brain_total", 0)
    execs = exec_.get("trades_opened", 0)

    # ── Top-level funnel row ──────────────────────────────────────────────────
    st.markdown("#### Pipeline Funnel")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Raw Signals Found",  raw,
              delta=f"Buy:{signals.get('raw_buy_signals',0)} Sell:{signals.get('raw_sell_signals',0)}",
              delta_color="off")
    f2.metric("Brain Accepted",     acc,
              delta=f"-{rej} rejected" if rej else "backtest mode (no live brain)",
              delta_color="inverse" if rej > 0 else "off")
    f3.metric("Trades Executed",    execs,
              delta=f"{exec_.get('trades_skipped_no_future_bars',0)} skip (no future bars)" if exec_.get('trades_skipped_no_future_bars',0) else None,
              delta_color="off")
    f4.metric("Days Skipped (Regime)", regime.get("days_skipped_by_regime", 0),
              delta=f"/{data.get('trading_days_found', 0)} days total",
              delta_color="off")

    # ── Root cause ────────────────────────────────────────────────────────────
    if root_cause:
        if execs == 0 and raw == 0:
            st.error(f"**Root cause:** {root_cause}")
        elif execs == 0 and acc == 0 and raw > 0:
            st.warning(f"**Root cause:** {root_cause}")
        elif execs == 0 and acc > 0:
            st.warning(f"**Root cause:** {root_cause}")
        else:
            st.success(f"**Status:** {root_cause}")

    # ── Regime distribution ───────────────────────────────────────────────────
    regime_dist = regime.get("distribution", {})
    if regime_dist:
        dist_str = " | ".join(f"{k}: {v}d" for k, v in sorted(regime_dist.items()))
        st.caption(f"Regime distribution: {dist_str}")

    skipped_strats = regime.get("strategies_skipped_by_regime", [])
    if skipped_strats:
        st.caption(f"Strategies blocked by regime: {', '.join(skipped_strats)}")

    # ── First-hour window (9:30–10:30 AM ET) ─────────────────────────────────
    fh = diag.get("first_hour", {})
    fh_raw  = fh.get("raw_signals", 0)
    fh_rej  = fh.get("brain_rejections", 0)
    fh_exec = fh.get("executed_trades", 0)
    fh_bars = fh.get("bars_loaded", 0)
    if fh_bars > 0 or fh_raw > 0:
        with st.expander(f"🕙 First Hour (9:30–10:30 AM) — {fh_raw} signals, {fh_exec} trades", expanded=(fh_raw == 0 or fh_exec == 0)):
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Bars in window",   fh_bars)
            h2.metric("Raw signals",      fh_raw,  delta="no signals — strategy warmup or filters" if fh_raw == 0 else None, delta_color="off")
            h3.metric("Brain rejections", fh_rej,  delta=fh.get("top_rejection_reason", "") or None, delta_color="off")
            h4.metric("Trades executed",  fh_exec)
            fh_counts = fh.get("rejection_counts", {})
            if fh_counts:
                st.caption("Rejection breakdown: " + " | ".join(f"{k}: {v}" for k, v in sorted(fh_counts.items(), key=lambda x: -x[1])))
            if fh_raw == 0 and fh_bars > 0:
                st.info("No raw signals generated in the first hour. Typical causes: indicator warmup bars not met (EMA/RSI need history), minimum-bars checks, or time-cutoff rules in the strategy.")
            elif fh_exec == 0 and fh_raw > 0:
                top = fh.get("top_rejection_reason", "unknown")
                st.warning(f"Signals generated but none executed — all rejected by brain. Top reason: **{top}**")

    # ── Brain rejection breakdown ─────────────────────────────────────────────
    if rej > 0:
        with st.expander(f"Brain rejection breakdown ({rej} rejected)", expanded=False):
            rb1, rb2, rb3, rb4, rb5, rb6 = st.columns(6)
            rb1.metric("Regime",    brain.get("rejected_by_regime", 0))
            rb2.metric("Volume",    brain.get("rejected_by_volume", 0))
            rb3.metric("R:R",       brain.get("rejected_by_rr", 0))
            rb4.metric("Time",      brain.get("rejected_by_time", 0))
            rb5.metric("Extension", brain.get("rejected_by_extension", 0))
            rb6.metric("Kill sw.",  brain.get("rejected_by_kill_switch", 0))
            reasons = brain.get("rejection_reasons", [])
            if reasons:
                st.markdown("**Sample rejection reasons:**")
                for r in reasons:
                    st.caption(f"• {r}")

    # ── Data info ─────────────────────────────────────────────────────────────
    with st.expander("Data quality", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("5m bars loaded",   data.get("bars_loaded_5m", 0))
        d2.metric("Mkt-hours bars",   data.get("bars_in_market_hours", 0))
        d3.metric("Trading days",     data.get("trading_days_found", 0))
        d4.metric("Days skipped<4bars", data.get("trading_days_skipped_short", 0))
        st.caption(
            f"Range: {data.get('earliest_bar','?')[:16]} → "
            f"{data.get('latest_bar','?')[:16]}  |  tz: {data.get('timezone','')}"
        )
        warn = data.get("data_warning", "")
        if warn:
            st.warning(warn)

    # ── Full pipeline trace ───────────────────────────────────────────────────
    if steps:
        with st.expander("Full pipeline trace", expanded=expanded):
            for step in steps:
                st.code(step, language=None)


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
        st.markdown(
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
        st.success(f"Market OPEN — {now_str}")
    elif is_pre_market():
        st.info(f"PRE-MARKET — {now_str}. Live signals only generated 9:30–3:45 PM ET.")
    else:
        st.warning(f"Market CLOSED — {now_str}. Showing most recent data.")

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
            st.markdown("### Brain Status")
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
                        st.success(s)
                with brow2:
                    st.markdown("**Blocked strategies:**")
                    for s in disabled:
                        st.error(s)

            reasons = brain.get("state_reasons", [])
            if reasons:
                with st.expander("Why this market state?"):
                    for r in reasons:
                        st.caption(f"• {r}")

            st.caption(brain.get("routing_summary", ""))
            st.markdown("---")

        # ── Signal counts (raw AND post-brain) ───────────────────────────────
        raw_count = result.get("raw_signal_count", 0)
        buys  = [s for s in signals if s.get("direction") == "BUY"]
        sells = [s for s in signals if s.get("direction") in ("SELL", "SELL_SHORT")]
        holds = [s for s in signals if s.get("direction") == "HOLD"]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Raw Signals",        raw_count)
        m2.metric("Brain Accepted",     len(signals))
        m3.metric("Brain Rejected",     len(rejected))
        m4.metric("Raw BUY",            result.get("diagnostics", {}).get("signals", {}).get("raw_buy_signals", 0))
        m5.metric("Raw SELL",           result.get("diagnostics", {}).get("signals", {}).get("raw_sell_signals", 0))
        m6.metric("Regime",             _regime_badge(regime))

        # ── Intraday candlestick with signal markers ─────────────────────────
        try:
            intraday_df = fetch_intraday(symbol_input, interval="5m", period="5d")
        except Exception:
            intraday_df = None
        if intraday_df is not None and not intraday_df.empty:
            # Combine accepted + rejected signals so traders can see both on the chart.
            combined_signals = []
            for s in signals:
                combined_signals.append({
                    "timestamp": s.get("timestamp") or s.get("bar_time") or s.get("entry_time"),
                    "direction": s.get("direction"),
                    "entry_price": s.get("entry_price"),
                })
            for s in rejected:
                combined_signals.append({
                    "timestamp": s.get("timestamp") or s.get("bar_time"),
                    "direction": s.get("direction"),
                    "entry_price": s.get("entry_price"),
                })
            st.markdown("### Price Action — 5m Candles")
            st.caption(
                "Latest 5m bars for the symbol with accepted (and rejected) signal markers anchored "
                "to the entry price. Read each fill in context: where in the range, what the bar shape "
                "looked like, where VWAP sat."
            )
            fig_intraday = charts.intraday_candles(
                intraday_df, signals=combined_signals,
                title=f"{symbol_input} — 5m", height=520,
            )
            st.plotly_chart(fig_intraday, use_container_width=True, theme=None)

        # ── Pipeline diagnostics panel ────────────────────────────────────────
        diag_data = result.get("diagnostics", {})
        if diag_data:
            st.markdown("---")
            _render_pipeline_diagnostics(diag_data)
            st.markdown("---")

        if not signals and raw_count == 0:
            st.info(
                "**No raw candidate setups found in current bars.** "
                "The strategies found no qualifying conditions. "
                "See pipeline diagnostics above for details."
            )
        elif not signals and raw_count > 0:
            st.warning(
                f"**{raw_count} raw signals found but all rejected by brain filters.** "
                "See pipeline diagnostics above for the breakdown."
            )
        else:
            st.markdown("### Accepted Signals")
            for i, sig in enumerate(signals):
                _render_signal_card(sig, i)
                brain_reason = sig.get("brain_reason", "")
                brain_size = sig.get("brain_size_multiplier", 1.0)
                if brain_reason:
                    st.caption(f"Brain: {brain_reason}  |  Size: {brain_size:.0%}")

        if rejected:
            with st.expander(f"{len(rejected)} signal(s) rejected by brain"):
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
            st.markdown(f"#### {bt_symbol} — Symbol Profile")
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
                st.success(f"**Recommended:** {' · '.join(best)}")
            if avoid:
                st.error(f"**Not recommended:** {' · '.join(avoid)}")

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

        # ── Quick signal counts — always visible, no scrolling needed ───────
        diag_data = bt_result.get("diagnostics", {})
        if diag_data:
            _dsig  = diag_data.get("signals", {})
            _dbr   = diag_data.get("brain_filter", {})
            _dex   = diag_data.get("execution", {})
            _dregm = diag_data.get("regime", {})
            _raw   = _dsig.get("raw_signals_generated", 0)
            _acc   = _dbr.get("accepted_signals", _raw)
            _exec  = _dex.get("trades_opened", bt_result.get("metrics", {}).get("total_trades", 0))
            _rej   = _dbr.get("rejected_by_brain_total", 0)
            _rskip = _dregm.get("days_skipped_by_regime", 0)
            qs1, qs2, qs3, qs4, qs5 = st.columns(5)
            qs1.metric("Raw Signals",    _raw,
                       delta=f"Buy:{_dsig.get('raw_buy_signals',0)} Sell:{_dsig.get('raw_sell_signals',0)}",
                       delta_color="off")
            qs2.metric("Brain Accepted", _acc,
                       delta=f"-{_rej} rejected" if _rej else "backtest mode (no live brain)",
                       delta_color="inverse" if _rej > 0 else "off")
            qs3.metric("Trades Executed", _exec)
            qs4.metric("Regime-Blocked Days", _rskip)
            _root = diag_data.get("diagnosis", {}).get("root_cause", "")
            qs5.metric("Status", "✅ OK" if _exec > 0 else ("⚠️ Signals, no trades" if _raw > 0 else "❌ No signals"))
            if _root and _exec == 0:
                _fn = st.error if _raw == 0 else st.warning
                _fn(f"**Why no trades:** {_root}")
            # Explain regime-blocked rejections clearly
            _rej_regime = _dbr.get("rejected_by_regime", 0)
            if _rej_regime > 0 and _rej_regime == _rej:
                st.info(
                    f"**All {_rej} brain rejections were regime-blocked.** "
                    "The brain classified each historical day's market state (TREND_UP/TREND_DOWN/CHOPPY) "
                    "and the selected strategy is not allowed in those states. "
                    "Try a strategy that works in CHOPPY markets (e.g. VWAPMeanReversion or BollingerMomentum), "
                    "or run in Raw Strategy mode to see unfiltered results."
                )

        # ── Full pipeline diagnostics panel ──────────────────────────────────
        if diag_data:
            with st.expander("📊 Pipeline Diagnostics", expanded=bt_result.get("metrics", {}).get("total_trades", 0) == 0):
                _render_pipeline_diagnostics(diag_data, expanded=False)

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

                with st.expander("Why no trades fired? (click to diagnose)", expanded=True):
                    st.markdown(exp.get("why_no_trades", ""))

                    st.markdown("#### Recommendations")
                    for rec in exp.get("recommendations", []):
                        st.markdown(f"• {rec}")

                    if alt:
                        st.markdown(f"#### Recommended alternative: **{alt}**")
                        alt_period = exp.get("alternative_period", "90d")
                        if st.button(
                            f"Try {alt} on {bt_symbol} ({alt_period})",
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
                # Fallback when no TradeExplainer output (brain-filtered path)
                diag_root = diag_data.get("diagnosis", {}).get("root_cause", "") if diag_data else ""
                if diag_root:
                    st.warning(f"**0 trades generated.** {diag_root}")
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
            st.markdown("### Trade Analyzer")

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
            st.markdown("### Brain Filter — Side-by-Side Comparison")
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
                    brain_diag_data = brain_result.get("diagnostics", {})
                    if brain_diag_data:
                        _render_pipeline_diagnostics(brain_diag_data, expanded=True)
                    else:
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
        with st.spinner(f"Running all {len(ALL_STRATEGIES)} strategies on {ca_symbol}…"):
            results = _cached_backtest_all(ca_symbol, ca_period, float(ca_capital))

        if results:
            df_compare = pd.DataFrame(results)
            df_compare.insert(0, "Rank", range(1, len(df_compare) + 1))

            def _status(row):
                if row.get("trades", 0) == 0:
                    return "⚪ No trades"
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

            _show_cols = ["Rank", "Strategy", "Trades", "Win%", "Profit Factor",
                          "Total P&L ($)", "Avg P&L%", "Sharpe", "Max DD%",
                          "Best Hour", "Best DoW", "Status"]
            _show_cols = [c for c in _show_cols if c in df_compare.columns]
            st.dataframe(df_compare[_show_cols], use_container_width=True, hide_index=True)

            # ── Why did strategies produce zero trades? ──────────────────────
            zero_rows = [r for r in results if r.get("trades", 0) == 0]
            if zero_rows:
                with st.expander(
                    f"⚪ {len(zero_rows)} strateg{'y' if len(zero_rows)==1 else 'ies'} "
                    "produced 0 trades — click to see why", expanded=False
                ):
                    for r in zero_rows:
                        st.markdown(f"**{r['strategy']}**")
                        expl = r.get("explanation", {})
                        if expl:
                            primary = expl.get("primary_reason", "")
                            detail  = expl.get("detail", "")
                            diag    = expl.get("diagnostics_summary", "")
                            recalt  = r.get("recommended_alternative", "")
                            if primary:
                                st.info(f"🔎 {primary}")
                            if detail:
                                st.caption(detail)
                            if diag:
                                st.caption(f"Pipeline: {diag}")
                            if recalt:
                                st.caption(f"💡 Better fit for this symbol: **{recalt}**")
                        else:
                            diag = r.get("diagnostics", {})
                            days = diag.get("trading_days_found", "?")
                            skipped = diag.get("days_skipped_by_regime", 0)
                            st.caption(
                                f"Trading days in period: {days} — "
                                f"{skipped} skipped by regime. "
                                "The strategy's entry conditions were not met on any day "
                                f"in the {ca_period} window."
                            )
                        st.divider()
        else:
            st.warning(
                f"No results for **{ca_symbol}** over **{ca_period}**. "
                "Most often this means no intraday bars were returned — "
                "try a shorter period, or for NSE symbols re-login at /upstox/login."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist Analyzer tab (inside tab_research)
# ─────────────────────────────────────────────────────────────────────────────
with tab_watchlist:
    st.markdown("### 🔍 Webull Watchlist Analyzer")
    st.caption(
        "Paste the symbols you see on Webull (top gainers, most active, unusual volume, etc.) "
        "and get an instant verdict on whether historical strategy backtests support trading them. "
        "**TRADE** = at least one strategy is statistically strong. "
        "**WATCH** = marginal edge, needs live confirmation. "
        "**SKIP** = no historical edge found in the selected period."
    )

    wl_col1, wl_col2, wl_col3 = st.columns([2, 1, 1])
    wl_symbols_raw = wl_col1.text_area(
        "Symbols (one per line or comma-separated)",
        placeholder="AAPL\nNVDA\nTSLA\nor: AAPL, NVDA, TSLA",
        height=140,
        key="wl_symbols",
    )
    wl_period = wl_col2.selectbox(
        "Backtest period",
        ["30d", "60d", "90d"],
        index=1,
        key="wl_period",
        help="Longer = more trades sampled = more reliable verdict. 60d is a good default.",
    )
    wl_capital = wl_col3.number_input(
        "Capital ($)", value=10_000, step=1_000, key="wl_capital"
    )

    st.caption(
        "⏱ Each symbol runs all 6 strategies × the full period. "
        "Allow ~5–15 seconds per symbol. Paste up to 10 symbols at once."
    )

    if st.button("▶ Analyze Watchlist", key="run_watchlist_analyze", type="primary"):
        # Parse symbols from textarea
        raw_text = wl_symbols_raw.strip()
        if not raw_text:
            st.warning("Paste at least one symbol.")
        else:
            # Support comma-separated or newline-separated
            import re as _re
            sym_list = [s.strip().upper() for s in _re.split(r"[\n,]+", raw_text) if s.strip()]
            sym_list = list(dict.fromkeys(sym_list))[:15]  # dedup, cap at 15

            with st.spinner(f"Analyzing {len(sym_list)} symbol(s)… this may take 10–30s"):
                try:
                    wl_resp = api._post(
                        "/daytrading/watchlist-analyze",
                        json={
                            "symbols": sym_list,
                            "period": wl_period,
                            "initial_capital": float(wl_capital),
                        },
                        timeout=120,  # allow up to 2 min for 10 symbols × 6 strategies
                    )
                    wl_results = wl_resp if isinstance(wl_resp, list) else []
                except Exception as e:
                    st.error(f"API error: {e}")
                    wl_results = []

            if not wl_results:
                st.warning("No results returned. Check that the symbols are valid US tickers.")
            else:
                # ── Summary metrics row ──────────────────────────────────────
                n_trade = sum(1 for r in wl_results if r["verdict"] == "TRADE")
                n_watch = sum(1 for r in wl_results if r["verdict"] == "WATCH")
                n_skip  = sum(1 for r in wl_results if r["verdict"] == "SKIP")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Symbols analyzed", len(wl_results))
                mc2.metric("✅ TRADE", n_trade)
                mc3.metric("🟡 WATCH", n_watch)
                mc4.metric("🔴 SKIP", n_skip)

                st.divider()

                # ── Per-symbol cards ─────────────────────────────────────────
                for row in wl_results:
                    v = row["verdict"]
                    color_icon = {"TRADE": "✅", "WATCH": "🟡", "SKIP": "🔴"}.get(v, "⚪")
                    with st.container(border=True):
                        h_col, v_col = st.columns([4, 1])
                        h_col.markdown(
                            f"**{row['symbol']}** &nbsp; {color_icon} **{v}**"
                        )
                        v_col.caption(f"Score: {row['score']:.2f}")

                        st.caption(row["reason"])

                        if row["best_strategy"] != "—":
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Best strategy", row["best_strategy"])
                            m2.metric("Trades", row["best_strategy_trades"])
                            m3.metric("Win %", f"{row['best_strategy_win_pct']:.0f}%")
                            m4.metric("Profit factor", f"{row['best_strategy_profit_factor']:.2f}")

                        if row.get("diagnostics_summary"):
                            st.caption(f"🔧 {row['diagnostics_summary']}")

                        if row["all_weak"]:
                            st.info(
                                f"No historical edge on **{row['symbol']}** in the last {wl_period}. "
                                "This doesn't mean the stock won't move — it means the strategy "
                                "rules didn't produce tradeable setups in that window. "
                                "Check live signals manually if it's gapping or on a mover list."
                            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Scanner
# ─────────────────────────────────────────────────────────────────────────────
with tab_scanner:
    import api as _api

    st.markdown("### Market Scanner — Ranked Day-Trade Candidates")
    st.caption(
        "Scans the full US-listed common-stock universe (~5,800 names from NASDAQ "
        "Trader, refreshed daily) and ranks tickers by volume, volatility, relative "
        "volume, gap size, and catalyst presence. Use the filters to focus on the "
        "price and float range you actually want to trade."
    )

    # ── Presets: pre-fill the filters with sensible day-trade profiles ───────
    # Float values are in MILLIONS, volume in shares, price in dollars.
    # universe_cap=500 on presets keeps scans snappy (~30-60s) while still
    # covering enough of the universe to find good candidates.
    SCANNER_PRESETS = {
        "Custom": None,
        "Low-float runners ($2-$20, 5-50M float)": {
            "min_price": 2.0, "max_price": 20.0,
            "min_vol": 500_000,
            "min_float_m": 5.0, "max_float_m": 50.0,
            "universe_cap": 0,
            "desc": "Small floats with room to move. Best for momentum/gap plays in HIGH_VOL regime.",
        },
        "Mid-cap movers ($20-$100, 50-500M float)": {
            "min_price": 20.0, "max_price": 100.0,
            "min_vol": 1_000_000,
            "min_float_m": 50.0, "max_float_m": 500.0,
            "universe_cap": 0,
            "desc": "Liquid mid-caps that still move meaningfully. Good for ORB and VWAP strategies.",
        },
        "Large-cap day trades ($50+, 1M+ vol)": {
            "min_price": 50.0, "max_price": 0.0,
            "min_vol": 5_000_000,
            "min_float_m": 0.0, "max_float_m": 0.0,
            "universe_cap": 0,
            "desc": "Mega-caps with deep liquidity. Tight spreads, smaller % moves, safest for size.",
        },
        "Pre-market gappers (any price, 500k+ vol)": {
            "min_price": 1.0, "max_price": 0.0,
            "min_vol": 500_000,
            "min_float_m": 0.0, "max_float_m": 0.0,
            "universe_cap": 0,
            "desc": "Wide net — let the gap/catalyst scoring do the filtering. Use in pre-market only.",
        },
    }

    mc1, mc2, mc3 = st.columns([1, 1, 2])
    ms_max = mc1.number_input("Top N", min_value=5, max_value=50, value=20, step=5, key="ms_max")
    ms_state = mc2.selectbox(
        "Force market state",
        ["auto", "TREND_UP", "TREND_DOWN", "CHOPPY", "HIGH_VOL", "NEWS_RISK"],
        index=0, key="ms_state",
    )
    ms_universe = mc3.text_input(
        "Override universe (optional, comma-separated)", value="", key="ms_universe",
        placeholder="leave empty to scan the full US-listed universe",
    )

    ms_preset_name = st.selectbox(
        "Filter preset",
        list(SCANNER_PRESETS.keys()),
        index=1,  # default to "Low-float runners" — most useful for the user's stated goal
        key="ms_preset",
        help="Pick a preset to auto-fill the filters below. Choose 'Custom' to set everything manually.",
    )
    _preset = SCANNER_PRESETS[ms_preset_name]
    if _preset:
        st.caption(f"_{_preset['desc']}_")

    # Streamlit widgets remember their last value via `key`. Reset session_state
    # when the preset changes so the inputs reflect the new preset values.
    # Setting state BEFORE the widget is rendered avoids the "value+key both set"
    # Streamlit warning.
    if st.session_state.get("_ms_last_preset") != ms_preset_name:
        st.session_state["_ms_last_preset"] = ms_preset_name
        if _preset:
            st.session_state["ms_min_price"] = _preset["min_price"]
            st.session_state["ms_max_price"] = _preset["max_price"]
            st.session_state["ms_min_vol"] = _preset["min_vol"]
            st.session_state["ms_min_float_m"] = _preset["min_float_m"]
            st.session_state["ms_max_float_m"] = _preset["max_float_m"]
            st.session_state["ms_universe_cap"] = _preset["universe_cap"]

    # ── Filters row 1: price + volume ────────────────────────────────────────
    # We rely on session_state for defaults (set above), so no `value=` param.
    # On first render with no session_state, fall back to Custom defaults.
    st.session_state.setdefault("ms_min_price", 5.0)
    st.session_state.setdefault("ms_max_price", 0.0)
    st.session_state.setdefault("ms_min_vol", 1_000_000)
    st.session_state.setdefault("ms_min_float_m", 0.0)
    st.session_state.setdefault("ms_max_float_m", 0.0)
    st.session_state.setdefault("ms_universe_cap", 0)

    f1, f2, f3 = st.columns(3)
    ms_min_price = f1.number_input(
        "Min price ($)", min_value=0.0, max_value=10_000.0, step=1.0, key="ms_min_price",
        help="Skip sub-$1 names (PDT/marginability issues) and pennies. Typical: $2 for runners, $20+ for mid-caps.",
    )
    ms_max_price = f2.number_input(
        "Max price ($, 0 = no cap)", min_value=0.0, max_value=10_000.0, step=10.0, key="ms_max_price",
        help="Cap to focus on a price band. Typical: $20 for low-float, $100 for mid-cap, 0 for large-cap.",
    )
    ms_min_vol = f3.number_input(
        "Min avg daily volume (shares)",
        min_value=0, max_value=500_000_000, step=100_000, key="ms_min_vol",
        help="Liquidity floor. 500k = thin but tradable, 1M = comfortable, 5M+ = institutional-grade.",
    )

    # ── Filters row 2: float ─────────────────────────────────────────────────
    # Inputs are in MILLIONS of shares to make the UX usable — entering
    # "10000000" for 10M is error-prone (an off-by-one zero is a 10x mistake).
    f4, f5, f6 = st.columns(3)
    ms_min_float_m = f4.number_input(
        "Min float (millions, 0 = no floor)",
        min_value=0.0, max_value=10_000.0, step=1.0, key="ms_min_float_m",
        help="Lower bound on shares float in millions. Low-float runners: 5–50M. Mid-caps: 50–500M.",
    )
    ms_max_float_m = f5.number_input(
        "Max float (millions, 0 = no cap)",
        min_value=0.0, max_value=10_000.0, step=10.0, key="ms_max_float_m",
        help="Upper bound on shares float. ~50 for low-float, ~500 for mid-caps, 0 for no cap.",
    )
    ms_universe_cap = f6.number_input(
        "Cap universe size (0 = all)",
        min_value=0, max_value=10_000, step=100, key="ms_universe_cap",
        help="Useful for quick test scans. 500 = fast (~30s). 0 = full ~5,800 universe (~2-3 min).",
    )

    # Convert millions to absolute share counts for the API
    ms_min_float = ms_min_float_m * 1_000_000
    ms_max_float = ms_max_float_m * 1_000_000

    # ── Native precheck controls ─────────────────────────────────────────────
    # The pre-check runs the 4 audited strategies (BollingerMomentum,
    # SupertrendTrend, EMAMomentum, ORBBreakout) on the top-K of the scanner
    # output, so rows with native_signal_active=True are "firing right now."
    pc1, pc2 = st.columns(2)
    ms_run_precheck = pc1.checkbox(
        "Run native pre-check on top-K", value=True, key="ms_run_precheck",
        help="Adds ~10–20s. Flags scanner rows whose audited strategies have an accepted signal at the current bar.",
    )
    ms_precheck_top_k = pc2.number_input(
        "Precheck top-K", min_value=0, max_value=50, value=10, step=1,
        key="ms_precheck_top_k",
        help="How many top-ranked candidates the pre-check runs on. 0 disables.",
    )

    if st.button("Run Market Scan", key="run_market_scan_btn", type="primary"):
        scan_failed = False
        with st.spinner("Scanning market — fetching bars, scoring, applying regime adjustments…"):
            try:
                ranked = _api.daytrading_scanner_watchlist(
                    max_symbols=int(ms_max),
                    universe=ms_universe.strip(),
                    market_state="" if ms_state == "auto" else ms_state,
                    min_price=float(ms_min_price),
                    max_price=float(ms_max_price) if ms_max_price > 0 else None,
                    min_avg_volume=float(ms_min_vol),
                    min_float=float(ms_min_float) if ms_min_float > 0 else None,
                    max_float=float(ms_max_float) if ms_max_float > 0 else None,
                    universe_max_symbols=int(ms_universe_cap) if ms_universe_cap > 0 else None,
                    run_native_precheck=bool(ms_run_precheck),
                    precheck_top_k=int(ms_precheck_top_k),
                )
            except Exception as e:
                ranked = []
                scan_failed = True
                msg = str(e)
                if "timed out" in msg.lower() or "timeout" in msg.lower():
                    st.error(
                        "Market scan timed out. The full ~5,800-symbol universe with "
                        "no cap can exceed the 10-min HTTP window. Try one of:\n\n"
                        "• Set **Cap universe size** to 500–1500 for a fast scan.\n"
                        "• Use an **Override universe** (comma-separated tickers) to "
                        "limit the work.\n"
                        "• Turn off **Run native pre-check on top-K** for the first "
                        "scan, then re-enable it once the list is narrowed."
                    )
                else:
                    st.error(f"Market scan failed: {e}")

        # Stash the latest scan in session_state so the Start-from-Scanner
        # button below stays useful after the next rerun.
        st.session_state["_ms_last_ranked"] = ranked

        if not ranked and not scan_failed:
            st.info("No candidates passed the hard filters (volume / price / ATR).")
        elif not ranked:
            # scan_failed branch — error already shown above; skip the
            # misleading "no candidates passed" info banner.
            pass
        else:
            rows = []
            any_precheck_ran = False
            for r in ranked:
                m = r.get("metrics", {}) or {}
                precheck_ran = bool(r.get("native_precheck_ran", False))
                any_precheck_ran = any_precheck_ran or precheck_ran
                signal_active = bool(r.get("native_signal_active", False))
                conf = r.get("best_native_confidence")
                rows.append({
                    "Symbol": r.get("symbol", ""),
                    "Signal?": "🟢" if signal_active else ("·" if precheck_ran else "—"),
                    "Native Strategy": r.get("best_native_strategy") or "—",
                    "Side": r.get("best_native_side") or "—",
                    "Native Conf": f"{conf:.2f}" if isinstance(conf, (int, float)) else "—",
                    "Score": round(float(r.get("score", 0.0)), 2),
                    "Adj Score": round(float(r.get("adjusted_score", 0.0)), 2),
                    "Bucket": r.get("recommended_strategy_bucket", "—") or "—",
                    "Tags": ", ".join(r.get("tags", []) or []),
                    "Price": f"${m.get('last_price', 0):.2f}" if m.get("last_price") else "—",
                    "ATR %": f"{m.get('atr_pct', 0):.2f}%" if m.get("atr_pct") else "—",
                    "Gap %": f"{m.get('premarket_gap_pct', 0):+.2f}%" if m.get("premarket_gap_pct") else "—",
                    "Pre-Mkt Rel Vol": f"{m.get('premarket_rel_vol', 0):.2f}x" if m.get("premarket_rel_vol") else "—",
                    "Avg Vol 30d": f"{int(m.get('avg_daily_volume_30d', 0)):,}" if m.get("avg_daily_volume_30d") else "—",
                    "Float": f"{int(m.get('shares_float', 0))/1e6:.1f}M" if m.get("shares_float") else "—",
                    "Catalyst": "yes" if m.get("has_catalyst") else "",
                })
            df_market = pd.DataFrame(rows)

            # Drop the precheck columns when nothing was actually checked —
            # otherwise the four extra "—" columns just clutter the table.
            if not any_precheck_ran:
                df_market = df_market.drop(
                    columns=["Signal?", "Native Strategy", "Side", "Native Conf"],
                    errors="ignore",
                )

            def _bucket_color(row):
                bucket = (row.get("Bucket") or "").lower()
                color = {
                    "gap": "rgba(255,152,0,0.18)",
                    "orb": "rgba(38,198,218,0.18)",
                    "momentum": "rgba(0,212,170,0.18)",
                    "vwap": "rgba(171,71,188,0.18)",
                }.get(bucket, "rgba(128,128,128,0.08)")
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df_market.style.apply(_bucket_color, axis=1),
                use_container_width=True, hide_index=True,
            )
            n_active = sum(1 for r in ranked if r.get("native_signal_active"))
            st.caption(
                f"{len(df_market)} candidate(s)"
                + (f" · {n_active} with active native signal" if any_precheck_ran else "")
                + " · Click a symbol from above and paste it into the per-symbol "
                  "scanner below to see live strategy signals."
            )

    # ── Start AutoTrader from scanner ────────────────────────────────────────
    # Reuses the most-recent scan stashed in session_state. The backend
    # re-runs the scanner internally (the cached UI list is just so the
    # button stays visible across reruns) and arms the autotrader with the
    # same selection policy.
    last_ranked = st.session_state.get("_ms_last_ranked") or []
    st.markdown("---")
    st.markdown("#### Arm AutoTrader from these candidates")
    sb1, sb2, sb3, sb4 = st.columns(4)
    sfs_max = sb1.number_input(
        "Symbols to arm", min_value=1, max_value=20, value=5, step=1, key="sfs_max",
    )
    sfs_dir = sb2.selectbox(
        "Direction", ["long_only", "short_only", "both"], index=0, key="sfs_dir",
    )
    sfs_require_signal = sb3.checkbox(
        "Require active native signal", value=True, key="sfs_require_signal",
        help="When on, only arm symbols whose pre-check found a firing audited strategy.",
    )
    sfs_min_conf = sb4.number_input(
        "Min native confidence (0 = off)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="sfs_min_conf",
    )
    sfs_force = st.checkbox(
        "Force outside RTH", value=False, key="sfs_force",
        help="Arm even if the market is closed (paper trader will idle until 9:30 ET).",
    )

    if st.button(
        "🚀 Start AutoTrader from Scanner",
        key="start_from_scanner_btn",
        disabled=not last_ranked,
        help="Runs the scanner again on the backend, then arms the autotrader with the top picks.",
    ):
        body = {
            "max_symbols": int(sfs_max),
            "direction_mode": sfs_dir,
            "broker_name": "paper",
            "entry_mode": "native_strategy",
            "require_native_signal": bool(sfs_require_signal),
            "prefer_native_signal": True,
            "force": bool(sfs_force),
        }
        if sfs_min_conf > 0:
            body["min_best_native_confidence"] = float(sfs_min_conf)
        with st.spinner("Re-running scanner on backend and arming autotrader…"):
            try:
                result = _api.autotrader_start_from_scanner(body)
                st.success(
                    f"Armed: {', '.join(result.get('selected_symbols', [])) or '—'} "
                    f"(market_state={result.get('market_state', '—')})"
                )
                with st.expander("Scanner summary"):
                    st.dataframe(
                        pd.DataFrame(result.get("scanner_summary", [])),
                        use_container_width=True, hide_index=True,
                    )
                with st.expander("Raw autotrader response"):
                    st.json(result.get("autotrader", {}))
            except Exception as e:
                st.error(f"Start from scanner failed: {e}")

    if not last_ranked:
        st.caption("Run a market scan first to enable this button.")

    st.markdown("---")
    st.markdown("### Per-Symbol Strategy Scanner")
    st.caption("Run all 5 day-trade strategies against a chosen list of symbols.")

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


# ─────────────────────────────────────────────────────────────────────────────
# TAB — 🎯 Single Stock Auto Trader
# ─────────────────────────────────────────────────────────────────────────────
with tab_autotrader:
    st.markdown("### Single Stock Auto Trader")
    st.caption(
        "Pick one symbol, configure risk, and let the bot manage the full intraday trade "
        "from entry to exit — stops, trailing, partial TP, and EOD flatten included."
    )

    # ── Session-state init ────────────────────────────────────────────────────
    if "at_trader" not in st.session_state:
        st.session_state["at_trader"] = None
    if "at_status" not in st.session_state:
        st.session_state["at_status"] = {}

    # ── Config panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Trader Configuration", expanded=True):
        ac1, ac2, ac3 = st.columns(3)
        at_symbol = ac1.text_input("Symbol", value="AAPL", key="at_symbol").strip().upper()
        at_mode   = ac2.radio("Trade Mode", ["Paper", "Live-ready"], horizontal=True, key="at_mode")
        at_dir    = ac3.radio(
            "Direction", ["Long only", "Short only", "Both"],
            horizontal=True, key="at_dir"
        )

        bc1, bc2, bc3, bc4 = st.columns(4)
        at_risk_pct    = bc1.number_input("Risk per trade %", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="at_risk")
        at_max_loss    = bc2.number_input("Max daily loss %", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="at_maxloss")
        at_capital     = bc3.number_input("Capital ($)", min_value=1000, value=10000, step=1000, key="at_capital")
        at_max_trades  = bc4.number_input("Max trades/day", min_value=1, max_value=20, value=6, step=1, key="at_maxtrades")

        cc1, cc2 = st.columns(2)
        at_partial_tp   = cc1.toggle("Partial take-profit at +1R", value=True, key="at_partial")
        at_trail_mode   = cc2.selectbox(
            "Trailing stop mode",
            ["atr", "ema", "candle"],
            key="at_trail",
        )

    # ── Control buttons ───────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    start_pressed    = btn_col1.button("▶ Start Bot", use_container_width=True, type="primary", key="at_start")
    stop_pressed     = btn_col2.button("⏹ Stop Bot", use_container_width=True, key="at_stop")
    flatten_pressed  = btn_col3.button("🚨 Force Flatten", use_container_width=True, key="at_flatten")

    if start_pressed:
        try:
            from app.services.strategy.daytrading.autotrader import SingleStockTrader
            from app.services.strategy.daytrading.autotrader.single_stock_trader import PolicyError
            from app.services.strategy.daytrading.brain.symbol_policy import get_policy
            dir_map = {"Long only": "long_only", "Short only": "short_only", "Both": "both"}
            trader = SingleStockTrader(
                symbol=at_symbol,
                broker=None,   # paper mode — no real broker needed
                direction_mode=dir_map[at_dir],
                trail_mode=at_trail_mode,
                partial_tp=at_partial_tp,
                risk_per_trade_pct=at_risk_pct / 100,
                max_daily_loss_pct=at_max_loss,
                max_trades_per_day=int(at_max_trades),
                initial_capital=float(at_capital),
                on_trade_update=lambda s: st.session_state.update({"at_status": s}),
            )
            old = st.session_state.get("at_trader")
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            st.session_state["at_trader"] = trader
            trader.start()
            st.success(f"Auto Trader started for {at_symbol} ({at_mode} mode)")
        except PolicyError as e:
            pol = get_policy(at_symbol)
            st.error(f"**Deployment policy blocked auto-trade for {at_symbol}**")
            st.warning(
                f"**Status:** {pol.badge()}  \n"
                f"**Reason:** {pol.reason}  \n"
                f"**Walk-forward verdict:** {pol.wf_verdict} (score {pol.wf_score:.0f}/100)  \n\n"
                f"To override: enable *Manual Override* in the Symbol Policy panel below."
            )
        except Exception as e:
            st.error(f"Failed to start trader: {e}")

    if stop_pressed:
        trader_obj = st.session_state.get("at_trader")
        if trader_obj:
            trader_obj.stop()
            st.info("Auto Trader stopped.")
        else:
            st.warning("No active trader to stop.")

    if flatten_pressed:
        trader_obj = st.session_state.get("at_trader")
        if trader_obj:
            trader_obj.force_flatten("Manual force flatten from UI")
            st.warning("Force flatten executed.")
        else:
            st.warning("No active trader.")

    # ── Refresh status ────────────────────────────────────────────────────────
    trader_obj = st.session_state.get("at_trader")
    if trader_obj is not None:
        try:
            st.session_state["at_status"] = trader_obj.get_status()
        except Exception:
            pass

    status = st.session_state.get("at_status", {})
    st.divider()

    # ── Status panels ─────────────────────────────────────────────────────────
    if not status:
        st.info("Start the bot to see live status.")
    else:
        # ── Summary row ────────────────────────────────────────────────────────
        s1, s2, s3, s4, s5 = st.columns(5)
        _state_emoji = {
            "FLAT": "⬜", "LONG": "🟢", "SHORT": "🔴",
            "PARTIAL_EXIT_TAKEN": "🟡", "TRAILING": "🔵",
            "EXITED": "✅", "BLOCKED": "🚫", "PENDING_ENTRY": "⏳",
        }
        state_val = status.get("state", "UNKNOWN")
        s1.metric("Status", f"{_state_emoji.get(state_val, '❓')} {state_val}")
        s2.metric("Symbol", status.get("symbol", "—"))
        s3.metric("Market Regime", status.get("market_state", "—"))
        s4.metric("Unrealized P&L", f"${status.get('unrealized_pnl', 0):+,.2f}")
        s5.metric("Realized P&L", f"${status.get('realized_pnl', 0):+,.2f}")

        # ── Management profile (always visible) ───────────────────────────────
        mgmt_profile = status.get("management_profile", "")
        if mgmt_profile:
            st.caption(f"**Management profile:** {mgmt_profile}")

        # ── Active trade panel ─────────────────────────────────────────────────
        if state_val in ("LONG", "SHORT", "PARTIAL_EXIT_TAKEN", "TRAILING"):
            st.markdown("#### Active Position")
            tp1, tp2, tp3, tp4, tp5, tp6, tp7 = st.columns(7)
            tp1.metric("Side",         status.get("side", "—"))
            tp2.metric("Entry",        f"${status.get('entry_price', 0):.2f}")
            tp3.metric("Stop",         f"${status.get('current_stop', 0):.2f}")
            tp4.metric("Target",       f"${status.get('first_target', 0):.2f}")
            r_val = status.get("r_multiple")
            tp5.metric("R Multiple",   f"{r_val:+.2f}R" if r_val is not None else "—")
            tp6.metric("Trail Mode",   status.get("active_trail_mode", "—"))
            tp7.metric("Strategy",     status.get("strategy", "—"))

        elif state_val == "FLAT":
            cooldown_left = status.get("cooldown_bars_remaining", 0)
            if cooldown_left > 0:
                st.info(f"**Cooldown:** {cooldown_left} bar(s) remaining before next entry")
            else:
                no_trade_reason = status.get("last_no_trade_reason", "")
                if no_trade_reason:
                    st.info(f"**Why no trade?** {no_trade_reason}")

        elif state_val == "BLOCKED":
            st.error(f"**Trading BLOCKED:** {status.get('block_reason', 'Risk limit hit')}")

        # ── Session stats ──────────────────────────────────────────────────────
        st.markdown("#### Session Summary")
        ss1, ss2, ss3, ss4 = st.columns(4)
        ss1.metric("Trades today",     status.get("trades_today", 0))
        ss2.metric("Consec. losses",   status.get("consecutive_losses", 0))
        hb_age = status.get("heartbeat_age_s", 0)
        hb_label = f"{hb_age:.0f}s ago" if hb_age < 120 else f"⚠️ {hb_age:.0f}s ago"
        ss3.metric("Last heartbeat",   hb_label)
        ss4.metric("Running",          "✅ Yes" if status.get("running") else "⏸ No")

        # ── Closed trades table ────────────────────────────────────────────────
        session_trades = status.get("session_trades", [])
        if session_trades:
            st.markdown("#### Today's Closed Trades")
            rows = []
            for t in session_trades:
                rows.append({
                    "Time": t.get("entry_time", "")[:19],
                    "Side": t.get("side", ""),
                    "Entry": t.get("entry_price", 0),
                    "Exit": t.get("exit_price", 0),
                    "Qty": t.get("qty", 0),
                    "P&L ($)": t.get("pnl", 0),
                    "P&L %": t.get("pnl_pct", 0),
                    "Exit Reason": t.get("exit_reason", ""),
                    "Strategy": t.get("strategy", ""),
                })
            trades_df = pd.DataFrame(rows)
            st.dataframe(
                # Styler.applymap was removed in pandas 2.1+; .map is the
                # drop-in replacement with the same signature.
                trades_df.style.map(
                    lambda v: "color: green" if isinstance(v, (int, float)) and v > 0 else
                              "color: red"   if isinstance(v, (int, float)) and v < 0 else "",
                    subset=["P&L ($)", "P&L %"],
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ── Decision log ──────────────────────────────────────────────────────
        log = status.get("decision_log", [])
        if log:
            with st.expander(f"📋 Decision Log ({len(log)} entries)"):
                for entry in reversed(log[-30:]):
                    icon = {
                        "ENTRY LONG": "🟢", "ENTRY SHORT": "🔴",
                        "EXIT": "✅", "EXIT LONG": "✅", "EXIT SHORT": "✅",
                        "MOVE_STOP": "🔧", "PARTIAL_EXIT": "📤",
                        "NO_TRADE": "⬜", "BLOCKED": "🚫",
                        "TRAIL_ACTIVATED": "🔵", "RESET": "🔄",
                    }.get(entry.get("event", ""), "ℹ️")
                    st.markdown(
                        f"`{entry['time']}` {icon} **{entry['event']}** — {entry['reason']}"
                    )

    # ── Auto-refresh while bot is running ─────────────────────────────────────
    if status.get("running"):
        import time as _t
        _t.sleep(0.1)
        st.rerun()


# ── Symbol Policy tab ─────────────────────────────────────────────────────────
with tab_policy:
    from app.services.strategy.daytrading.brain.symbol_policy import (
        all_policies, get_policy, set_override,
        ENABLED, MONITOR_ONLY, REGIME_DEPENDENT, DISABLED,
        SymbolPolicy, set_policy,
    )

    st.markdown("### Symbol Deployment Policy")
    st.caption(
        "Controls which symbols are cleared for live auto-trading. "
        "Policies are derived from walk-forward validation results. "
        "Changes here take effect immediately but reset on server restart."
    )

    # ── Summary table ──────────────────────────────────────────────────────────
    policies = all_policies()
    if policies:
        rows = []
        for p in policies:
            rows.append({
                "Symbol":    p["symbol"],
                "Status":    p["badge"],
                "WF Verdict": p["wf_verdict"].title(),
                "WF Score":  f"{p['wf_score']:.0f}/100",
                "Override":  "YES" if p["override_live"] else "—",
                "Scan":      "✅" if p["allows_scan"] else "🚫",
                "Reason":    p["reason"],
            })
        policy_df = pd.DataFrame(rows)
        st.dataframe(
            policy_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Reason": st.column_config.TextColumn(width="large"),
                "Status": st.column_config.TextColumn(width="small"),
            },
        )

    st.divider()

    # ── Per-symbol detail + override ──────────────────────────────────────────
    st.markdown("#### Adjust a Symbol")
    pol_col1, pol_col2 = st.columns([1, 2])

    with pol_col1:
        known_syms = [p["symbol"] for p in policies]
        extra_sym  = st.text_input("Symbol (add or edit)", value="", key="pol_sym_input").upper().strip()
        edit_sym   = extra_sym if extra_sym else (known_syms[0] if known_syms else "AAPL")

    pol = get_policy(edit_sym)

    with pol_col2:
        st.markdown(f"**Current policy for {edit_sym}:** {pol.badge()}")
        st.caption(pol.reason)
        if pol.wf_verdict:
            st.caption(f"Walk-forward: **{pol.wf_verdict}** · score {pol.wf_score:.0f}/100")

    st.markdown("")
    ov_col, st_col = st.columns(2)
    with ov_col:
        new_override = st.checkbox(
            "Manual live override (bypasses policy for this symbol)",
            value=pol.override_live,
            key=f"pol_override_{edit_sym}",
            help="When checked, this symbol is allowed to auto-trade regardless of its status.",
        )

    with st_col:
        status_options = [ENABLED, MONITOR_ONLY, REGIME_DEPENDENT, DISABLED]
        status_labels  = {
            ENABLED:          "✅ Enabled — live auto-trade allowed",
            MONITOR_ONLY:     "👁 Monitor only — scans yes, auto-trade no",
            REGIME_DEPENDENT: "🔀 Regime dependent — live only in allowed regimes",
            DISABLED:         "🚫 Disabled — excluded from all live flows",
        }
        new_status = st.selectbox(
            "Deployment status",
            options=status_options,
            index=status_options.index(pol.status) if pol.status in status_options else 1,
            format_func=lambda s: status_labels[s],
            key=f"pol_status_{edit_sym}",
        )

    new_reason = st.text_area(
        "Reason / notes",
        value=pol.reason,
        height=80,
        key=f"pol_reason_{edit_sym}",
    )

    if st.button("💾 Save policy change", key="pol_save"):
        updated = SymbolPolicy(
            symbol=edit_sym,
            status=new_status,
            reason=new_reason,
            wf_verdict=pol.wf_verdict,
            wf_score=pol.wf_score,
            override_live=new_override,
            allowed_regimes=pol.allowed_regimes,
        )
        set_policy(edit_sym, updated)
        st.success(f"Policy for {edit_sym} updated to: {updated.badge()}")
        st.rerun()

    # ── Live signals policy note ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### How policies are enforced")
    st.markdown("""
| Flow | Gate | Effect when blocked |
|------|------|---------------------|
| **Live Signals** (`run_signals`) | `allows_live(symbol, regime)` | Returns empty signal list with `policy_blocked=True` and reason |
| **Compare All** (`run_backtest_all`) | `allows_scan(symbol)` | Returns empty results list |
| **Auto Trader** (`SingleStockTrader.start`) | `allows_live(symbol)` | Raises `PolicyError` — UI shows reason and walk-forward verdict |
| **Backtest** (single strategy) | Not gated — research always allowed | Backtest runs regardless of live policy |
""")
    st.caption(
        "Policy changes are in-memory only and reset on server restart. "
        "To make permanent changes, edit `brain/symbol_policy.py` directly."
    )
