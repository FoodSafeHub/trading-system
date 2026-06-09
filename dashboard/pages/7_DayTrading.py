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
    IST,
    is_market_open,
    is_pre_market,
    market_status,
    market_session,
    get_spy_regime,
    compute_vwap,
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
import _lightweight_chart as lwc  # noqa: E402
import api  # noqa: E402
from _broker_routing import render_broker_routing_toggle  # noqa: E402
from _components import (  # noqa: E402
    page_header, stat_band, filter_cols, empty_state,
    regime_chip, eligibility_chip, bot_state_chip, blocker_chip, blocker_label,
)

apply_theme("Day Trading")

# ── Page header + data-source status ─────────────────────────────────────────
_ds_label, _ds_color = "unknown", "grey"
_ds_detail = ""
try:
    _ds = api.daytrading_data_source_status() or {}
    _ds_status = _ds.get("status", "idle")
    _ds_label  = _ds.get("label", "unknown")
    _ctr       = _ds.get("counters", {}) or {}
    _ds_color  = {"ok": "green", "degraded": "amber", "idle": "grey"}.get(_ds_status, "grey")
    _ds_detail = (
        f"TD: {_ctr.get('twelvedata', 0)}  "
        f"Webull: {_ctr.get('webull', 0)}  "
        f"yfinance: {_ctr.get('yfinance', 0)}"
    )
except Exception:
    pass

page_header(
    "Day Trading",
    subtitle=(
        f"{len(ALL_STRATEGIES)} intraday strategies · 5m and 15m bars · "
        "All signals expire at market close · No overnight holds"
    ),
)

# Compact stat band: market session clocks + data provider health
_mkt_status_bits = []
from app.services.strategy.daytrading.market_open import now_et as _now_et
_now_str = _now_et().strftime("%H:%M ET")
_mkt_bits: list[tuple[str,str,str]] = [
    ("Data", _ds_label, _ds_color),
]
if _ds_detail:
    _mkt_bits.append(("Providers", _ds_detail, "grey"))
from _theme import pill as _pill
_mkt_html = "".join(
    f"<span class='tx-stat'><span class='tx-stat-label'>{l}</span>{_pill(v, c)}</span>"
    for l, v, c in _mkt_bits
)
st.markdown(
    f"<div class='tx-statband' style='margin-top:var(--sp-2)'>{_mkt_html}"
    f"<span style='margin-left:auto;font-size:0.75rem;color:var(--text-3)'>{_now_str}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

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
    tab_backtest, tab_compare, tab_sizer, tab_watchlist, tab_wf = st.tabs([
        "Backtest", "Compare All", "Position Sizer", "🔍 Watchlist Analyzer", "📈 Walk-Forward",
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


def _render_signal_card(sig: dict, idx: int, symbol: str = "") -> None:
    direction = sig.get("direction", "HOLD")
    confidence = sig.get("confidence", 0.0)
    rr = sig.get("r_multiple", 0)
    label = (
        f"**{sig['strategy']}** · {direction} · "
        f"conf {confidence:.0%} · R:R {rr:.1f} · {sig.get('timeframe', '')} · "
        f"{_regime_badge(sig.get('regime', ''))}"
    )
    with st.expander(label, expanded=idx == 0):
        c1, c2, c3, c4 = st.columns(4)
        entry = sig.get("entry_price", 0)
        stop  = sig.get("stop_price", 0)
        tgt   = sig.get("target_price", 0)
        c1.metric("Entry",  f"${entry:.2f}")
        c2.metric("Stop",   f"${stop:.2f}",  delta=f"-{abs(entry-stop):.2f}" if entry else None, delta_color="inverse")
        c3.metric("Target", f"${tgt:.2f}",   delta=f"+{abs(tgt-entry):.2f}"  if entry else None, delta_color="normal")
        c4.metric("R:R",    f"{rr:.1f}:1")

        st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.0%}")
        st.caption(f"Reason: {sig.get('reason', '')}")

        # Market-aware expiry label
        _sym = symbol or sig.get("symbol", "")
        if _sym:
            try:
                _sess = market_session(_sym)
                _close_str = _sess.close_time.strftime("%H:%M")
                _tz_str = "IST" if _sess.tz is IST else "ET"
                st.info(f"⏱ **DAY ORDER — expires at close ({_close_str} {_tz_str})**")
            except Exception:
                st.info("⏱ **DAY ORDER — expires at close (3:45 PM ET)**")
        else:
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
    _hold_bars = m.get("avg_hold_bars", 0)
    _hold_min  = _hold_bars * 5   # 5m bars → minutes
    cols[7].metric("Avg Hold", f"{_hold_bars:.1f} bars", delta=f"≈{_hold_min:.0f} min", delta_color="off")

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
# TAB 1 — Live Signals  (day-trading workstation)
# ─────────────────────────────────────────────────────────────────────────────
with tab_signals:
    # ── Top control bar ───────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([3, 1, 1, 1, 1])
    symbol_input = ctrl1.text_input(
        "Symbol", value="SPY", key="signals_symbol",
        placeholder="AAPL, MTEN, SPY …",
    ).upper().strip() or "SPY"
    tf_pick      = ctrl2.selectbox("Timeframe", ["5m", "1m", "15m"], index=0, key="signals_tf")
    auto_refresh = ctrl3.toggle("Live", value=False, key="signals_auto_refresh",
                                help="Auto-refresh every 5 min during market hours.")
    show_rejected = ctrl4.toggle("Show rejected", value=True, key="signals_show_rej",
                                 help="Overlay rejected signal markers on chart.")
    get_btn = ctrl5.button("🔄 Refresh", use_container_width=True)

    # ── Status band ───────────────────────────────────────────────────────────
    _mkt_status_obj = market_status()
    _now_str2    = _mkt_status_obj.get("time_et", "")
    _mkt_is_open = is_market_open(symbol_input)
    _mkt_pre     = is_pre_market(symbol_input)
    _mkt_state   = "OPEN" if _mkt_is_open else ("PRE-MARKET" if _mkt_pre else "CLOSED")
    _mkt_clr     = "green" if _mkt_is_open else ("blue" if _mkt_pre else "grey")
    _last_refresh = st.session_state.get("_signals_last_refresh")
    _refresh_str  = _last_refresh.strftime("%H:%M:%S") if _last_refresh else "—"
    stat_band([
        ("Session",      _mkt_state, _mkt_clr),
        ("Clock",        _now_str2,  "grey"),
        ("Timeframe",    tf_pick,    "grey"),
        ("Last refresh", _refresh_str, "grey"),
    ])

    # ── Auto-refresh trigger ──────────────────────────────────────────────────
    if auto_refresh and _mkt_is_open:
        _last = st.session_state.get("_signals_last_refresh")
        if _last is None or (datetime.now(ET) - _last).total_seconds() >= 300:
            get_btn = True

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN WORKSTATION — chart left, signals right
    # ─────────────────────────────────────────────────────────────────────────
    chart_col, side_col = st.columns([7, 3], gap="medium")

    with chart_col:
        # ── TradingView lightweight-charts via /chart/intraday endpoint ───────
        # This is the same engine used on the Charts page: proper candles,
        # auto-zoom to the current session, backend-computed VWAP/EMA overlays,
        # and strategy markers anchored to the correct bar.
        if not get_btn and not st.session_state.get("_dt_signals_loaded"):
            st.markdown(
                "<div style='padding:60px 24px;text-align:center;"
                "color:var(--text-3);font-size:0.9rem'>"
                "Enter a symbol and click <b>Refresh</b> to load the live chart."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.session_state["_dt_signals_loaded"] = True
            st.session_state["_signals_last_refresh"] = datetime.now(ET)

            with st.spinner(f"Loading {symbol_input} {tf_pick} chart…"):
                try:
                    _chart_payload = api.intraday_chart(
                        symbol_input,
                        timeframe=tf_pick,
                        strategies="all",
                        include_rejected=show_rejected,
                    )
                except Exception as _ce:
                    _chart_payload = None
                    st.error(f"Chart fetch failed: {_ce}", icon="⚠️")

            if _chart_payload:
                # Warn about policy block but still show whatever came back
                if _chart_payload.get("warning") and "policy" in str(_chart_payload.get("warning","")).lower():
                    st.warning(_chart_payload["warning"], icon="⚠️")

                lwc.render_strategy_chart(
                    _chart_payload,
                    overlays_enabled=["vwap", "ema9", "ema21"],
                    show_trades=True,
                    show_rejected=show_rejected,
                    show_levels=True,
                    height=560,
                )
                _n_markers = len(_chart_payload.get("markers") or [])
                _n_rej = len(_chart_payload.get("rejected_markers") or [])
                st.caption(
                    f"▲ green = BUY  ·  ▼ red = SELL  ·  ✕ yellow = rejected  ·  "
                    f"{_n_markers} signal(s)  ·  {_n_rej} rejected  ·  "
                    f"Click any marker for entry/stop/target detail"
                )

    with side_col:
        # ── Signal panel: run signals for the accepted list + brain status ────
        if st.session_state.get("_dt_signals_loaded"):
            with st.spinner("Running strategy signals…"):
                result = _cached_signals(symbol_input)

            # Policy block with 1-click override
            if result.get("policy_blocked"):
                _pol_reason = result.get("policy_reason", "Symbol blocked by deployment policy.")
                st.warning(_pol_reason, icon="⚠️")
                if st.button("Enable signals for this symbol", key="policy_override_btn",
                             type="primary", use_container_width=True):
                    try:
                        from app.services.strategy.daytrading.brain.symbol_policy import (
                            set_policy, get_policy, SymbolPolicy, ENABLED,
                        )
                        _cur = get_policy(symbol_input)
                        set_policy(symbol_input, SymbolPolicy(
                            symbol=symbol_input, status=ENABLED, override_live=True,
                            reason=f"Enabled via UI {datetime.now(ET).strftime('%H:%M ET')}",
                            wf_verdict=_cur.wf_verdict, wf_score=_cur.wf_score,
                        ))
                        _cached_signals.clear()
                        st.rerun()
                    except Exception as _oe:
                        st.error(f"Override failed: {_oe}")
                st.caption("In-memory only — resets on server restart.")
                st.stop()

            regime   = result.get("regime", "CHOPPY")
            signals  = result.get("signals", [])
            rejected = result.get("rejected_signals", [])
            brain    = result.get("brain", {})
            raw_count = result.get("raw_signal_count", 0)

            # ── Brain status (compact) ────────────────────────────────────────
            if brain:
                ms_state  = brain.get("market_state", "UNKNOWN")
                ms_conf   = brain.get("state_confidence", 0)
                kill      = brain.get("kill_switch", False)
                size_mult = brain.get("size_multiplier", 1.0)
                daily_pnl = brain.get("daily_pnl_pct", 0.0)

                if kill:
                    _elig_chip = blocker_chip("KILL_SWITCH")
                elif ms_state == "NEWS_RISK":
                    _elig_chip = blocker_chip("NEWS_RISK")
                elif size_mult < 0.6:
                    _elig_chip = eligibility_chip("watch", "Reduced size")
                elif ms_state in ("CHOPPY", "HIGH_VOL"):
                    _elig_chip = eligibility_chip("watch", ms_state)
                else:
                    _elig_chip = eligibility_chip("ready")

                st.markdown(
                    f"{regime_chip(ms_state)} &nbsp;"
                    f"<span style='color:var(--text-3);font-size:0.75rem'>conf {ms_conf:.0%}</span>"
                    f"&nbsp;&nbsp; {_elig_chip}",
                    unsafe_allow_html=True,
                )
                b1, b2, b3 = st.columns(3)
                b1.metric("Size", f"{size_mult:.0%}")
                b2.metric("Trades", str(brain.get("trades_today", 0)))
                pnl_c = "normal" if daily_pnl >= 0 else "inverse"
                b3.metric("P&L", f"{daily_pnl:+.2f}%", delta_color=pnl_c)

                if kill:
                    st.error("🛑 Kill switch active", icon="🛑")

                # Strategy routing — one line
                enabled  = brain.get("enabled_strategies", [])
                disabled = brain.get("disabled_strategies", [])
                if enabled or disabled:
                    with st.expander(
                        f"Routing: {len(enabled)} on · {len(disabled)} off",
                        expanded=False,
                    ):
                        _dis_reasons = brain.get("disabled_strategies_reasons", {}) or {}
                        for s in enabled:
                            st.markdown(
                                f"<span class='tx-pill green' style='margin:2px;display:inline-block'>{s}</span>",
                                unsafe_allow_html=True,
                            )
                        for s in disabled:
                            _r = _dis_reasons.get(s, "regime")
                            st.markdown(
                                f"<span class='tx-pill red' title='{_r}' "
                                f"style='margin:2px;display:inline-block'>{s}</span>",
                                unsafe_allow_html=True,
                            )

                reasons = brain.get("state_reasons", [])
                if reasons:
                    with st.expander("Why this state?", expanded=False):
                        for r in reasons:
                            st.caption(f"• {r}")

            st.markdown("---")

            # ── Signal cards ──────────────────────────────────────────────────
            st.markdown(
                f"<div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.08em;color:var(--text-3);margin-bottom:8px'>"
                f"Signals — {len(signals)} accepted · {raw_count} raw · {len(rejected)} rejected"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"Regime: {regime_chip(regime)}",
                unsafe_allow_html=True,
            )

            if not signals and raw_count == 0:
                empty_state(
                    "No setups found",
                    "Strategies found no qualifying conditions in current bars. "
                    "Check diagnostics below.",
                    icon="📭",
                )
            elif not signals and raw_count > 0:
                st.warning(
                    f"{raw_count} raw signal(s) all rejected by brain.",
                    icon="⚠️",
                )
            else:
                for i, sig in enumerate(signals):
                    _render_signal_card(sig, i, symbol=symbol_input)
                    brain_reason = sig.get("brain_reason", "")
                    brain_size   = sig.get("brain_size_multiplier", 1.0)
                    if brain_reason:
                        st.caption(f"Brain: {brain_reason}  |  Size: {brain_size:.0%}")

            if rejected:
                with st.expander(f"{len(rejected)} rejected", expanded=False):
                    for sig in rejected:
                        st.caption(
                            f"**{sig.get('strategy')}** {sig.get('direction')} "
                            f"@ ${sig.get('entry_price', 0):.2f} — "
                            f"{sig.get('brain_reason', 'filtered')}"
                        )

            # ── Export + diagnostics ──────────────────────────────────────────
            st.markdown("---")
            _exp_rows = [
                {k: s.get(k) for k in ["strategy","direction","entry_price","stop_price",
                                        "target_price","r_multiple","confidence","regime","signal_time"]}
                for s in signals
            ]
            if _exp_rows:
                st.download_button(
                    "⬇ Export signals CSV",
                    data=pd.DataFrame(_exp_rows).to_csv(index=False),
                    file_name=f"{symbol_input}_signals_{datetime.now(ET).strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            diag_data = result.get("diagnostics", {})
            if diag_data:
                with st.expander("Pipeline diagnostics", expanded=False):
                    _render_pipeline_diagnostics(diag_data)

    # Auto-rerun schedule
    if auto_refresh and _mkt_is_open and st.session_state.get("_dt_signals_loaded"):
        import time as _t2
        _t2.sleep(0.5)
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
                    hovertemplate="Hour %{x}<br>Win rate: %{y:.0f}%<br>Trades: %{text}<extra></extra>",
                ))
                # Reference lines: market open (9) and last-entry cutoff (15)
                for _hr, _lbl, _clr in [(9, "Open 9:30", "#26C6DA"), (15, "Cutoff 15:15", "#FF9800")]:
                    _hr_str = f"{_hr}:00"
                    if _hr_str in [f"{h}:00" for h in hours]:
                        fig_h.add_vline(x=_hr_str, line_dash="dash", line_color=_clr,
                                        annotation_text=_lbl, annotation_position="top")
                fig_h.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis=dict(title="Win%", range=[0, 112]),
                    xaxis=dict(title="Entry Hour (ET)"),
                    showlegend=False,
                )
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption("Green bars ≥55% win rate · Yellow 45–55% · Red <45% · Dashed lines: market open and last-entry cutoff")

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

        # PDT warning
        if ps_account < 25_000:
            st.warning(
                f"⚠️ **PDT Rule**: Account (${ps_account:,.0f}) is below $25,000. "
                "You are limited to **3 day trades per 5 rolling business days**. "
                f"You need **${25_000 - ps_account:,.0f}** more to remove this restriction."
            )

        # Scaling plan
        st.markdown("#### Scaling Plan (recommended)")
        _scale_rows = []
        for _tier_pct, _label in [(0.50, "Tier 1 (50% at 1R)"), (0.30, "Tier 2 (30% at 2R)"), (0.20, "Tier 3 (20% trail)")]:
            _tier_shares = max(1, int(shares * _tier_pct))
            _tier_value  = _tier_shares * ps_entry
            _scale_rows.append({"Action": _label, "Shares": _tier_shares,
                                 "Value ($)": f"${_tier_value:,.0f}",
                                 "Notes": "Move stop to breakeven after this fill" if _tier_pct == 0.50 else
                                          "Trail remaining with ATR stop" if _tier_pct == 0.20 else ""})
        st.dataframe(pd.DataFrame(_scale_rows), use_container_width=True, hide_index=True)
        st.caption("Scaling reduces average hold risk while letting winners run. Adjust percentages to your style.")

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

                        # Per-strategy breakdown
                        all_strats = row.get("all_strategies", [])
                        if all_strats:
                            with st.expander("Per-strategy breakdown"):
                                strat_rows = []
                                for sr in all_strats:
                                    _spf = sr.get("profit_factor", 0)
                                    _swr = sr.get("win_rate", 0)
                                    _st  = sr.get("trades", 0)
                                    strat_rows.append({
                                        "Strategy": sr.get("strategy", ""),
                                        "Trades": _st,
                                        "Win%": f"{_swr:.0f}%",
                                        "PF": f"{_spf:.2f}",
                                        "P&L ($)": f"${sr.get('total_pnl', 0):,.0f}",
                                        "Status": "✅" if (_spf >= 1.5 and _swr >= 50) else
                                                  ("🟡" if (_spf >= 1.0 and _swr >= 40) else
                                                   ("⚪" if _st == 0 else "🔴")),
                                    })
                                st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

                        if row["all_weak"]:
                            st.info(
                                f"No historical edge on **{row['symbol']}** in the last {wl_period}. "
                                "This doesn't mean the stock won't move — it means the strategy "
                                "rules didn't produce tradeable setups in that window. "
                                "Check live signals manually if it's gapping or on a mover list."
                            )
                            # Period comparison suggestion
                            other_period = "90d" if wl_period == "30d" else "30d"
                            if st.button(f"Try {other_period} period for {row['symbol']}", key=f"try_period_{row['symbol']}"):
                                st.session_state["wl_period"] = other_period
                                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB — Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────
with tab_wf:
    st.markdown("### Walk-Forward Validation")
    st.caption(
        "Splits the historical period into in-sample (IS) training windows and "
        "out-of-sample (OOS) test windows to measure how well the strategy generalises. "
        "A **Walk-Forward Efficiency (WFE) ≥ 60%** means the edge found in IS persists OOS. "
        "WFE < 40% is a sign of overfitting."
    )

    wf1, wf2, wf3, wf4 = st.columns(4)
    wf_symbol   = wf1.text_input("Symbol", value="SPY", key="wf_symbol").upper()
    wf_strategy = wf2.selectbox("Strategy", [s.name for s in ALL_STRATEGIES], key="wf_strategy")
    wf_years    = wf3.selectbox("History (years)", [1, 2], index=1, key="wf_years",
                                 help="5m data is limited to ~60 days by most providers; longer periods use daily-bar approximation.")
    wf_step     = wf4.selectbox("Step (months)", [1, 2, 3, 6], index=2, key="wf_step",
                                 help="Slide the window forward by this many months each iteration.")
    wf_capital  = st.number_input("Capital ($)", value=10_000, step=1_000, key="wf_capital")

    if st.button("▶ Run Walk-Forward", key="run_wf", type="primary"):
        with st.spinner(f"Running walk-forward for {wf_strategy} on {wf_symbol}… (may take 1–2 min)"):
            try:
                wf_result = api.daytrading_walkforward(
                    strategy=wf_strategy, symbol=wf_symbol,
                    years=int(wf_years), step_months=int(wf_step),
                    initial_capital=float(wf_capital),
                )
                st.session_state["wf_result"] = wf_result
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")
                st.session_state["wf_result"] = None

    wf_res = st.session_state.get("wf_result")
    if wf_res:
        if wf_res.get("error"):
            st.error(wf_res["error"])
        else:
            # Top-level summary
            wfe = wf_res.get("wfe_score", 0)
            verdict = wf_res.get("verdict", "")
            wfe_color = "normal" if wfe >= 60 else ("off" if wfe >= 40 else "inverse")
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("WFE Score", f"{wfe:.0f} / 100", delta=verdict, delta_color=wfe_color)
            w2.metric("IS Win Rate",  f"{wf_res.get('is_win_rate', 0):.1f}%")
            w3.metric("OOS Win Rate", f"{wf_res.get('oos_win_rate', 0):.1f}%")
            w4.metric("Windows",      str(wf_res.get("n_windows", 0)))

            if wfe >= 60:
                st.success(f"✅ **{verdict}** — Edge found in-sample generalises out-of-sample. Confidence in live use: HIGH.")
            elif wfe >= 40:
                st.warning(f"🟡 **{verdict}** — Partial generalisation. Reduce position size in live use.")
            else:
                st.error(f"🔴 **{verdict}** — Strategy is overfit to history. Do NOT use live without further tuning.")

            # Per-window table
            windows = wf_res.get("windows", [])
            if windows:
                st.markdown("#### Per-Window Results")
                wf_rows = []
                for w in windows:
                    wf_rows.append({
                        "Window": w.get("window", ""),
                        "IS Period": w.get("is_period", ""),
                        "OOS Period": w.get("oos_period", ""),
                        "IS Trades": w.get("is_trades", 0),
                        "IS Win%": f"{w.get('is_win_rate', 0):.1f}%",
                        "IS PF": f"{w.get('is_profit_factor', 0):.2f}",
                        "OOS Trades": w.get("oos_trades", 0),
                        "OOS Win%": f"{w.get('oos_win_rate', 0):.1f}%",
                        "OOS PF": f"{w.get('oos_profit_factor', 0):.2f}",
                        "OOS P&L ($)": f"${w.get('oos_pnl', 0):,.0f}",
                        "Pass": "✅" if w.get("oos_pass") else "❌",
                    })
                st.dataframe(pd.DataFrame(wf_rows), use_container_width=True, hide_index=True)

            notes = wf_res.get("notes", [])
            if notes:
                with st.expander("Analysis notes"):
                    for n in notes:
                        st.caption(f"• {n}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Scanner
# ─────────────────────────────────────────────────────────────────────────────
with tab_scanner:
    import api as _api
    from _theme import section as _section

    _section(
        "Market Scanner",
        "Full US universe (~5,800 names) ranked by volume, volatility, RVOL, gap, and catalyst. "
        "Filters narrow to your exact price/float band. Native pre-check flags setups firing right now.",
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

            # Sort: active signals first, then by adj score descending
            if "Signal?" in df_market.columns:
                df_market["_sig_sort"] = df_market["Signal?"].apply(lambda v: 0 if v == "🟢" else 1)
                df_market = df_market.sort_values(["_sig_sort", "Adj Score"], ascending=[True, False])
                df_market = df_market.drop(columns=["_sig_sort"])

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
                + " · Signals-active candidates sorted to top."
            )

            # Quick-backtest row: one button per top symbol, pre-fills Backtest tab
            _top_syms = [r.get("symbol", "") for r in ranked[:8] if r.get("symbol")]
            if _top_syms:
                st.markdown("**Quick Backtest** — click a symbol to pre-fill the Backtest tab:")
                _bt_cols = st.columns(min(len(_top_syms), 8))
                for _ci, _sym in enumerate(_top_syms):
                    _sig_icon = "🟢 " if any(r.get("symbol") == _sym and r.get("native_signal_active") for r in ranked) else ""
                    if _bt_cols[_ci].button(f"{_sig_icon}{_sym}", key=f"qbt_{_sym}"):
                        st.session_state["bt_symbol"] = _sym
                        st.session_state["ca_symbol"] = _sym
                        st.info(f"Pre-filled '{_sym}' in Backtest and Compare All tabs — switch to Research tab to run.")

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

    st.caption("Toggle changes are applied immediately in-memory and reset on server restart.")
    for strategy in ALL_STRATEGIES:
        with st.expander(f"⚙️ {strategy.name}  —  {STRATEGY_DESCRIPTIONS.get(strategy.name, '')}"):
            enabled_key  = f"cfg_enabled_{strategy.name}"
            _was_enabled = st.session_state.get(enabled_key, True)
            _now_enabled = st.toggle(f"Enable {strategy.name}", value=_was_enabled, key=enabled_key)

            # Wire toggle to backend when it changes
            if _now_enabled != _was_enabled:
                try:
                    api.daytrading_toggle_strategy(strategy.name, _now_enabled)
                    st.toast(f"{'Enabled' if _now_enabled else 'Disabled'} {strategy.name}", icon="✅" if _now_enabled else "🚫")
                except Exception as _te:
                    st.warning(f"Could not update backend: {_te}")

            cfg = strategy.default_config
            st.markdown("**Parameters (auto-tuned per symbol in live/backtest; defaults shown):**")
            cfg_df = pd.DataFrame(
                [{"Parameter": k, "Default Value": v} for k, v in cfg.items()]
            )
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB — 🎯 Single Stock Auto Trader
# ─────────────────────────────────────────────────────────────────────────────
with tab_autotrader:

    # ── Session state init ────────────────────────────────────────────────────
    if "at_trader" not in st.session_state:
        st.session_state["at_trader"] = None
    if "at_status" not in st.session_state:
        st.session_state["at_status"] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TOP CONTROL BAR — symbol, mode toggle, broker, start/stop/flatten
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cb1, cb2, cb3, cb4, cb5, cb6, cb7 = st.columns([2, 1, 1, 1, 1, 1, 1])

    at_symbol = cb1.text_input(
        "Symbol", value=st.session_state.get("signals_symbol", "AAPL"),
        key="at_symbol", placeholder="AAPL, MTEN, SPY …",
    ).strip().upper() or "AAPL"

    at_simulation = cb2.toggle(
        "Simulation", value=True, key="at_simulation",
        help=(
            "ON = paper simulation: bot runs all logic, shows where it would buy/sell "
            "and the P&L it would realize — no real orders placed.\n\n"
            "OFF = live trading: real orders sent to the selected broker."
        ),
    )

    at_dir = cb3.selectbox(
        "Direction", ["Long only", "Both", "Short only"],
        index=0, key="at_dir",
    )

    at_tf = cb4.selectbox("Timeframe", ["5m", "1m", "15m"], index=0, key="at_tf")

    start_pressed   = cb5.button("▶ Start", use_container_width=True, type="primary", key="at_start")
    stop_pressed    = cb6.button("⏹ Stop",  use_container_width=True, key="at_stop")
    flatten_pressed = cb7.button("🚨 Flatten", use_container_width=True, key="at_flatten")

    # Mode indicator bar
    if at_simulation:
        st.markdown(
            "<div style='background:rgba(91,146,209,0.12);border:1px solid rgba(91,146,209,0.3);"
            "border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:0.84rem'>"
            "🔵 <b>SIMULATION MODE</b> — paper trading only. "
            "The bot runs all strategy logic and shows exactly where it would enter, "
            "manage stops, take profits, and exit. <b>No real orders are placed.</b> "
            "Switch Simulation OFF + select a broker to go live."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:rgba(208,122,122,0.12);border:1px solid rgba(208,122,122,0.4);"
            "border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:0.84rem'>"
            "🔴 <b>LIVE TRADING MODE</b> — real orders will be sent to your broker. "
            "Verify broker routing and risk limits before starting."
            "</div>",
            unsafe_allow_html=True,
        )
        render_broker_routing_toggle(key_suffix="at_live")

    # ── Risk configuration (compact, collapsed by default when running) ───────
    trader_running = bool(st.session_state.get("at_status", {}).get("running"))
    with st.expander("⚙️ Risk & Strategy Config", expanded=not trader_running):
        rc1, rc2, rc3, rc4 = st.columns(4)
        at_risk_pct   = rc1.number_input("Risk per trade %", 0.1, 5.0, 1.0, 0.1, key="at_risk",
                                          help="% of capital risked per trade (stop distance × shares).")
        at_max_loss   = rc2.number_input("Max daily loss %", 0.5, 10.0, 2.0, 0.5, key="at_maxloss",
                                          help="Bot stops trading for the day when realized loss exceeds this.")
        at_capital    = rc3.number_input("Capital ($)", 1000, 10_000_000, 10_000, 1000, key="at_capital")
        at_max_trades = rc4.number_input("Max trades/day", 1, 20, 6, 1, key="at_maxtrades")

        sc1, sc2, sc3 = st.columns(3)
        at_partial_tp = sc1.toggle("Partial TP at +1R", value=True, key="at_partial",
                                    help="Scale out 50% of position at 1× risk gained.")
        at_trail_mode = sc2.selectbox("Trail stop mode", ["atr", "ema", "candle"], key="at_trail")
        at_entry_mode = sc3.selectbox(
            "Entry mode",
            ["native_strategy", "legacy_entry_decider"],
            index=0, key="at_entry_mode",
            help=(
                "native_strategy: uses the same strategy.generate_signals() as the backtest "
                "(recommended — most consistent with what you see on the signals chart).\n\n"
                "legacy_entry_decider: uses the older indicator-scoring entry path."
            ),
        )

    # ── Start / stop logic ────────────────────────────────────────────────────
    if start_pressed:
        try:
            from app.services.strategy.daytrading.autotrader import SingleStockTrader
            from app.services.strategy.daytrading.autotrader.single_stock_trader import PolicyError
            from app.services.strategy.daytrading.brain.symbol_policy import get_policy, set_policy, SymbolPolicy, ENABLED

            # Auto-enable policy so the bot can always trade what you pick
            _pol = get_policy(at_symbol)
            if not _pol.allows_live():
                set_policy(at_symbol, SymbolPolicy(
                    symbol=at_symbol, status=ENABLED, override_live=True,
                    reason=f"Auto-enabled for bot run {datetime.now(ET).strftime('%H:%M ET')}",
                    wf_verdict=_pol.wf_verdict, wf_score=_pol.wf_score,
                ))

            dir_map = {"Long only": "long_only", "Short only": "short_only", "Both": "both"}

            # Broker: None = paper sim; real broker instance for live
            _broker = None
            _exec_svc = None
            _acct_id  = ""
            if not at_simulation:
                from app.services.broker import get_broker as _get_broker
                from app.services.execution.service import ExecutionService
                import asyncio as _aio
                _broker = _get_broker()
                _aio.run(_broker.authenticate())
                _accts  = _aio.run(_broker.get_accounts())
                _acct_id = _accts[0].account_id if _accts else ""
                _exec_svc = ExecutionService(_broker)

            trader = SingleStockTrader(
                symbol=at_symbol,
                broker=_broker,
                execution_service=_exec_svc,
                account_id=_acct_id,
                direction_mode=dir_map[at_dir],
                trail_mode=at_trail_mode,
                partial_tp=at_partial_tp,
                risk_per_trade_pct=at_risk_pct / 100,
                max_daily_loss_pct=at_max_loss,
                max_trades_per_day=int(at_max_trades),
                initial_capital=float(at_capital),
                entry_mode=at_entry_mode,
                on_trade_update=lambda s: st.session_state.update({"at_status": s}),
            )
            # Stop previous trader if running
            old = st.session_state.get("at_trader")
            if old is not None:
                try: old.stop()
                except Exception: pass

            st.session_state["at_trader"] = trader
            trader.start(force=True)   # force=True allows start outside RTH for testing
            _mode_label = "SIMULATION" if at_simulation else "LIVE"
            st.success(
                f"✅ Bot started for **{at_symbol}** in **{_mode_label}** mode. "
                f"Polling every 30s · EOD flatten at market close."
            )
        except Exception as e:
            st.error(f"Failed to start: {e}", icon="❌")

    if stop_pressed:
        _trader = st.session_state.get("at_trader")
        if _trader:
            _trader.stop()
            st.info("Bot stopped.")
        else:
            st.warning("No active bot.")

    if flatten_pressed:
        _trader = st.session_state.get("at_trader")
        if _trader:
            _trader.force_flatten("Manual flatten from UI")
            st.warning("🚨 Force flatten executed — all positions closed at market.")
        else:
            st.warning("No active bot.")

    # ── Refresh status from running trader ────────────────────────────────────
    _trader_obj = st.session_state.get("at_trader")
    if _trader_obj is not None:
        try:
            st.session_state["at_status"] = _trader_obj.get_status()
        except Exception:
            pass

    status = st.session_state.get("at_status", {})

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN AREA — chart (left) | position + log (right)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    chart_col, ctrl_col = st.columns([7, 3], gap="medium")

    with chart_col:
        # ── Live TradingView chart with signal + trade markers ────────────────
        # Pulls /chart/intraday which runs strategies server-side and returns
        # marker positions with entry/stop/target. Refreshes every poll cycle.
        _at_chart_sym = at_symbol
        try:
            _at_payload = api.intraday_chart(
                _at_chart_sym,
                timeframe=at_tf,
                strategies="all",
                include_rejected=True,
            )
        except Exception as _cex:
            _at_payload = None
            st.warning(f"Chart load failed: {_cex}", icon="⚠️")

        if _at_payload:
            # Inject bot trade markers (sim fills) on top of strategy signals
            _bot_markers = []
            _session_trades = status.get("session_trades", []) or []
            for _t in _session_trades:
                _fill_ts = _t.get("entry_time", "")
                _exit_ts = _t.get("exit_time", "")
                if _fill_ts and _t.get("entry_price"):
                    _bot_markers.append({
                        "time": _fill_ts, "side": _t.get("side", "BUY").upper(),
                        "entry_price": _t.get("entry_price"), "strategy": "BOT ENTRY",
                        "reason": f"Bot entered @ ${_t.get('entry_price',0):.2f}",
                        "_accepted": True,
                    })
                if _exit_ts and _t.get("exit_price"):
                    _exit_side = "SELL" if _t.get("side","BUY").upper() == "BUY" else "BUY"
                    _bot_markers.append({
                        "time": _exit_ts, "side": _exit_side,
                        "entry_price": _t.get("exit_price"),
                        "strategy": f"BOT EXIT ({_t.get('exit_reason','')[:20]})",
                        "reason": (
                            f"P&L: ${_t.get('pnl', 0):+.2f} ({_t.get('pnl_pct', 0):+.2f}%) — "
                            f"{_t.get('exit_reason', '')}"
                        ),
                        "_accepted": True,
                    })
            # Merge bot fills into the payload's marker list
            if _bot_markers:
                _at_payload = dict(_at_payload)
                _at_payload["markers"] = list(_at_payload.get("markers") or []) + _bot_markers

            lwc.render_strategy_chart(
                _at_payload,
                overlays_enabled=["vwap", "ema9", "ema21"],
                show_trades=True,
                show_rejected=True,
                show_levels=True,     # draws entry/stop/target lines
                height=540,
            )
            _mode_note = "SIMULATION — showing where bot would trade" if at_simulation else "LIVE — real orders"
            st.caption(
                f"▲ green = BUY signal  ·  ▼ red = SELL signal  ·  ✕ yellow = rejected  ·  "
                f"BOT fills overlaid in same colors  ·  {_mode_note}  ·  "
                "Click any marker for entry/stop/target detail"
            )

    with ctrl_col:
        # ── Bot state header ──────────────────────────────────────────────────
        if not status:
            empty_state(
                "Bot not running",
                "Configure a symbol above and click ▶ Start.",
                icon="🤖",
            )
        else:
            state_val  = status.get("state", "UNKNOWN")
            at_regime  = status.get("market_state", "UNKNOWN")
            unreal_pnl = status.get("unrealized_pnl", 0) or 0
            real_pnl   = status.get("realized_pnl", 0) or 0
            block_rsn  = status.get("block_reason", "")
            is_sim     = _trader_obj._broker is None if _trader_obj else True

            # Mode + state chips
            mode_chip = (
                "<span class='tx-pill blue'>SIMULATION</span>"
                if is_sim else
                "<span class='tx-pill red'>LIVE</span>"
            )
            st.markdown(
                f"{mode_chip} &nbsp; {regime_chip(at_regime)} &nbsp; {bot_state_chip(state_val, block_rsn)}",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # P&L metrics
            pnl1, pnl2, pnl3 = st.columns(3)
            pnl1.metric("Unrealized", f"${unreal_pnl:+,.2f}",
                        delta_color="normal" if unreal_pnl >= 0 else "inverse")
            pnl2.metric("Realized",   f"${real_pnl:+,.2f}",
                        delta_color="normal" if real_pnl >= 0 else "inverse")
            pnl3.metric("Total",      f"${unreal_pnl + real_pnl:+,.2f}",
                        delta_color="normal" if (unreal_pnl + real_pnl) >= 0 else "inverse")

            st.markdown("---")

            # ── Active position panel ─────────────────────────────────────────
            if state_val in ("LONG", "SHORT", "PARTIAL_EXIT_TAKEN", "TRAILING"):
                entry_px = status.get("entry_price", 0) or 0
                curr_px  = status.get("current_price", 0) or status.get("last_price", 0) or 0
                stop_px  = status.get("current_stop", 0) or 0
                tgt_px   = status.get("first_target", 0) or 0
                r_val    = status.get("r_multiple")
                qty      = status.get("qty", 0) or 0

                st.markdown(
                    f"<div style='font-size:0.69rem;font-weight:700;text-transform:uppercase;"
                    f"letter-spacing:0.08em;color:var(--text-3);margin-bottom:6px'>"
                    f"Active position — {status.get('side','—')} {at_symbol}</div>",
                    unsafe_allow_html=True,
                )
                p1, p2 = st.columns(2)
                p1.metric("Entry",   f"${entry_px:.2f}")
                p2.metric("Current", f"${curr_px:.2f}",
                          delta=f"{curr_px - entry_px:+.2f}" if curr_px and entry_px else None,
                          delta_color="normal" if curr_px >= entry_px else "inverse")
                p3, p4 = st.columns(2)
                p3.metric("Stop",   f"${stop_px:.2f}",
                          delta=f"−${abs(curr_px - stop_px):.2f} risk" if curr_px and stop_px else None,
                          delta_color="off")
                p4.metric("Target", f"${tgt_px:.2f}",
                          delta=f"+${abs(tgt_px - curr_px):.2f} reward" if curr_px and tgt_px else None,
                          delta_color="off")
                p5, p6 = st.columns(2)
                p5.metric("R Multiple", f"{r_val:+.2f}R" if r_val is not None else "—")
                p6.metric("Qty", f"{qty:g}")

                # Time in trade + scenario analysis
                _entry_time = status.get("entry_time", "")
                if _entry_time:
                    try:
                        _et_ts = datetime.fromisoformat(_entry_time.replace("Z", "+00:00"))
                        _mins = int((datetime.now(ET) - _et_ts.astimezone(ET)).total_seconds() // 60)
                        _max_profit = abs(tgt_px - entry_px) * qty if tgt_px and entry_px else 0
                        _max_loss   = abs(entry_px - stop_px) * qty if entry_px and stop_px else 0
                        st.caption(
                            f"⏱ {_mins}m in trade · "
                            f"If target: **${_max_profit:,.0f}** · "
                            f"If stopped: **−${_max_loss:,.0f}**"
                        )
                        if state_val == "TRAILING":
                            st.caption(f"Trail mode: {status.get('active_trail_mode','—')}")
                    except Exception:
                        pass

            elif state_val == "FLAT":
                cooldown_left   = status.get("cooldown_bars_remaining", 0) or 0
                no_trade_reason = status.get("last_no_trade_reason", "") or ""
                if cooldown_left > 0:
                    st.markdown(
                        f"{eligibility_chip('cooldown')} "
                        f"<span style='color:var(--text-3);font-size:0.82rem'>"
                        f"{cooldown_left} bar(s) before next entry</span>",
                        unsafe_allow_html=True,
                    )
                elif no_trade_reason:
                    st.markdown(
                        f"{eligibility_chip('idle')} "
                        f"<span style='color:var(--text-3);font-size:0.82rem'>{no_trade_reason}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"{eligibility_chip('idle')} "
                        f"<span style='color:var(--text-3);font-size:0.82rem'>Scanning for setup…</span>",
                        unsafe_allow_html=True,
                    )

            elif state_val == "BLOCKED":
                _block_rsn = status.get("block_reason", "Risk limit hit")
                st.error(f"BLOCKED — {_block_rsn}", icon="🚫")

            st.markdown("---")

            # ── Session stats ─────────────────────────────────────────────────
            ss1, ss2 = st.columns(2)
            ss1.metric("Trades today",    status.get("trades_today", 0))
            ss2.metric("Consec. losses",  status.get("consecutive_losses", 0))
            hb = status.get("heartbeat_age_s", 0) or 0
            hb_label = f"{hb:.0f}s ago" if hb < 120 else f"⚠️ {hb:.0f}s — may be stale"
            st.caption(f"Heartbeat: {hb_label} · Running: {'✅' if status.get('running') else '⏸'}")

            st.markdown("---")

            # ── Closed trades table ───────────────────────────────────────────
            session_trades = status.get("session_trades", []) or []
            if session_trades:
                st.markdown(
                    "<div style='font-size:0.69rem;font-weight:700;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:var(--text-3);margin-bottom:6px'>"
                    f"Closed trades today ({len(session_trades)})</div>",
                    unsafe_allow_html=True,
                )
                _trade_rows = []
                for _t in session_trades:
                    _pnl = _t.get("pnl", 0) or 0
                    _trade_rows.append({
                        "Time":   (_t.get("entry_time") or "")[:16].replace("T", " "),
                        "Side":   _t.get("side", ""),
                        "Entry":  _t.get("entry_price", 0),
                        "Exit":   _t.get("exit_price", 0),
                        "Qty":    _t.get("qty", 0),
                        "P&L $":  _pnl,
                        "P&L %":  _t.get("pnl_pct", 0),
                        "Reason": (_t.get("exit_reason") or "")[:25],
                    })
                st.dataframe(
                    pd.DataFrame(_trade_rows).style.map(
                        lambda v: "color:#4db896" if isinstance(v, (int, float)) and v > 0
                                  else "color:#d07a7a" if isinstance(v, (int, float)) and v < 0
                                  else "",
                        subset=["P&L $", "P&L %"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Entry": st.column_config.NumberColumn("Entry", format="$%.2f", width="small"),
                        "Exit":  st.column_config.NumberColumn("Exit",  format="$%.2f", width="small"),
                        "Qty":   st.column_config.NumberColumn("Qty",   format="%.1f",  width="small"),
                        "P&L $": st.column_config.NumberColumn("P&L $", format="$%+.2f",width="small"),
                        "P&L %": st.column_config.NumberColumn("P&L %", format="%+.2f%%",width="small"),
                    },
                )

            # ── Decision log ──────────────────────────────────────────────────
            log = status.get("decision_log", []) or []
            if log:
                with st.expander(f"Decision log ({len(log)} entries)", expanded=False):
                    _ICONS = {
                        "ENTRY LONG": "🟢", "ENTRY SHORT": "🔴",
                        "EXIT": "✅", "EXIT LONG": "✅", "EXIT SHORT": "✅",
                        "MOVE_STOP": "🔧", "PARTIAL_EXIT": "📤",
                        "NO_TRADE": "⬜", "BLOCKED": "🚫",
                        "TRAIL_ACTIVATED": "🔵", "RESET": "🔄", "COOLDOWN": "⏳",
                    }
                    for _entry in reversed(log[-50:]):
                        _event  = _entry.get("event", "")
                        _icon   = _ICONS.get(_event, "ℹ️")
                        _reason = _entry.get("reason", "")
                        _checks = _entry.get("checks", {}) or {}
                        _extra  = ""
                        if "ENTRY" in _event:
                            _s = _checks.get("strategy") or _entry.get("strategy", "")
                            _c = _checks.get("confidence")
                            _r = _checks.get("rr")
                            if _s: _extra += f" · {_s}"
                            if _c is not None: _extra += f" · {float(_c):.0%}"
                            if _r is not None: _extra += f" · {float(_r):.1f}R"
                        elif _event == "NO_TRADE":
                            _gr = (_checks.get("guard") or {}).get("reason", "")
                            if _gr and _gr not in _reason:
                                _extra = f" [{_gr[:50]}]"
                        st.markdown(
                            f"`{_entry.get('time','')[-8:]}` {_icon} **{_event}**{_extra} — {_reason}"
                        )

    # ── Auto-refresh while bot is running ─────────────────────────────────────
    if status.get("running"):
        import time as _t
        _t.sleep(0.3)
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
