from __future__ import annotations

import sys
import os

dashboard_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
workspace_root = os.path.abspath(os.path.join(dashboard_root, ".."))
sys.path.insert(0, workspace_root)
sys.path.insert(0, dashboard_root)

import api

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.config import get_settings

st.set_page_config(page_title="Perplexity Strategies", page_icon="🧠", layout="wide")
st.title("🧠 Perplexity Swing Strategies")
st.caption(
    "8 swing trading strategies — EMA mean reversion, MA crossover, consolidation breakout, "
    "BB mean reversion, Fibonacci pullback, RSI swing reversal, Supertrend, and BB breakout. "
    "Daily bars, 3–20 day holds. Scan the full market or analyse a single symbol."
)

STRATEGY_DESCRIPTIONS = {
    "EMA_Mean_Reversion":    "Connors RSI(2) mean reversion — BULL market only (price > SMA200). "
                              "Buys when RSI(2) drops below 10 (extreme oversold), skips high-volatility spike days (ATR% guard). "
                              "Exit: RSI(2) > 70 OR close > SMA(5). Stop: 5% hard stop. Target: +8%. Max hold: 10 bars. "
                              "Highest-frequency strategy, 7–8 trades/year. Best on: AAPL, MSFT, GOOGL, SPY.",
    "MA_Crossover_RSI":      "EMA(9/21) bullish crossover confirmed by MACD above its signal line. "
                              "BULL only. Requires RSI in 45–65 zone + volume ≥ 1.1× average at crossover. "
                              "Stop: entry − 1.5×ATR. Target: entry + 2×ATR. Max hold: 20 bars. "
                              "Best on: SPY, QQQ, NVDA — trending names with clean momentum transitions.",
    "Breakout_Consolidation":"Bollinger Band Width squeeze (5+ consecutive bars of narrowing bandwidth) "
                              "followed by close above the upper BB with volume ≥ 1.3× average. BULL only. "
                              "Stop: lower BB at entry. Target: entry + one full BB width. Max hold: 15 bars. "
                              "Best on: TSLA, META, AMZN — stocks that compress then explode.",
    "BB_Mean_Reversion":     "Pullback to rising EMA(50) with bullish rejection wick. Works in BULL and mild BEAR. "
                              "Requires: EMA(50) slope up + price within ±1% of EMA(50) + RSI 35–55 "
                              "+ lower wick ≥ 40% of bar range. Stop: 2% hard stop. Target: +3%. Max hold: 20 bars. "
                              "Highest-frequency strategy. Best on: AAPL, MSFT, GOOGL — smooth EMA50 tapestry.",
    "Fib_Pullback_Support":  "ATR-spike / fear-capitulation reversal. Works in BULL and mild BEAR. "
                              "Triggers when ATR% > 3% (panic day) + RSI < 30 + BB%B < 0.15 + lower rejection wick ≥ 50% "
                              "+ 3-bar prior decline ≥ 2%. Stop: 4% hard stop. Target: +6%. Max hold: 8 bars. "
                              "Best on: SPY, AAPL, MSFT — liquid names that recover fast after selloffs.",
    "RSI_Swing_Reversal":    "RSI(14) pullback reversal in an uptrend. Price must be above rising EMA(50). "
                              "Buys when RSI dips below 40 on the prior bar then turns back up — confirming the dip "
                              "is a pullback rather than a breakdown. Stop: 1.5×ATR. Target: 2.5×ATR. Max hold: 15 bars. "
                              "Best on: AAPL, MSFT, SPY — smooth EMA(50) uptrends.",
    "Supertrend_Swing":      "Supertrend indicator (ATR-based trailing stop) trend-follow on daily bars. "
                              "Enters when Supertrend flips from bearish to bullish with ADX > 20 (filters trendless chop). "
                              "Exit: Supertrend flips bearish. Stop: Supertrend line. Target: 2.5×risk. Max hold: 20 bars. "
                              "Best on: trending names — NVDA, TSLA, AMD during strong directional moves.",
    "BB_Breakout":           "Bollinger Band upper-band breakout with expanding bands and above-average volume. "
                              "Enters when close breaks above BB(20,2) upper band after being inside the bands, "
                              "with RSI 52–80 (momentum confirmation) and volume ≥ 1.2×. "
                              "Stop: BB midline. Target: upper band + one full band width. Max hold: 12 bars. "
                              "Best on: TSLA, NVDA, AMZN — volatile names with strong breakout momentum.",
}

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL"]

# ── Shared indicator label map ─────────────────────────────────
INDICATOR_LABELS = {
    "rsi":             "RSI at Entry",
    "atr_pct":         "Volatility (ATR%)",
    "ema_dist_pct":    "EMA20 Distance %",
    "sma200_dist_pct": "SMA200 Distance %",
    "bb_pct":          "BB Band Position (0=lower, 1=upper)",
    "volume_ratio":    "Volume vs 20d Avg",
    "body_pct":        "Candle Body Size (x ATR)",
    "lower_wick_pct":  "Lower Wick % of Range (rejection)",
    "upper_wick_pct":  "Upper Wick % of Range",
    "hold_bars":       "Hold Duration (bars)",
    "ema_spread_pct":  "EMA20/50 Spread % (MA Crossover momentum)",
    "range_atr_ratio": "Base Range / ATR (Breakout tightness)",
    "bb_depth_pct":    "BB Lower Band Depth % (BB oversold severity)",
    "fib_level":       "Fibonacci Level Hit (0.382 / 0.50 / 0.618)",
}


def _timing_chart(data: dict, title: str, container):
    """Bar chart of win rate by time bucket (month / quarter / weekday)."""
    if not data:
        return
    labels = list(data.keys())
    wr     = [data[k]["win_rate"] for k in labels]
    trades = [data[k]["trades"]   for k in labels]
    avg_p  = [data[k].get("avg_pnl_pct", 0) for k in labels]
    colors = ["#00d4aa" if w >= 55 else ("#FFD700" if w >= 45 else "#ff4b4b") for w in wr]
    fig = go.Figure(go.Bar(
        x=labels, y=wr, marker_color=colors,
        text=[f"{w:.0f}%<br>({t}t)" for w, t in zip(wr, trades)],
        textposition="outside",
        hovertext=[f"Win rate: {w:.0f}%<br>Trades: {t}<br>Avg P&L: {p:+.1f}%"
                   for w, t, p in zip(wr, trades, avg_p)],
        hoverinfo="text",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="50%")
    fig.update_layout(
        title=title, height=250, template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 110], title="Win Rate %"),
    )
    container.plotly_chart(fig, use_container_width=True)


def _breakdown_tables(bd: dict):
    """Render by-regime / by-volatility / by-R-bucket breakdown tables from a backtest result."""
    def _bt_table(rows_data: dict, label_key: str, section_title: str):
        if not rows_data:
            return
        rows = [{
            label_key:   key.title(),
            "Trades":    v["total_trades"],
            "Win Rate":  f"{v['win_rate_pct']:.1f}%",
            "PF":        v["profit_factor"] if v["profit_factor"] else "—",
            "Expectancy":f"{v['expectancy_pct']:+.2f}%",
            "Avg Hold":  f"{v['average_holding_days']:.1f}",
        } for key, v in rows_data.items()]
        st.write(f"**{section_title}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    _bt_table(bd.get("by_regime",     {}), "Regime",     "By regime")
    _bt_table(bd.get("by_volatility", {}), "Volatility", "By volatility bucket")
    if bd.get("by_r_bucket"):
        rows = [{
            "R bucket":  key,
            "Trades":    v["total_trades"],
            "Win Rate":  f"{v['win_rate_pct']:.1f}%",
            "PF":        v["profit_factor"] if v["profit_factor"] else "—",
            "Expectancy":f"{v['expectancy_pct']:+.2f}%",
            "Avg Hold":  f"{v['average_holding_days']:.1f}",
        } for key, v in bd["by_r_bucket"].items()]
        st.write("**By R bucket**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _plot_equity_curve(result: dict, height: int = 380, show_markers: bool = True) -> None:
    """Render a dark-themed equity curve from a backtest result dict."""
    if not result.get("equity_curve"):
        return
    eq_df = pd.DataFrame(result["equity_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["date"], y=eq_df["equity"],
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        line=dict(color="#00d4aa", width=2), name="Portfolio Value",
    ))
    if result.get("initial_capital"):
        fig.add_hline(y=result["initial_capital"], line_dash="dash",
                      line_color="rgba(255,255,255,0.3)", annotation_text="Starting Capital")
    if show_markers and result.get("trades"):
        buys  = [t for t in result["trades"] if t["side"] == "BUY"]
        sells = [t for t in result["trades"] if "SELL" in t["side"]]
        if buys:
            bx = [t["date"] for t in buys]
            by = [next((e["equity"] for e in result["equity_curve"] if e["date"] == d), None) for d in bx]
            fig.add_trace(go.Scatter(x=bx, y=by, mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00d4aa"), name="BUY"))
        if sells:
            sx = [t["date"] for t in sells]
            sy = [next((e["equity"] for e in result["equity_curve"] if e["date"] == d), None) for d in sx]
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ff4b4b"), name="SELL"))
    fig.update_layout(
        height=height, template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickprefix="$",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab layout ────────────────────────────────────────────────
tab_signals, tab_scanner, tab_sizer, tab_backtest, tab_compare, tab_portfolio, tab_walkforward, tab_profiles, tab_analysis, tab_config = st.tabs([
    "📡 Live Signals", "🔭 Market Scanner", "📐 Position Sizer", "🔬 Backtest", "📊 Compare All",
    "🗂 Portfolio", "🔀 Walk-Forward", "🎯 Symbol Profiles", "🔍 Analysis", "⚙️ Config"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE SIGNALS
# ══════════════════════════════════════════════════════════════
with tab_signals:
    st.subheader("Current Signals")
    st.caption("Runs all 5 strategies on the latest market data — no trade is placed, just analysis.")

    col1, col2 = st.columns([2, 1])
    with col1:
        sig_symbol = st.text_input("Symbol", value="AAPL", key="sig_sym",
                                   help="Type any US stock ticker — not limited to the default list.").upper().strip()
    with col2:
        st.write("")
        st.write("")
        run_sig = st.button("▶ Get Signals", type="primary", use_container_width=True)

    if run_sig:
        with st.spinner(f"Fetching signals for {sig_symbol}..."):
            try:
                signals = api.perplexity_signals(sig_symbol)
                st.session_state["px_signals"] = signals
            except Exception as e:
                st.error(f"Failed: {e}")

    sigs = st.session_state.get("px_signals")
    if sigs:
        buy_count  = sum(1 for s in sigs if s["direction"] == "BUY")
        sell_count = sum(1 for s in sigs if s["direction"] == "SELL")
        hold_count = sum(1 for s in sigs if s["direction"] == "HOLD")
        regime_label = sigs[0].get("regime", "unknown").replace("_", " ").title() if sigs else "Unknown"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY signals",  buy_count)
        c2.metric("🔴 SELL signals", sell_count)
        c3.metric("⬜ HOLD",         hold_count)
        c4.metric("📈 Market Regime", regime_label)
        st.divider()

        for s in sigs:
            direction = s["direction"]
            name = s["strategy"]
            blocked = s.get("suitability_blocked", False)
            icon = "🚫" if blocked else ("🟢" if direction == "BUY" else ("🔴" if direction == "SELL" else "⬜"))
            conf_str = f"confidence: {s['confidence']:.0%}" if s["confidence"] else ""
            bucket_str = f" | Vol bucket: {s['volatility_bucket']}" if s.get("volatility_bucket") else ""

            with st.expander(
                f"{icon} **{name}** — {direction}{bucket_str}  |  {conf_str}",
                expanded=(direction == "BUY"),
            ):
                st.caption(STRATEGY_DESCRIPTIONS.get(name, ""))
                if s.get("suitability_blocked"):
                    st.warning(f"Blocked by suitability: {s.get('suitability_reason', 'Rule mismatch')}")
                if s["reason"] and not s.get("suitability_blocked"):
                    st.info(f"**Reason:** {s['reason']}")

                # Entry / Stop / Target
                if s["entry_price"] or s["stop_price"] or s["target_price"]:
                    c1, c2, c3 = st.columns(3)
                    if s["entry_price"]:
                        c1.metric("Entry", f"${s['entry_price']:,.2f}")
                    if s["stop_price"] and s["entry_price"]:
                        stop_pct = (s["entry_price"] - s["stop_price"]) / s["entry_price"] * 100
                        c2.metric("Stop", f"${s['stop_price']:,.2f}",
                                  delta=f"-{stop_pct:.1f}% from entry", delta_color="inverse")
                    if s["target_price"] and s["entry_price"]:
                        tgt_pct = (s["target_price"] - s["entry_price"]) / s["entry_price"] * 100
                        c3.metric("Target", f"${s['target_price']:,.2f}",
                                  delta=f"+{tgt_pct:.1f}% from entry")

                # Position sizing block (BUY signals only)
                sz = s.get("position_size")
                if sz:
                    st.divider()
                    if not sz["viable"]:
                        st.warning(f"⚠️ Position sizing skipped: {sz['skip_reason']}")
                    else:
                        st.markdown("**📐 Position Size (based on your account settings)**")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Shares to Buy",    f"{sz['shares']:,.4f}")
                        s2.metric("Position Value",   f"${sz['position_value']:,.2f}")
                        s3.metric("$ at Risk",        f"${sz['risk_amount']:,.2f}",
                                  delta=f"{sz['risk_pct_of_account']:.2f}% of account",
                                  delta_color="off")
                        rr = ((s["target_price"] - s["entry_price"]) / sz["stop_distance"]
                              if s.get("target_price") and sz["stop_distance"] > 0 else None)
                        s4.metric("Risk:Reward", f"1 : {rr:.1f}" if rr else "—")
                        if sz["capped"]:
                            st.caption(f"ℹ️ {sz['cap_reason']}")

                if s["indicators"] and direction != "HOLD":
                    with st.expander("Indicator values", expanded=False):
                        st.json(s["indicators"])


# ══════════════════════════════════════════════════════════════
# TAB 2 — MARKET SCANNER
# ══════════════════════════════════════════════════════════════
from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES as _ALL_STRATEGIES
_STRATEGY_NAMES = [s.name for s in _ALL_STRATEGIES]

with tab_scanner:
    st.subheader("🔭 Market Scanner")
    st.caption(
        "Enter any tickers you want to scan. Live prices are fetched from Schwab; "
        "historical bars (indicators) come from yfinance. Results show actionable signals ready for auto-trading."
    )

    st.info(
        "**How to use:** Paste tickers you want to watch — one per line or comma-separated. "
        "The scanner runs all 8 strategies and shows only signals that pass your confidence threshold. "
        "Signals from this scanner can be auto-executed by assigning them in the **Strategy** page.",
        icon="ℹ️",
    )

    sc1, sc2, sc3 = st.columns([2, 1, 1])
    scan_direction  = sc1.selectbox("Signal direction", ["BUY", "SELL", "ALL"], index=0, key="scan_dir")
    scan_min_conf   = sc2.slider("Min confidence", 0.0, 1.0, 0.55, 0.05, key="scan_conf")
    scan_strategies = sc3.multiselect(
        "Strategies (blank = all)", options=_STRATEGY_NAMES, default=[], key="scan_strats",
    )

    st.markdown("**Tickers to scan** — enter comma-separated or one per line:")
    scan_universe_input = st.text_area(
        "Tickers",
        placeholder="AAPL\nMSFT\nNVDA\nTSLA\nSPY,QQQ,AMD",
        height=140,
        key="scan_universe",
        label_visibility="collapsed",
    )

    sc_run = st.button("▶ Run Scan", type="primary", key="scan_run")

    if sc_run:
        raw_tickers = [t.strip().upper() for t in scan_universe_input.replace("\n", ",").split(",") if t.strip()]
        raw_tickers = list(dict.fromkeys(raw_tickers))
        if not raw_tickers:
            st.error("No tickers entered. Paste at least one symbol above.")
        elif len(raw_tickers) > 500:
            st.error("Maximum 500 symbols per scan.")
        else:
            strat_filter = ",".join(scan_strategies) if scan_strategies else None
            with st.spinner(f"Fetching Schwab live prices and scanning {len(raw_tickers)} symbols…"):
                try:
                    scan_result = api.perplexity_scan(
                        symbols=raw_tickers,
                        direction=scan_direction,
                        min_confidence=scan_min_conf,
                        strategies=strat_filter,
                        max_workers=10,
                    )
                    st.session_state["px_scan_result"] = scan_result
                    st.session_state["px_scan_tickers"] = raw_tickers
                except Exception as e:
                    st.error(f"Scan failed: {e}")

    scan_data = st.session_state.get("px_scan_result")
    if scan_data:
        hits    = scan_data.get("results", [])
        scanned = scan_data.get("scanned", 0)
        schwab_loaded = scan_data.get("schwab_prices_loaded", 0)

        sc_c1, sc_c2, sc_c3, sc_c4, sc_c5 = st.columns(5)
        sc_c1.metric("Symbols scanned", scanned)
        sc_c2.metric("Signals found",   len(hits))
        sc_c3.metric("BUY signals",  sum(1 for h in hits if h["direction"] == "BUY"))
        sc_c4.metric("SELL signals", sum(1 for h in hits if h["direction"] == "SELL"))
        sc_c5.metric("Schwab live prices", schwab_loaded,
                     delta="yfinance fallback" if schwab_loaded < scanned else "all live",
                     delta_color="off")

        if not hits:
            st.info("No signals matched your filters. Try lowering min confidence, adding more tickers, or selecting ALL direction.")
        else:
            rows = []
            for h in hits:
                rr = None
                if h["entry_price"] and h["stop_price"] and h["target_price"]:
                    risk   = abs(h["entry_price"] - h["stop_price"])
                    reward = abs(h["target_price"] - h["entry_price"])
                    rr = round(reward / risk, 1) if risk > 0 else None
                price_src = "🟢 Schwab" if h.get("price_source") == "schwab" else "⚪ yfinance"
                rows.append({
                    "Symbol":      h["symbol"],
                    "Strategy":    h["strategy"],
                    "Signal":      h["direction"],
                    "Confidence":  f"{h['confidence']:.0%}",
                    "Live Price":  f"${h['live_price']:,.2f}" if h.get("live_price") else "—",
                    "Entry":       f"${h['entry_price']:,.2f}" if h["entry_price"] else "—",
                    "Stop":        f"${h['stop_price']:,.2f}"  if h["stop_price"]   else "—",
                    "Target":      f"${h['target_price']:,.2f}" if h["target_price"] else "—",
                    "R:R":         f"{rr:.1f}" if rr else "—",
                    "Price Src":   price_src,
                    "Regime":      h.get("regime", "").replace("_", " ").title(),
                    "Reason":      h.get("reason", ""),
                })
            df_hits = pd.DataFrame(rows)

            def _color_row(row):
                if row["Signal"] == "BUY":
                    return ["background-color: rgba(0,212,170,0.08)"] * len(row)
                if row["Signal"] == "SELL":
                    return ["background-color: rgba(255,75,75,0.08)"] * len(row)
                return [""] * len(row)

            st.dataframe(df_hits.style.apply(_color_row, axis=1),
                         use_container_width=True, hide_index=True)

            # Detail cards for BUY signals
            buy_hits_list = [h for h in hits if h["direction"] == "BUY"]
            if buy_hits_list:
                st.markdown("---")
                st.markdown("### BUY Signal Details")
                for h in buy_hits_list:
                    ps = h.get("position_size") or {}
                    price_badge = "🟢 Schwab live" if h.get("price_source") == "schwab" else "⚪ yfinance close"
                    with st.expander(
                        f"🟢 **{h['symbol']}** — {h['strategy']} | conf {h['confidence']:.0%} | {price_badge}",
                        expanded=False,
                    ):
                        if h.get("live_price"):
                            st.metric("Live Price (Schwab)", f"${h['live_price']:,.2f}")
                        d1, d2, d3 = st.columns(3)
                        if h["entry_price"]:
                            d1.metric("Entry", f"${h['entry_price']:,.2f}")
                        if h["stop_price"] and h["entry_price"]:
                            stop_pct = (h["entry_price"] - h["stop_price"]) / h["entry_price"] * 100
                            d2.metric("Stop", f"${h['stop_price']:,.2f}",
                                      delta=f"-{stop_pct:.1f}%", delta_color="inverse")
                        if h["target_price"] and h["entry_price"]:
                            tgt_pct = (h["target_price"] - h["entry_price"]) / h["entry_price"] * 100
                            d3.metric("Target", f"${h['target_price']:,.2f}", delta=f"+{tgt_pct:.1f}%")
                        st.info(f"**Signal reason:** {h['reason']}")
                        if ps and ps.get("viable"):
                            p1, p2, p3 = st.columns(3)
                            p1.metric("Shares",         f"{ps.get('shares', 0):,.0f}")
                            p2.metric("Position Value", f"${ps.get('position_value', 0):,.0f}")
                            p3.metric("Risk Amount",    f"${ps.get('risk_amount', 0):,.2f}")
                        st.caption(
                            f"Strategy: {h['strategy']} | Regime: {h.get('regime','').replace('_',' ').title()} "
                            f"| Vol bucket: {h.get('volatility_bucket','')} | Price: {price_badge}"
                        )
                        st.caption(
                            "To auto-trade this signal, go to the **Strategy** page → Assignments "
                            f"and assign **{h['strategy']}** to **{h['symbol']}** with the Perplexity system."
                        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — POSITION SIZER
# ══════════════════════════════════════════════════════════════
with tab_sizer:
    st.subheader("📐 Position Sizer")
    st.caption(
        "Enter a symbol to auto-load the current price and ATR-based stop suggestions. "
        "The system calculates shares so you risk a fixed % of your account — no guessing."
    )

    with st.expander("ℹ️ How position sizing works", expanded=False):
        st.markdown("""
**The formula:**
```
Risk per trade  = Account Value × Risk % (e.g. $10,000 × 1% = $100)
Stop distance   = Entry price − Stop price
Shares          = Risk per trade ÷ Stop distance
```

**Example:**
- Account: $10,000  |  Risk: 1% = $100
- Entry: $50  |  Stop: $47.50  →  Stop distance = $2.50
- Shares = $100 ÷ $2.50 = **40 shares**  |  Position = $2,000

**What is ATR?**
ATR (Average True Range) measures how much a stock normally moves in a day.
Using 2×ATR as your stop means the stock has to move unusually far against you before stopping out.
It's the standard way professional traders set stops — no guesswork needed.

No matter how volatile the stock, you always risk the same dollar amount.
If the stop is hit you lose ~1% of your account. If the target is hit you typically gain 2–3×.
        """)

    st.divider()

    # ── Step 1: Symbol + auto-load ────────────────────────────
    st.markdown("**Step 1 — Enter symbol and load current price**")
    sz_col1, sz_col2 = st.columns([2, 1])
    with sz_col1:
        sz_symbol = st.text_input("Symbol", value="AAPL", key="sz_sym",
                                  help="Type any US stock ticker.").upper().strip()
    with sz_col2:
        st.write("")
        st.write("")
        load_atr = st.button("🔄 Load Price & ATR Stops", use_container_width=True, key="sz_load")

    if load_atr and sz_symbol:
        with st.spinner(f"Fetching ATR data for {sz_symbol}..."):
            try:
                atr_data = api.get_atr_stops(sz_symbol)
                st.session_state["sz_atr"] = atr_data
                # Pre-fill entry with current price
                st.session_state["sz_entry_val"] = atr_data["current_price"]
            except Exception as e:
                st.error(f"Could not load data: {e}")

    atr = st.session_state.get("sz_atr")
    if atr and atr.get("symbol") == sz_symbol:
        st.info(
            f"**{sz_symbol}** current price: **${atr['current_price']:,.2f}**  |  "
            f"ATR(14): **${atr['atr14']:,.2f}**"
        )
        st.markdown("**Suggested stops based on ATR — pick one:**")
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button(
                f"1× ATR stop: ${atr['stop_1x_atr']:,.2f}  (-{atr['stop_pct_1x']:.1f}%)\n"
                f"Tighter — good for low-volatility stocks",
                key="atr1x", use_container_width=True
            ):
                st.session_state["sz_stop_val"] = atr["stop_1x_atr"]
        with a2:
            if st.button(
                f"1.5× ATR stop: ${atr['stop_1_5x_atr']:,.2f}  (-{atr['stop_pct_1_5x']:.1f}%)\n"
                f"Balanced — most common choice",
                key="atr15x", use_container_width=True, type="primary"
            ):
                st.session_state["sz_stop_val"] = atr["stop_1_5x_atr"]
        with a3:
            if st.button(
                f"2× ATR stop: ${atr['stop_2x_atr']:,.2f}  (-{atr['stop_pct_2x']:.1f}%)\n"
                f"Wider — used by Perplexity strategies",
                key="atr2x", use_container_width=True
            ):
                st.session_state["sz_stop_val"] = atr["stop_2x_atr"]
        st.caption("Clicking a button above fills the Stop Price field below automatically.")

    st.divider()

    # ── Step 2: Trade details ─────────────────────────────────
    st.markdown("**Step 2 — Confirm prices**")
    col1, col2 = st.columns([1, 1])
    with col1:
        default_entry = st.session_state.get("sz_entry_val",
                        atr["current_price"] if atr and atr.get("symbol") == sz_symbol else 100.0)
        sz_entry = st.number_input("Entry price ($)", value=float(default_entry),
                                   min_value=0.01, step=1.0, format="%.2f", key="sz_entry")

        default_stop = st.session_state.get("sz_stop_val",
                       atr["stop_2x_atr"] if atr and atr.get("symbol") == sz_symbol else float(default_entry) * 0.95)
        sz_stop = st.number_input("Stop price ($)", value=float(default_stop),
                                  min_value=0.01, step=0.5, format="%.2f", key="sz_stop",
                                  help="Price where you exit if the trade goes wrong. Use ATR buttons above to auto-fill.")

        sz_target = st.number_input(
            "Target price ($ optional)", value=0.0, min_value=0.0, step=1.0, format="%.2f",
            help="Your profit target. Leave at 0 to skip R:R calculation."
        )

    with col2:
        st.markdown("**Account settings**")
        sz_account = st.number_input("Account value ($)", value=10000, min_value=1000, step=1000,
                                     help="Total cash in your trading account.")
        sz_risk = st.slider("Risk per trade (%)", min_value=0.25, max_value=3.0,
                            value=1.0, step=0.25,
                            help="% of account you are willing to lose if stop is hit. "
                                 "Recommended: 1% when starting out. Never exceed 2%.")
        sz_maxpos = st.number_input(
            "Max position size ($)", value=2000, min_value=100, step=500,
            help="Hard cap — position will never exceed this value even if the formula says more. "
                 "Set to your per-stock budget."
        )
        st.caption(
            f"Max loss on this trade: **${sz_account * sz_risk / 100:,.2f}** "
            f"({sz_risk:.2f}% of ${sz_account:,})"
        )

    calc = st.button("📐 Calculate Position Size", type="primary", key="sz_calc")

    if calc:
        if sz_stop >= sz_entry:
            st.error("Stop price must be below entry price.")
        else:
            try:
                result = api.perplexity_size(
                    sz_symbol, sz_entry, sz_stop,
                    account_value=sz_account, risk_pct=sz_risk / 100,
                    max_position_size_usd=sz_maxpos,
                )
                st.session_state["sz_result"] = {
                    **result, "_target": sz_target, "_account": sz_account, "_maxpos": sz_maxpos
                }
            except Exception as e:
                st.error(f"Failed: {e}")

    r = st.session_state.get("sz_result")
    if r:
        st.divider()
        if not r["viable"]:
            st.error(f"❌ Cannot size this trade: {r['skip_reason']}")
            if "too wide" in r.get("skip_reason", ""):
                st.info("Tip: your stop is more than 20% below entry — try using a tighter stop (1× or 1.5× ATR).")
            elif "too tight" in r.get("skip_reason", ""):
                st.info("Tip: your stop is less than 0.1% below entry — it will almost certainly get hit immediately. Use a wider stop.")
        else:
            target = r.get("_target", 0)
            account = r.get("_account", sz_account)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Shares to Buy",  f"{r['shares']:,.4f}")
            m2.metric("Position Value", f"${r['position_value']:,.2f}",
                      delta=f"{r['position_value']/account*100:.1f}% of account",
                      delta_color="off")
            m3.metric("Max Loss (if stop hit)", f"${r['risk_amount']:,.2f}",
                      delta=f"{r['risk_pct_of_account']:.2f}% of account",
                      delta_color="inverse")
            if target and target > r["entry_price"]:
                reward = (target - r["entry_price"]) * r["shares"]
                rr = (target - r["entry_price"]) / r["stop_distance"]
                m4.metric("Max Gain (if target hit)", f"${reward:,.2f}",
                          delta=f"+{reward/account*100:.2f}% of account")
                m5.metric("Risk : Reward", f"1 : {rr:.1f}",
                          delta="Good" if rr >= 2 else "Low — consider wider target",
                          delta_color="normal" if rr >= 2 else "inverse")
            else:
                m4.metric("Max Gain", "—")
                m5.metric("Risk : Reward", "—")

            if r["capped"]:
                st.warning(f"⚠️ Position capped: {r['cap_reason']}")

            st.divider()
            st.markdown("**Full breakdown**")
            breakdown = {
                "Entry price":    f"${r['entry_price']:,.2f}",
                "Stop price":     f"${r['stop_price']:,.2f}",
                "Stop distance":  f"${r['stop_distance']:,.2f}  ({r['stop_distance_pct']:.2f}% below entry)",
                "Account value":  f"${account:,.2f}",
                "Risk %":         f"{sz_risk:.2f}%",
                "$ at risk":      f"${r['risk_amount']:,.2f}  ({r['risk_pct_of_account']:.2f}% of account)",
                "Shares":         f"{r['shares']:,.4f}",
                "Position value": f"${r['position_value']:,.2f}",
                "Max position cap": f"${r.get('_maxpos', sz_maxpos):,}",
            }
            if target and target > r["entry_price"]:
                reward = (target - r["entry_price"]) * r["shares"]
                breakdown["Target price"]  = f"${target:,.2f}"
                breakdown["Expected gain"] = f"${reward:,.2f}"
                breakdown["R:R ratio"]     = f"1 : {(target - r['entry_price']) / r['stop_distance']:.1f}"

            tbl = pd.DataFrame(list(breakdown.items()), columns=["Field", "Value"])
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            st.success(
                f"**Order:** BUY **{r['shares']:,.4f}** shares of **{r['symbol']}** @ market  |  "
                f"Set stop loss @ **${r['stop_price']:,.2f}**"
                + (f"  |  Target @ **${target:,.2f}**" if target and target > r["entry_price"] else "")
            )


# ══════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST SINGLE STRATEGY
# ══════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Backtest a Single Strategy")

    try:
        strat_list = api.perplexity_strategies()
        strat_names = [s["name"] for s in strat_list]
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        chosen_strat = st.selectbox("Strategy", strat_names, key="px_strat")
    with col2:
        bt_symbol = st.text_input("Symbol", value="AAPL", key="px_sym",
                                  help="Type any US stock ticker.").upper().strip()
    with col3:
        bt_period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=2, key="px_period")
    with col4:
        bt_capital = st.number_input("Capital ($)", value=10000, min_value=1000, step=1000, key="px_cap")

    sz_col1, sz_col2 = st.columns([2, 3])
    with sz_col1:
        bt_sizing_mode = st.radio(
            "Position sizing",
            ["Risk-based (1% risk/trade)", "Fixed % of equity per trade"],
            key="px_sizing_mode", horizontal=True,
        )
    with sz_col2:
        if bt_sizing_mode.startswith("Fixed"):
            bt_pos_pct = st.slider("Position size (% of equity)", 5, 95, 20, 5, key="px_pos_pct") / 100
            st.caption(f"Each trade deploys **{bt_pos_pct*100:.0f}%** of current equity → "
                       f"**${bt_capital * bt_pos_pct:,.0f}** at start")
        else:
            bt_pos_pct = 0.0
            st.caption("Sizes each trade to risk **1%** of account on the stop distance. "
                       "Conservative — most capital stays idle.")

    st.caption(STRATEGY_DESCRIPTIONS.get(chosen_strat, ""))

    run_bt = st.button("▶ Run Backtest", type="primary", key="px_run_bt")

    if run_bt:
        with st.spinner(f"Backtesting {chosen_strat} on {bt_symbol} over {bt_period}..."):
            try:
                r = api.perplexity_backtest(chosen_strat, bt_symbol, period=bt_period,
                                            initial_capital=bt_capital, position_pct=bt_pos_pct,
                                            breakdown=True)
                st.session_state["px_bt_result"] = r
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    r = st.session_state.get("px_bt_result")
    if r and not r.get("error"):
        st.divider()
        st.subheader(f"{r['strategy_name']} on {r['symbol']}  —  {r['start_date']} → {r['end_date']}")

        pnl = r["total_pnl"]
        cap_emp = r.get("capital_employed", 0)
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric("Total P&L",    f"${pnl:,.2f}")
        c2.metric("Total Return", f"{r['total_return_pct']:+.2f}%",
                  help="(Final equity ÷ Initial equity) − 1, based on the full equity curve")
        c3.metric("CAGR",         f"{r.get('cagr', 0):+.2f}%",
                  help="Compound annual growth rate (252 trading days/year)")
        c4.metric("Win Rate",     f"{r['win_rate_pct']:.1f}%",
                  delta=f"{r['winning_trades']}W / {r['losing_trades']}L")
        c5.metric("Profit Factor", r["profit_factor"] if r["profit_factor"] else "—")
        c6.metric("Max Drawdown", f"{r['max_drawdown_pct']:.1f}%", delta_color="inverse")
        c7.metric("Sharpe",       r["sharpe_ratio"] if r["sharpe_ratio"] else "—")
        c8.metric("Trades",       r["total_trades"])
        c9, c10, c11, c12 = st.columns([1, 1, 1, 1])
        c9.metric("Avg Win",      f"{r.get('avg_win_pct', 0):+.2f}%")
        c10.metric("Avg Loss",    f"{r.get('avg_loss_pct', 0):+.2f}%")
        c11.metric("Expectancy",  f"{r.get('expectancy_pct', 0):+.2f}%")
        c12.metric("Avg Hold",    f"{r.get('average_holding_days', 0):.1f} bars")

        _plot_equity_curve(r)

        # Trade log
        if r.get("trades"):
            st.divider()
            st.subheader(f"Trade Log ({r['total_trades']} trades)")
            rows = []
            cumulative_pnl = 0.0
            for t in r["trades"]:
                pnl_val = t.get("pnl")
                if pnl_val is not None:
                    cumulative_pnl += pnl_val
                rows.append({
                    "date":           t["date"],
                    "side":           "🟢 BUY" if t["side"] == "BUY" else "🔴 " + t["side"],
                    "price":          f"${t['price']:,.2f}",
                    "qty":            round(t["quantity"], 4),
                    "value":          f"${t['value']:,.2f}",
                    "stop":           f"${t['stop']:,.2f}" if t.get("stop") else "—",
                    "target":         f"${t['target']:,.2f}" if t.get("target") else "—",
                    "P&L":            (f"🟢 +${pnl_val:,.2f}" if pnl_val > 0 else f"🔴 -${abs(pnl_val):,.2f}")
                                      if pnl_val is not None else "—",
                    "cumulative P&L": (f"🟢 +${cumulative_pnl:,.2f}" if cumulative_pnl >= 0 else f"🔴 -${abs(cumulative_pnl):,.2f}")
                                      if pnl_val is not None else "—",
                    "$ risked":       f"${t['risk_usd']:,.2f}" if t.get("risk_usd") else "—",
                    "reason":         t.get("reason", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if r.get("breakdown"):
            st.divider()
            with st.expander("📊 Performance Breakdown", expanded=False):
                _breakdown_tables(r["breakdown"])

        st.divider()
        st.info(
            f"**Want to understand why trades win or lose?**  "
            f"Go to the **🔍 Analysis** tab — enter **{bt_symbol}** and **{chosen_strat}**, "
            f"run a full winner/loser breakdown, then hit **Optimize Filters → Save Profile** "
            f"to automatically tighten entry conditions."
        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — COMPARE ALL STRATEGIES ON ONE SYMBOL
# ══════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Compare All 5 Strategies")
    st.caption("Runs all strategies on the same symbol and period — easy to see which works best.")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        cmp_symbol = st.text_input("Symbol", value="AAPL", key="cmp_sym",
                                   help="Type any US stock ticker.").upper().strip()
    with col2:
        cmp_period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y"], index=2, key="cmp_period")
    with col3:
        cmp_capital = st.number_input("Capital ($)", value=10000, min_value=1000,
                                       step=1000, key="cmp_cap")

    cmp_sz1, cmp_sz2 = st.columns([2, 3])
    with cmp_sz1:
        cmp_sizing_mode = st.radio("Position sizing", ["Risk-based", "Fixed % of equity"],
                                   key="cmp_sizing_mode", horizontal=True)
    with cmp_sz2:
        if cmp_sizing_mode.startswith("Fixed"):
            cmp_pos_pct = st.slider("Position size (%)", 5, 95, 20, 5, key="cmp_pos_pct") / 100
        else:
            cmp_pos_pct = 0.0
            st.caption("1% risk per trade on stop distance.")

    cmp_btn1, cmp_btn2 = st.columns(2)
    with cmp_btn1:
        run_cmp = st.button("▶ Compare Backtests", type="primary", key="px_run_cmp", use_container_width=True)
    with cmp_btn2:
        run_wf_cmp = st.button("🔀 Compare Walk-Forward (all strategies)", key="px_run_wf_cmp",
                                use_container_width=True,
                                help="Runs rolling walk-forward for all 5 strategies on this symbol. "
                                     "Shows which strategy has real edge right now vs which is overfit.")

    if run_cmp:
        with st.spinner(f"Running all 5 strategies on {cmp_symbol} over {cmp_period}..."):
            try:
                results = api.perplexity_backtest_all(cmp_symbol, period=cmp_period,
                                                      initial_capital=cmp_capital,
                                                      position_pct=cmp_pos_pct)
                st.session_state["px_compare"] = results
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    if run_wf_cmp:
        with st.spinner(f"Running walk-forward for all 5 strategies on {cmp_symbol} (10y, 3y IS / 1y OOS)... this takes ~2 min"):
            try:
                wf_results = api.perplexity_walkforward_all(cmp_symbol, period="10y",
                                                             train_years=3.0, test_years=1.0, step_years=1.0,
                                                             initial_capital=cmp_capital,
                                                             position_pct=cmp_pos_pct)
                st.session_state["px_wf_compare"] = {"symbol": cmp_symbol, "data": wf_results}
            except Exception as e:
                st.error(f"Walk-forward comparison failed: {e}")

    cmp = st.session_state.get("px_compare")
    if cmp:
        st.divider()

        # Require at least 8 trades for statistical reliability.
        # 5 trades is too small a sample to trust profit factor or win rate.
        valid = [r for r in cmp if not r.get("error") and r["total_trades"] >= 8]

        # ── Best Strategy Recommendation ─────────────────────
        if valid:
            # Rank-based scoring: each metric is ranked 1..N independently,
            # so no single metric dominates due to scale differences.
            # Weights: Return 40% | Profit Factor 25% | Win Rate 20% | Sharpe 10% | Trades 5%
            # Trades weight ensures a 12-trade strategy beats a 5-trade one when
            # other metrics are close — more trades = more statistically reliable.
            def _ranked_score(strategies):
                metrics = {
                    "total_return_pct": 0.40,
                    "profit_factor":    0.25,
                    "win_rate_pct":     0.20,
                    "sharpe_ratio":     0.10,
                    "total_trades":     0.05,
                }
                scores = {r["strategy_name"]: 0.0 for r in strategies}
                n = len(strategies)
                for metric, weight in metrics.items():
                    # profit_factor None = no losing trades = best possible → rank 1st
                    def _val(r, m=metric):
                        v = r.get(m)
                        return float("inf") if v is None else float(v)
                    ranked = sorted(strategies, key=_val, reverse=True)
                    for rank, r in enumerate(ranked, start=1):
                        # rank 1 = best → highest score = (n - rank + 1) / n
                        scores[r["strategy_name"]] += weight * (n - rank + 1) / n
                return scores

            profitable = [r for r in valid if r["total_return_pct"] > 0]
            pool = profitable if profitable else valid
            scores = _ranked_score(pool)
            best = max(pool, key=lambda r: scores[r["strategy_name"]])

            b_pf = best["profit_factor"] if best["profit_factor"] else 0
            b_ret = best["total_return_pct"]
            b_wr = best["win_rate_pct"]
            color = "green" if b_ret > 0 else "orange"

            st.success(
                f"**Recommended strategy for {cmp_symbol}: {best['strategy_name'].replace('_', ' ')}**  \n"
                f"Win Rate: **{b_wr:.1f}%** | Profit Factor: **{b_pf:.2f}** | "
                f"Return over {cmp_period}: **{b_ret:+.2f}%** | Trades: **{best['total_trades']}**  \n"
                f"Use this strategy on the Live Signals tab or Backtest tab for {cmp_symbol}."
            )
            st.divider()

        rows = []
        for r in cmp:
            if r.get("error"):
                rows.append({"Strategy": r["strategy_name"], "Trades": "—", "Win Rate": "—",
                             "Profit Factor": "—", "Avg Win": "—", "Expectancy": "—",
                             "Total Return": "ERROR", "CAGR": "—", "Total P&L": r["error"],
                             "Max Drawdown": "—", "Sharpe": "—"})
                continue
            pnl = r["total_pnl"]
            is_best = valid and r["strategy_name"] == best["strategy_name"]
            rows.append({
                "Strategy":      ("⭐ " if is_best else "") + r["strategy_name"],
                "Trades":        r["total_trades"],
                "Win Rate":      f"{r['win_rate_pct']:.1f}%",
                "Profit Factor": r["profit_factor"] if r["profit_factor"] else "—",
                "Avg Win":       f"{r.get('avg_win_pct', 0):+.2f}%",
                "Expectancy":    f"{r.get('expectancy_pct', 0):+.2f}%",
                "Total Return":  f"{r['total_return_pct']:+.2f}%",
                "CAGR":          f"{r.get('cagr', 0):+.2f}%",
                "Total P&L":     f"+${pnl:,.0f}" if pnl >= 0 else f"-${abs(pnl):,.0f}",
                "Max Drawdown":  f"{r['max_drawdown_pct']:.1f}%",
                "Sharpe":        r["sharpe_ratio"] if r["sharpe_ratio"] else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Bar chart of returns
        if valid:
            best_name = best["strategy_name"]
            fig = go.Figure(go.Bar(
                x=[r["strategy_name"].replace("_", " ") for r in valid],
                y=[r["total_return_pct"] for r in valid],
                marker_color=[
                    "#FFD700" if r["strategy_name"] == best_name
                    else ("#00d4aa" if r["total_return_pct"] >= 0 else "#ff4b4b")
                    for r in valid
                ],
                text=[("⭐ " if r["strategy_name"] == best_name else "") + f"{r['total_return_pct']:+.1f}%"
                      for r in valid],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Total Return — {cmp_symbol} ({cmp_period})  |  Gold = Recommended",
                height=350, template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Return (%)",
            )
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Walk-Forward Comparison results ──────────────────────
    wf_cmp = st.session_state.get("px_wf_compare")
    if wf_cmp:
        st.divider()
        st.subheader(f"Walk-Forward Strategy Comparison — {wf_cmp['symbol']}  |  10y  |  3y IS / 1y OOS")
        st.caption(
            "**How to read this:** Latest IS CAGR = what the strategy earned in its most recent training window. "
            "Latest OOS CAGR = how it performed in the immediately following held-out period. "
            "Prev OOS CAGR = the period before that. "
            "A strategy is **currently robust** if Latest IS + OOS are both positive and Latest WFE ≥ 0.5."
        )

        wf_data = [r for r in wf_cmp["data"] if not r.get("error")]
        wf_errors = [r for r in wf_cmp["data"] if r.get("error")]

        for e in wf_errors:
            st.warning(f"⚠️ {e['strategy_name']}: {e['error']}")

        if wf_data:
            def _wfe_badge(wfe, label):
                if wfe is None:
                    return "N/A"
                icon = "🟢" if wfe >= 0.7 else ("🟡" if wfe >= 0.5 else "🔴")
                return f"{icon} {wfe:.2f}× ({label})"

            def _cagr_fmt(v):
                if v is None: return "—"
                return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"

            # Sort by latest OOS CAGR descending — "what's working NOW"
            wf_sorted = sorted(wf_data, key=lambda r: r["latest_oos_cagr"], reverse=True)
            best_now = wf_sorted[0]["strategy_name"] if wf_sorted else None

            # Best strategy callout
            if best_now:
                b = wf_sorted[0]
                b_latest_wfe = b["latest_wfe"]
                verdict = "strong" if (b_latest_wfe or 0) >= 0.7 else ("moderate" if (b_latest_wfe or 0) >= 0.5 else "weak")
                st.success(
                    f"**Best strategy right now on {wf_cmp['symbol']}: "
                    f"{best_now.replace('_', ' ')}**  \n"
                    f"Latest OOS CAGR: **{_cagr_fmt(b['latest_oos_cagr'])}**  |  "
                    f"Latest WFE: **{_wfe_badge(b['latest_wfe'], b['latest_wfe_label'])}**  |  "
                    f"Global OOS CAGR: **{_cagr_fmt(b['global_oos_cagr'])}**  \n"
                    f"OOS edge: **{verdict}** — "
                    + ("confident to trade live." if verdict == "strong"
                       else "monitor closely before trading live." if verdict == "moderate"
                       else "be cautious — possible overfit in recent window.")
                )

            # Comparison table
            tbl = []
            for r in wf_sorted:
                is_best = r["strategy_name"] == best_now
                tbl.append({
                    "Strategy": ("⭐ " if is_best else "") + r["strategy_name"].replace("_", " "),
                    "Windows": r["n_windows"],
                    "Latest IS CAGR": _cagr_fmt(r["latest_is_cagr"]),
                    "Latest OOS CAGR": _cagr_fmt(r["latest_oos_cagr"]),
                    "Latest WFE": _wfe_badge(r["latest_wfe"], r["latest_wfe_label"]),
                    "Prev OOS CAGR": _cagr_fmt(r["prev_oos_cagr"]),
                    "Global OOS CAGR": _cagr_fmt(r["global_oos_cagr"]),
                    "Global WFE": _wfe_badge(r["global_wfe"], r["global_wfe_label"]),
                    "Avg IS CAGR": _cagr_fmt(r["avg_is_cagr"]),
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

            # Visual: Latest OOS CAGR bar chart
            fig_wf = go.Figure(go.Bar(
                x=[r["strategy_name"].replace("_", " ") for r in wf_sorted],
                y=[r["latest_oos_cagr"] for r in wf_sorted],
                marker_color=[
                    "#FFD700" if r["strategy_name"] == best_now
                    else ("#00d4aa" if r["latest_oos_cagr"] >= 0 else "#ff4b4b")
                    for r in wf_sorted
                ],
                text=[_cagr_fmt(r["latest_oos_cagr"]) for r in wf_sorted],
                textposition="outside",
            ))
            fig_wf.update_layout(
                title=f"Latest OOS CAGR by Strategy — {wf_cmp['symbol']}  (Gold = best current edge)",
                height=320, template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="OOS CAGR (%)",
            )
            fig_wf.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dash")
            fig_wf.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_wf, use_container_width=True)

            # Global WFE bar chart
            fig_gwfe = go.Figure(go.Bar(
                x=[r["strategy_name"].replace("_", " ") for r in wf_data],
                y=[r["global_wfe"] if r["global_wfe"] is not None else 0 for r in wf_data],
                marker_color=[
                    "#00d4aa" if (r["global_wfe"] or 0) >= 0.7
                    else "#FFD700" if (r["global_wfe"] or 0) >= 0.5
                    else "#ff4b4b"
                    for r in wf_data
                ],
                text=[_wfe_badge(r["global_wfe"], r["global_wfe_label"]) for r in wf_data],
                textposition="outside",
            ))
            fig_gwfe.add_hline(y=0.7, line_dash="dash", line_color="#00d4aa",
                               annotation_text="0.70 threshold")
            fig_gwfe.update_layout(
                title="Global WFE by Strategy  (green ≥ 0.70 | yellow 0.50–0.70 | red < 0.50)",
                height=280, template="plotly_dark",
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Global WFE",
            )
            fig_gwfe.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_gwfe, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — PORTFOLIO BACKTEST
# ══════════════════════════════════════════════════════════════
with tab_portfolio:
    st.subheader("Portfolio Backtest")
    st.caption(
        "Run one strategy across multiple symbols simultaneously with a shared capital pool. "
        "Positions are sized as a fixed % of portfolio equity — capital stays deployed."
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        port_strat = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()), key="port_strat")
    with pc2:
        port_period = st.selectbox("Period", ["2y", "5y", "10y"], index=1, key="port_period")
    with pc3:
        port_capital = st.number_input("Capital ($)", min_value=1000, value=10000, step=1000, key="port_cap")

    pc4, pc5, pc6 = st.columns(3)
    with pc4:
        port_pos_pct = st.slider("Position size (% of equity per trade)", 5, 50, 20, 5, key="port_pos_pct") / 100
    with pc5:
        port_max_pos = st.slider("Max simultaneous positions", 1, 10, 5, 1, key="port_max_pos")
    with pc6:
        port_symbols_input = st.text_input("Symbols (comma-separated)", value="SPY,QQQ,IWM,AAPL,NVDA", key="port_syms")

    if st.button("▶ Run Portfolio Backtest", type="primary", key="port_run"):
        with st.spinner(f"Running {port_strat} across portfolio..."):
            try:
                r = api.perplexity_portfolio(port_strat, symbols=port_symbols_input,
                                             period=port_period, initial_capital=port_capital,
                                             position_pct=port_pos_pct,
                                             max_open_positions=port_max_pos)
                st.session_state["px_portfolio"] = r
            except Exception as e:
                st.error(f"Portfolio backtest failed: {e}")

    pr = st.session_state.get("px_portfolio")
    if pr and not pr.get("error"):
        st.divider()
        st.subheader(f"{pr['strategy_name']} — {', '.join(pr['symbols'])}  |  {pr['start_date']} → {pr['end_date']}")

        pnl = pr["total_pnl"]
        pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8 = st.columns(8)
        pp1.metric("Total P&L",     f"${pnl:,.0f}")
        pp2.metric("Total Return",  f"{pr['total_return_pct']:+.2f}%")
        pp3.metric("CAGR",          f"{pr.get('cagr', 0):+.2f}%")
        pp4.metric("Win Rate",      f"{pr['win_rate_pct']:.1f}%",
                   delta=f"{pr['winning_trades']}W / {pr['losing_trades']}L")
        pp5.metric("Profit Factor", pr["profit_factor"] if pr["profit_factor"] else "—")
        pp6.metric("Max Drawdown",  f"{pr['max_drawdown_pct']:.1f}%", delta_color="inverse")
        pp7.metric("Trades",        pr["total_trades"])
        pp8.metric("Capital Util.", f"{pr.get('capital_utilisation_pct', 0):.1f}%",
                   help="Average % of capital deployed across all days")

        _plot_equity_curve(pr, height=300, show_markers=False)

        # Trades table
        if pr.get("trades"):
            st.subheader("Trade Log")
            sell_only = [t for t in pr["trades"] if "SELL" in t["side"] and t.get("pnl") is not None]
            if sell_only:
                df_trades = pd.DataFrame(sell_only)[["date", "symbol", "side", "price", "quantity", "value", "pnl", "reason"]]
                df_trades["pnl"] = df_trades["pnl"].apply(lambda x: f"+${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}")
                st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════
with tab_walkforward:
    st.subheader("Walk-Forward Validation")
    st.caption(
        "Tests whether a strategy's in-sample edge holds out-of-sample. "
        "**Simple**: one 70/30 split. **Rolling**: multiple IS/OOS windows slide across full history — "
        "OOS curves are stitched into one composite. "
        "**WFE** (Walk-Forward Efficiency) = OOS CAGR ÷ IS CAGR. "
        "≥0.7 = acceptable  |  0.5–0.7 = investigate  |  <0.5 = likely overfit."
    )

    # ── Controls ──────────────────────────────────────────────
    wf_c1, wf_c2, wf_c3 = st.columns(3)
    with wf_c1:
        wf_strat = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()), key="wf_strat")
    with wf_c2:
        wf_sym = st.text_input("Symbol", value="SPY", key="wf_sym").upper().strip()
    with wf_c3:
        wf_period = st.selectbox("Full Period", ["5y", "10y"], index=1, key="wf_period")

    wf_mode = st.radio("Mode", ["Simple 70/30 Split", "Rolling Walk-Forward"],
                       key="wf_mode", horizontal=True)

    if wf_mode == "Simple 70/30 Split":
        wf_m1, wf_m2, wf_m3 = st.columns(3)
        with wf_m1:
            wf_train_pct = st.slider("Train split (%)", 50, 80, 70, 5, key="wf_train_pct") / 100
        with wf_m2:
            wf_capital = st.number_input("Capital ($)", min_value=1000, value=10000,
                                          step=1000, key="wf_cap")
        with wf_m3:
            wf_pos_pct = st.slider("Position size (% equity, 0=risk-based)",
                                    0, 50, 0, 5, key="wf_pos_pct") / 100
    else:
        wf_r1, wf_r2, wf_r3, wf_r4, wf_r5 = st.columns(5)
        with wf_r1:
            wf_train_years = st.number_input("Train window (years)", 1.0, 8.0, 3.0, 0.5,
                                              key="wf_train_y")
        with wf_r2:
            wf_test_years = st.number_input("Test window (years)", 0.5, 3.0, 1.0, 0.5,
                                             key="wf_test_y")
        with wf_r3:
            wf_step_years = st.number_input("Step size (years)", 0.5, 2.0, 1.0, 0.5,
                                             key="wf_step_y")
        with wf_r4:
            wf_capital = st.number_input("Capital ($)", min_value=1000, value=10000,
                                          step=1000, key="wf_cap_r")
        with wf_r5:
            wf_pos_pct = st.slider("Position size (% equity, 0=risk-based)",
                                    0, 50, 0, 5, key="wf_pos_pct_r") / 100

    if st.button("▶ Run Walk-Forward", type="primary", key="wf_run"):
        with st.spinner("Running walk-forward validation..."):
            try:
                if wf_mode == "Simple 70/30 Split":
                    result = api.perplexity_walkforward(
                        wf_strat, wf_sym, mode="simple", period=wf_period,
                        train_pct=wf_train_pct, initial_capital=wf_capital,
                        position_pct=wf_pos_pct,
                    )
                else:
                    result = api.perplexity_walkforward(
                        wf_strat, wf_sym, mode="rolling", period=wf_period,
                        train_years=wf_train_years, test_years=wf_test_years,
                        step_years=wf_step_years, initial_capital=wf_capital,
                        position_pct=wf_pos_pct,
                    )
                st.session_state["px_walkforward"] = result
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")

    # ── Results ───────────────────────────────────────────────
    def _fmt_pf(v):
        return f"{v:.2f}" if v is not None else "—"

    def _wfe_color(wfe):
        if wfe is None: return "off"
        if wfe >= 0.7:  return "normal"
        if wfe >= 0.5:  return "off"
        return "inverse"

    def _render_segment_cols(seg: dict, label: str, color: str):
        st.markdown(f"#### {label}  `{seg['start']} → {seg['end']}`")
        st.caption(f"{seg['bars']} bars")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", f"{seg['total_return_pct']:+.2f}%")
        c2.metric("CAGR",         f"{seg['cagr']:+.2f}%")
        c3.metric("Trades",       seg["trades"])
        c4, c5, c6 = st.columns(3)
        c4.metric("Win Rate",      f"{seg['win_rate_pct']:.1f}%")
        c5.metric("Profit Factor", _fmt_pf(seg.get("profit_factor")))
        c6.metric("Max Drawdown",  f"{seg['max_drawdown_pct']:.1f}%", delta_color="inverse")

    def _equity_chart(traces: list, title: str):
        fig = go.Figure()
        for t in traces:
            fig.add_trace(go.Scatter(x=t["x"], y=t["y"], mode="lines",
                                     name=t["name"], line=t["line"]))
        fig.update_layout(
            title=title, height=320, template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    wfr = st.session_state.get("px_walkforward")
    if wfr:
        st.divider()
        mode = wfr.get("mode", "simple")

        # ── SIMPLE MODE ──────────────────────────────────────
        if mode == "simple":
            is_seg  = wfr.get("is") or wfr.get("train")
            oos_seg = wfr.get("oos") or wfr.get("test")
            wfe     = wfr.get("wfe") or wfr.get("oos_return_ratio", 0)
            oos_pf_ratio = wfr.get("oos_pf_ratio")
            wfe_label = wfr.get("wfe_label", "")

            st.subheader(f"{wfr['strategy_name']} on {wfr['symbol']}  |  "
                         f"{wfr['full_period']}  |  "
                         f"Split {int(wfr.get('train_pct', 0.7)*100)}/{int((1-wfr.get('train_pct',0.7))*100)}")

            col_is, col_oos = st.columns(2)
            with col_is:
                _render_segment_cols(is_seg, "In-Sample (IS)", "#00d4aa")
            with col_oos:
                _render_segment_cols(oos_seg, "Out-of-Sample (OOS)", "#FFD700")

            st.divider()
            wfe_val  = wfe if wfe is not None else 0.0
            pf_ratio = oos_pf_ratio if oos_pf_ratio is not None else 0.0
            d1, d2, d3 = st.columns(3)
            d1.metric(
                "WFE (Walk-Forward Efficiency)",
                f"{wfe_val:.2f}×" if wfe is not None else "N/A",
                delta=wfe_label,
                delta_color=_wfe_color(wfe),
                help="OOS CAGR ÷ IS CAGR. ≥0.7 = acceptable. <0.5 = likely overfit.",
            )
            d2.metric(
                "OOS Profit Factor Ratio",
                f"{pf_ratio:.2f}×" if oos_pf_ratio is not None else "N/A",
                help="OOS Profit Factor ÷ IS Profit Factor.",
            )
            d3.metric(
                "IS CAGR vs OOS CAGR",
                f"{is_seg['cagr']:+.2f}% → {oos_seg['cagr']:+.2f}%",
            )

            is_curve  = is_seg.get("equity_curve", [])
            oos_curve = oos_seg.get("equity_curve", [])
            if is_curve or oos_curve:
                traces = []
                if is_curve:
                    df_is = pd.DataFrame(is_curve)
                    traces.append({"x": df_is["date"], "y": df_is["equity"],
                                   "name": "In-Sample", "line": dict(color="#00d4aa", width=2)})
                if oos_curve:
                    df_oos = pd.DataFrame(oos_curve)
                    traces.append({"x": df_oos["date"], "y": df_oos["equity"],
                                   "name": "Out-of-Sample", "line": dict(color="#FFD700", width=2, dash="dot")})
                _equity_chart(traces, "IS vs OOS Equity Curve")

        # ── ROLLING MODE ─────────────────────────────────────
        else:
            segs = wfr.get("segments", [])
            composite = wfr.get("oos_composite_curve", [])
            g_cagr = wfr.get("global_oos_cagr", 0)
            g_tr   = wfr.get("global_oos_total_return_pct", 0)
            avg_is = wfr.get("avg_is_cagr", 0)
            g_wfe  = wfr.get("global_wfe")
            g_label= wfr.get("global_wfe_label", "")

            st.subheader(f"{wfr['strategy_name']} on {wfr['symbol']}  |  "
                         f"{wfr['full_period']}  |  "
                         f"{len(segs)} windows  "
                         f"({wfr.get('train_years',3)}y IS / {wfr.get('test_years',1)}y OOS "
                         f"step {wfr.get('step_years',1)}y)")

            # Global summary
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Global OOS Total Return", f"{g_tr:+.2f}%")
            g2.metric("Global OOS CAGR",         f"{g_cagr:+.2f}%")
            g3.metric("Avg IS CAGR",              f"{avg_is:+.2f}%")
            g4.metric(
                "Global WFE",
                f"{g_wfe:.2f}×" if g_wfe is not None else "N/A",
                delta=g_label,
                delta_color=_wfe_color(g_wfe),
                help="Global OOS CAGR ÷ Average IS CAGR across all windows.",
            )

            # Composite OOS equity curve
            if composite:
                df_comp = pd.DataFrame(composite)
                _equity_chart(
                    [{"x": df_comp["date"], "y": df_comp["equity"],
                      "name": "Composite OOS Equity",
                      "line": dict(color="#FFD700", width=2)}],
                    "Composite OOS Equity Curve (all windows stitched)",
                )

            # Per-segment table
            if segs:
                st.subheader("Per-Window Results")
                tbl_rows = []
                for s in segs:
                    wfe_s = s.get("wfe")
                    tbl_rows.append({
                        "Window": s["window"],
                        "IS":     f"{s['is_start']} → {s['is_end']}",
                        "IS CAGR": f"{s['is_cagr']:+.2f}%",
                        "IS PF":   _fmt_pf(s.get("is_profit_factor")),
                        "IS Trades": s["is_trades"],
                        "OOS":    f"{s['oos_start']} → {s['oos_end']}",
                        "OOS CAGR": f"{s['oos_cagr']:+.2f}%",
                        "OOS PF":   _fmt_pf(s.get("oos_profit_factor")),
                        "OOS Trades": s["oos_trades"],
                        "WFE":    f"{wfe_s:.2f}×" if wfe_s is not None else "N/A",
                        "Verdict": s.get("wfe_label", ""),
                    })
                st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

                # WFE bar chart per window
                wfe_vals = [s.get("wfe") or 0 for s in segs]
                colors = [
                    "#00d4aa" if v >= 0.7 else "#FFD700" if v >= 0.5 else "#ff4b4b"
                    for v in wfe_vals
                ]
                fig_wfe = go.Figure(go.Bar(
                    x=[f"W{s['window']}" for s in segs],
                    y=wfe_vals,
                    marker_color=colors,
                    text=[f"{v:.2f}×" for v in wfe_vals],
                    textposition="outside",
                ))
                fig_wfe.add_hline(y=0.7, line_dash="dash", line_color="#00d4aa",
                                  annotation_text="0.70 threshold")
                fig_wfe.update_layout(
                    title="WFE per Window  (green ≥ 0.70 | yellow 0.50–0.70 | red < 0.50)",
                    height=280, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="WFE",
                )
                st.plotly_chart(fig_wfe, use_container_width=True)

    # ── Before / After Filter Comparison ─────────────────────
    # Available for all strategies — filter inputs auto-populate from saved calibration
    # profile for the selected symbol, falling back to strategy config defaults.
    st.divider()
    with st.expander("🔬 Before vs After Filter Comparison — do the calibrated filters improve WFE?", expanded=False):
        st.caption(
            f"Runs rolling walk-forward **twice** on **{wf_strat}**: once with filters OFF, "
            "once with filters ON using the values from the saved Symbol Profile for this symbol "
            "(or strategy defaults if no profile exists). A genuine improvement in OOS WFE "
            "means the filters are real and not overfit."
        )

        baf_col1, baf_col2, baf_col3 = st.columns(3)
        with baf_col1:
            baf_sym = st.text_input("Symbol", value=wf_sym, key="baf_sym").upper().strip()
        with baf_col2:
            baf_period = st.selectbox("Period", ["5y", "10y"], index=1, key="baf_period")
        with baf_col3:
            baf_capital = st.number_input("Capital ($)", min_value=1000, value=10000,
                                          step=1000, key="baf_cap")

        # ── Auto-load saved profile for this symbol/strategy ──
        _baf_profile = {}
        try:
            _baf_profile = api.perplexity_profile(wf_strat, baf_sym) or {}
        except Exception:
            pass

        # Helper: read from profile first, then strategy config, then hard fallback
        def _baf_default(profile_key: str, config_key: str, fallback: float) -> float:
            v = _baf_profile.get(profile_key)
            if v and float(v) > 0:
                return float(v)
            try:
                from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES as _PS
                _cfgmap = {s.name: s for s in _PS}
                cfg_v = _cfgmap.get(wf_strat, None)
                if cfg_v:
                    cv = cfg_v.config.get(config_key, 0.0)
                    if cv and float(cv) > 0:
                        return float(cv)
            except Exception:
                pass
            return fallback

        if _baf_profile:
            st.info(f"✅ Loaded saved profile for **{wf_strat} / {baf_sym}** — filter defaults pre-filled from calibration results.")
        else:
            st.warning(f"No saved profile found for **{wf_strat} / {baf_sym}**. Run Auto-Calibrate in the Symbol Profiles tab first for best results. Using strategy defaults.")

        # ── Strategy-specific filter inputs ───────────────────
        baf_params: dict = {}

        if wf_strat == "EMA_Mean_Reversion":
            f1, f2, f3 = st.columns(3)
            with f1:
                baf_params["ema_dist_min"] = st.number_input(
                    "Min EMA distance % (≥)", 0.0, 10.0,
                    _baf_default("ema_dist_min", "filter_ema_dist_min", 1.5), 0.1, key="baf_f1")
            with f2:
                baf_params["vol_min"] = st.number_input(
                    "Min volume ratio (≥)", 0.0, 3.0,
                    _baf_default("vol_min", "filter_vol_min", 1.0), 0.1, key="baf_f2")
            with f3:
                baf_params["bb_pos_min"] = st.number_input(
                    "Min BB position (≥)", 0.0, 1.5,
                    _baf_default("bb_pos_min", "filter_bb_pos_min", 0.72), 0.05, key="baf_f3")
            _filter_label = (f"EMA dist≥{baf_params['ema_dist_min']:.1f}%, "
                             f"Vol≥{baf_params['vol_min']:.1f}×, "
                             f"BB pos≥{baf_params['bb_pos_min']:.2f}")

        elif wf_strat == "MA_Crossover_RSI":
            f1, f2 = st.columns(2)
            with f1:
                baf_params["vol_min"] = st.number_input(
                    "Min volume ratio (≥)", 0.0, 3.0,
                    _baf_default("vol_min", "filter_vol_min", 1.0), 0.1, key="baf_f1")
            with f2:
                baf_params["ema_spread_min"] = st.number_input(
                    "Min EMA spread % (≥)", 0.0, 10.0,
                    _baf_default("ema_spread_min", "filter_ema_spread_min", 1.0), 0.1, key="baf_f2")
            _filter_label = (f"Vol≥{baf_params['vol_min']:.1f}×, "
                             f"EMA spread≥{baf_params['ema_spread_min']:.1f}%")

        elif wf_strat == "Breakout_Consolidation":
            f1, f2 = st.columns(2)
            with f1:
                baf_params["vol_min"] = st.number_input(
                    "Min volume ratio (≥)", 0.0, 3.0,
                    _baf_default("vol_min", "filter_vol_min", 1.5), 0.1, key="baf_f1")
            with f2:
                baf_params["range_atr_max"] = st.number_input(
                    "Max range/ATR ratio (≤, 0=off)", 0.0, 5.0,
                    _baf_default("range_atr_max", "filter_range_atr_max", 0.0), 0.1, key="baf_f2")
            _filter_label = (f"Vol≥{baf_params['vol_min']:.1f}×" +
                             (f", Range/ATR≤{baf_params['range_atr_max']:.1f}" if baf_params["range_atr_max"] > 0 else ""))

        elif wf_strat == "BB_Mean_Reversion":
            f1, f2, f3 = st.columns(3)
            with f1:
                baf_params["vol_min"] = st.number_input(
                    "Min volume ratio (≥)", 0.0, 3.0,
                    _baf_default("vol_min", "filter_vol_min", 1.0), 0.1, key="baf_f1")
            with f2:
                baf_params["atr_pct_max"] = st.number_input(
                    "Max ATR% (≤, 0=off)", 0.0, 10.0,
                    _baf_default("atr_pct_max", "filter_atr_pct_max", 0.0), 0.1, key="baf_f2")
            with f3:
                baf_params["bb_depth_min"] = st.number_input(
                    "Min BB depth below lower band (≥)", 0.0, 1.0,
                    _baf_default("bb_depth_min", "filter_bb_depth_min", 0.0), 0.05, key="baf_f3")
            _filter_label = (f"Vol≥{baf_params['vol_min']:.1f}×" +
                             (f", ATR%≤{baf_params['atr_pct_max']:.1f}" if baf_params["atr_pct_max"] > 0 else "") +
                             (f", BB depth≥{baf_params['bb_depth_min']:.2f}" if baf_params["bb_depth_min"] > 0 else ""))

        elif wf_strat == "Fib_Pullback_Support":
            f1, f2 = st.columns(2)
            with f1:
                baf_params["lower_wick_min"] = st.number_input(
                    "Min lower wick % of range (≥)", 0.0, 50.0,
                    _baf_default("lower_wick_min", "filter_lower_wick_min", 0.0), 1.0, key="baf_f1")
            with f2:
                baf_params["vol_min"] = st.number_input(
                    "Min volume ratio (≥)", 0.0, 3.0,
                    _baf_default("vol_min", "filter_vol_min", 1.0), 0.1, key="baf_f2")
            _filter_label = (
                (f"Lower wick≥{baf_params['lower_wick_min']:.0f}%" if baf_params["lower_wick_min"] > 0 else "") +
                (", " if baf_params["lower_wick_min"] > 0 and baf_params["vol_min"] > 0 else "") +
                (f"Vol≥{baf_params['vol_min']:.1f}×" if baf_params["vol_min"] > 0 else "")
            ) or "no filters set"
        else:
            baf_params["vol_min"] = st.number_input("Min volume ratio (≥)", 0.0, 3.0, 1.0, 0.1, key="baf_f1")
            _filter_label = f"Vol≥{baf_params['vol_min']:.1f}×"

        # Build query string from non-zero params
        _baf_qs = "&".join(f"{k}={v}" for k, v in baf_params.items() if v is not None)

        if st.button("▶ Run Before/After Comparison", type="primary", key="baf_run"):
            with st.spinner(f"Running both walk-forwards for {wf_strat} / {baf_sym}... ~2–4 min"):
                try:
                    result = api.perplexity_filter_comparison(
                        wf_strat, baf_sym,
                        period=baf_period, train_years=3.0, test_years=1.0, step_years=1.0,
                        initial_capital=baf_capital, position_pct=0.0,
                        **{k: v for k, v in baf_params.items() if v is not None},
                    )
                    st.session_state["baf_result"] = result
                    st.session_state["baf_filter_label"] = _filter_label
                    st.session_state["baf_strat"] = wf_strat
                    st.session_state["baf_sym_used"] = baf_sym
                except Exception as e:
                    st.error(f"Comparison failed: {e}")

        baf_res = st.session_state.get("baf_result")
        _saved_label = st.session_state.get("baf_filter_label", _filter_label)
        _saved_strat = st.session_state.get("baf_strat", wf_strat)
        _saved_sym   = st.session_state.get("baf_sym_used", baf_sym)
        bef = baf_res.get("before") if baf_res else None
        aft = baf_res.get("after")  if baf_res else None

        if bef and aft:
            st.divider()
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("WITHOUT filters — Global WFE",
                      f"{bef.get('global_wfe', 0):.2f}×" if bef.get("global_wfe") else "N/A",
                      delta=bef.get("global_wfe_label", ""))
            b2.metric("WITHOUT filters — OOS CAGR",
                      f"{bef.get('global_oos_cagr', 0):+.2f}%")
            b3.metric("WITH filters — Global WFE",
                      f"{aft.get('global_wfe', 0):.2f}×" if aft.get("global_wfe") else "N/A",
                      delta=aft.get("global_wfe_label", ""),
                      delta_color="normal" if (aft.get("global_wfe") or 0) > (bef.get("global_wfe") or 0) else "inverse")
            b4.metric("WITH filters — OOS CAGR",
                      f"{aft.get('global_oos_cagr', 0):+.2f}%",
                      delta=f"{aft.get('global_oos_cagr', 0) - bef.get('global_oos_cagr', 0):+.2f}% vs no filters",
                      delta_color="normal" if aft.get("global_oos_cagr", 0) > bef.get("global_oos_cagr", 0) else "inverse")

            bef_curve = bef.get("oos_composite_curve", [])
            aft_curve = aft.get("oos_composite_curve", [])
            if bef_curve or aft_curve:
                fig_baf = go.Figure()
                if bef_curve:
                    df_b = pd.DataFrame(bef_curve)
                    fig_baf.add_trace(go.Scatter(
                        x=df_b["date"], y=df_b["equity"], mode="lines",
                        name="Without filters", line=dict(color="#ff4b4b", width=2, dash="dot")))
                if aft_curve:
                    df_a = pd.DataFrame(aft_curve)
                    fig_baf.add_trace(go.Scatter(
                        x=df_a["date"], y=df_a["equity"], mode="lines",
                        name=f"With filters ({_saved_label})",
                        line=dict(color="#00d4aa", width=2)))
                fig_baf.update_layout(
                    title=f"Composite OOS Equity — {_saved_strat} on {_saved_sym}",
                    height=320, template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                fig_baf.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                fig_baf.update_yaxes(gridcolor="rgba(255,255,255,0.05)", tickprefix="$")
                st.plotly_chart(fig_baf, use_container_width=True)

            bef_segs = bef.get("segments", [])
            aft_segs = aft.get("segments", [])
            if bef_segs and aft_segs:
                st.subheader("Per-Window WFE: Before vs After")
                cmp_rows = []
                for i, (bs, as_) in enumerate(zip(bef_segs, aft_segs)):
                    bwfe = bs.get("wfe")
                    awfe = as_.get("wfe")
                    improved = (awfe or 0) > (bwfe or 0)
                    cmp_rows.append({
                        "Window": f"W{i+1}  {bs['oos_start']}→{bs['oos_end']}",
                        "Before OOS CAGR": f"{bs['oos_cagr']:+.2f}%",
                        "Before WFE": f"{bwfe:.2f}×" if bwfe is not None else "N/A",
                        "After OOS CAGR": f"{as_['oos_cagr']:+.2f}%",
                        "After WFE": f"{awfe:.2f}×" if awfe is not None else "N/A",
                        "Improved?": "✅ Yes" if improved else "❌ No",
                    })
                st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 7 — SYMBOL PROFILES
# ══════════════════════════════════════════════════════════════
with tab_profiles:
    st.subheader("🎯 Per-Symbol Filter Profiles")
    st.caption(
        "Each symbol gets its own calibrated entry filters for EMA Mean Reversion. "
        "Run **Auto-Calibrate** on any symbol — it analyzes winning vs losing trades, "
        "derives optimal thresholds, verifies with walk-forward, and saves automatically. "
        "The strategy uses these filters live whenever it trades that symbol."
    )

    pr_col1, pr_col2 = st.columns([2, 1])
    with pr_col1:
        prof_strat = st.selectbox("Strategy", [
            "EMA_Mean_Reversion", "MA_Crossover_RSI", "Breakout_Consolidation",
            "BB_Mean_Reversion", "Fib_Pullback_Support"
        ], key="prof_strat")
    with pr_col2:
        if st.button("🔄 Refresh profiles", key="prof_refresh"):
            st.rerun()

    # ── Auto-Calibrate section ────────────────────────────────
    st.divider()
    st.markdown("### Auto-Calibrate a Symbol")
    st.caption(
        "Runs backtest → analyzes trade patterns → derives thresholds → "
        "verifies with walk-forward → saves profile. Takes ~2 min per symbol."
    )

    ac1, ac2, ac3, ac4 = st.columns([2, 1, 1, 1])
    with ac1:
        cal_sym = st.text_input("Symbol to calibrate", value="AAPL",
                                key="cal_sym").upper().strip()
    with ac2:
        cal_period = st.selectbox("Data period", ["3y", "5y", "10y"], index=1, key="cal_period")
    with ac3:
        cal_capital = st.number_input("Capital ($)", value=10000, min_value=1000,
                                      step=1000, key="cal_cap")
    with ac4:
        cal_verify = st.toggle("Verify with WF", value=True, key="cal_verify",
                               help="Run before/after walk-forward to confirm filters improve WFE. "
                                    "Adds ~90s but gives confidence the thresholds are real.")

    if st.button("⚡ Auto-Calibrate", type="primary", key="cal_run", use_container_width=True):
        with st.spinner(f"Calibrating {prof_strat} for {cal_sym}... analyzing trades + {'verifying WF' if cal_verify else 'skipping WF'}"):
            try:
                cal_result = api.perplexity_calibrate(prof_strat, cal_sym, period=cal_period,
                                                      initial_capital=cal_capital, verify_wf=cal_verify)
                st.session_state["cal_result"] = cal_result
                st.success(f"✅ Profile saved for {cal_sym}!")
            except Exception as e:
                st.error(f"Calibration failed: {e}")

    cal = st.session_state.get("cal_result")
    if cal and cal.get("symbol") == cal_sym:
        st.divider()
        st.markdown(f"#### Calibration result — **{cal['symbol']}** ({cal['n_trades']} trades, {cal['win_rate_pct']:.1f}% win rate)")

        thr = cal.get("thresholds", {})
        ev  = cal.get("evidence", {})
        ver = cal.get("verification", {})

        # Thresholds — shown dynamically per strategy
        _strat = cal.get("strategy", prof_strat)
        t_cols = st.columns(3)

        def _thresh_metric(col, label, val, fmt, win_v, loss_v, win_fmt="", loss_fmt=""):
            col.metric(label,
                       f"{fmt.format(val)}" if val > 0 else "OFF",
                       delta=f"wins {win_fmt.format(win_v) if win_v else '—'} vs losses {loss_fmt.format(loss_v) if loss_v else '—'}",
                       delta_color="off")

        if _strat == "EMA_Mean_Reversion":
            t_cols[0].metric("EMA dist filter",
                f"≥ {thr.get('ema_dist_min', 0):.2f}%" if thr.get('ema_dist_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_ema_dist_mean', 0):.2f}% vs losses {ev.get('loss_ema_dist_mean', 0):.2f}%", delta_color="off")
            t_cols[1].metric("Volume filter",
                f"≥ {thr.get('vol_min', 0):.2f}×" if thr.get('vol_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_vol_mean', 0):.2f}× vs losses {ev.get('loss_vol_mean', 0):.2f}×", delta_color="off")
            t_cols[2].metric("BB position filter",
                f"≥ {thr.get('bb_pos_min', 0):.2f}" if thr.get('bb_pos_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_bb_pos_mean', 0):.2f} vs losses {ev.get('loss_bb_pos_mean', 0):.2f}", delta_color="off")
        elif _strat == "MA_Crossover_RSI":
            t_cols[0].metric("RSI min filter",
                f"≥ {thr.get('rsi_min', 0):.0f}" if thr.get('rsi_min', 0) > 0 else "OFF", delta_color="off")
            t_cols[1].metric("Volume filter",
                f"≥ {thr.get('vol_min', 0):.2f}×" if thr.get('vol_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_vol_mean', 0):.2f}× vs losses {ev.get('loss_vol_mean', 0):.2f}×", delta_color="off")
            t_cols[2].metric("EMA spread min",
                f"≥ {thr.get('ema_spread_min', 0):.2f}%" if thr.get('ema_spread_min', 0) > 0 else "OFF", delta_color="off")
        elif _strat == "Breakout_Consolidation":
            t_cols[0].metric("Volume filter",
                f"≥ {thr.get('vol_min', 0):.2f}×" if thr.get('vol_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_vol_mean', 0):.2f}× vs losses {ev.get('loss_vol_mean', 0):.2f}×", delta_color="off")
            t_cols[1].metric("RSI min filter",
                f"≥ {thr.get('rsi_min', 0):.0f}" if thr.get('rsi_min', 0) > 0 else "OFF", delta_color="off")
            t_cols[2].metric("Max range/ATR",
                f"≤ {thr.get('range_atr_max', 0):.1f}" if thr.get('range_atr_max', 0) > 0 else "OFF", delta_color="off")
        elif _strat == "BB_Mean_Reversion":
            t_cols[0].metric("RSI max filter",
                f"≤ {thr.get('rsi_max', 0):.0f}" if thr.get('rsi_max', 0) > 0 else "OFF", delta_color="off")
            t_cols[1].metric("Volume filter",
                f"≥ {thr.get('vol_min', 0):.2f}×" if thr.get('vol_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_vol_mean', 0):.2f}× vs losses {ev.get('loss_vol_mean', 0):.2f}×", delta_color="off")
            t_cols[2].metric("Max ATR%",
                f"≤ {thr.get('atr_pct_max', 0):.1f}%" if thr.get('atr_pct_max', 0) > 0 else "OFF", delta_color="off")
        elif _strat == "Fib_Pullback_Support":
            t_cols[0].metric("RSI min filter",
                f"≥ {thr.get('rsi_min', 0):.0f}" if thr.get('rsi_min', 0) > 0 else "OFF", delta_color="off")
            t_cols[1].metric("Min lower wick%",
                f"≥ {thr.get('lower_wick_min', 0):.0f}%" if thr.get('lower_wick_min', 0) > 0 else "OFF", delta_color="off")
            t_cols[2].metric("Volume filter",
                f"≥ {thr.get('vol_min', 0):.2f}×" if thr.get('vol_min', 0) > 0 else "OFF",
                delta=f"wins {ev.get('win_vol_mean', 0):.2f}× vs losses {ev.get('loss_vol_mean', 0):.2f}×", delta_color="off")

        # Walk-forward verification
        if ver.get("wfe_before") is not None or ver.get("wfe_after") is not None:
            st.divider()
            st.markdown("**Walk-Forward Verification**")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("WFE without filters",
                      f"{ver['wfe_before']:.2f}×" if ver['wfe_before'] is not None else "N/A")
            v2.metric("OOS CAGR without",
                      f"{ver['oos_cagr_before']:+.2f}%" if ver['oos_cagr_before'] is not None else "N/A")
            v3.metric("WFE with filters",
                      f"{ver['wfe_after']:.2f}×" if ver['wfe_after'] is not None else "N/A",
                      delta_color="normal" if (ver.get("wfe_after") or 0) >= (ver.get("wfe_before") or 0) else "inverse")
            v4.metric("OOS CAGR with",
                      f"{ver['oos_cagr_after']:+.2f}%" if ver['oos_cagr_after'] is not None else "N/A",
                      delta_color="normal" if (ver.get("oos_cagr_after") or 0) >= (ver.get("oos_cagr_before") or 0) else "inverse")

            if ver.get("verified"):
                st.success("✅ Verified — filters improve walk-forward performance. Profile is active.")
            else:
                st.warning("⚠️ Filters did not clearly improve walk-forward on this symbol. "
                           "Profile saved but treat with caution — consider disabling individual filters.")

    # ── Saved profiles table ──────────────────────────────────
    st.divider()
    st.markdown("### Saved Profiles")

    try:
        all_profiles = api.perplexity_profiles(prof_strat)
    except Exception as e:
        st.error(f"Could not load profiles: {e}")
        all_profiles = []

    if not all_profiles:
        st.info("No profiles saved yet. Run Auto-Calibrate on a symbol above to create the first one.")
    else:
        prof_rows = []
        for p in all_profiles:
            strat = p.get("strategy", "")
            if strat == "EMA_Mean_Reversion":
                filters_active = [
                    f"EMA≥{p['ema_dist_min']:.1f}%" if p.get('ema_dist_min', 0) > 0 else None,
                    f"Vol≥{p['vol_min']:.1f}×"      if p.get('vol_min', 0)      > 0 else None,
                    f"BB≥{p['bb_pos_min']:.2f}"      if p.get('bb_pos_min', 0)   > 0 else None,
                ]
            elif strat == "MA_Crossover_RSI":
                filters_active = [
                    f"Vol≥{p['vol_min']:.1f}×"           if p.get('vol_min', 0)         > 0 else None,
                    f"Spread≥{p['ema_spread_min']:.1f}%" if p.get('ema_spread_min', 0)  > 0 else None,
                ]
            elif strat == "Breakout_Consolidation":
                filters_active = [
                    f"Vol≥{p['vol_min']:.1f}×" if p.get('vol_min', 0) > 0 else None,
                ]
            elif strat == "BB_Mean_Reversion":
                filters_active = [
                    f"Vol≥{p['vol_min']:.1f}×"        if p.get('vol_min', 0)      > 0 else None,
                    f"ATR%≤{p['atr_pct_max']:.1f}%"   if p.get('atr_pct_max', 0)  > 0 else None,
                ]
            elif strat == "Fib_Pullback_Support":
                filters_active = [
                    f"Wick≥{p['lower_wick_min']:.0f}%" if p.get('lower_wick_min', 0) > 0 else None,
                    f"Vol≥{p['vol_min']:.1f}×"          if p.get('vol_min', 0)        > 0 else None,
                ]
            else:
                filters_active = []
            filters_str = " | ".join(f for f in filters_active if f) or "None (all disabled)"
            wfe_change = ""
            if p.get("wfe_before") is not None and p.get("wfe_after") is not None:
                delta = (p["wfe_after"] or 0) - (p["wfe_before"] or 0)
                wfe_change = f"{delta:+.2f}×"
            prof_rows.append({
                "Symbol":        p["symbol"],
                "Calibrated":    p["calibrated_at"],
                "Trades":        p["n_trades"],
                "Win Rate":      f"{p['win_rate_pct']:.1f}%",
                "Active Filters": filters_str,
                "WFE change":    wfe_change,
                "Verified":      "✅" if p.get("verified") else "⚠️ Unverified",
            })
        st.dataframe(pd.DataFrame(prof_rows), use_container_width=True, hide_index=True)

        # Delete a profile
        st.divider()
        del_sym = st.selectbox("Delete profile for symbol",
                               [""] + [p["symbol"] for p in all_profiles],
                               key="del_sym")
        if del_sym and st.button(f"🗑 Delete {del_sym} profile", key="del_prof"):
            try:
                api.perplexity_delete_profile(prof_strat, del_sym)
                st.success(f"Deleted profile for {del_sym}")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── Batch calibrate ───────────────────────────────────────
    st.divider()
    st.markdown("### Batch Calibrate Multiple Symbols")
    st.caption("Runs Auto-Calibrate on all symbols in the list sequentially. Takes ~2 min per symbol.")
    batch_input = st.text_input("Symbols (comma-separated)", value="AAPL,SPY,QQQ,NVDA,MSFT",
                                key="batch_syms")
    batch_period = st.selectbox("Period", ["3y", "5y"], index=1, key="batch_period")
    if st.button("⚡ Batch Calibrate All", key="batch_run"):
        syms = [s.strip().upper() for s in batch_input.split(",") if s.strip()]
        progress = st.progress(0, text="Starting...")
        batch_results = []
        for i, sym in enumerate(syms):
            progress.progress((i) / len(syms), text=f"Calibrating {sym}...")
            try:
                r = api.perplexity_calibrate(prof_strat, sym, period=batch_period,
                                             initial_capital=10000, verify_wf=True)
                thr = r.get("thresholds", {})
                filters_str = " | ".join(f"{k}={v:.2f}" for k, v in thr.items() if v and v > 0) or "none"
                batch_results.append({"Symbol": sym, "Status": "✅ Done",
                                      "Filters": filters_str,
                                      "Verified": "✅" if r.get("verification", {}).get("verified") else "⚠️"})
            except Exception as e:
                batch_results.append({"Symbol": sym, "Status": f"❌ {e}", "Filters": "—", "Verified": "—"})
        progress.progress(1.0, text="Done!")
        st.dataframe(pd.DataFrame(batch_results), use_container_width=True, hide_index=True)
        st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 8 — TRADE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab_analysis:
    st.subheader("Trade Analysis — Winner vs Loser Breakdown")
    st.caption(
        "Runs a backtest then compares entry-bar indicators between winning and losing trades. "
        "High Cohen's d features are the strongest filters — use them to tighten entry conditions "
        "and improve win rate. Run **Optimize Filters** to auto-apply the best thresholds as a profile."
    )

    # ── Step 1: Symbol + period controls ────────────────────────
    s1c1, s1c2, s1c3 = st.columns([2, 1, 1])
    with s1c1:
        an_symbol = st.text_input("Symbol", value="NVDA", key="an_sym").upper().strip()
    with s1c2:
        an_period = st.selectbox("Period", ["3y", "5y", "8y", "10y"], index=1, key="an_period")
    with s1c3:
        an_pos_pct = st.number_input("Position %", min_value=5, max_value=50, value=20, step=5, key="an_pos") / 100

    compare_btn = st.button("📊 Compare All Strategies", key="compare_strats")

    if compare_btn and an_symbol:
        with st.spinner(f"Running all 5 strategies on {an_symbol} ({an_period})…"):
            compare_rows = []
            for sname in STRATEGY_DESCRIPTIONS.keys():
                try:
                    r = api.perplexity_analyze(sname, an_symbol, period=an_period, position_pct=an_pos_pct)
                    snaps_c = r.get("snapshots", [])
                    wins_c  = [s for s in snaps_c if s["outcome"] == "win"]
                    losses_c= [s for s in snaps_c if s["outcome"] == "loss"]
                    wr_c    = r.get("win_rate_pct", 0)
                    n_c     = r.get("total_trades", 0)
                    avg_w   = round(sum(s["pnl_pct"] for s in wins_c)   / len(wins_c),   2) if wins_c   else 0
                    avg_l   = round(sum(s["pnl_pct"] for s in losses_c) / len(losses_c), 2) if losses_c else 0
                    exp     = round(wr_c / 100 * avg_w + (1 - wr_c / 100) * avg_l, 3)
                    top_feat= r.get("patterns", [{}])[0].get("indicator", "—") if r.get("patterns") else "—"
                    top_d   = r.get("patterns", [{}])[0].get("separation", 0)   if r.get("patterns") else 0
                    score   = round(wr_c * (n_c ** 0.5) / 10, 1)  # composite: WR × sqrt(trades)
                    compare_rows.append({
                        "Strategy":      sname,
                        "Trades":        n_c,
                        "Win Rate %":    f"{wr_c:.1f}",
                        "Avg Win %":     f"+{avg_w:.2f}",
                        "Avg Loss %":    f"{avg_l:.2f}",
                        "Expectancy %":  f"{exp:+.3f}",
                        "Top Indicator": f"{top_feat} (d={top_d:.2f})" if top_feat != "—" else "—",
                        "Score":         score,
                    })
                except Exception as e:
                    compare_rows.append({"Strategy": sname, "Trades": 0, "Win Rate %": "ERR",
                                         "Avg Win %": "—", "Avg Loss %": "—", "Expectancy %": "—",
                                         "Top Indicator": str(e)[:40], "Score": 0})

        if compare_rows:
            st.markdown(f"#### Strategy Comparison — {an_symbol} / {an_period}")
            st.caption("Score = Win Rate × √Trades ÷ 10. Higher is better. Pick the top scorer for detailed analysis below.")
            cdf = pd.DataFrame(compare_rows).sort_values("Score", ascending=False)
            st.dataframe(cdf, use_container_width=True, hide_index=True)
            best = cdf.iloc[0]["Strategy"]
            st.info(f"**Recommended:** {best} has the highest composite score on {an_symbol}. Select it below to run a full analysis.")
            st.session_state["an_best_strategy"] = best

    st.divider()

    # ── Step 2: Detailed analysis for chosen strategy ─────────
    st.markdown("#### Detailed Analysis")
    default_strat = st.session_state.get("an_best_strategy", list(STRATEGY_DESCRIPTIONS.keys())[0])
    default_idx   = list(STRATEGY_DESCRIPTIONS.keys()).index(default_strat) if default_strat in STRATEGY_DESCRIPTIONS else 0
    an_strategy   = st.selectbox("Strategy", list(STRATEGY_DESCRIPTIONS.keys()), index=default_idx, key="an_strat")

    btn_col1, btn_col2 = st.columns([1, 2])
    with btn_col1:
        run_analysis_btn = st.button("🔍 Run Analysis", type="primary", key="run_analysis")
    with btn_col2:
        run_optimize_btn = st.button("⚡ Optimize Filters → Save Profile", key="run_optimize")

    if run_analysis_btn or run_optimize_btn:
        with st.spinner(f"Running backtest + analysis for {an_strategy}/{an_symbol}..."):
            try:
                result = api.perplexity_analyze(
                    an_strategy, an_symbol,
                    period=an_period,
                    position_pct=an_pos_pct,
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                result = None

        if result:
            n       = result.get("total_trades", 0)
            wr      = result.get("win_rate_pct", 0)
            patterns = result.get("patterns", [])
            timing   = result.get("timing", {})
            snaps    = result.get("snapshots", [])
            msg      = result.get("message")

            if msg:
                st.warning(msg)
            elif n == 0:
                st.warning("No trades generated — try a different period or symbol.")
            else:
                # ── Summary metrics ──────────────────────────────────
                wins   = [s for s in snaps if s["outcome"] == "win"]
                losses = [s for s in snaps if s["outcome"] == "loss"]
                avg_win  = round(sum(s["pnl_pct"] for s in wins)   / len(wins),   2) if wins   else 0
                avg_loss = round(sum(s["pnl_pct"] for s in losses) / len(losses), 2) if losses else 0
                expectancy = round(wr / 100 * avg_win + (1 - wr / 100) * avg_loss, 3)

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Trades", n)
                mc2.metric("Win Rate", f"{wr:.1f}%")
                mc3.metric("Avg Win", f"+{avg_win:.2f}%")
                mc4.metric("Avg Loss", f"{avg_loss:.2f}%")
                mc5.metric("Expectancy", f"{expectancy:+.3f}%",
                           delta="positive" if expectancy > 0 else "negative",
                           delta_color="normal" if expectancy > 0 else "inverse")

                st.divider()

                # ── Pattern insights table ────────────────────────────
                if patterns:
                    st.markdown("#### Feature Separation — Entry Conditions")
                    st.caption(
                        "Cohen's d measures how well each indicator separates winners from losers. "
                        "**Strong** (|d|≥0.5) features are the best candidates for entry filters."
                    )
                    pat_rows = []
                    for p in patterns:
                        strength = "🔴 Strong" if abs(p["separation"]) >= 0.5 else (
                                   "🟡 Moderate" if abs(p["separation"]) >= 0.2 else "⬜ Weak")
                        direction = "↑ Higher→Win" if p["direction"] == "higher_is_better" else "↓ Lower→Win"
                        pat_rows.append({
                            "Indicator":     p["indicator"],
                            "Description":   p["description"],
                            "Win avg":       p["win_mean"],
                            "Loss avg":      p["loss_mean"],
                            "Cohen's d":     p["separation"],
                            "Strength":      strength,
                            "Signal":        direction,
                            "Suggested filter": (
                                f"≥ {p['suggested_min']}" if p.get("suggested_min") else
                                f"≤ {p['suggested_max']}" if p.get("suggested_max") else "—"
                            ),
                        })
                    st.dataframe(pd.DataFrame(pat_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No meaningful feature separation found — not enough trades or all conditions perform similarly.")

                st.divider()

                # ── Timing charts ─────────────────────────────────────
                tc1, tc2, tc3 = st.columns(3)
                _timing_chart(timing.get("by_month",   {}), "By Month",   tc1)
                _timing_chart(timing.get("by_quarter", {}), "By Quarter", tc2)
                _timing_chart(timing.get("by_day",     {}), "By Weekday", tc3)

                if snaps:
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Histogram(
                        x=[s["pnl_pct"] for s in wins],
                        name="Wins", marker_color="rgba(0,212,170,0.75)", opacity=0.8, nbinsx=20,
                    ))
                    fig_pnl.add_trace(go.Histogram(
                        x=[s["pnl_pct"] for s in losses],
                        name="Losses", marker_color="rgba(255,75,75,0.75)", opacity=0.8, nbinsx=20,
                    ))
                    fig_pnl.add_vline(x=0, line_color="rgba(255,255,255,0.4)", line_dash="dash")
                    fig_pnl.update_layout(
                        barmode="overlay", height=280,
                        title="P&L % Distribution — Wins vs Losses",
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="P&L %", yaxis_title="# Trades",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    fig_pnl.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                    fig_pnl.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                    st.plotly_chart(fig_pnl, use_container_width=True)

                st.divider()

                # ── Scatter: top feature vs P&L ───────────────────────
                if patterns and snaps:
                    top_feat = patterns[0]["indicator"]
                    top_feat_label = INDICATOR_LABELS.get(top_feat, top_feat)
                    st.markdown(f"#### {top_feat_label} vs Trade P&L (top discriminating feature)")
                    feat_vals = [s.get(top_feat) for s in snaps]
                    pnl_vals2 = [s["pnl_pct"] for s in snaps]
                    colors    = ["#00cc88" if s["outcome"] == "win" else "#ff4444" for s in snaps]
                    dates     = [s.get("date", "") for s in snaps]
                    valid = [(f, p, c, d) for f, p, c, d in zip(feat_vals, pnl_vals2, colors, dates)
                             if f is not None]
                    if valid:
                        fx, py, cx, dx = zip(*valid)
                        fig_sc = go.Figure(go.Scatter(
                            x=fx, y=py, mode="markers",
                            marker=dict(color=cx, size=8, opacity=0.75),
                            text=[f"{d}<br>P&L: {p:+.2f}%" for d, p in zip(dx, py)],
                            hovertemplate="%{text}<extra></extra>",
                        ))
                        smin = patterns[0].get("suggested_min")
                        smax = patterns[0].get("suggested_max")
                        if smin:
                            fig_sc.add_vline(x=smin, line_dash="dash", line_color="yellow",
                                             annotation_text=f"filter ≥{smin}")
                        if smax:
                            fig_sc.add_vline(x=smax, line_dash="dash", line_color="yellow",
                                             annotation_text=f"filter ≤{smax}")
                        fig_sc.update_layout(
                            height=320,
                            xaxis_title=top_feat, yaxis_title="P&L %",
                            margin=dict(t=20, b=30, l=40, r=10),
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

                # ── Raw trade log ─────────────────────────────────────
                with st.expander("View all trade snapshots", expanded=False):
                    if snaps:
                        snap_df = pd.DataFrame(snaps)
                        outcome_icon = snap_df["outcome"].map({"win": "✅ Win", "loss": "❌ Loss"})
                        snap_df["outcome"] = outcome_icon
                        st.dataframe(snap_df, use_container_width=True, hide_index=True)

        # ── Optimize & save profile ───────────────────────────────
        if run_optimize_btn:
            with st.spinner(f"Optimizing filters and saving profile for {an_strategy}/{an_symbol}..."):
                try:
                    cal = api.perplexity_calibrate(
                        an_strategy, an_symbol,
                        period=an_period,
                        position_pct=an_pos_pct,
                        verify_wf=True,
                    )
                    if cal.get("saved"):
                        thr = cal.get("thresholds", {})
                        ver = cal.get("verification", {})
                        base_wr    = cal.get("win_rate_pct", 0)
                        filt_wr    = cal.get("filtered_win_rate_pct", base_wr)
                        filt_trades= cal.get("filtered_trades", cal.get("n_trades", 0))
                        survival   = cal.get("survival_rate_pct", 100)
                        st.success(
                            f"✅ Profile saved for **{an_strategy}/{an_symbol}** — "
                            f"Win rate: **{base_wr:.1f}% → {filt_wr:.1f}%** (+{filt_wr-base_wr:.1f}pp)  |  "
                            f"Trades kept: {filt_trades} ({survival:.0f}% survival)  |  "
                            f"OOS verified: {'✅' if ver.get('verified') else '—'}"
                        )
                        if thr:
                            st.markdown("**Applied entry filters:**")
                            st.json(thr)
                        if ver.get("wfe_before") is not None:
                            vc1, vc2, vc3 = st.columns(3)
                            vc1.metric("WFE Before", f"{ver['wfe_before']:.3f}" if ver.get("wfe_before") else "—")
                            vc2.metric("WFE After",  f"{ver['wfe_after']:.3f}"  if ver.get("wfe_after")  else "—",
                                       delta="improved" if ver.get("verified") else "informational only")
                            vc3.metric("OOS CAGR After", f"{ver.get('oos_cagr_after', 0):.1f}%")
                    else:
                        st.warning(
                            cal.get("skip_reason") or
                            "Calibration ran but filters did not improve win rate by ≥3pp with ≥40% trade survival. "
                            "The strategy already performs well on this symbol — no filtering needed. "
                            "Try a longer period (8y/10y) for more historical trades."
                        )
                except Exception as e:
                    st.error(f"Optimization failed: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 9 — CONFIG
# ══════════════════════════════════════════════════════════════
with tab_config:
    st.subheader("Strategy Configuration")
    st.caption(
        "Enable/disable strategies and tune their parameters. "
        "Parameter changes take effect on the next Live Signals or Backtest run — no restart needed."
    )

    try:
        strats = api.perplexity_strategies()
        strat_enabled = {s["name"]: s["enabled"] for s in strats}
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    # Import strategy instances so we can read/write their config dicts
    try:
        from app.services.strategy.perplexity.runner import PERPLEXITY_STRATEGIES as _PXSTRATS
        _cfg_map = {s.name: s for s in _PXSTRATS}
    except Exception as e:
        st.error(f"Could not load strategy objects: {e}")
        st.stop()

    # ── Common risk parameters ────────────────────────────────
    st.markdown("### Common Risk Parameters")
    st.caption("These apply to all strategies for the backtest engine.")
    rc1, rc2 = st.columns(2)
    with rc1:
        risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25,
                             help="% of account risked per trade in backtests.")
    with rc2:
        max_pos_pct = st.slider("Max position size (% of capital)", 5, 50, 20, 5,
                                help="Hard cap — no single trade exceeds this % of capital.")
    st.caption(
        f"With $10,000 account: max risk/trade = **${10000 * risk_pct / 100:.0f}**  |  "
        f"max position = **${10000 * max_pos_pct / 100:.0f}**"
    )

    settings = get_settings()
    st.markdown("### Market Regime Settings")
    st.caption("Regime detection adjusts sizing and strategy aggressiveness using the benchmark trend.")
    mr1, mr2, mr3 = st.columns(3)
    with mr1:
        st.write("**Benchmark symbol**")
        st.write(settings.regime_benchmark_symbol)
        st.caption("Daily benchmark used for regime detection.")
    with mr2:
        st.write("**Bull sizing**")
        st.write(
            f"{settings.regime_risk_pct_bull*100:.1f}% risk / "
            f"{settings.regime_max_account_risk_bull*100:.0f}% max account exposure"
        )
    with mr3:
        st.write("**Bear / Deep Bear sizing**")
        st.write(
            f"Bear: {settings.regime_risk_pct_bear*100:.1f}% risk, "
            f"{settings.regime_max_account_risk_bear*100:.0f}% max exposure\n"
            f"Deep bear: {settings.regime_risk_pct_deep_bear*100:.2f}% risk, "
            f"{settings.regime_max_account_risk_deep_bear*100:.0f}% max exposure"
        )
    st.divider()

    # ── Per-strategy panels ───────────────────────────────────
    st.markdown("### Per-Strategy Parameters")

    for s_name, s_obj in _cfg_map.items():
        enabled = strat_enabled.get(s_name, True)
        with st.expander(
            f"{'✅' if enabled else '⬜'} **{s_name.replace('_', ' ')}**",
            expanded=False
        ):
            st.caption(STRATEGY_DESCRIPTIONS.get(s_name, ""))

            # Enable / disable toggle
            col_tog, _ = st.columns([1, 3])
            with col_tog:
                new_enabled = st.toggle("Enabled", value=enabled, key=f"tog_{s_name}")
                if new_enabled != enabled:
                    try:
                        api.perplexity_toggle_strategy(s_name, new_enabled)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            st.divider()
            cfg = s_obj.config
            c1, c2, c3 = st.columns(3)

            # ── Strategy-specific params ──────────────────────
            if s_name == "EMA_Mean_Reversion":
                st.caption("RSI(2) Mean Reversion — BULL only. Entry: RSI(2) < threshold. Exit: RSI(2) > exit OR close > SMA(5).")
                with c1:
                    cfg["rsi_entry_threshold"] = st.number_input(
                        "RSI(2) entry threshold (<)", 1, 20,
                        int(cfg.get("rsi_entry_threshold", 10)), 1, key=f"{s_name}_rsi_e",
                        help="Buy when RSI(2) falls below this. Lower = more extreme oversold, fewer trades.")
                    cfg["rsi_exit_threshold"] = st.number_input(
                        "RSI(2) exit threshold (>)", 50, 95,
                        int(cfg.get("rsi_exit_threshold", 70)), 5, key=f"{s_name}_rsi_x",
                        help="Exit when RSI(2) recovers above this. 70 = full mean reversion.")
                with c2:
                    cfg["hard_stop_pct"] = st.number_input(
                        "Hard stop (%)", 1.0, 10.0,
                        float(cfg.get("hard_stop_pct", 5.0)), 0.5, key=f"{s_name}_stp",
                        help="% below entry price for emergency stop.")
                    cfg["take_profit_pct"] = st.number_input(
                        "Take profit (%)", 2.0, 20.0,
                        float(cfg.get("take_profit_pct", 8.0)), 0.5, key=f"{s_name}_tp",
                        help="% above entry price for target.")
                with c3:
                    cfg["max_hold_bars"] = st.number_input(
                        "Max hold (bars)", 3, 30,
                        int(cfg.get("max_hold_bars", 10)), 1, key=f"{s_name}_hold")
                    cfg["atr_skip_threshold"] = st.number_input(
                        "ATR% spike skip (>)", 1.0, 10.0,
                        float(cfg.get("atr_skip_threshold", 5.0)), 0.5, key=f"{s_name}_atr_skip",
                        help="Skip entry if ATR% exceeds this — avoids buying into market panic.")

            elif s_name == "MA_Crossover_RSI":
                st.caption("EMA(9/21) crossover + MACD confirmation — BULL only. ATR-based stop/target.")
                with c1:
                    cfg["ema_fast_new"] = st.number_input(
                        "Fast EMA", 5, 20,
                        int(cfg.get("ema_fast_new", 9)), 1, key=f"{s_name}_ef")
                    cfg["ema_slow_new"] = st.number_input(
                        "Slow EMA", 15, 50,
                        int(cfg.get("ema_slow_new", 21)), 1, key=f"{s_name}_es")
                with c2:
                    cfg["rsi_min"] = st.number_input(
                        "RSI min at crossover", 30, 60,
                        int(cfg.get("rsi_min", 45)), 5, key=f"{s_name}_rl")
                    cfg["rsi_max"] = st.number_input(
                        "RSI max at crossover", 50, 80,
                        int(cfg.get("rsi_max", 65)), 5, key=f"{s_name}_rh")
                with c3:
                    cfg["atr_stop_multiplier"] = st.number_input(
                        "Stop ATR multiple", 0.5, 3.0,
                        float(cfg.get("atr_stop_multiplier", 1.5)), 0.25, key=f"{s_name}_atr_s")
                    cfg["atr_tp_multiplier"] = st.number_input(
                        "Target ATR multiple", 1.0, 5.0,
                        float(cfg.get("atr_tp_multiplier", 2.0)), 0.25, key=f"{s_name}_atr_t")
                cfg["vol_ratio_min"] = st.number_input(
                    "Min volume ratio at crossover", 0.5, 3.0,
                    float(cfg.get("vol_ratio_min", 1.1)), 0.1, key=f"{s_name}_vol",
                    help="Volume must be ≥ this × 20-day average at the crossover bar.")

            elif s_name == "Breakout_Consolidation":
                st.caption("BB Width squeeze → expansion breakout — BULL only. Close above upper BB after 5+ contracting bars.")
                with c1:
                    cfg["bb_period"] = st.number_input(
                        "BB period", 10, 30,
                        int(cfg.get("bb_period", 20)), 1, key=f"{s_name}_bbp")
                    cfg["squeeze_bars"] = st.number_input(
                        "Squeeze bars (consecutive narrowing)", 3, 10,
                        int(cfg.get("squeeze_bars", 5)), 1, key=f"{s_name}_sq",
                        help="Consecutive bars of narrowing BB width before watching for breakout.")
                with c2:
                    cfg["rsi_entry_min"] = st.number_input(
                        "RSI min at breakout", 40, 70,
                        int(cfg.get("rsi_entry_min", 50)), 5, key=f"{s_name}_rsi_e",
                        help="RSI must be above this at the breakout bar — below 50 lacks momentum.")
                    cfg["rsi_overbought"] = st.number_input(
                        "RSI exit (overbought)", 65, 90,
                        int(cfg.get("rsi_overbought", 80)), 5, key=f"{s_name}_rsi_x")
                with c3:
                    cfg["vol_ratio_min"] = st.number_input(
                        "Min volume ratio at breakout", 0.5, 3.0,
                        float(cfg.get("vol_ratio_min", 1.3)), 0.1, key=f"{s_name}_vol")
                    cfg["max_hold_bars"] = st.number_input(
                        "Max hold (bars)", 5, 30,
                        int(cfg.get("max_hold_bars", 15)), 1, key=f"{s_name}_hold")

            elif s_name == "BB_Mean_Reversion":
                st.caption("Pullback to rising EMA(50) with rejection wick — BULL + mild BEAR. Highest-frequency strategy.")
                with c1:
                    cfg["ema_trend"] = st.number_input(
                        "Trend EMA period", 20, 100,
                        int(cfg.get("ema_trend", 50)), 5, key=f"{s_name}_ema_t")
                    cfg["price_ema_proximity_pct"] = st.number_input(
                        "Price ↔ EMA proximity (%)", 0.2, 5.0,
                        float(cfg.get("price_ema_proximity_pct", 1.0)), 0.1, key=f"{s_name}_prox",
                        help="Price must be within this % of the EMA to qualify as a pullback entry.")
                with c2:
                    cfg["rsi_min"] = st.number_input(
                        "RSI min", 20, 50,
                        int(cfg.get("rsi_min", 35)), 5, key=f"{s_name}_rl")
                    cfg["rsi_max"] = st.number_input(
                        "RSI max", 40, 70,
                        int(cfg.get("rsi_max", 55)), 5, key=f"{s_name}_rh")
                with c3:
                    cfg["wick_ratio_min"] = st.number_input(
                        "Min lower wick ratio", 0.1, 0.8,
                        float(cfg.get("wick_ratio_min", 0.4)), 0.05, key=f"{s_name}_wick",
                        help="Lower wick must be ≥ this fraction of the bar's total range (bullish rejection).")
                    cfg["hard_stop_pct"] = st.number_input(
                        "Hard stop (%)", 0.5, 5.0,
                        float(cfg.get("hard_stop_pct", 2.0)), 0.25, key=f"{s_name}_stp")
                cfg["bear_skip_threshold_pct"] = st.number_input(
                    "Deep bear skip threshold (% below SMA200)", 5.0, 20.0,
                    float(cfg.get("bear_skip_threshold_pct", 10.0)), 1.0, key=f"{s_name}_bear_skip",
                    help="Skip entries when price is this far below SMA(200). Prevents buying falling knives.")

            elif s_name == "Fib_Pullback_Support":
                st.caption("ATR-spike / fear-capitulation reversal — BULL + mild BEAR. Buys panic selling with tight stops.")
                with c1:
                    cfg["atr_spike_threshold"] = st.number_input(
                        "ATR% spike threshold (>)", 1.0, 8.0,
                        float(cfg.get("atr_spike_threshold", 3.0)), 0.25, key=f"{s_name}_atr_e",
                        help="ATR% must exceed this to qualify as a fear/panic spike day.")
                    cfg["rsi_entry_max"] = st.number_input(
                        "RSI max at entry (<)", 15, 45,
                        int(cfg.get("rsi_entry_max", 30)), 5, key=f"{s_name}_rsi_e",
                        help="RSI must be below this — deeply oversold panic only.")
                with c2:
                    cfg["bb_pos_max"] = st.number_input(
                        "BB%B max (near lower band)", 0.05, 0.40,
                        float(cfg.get("bb_pos_max", 0.15)), 0.05, key=f"{s_name}_bb",
                        help="Price must be near the lower Bollinger Band (0=lower, 1=upper).")
                    cfg["wick_ratio_min"] = st.number_input(
                        "Min lower wick ratio", 0.2, 0.8,
                        float(cfg.get("wick_ratio_min", 0.5)), 0.05, key=f"{s_name}_wick")
                with c3:
                    cfg["hard_stop_pct"] = st.number_input(
                        "Hard stop (%)", 1.0, 8.0,
                        float(cfg.get("hard_stop_pct", 4.0)), 0.5, key=f"{s_name}_stp")
                    cfg["take_profit_pct"] = st.number_input(
                        "Take profit (%)", 2.0, 15.0,
                        float(cfg.get("take_profit_pct", 6.0)), 0.5, key=f"{s_name}_tp")
                cfg["prior_decline_pct"] = st.number_input(
                    "Prior N-bar decline required (%)", 0.5, 8.0,
                    float(cfg.get("prior_decline_pct", 2.0)), 0.25, key=f"{s_name}_dec",
                    help="Price must have declined at least this % over the prior 3 bars to confirm a selloff setup.")

            st.caption(
                f"Current config: `{cfg}`"
            )
