from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Perplexity Strategies", page_icon="🧠", layout="wide")
st.title("🧠 Perplexity Swing Strategies")
st.caption(
    "5 advanced swing trading strategies with SMA(200) trend filter, ATR-based stops, and defined targets. "
    "Designed for 3–10 day holds on daily bars."
)

STRATEGY_DESCRIPTIONS = {
    "High_Volume_Momentum_Breakout": "Price breaks above the 20-day high with volume > 1.5× average and RSI 50–75. "
                                      "Only trades when EMA(20) > EMA(50) — confirmed uptrend. "
                                      "Stop: 2×ATR below entry. Target: 2.5×ATR (2.5:1 R:R). "
                                      "Best on: NVDA, TSLA, GOOGL, META.",
    "Bollinger_Squeeze_Breakout":    "Waits for Bollinger Bands to squeeze to a 6-month low (low volatility), "
                                      "then buys the breakout above the upper band with volume confirmation. "
                                      "Stop: BB middle. Best on: MSFT, NVDA.",
    "MACD_RSI_Momentum":             "Enters when MACD is bullish (line > signal, histogram rising) and RSI > 50. "
                                      "Tighter than before — requires accelerating momentum. "
                                      "Stop: 1.5×ATR. Target: 3×ATR (3:1 R:R). "
                                      "Best on: GOOGL, NVDA, JPM, TSLA.",
    "EMA_Pullback_Support":          "Waits for price to pull back to EMA(20) and close above it on a bullish candle. "
                                      "Now requires EMA(20) > EMA(50) and RSI 40–65 to avoid overbought entries. "
                                      "Stop: below EMA(20). Best on: GOOGL, AAPL, JPM.",
    "Bollinger_Reversion_Uptrend":   "Buys when price touches below the lower BB then closes back inside — "
                                      "mean-reversion only in confirmed uptrends (close > SMA200). "
                                      "Partial exit signal at BB middle. Stop: 2×ATR. "
                                      "Best on: AAPL, NVDA, AMZN.",
}

SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "GOOGL"]

# ── Tab layout ────────────────────────────────────────────────
tab_signals, tab_sizer, tab_backtest, tab_compare, tab_config = st.tabs([
    "📡 Live Signals", "📐 Position Sizer", "🔬 Backtest", "📊 Compare All", "⚙️ Config"
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
                signals = api._get(f"/perplexity/signals/{sig_symbol}")
                st.session_state["px_signals"] = signals
            except Exception as e:
                st.error(f"Failed: {e}")

    sigs = st.session_state.get("px_signals")
    if sigs:
        buy_count  = sum(1 for s in sigs if s["direction"] == "BUY")
        sell_count = sum(1 for s in sigs if s["direction"] == "SELL")
        hold_count = sum(1 for s in sigs if s["direction"] == "HOLD")
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY signals",  buy_count)
        c2.metric("🔴 SELL signals", sell_count)
        c3.metric("⬜ HOLD",         hold_count)
        st.divider()

        for s in sigs:
            direction = s["direction"]
            name = s["strategy"]
            icon = "🟢" if direction == "BUY" else ("🔴" if direction == "SELL" else "⬜")
            conf_str = f"confidence: {s['confidence']:.0%}" if s["confidence"] else ""

            with st.expander(f"{icon} **{name}** — {direction}  |  {conf_str}",
                             expanded=(direction == "BUY")):
                st.caption(STRATEGY_DESCRIPTIONS.get(name, ""))
                if s["reason"]:
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
# TAB 2 — POSITION SIZER
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
                result = api._get(
                    f"/perplexity/size?symbol={sz_symbol}"
                    f"&entry_price={sz_entry}&stop_price={sz_stop}"
                    f"&account_value={sz_account}&risk_pct={sz_risk/100}"
                    f"&max_position_size_usd={sz_maxpos}"
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
        strat_list = api._get("/perplexity/strategies")
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
        bt_period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="px_period")
    with col4:
        bt_capital = st.number_input("Capital ($)", value=10000, min_value=1000, step=1000, key="px_cap")

    st.caption(STRATEGY_DESCRIPTIONS.get(chosen_strat, ""))

    run_bt = st.button("▶ Run Backtest", type="primary", key="px_run_bt")

    if run_bt:
        with st.spinner(f"Backtesting {chosen_strat} on {bt_symbol} over {bt_period}..."):
            try:
                r = api._get(
                    f"/perplexity/backtest/{chosen_strat}/{bt_symbol}"
                    f"?period={bt_period}&initial_capital={bt_capital}"
                )
                st.session_state["px_bt_result"] = r
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    r = st.session_state.get("px_bt_result")
    if r and not r.get("error"):
        st.divider()
        st.subheader(f"{r['strategy_name']} on {r['symbol']}  —  {r['start_date']} → {r['end_date']}")

        pnl = r["total_pnl"]
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Final Capital",   f"${r['final_capital']:,.0f}")
        c2.metric("Total P&L",       f"${pnl:,.2f}", delta=f"{r['total_return_pct']:+.2f}%",
                  delta_color="normal" if pnl >= 0 else "inverse")
        c3.metric("Win Rate",        f"{r['win_rate_pct']:.1f}%",
                  delta=f"{r['winning_trades']}W / {r['losing_trades']}L")
        c4.metric("Profit Factor",   r["profit_factor"] if r["profit_factor"] else "—")
        c5.metric("Max Drawdown",    f"{r['max_drawdown_pct']:.1f}%", delta_color="inverse")
        c6.metric("Sharpe",          r["sharpe_ratio"] if r["sharpe_ratio"] else "—")
        c7.metric("Trades",          r["total_trades"])

        # Equity curve
        if r.get("equity_curve"):
            eq_df = pd.DataFrame(r["equity_curve"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df["date"], y=eq_df["equity"],
                fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
                line=dict(color="#00d4aa", width=2), name="Portfolio Value",
            ))
            fig.add_hline(y=r["initial_capital"], line_dash="dash",
                          line_color="rgba(255,255,255,0.3)", annotation_text="Starting Capital")

            buys  = [t for t in r["trades"] if t["side"] == "BUY"]
            sells = [t for t in r["trades"] if "SELL" in t["side"]]
            if buys:
                bx = [t["date"] for t in buys]
                by = [next((e["equity"] for e in r["equity_curve"] if e["date"] == d), None) for d in bx]
                fig.add_trace(go.Scatter(x=bx, y=by, mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#00d4aa"), name="BUY"))
            if sells:
                sx = [t["date"] for t in sells]
                sy = [next((e["equity"] for e in r["equity_curve"] if e["date"] == d), None) for d in sx]
                fig.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="#ff4b4b"), name="SELL"))

            fig.update_layout(height=380, template="plotly_dark",
                              margin=dict(l=0, r=0, t=20, b=0),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis_tickprefix="$")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig, use_container_width=True)

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
        cmp_period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="cmp_period")
    with col3:
        cmp_capital = st.number_input("Capital ($)", value=10000, min_value=1000,
                                       step=1000, key="cmp_cap")

    run_cmp = st.button("▶ Compare All", type="primary", key="px_run_cmp")

    if run_cmp:
        with st.spinner(f"Running all 5 strategies on {cmp_symbol} over {cmp_period}..."):
            try:
                results = api._get(
                    f"/perplexity/backtest-all/{cmp_symbol}"
                    f"?period={cmp_period}&initial_capital={cmp_capital}"
                )
                st.session_state["px_compare"] = results
            except Exception as e:
                st.error(f"Comparison failed: {e}")

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
                             "Profit Factor": "—", "Return": "ERROR", "Total P&L": r["error"],
                             "Max Drawdown": "—", "Sharpe": "—", "Recommended": ""})
                continue
            pnl = r["total_pnl"]
            is_best = valid and r["strategy_name"] == best["strategy_name"]
            rows.append({
                "Strategy":      ("⭐ " if is_best else "") + r["strategy_name"],
                "Trades":        r["total_trades"],
                "Win Rate":      f"{r['win_rate_pct']:.1f}%",
                "Profit Factor": r["profit_factor"] if r["profit_factor"] else "—",
                "Return":        f"{r['total_return_pct']:+.2f}%",
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


# ══════════════════════════════════════════════════════════════
# TAB 4 — CONFIG (enable/disable strategies)
# ══════════════════════════════════════════════════════════════
with tab_config:
    st.subheader("Enable / Disable Strategies")
    st.caption("Changes apply immediately to live signals. Reload the page to confirm.")

    try:
        strats = api._get("/perplexity/strategies")
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    for s in strats:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{s['name']}**")
            st.caption(STRATEGY_DESCRIPTIONS.get(s["name"], ""))
        with col2:
            toggled = st.toggle("Enabled", value=s["enabled"], key=f"tog_{s['name']}")
            if toggled != s["enabled"]:
                try:
                    api._post(f"/perplexity/strategies/{s['name']}/toggle?enabled={str(toggled).lower()}", {})
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        st.divider()
