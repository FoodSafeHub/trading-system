from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Backtest", page_icon="🔬", layout="wide")
st.title("🔬 Strategy Backtester")
st.caption("Simulates how a strategy would have performed on historical data — no real money involved.")


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def _equity_chart(r: dict) -> None:
    if not r.get("equity_curve"):
        return
    st.divider()
    st.subheader("Equity Curve")
    eq_df = pd.DataFrame(r["equity_curve"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["date"], y=eq_df["equity"],
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        line=dict(color="#00d4aa", width=2),
        name="Portfolio Value",
    ))
    fig.add_hline(y=r["initial_capital"], line_dash="dash",
                  line_color="rgba(255,255,255,0.3)",
                  annotation_text="Starting Capital")

    trades = r.get("trades", [])
    buys  = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if "SELL" in str(t["side"])]

    if buys:
        buy_dates  = [t["date"] for t in buys]
        buy_equity = [next((e["equity"] for e in r["equity_curve"] if e["date"] == d), None) for d in buy_dates]
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_equity, mode="markers",
            marker=dict(symbol="triangle-up", size=10, color="#00d4aa"),
            name="BUY",
        ))

    if sells:
        sell_dates  = [t["date"] for t in sells]
        sell_equity = [next((e["equity"] for e in r["equity_curve"] if e["date"] == d), None) for d in sell_dates]
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_equity, mode="markers",
            marker=dict(symbol="triangle-down", size=10, color="#ff4b4b"),
            name="SELL",
        ))

    fig.update_layout(
        height=400, template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickprefix="$",
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)


def _side_tag(v: str) -> str:
    if str(v) == "BUY":       return "🟢 BUY"
    if "SELL" in str(v):      return "🔴 SELL"
    return str(v)


def _pnl_tag(v) -> str:
    if v is None: return "—"
    return f"🟢 +${v:,.2f}" if v >= 0 else f"🔴 -${abs(v):,.2f}"


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
    st.dataframe(df, use_container_width=True, hide_index=True)


def _consensus_trades_table(trades: list) -> None:
    st.divider()
    st.subheader(f"All Trades ({len(trades)} total)")
    if not trades:
        st.info("No trades were generated.")
        return

    rows = [{
        "date":      t.get("date"),
        "side":      _side_tag(t.get("side", "")),
        "price":     f"${t['price']:,.2f}" if t.get("price") is not None else "—",
        "quantity":  t.get("quantity"),
        "value":     f"${t['value']:,.2f}" if t.get("value") is not None else "—",
        "agreed by": ", ".join(t.get("agreeing", [])) or "—",
        "profit / loss": _pnl_tag(t.get("pnl")),
    } for t in trades]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# MODE TOGGLE
# ══════════════════════════════════════════════════════════════
mode = st.radio(
    "Backtest Mode",
    ["Single Strategy", "Consensus Mode"],
    horizontal=True,
    help="Single: test one strategy alone.\n"
         "Consensus: test a symbol using all its strategies with agreement filtering.",
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

    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    with col1:
        chosen = st.selectbox("Strategy", strategy_names)
    with col2:
        period = st.selectbox("Historical Period", ["6mo", "1y", "2y"], index=1)
    with col3:
        capital = st.number_input("Starting Capital ($)", value=100000, min_value=1000, step=10000)
    with col4:
        st.write("")
        st.write("")
        run = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if not run and "bt_result" not in st.session_state:
        st.info("Select a strategy and click Run Backtest to see historical performance.")
        st.stop()

    if run:
        with st.spinner(f"Running backtest for {chosen} over {period}..."):
            try:
                result = api._get(f"/backtest/run/{chosen}?period={period}&initial_capital={capital}&quantity=1")
                st.session_state["bt_result"] = result
                st.session_state.pop("bt_consensus", None)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

    r = st.session_state.get("bt_result")
    if not r:
        st.stop()

    st.subheader(f"Results — {r['strategy_name']} ({r['start_date']} → {r['end_date']})")

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

    _equity_chart(r)
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

    info = symbol_info.get(chosen_sym, {})
    strat_count = info.get("strategy_count", 0)
    strat_names = info.get("strategies", [])
    st.caption(f"**{chosen_sym}** has **{strat_count} strategies**: {', '.join(strat_names)}")

    min_agreement = st.slider(
        "Minimum strategies that must agree before placing a trade",
        min_value=1,
        max_value=max(strat_count, 1),
        value=min(2, strat_count),
        help=(
            "1 = any single strategy fires a trade (most trades, more noise).\n"
            "2 = two strategies must agree (balanced — this is the live default).\n"
            "3 = all three must agree (fewest trades, highest confidence)."
        ),
    )

    agree_labels = {1: "any signal fires (high noise)", 2: "two must agree (balanced)", 3: "all must agree (high confidence)"}
    st.info(
        f"**min_agreement = {min_agreement}** — {agree_labels.get(min_agreement, '')}. "
        f"A trade is placed only when **{min_agreement} out of {strat_count}** strategies "
        f"give the same BUY or SELL signal on the same day."
    )

    run_c = st.button("▶ Run Consensus Backtest", type="primary")

    if not run_c and "bt_consensus" not in st.session_state:
        st.stop()

    if run_c:
        with st.spinner(f"Running consensus backtest for {chosen_sym} (need {min_agreement} to agree)..."):
            try:
                result = api._get(
                    f"/backtest/consensus/{chosen_sym}"
                    f"?min_agreement={min_agreement}&period={period}&initial_capital={capital}"
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

    if r["total_trades"] == 0:
        st.warning(
            f"No trades were placed with min_agreement={r['min_agreement']}. "
            "The strategies never agreed enough times in this period. "
            "Try lowering the agreement level or selecting a longer period (2y)."
        )

    _equity_chart(r)
    _consensus_trades_table(r["trades"])
