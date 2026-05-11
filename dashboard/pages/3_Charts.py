from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api

st.set_page_config(page_title="Charts", page_icon="📊", layout="wide")
st.title("📊 Price Charts & Indicators")

# ── Controls ──────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    symbol = st.text_input("Symbol", value="SPY").upper()
with col2:
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
with col3:
    st.write("")
    st.write("")
    load = st.button("Load Chart", type="primary", use_container_width=True)

if not load and "chart_data" not in st.session_state:
    st.info("Enter a symbol and click Load Chart.")
    st.stop()

if load:
    with st.spinner(f"Fetching {symbol} data..."):
        try:
            data = api._get(f"/strategy/chart/{symbol}?period={period}")
            st.session_state["chart_data"] = data
            st.session_state["chart_symbol"] = symbol
        except Exception as e:
            st.error(f"Failed to load chart data: {e}")
            st.stop()

data = st.session_state.get("chart_data")
if not data:
    st.stop()

dates = data["dates"]
ind = data["indicators"]

# ── Overlay toggles ───────────────────────────────────────────
st.subheader(f"{data['symbol']} — {period}")
c1, c2, c3, c4, c5 = st.columns(5)
show_sma = c1.checkbox("SMA 10/30", value=True)
show_ema = c2.checkbox("EMA 9", value=False)
show_bb  = c3.checkbox("Bollinger Bands", value=True)
show_vol = c4.checkbox("Volume", value=True)
show_macd = c5.checkbox("MACD", value=False)

# ── Build figure ──────────────────────────────────────────────
row_count = 2  # price + RSI always
if show_vol:
    row_count += 1
if show_macd:
    row_count += 1

row_heights = [0.5, 0.2]
if show_vol:
    row_heights.append(0.15)
if show_macd:
    row_heights.append(0.15)

subplot_titles = [data["symbol"], "RSI (14)"]
if show_vol:
    subplot_titles.append("Volume")
if show_macd:
    subplot_titles.append("MACD")

fig = make_subplots(
    rows=row_count, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=subplot_titles,
    row_heights=row_heights,
)

# ── Candlestick ───────────────────────────────────────────────
fig.add_trace(go.Candlestick(
    x=dates, open=data["open"], high=data["high"],
    low=data["low"], close=data["close"],
    name=data["symbol"],
    increasing_line_color="#00d4aa",
    decreasing_line_color="#ff4b4b",
), row=1, col=1)

if show_sma:
    fig.add_trace(go.Scatter(x=dates, y=ind["sma10"], name="SMA 10",
                             line=dict(color="#f7c948", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ind["sma30"], name="SMA 30",
                             line=dict(color="#a78bfa", width=1.5)), row=1, col=1)

if show_ema:
    fig.add_trace(go.Scatter(x=dates, y=ind["ema9"], name="EMA 9",
                             line=dict(color="#38bdf8", width=1.5, dash="dot")), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=dates, y=ind["bb_upper"], name="BB Upper",
                             line=dict(color="rgba(148,163,184,0.6)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ind["bb_middle"], name="BB Mid",
                             line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ind["bb_lower"], name="BB Lower",
                             line=dict(color="rgba(148,163,184,0.6)", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(148,163,184,0.05)"), row=1, col=1)

# ── RSI ───────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=dates, y=ind["rsi14"], name="RSI 14",
                         line=dict(color="#f97316", width=1.5)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,75,75,0.5)", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,212,170,0.5)", row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,75,75,0.05)", line_width=0, row=2, col=1)
fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,212,170,0.05)", line_width=0, row=2, col=1)

# ── Volume ────────────────────────────────────────────────────
next_row = 3
if show_vol:
    colors = ["#00d4aa" if c >= o else "#ff4b4b"
              for c, o in zip(data["close"], data["open"])]
    fig.add_trace(go.Bar(x=dates, y=data["volume"], name="Volume",
                         marker_color=colors, opacity=0.6), row=next_row, col=1)
    next_row += 1

# ── MACD ──────────────────────────────────────────────────────
if show_macd:
    hist = ind["macd_hist"]
    hist_colors = ["#00d4aa" if (v or 0) >= 0 else "#ff4b4b" for v in hist]
    fig.add_trace(go.Bar(x=dates, y=hist, name="MACD Hist",
                         marker_color=hist_colors, opacity=0.7), row=next_row, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ind["macd"], name="MACD",
                             line=dict(color="#38bdf8", width=1.5)), row=next_row, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ind["macd_signal"], name="Signal",
                             line=dict(color="#f97316", width=1.5)), row=next_row, col=1)

# ── Layout ────────────────────────────────────────────────────
fig.update_layout(
    height=700,
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

st.plotly_chart(fig, use_container_width=True)

# ── Latest indicator values ───────────────────────────────────
st.divider()
st.subheader("Latest Values")
latest = {}
for key, vals in ind.items():
    clean = [v for v in vals if v is not None]
    latest[key] = round(clean[-1], 2) if clean else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("SMA 10", latest.get("sma10"))
c2.metric("SMA 30", latest.get("sma30"))
c3.metric("EMA 9",  latest.get("ema9"))
rsi_val = latest.get("rsi14")
rsi_label = " 🔴 Overbought" if rsi_val and rsi_val > 70 else (" 🟢 Oversold" if rsi_val and rsi_val < 30 else "")
c4.metric("RSI 14", f"{rsi_val}{rsi_label}" if rsi_val else "—")
c5.metric("MACD", latest.get("macd"))
