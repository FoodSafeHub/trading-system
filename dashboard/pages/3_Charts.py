from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme

apply_theme("Charts")
st.title("Price Charts")

col1, col2 = st.columns([2, 6])
with col1:
    symbol = st.text_input("Symbol", value="SPY", placeholder="AAPL, NVDA, SPY …").upper().strip() or "SPY"
with col2:
    st.write("")
    st.caption("Change the symbol above to update financials. To change the chart symbol without losing indicators, use the search bar **inside** the chart. Log in to TradingView inside the chart to save your indicator layout permanently.")

# ── Build watchlist: assigned symbols first, then defaults ────────
_EXCHANGE_MAP = {
    "SPY": "AMEX", "QQQ": "NASDAQ", "IWM": "AMEX",
}
_DEFAULT_WATCHLIST = ["AMEX:SPY","NASDAQ:QQQ","NASDAQ:AAPL","NASDAQ:NVDA",
                      "NASDAQ:MSFT","NASDAQ:TSLA","NASDAQ:AMZN","NASDAQ:META",
                      "NASDAQ:GOOGL","NYSE:JPM","NASDAQ:AMD","NYSE:NFLX"]

try:
    _assignments = api.list_assignments()
    _assigned_syms = [a["symbol"].upper() for a in _assignments if a.get("enabled")]
except Exception:
    _assigned_syms = []

def _tv_sym(sym):
    if sym in _EXCHANGE_MAP:
        return f"{_EXCHANGE_MAP[sym]}:{sym}"
    # NYSE-listed ETFs and financials vs NASDAQ tech — best-effort default
    _nyse = {"JPM","BAC","GS","MS","WFC","XOM","CVX","JNJ","UNH","V","MA"}
    return f"NYSE:{sym}" if sym in _nyse else f"NASDAQ:{sym}"

_assigned_tv  = [_tv_sym(s) for s in _assigned_syms]
_extra        = [s for s in _DEFAULT_WATCHLIST if not any(s.endswith(f":{sym}") for sym in _assigned_syms)]
_watchlist    = _assigned_tv + _extra
_watchlist_js = str(_watchlist).replace("'", '"')

# ── TradingView Advanced Chart ────────────────────────────────────
import json as _json

_tv_config = _json.dumps({
    "autosize": True,
    "symbol": "SPY",
    "interval": "D",
    "timezone": "America/New_York",
    "theme": "dark",
    "style": "1",
    "locale": "en",
    "backgroundColor": "rgba(19,23,34,1)",
    "gridColor": "rgba(255,255,255,0.06)",
    "hide_top_toolbar": False,
    "hide_legend": False,
    "range": "YTD",
    "allow_symbol_change": True,
    "save_image": True,
    "watchlist": _watchlist,
    "studies": [
        {"name": "Moving Average Exponential", "override": {"length": 9,   "linecolor": "#26C6DA", "linewidth": 1}},
        {"name": "Moving Average Exponential", "override": {"length": 21,  "linecolor": "#FF9800", "linewidth": 1}},
        {"name": "Moving Average Exponential", "override": {"length": 50,  "linecolor": "#AB47BC", "linewidth": 1}},
        {"name": "Moving Average Exponential", "override": {"length": 200, "linecolor": "#EF5350", "linewidth": 2}},
        {"name": "Bollinger Bands",            "override": {"length": 20, "mult": 2}},
        {"name": "VWAP",                       "override": {}},
        {"name": "Supertrend",                 "override": {"Factor": 3, "ATR Length": 10}},
        {"name": "Relative Strength Index",    "override": {"length": 14}},
        {"name": "MACD",                       "override": {"fast length": 12, "slow length": 26, "signal smoothing": 9}},
        {"name": "Stochastic RSI",             "override": {}},
        {"name": "Average True Range",         "override": {"length": 14}},
        {"name": "On Balance Volume",          "override": {}},
        {"name": "Volume",                     "override": {}},
    ],
    "support_host": "https://www.tradingview.com",
})

components.html(f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>html,body{{margin:0;padding:0;height:100%;overflow:hidden;background:transparent;}}</style>
</head>
<body>
  <div class="tradingview-widget-container" style="height:860px;width:100%">
    <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
    <div class="tradingview-widget-copyright">
      <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
        <span class="blue-text">Track all markets on TradingView</span>
      </a>
    </div>
  </div>
  <script type="text/javascript"
    src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {_tv_config}
  </script>
</body>
</html>""", height=880, scrolling=False)

# ── Financials ────────────────────────────────────────────────────
st.divider()

with st.spinner(f"Loading {symbol} fundamentals…"):
    try:
        data_fin = api.chart_data(symbol, period="1mo")
        fund = data_fin.get("fundamentals", {})
    except Exception:
        fund = {}

if not fund:
    st.caption(f"No fundamental data available for {symbol}.")
    st.stop()

def _fmt_large(v):
    if v is None: return "—"
    if v >= 1e12: return f"${v/1e12:.2f}T"
    if v >= 1e9:  return f"${v/1e9:.2f}B"
    if v >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def _pct(v):
    return f"{v*100:.2f}%" if v is not None else "—"

def _val(v, fmt=None):
    if v is None: return "—"
    return fmt.format(v) if fmt else str(v)

name     = fund.get("company_name") or symbol
sector   = fund.get("sector")   or "—"
industry = fund.get("industry") or "—"
st.subheader(f"{name}  ({symbol})")
st.caption(f"**Sector:** {sector}  ·  **Industry:** {industry}")

st.markdown("#### Valuation")
vc = st.columns(5)
vc[0].metric("Market Cap",   _fmt_large(fund.get("market_cap")))
vc[1].metric("P/E (TTM)",    _val(fund.get("pe_ratio"),   "{:.2f}"))
vc[2].metric("Forward P/E",  _val(fund.get("forward_pe"), "{:.2f}"))
vc[3].metric("PEG Ratio",    _val(fund.get("peg_ratio"),  "{:.2f}"))
vc[4].metric("EPS (TTM)",    _val(fund.get("eps"),        "${:.2f}"))

st.markdown("#### Price Statistics")
pc = st.columns(5)
pc[0].metric("52W High",       _val(fund.get("52w_high"),    "${:.2f}"))
pc[1].metric("52W Low",        _val(fund.get("52w_low"),     "${:.2f}"))
pc[2].metric("Beta",           _val(fund.get("beta"),        "{:.2f}"))
pc[3].metric("Dividend Yield", _pct(fund.get("dividend_yield")))
pc[4].metric("Short Ratio",    _val(fund.get("short_ratio"), "{:.2f}x"))

st.markdown("#### Fundamentals")
fc = st.columns(5)
fc[0].metric("Revenue",       _fmt_large(fund.get("revenue")))
fc[1].metric("Profit Margin", _pct(fund.get("profit_margin")))
fc[2].metric("Debt/Equity",   _val(fund.get("debt_to_equity"), "{:.2f}"))
fc[3].metric("ROE",           _pct(fund.get("roe")))
fc[4].metric("Avg Volume",    f"{fund.get('avg_volume'):,}" if fund.get("avg_volume") else "—")

st.markdown("#### Analyst Consensus")
ac = st.columns(3)
rating     = (fund.get("analyst_rating") or "—").replace("_", " ").title()
target     = fund.get("analyst_target")
last_price = next((v for v in reversed(data_fin.get("close", [])) if v), None)
upside     = round((target - last_price) / last_price * 100, 1) if target and last_price else None
ac[0].metric("Rating",       rating)
ac[1].metric("Price Target", f"${target:.2f}" if target else "—")
ac[2].metric("Upside",       f"{upside:+.1f}%" if upside is not None else "—")
