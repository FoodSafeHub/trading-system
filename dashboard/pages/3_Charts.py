from __future__ import annotations

import streamlit as st

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme
import _charts as charts

apply_theme("Charts")
st.title("Price Charts")

col1, col2 = st.columns([2, 6])
with col1:
    symbol = st.text_input("Symbol", value="SPY", placeholder="AAPL, NVDA, SPY …").upper().strip() or "SPY"
with col2:
    st.write("")
    st.caption(
        "Two engines: **TradingView** for live streaming candlesticks + drawing tools, and "
        "**Schwab-style native candles** (Plotly) for our own indicators and recent strategy signals. "
        "Use the search bar inside the TradingView chart to change symbol without losing studies."
    )

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
    _nyse = {"JPM","BAC","GS","MS","WFC","XOM","CVX","JNJ","UNH","V","MA"}
    return f"NYSE:{sym}" if sym in _nyse else f"NASDAQ:{sym}"

_assigned_tv  = [_tv_sym(s) for s in _assigned_syms]
_extra        = [s for s in _DEFAULT_WATCHLIST if not any(s.endswith(f":{sym}") for sym in _assigned_syms)]
_watchlist    = _assigned_tv + _extra

tv_tab, native_tab = st.tabs(["TradingView Advanced", "Native Candles + Signals"])

with tv_tab:
    tv_symbol = _tv_sym(symbol)
    charts.tradingview_embed(tv_symbol, interval="D", watchlist=_watchlist, height=820)

with native_tab:
    nc1, nc2, nc3 = st.columns([1.5, 1.5, 5])
    with nc1:
        nc_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with nc2:
        overlay_keys = st.multiselect(
            "Indicator overlays",
            options=["ema9", "ema21", "ema50", "ema200", "vwap",
                     "bb_upper", "bb_lower", "bb_middle", "supertrend",
                     "sma50", "sma200"],
            default=["ema21", "ema50", "vwap", "bb_upper", "bb_lower"],
            format_func=lambda k: {
                "ema9": "EMA 9", "ema21": "EMA 21", "ema50": "EMA 50", "ema200": "EMA 200",
                "vwap": "VWAP",
                "bb_upper": "Bollinger ↑", "bb_lower": "Bollinger ↓", "bb_middle": "Bollinger mid",
                "supertrend": "Supertrend",
                "sma50": "SMA 50", "sma200": "SMA 200",
            }[k],
        )
    with nc3:
        st.caption(
            "Loads OHLCV from our backend `/strategy/chart/{symbol}` and renders a multi-pane "
            "candle stack — Price + Volume + RSI(14) + MACD. Recent strategy signals (if available) "
            "are overlaid as ▲/▼ markers on the candle at the fill price."
        )

    with st.spinner(f"Loading {symbol} OHLCV…"):
        try:
            payload = api.chart_data(symbol, period=nc_period)
        except Exception as e:
            st.error(f"Could not load chart data: {e}")
            payload = None

    if payload and payload.get("dates"):
        # Pull recent strategy signals for this symbol so traders can read each
        # fill on the candle — falls back silently if none exist yet.
        recent_signals: list[dict] = []
        try:
            sigs = api.signals() or []
            for s in sigs:
                if str(s.get("symbol", "")).upper() != symbol:
                    continue
                d = (s.get("ts") or s.get("timestamp") or s.get("created_at") or "")[:10]
                if not d:
                    continue
                recent_signals.append({
                    "date":  d,
                    "side":  str(s.get("direction") or s.get("side") or "").upper(),
                    "price": s.get("price") or s.get("entry_price") or s.get("close") or None,
                })
            # If we don't have a price, anchor to that day's close.
            close_by_date = dict(zip(payload["dates"], payload["close"]))
            for r in recent_signals:
                if r["price"] is None:
                    r["price"] = close_by_date.get(r["date"])
        except Exception:
            recent_signals = []

        charts.render_price_chart(
            payload,
            trades=recent_signals,
            overlays=tuple(overlay_keys),
            include_volume=True,
            include_rsi=True,
            include_macd=True,
            title=f"{symbol} — {nc_period}",
        )
    else:
        st.caption(f"No OHLC data available for {symbol}.")

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
