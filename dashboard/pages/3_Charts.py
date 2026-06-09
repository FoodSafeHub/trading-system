from __future__ import annotations

import streamlit as st

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section
import _charts as charts
import _lightweight_chart as lwc

# India detection — UI-only copy kept in sync with app.services.markets, so the
# dashboard never imports backend modules (same convention as 11_India.py).
_NIFTY_50 = {
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "HCLTECH", "BAJAJFINSV", "TECHM", "GRASIM",
    "INDUSINDBK", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "DIVISLAB", "HINDALCO", "BPCL", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN",
}
_INDIA_PREFIXES = ("NSE:", "BSE:")
_INDIA_SUFFIXES = (".NS", ".BO")

def _normalize_sym(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    for p in _INDIA_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    for suf in _INDIA_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s

def is_india_symbol(symbol: str) -> bool:
    s = (symbol or "").upper().strip()
    if s.startswith(_INDIA_PREFIXES) or s.endswith(_INDIA_SUFFIXES):
        return True
    return _normalize_sym(s) in _NIFTY_50

from _components import page_header, filter_cols  # noqa: E402

apply_theme("Charts")

from _sidebar import render_sidebar
render_sidebar()

page_header(
    "Price Charts",
    subtitle=(
        "Candlesticks powered by your market data — Schwab/yfinance for US, "
        "Upstox for India — with indicator overlays and strategy-signal markers. "
        "India symbols render in ₹."
    ),
)

sym_col, _ = st.columns([3, 7])
symbol = sym_col.text_input("Symbol", value="SPY", placeholder="AAPL, NVDA, SPY …").upper().strip() or "SPY"

chart_tab, live_tab, tv_tab = st.tabs(
    ["Chart (daily + signals)", "Strategy Live (intraday)", "TradingView (full UI)"]
)

with chart_tab:
    _india = is_india_symbol(symbol)
    _cur   = "₹" if _india else "$"
    _src   = "upstox / yfinance .NS" if _india else "yfinance"

    nc0, nc1, nc2 = st.columns([1.3, 1.3, 5])
    with nc0:
        _INTERVALS = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo", "Hourly (1h)": "1h"}
        nc_interval_label = st.selectbox("Candles", list(_INTERVALS), index=0)
        nc_interval = _INTERVALS[nc_interval_label]
    with nc1:
        # Hourly needs a short window; daily/weekly/monthly span wider.
        if nc_interval == "1h":
            _periods, _pidx = ["5d", "1mo", "3mo", "6mo"], 1
        else:
            _periods, _pidx = ["1mo", "3mo", "6mo", "1y", "2y", "5y"], 3
        nc_period = st.selectbox("Period", _periods, index=_pidx)
    with nc2:
        overlay_keys = st.multiselect(
            "Price overlays",
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

    oc1, oc2 = st.columns([5, 3])
    with oc1:
        osc_keys = st.multiselect(
            "Sub-pane indicators (stacked below price)",
            options=["rsi", "macd", "stoch", "atr", "obv"],
            default=["rsi", "macd"],
            format_func=lambda k: {
                "rsi": "RSI (14)", "macd": "MACD (12,26,9)", "stoch": "Stochastic (14,3)",
                "atr": "ATR (14)", "obv": "OBV",
            }[k],
        )
    with oc2:
        st.caption(
            f"Backend `/strategy/chart` (**{_src}**), same lightweight-charts engine as the live "
            "tab. ▲/▼ markers = recent strategy signals."
        )

    # More panes need more vertical room so price isn't squeezed.
    _chart_h = 720 + 130 * len(osc_keys)

    with st.spinner(f"Loading {symbol} {nc_interval_label.lower()} OHLCV…"):
        try:
            payload = api.chart_data(symbol, period=nc_period, interval=nc_interval)
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

        lwc.render_daily_chart(
            payload,
            overlays_enabled=overlay_keys,
            oscillators_enabled=osc_keys,
            trades=recent_signals,
            data_source=_src,
            currency=_cur,
            height=_chart_h,
        )
        st.markdown(
            f"<div style='text-align:right;margin-top:-8px'>"
            f"<a href='{charts.tradingview_url(symbol)}' target='_blank' "
            f"style='color:#42A5F5;font-size:12px;text-decoration:none'>"
            f"Open {charts.tv_symbol(symbol)} in TradingView ↗</a></div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"No OHLC data available for {symbol}.")

# ── Live strategy chart (lightweight-charts + backend overlays/markers) ──
with live_tab:
    st.caption(
        "Live candlestick chart powered by TradingView's lightweight-charts. "
        "**All overlays, markers, stops, and targets are computed by the backend** — "
        "this tab is a renderer, not a decision engine. Click any ▲/▼/✕ marker "
        "to see the strategy's entry/stop/target and the explanation (or rejection reason)."
    )

    lc1, lc2, lc3, lc4 = st.columns([1.2, 1.2, 3.2, 1.4])
    with lc1:
        tf = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1, key="live_tf")
    with lc2:
        refresh_secs = st.selectbox(
            "Refresh", [0, 5, 10, 30, 60],
            index=2,  # default 10s
            format_func=lambda s: "off" if s == 0 else f"{s}s",
            key="live_refresh",
        )
    with lc3:
        try:
            avail = [s["name"] for s in api.chart_strategies()]
        except Exception:
            avail = []
        strat_filter = st.multiselect(
            "Strategy filter (empty = all)",
            options=avail, default=[], key="live_strats",
        )
    with lc4:
        st.write("")
        if st.button("Refresh now", key="live_refresh_btn"):
            st.rerun()

    if tf == "1m":
        st.markdown(
            '<span style="background:#3a3526;color:#ffca28;border:1px solid #ffca28;'
            'padding:2px 8px;border-radius:10px;font-size:11px">'
            'signals computed on 5m / 15m</span>'
            '<span style="opacity:.65;margin-left:8px;font-size:12px">'
            'Candles render at 1m; markers anchor to the 1m bar containing each '
            'strategy signal.</span>',
            unsafe_allow_html=True,
        )

    ov1, ov2, ov3 = st.columns([3, 1.2, 1.2])
    with ov1:
        overlays_on = st.multiselect(
            "Overlays",
            options=["ema9", "ema21", "ema50", "vwap", "bb_upper", "bb_middle",
                     "bb_lower", "supertrend"],
            default=["ema9", "ema21", "vwap", "bb_upper", "bb_lower"],
            format_func=lambda k: {
                "ema9": "EMA 9", "ema21": "EMA 21", "ema50": "EMA 50",
                "vwap": "VWAP",
                "bb_upper": "BB Upper", "bb_middle": "BB Mid", "bb_lower": "BB Lower",
                "supertrend": "Supertrend",
            }[k],
            key="live_overlays",
        )
    with ov2:
        show_trades = st.checkbox("Show trades", value=True, key="live_show_trades")
    with ov3:
        show_rejected = st.checkbox("Show rejected", value=True, key="live_show_rej")

    strat_arg = ",".join(strat_filter) if strat_filter else "all"
    with st.spinner(f"Loading {symbol} {tf} chart…"):
        try:
            live_payload = api.intraday_chart(
                symbol, timeframe=tf, strategies=strat_arg,
                include_rejected=show_rejected,
            )
        except Exception as e:
            st.error(f"Live chart fetch failed: {e}")
            live_payload = None

    if live_payload:
        rgm = live_payload.get("regime") or "—"
        mkt = (live_payload.get("market_status") or {}).get("session_label", "—")
        diag = live_payload.get("diagnostics") or {}
        data_src = live_payload.get("data_source") or "—"
        fb_anchored = int(live_payload.get("fallback_anchored") or 0)
        mt1, mt2, mt3, mt4, mt5 = st.columns(5)
        mt1.metric("Symbol", live_payload.get("symbol", symbol))
        mt2.metric("Regime", rgm)
        mt3.metric("Accepted", len(live_payload.get("markers") or []))
        mt4.metric("Rejected", len(live_payload.get("rejected_markers") or []))
        mt5.metric("Data", data_src, delta=f"{fb_anchored} fallback" if fb_anchored else None,
                   delta_color="inverse" if fb_anchored else "off")

        if live_payload.get("policy_blocked"):
            st.warning(f"Policy blocked: {live_payload.get('policy_reason') or '—'}")
        if live_payload.get("warning"):
            st.info(live_payload["warning"])
        if fb_anchored:
            st.warning(
                f"{fb_anchored} marker(s) had an unparseable `signal_time` and were "
                "anchored to the latest bar. Check rows flagged `anchor_fallback` below."
            )

        lwc.render_strategy_chart(
            live_payload,
            overlays_enabled=overlays_on,
            show_trades=show_trades,
            show_rejected=show_rejected,
            show_levels=True,
            height=720,
        )
        st.markdown(
            f"<div style='text-align:right;margin-top:-8px'>"
            f"<a href='{charts.tradingview_url(symbol)}' target='_blank' "
            f"style='color:#42A5F5;font-size:12px;text-decoration:none'>"
            f"Open {charts.tv_symbol(symbol)} in TradingView ↗</a></div>",
            unsafe_allow_html=True,
        )

        # ── Explainability table: all markers in a sortable list ──
        rows = []
        for m in (live_payload.get("markers") or []):
            rows.append({**m, "status": "ACCEPTED"})
        for m in (live_payload.get("rejected_markers") or []):
            rows.append({**m, "status": "REJECTED"})
        if rows:
            import pandas as _pd
            df_rows = _pd.DataFrame(rows)
            keep = [c for c in ["status", "strategy", "side", "timeframe", "regime",
                                 "entry_price", "stop_price", "target_price",
                                 "confidence", "r_multiple", "anchor_fallback", "reason"]
                    if c in df_rows.columns]
            st.dataframe(df_rows[keep], use_container_width=True, hide_index=True)
        else:
            st.caption("No signals yet for this symbol/timeframe.")

        if diag and diag.get("root_cause"):
            with st.expander("Pipeline diagnostics", expanded=False):
                st.write(diag.get("root_cause"))
                for step in diag.get("diagnosis_steps", []):
                    st.text(f"· {step}")
    else:
        st.caption("No live payload available yet.")

    # Poll → just trigger another Streamlit rerun on the chosen interval.
    if refresh_secs and refresh_secs > 0:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=refresh_secs * 1000, key="live_chart_autorefresh")
        except Exception:
            # streamlit-autorefresh is optional; if missing, fall back to a
            # plain meta refresh so the page still polls.
            import streamlit.components.v1 as _components
            _components.html(
                f"<meta http-equiv='refresh' content='{refresh_secs}'>",
                height=0,
            )

# ── TradingView Advanced Chart (full pro UI) ──────────────────────
with tv_tab:
    _tv_sym = charts.tv_symbol(symbol)
    st.caption(
        "Full TradingView Advanced Chart — drawing tools, alerts, watchlists, "
        "multi-timeframe, and TradingView's own indicator library. "
        "**Independent from the engine**: no platform overlays or signal markers are shown here. "
        "For engine-faithful indicators + ▲/▼ markers + ENTRY/STOP/TARGET lines, use the other tabs."
    )
    st.markdown(
        f"<div style='margin-bottom:6px'>"
        f"Routed as <code>{_tv_sym}</code> · "
        f"<a href='{charts.tradingview_url(symbol)}' target='_blank' "
        f"style='color:#42A5F5;text-decoration:none'>Open {_tv_sym} in TradingView ↗</a></div>",
        unsafe_allow_html=True,
    )
    charts.tradingview_embed(_tv_sym, interval="D", height=720)

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

_CUR = "₹" if is_india_symbol(symbol) else "$"

def _fmt_large(v):
    if v is None: return "—"
    if v >= 1e12: return f"{_CUR}{v/1e12:.2f}T"
    if v >= 1e9:  return f"{_CUR}{v/1e9:.2f}B"
    if v >= 1e6:  return f"{_CUR}{v/1e6:.2f}M"
    return f"{_CUR}{v:,.0f}"

def _pct(v):
    return f"{v*100:.2f}%" if v is not None else "—"

def _val(v, fmt=None):
    if v is None: return "—"
    return fmt.format(v) if fmt else str(v)

name     = fund.get("company_name") or symbol
sector   = fund.get("sector")   or "—"
industry = fund.get("industry") or "—"
section(f"{name}  ({symbol})")
st.caption(f"**Sector:** {sector}  ·  **Industry:** {industry}")

st.markdown("#### Valuation")
vc = st.columns(5)
vc[0].metric("Market Cap",   _fmt_large(fund.get("market_cap")))
vc[1].metric("P/E (TTM)",    _val(fund.get("pe_ratio"),   "{:.2f}"))
vc[2].metric("Forward P/E",  _val(fund.get("forward_pe"), "{:.2f}"))
vc[3].metric("PEG Ratio",    _val(fund.get("peg_ratio"),  "{:.2f}"))
vc[4].metric("EPS (TTM)",    _val(fund.get("eps"),        _CUR + "{:.2f}"))

st.markdown("#### Price Statistics")
pc = st.columns(5)
pc[0].metric("52W High",       _val(fund.get("52w_high"),    _CUR + "{:.2f}"))
pc[1].metric("52W Low",        _val(fund.get("52w_low"),     _CUR + "{:.2f}"))
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
ac[1].metric("Price Target", f"{_CUR}{target:.2f}" if target else "—")
ac[2].metric("Upside",       f"{upside:+.1f}%" if upside is not None else "—")
