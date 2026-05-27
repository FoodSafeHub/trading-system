from __future__ import annotations

"""India (Zerodha) cockpit — a single page bundling everything India-specific:

  * Daily-login status + nudges (orders via Zerodha, data via Upstox)
  * Positions / account for the Zerodha broker, formatted in ₹
  * Nifty 50 scanner (runs through the same /scanner API, prices in ₹)
  * Backtest on any NSE symbol (data flows through the exchange-aware provider)

Orders route to Zerodha automatically for India symbols (see app.services.markets
+ the scheduler's auto-route). US workflow stays on the other pages.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")

import pandas as pd
import streamlit as st

import api
from _theme import apply_theme, section, kpi_row, money, market_status_bar

apply_theme("India (Zerodha)")
st.title("🇮🇳 India — Zerodha")
st.caption(
    "India-only cockpit. Orders route to Zerodha; market data comes from Upstox "
    "(or yfinance .NS as a free fallback). All values shown in ₹."
)

market_status_bar()

RUPEE = "₹"


# ── India momentum regime banner ──────────────────────────────────────────────
# Momentum strategies (Supertrend, BB breakout, pattern setups) only have an edge
# when the Nifty itself is trending. This reads the Nifty-50/India-VIX regime so
# you know at a glance whether to deploy momentum or stand down.
def _momentum_banner() -> None:
    snap = _safe(lambda: api.momentum_regime("india"), None)
    if not snap:
        return
    regime = snap.get("regime", "")
    idx = snap.get("spy_close")        # ^NSEI close (field name is generic)
    sma200 = snap.get("spy_sma200")
    vix = snap.get("vix")
    breadth = snap.get("breadth_pct")
    bits = []
    if idx and sma200:
        rel = "above" if idx > sma200 else "below"
        bits.append(f"Nifty {idx:,.0f} {rel} 200-DMA {sma200:,.0f}")
    if vix is not None:
        bits.append(f"India VIX {vix:.1f}")
    if breadth is not None:
        bits.append(f"breadth {breadth:.0f}%")
    detail = " · ".join(bits)
    if regime == "bull_momentum":
        st.success(f"🟢 **India momentum: ON** (full size) — {detail}")
    elif regime == "bull_caution":
        st.warning(f"🟡 **India momentum: CAUTION** (half size) — {detail}")
    elif regime == "bear_momentum":
        st.warning(f"🔴 **India momentum: SHORTS ONLY** — {detail}")
    else:
        st.info(f"⚪ **India momentum: OFF** (stand down — chop/transitional) — {detail}")


# Nifty 50 — kept in sync with app.services.markets.NIFTY_50. UI-only copy so
# the dashboard doesn't import backend modules.
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE", "AXISBANK", "ASIANPAINT",
    "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ONGC",
    "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "HCLTECH", "BAJAJFINSV", "TECHM", "GRASIM",
    "INDUSINDBK", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "DIVISLAB", "HINDALCO", "BPCL", "APOLLOHOSP", "BAJAJ-AUTO", "TATACONSUM",
    "SBILIFE", "HDFCLIFE", "LTIM", "SHRIRAMFIN",
]


def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


def _inr(v) -> str:
    return money(v, currency=RUPEE) if v is not None else "—"


# Render the momentum banner now that _safe is defined (it sits visually at the
# top of the page, just under the market status bar).
_momentum_banner()


# The 7 scanner strategy types (5 regime-aware + 2 legacy) — same set the US
# Backtest page exposes. Used for both compare-all and single-strategy modes.
SCANNER_STRATEGIES: list[tuple[str, str]] = [
    ("rsi2_mean_reversion",  "RSI-2 Mean Reversion"),
    ("ema_macd_crossover",   "EMA + MACD Crossover"),
    ("bb_squeeze_breakout",  "Bollinger Squeeze Breakout"),
    ("pullback_ema50",       "Pullback to EMA(50)"),
    ("vix_spike_reversal",   "VIX Spike Reversal"),
    ("bollinger",            "Legacy: Bollinger Mean Reversion"),
    ("fib_pullback",         "Legacy: Fibonacci Pullback"),
]
_SCANNER_LABEL = {t: lbl for t, lbl in SCANNER_STRATEGIES}


def _num(d: dict, *keys):
    """First present numeric value among keys, else None."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return v
    return None


def _render_bt_metrics(m: dict) -> None:
    """KPI row from a backtest result dict, tolerant of differing field names.
    Money values shown in ₹ since India bars are rupee-denominated."""
    total_ret = _num(m, "total_return_pct", "total_return")
    win_rate = _num(m, "win_rate_pct", "win_rate")
    trades = _num(m, "total_trades", "num_trades")
    final_eq = _num(m, "final_capital", "final_equity", "ending_equity")
    max_dd = _num(m, "max_drawdown_pct", "max_drawdown")
    kpi_row([
        ("Total return", f"{total_ret:.1f}%" if total_ret is not None else "—"),
        ("Win rate",     f"{win_rate:.0f}%" if win_rate is not None else "—"),
        ("Trades",       str(int(trades)) if trades is not None else "—"),
        ("Final capital", _inr(final_eq) if final_eq is not None else "—"),
        ("Max drawdown", f"{max_dd:.1f}%" if max_dd is not None else "—"),
    ])


def _side_tag_inr(v: str) -> str:
    if str(v) == "BUY":  return "🟢 BUY"
    if "SELL" in str(v): return "🔴 SELL"
    return str(v)


def _pnl_tag_inr(v) -> str:
    # Entry rows carry no realised P/L; None upcasts to NaN inside a DataFrame,
    # so guard for both to avoid rendering "-₹nan" on BUY rows.
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"🟢 +{_inr(v)}" if v >= 0 else f"🔴 -{_inr(abs(v))}"


def _render_trades_inr(trades: list) -> None:
    """Trade-by-trade table, ₹-formatted, matching the US Backtest layout:
    colored side tags + a derived realised profit/loss column.

    Scanner trades carry only date/side/price/quantity/value (no per-trade
    P/L), so pair each SELL with its preceding BUY by the value delta. Falls
    back to whatever columns exist for Perplexity-shaped (entry/exit) trades."""
    if not trades:
        st.info("No trades were generated in this period. Try a longer period or another strategy.")
        return

    sample = trades[0]
    # Scanner shape: BUY/SELL legs with a `value` we can difference into P/L.
    if "side" in sample and "value" in sample and "pnl" not in sample:
        rows = []
        pending_buy = None
        for t in trades:
            if t.get("side") == "BUY":
                pending_buy = t
                rows.append({**t, "profit / loss": None})
            elif "SELL" in str(t.get("side", "")):
                pnl = round(t["value"] - pending_buy["value"], 2) if pending_buy else None
                rows.append({**t, "profit / loss": pnl})
                pending_buy = None
            else:
                rows.append({**t, "profit / loss": None})
        df = pd.DataFrame(rows)
        df["side"] = df["side"].apply(_side_tag_inr)
        for c in ("price", "value"):
            if c in df.columns:
                df[c] = df[c].apply(lambda v: _inr(v) if isinstance(v, (int, float)) else "—")
        df["profit / loss"] = df["profit / loss"].apply(_pnl_tag_inr)
        for col in ("signal_from", "regime"):
            if col in df.columns:
                df = df.drop(columns=[col])
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    # Generic / Perplexity shape: format money + percent columns in place.
    df = pd.DataFrame(trades)
    money_cols = [c for c in df.columns if c.lower() in (
        "price", "value", "entry_price", "exit_price", "pnl", "pnl_inr",
        "entry", "exit", "stop", "target", "capital")]
    for c in money_cols:
        df[c] = df[c].apply(lambda v: _inr(v) if isinstance(v, (int, float)) else (v if v is not None else "—"))
    for c in [c for c in df.columns if c.lower() in ("pnl_pct", "return_pct", "r_multiple")]:
        df[c] = df[c].apply(lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else (v if v is not None else "—"))
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Daily login status ──────────────────────────────────────────────────────
# Zerodha + Upstox tokens both expire daily (SEBI rule, no refresh). Surface a
# clear re-login nudge with the right links.
section("Daily authorization")
st.caption(
    "Both India connections need a fresh login each trading day — Zerodha for "
    "orders (~07:30 IST expiry), Upstox for data (~03:30 IST expiry). No "
    "auto-refresh is allowed by SEBI."
)
lc1, lc2 = st.columns(2)
with lc1:
    st.markdown("**Zerodha (orders)**")
    st.link_button("Log in to Zerodha", f"{api.BASE}/zerodha/login", use_container_width=True)
    st.caption("Opens Kite login; redirects back and stores today's token.")
with lc2:
    st.markdown("**Upstox (market data)**")
    st.link_button("Log in to Upstox", f"{api.BASE}/upstox/login", use_container_width=True)
    st.caption("Free India data. If not yet configured, the page falls back to yfinance .NS.")


# ── India positions / account ────────────────────────────────────────────────
section("India positions (Zerodha)")
# Query Zerodha directly (not the global-routing endpoints) so India data shows
# regardless of where the global toggle points. Both degrade to [] on failure.
india_positions = _safe(lambda: api.broker_positions("zerodha"), []) or []
india_accts = _safe(lambda: api.broker_account_summary("zerodha"), []) or []

if india_accts:
    eq = sum(a.get("equity") or 0 for a in india_accts) or None
    cash = sum(a.get("cash") or 0 for a in india_accts) or None
    bp = sum(a.get("buying_power") or 0 for a in india_accts) or None
    kpi_row([
        ("Equity",       _inr(eq)),
        ("Cash",         _inr(cash)),
        ("Buying power", _inr(bp)),
        ("Open positions", str(len(india_positions))),
    ])
    acct_ids = ", ".join(str(a.get("account_id") or "") for a in india_accts if a.get("account_id"))
    if acct_ids:
        st.caption(f"Account {acct_ids}")
else:
    st.info(
        "No Zerodha account data yet. Log in above, then ensure the API has been "
        "restarted since the India integration. Cash/positions appear once "
        "Zerodha authenticates."
    )

if india_positions:
    df = pd.DataFrame(india_positions)
    keep = [c for c in ["symbol", "quantity", "average_cost", "current_price",
                        "market_value", "unrealized_pnl"] if c in df.columns]
    if keep:
        df = df[keep]
        for col in ("average_cost", "current_price", "market_value", "unrealized_pnl"):
            if col in df.columns:
                df[col] = df[col].apply(_inr)
        df = df.rename(columns={
            "symbol": "Symbol", "quantity": "Qty", "average_cost": "Avg cost",
            "current_price": "Last", "market_value": "Value",
            "unrealized_pnl": "Unrealised P&L",
        })
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No open India positions.")


# ── Tabs: Scanner + Backtest ──────────────────────────────────────────────────
tab_scan, tab_bt = st.tabs(["Nifty 50 Scanner", "Backtest (NSE)"])

with tab_scan:
    st.caption(
        "Scans an India universe for strategy signals — same engine as the US scanner, "
        "but orders route to Zerodha. Data via Upstox. Prices in ₹. "
        "Wider universes (Nifty 500 / All NSE) take longer and run in the background."
    )
    _UNIVERSE_LABELS = {
        "nifty50":  "Nifty 50",
        "nifty100": "Nifty 100",
        "nifty200": "Nifty 200",
        "nifty500": "Nifty 500",
        "nse_all":  "All NSE (~2,466)",
    }
    su0, sc1, sc2, sc3 = st.columns([1.4, 1, 1, 1])
    with su0:
        scan_universe = st.selectbox(
            "Universe", list(_UNIVERSE_LABELS),
            format_func=lambda k: _UNIVERSE_LABELS[k],
            key="india_scan_universe",
            help="Curated Nifty tiers are fast and liquid. 'All NSE' sweeps every "
                 "Upstox-resolvable equity — most thorough but slow and noisier.",
        )
    with sc1:
        min_price = st.number_input("Min price (₹)", min_value=1.0, value=50.0, step=10.0,
                                    key="india_scan_minprice")
    with sc2:
        direction = st.radio("Direction", ["ANY", "BUY", "SELL"], index=0,
                             horizontal=True, key="india_scan_dir")
    with sc3:
        top_n = st.slider("Top N", 1, 20, 5, key="india_scan_topn")

    _u_label = _UNIVERSE_LABELS[scan_universe]
    if st.button(f"🔍 Scan {_u_label}", key="india_scan_btn", type="primary"):
        config = {
            "universe": scan_universe,
            "custom_symbols": [],
            "min_price": float(min_price),
            "min_avg_volume": 0.0,   # India volumes differ; don't over-filter
            "top_n": int(top_n),
            "scan_direction": direction,
            "auto_trade_top": False,
            "auto_trade_direction": "ANY",
            "batch_size": 20,
        }
        with st.spinner(f"Scanning {_u_label} (runs in background)..."):
            try:
                api.scanner_run(config)
                st.info(f"{_u_label} scan started in the background. Click 'Refresh results' shortly.")
            except Exception as e:
                st.error(f"Scan failed to start: {e}")

    if st.button("🔄 Refresh results", key="india_scan_refresh"):
        st.rerun()

    _india_universes = set(_UNIVERSE_LABELS)
    results = _safe(lambda: api.scanner_results(limit=50), []) or []
    india_results = [r for r in results if (r.get("universe") or "") in _india_universes]
    if india_results:
        rows = [{
            "Symbol":      r.get("symbol"),
            "Direction":   r.get("direction"),
            "Score":       r.get("score"),
            "Agree":       r.get("strategies_agreeing"),
            "Price":       _inr(r.get("price")),
            "Strategy":    r.get("strategy_name"),
        } for r in india_results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No Nifty 50 scan results yet — run a scan above.")

with tab_bt:
    st.caption(
        "Backtest any NSE symbol. Pick a **strategy source** (the 7 scanner "
        "strategies or the Perplexity set) and a **mode** (compare all, or one "
        "strategy). Data flows through the exchange-aware provider "
        "(Upstox → yfinance .NS); results shown in ₹."
    )

    # Controls row 1: source + mode.
    mc1, mc2 = st.columns(2)
    with mc1:
        bt_source = st.radio(
            "Strategy source", ["Scanner (7 strategies)", "Perplexity strategies"],
            horizontal=True, key="india_bt_source",
            help="Scanner = the 7 regime-aware/legacy strategies. Perplexity = the "
                 "Perplexity strategy set. Both fetch India data the same way.",
        )
    with mc2:
        bt_mode = st.radio(
            "Mode", ["Compare all", "Single strategy"],
            horizontal=True, key="india_bt_mode",
            help="Compare all = run every strategy in the source and rank them. "
                 "Single = deep-dive one strategy with full trade details.",
        )

    is_perplexity = bt_source.startswith("Perplexity")

    # Controls row 2: symbol (free-type any NSE ticker), period, capital.
    bc1, bc2, bc3 = st.columns([2, 1, 1])
    with bc1:
        bt_symbol = (st.text_input(
            "NSE symbol", value="RELIANCE", key="india_bt_sym",
            placeholder="Any NSE ticker — RELIANCE, DMART, IRCTC, POLYCAB …",
            help="Type any of the ~2,466 NSE equities. Validated against Upstox. "
                 "Quick-pick a Nifty 50 name below if you'd rather browse.",
        ) or "").upper().strip()
    with bc2:
        bt_period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1, key="india_bt_period")
    with bc3:
        bt_capital = st.number_input("Capital (₹)", min_value=1000, value=100000,
                                     step=10000, key="india_bt_capital")

    # Quick-pick row: drop a Nifty 50 name into the box without typing.
    qp = st.selectbox("…or quick-pick a Nifty 50 name", ["—"] + NIFTY_50,
                      key="india_bt_quickpick")
    if qp != "—" and qp != bt_symbol:
        st.session_state["india_bt_sym"] = qp
        st.rerun()

    # Validate the ticker against the Upstox instrument map so a typo fails fast
    # rather than producing an empty/garbage backtest.
    if bt_symbol:
        _res = _safe(lambda: api.upstox_resolve(bt_symbol), {}) or {}
        if _res.get("tradeable"):
            st.caption(f"✅ **{bt_symbol}** resolves on NSE (Upstox).")
        else:
            st.warning(
                f"⚠️ **{bt_symbol}** didn't resolve in the Upstox NSE map. It may be "
                "renamed/delisted, or the symbol differs from the NSE tradingsymbol. "
                "The backtest will fall back to yfinance .NS, which may return no data."
            )

    # Strategy picker only in single mode.
    single_strategy = None
    if bt_mode == "Single strategy":
        if is_perplexity:
            perp_names = _safe(lambda: [s["name"] for s in api.perplexity_strategies()
                                        if isinstance(s, dict) and "name" in s], [])
            if not perp_names:
                st.warning("Could not load Perplexity strategy names from the API.")
            single_strategy = st.selectbox("Perplexity strategy", perp_names or ["—"],
                                           key="india_bt_perp_pick")
        else:
            single_strategy = st.selectbox(
                "Scanner strategy", [t for t, _ in SCANNER_STRATEGIES],
                format_func=lambda t: _SCANNER_LABEL.get(t, t),
                index=3, key="india_bt_scan_pick",
            )

    if st.button("📊 Run backtest", key="india_bt_btn", type="primary"):
        with st.spinner(f"Backtesting {bt_symbol} ({bt_period})..."):
            try:
                if is_perplexity and bt_mode == "Compare all":
                    res = ("perp_all", api.perplexity_backtest_all(
                        bt_symbol, period=bt_period, initial_capital=float(bt_capital)))
                elif is_perplexity:
                    res = ("perp_one", api.perplexity_backtest(
                        single_strategy, bt_symbol, period=bt_period,
                        initial_capital=float(bt_capital)))
                elif bt_mode == "Compare all":
                    res = ("scan_all", api.backtest_custom_compare_all(
                        bt_symbol, period=bt_period, initial_capital=float(bt_capital)))
                else:
                    res = ("scan_one", api.backtest_run_generic(
                        bt_symbol, single_strategy, period=bt_period,
                        initial_capital=float(bt_capital)))
                st.session_state["india_bt_result"] = res
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.session_state.pop("india_bt_result", None)

    stored = st.session_state.get("india_bt_result")
    if stored:
        kind, res = stored
        st.divider()

        # ── Compare-all: a ranked table of every strategy ──────────────────
        if kind in ("perp_all", "scan_all"):
            rows = res if isinstance(res, list) else res.get("results", res)
            if isinstance(rows, dict):
                rows = rows.get("results", [])
            if not rows:
                st.info("No results returned. Try a longer period.")
            else:
                # Rank by total return (best first), keep only ones that ran.
                ranked = sorted(
                    [r for r in rows if not r.get("error")],
                    key=lambda r: _num(r, "total_return_pct", "total_return") or float("-inf"),
                    reverse=True,
                )
                errored = [r for r in rows if r.get("error")]
                best_name = ranked[0].get("strategy_name") if ranked else None

                table = []
                for r in ranked:
                    pnl = _num(r, "total_pnl")
                    sh = _num(r, "sharpe_ratio")
                    name = r.get("strategy_name") or r.get("name") or "—"
                    table.append({
                        "Strategy":    ("🏆 " + name) if name == best_name else name,
                        "Total return": (f"{_num(r, 'total_return_pct', 'total_return'):+.1f}%"
                                         if _num(r, 'total_return_pct', 'total_return') is not None else "—"),
                        "Total P&L":   (_inr(pnl) if pnl is not None and pnl >= 0
                                        else (f"-{_inr(abs(pnl))}" if pnl is not None else "—")),
                        "Win rate":    (f"{_num(r, 'win_rate_pct', 'win_rate'):.0f}%"
                                        if _num(r, 'win_rate_pct', 'win_rate') is not None else "—"),
                        "Trades":      (int(_num(r, 'total_trades', 'num_trades'))
                                        if _num(r, 'total_trades', 'num_trades') is not None else "—"),
                        "Profit factor": (f"{_num(r, 'profit_factor'):.2f}"
                                          if _num(r, 'profit_factor') is not None else "—"),
                        "Sharpe":      (f"{sh:.2f}" if sh is not None else "—"),
                        "Max DD":      (f"{_num(r, 'max_drawdown_pct', 'max_drawdown'):.1f}%"
                                        if _num(r, 'max_drawdown_pct', 'max_drawdown') is not None else "—"),
                    })
                st.subheader(f"Compare All — {bt_symbol} ({bt_period})")
                st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
                if errored:
                    st.caption("⚠️ Did not run: " + ", ".join(
                        f"{r.get('strategy_name', '?')} ({r['error']})" for r in errored))

                # Per-strategy trade drill-down — inspect any strategy's trades
                # without switching to Single mode.
                with_trades = [r for r in ranked if r.get("trades")]
                if with_trades:
                    st.markdown("**Trade details**")
                    pick = st.selectbox(
                        "Strategy to inspect", with_trades,
                        format_func=lambda r: f"{r.get('strategy_name', '?')} "
                                              f"({len(r.get('trades', []))} trades)",
                        key="india_cmp_pick",
                    )
                    if pick:
                        _render_bt_metrics(pick)
                        _render_trades_inr(pick.get("trades", []))

        # ── Single strategy: metrics + full trade-by-trade details ─────────
        else:
            m = res if isinstance(res, dict) else {}
            st.subheader(f"{m.get('strategy_name', single_strategy)} — {bt_symbol} ({bt_period})")
            if m.get("start_date") and m.get("end_date"):
                st.caption(f"{m['start_date']} → {m['end_date']}")
            _render_bt_metrics(m)
            st.markdown(f"**All trades ({len(m.get('trades', []))} total)**")
            _render_trades_inr(m.get("trades", []))
            with st.expander("Raw backtest response"):
                st.json(res)
