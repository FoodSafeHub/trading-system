from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme
import _charts as charts

import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _fmt_et(ts) -> str:
    """Convert any timestamp (str/datetime, naive or UTC-aware) to 'YYYY-MM-DD HH:MM:SS ET'."""
    if ts is None or ts == "":
        return "—"
    try:
        dt = pd.to_datetime(ts, utc=True)
        return dt.tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return str(ts)[:19].replace("T", " ")


apply_theme("Market Scanner")
st.title("Market Scanner")
st.caption("Scans a universe of stocks for strategy signals, scores them, and shows the top candidates.")


_EXCHANGE_GUESS = {
    "SPY": "AMEX", "QQQ": "NASDAQ", "IWM": "AMEX",
}
_NYSE_HINTS = {"JPM","BAC","GS","MS","WFC","XOM","CVX","JNJ","UNH","V","MA"}


def _tv_symbol(sym: str) -> str:
    """Backwards-compat shim. Routing now lives in charts.tv_symbol so the
    Scanner and the Charts page share one rule (incl. India NSE:/BSE: routing)."""
    return charts.tv_symbol(sym)


def _parse_strategies_from_reason(reason: str | None) -> list[str]:
    """Extract strategy names from the scanner's reason string.

    Reason format: "N strategies agree: A, B, C +K more" — we want A, B, C
    so we can compare each against the recommended winner for the symbol.
    """
    if not reason or ":" not in reason:
        return []
    tail = reason.split(":", 1)[1]
    # Drop the "+K more" suffix if present
    tail = tail.split(" +")[0]
    return [s.strip() for s in tail.split(",") if s.strip()]


def _show_candidates(candidates, *, key_prefix: str = "cands"):
    # Direction filter — lets the user focus on buy-only or sell-only signals
    # without re-running the scan. ANY shows everything.
    direction_pick = st.radio(
        "Direction filter",
        ["All", "BUY", "SELL"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_dirfilter",
    )
    if direction_pick != "All":
        candidates = [c for c in candidates if str(c.get("direction", "")).upper() == direction_pick]
        if not candidates:
            st.info(f"No {direction_pick} candidates in this result set.")
            return

    # Pull the cached "best historical strategy per symbol" so rows whose
    # firing strategy matches the historical winner can be starred.
    try:
        recs = {r["symbol"]: r for r in (api.recommendations_list() or [])}
    except Exception:
        recs = {}

    rows = []
    for c in candidates:
        direction = c.get("direction", "")
        sym = c.get("symbol", "")
        rec = recs.get(sym)
        firing_strats = _parse_strategies_from_reason(c.get("reason"))
        is_match = bool(rec) and any(s == rec["strategy_name"] for s in firing_strats)

        symbol_label = f"⭐ {sym}" if is_match else sym
        reason_label = c.get("reason", "—") or "—"
        if is_match:
            reason_label = f"★ MATCHES RECOMMENDED — {reason_label}"

        if rec:
            wr = rec.get("win_rate_pct")
            pf = rec.get("profit_factor")
            tr = rec.get("total_return_pct")
            best_label = rec["strategy_name"].replace("_", " ")
            metrics_bits = []
            if wr is not None: metrics_bits.append(f"WR {wr:.0f}%")
            if pf is not None: metrics_bits.append(f"PF {pf:.2f}")
            if tr is not None: metrics_bits.append(f"{tr:+.0f}%")
            best_full = best_label + (f" ({', '.join(metrics_bits)})" if metrics_bits else "")
        else:
            best_full = "—"

        rows.append({
            "Symbol":       symbol_label,
            "Direction":    direction,
            "Score":        f"{c['score']:.0f} / 100",
            "Strategies":   c["strategies_agreeing"],
            "Price":        f"${c['price']:,.2f}" if c.get("price") else "—",
            "Avg Volume":   f"{int(c['avg_volume'] or 0):,}" if c.get("avg_volume") else "—",
            "Universe":     c.get("universe", "—"),
            "Reason":       reason_label,
            "Best (hist.)": best_full,
            "Auto-Traded":  "yes" if c.get("auto_traded") else "—",
            "Scanned At":   _fmt_et(c.get("scanned_at")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Recompute recommendations for the symbols on screen ─────────────────
    cand_symbols = sorted({c["symbol"] for c in candidates if c.get("symbol")})
    missing = [s for s in cand_symbols if s not in recs]
    rc_cols = st.columns([3, 2, 2])
    with rc_cols[0]:
        st.caption(
            f"{len(recs)} symbol(s) cached · {len(missing)} of {len(cand_symbols)} "
            f"on-screen symbol(s) have no recommendation yet."
        )
    with rc_cols[1]:
        period_pick = st.selectbox(
            "Backtest period", ["2y", "5y", "1y"], index=1,
            key=f"{key_prefix}_rec_period",
            help="Period passed to Compare All. 5y is the dashboard default.",
        )
    with rc_cols[2]:
        if missing and st.button(
            f"⚙ Recompute {len(missing)} missing", key=f"{key_prefix}_rec_missing",
            use_container_width=True,
        ):
            with st.spinner(
                f"Running Compare All on {len(missing)} symbol(s) — "
                "this takes ~30–90s each."
            ):
                try:
                    api.recommendations_recompute_many(missing, period=period_pick)
                    st.success(f"Recomputed {len(missing)} recommendation(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Recompute failed: {exc}")

    # ── Inline candle drill-in for any candidate ──────────────────────────
    if not candidates:
        return
    symbols = sorted({c["symbol"] for c in candidates if c.get("symbol")})
    if not symbols:
        return
    st.markdown("#### Inspect a candidate on candles")
    pick = st.selectbox(
        "Symbol", symbols, index=0, key=f"{key_prefix}_pick",
        help="Pulls daily OHLCV from the backend and renders candles + EMAs/VWAP/Bollinger "
              "plus RSI and MACD panes.",
    )
    view = st.radio(
        "View", ["Native candles + indicators", "TradingView mini"],
        horizontal=True, key=f"{key_prefix}_view",
    )
    if view == "TradingView mini":
        charts.tradingview_mini(_tv_symbol(pick), height=320)
    else:
        try:
            payload = api.chart_data(pick, period="6mo")
        except Exception as e:
            st.caption(f"Could not load OHLC for {pick}: {e}")
            return
        if not payload or not payload.get("dates"):
            st.caption(f"No OHLC available for {pick}.")
            return

        # Pin a marker at today's bar for the scanned direction so traders see
        # *where* the scanner says to act.
        cand = next((c for c in candidates if c["symbol"] == pick), {})
        scan_trade = []
        if cand.get("price"):
            scan_trade.append({
                "date":  payload["dates"][-1],
                "side":  "BUY" if str(cand.get("direction", "")).upper() == "BUY" else "SELL",
                "price": cand["price"],
            })
        charts.render_price_chart(
            payload,
            trades=scan_trade,
            overlays=("ema21", "ema50", "vwap", "bb_upper", "bb_lower"),
            include_volume=True, include_rsi=True, include_macd=True,
            title=f"{pick} — last 6mo (scanner candidate)",
        )


# ── Scanner status ─────────────────────────────────────────────────────────
try:
    status = api._get("/scanner/status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Scanner", "Running..." if status["running"] else "Ready")
    c2.metric("Last Scan", _fmt_et(status["last_scan"]) if status["last_scan"] else "Never")
    c3.metric("Last Matches", status["last_matches"] or 0)
except Exception as e:
    st.warning(f"Cannot reach scanner API: {e}")

st.divider()

# ── Scan config ────────────────────────────────────────────────────────────
st.subheader("Run a Scan")

col1, col2 = st.columns([2, 2])
with col1:
    universe = st.selectbox(
        "Universe",
        ["watchlist", "sp500", "nasdaq100", "nifty50", "custom"],
        format_func=lambda v: "nifty50 (India)" if v == "nifty50" else v,
        help="watchlist = your current strategies.json stocks. sp500/nasdaq100 = full US index scan (~2–5 min). "
             "nifty50 = India NSE top-50 (orders route to Zerodha; prices in ₹).",
    )
    custom_input = ""
    if universe == "custom":
        custom_input = st.text_area(
            "Custom symbols (one per line or comma-separated)",
            placeholder="AAPL\nTSLA\nNVDA",
        )

with col2:
    # India (Nifty 50) prices/volumes are in ₹ and are much larger in absolute
    # terms than US-dollar thresholds, so default the filters lower for India.
    _is_india_scan = universe == "nifty50"
    _price_label = "Min price (₹)" if _is_india_scan else "Min price ($)"
    _price_default = 50.0 if _is_india_scan else 5.0
    min_price = st.number_input(_price_label, min_value=1.0, value=_price_default, step=1.0)
    min_volume = st.number_input("Min avg daily volume", min_value=0, value=500000, step=100000)
    scan_direction = st.radio(
        "Scan direction",
        ["ANY", "BUY", "SELL"],
        index=0,
        horizontal=True,
        help="ANY ranks BUY + SELL together. BUY or SELL drops the other side entirely so "
             "the Top N window is filled exclusively with the requested direction.",
    )
    top_n = st.slider("Top N candidates to return", min_value=1, max_value=20, value=5)
    # The Scanner is DISCOVERY ONLY. The auto-scheduler (every 15 min) is the
    # sole execution authority. Scanner candidates write signal rows that the
    # scheduler picks up on its next cycle if the symbol is assigned. The old
    # "Auto-trade top candidate" toggle was removed because it placed orders
    # in parallel with the scheduler, with a separate (and weaker) cap +
    # SELL-policy contract -- which is how BNY got flattened by an
    # unintended MARKET sell on 2026-06-04.
    auto_trade = False
    auto_trade_direction = "ANY"
    st.caption(
        "ℹ Scanner is discovery only. Candidates surface as signals; the "
        "auto-scheduler (every 15 min) is the sole execution authority and "
        "trades only on assignments you've configured."
    )

run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🔭 Run Scan Now", type="primary", use_container_width=True)

if run_btn:
    custom_symbols = []
    if universe == "custom" and custom_input:
        custom_symbols = [s.strip().upper() for s in custom_input.replace(",", "\n").splitlines() if s.strip()]

    config_payload = {
        "universe": universe,
        "custom_symbols": custom_symbols,
        "min_price": min_price,
        "min_avg_volume": min_volume,
        "top_n": top_n,
        "scan_direction": scan_direction,
        "auto_trade_top": auto_trade,
        "auto_trade_direction": auto_trade_direction,
        "batch_size": 20,
    }

    is_large = universe in ("sp500", "nasdaq100", "nifty50") or len(custom_symbols) > 20

    with st.spinner(f"Scanning {universe} universe... {'(running in background for large universe)' if is_large else ''}"):
        try:
            result = api._post("/scanner/run", json=config_payload)
            if result.get("scan_run_id") == "pending":
                st.info("Large universe scan is running in the background. Refresh this page in 2–5 minutes to see results.")
            else:
                st.success(
                    f"✅ Scan complete in {result['duration_seconds']}s — "
                    f"scanned {result['total_scanned']} symbols, "
                    f"{result['total_passed_filters']} passed filters, "
                    f"{result['total_matches']} matches found."
                )
                if result["top_candidates"]:
                    st.subheader(f"Top {len(result['top_candidates'])} Candidates")
                    _show_candidates(result["top_candidates"], key_prefix="run_cands")
                else:
                    st.warning("No symbols met the signal criteria in this scan.")
        except Exception as e:
            st.error(f"Scan failed: {e}")

st.divider()

# ── Latest scan results from DB ────────────────────────────────────────────
st.subheader("Recent Scan Results")

try:
    results = api._get("/scanner/results?limit=50")
    if results:
        _show_candidates(results, key_prefix="recent_cands")
    else:
        st.info("No scan results yet. Run a scan above or wait for the auto-scheduler "
                "(watchlist every 15 min, S&P 500 + NASDAQ 100 every 4 h, market hours only).")
except Exception as e:
    st.warning(f"Could not load scan results: {e}")

st.divider()

# ── Info box ───────────────────────────────────────────────────────────────
with st.expander("How the scanner works"):
    st.markdown("""
**Scoring system (0–100):**

| Factor | Points |
|---|---|
| Each strategy agreeing (up to 3) | 20 pts each |
| Perplexity strategy agreement | 15 pts |
| Avg volume > 2M | 10 pts |
| Price above 50-day SMA (uptrend) | 10 pts |

**Scan flow:**
1. Build symbol universe (watchlist / S&P 500 / NASDAQ 100 / custom)
2. Apply liquidity filters: min price, min avg volume, min history
3. Run all 5 Bollinger strategy types + 5 Perplexity strategies per symbol
4. Symbols where 1+ strategies agree on BUY or SELL are candidates
5. Score and rank — top N returned
6. Results saved to database

**Auto-scan:** During market hours (9:30am–4pm ET) the scheduler runs
- watchlist every 15 minutes
- NASDAQ 100 every 4 hours
- S&P 500 every 4 hours (staggered 10 min after NASDAQ 100)

**Large universe scans** (S&P 500 = ~500 stocks) take 2–5 minutes and run in the background.
    """)
