from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, divider, pill, empty_state
from _components import page_header, stat_band, filter_cols
import _charts as charts

import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _fmt_et(ts) -> str:
    if ts is None or ts == "":
        return "—"
    try:
        dt = pd.to_datetime(ts, utc=True)
        return dt.tz_convert(ET).strftime("%H:%M ET")
    except Exception:
        return str(ts)[:19].replace("T", " ")


def _parse_strategies_from_reason(reason: str | None) -> list[str]:
    if not reason or ":" not in reason:
        return []
    tail = reason.split(":", 1)[1]
    tail = tail.split(" +")[0]
    return [s.strip() for s in tail.split(",") if s.strip()]


def _show_candidates(candidates, *, key_prefix: str = "cands"):
    dir_col, _ = st.columns([2, 6])
    direction_pick = dir_col.radio(
        "Direction filter", ["All", "BUY", "SELL"],
        index=0, horizontal=True, key=f"{key_prefix}_dirfilter",
    )
    if direction_pick != "All":
        candidates = [c for c in candidates if str(c.get("direction", "")).upper() == direction_pick]
        if not candidates:
            empty_state(
                f"No {direction_pick} candidates",
                "Try switching to 'All' or re-running the scan.",
                icon="🔍",
            )
            return

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

        dir_icon = "🟢" if direction == "BUY" else ("🔴" if direction == "SELL" else "")
        match_icon = "⭐ " if is_match else ""
        reason_label = c.get("reason", "—") or "—"
        if is_match:
            reason_label = f"★ matches recommended — {reason_label}"

        if rec:
            wr = rec.get("win_rate_pct")
            pf = rec.get("profit_factor")
            tr = rec.get("total_return_pct")
            best_label = rec["strategy_name"].replace("_", " ")
            bits = []
            if wr is not None: bits.append(f"WR {wr:.0f}%")
            if pf is not None: bits.append(f"PF {pf:.2f}")
            if tr is not None: bits.append(f"{tr:+.0f}%")
            best_full = best_label + (f" ({', '.join(bits)})" if bits else "")
        else:
            best_full = "—"

        rows.append({
            "Symbol":       f"{match_icon}{sym}",
            "Dir":          f"{dir_icon} {direction}",
            "Score":        c.get("score", 0),
            "Strategies":   c.get("strategies_agreeing", 0),
            "Price":        c.get("price") or None,
            "Avg Vol":      c.get("avg_volume") or None,
            "Universe":     c.get("universe", "—"),
            "Reason":       reason_label,
            "Best (hist.)": best_full,
            "Auto-Traded":  "yes" if c.get("auto_traded") else "—",
            "Scanned":      _fmt_et(c.get("scanned_at")),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol":       st.column_config.TextColumn("Symbol",       width="small"),
            "Dir":          st.column_config.TextColumn("Dir",          width="small"),
            "Score":        st.column_config.NumberColumn("Score",      format="%d / 100", width="small"),
            "Strategies":   st.column_config.NumberColumn("Agree",      format="%d", width="small"),
            "Price":        st.column_config.NumberColumn("Price",      format="$%.2f", width="small"),
            "Avg Vol":      st.column_config.NumberColumn("Avg Vol",    format="%d", width="medium"),
            "Universe":     st.column_config.TextColumn("Universe",     width="small"),
            "Reason":       st.column_config.TextColumn("Reason",       width="large"),
            "Best (hist.)": st.column_config.TextColumn("Best (hist.)", width="large"),
            "Auto-Traded":  st.column_config.TextColumn("Auto",        width="small"),
            "Scanned":      st.column_config.TextColumn("Scanned",     width="small"),
        },
    )

    # Recompute recommendations for on-screen symbols
    cand_symbols = sorted({c["symbol"] for c in candidates if c.get("symbol")})
    missing = [s for s in cand_symbols if s not in recs]
    rc1, rc2, rc3 = st.columns([4, 2, 2])
    rc1.caption(
        f"{len(recs)} cached · {len(missing)} of {len(cand_symbols)} on-screen missing recommendation"
    )
    period_pick = rc2.selectbox(
        "Backtest period", ["2y", "5y", "1y"], index=1, key=f"{key_prefix}_rec_period",
    )
    if missing and rc3.button(
        f"⚙ Compute {len(missing)} missing", key=f"{key_prefix}_rec_missing", use_container_width=True,
    ):
        with st.spinner(f"Running Compare All on {len(missing)} symbol(s)…"):
            try:
                api.recommendations_recompute_many(missing, period=period_pick)
                st.success(f"Recomputed {len(missing)} recommendation(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Recompute failed: {exc}")

    # Inline candle drill-in
    symbols = sorted({c["symbol"] for c in candidates if c.get("symbol")})
    if not symbols:
        return
    section("Inspect Candidate", level=3)
    pick_col, view_col = st.columns([2, 2])
    pick = pick_col.selectbox("Symbol", symbols, key=f"{key_prefix}_pick")
    view = view_col.radio(
        "View", ["Native candles", "TradingView mini"],
        horizontal=True, key=f"{key_prefix}_view",
    )
    if view == "TradingView mini":
        charts.tradingview_mini(charts.tv_symbol(pick), height=320)
    else:
        try:
            payload = api.chart_data(pick, period="6mo")
        except Exception as e:
            st.warning(f"Could not load chart for {pick}: {e}", icon="⚠️")
            return
        if not payload or not payload.get("dates"):
            empty_state(f"No chart data for {pick}", "Try a different symbol or period.", icon="📉")
            return
        cand = next((c for c in candidates if c["symbol"] == pick), {})
        scan_trade = []
        if cand.get("price"):
            scan_trade.append({
                "date":  payload["dates"][-1],
                "side":  "BUY" if str(cand.get("direction", "")).upper() == "BUY" else "SELL",
                "price": cand["price"],
            })
        charts.render_price_chart(
            payload, trades=scan_trade,
            overlays=("ema21", "ema50", "vwap", "bb_upper", "bb_lower"),
            include_volume=True, include_rsi=True, include_macd=True,
            title=f"{pick} — last 6mo",
        )


# ── Page header ───────────────────────────────────────────────────────────────
apply_theme("Market Scanner")

page_header(
    "Market Scanner",
    subtitle=(
        "Scans a universe of stocks for strategy signals, scores them 0–100, "
        "and shows the top candidates. Discovery only — the auto-scheduler executes."
    ),
)

# ── Scanner status stat band ──────────────────────────────────────────────────
try:
    sc_status = api._get("/scanner/status")
    _running   = sc_status.get("running", False)
    _last_scan = _fmt_et(sc_status.get("last_scan")) if sc_status.get("last_scan") else "Never"
    _matches   = str(sc_status.get("last_matches") or 0)
    stat_band([
        ("Status",      "Scanning…" if _running else "Ready",   "amber" if _running else "green"),
        ("Last scan",   _last_scan,                              "grey"),
        ("Last matches", _matches,                               "teal" if int(_matches) > 0 else "grey"),
    ])
except Exception:
    pass

divider()

# ── Scan configuration ────────────────────────────────────────────────────────
section("Run a Scan")

row1_cols = filter_cols(2, 2, 1, 1, 1)
universe = row1_cols[0].selectbox(
    "Universe",
    ["watchlist", "sp500", "nasdaq100", "nifty50", "custom"],
    format_func=lambda v: "nifty50 (India)" if v == "nifty50" else v,
    help="watchlist = your assigned symbols. sp500/nasdaq100 = full US index (~2–5 min). nifty50 = NSE top-50.",
)
_is_india = universe == "nifty50"
min_price  = row1_cols[1].number_input(
    "Min price (₹)" if _is_india else "Min price ($)",
    min_value=1.0, value=50.0 if _is_india else 5.0, step=1.0,
)
min_volume = row1_cols[2].number_input(
    "Min avg vol", min_value=0, value=500_000, step=100_000,
)
top_n = row1_cols[3].number_input("Top N", min_value=1, max_value=20, value=5, step=1)
scan_direction = row1_cols[4].radio(
    "Direction", ["ANY", "BUY", "SELL"], horizontal=True,
    help="ANY returns both sides. BUY or SELL filters the top-N window.",
)

custom_input = ""
if universe == "custom":
    custom_input = st.text_area(
        "Custom symbols (one per line or comma-separated)",
        placeholder="AAPL\nTSLA\nNVDA",
    )

# Discovery-only notice (auto_trade removed — see comment in original file)
auto_trade, auto_trade_direction = False, "ANY"
st.caption(
    "ℹ Scanner is **discovery only**. Candidates surface as signals; "
    "the auto-scheduler (every 15 min) is the sole execution authority."
)

run_col, _ = st.columns([2, 8])
run_btn = run_col.button("🔭 Run Scan Now", type="primary", use_container_width=True)

if run_btn:
    custom_symbols = []
    if universe == "custom" and custom_input:
        custom_symbols = [s.strip().upper() for s in custom_input.replace(",", "\n").splitlines() if s.strip()]

    config_payload = {
        "universe": universe,
        "custom_symbols": custom_symbols,
        "min_price": min_price,
        "min_avg_volume": int(min_volume),
        "top_n": int(top_n),
        "scan_direction": scan_direction,
        "auto_trade_top": auto_trade,
        "auto_trade_direction": auto_trade_direction,
        "batch_size": 20,
    }
    is_large = universe in ("sp500", "nasdaq100", "nifty50") or len(custom_symbols) > 20

    with st.spinner(f"Scanning {universe}…{'(large universe, running in background)' if is_large else ''}"):
        try:
            result = api._post("/scanner/run", json=config_payload)
            if result.get("scan_run_id") == "pending":
                st.info(
                    "Large universe scan is running in the background. "
                    "Refresh in 2–5 minutes to see results."
                )
            else:
                st.success(
                    f"Scan complete in {result['duration_seconds']}s — "
                    f"{result['total_scanned']} symbols · "
                    f"{result['total_passed_filters']} passed filters · "
                    f"{result['total_matches']} matches."
                )
                if result["top_candidates"]:
                    section(f"Top {len(result['top_candidates'])} Candidates", level=3)
                    _show_candidates(result["top_candidates"], key_prefix="run_cands")
                else:
                    empty_state(
                        "No matches found",
                        "No symbols met the signal criteria. Try a wider universe or lower filters.",
                        icon="🔍",
                    )
        except Exception as e:
            st.error(f"Scan failed: {e}")

divider()

# ── Recent scan results ────────────────────────────────────────────────────────
section(
    "Recent Scan Results",
    "Last 50 candidates from the database — auto-populated by the background scheduler.",
)

try:
    results = api._get("/scanner/results?limit=50")
    if results:
        _show_candidates(results, key_prefix="recent_cands")
    else:
        empty_state(
            "No scan results yet",
            "Run a scan above or wait for the auto-scheduler "
            "(watchlist every 15 min, S&P 500 + NASDAQ 100 every 4 h, market hours only).",
            icon="📡",
        )
except Exception as e:
    st.warning(f"Could not load scan results: {e}", icon="⚠️")

divider()

# ── How it works ──────────────────────────────────────────────────────────────
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
4. Symbols where 1+ strategies agree on BUY or SELL become candidates
5. Score and rank — top N returned, saved to database

**Auto-scan schedule (market hours only):**
- Watchlist every 15 minutes
- NASDAQ 100 every 4 hours
- S&P 500 every 4 hours (staggered 10 min after NASDAQ 100)

Large universe scans (S&P 500 ≈ 500 stocks) take 2–5 minutes and run in the background.
    """)
