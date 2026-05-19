from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme

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


def _show_candidates(candidates):
    rows = []
    for c in candidates:
        direction = c.get("direction", "")
        dir_label = direction
        rows.append({
            "Symbol":       c["symbol"],
            "Direction":    dir_label,
            "Score":        f"{c['score']:.0f} / 100",
            "Strategies":   c["strategies_agreeing"],
            "Price":        f"${c['price']:,.2f}" if c.get("price") else "—",
            "Avg Volume":   f"{int(c['avg_volume'] or 0):,}" if c.get("avg_volume") else "—",
            "Universe":     c.get("universe", "—"),
            "Reason":       c.get("reason", "—"),
            "Auto-Traded":  "yes" if c.get("auto_traded") else "—",
            "Scanned At":   _fmt_et(c.get("scanned_at")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
        ["watchlist", "sp500", "nasdaq100", "custom"],
        help="watchlist = your current strategies.json stocks. sp500/nasdaq100 = full index scan (takes ~2–5 min).",
    )
    custom_input = ""
    if universe == "custom":
        custom_input = st.text_area(
            "Custom symbols (one per line or comma-separated)",
            placeholder="AAPL\nTSLA\nNVDA",
        )

with col2:
    min_price = st.number_input("Min price ($)", min_value=1.0, value=5.0, step=1.0)
    min_volume = st.number_input("Min avg daily volume", min_value=0, value=500000, step=100000)
    top_n = st.slider("Top N candidates to return", min_value=1, max_value=20, value=5)
    auto_trade = st.toggle(
        "Auto-trade top candidate",
        value=False,
        help="If enabled, the #1 ranked candidate will be paper-traded automatically if risk checks pass.",
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
        "auto_trade_top": auto_trade,
        "batch_size": 20,
    }

    is_large = universe in ("sp500", "nasdaq100") or len(custom_symbols) > 20

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
                    _show_candidates(result["top_candidates"])
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
        _show_candidates(results)
    else:
        st.info("No scan results yet. Run a scan above or wait for the auto-scheduler (runs every 15 min during market hours).")
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

**Auto-scan:** Runs automatically every 15 minutes during market hours (9:30am–4pm ET) on your watchlist.

**Large universe scans** (S&P 500 = ~500 stocks) take 2–5 minutes and run in the background.
    """)
