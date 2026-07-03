from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, divider, pill, empty_state
from _components import (
    page_header, stat_band, filter_cols,
    eligibility_chip, regime_badge, candidate_state,
)
import _charts as charts

import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── Eligibility sort priority (lower = higher in table) ──────────────────────
_ELIG_SORT = {"active": 0, "ready": 1, "watch": 2, "idle": 3, "blocked": 4}


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


def _show_candidates(candidates: list, *, key_prefix: str = "cands") -> None:
    """Render scanner candidates table with eligibility chips and priority ordering."""
    # ── Direction filter ──────────────────────────────────────────────────────
    dir_col, _ = st.columns([3, 7])
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

    # ── Load cached recommendations ───────────────────────────────────────────
    try:
        recs = {r["symbol"]: r for r in (api.recommendations_list() or [])}
    except Exception:
        recs = {}

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for c in candidates:
        sym        = c.get("symbol", "")
        direction  = (c.get("direction") or "").upper()
        score      = int(c.get("score") or 0)
        auto_traded = bool(c.get("auto_traded"))
        rec        = recs.get(sym)
        firing_strats = _parse_strategies_from_reason(c.get("reason"))
        is_match   = bool(rec) and any(s == rec["strategy_name"] for s in firing_strats)

        # Eligibility — the primary triage column
        elig_state, elig_reason = candidate_state(score, direction, auto_traded, is_match)

        # Direction display
        dir_text = {"BUY": "▲ BUY", "SELL": "▼ SELL"}.get(direction, direction or "—")

        # Best historical strategy compact
        if rec:
            wr = rec.get("win_rate_pct")
            pf = rec.get("profit_factor")
            strat_name = rec["strategy_name"].replace("_", " ")
            bits = []
            if wr is not None: bits.append(f"{wr:.0f}%WR")
            if pf is not None: bits.append(f"{pf:.2f}PF")
            best_short = strat_name[:22] + ("…" if len(strat_name) > 22 else "")
            best_str   = best_short + (f" ({', '.join(bits)})" if bits else "")
            match_flag = "⭐ " if is_match else ""
        else:
            best_str   = "—"
            match_flag = ""

        # Reason: strip verbose prefix, keep signal core
        raw_reason = c.get("reason") or "—"
        # Tape-gate verdict (appended by the scanner) → its own column so the
        # 60-char Signal truncation can't hide it.
        tape_col = ""
        if " · Tape gate: " in raw_reason:
            raw_reason, _, _tape = raw_reason.partition(" · Tape gate: ")
            if _tape.startswith("PASSED"):
                tape_col = "🟢 clear"
            elif _tape.startswith("BLOCKED"):
                tape_col = "🔴 " + _tape.replace("BLOCKED — ", "")
            else:
                tape_col = _tape
        if raw_reason.startswith("N strategies agree:"):
            # "N strategies agree: A, B, C" → just the strategy names
            raw_reason = raw_reason.split(":", 1)[-1].strip()
        reason_short = raw_reason[:60] + ("…" if len(raw_reason) > 60 else "")

        rows.append({
            "_elig_sort": _ELIG_SORT.get(elig_state, 9),
            "_elig_html": eligibility_chip(elig_state, elig_reason),
            "Symbol":     f"{match_flag}{sym}",
            "State":      elig_state.upper(),       # plain text for column_config sorting
            "Dir":        dir_text,
            "Score":      score,
            "Agree":      c.get("strategies_agreeing") or 0,
            "Price":      c.get("price") or None,
            "Avg Vol M":  (c.get("avg_volume") or 0) / 1_000_000,   # shown as xM
            "Best strategy": best_str,
            "Tape":       tape_col,
            "Signal":     reason_short,
            "Scanned":    _fmt_et(c.get("scanned_at")),
        })

    # Sort: ACTIVE first, then READY, WATCH, IDLE, then by score desc within tier
    rows.sort(key=lambda r: (r["_elig_sort"], -r["Score"]))

    # ── Summary header bar ────────────────────────────────────────────────────
    n_ready   = sum(1 for r in rows if r["State"] in ("READY", "ACTIVE"))
    n_watch   = sum(1 for r in rows if r["State"] == "WATCH")
    n_idle    = sum(1 for r in rows if r["State"] == "IDLE")
    st.markdown(
        f"<div style='display:flex;gap:var(--sp-4);margin-bottom:var(--sp-3);align-items:center'>"
        f"{pill(f'{n_ready} READY', 'green') if n_ready else ''}"
        f"{pill(f'{n_watch} WATCH', 'amber') if n_watch else ''}"
        f"{pill(f'{n_idle} IDLE',  'grey')  if n_idle  else ''}"
        f"<span style='color:var(--text-3);font-size:0.78rem;margin-left:auto'>"
        f"{len(rows)} total · sorted by eligibility then score</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    # Drop internal sort/html columns before display
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol":        st.column_config.TextColumn("Symbol",      width="small"),
            "State":         st.column_config.TextColumn("State",       width="small"),
            "Dir":           st.column_config.TextColumn("Dir",         width="small"),
            "Score":         st.column_config.NumberColumn("Score /100", format="%d",    width="small"),
            "Agree":         st.column_config.NumberColumn("Agree",      format="%d",    width="small"),
            "Price":         st.column_config.NumberColumn("Price",      format="$%.2f", width="small"),
            "Avg Vol M":     st.column_config.NumberColumn("Vol (M)",    format="%.1f",  width="small"),
            "Best strategy": st.column_config.TextColumn("Best (hist.)", width="medium"),
            "Signal":        st.column_config.TextColumn("Signal",       width="large"),
            "Scanned":       st.column_config.TextColumn("Scanned",      width="small"),
        },
    )

    # ── Recommendations recompute ─────────────────────────────────────────────
    cand_symbols = sorted({c["symbol"] for c in candidates if c.get("symbol")})
    missing      = [s for s in cand_symbols if s not in recs]
    rc1, rc2, rc3 = st.columns([4, 2, 2])
    rc1.caption(
        f"{len(recs)} historical recs cached · "
        f"{len(missing)} of {len(cand_symbols)} on-screen symbols missing"
    )
    period_pick = rc2.selectbox(
        "Backtest period", ["2y", "5y", "1y"], index=1, key=f"{key_prefix}_rec_period",
    )
    if missing and rc3.button(
        f"⚙ Compute {len(missing)} missing", key=f"{key_prefix}_rec_missing",
        use_container_width=True,
    ):
        with st.spinner(f"Running Compare All on {len(missing)} symbol(s)…"):
            try:
                api.recommendations_recompute_many(missing, period=period_pick)
                st.success(f"Recomputed {len(missing)} recommendation(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Recompute failed: {exc}")

    # ── Inline candle drill-in ────────────────────────────────────────────────
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


# ── Page setup ────────────────────────────────────────────────────────────────
apply_theme("Market Scanner")

from _sidebar import render_sidebar
render_sidebar()

page_header(
    "Market Scanner",
    subtitle=(
        "Scans a universe of stocks for strategy signals, scores 0–100, "
        "and ranks candidates by eligibility. Discovery only — the scheduler executes."
    ),
)

# ── Status stat band ──────────────────────────────────────────────────────────
try:
    sc_status  = api._get("/scanner/status")
    _running   = sc_status.get("running", False)
    _last_scan = _fmt_et(sc_status.get("last_scan")) if sc_status.get("last_scan") else "Never"
    _matches   = str(sc_status.get("last_matches") or 0)
    stat_band([
        ("Status",       "Scanning…" if _running else "Ready",                   "amber" if _running else "green"),
        ("Last scan",    _last_scan,                                               "grey"),
        ("Last matches", _matches,                                                 "teal" if int(_matches) > 0 else "grey"),
        ("Execution",    "Scheduler only",                                        "grey"),
    ])
except Exception:
    pass

divider()

# ── Scan configuration ────────────────────────────────────────────────────────
section("Run a Scan")

# Scan lens. Consensus = legacy symbol-first (rank by 0–100 consensus score).
# By signal = strategy-first: list every symbol where one or more chosen
# strategies fired the selected direction (ANY match).
scan_mode_label = st.radio(
    "Scan mode",
    ["Consensus", "By signal"],
    horizontal=True,
    help=(
        "Consensus ranks symbols by how many strategies agree. "
        "By signal flips it: pick the strategy(s) you trust + a direction, and "
        "the scan lists every symbol where any of them fired that side."
    ),
)
_signal_mode = scan_mode_label == "By signal"

# Strategy multiselect — only shown in signal mode. Built from the engine via
# GET /scanner/strategies so it can't drift from the live strategy set.
selected_strategy_ids: list[str] = []
if _signal_mode:
    try:
        _strat_opts = api.scanner_strategies()
    except Exception as exc:
        _strat_opts = {"generic": [], "perplexity": []}
        st.warning(f"Could not load strategy list: {exc}")
    # id -> label map for the picker; group perplexity under a clear prefix.
    _id_to_label: dict[str, str] = {}
    for g in _strat_opts.get("generic", []):
        _id_to_label[g["id"]] = g["label"]
    for p in _strat_opts.get("perplexity", []):
        _id_to_label[p["id"]] = f"Swing · {p['label']}"
    selected_strategy_ids = st.multiselect(
        "Strategies (signal)",
        options=list(_id_to_label.keys()),
        format_func=lambda i: _id_to_label.get(i, i),
        help="A symbol matches if ANY selected strategy fired the chosen direction.",
    )

row1_cols = filter_cols(2, 1, 1, 1, 1)
universe = row1_cols[0].selectbox(
    "Universe",
    ["watchlist", "sp500", "nasdaq100", "sp400", "sp600", "sp1500",
     "nifty50", "nifty100", "nifty200", "nifty500", "nse_all", "custom"],
    format_func=lambda v: {
        "nifty50":  "nifty50 (India)",
        "nifty100": "nifty100 (India)",
        "nifty200": "nifty200 (India)",
        "nifty500": "nifty500 (India — full NSE ~2466)",
        "nse_all":  "nse_all (India — full NSE)",
        "sp1500":   "sp1500 (~1500)",
        "sp400":    "sp400 (MidCap)",
        "sp600":    "sp600 (SmallCap)",
    }.get(v, v),
    help=(
        "watchlist = assigned symbols. sp500/nasdaq100 = full US index (~2–5 min). "
        "sp1500 = S&P Composite 1500 (~1500 stocks, ~8–15 min). "
        "nifty50/100/200/500 = NSE tiers; nse_all = full NSE (slow)."
    ),
)
_is_india  = universe in ("nifty50", "nifty100", "nifty200", "nifty500", "nse_all")
_cur       = "₹" if _is_india else "$"
min_price  = row1_cols[1].number_input(
    f"Min price ({_cur})",
    min_value=1.0, value=50.0 if _is_india else 5.0, step=1.0,
)
max_price  = row1_cols[2].number_input(
    f"Max price ({_cur})",
    min_value=0.0, value=0.0, step=1.0,
    help="0 = no ceiling. Must be above Min price to take effect.",
)
min_volume = row1_cols[3].number_input("Min avg vol", min_value=0, value=500_000, step=100_000)
top_n      = row1_cols[4].number_input("Top N", min_value=1, max_value=20, value=5, step=1)

row2_cols = filter_cols(1, 1, 2)
min_float_m = row2_cols[0].number_input(
    "Min float (M)", min_value=0.0, value=0.0, step=1.0,
    help="Shares float in millions. 0 = no minimum. US symbols only (no India float feed).",
    disabled=_is_india,
)
max_float_m = row2_cols[1].number_input(
    "Max float (M)", min_value=0.0, value=0.0, step=1.0,
    help="Shares float in millions. 0 = no maximum. Use a low cap (e.g. 50) for low-float momentum names.",
    disabled=_is_india,
)
# Signal mode needs a concrete side (BUY/SELL) — "scan for X firing" is
# inherently directional. Consensus mode keeps the ANY option.
_dir_options = ["BUY", "SELL"] if _signal_mode else ["ANY", "BUY", "SELL"]
scan_direction = row2_cols[2].radio(
    "Direction", _dir_options, horizontal=True,
    help=(
        "Pick the side the selected strategies must fire."
        if _signal_mode else
        "ANY returns both sides. BUY or SELL fills Top N from that side only."
    ),
)
if (min_float_m or max_float_m) and not _is_india:
    st.caption(
        "ℹ Float filter fetches shares-outstanding per symbol (cached daily). "
        "It only runs on symbols that clear the price/volume gates, but a first "
        "1500-symbol run can still take a few extra minutes."
    )

custom_input = ""
if universe == "custom":
    custom_input = st.text_area(
        "Custom symbols (one per line or comma-separated)",
        placeholder="AAPL\nTSLA\nNVDA",
    )

auto_trade, auto_trade_direction = False, "ANY"
st.caption(
    "ℹ Scanner is **discovery only** — candidates become signals. "
    "The auto-scheduler (every 15 min) is the sole execution authority."
)

run_col, _ = st.columns([2, 8])
run_btn = run_col.button("🔭 Run Scan Now", type="primary", use_container_width=True)

if run_btn and _signal_mode and not selected_strategy_ids:
    # Guard: signal mode is meaningless without at least one strategy picked.
    st.error("Pick at least one strategy to run a signal scan.")
    run_btn = False

if run_btn:
    custom_symbols = []
    if universe == "custom" and custom_input:
        custom_symbols = [s.strip().upper() for s in custom_input.replace(",", "\n").splitlines() if s.strip()]

    config_payload = {
        "universe": universe,
        "custom_symbols": custom_symbols,
        "min_price": min_price,
        "max_price": max_price,
        "min_avg_volume": int(min_volume),
        "min_float": float(min_float_m) * 1_000_000 if not _is_india else 0.0,
        "max_float": float(max_float_m) * 1_000_000 if not _is_india else 0.0,
        "top_n": int(top_n),
        "scan_direction": scan_direction,
        "scan_mode": "signal" if _signal_mode else "consensus",
        "signal_strategies": selected_strategy_ids if _signal_mode else [],
        "auto_trade_top": auto_trade,
        "auto_trade_direction": auto_trade_direction,
        "batch_size": 20,
    }
    is_large = (
        universe in ("sp500", "nasdaq100", "sp400", "sp600", "sp1500",
                     "nifty50", "nifty100", "nifty200", "nifty500", "nse_all")
        or len(custom_symbols) > 20
    )

    with st.spinner(f"Scanning {universe}…" + (" (large — background)" if is_large else "")):
        try:
            result = api._post("/scanner/run", json=config_payload)
            if result.get("scan_run_id") == "pending":
                _eta = "8–15 minutes" if universe == "sp1500" else "2–5 minutes"
                st.info(
                    "Large universe scan running in the background. "
                    f"Refresh in {_eta} to see results."
                )
            else:
                dur  = result.get("duration_seconds", "?")
                tot  = result.get("total_scanned", 0)
                passed = result.get("total_passed_filters", 0)
                matches = result.get("total_matches", 0)
                st.success(
                    f"Scan complete in {dur}s — "
                    f"{tot:,} scanned · {passed:,} passed filters · {matches} matches."
                )
                if result.get("top_candidates"):
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
            "Run a scan above, or wait for the auto-scheduler "
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

| Factor | Points | Eligibility threshold |
|---|---|---|
| Each strategy agreeing (up to 3) | 20 pts each | ≥60 → READY, 35–59 → WATCH |
| Perplexity strategy agreement | 15 pts | |
| Avg volume > 2M | 10 pts | |
| Price above 50-day SMA (uptrend) | 10 pts | |

**Row states:**
- **ACTIVE** — symbol is currently auto-traded by the bot
- **READY** — score ≥ 60; conditions met for entry (especially if it matches the historically best strategy ⭐)
- **WATCH** — score 35–59; marginal signal, monitor closely
- **IDLE** — score < 35; surfaced by scanner but signal is weak

**Scan flow:**
1. Build universe (watchlist / S&P 500 / 400 / 600 / 1500 / NASDAQ 100 / nifty50 / custom)
2. Apply liquidity filters: min **and max** price, min avg volume, min history
3. Optional shares-**float band** (US only; min/max in millions, cached daily)
4. Run all 5 Bollinger + 5 Perplexity strategies per symbol
5. Score and rank — top N returned, saved to database

**Universes:** `sp1500` is the S&P Composite 1500 (S&P 500 + MidCap 400 +
SmallCap 600 ≈ 1500 stocks) — the widest single-index sweep, ~8–15 min.

**Auto-scan (market hours only):** watchlist every 15 min · NASDAQ 100 every 4 h · S&P 500 every 4 h
    """)
