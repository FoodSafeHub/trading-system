"""Analyst Ratings — Wall-Street consensus for holdings + assigned symbols.

Reads the analyst_ratings cache (/ratings). Data comes from yfinance (free):
consensus recommendation, buy/hold/sell counts, price targets and recent
upgrades/downgrades. NSE names usually have thin/no coverage — those rows
show a note instead of numbers. The scheduler refreshes the cache twice a
day; the Refresh button forces it now.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, kpi_row, market_status_bar, empty_state
from _components import page_header

import pandas as pd
import streamlit as st

apply_theme("Analyst Ratings")

from _sidebar import render_sidebar
render_sidebar()
page_header(
    "Analyst Ratings",
    subtitle=(
        "Wall-Street consensus for your holdings and assigned symbols · "
        "source: Yahoo Finance · cache refreshed twice daily"
    ),
)
market_status_bar()

# ── Load + refresh ───────────────────────────────────────────────────────────
_c1, _c2 = st.columns([1, 5])
with _c1:
    _do_refresh = st.button("🔄 Refresh now", help=(
        "Re-pull ratings from Yahoo Finance for every held + assigned symbol. "
        "Takes a few seconds per symbol."
    ))
if _do_refresh:
    with st.spinner("Refreshing analyst ratings from Yahoo Finance…"):
        try:
            out = api.ratings_recompute()
            st.success(
                f"Refreshed {out.get('refreshed', 0)} symbol(s)"
                + (f" · {len(out['errors'])} error(s)" if out.get("errors") else "")
            )
        except Exception as exc:
            st.error(f"Refresh failed: {exc}")

try:
    rows = api.ratings_list()
except Exception as exc:
    st.error(f"Cannot load /ratings: {exc}")
    st.stop()

if not rows:
    empty_state(
        "No ratings cached yet",
        "Click **Refresh now** to pull analyst data for your holdings and "
        "assigned symbols. The scheduler also refreshes this twice a day.",
    )
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
_covered = [r for r in rows if r.get("recommendation_key") or r.get("analyst_count")]
_upsides = [r["upside_pct"] for r in _covered if r.get("upside_pct") is not None]
_avg_upside = round(sum(_upsides) / len(_upsides), 1) if _upsides else None


def _bullish(r) -> bool | None:
    key = (r.get("recommendation_key") or "").lower()
    if not key or key == "none":
        return None
    return key in ("strong_buy", "buy")


_n_buy = sum(1 for r in _covered if _bullish(r) is True)
_n_not_buy = sum(1 for r in _covered if _bullish(r) is False)
kpi_row([
    ("Symbols tracked", str(len(rows))),
    ("With coverage", str(len(_covered))),
    ("Consensus buy/strong-buy", f"{_n_buy} of {_n_buy + _n_not_buy}" if _covered else "—"),
    ("Avg upside to mean target", f"{_avg_upside:+.1f}%" if _avg_upside is not None else "—"),
])

# ── Main table ───────────────────────────────────────────────────────────────
section(
    "Consensus & Price Targets",
    "One row per symbol — holdings first. **Consensus** is Yahoo's aggregated "
    "recommendation; **counts** are this month's analyst distribution; "
    "**Upside** compares the mean 12-month target to the current price.",
)

_KEY_LABELS = {
    "strong_buy": "🟢 Strong Buy", "buy": "🟢 Buy", "hold": "🟡 Hold",
    "underperform": "🔴 Underperform", "sell": "🔴 Sell", "strong_sell": "🔴 Strong Sell",
    "none": "—",
}


def _row_fmt(r: dict) -> dict:
    def _n(v):
        return int(v) if v is not None else 0
    counts = "—"
    if any(r.get(k) is not None for k in ("strong_buy", "buy", "hold", "sell", "strong_sell")):
        counts = (
            f"{_n(r.get('strong_buy'))}/{_n(r.get('buy'))}/{_n(r.get('hold'))}"
            f"/{_n(r.get('sell'))}/{_n(r.get('strong_sell'))}"
        )
    src = []
    if r.get("is_holding"):
        src.append("Holding")
    if r.get("is_assigned"):
        src.append("Assigned")
    key = (r.get("recommendation_key") or "").lower()
    return {
        "Symbol": r["symbol"],
        "Why tracked": " + ".join(src) or "—",
        "Consensus": _KEY_LABELS.get(key, key.title() if key else "—"),
        "SB/B/H/S/SS": counts,
        "Analysts": r.get("analyst_count"),
        "Price": r.get("current_price"),
        "Target (mean)": r.get("target_mean"),
        "Target (low–high)": (
            f"{r['target_low']:,.0f}–{r['target_high']:,.0f}"
            if r.get("target_low") is not None and r.get("target_high") is not None
            else "—"
        ),
        "Upside %": r.get("upside_pct"),
        "Note": r.get("note") or "",
        "As of": str(r.get("computed_at") or "")[:16].replace("T", " "),
    }


df = pd.DataFrame([_row_fmt(r) for r in rows])
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="%.2f"),
        "Target (mean)": st.column_config.NumberColumn(format="%.2f"),
        "Upside %": st.column_config.NumberColumn(format="%+.1f%%"),
    },
)
st.caption(
    "SB/B/H/S/SS = Strong Buy / Buy / Hold / Sell / Strong Sell analyst counts "
    "(current month). NSE (India) names typically have little or no Yahoo analyst "
    "coverage — that's a data-source limit, not a signal."
)

# ── Recent upgrades / downgrades ─────────────────────────────────────────────
st.divider()
section(
    "Recent Upgrades & Downgrades",
    "Latest firm actions per symbol (most recent first). An 'up' on a name you "
    "hold in the red supports patience; a wave of downgrades is worth a manual "
    "review even if the strategy hasn't fired a SELL.",
)

_with_actions = [r for r in rows if r.get("upgrades")]
if not _with_actions:
    st.info("No upgrade/downgrade history cached — hit Refresh, or the covered "
            "names simply have no recent actions.")
else:
    _sel = st.selectbox(
        "Symbol",
        [r["symbol"] for r in _with_actions],
        help="Symbols with cached upgrade/downgrade history.",
    )
    _r = next(r for r in _with_actions if r["symbol"] == _sel)
    _df_u = pd.DataFrame(_r["upgrades"])
    if not _df_u.empty:
        _df_u = _df_u.rename(columns={
            "date": "Date", "firm": "Firm", "action": "Action",
            "from_grade": "From", "to_grade": "To",
        })
        st.dataframe(_df_u, use_container_width=True, hide_index=True)
