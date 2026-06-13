from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, empty_state
from _components import page_header, stat_band

import pandas as pd
import streamlit as st

# ── Page setup ────────────────────────────────────────────────────────────────
apply_theme("M1 Portfolio")

try:
    from _sidebar import render_sidebar
    render_sidebar()
except Exception:
    pass

page_header(
    "M1 Portfolio Advisor",
    subtitle=(
        "Runs the trend strategy panel over your M1 holdings and recommends where "
        "to put your next contribution. Advisory only — M1 has no trading API, so "
        "you fund these manually. Not financial advice."
    ),
    badge="ADVISORY",
    badge_color="amber",
)

vcol, scol = st.columns([3, 1])
with vcol:
    view = st.radio(
        "View", ["By pie (4 pies)", "Target weights", "Flat (all holdings)"], horizontal=True,
        label_visibility="collapsed",
    )
with scol:
    if st.button("📨 Run daily scan now", use_container_width=True,
                 help="Runs the SIP dip scan and posts a notification digest now."):
        try:
            summary = api._post("/m1/scan_now", timeout=300, params={"session_label": "manual"})
            st.success(
                f"Scan done — {summary.get('funded_count', 0)} buys, "
                f"{summary.get('dip_count', 0)} dips, "
                f"{summary.get('laggard_count', 0)} laggards. Check Notifications."
            )
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
pie_view = view.startswith("By pie")
target_view = view.startswith("Target")

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns([1.3, 1.1, 1.2, 0.9, 0.9])
with c1:
    contribution = st.number_input(
        "Next contribution ($)", min_value=0.0, value=500.0, step=50.0,
        help="New money to allocate.",
    )
with c2:
    tilt_mode = st.selectbox(
        "Tilt mode", ["dip", "aggressive", "moderate", "gentle"], index=0,
        help="dip = SIP buy-the-dip (oversold pullbacks in uptrend); "
             "aggressive = fund only BUY names.",
    )
with c3:
    pie_split_mode = st.selectbox(
        "Pie split", ["conviction", "equal", "value"], index=0,
        help="How the contribution divides across the 4 pies (pie view only).",
        disabled=not pie_view,
    )
with c4:
    period = st.selectbox("History", ["6mo", "1y", "2y"], index=1)
with c5:
    st.write("")
    run = st.button("Analyze", type="primary", use_container_width=True)

for _k in ("m1_result", "m1_pie_result", "m1_target_result"):
    if _k not in st.session_state:
        st.session_state[_k] = None

if run:
    with st.spinner("Analyzing — fetching prices, momentum and running the strategy panel…"):
        try:
            if target_view:
                st.session_state["m1_target_result"] = api.m1_targets(
                    contribution=contribution, period=period,
                )
            elif pie_view:
                st.session_state["m1_pie_result"] = api.m1_analyze_pies(
                    contribution=contribution, tilt_mode=tilt_mode,
                    pie_split_mode=pie_split_mode, period=period,
                )
            else:
                st.session_state["m1_result"] = api.m1_analyze(
                    contribution=contribution, tilt_mode=tilt_mode, period=period
                )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

# ── Target-weights view ───────────────────────────────────────────────────────
if target_view:
    tres = st.session_state["m1_target_result"]
    if not tres:
        empty_state("Click **Analyze** to compute recommended target weights for every pie and stock.")
        st.stop()

    pies = tres.get("pies", [])
    stat_band([
        ("As of", str(tres.get("as_of") or "—"), "grey"),
        ("Pies", str(len(pies)), "blue"),
        ("Max/stock", f"{tres.get('max_stock_weight', 0)*100:.0f}%", "amber"),
        ("Max/pie", f"{tres.get('max_pie_weight', 0)*100:.0f}%", "amber"),
    ])

    section("Pie targets — current vs recommended")
    pdf = pd.DataFrame([{
        "Pie": p["name"],
        "Current %": round(p["current_weight"] * 100, 1),
        "Target %": round(p["target_weight"] * 100, 1),
        "Drift %": round(p["drift"] * 100, 1),
        "Value $": p["current_value"],
        "Add $": p["suggested_dollars"],
    } for p in pies])
    st.dataframe(
        pdf, use_container_width=True, hide_index=True,
        column_config={
            "Current %": st.column_config.NumberColumn(format="%.1f%%"),
            "Target %": st.column_config.NumberColumn(format="%.1f%%"),
            "Drift %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Value $": st.column_config.NumberColumn(format="$%.0f"),
            "Add $": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    st.caption(
        "Target % = recommended pie weight (signal × momentum × inverse-vol, capped). "
        "Drift = target − current. Add $ moves you toward target without selling."
    )

    section("Stock targets — within each pie")
    for p in pies:
        with st.expander(f"**{p['name']}** — target {p['target_weight']*100:.1f}% "
                         f"(drift {p['drift']*100:+.1f}%), add ${p['suggested_dollars']:,.2f}",
                         expanded=(p["suggested_dollars"] > 0)):
            sdf = pd.DataFrame([{
                "Symbol": s["symbol"],
                "Signal": s["direction"],
                "Current %": round(s["current_weight"] * 100, 1),
                "Target %": round(s["target_weight"] * 100, 1),
                "Drift %": round(s["drift"] * 100, 1),
                "RS": s.get("momentum_rs"),
                "Vol": s.get("volatility"),
                "Add $": s["suggested_dollars"],
                "Flag": "⚠" if s.get("laggard") else "",
            } for s in p["slices"]])
            st.dataframe(
                sdf, use_container_width=True, hide_index=True,
                column_config={
                    "Current %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Target %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Drift %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Add $": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
    st.caption(
        "To act: set each pie's Target % in M1's pie editor, and within each pie set "
        "the slice Target %s. Then fund with the Add $ amounts. Advisory only; not financial advice."
    )
    st.stop()

# ── Pie view ──────────────────────────────────────────────────────────────────
if pie_view:
    pres = st.session_state["m1_pie_result"]
    if not pres:
        empty_state("Click **Analyze** to score your 4 pies and get a two-level funding plan.")
        st.stop()

    pies = pres.get("pies", [])
    stat_band([
        ("As of", str(pres.get("as_of") or "—"), "grey"),
        ("Pies", str(len(pies)), "blue"),
        ("Symbols", str(pres.get("analyzed_symbols") or 0), "green"),
        ("Failed", str(pres.get("failed_symbols") or 0), "grey"),
        ("Split", str(pres.get("pie_split_mode") or ""), "amber"),
    ])

    section("Step 1 — how your contribution splits across pies")
    pie_df = pd.DataFrame([{
        "Pie": p["name"],
        "Pie value $": p["value"],
        "Conviction": round(p["conviction"], 0),
        "BUY/HOLD/SELL": f"{p['buy_slices']}/{p['hold_slices']}/{p['sell_slices']}",
        "Pie share %": round(p["pie_weight"] * 100, 1),
        "Allocate $": p["suggested_dollars"],
    } for p in pies])
    st.dataframe(
        pie_df, use_container_width=True, hide_index=True,
        column_config={
            "Pie value $": st.column_config.NumberColumn(format="$%.0f"),
            "Allocate $": st.column_config.NumberColumn(format="$%.2f"),
            "Conviction": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        },
    )
    st.caption(
        f"Splitting **${pres.get('contribution', 0):,.0f}** across pies by "
        f"**{pres.get('pie_split_mode')}**, then tilting within each pie by **{pres.get('tilt_mode')}**."
    )

    section("Step 2 — within each pie")
    for p in pies:
        funded = [s for s in p["slices"] if (s.get("suggested_dollars") or 0) > 0]
        head = (f"**{p['name']}** — ${p['suggested_dollars']:,.2f} "
                f"(conviction {round(p['conviction'])}, {len(funded)} funded)")
        with st.expander(head, expanded=(p["suggested_dollars"] > 0)):
            if not funded:
                st.info("No BUY signals in this pie this cycle — it would receive nothing under aggressive tilt.")
            sl_df = pd.DataFrame([{
                "Symbol": s["symbol"],
                "Signal": s["direction"],
                "Dip": round(s.get("dip_score") or 0, 0),
                "RSI": s.get("rsi"),
                "% off high": s.get("pct_from_high"),
                "Conviction": round(s["conviction"], 0),
                "Slice $": s["value"],
                "Allocate $": s["suggested_dollars"],
                "Flag": "⚠ review" if s.get("laggard") else "",
            } for s in p["slices"]])
            st.dataframe(
                sl_df, use_container_width=True, hide_index=True,
                column_config={
                    "Slice $": st.column_config.NumberColumn(format="$%.0f"),
                    "Allocate $": st.column_config.NumberColumn(format="$%.2f"),
                    "% off high": st.column_config.NumberColumn(format="%.1f%%"),
                    "Dip": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                    "Conviction": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                },
            )
    st.caption(
        "Panel: trend_follow · momentum_breakout · trend_pullback · squeeze_breakout. "
        "Pies share tickers; each is funded independently. Advisory only; not financial advice."
    )
    st.stop()

# ── Flat view ─────────────────────────────────────────────────────────────────
result = st.session_state["m1_result"]

if not result:
    empty_state("Click **Analyze** to score your M1 holdings and get a funding plan.")
    st.stop()

holdings = result.get("holdings", [])
if not holdings:
    empty_state("No holdings found. Check m1_holdings.json.")
    st.stop()

# ── Summary band ──────────────────────────────────────────────────────────────
buys = [h for h in holdings if h["direction"] == "BUY"]
sells = [h for h in holdings if h["direction"] == "SELL"]
funded = [h for h in holdings if (h.get("suggested_dollars") or 0) > 0]
total_val = sum(h.get("value") or 0 for h in holdings)

stat_band([
    ("As of", str(result.get("as_of") or "—"), "grey"),
    ("Holdings", str(len(holdings)), "blue"),
    ("BUY", str(len(buys)), "green"),
    ("SELL", str(len(sells)), "red"),
    ("Funded", str(len(funded)), "amber"),
    ("Failed", str(result.get("failed") or 0), "grey"),
])

# ── Funding plan ──────────────────────────────────────────────────────────────
section("Funding plan — where your next contribution goes")
if not funded:
    st.info("No BUY signals this cycle — the tilt would fund nothing. Consider holding the cash or switching tilt mode.")
else:
    fund_df = pd.DataFrame([{
        "Symbol": h["symbol"],
        "Name": h["name"],
        "Signal": h["direction"],
        "Conviction": round(h["conviction"], 0),
        "Allocate $": h["suggested_dollars"],
        "Weight %": round(h["tilt_weight"] * 100, 1),
        "Price": h.get("price"),
    } for h in funded])
    st.dataframe(
        fund_df, use_container_width=True, hide_index=True,
        column_config={
            "Allocate $": st.column_config.NumberColumn(format="$%.2f"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Conviction": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        },
    )
    st.caption(
        f"Allocating **${contribution:,.0f}** across **{len(funded)}** BUY-signal holdings "
        f"using **{tilt_mode}** tilt. Place these as one-time buys (or set pie weights) in M1."
    )

# ── All holdings signal table ─────────────────────────────────────────────────
section("All holdings — signals")
_dir_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
all_df = pd.DataFrame([{
    "Symbol": h["symbol"],
    "Name": h["name"],
    "Signal": h["direction"],
    "Dip": round(h.get("dip_score") or 0, 0),
    "RSI": h.get("rsi"),
    "% off high": h.get("pct_from_high"),
    "Conviction": round(h["conviction"], 0),
    "Value $": h.get("value"),
    "Price": h.get("price"),
    "Note": h.get("note") or h.get("error") or "",
} for h in sorted(holdings, key=lambda x: (_dir_order.get(x["direction"], 9), -x["conviction"]))])

st.dataframe(
    all_df, use_container_width=True, hide_index=True,
    column_config={
        "Value $": st.column_config.NumberColumn(format="$%.2f"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "% off high": st.column_config.NumberColumn(format="%.1f%%"),
        "Dip": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Conviction": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
    },
)

st.caption(
    "Panel: trend_follow · momentum_breakout · trend_pullback · squeeze_breakout. "
    "Consensus = majority vote. Conviction = net agreement of the winning side. "
    "Advisory only; not financial advice."
)
