"""Realized + unrealized P/L dashboard.

Pulls everything from /pnl/* — no local state. Top-line tiles, equity curve,
open positions with live unrealized, then breakdowns by symbol and strategy
followed by the full closed-trade log.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + "/dashboard")
import api
from _theme import apply_theme, section, kpi_row, money, pct, divider, currency_symbol, market_status_bar, empty_state
from _components import page_header, stat_band

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

apply_theme("P/L")

from _sidebar import render_sidebar
render_sidebar()
page_header(
    "P&L Dashboard",
    subtitle=(
        "Realized P&L computed FIFO from fills · "
        "Unrealized = open size × (last quote − avg cost) · refreshed each load"
    ),
)
market_status_bar()


# ── Load ────────────────────────────────────────────────────────────────────
try:
    summary = api.pnl_summary(include_unrealized=True)
except Exception as exc:
    st.error(f"Cannot load /pnl/summary: {exc}")
    st.stop()

realized = summary.get("realized") or {}
total_unrealized = summary.get("total_unrealized_pnl") or 0.0
closed_count = summary.get("closed_trade_count") or 0
open_count = summary.get("open_position_count") or 0
has_live_prices = summary.get("has_live_prices", False)

total_realized = realized.get("total_realized_pnl") or 0.0
win_rate = realized.get("win_rate_pct") or 0.0
profit_factor = realized.get("profit_factor")
best_trade = realized.get("best_trade") or 0.0
worst_trade = realized.get("worst_trade") or 0.0
avg_hold = realized.get("avg_hold_days") or 0.0

# ── Top tiles ───────────────────────────────────────────────────────────────
kpi_row([
    ("Realized P/L", money(total_realized)),
    ("Unrealized P/L", money(total_unrealized) if has_live_prices else "—"),
    ("Total P/L", money(total_realized + total_unrealized) if has_live_prices else money(total_realized)),
    ("Closed trades", f"{closed_count:,}"),
])

kpi_row([
    ("Win rate", f"{win_rate:.1f}%"),
    ("Profit factor", f"{profit_factor:.2f}" if profit_factor is not None else "—"),
    ("Best trade", money(best_trade)),
    ("Worst trade", money(worst_trade)),
])

if not has_live_prices and open_count > 0:
    st.info(f"{open_count} open position(s) — live quotes unavailable, unrealized P/L shown as —.")

divider()


# ── Equity curve ────────────────────────────────────────────────────────────
section("Equity Curve", "Cumulative realized P/L + running drawdown from peak.")

bcol1, bcol2 = st.columns([1, 5])
with bcol1:
    bucket_label = st.selectbox("Bucket", ["Per trade", "Per day"], index=0, key="pnl_eq_bucket")
bucket = "day" if bucket_label == "Per day" else "trade"

try:
    eq = api.pnl_equity_curve(bucket=bucket)
except Exception as exc:
    st.error(f"Cannot load /pnl/equity-curve: {exc}")
    eq = []

if not eq:
    st.info("No realized trades yet. The curve fills in as round-trips close.")
else:
    df_eq = pd.DataFrame(eq)
    # Timestamps may arrive as tz-aware ISO-8601 with offsets (e.g.
    # "2026-06-15T09:30:03-04:00") or naive; format="ISO8601" parses both,
    # then drop tz so all points share one naive axis (avoids mixed-tz errors).
    df_eq["at"] = pd.to_datetime(df_eq["at"], format="ISO8601", utc=True).dt.tz_localize(None)

    max_dd = float(df_eq["drawdown"].max()) if "drawdown" in df_eq else 0.0
    max_dd_pct = df_eq["drawdown_pct"].max() if "drawdown_pct" in df_eq else None

    # With <2 points Plotly can't draw a line — show markers so a single trade
    # is visible. Also pad the x range so a lone point doesn't get zoomed to
    # microseconds on the axis.
    single_point = len(df_eq) < 2
    mode = "lines+markers" if single_point else "lines"
    ts = df_eq["at"]
    if single_point:
        t0 = ts.iloc[0]
        x_range = [t0 - pd.Timedelta(hours=12), t0 + pd.Timedelta(hours=12)]
    else:
        span = ts.iloc[-1] - ts.iloc[0]
        pad = max(span * 0.04, pd.Timedelta(minutes=30))
        x_range = [ts.iloc[0] - pad, ts.iloc[-1] + pad]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(go.Scatter(
        x=ts, y=df_eq["realized_pnl"],
        mode=mode,
        line=dict(color="#26a69a", width=2),
        marker=dict(color="#26a69a", size=8),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.15)",
        hovertemplate="%{x|%b %d %Y %H:%M}<br>P/L $%{y:,.2f}<extra></extra>",
        name="Realized P/L",
    ), row=1, col=1)
    if not single_point:
        fig.add_trace(go.Scatter(
            x=ts, y=df_eq["peak_pnl"],
            mode="lines", line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
            hovertemplate="%{x|%b %d %Y %H:%M}<br>Peak $%{y:,.2f}<extra></extra>",
            name="Peak",
        ), row=1, col=1)
    # Drawdown ribbon underneath. Hidden when there's nothing to show.
    fig.add_trace(go.Scatter(
        x=ts, y=-df_eq["drawdown"],
        mode=mode,
        line=dict(color="#ef5350", width=1.5),
        marker=dict(color="#ef5350", size=6),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.22)",
        hovertemplate="%{x|%b %d %Y %H:%M}<br>DD $%{customdata:,.2f}<extra></extra>",
        customdata=df_eq["drawdown"],
        name="Drawdown",
    ), row=2, col=1)

    # Y range padding so a single point isn't pasted on the gridline.
    y_min = float(df_eq["realized_pnl"].min())
    y_max = float(df_eq["peak_pnl"].max()) if "peak_pnl" in df_eq else float(df_eq["realized_pnl"].max())
    y_span = max(y_max - y_min, abs(y_max), 1.0)
    y_pad = y_span * 0.25

    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(230,230,230,0.9)"),
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        range=x_range,
        tickformat="%b %d %H:%M",
        showspikes=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)", tickprefix="$",
        range=[min(0, y_min) - y_pad, y_max + y_pad],
        row=1, col=1,
    )
    # Drawdown axis: always show 0 at the top, dynamic floor.
    dd_floor = -max(float(df_eq["drawdown"].max()) * 1.25, 1.0)
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)", tickprefix="$",
        range=[dd_floor, 0],
        row=2, col=1,
    )
    st.plotly_chart(fig, use_container_width=True)

    dd_pct_str = f" ({max_dd_pct * 100:.1f}%)" if max_dd_pct else ""
    st.caption(f"Max drawdown from peak: {money(max_dd)}{dd_pct_str}")


# ── Open positions ──────────────────────────────────────────────────────────
section("Open Positions", "Currently long lots aggregated per symbol, unrealized P/L vs last broker quote.")

try:
    opens = api.pnl_open_positions()
except Exception as exc:
    st.error(f"Cannot load /pnl/open-positions: {exc}")
    opens = []

if not opens:
    empty_state("No open positions", "Open lots appear here once you have live or paper trades.", icon="📊")
else:
    df_open = pd.DataFrame(opens)
    # Per-row currency: format money columns to strings prefixed with the row's
    # broker glyph (₹ for zerodha, $ otherwise). NumberColumn can't vary the
    # symbol per row, so we pre-format and render as text.
    def _cur(row) -> str:
        return currency_symbol(row.get("broker"))

    def _fmt_money(row, col, signed=False) -> str:
        cur = _cur(row)
        return money(row.get(col), currency=cur, decimals=2) if not signed else (
            "—" if row.get(col) is None else f"{cur}{float(row[col]):+,.2f}"
        )

    df_open["_avg"] = df_open.apply(lambda r: _fmt_money(r, "avg_cost"), axis=1)
    df_open["_last"] = df_open.apply(lambda r: _fmt_money(r, "last_price"), axis=1)
    df_open["_mv"] = df_open.apply(lambda r: _fmt_money(r, "market_value"), axis=1)
    df_open["_upnl"] = df_open.apply(lambda r: _fmt_money(r, "unrealized_pnl", signed=True), axis=1)

    df_show = df_open[[
        "symbol", "quantity", "_avg", "_last", "_mv",
        "_upnl", "unrealized_pct", "broker", "is_paper",
    ]].copy()
    df_show.columns = ["Symbol", "Qty", "Avg cost", "Last", "Mkt value",
                       "Unrealized", "Unrealized %", "Broker", "Paper"]
    st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty": st.column_config.NumberColumn(format="%.4f"),
            "Unrealized %": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )


# ── By symbol / strategy ────────────────────────────────────────────────────
col_sym, col_strat = st.columns(2)

with col_sym:
    section("By Symbol", "Realized P/L per ticker, best first.", level=3)
    try:
        sym_rows = api.pnl_by_symbol()
    except Exception as exc:
        st.error(f"Cannot load /pnl/by-symbol: {exc}")
        sym_rows = []
    if not sym_rows:
        st.info("No closed trades yet.")
    else:
        df_sym = pd.DataFrame(sym_rows)[[
            "key", "trade_count", "win_rate_pct",
            "total_realized_pnl", "avg_pnl", "profit_factor", "avg_hold_days",
        ]]
        df_sym.columns = ["Symbol", "Trades", "Win %", "Total $", "Avg $", "PF", "Avg hold (d)"]
        st.dataframe(
            df_sym, use_container_width=True, hide_index=True,
            column_config={
                "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                "Total $": st.column_config.NumberColumn(format="$%+.2f"),
                "Avg $": st.column_config.NumberColumn(format="$%+.2f"),
                "PF": st.column_config.NumberColumn(format="%.2f"),
                "Avg hold (d)": st.column_config.NumberColumn(format="%.1f"),
            },
        )

with col_strat:
    section("By Strategy", "Attributed to whichever strategy opened the trade.", level=3)
    try:
        strat_rows = api.pnl_by_strategy()
    except Exception as exc:
        st.error(f"Cannot load /pnl/by-strategy: {exc}")
        strat_rows = []
    if not strat_rows:
        st.info("No closed trades yet.")
    else:
        df_strat = pd.DataFrame(strat_rows)[[
            "key", "trade_count", "win_rate_pct",
            "total_realized_pnl", "avg_pnl", "profit_factor", "avg_hold_days",
        ]]
        df_strat.columns = ["Strategy", "Trades", "Win %", "Total $", "Avg $", "PF", "Avg hold (d)"]
        st.dataframe(
            df_strat, use_container_width=True, hide_index=True,
            column_config={
                "Win %": st.column_config.NumberColumn(format="%.1f%%"),
                "Total $": st.column_config.NumberColumn(format="$%+.2f"),
                "Avg $": st.column_config.NumberColumn(format="$%+.2f"),
                "PF": st.column_config.NumberColumn(format="%.2f"),
                "Avg hold (d)": st.column_config.NumberColumn(format="%.1f"),
            },
        )


# ── Closed trades log ───────────────────────────────────────────────────────
divider()
section("Closed Trades", "Every realized round-trip, most recent first.")

f1, f2, f3 = st.columns([2, 2, 2])
with f1:
    flt_sym = st.text_input("Symbol filter", value="", key="pnl_flt_sym").strip().upper() or None
with f2:
    strat_options = ["(all)"] + [r["key"] for r in (strat_rows or [])]
    flt_strat_pick = st.selectbox("Strategy filter", strat_options, index=0, key="pnl_flt_strat")
    flt_strat = None if flt_strat_pick == "(all)" else flt_strat_pick
with f3:
    flt_limit = st.selectbox("Limit", [100, 250, 500, 1000, 5000], index=2, key="pnl_flt_limit")

try:
    closed = api.pnl_closed_trades(symbol=flt_sym, strategy=flt_strat, limit=flt_limit)
except Exception as exc:
    st.error(f"Cannot load /pnl/closed-trades: {exc}")
    closed = []

if not closed:
    st.info("No closed trades match the current filters.")
else:
    df_cl = pd.DataFrame(closed)

    def _cur_cl(row) -> str:
        return currency_symbol(row.get("broker"))

    def _m(row, col) -> str:
        return money(row.get(col), currency=_cur_cl(row), decimals=2)

    def _m_signed(row, col) -> str:
        v = row.get(col)
        return "—" if v is None else f"{_cur_cl(row)}{float(v):+,.2f}"

    df_cl["_buy"] = df_cl.apply(lambda r: _m(r, "buy_price"), axis=1)
    df_cl["_sell"] = df_cl.apply(lambda r: _m(r, "sell_price"), axis=1)
    df_cl["_pnl"] = df_cl.apply(lambda r: _m_signed(r, "realized_pnl"), axis=1)

    df_show = df_cl[[
        "sell_at", "symbol", "quantity", "_buy", "_sell",
        "_pnl", "realized_pct", "hold_days",
        "buy_strategy", "broker", "is_paper",
    ]].copy()
    df_show.columns = ["Closed", "Symbol", "Qty", "Buy", "Sell",
                       "P/L", "P/L %", "Hold (d)",
                       "Strategy", "Broker", "Paper"]
    st.dataframe(
        df_show, use_container_width=True, hide_index=True,
        column_config={
            "Qty": st.column_config.NumberColumn(format="%.4f"),
            "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Hold (d)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(f"{len(df_show):,} trade(s) shown. Money shown in each row's broker currency (₹ for Zerodha, $ otherwise).")

# ── Tight Trails on Open Positions ────────────────────────────────────────────
st.divider()
section(
    "Tight Trails — Open Positions",
    "Held positions whose assigned strategy fired a SELL signal. "
    "**Protection** tells you whether a real trailing stop is actually resting on the "
    "broker (🟢 ARMED) or whether the trail trigger shown is only a preview (🔴 NOT ARMED). "
    "A NOT-ARMED position is currently unprotected — nothing will auto-close it.",
)
st.caption(
    "**Protection** — 🟢 ARMED means a live STOP/TRAILING_STOP is resting at the broker; it "
    "ratchets up automatically and auto-closes when price hits it. 🔴 NOT ARMED means no order "
    "is resting yet — the trail trigger is only an estimate and will NOT close the position. "
    "**Trail trigger** — the live order's level (marked *live*) once armed, or a peak-based "
    "**estimate** (peak since signal × (1 − trail%), floored at signal + 0.25%) marked "
    "*NOT live* before arming. **Move since signal** = (last − signal) / signal."
)

try:
    _armed = api.pnl_open_trails()
except Exception as exc:
    st.error(f"Cannot load /pnl/open-trails: {exc}")
    _armed = []

if not _armed:
    st.info(
        "No held position currently has a SELL signal from its assigned strategy. "
        "This populates when an assigned strategy fires a SELL on a position you hold."
    )
else:
    # Protection split — surface unprotected positions loudly.
    _n_armed = sum(1 for r in _armed if r.get("order_type") in ("STOP", "TRAILING_STOP"))
    _n_unarmed = len(_armed) - _n_armed
    if _n_unarmed:
        _syms = ", ".join(r["symbol"] for r in _armed
                          if r.get("order_type") not in ("STOP", "TRAILING_STOP"))
        st.warning(
            f"⚠️ **{_n_unarmed} position(s) NOT protected:** {_syms}. "
            "A SELL signal fired but no trailing stop is resting on the broker yet, so "
            "nothing will auto-close them. The scheduler arms the trail on its next live "
            "cycle (every 15 min, market hours) — restart the API server if it hasn't picked "
            "up the latest code. The 'Trail trigger' below is only an estimate until armed.",
            icon="⚠️",
        )

    _n_help = sum(1 for r in _armed if r.get("trail_helping") is True)
    _n_hurt = sum(1 for r in _armed if r.get("trail_helping") is False)
    _moves = [r["move_since_signal_pct"] for r in _armed if r.get("move_since_signal_pct") is not None]
    _avg_move = round(sum(_moves) / len(_moves), 2) if _moves else 0.0
    kpi_row([
        ("Positions", str(len(_armed))),
        ("🟢 Protected (armed)", f"{_n_armed}"),
        ("🔴 Unprotected", f"{_n_unarmed}"),
        ("Avg move since signal", f"{_avg_move:+.2f}%"),
    ])

    df_at = pd.DataFrame(_armed)

    def _cur_at(row) -> str:
        return currency_symbol(row.get("broker"))

    def _money_at(row, col, signed=False) -> str:
        if row.get(col) is None:
            return "—"
        cur = _cur_at(row)
        return f"{cur}{float(row[col]):+,.2f}" if signed else f"{cur}{float(row[col]):,.2f}"

    def _is_armed(row) -> bool:
        return row.get("order_type") in ("STOP", "TRAILING_STOP")

    def _is_india(row) -> bool:
        return str(row.get("broker", "")).lower() == "zerodha"

    def _protection(row) -> str:
        ot = row.get("order_type")
        if ot == "TRAILING_STOP":
            return "🟢 ARMED (broker trails)"
        if ot == "STOP":
            # A static STOP is the EXPECTED protection on Zerodha (no native
            # trailing on Kite) — the Chandelier job ratchets it every 15 min.
            if _is_india(row):
                return "🟢 ARMED (static stop · Chandelier ratchets)"
            return "🟢 ARMED (static stop)"
        return "🔴 NOT ARMED (estimate only)"

    def _trigger_at(row) -> str:
        ot = row.get("order_type")
        cur = _cur_at(row)
        # Real resting order → the live trigger.
        if ot == "TRAILING_STOP" and row.get("trail_pct"):
            return f"{float(row['trail_pct']):.1f}% trail (live)"
        if ot == "STOP" and row.get("stop_price") is not None:
            return f"{_money_at(row, 'stop_price')} (live)"
        # No order resting → estimate only, clearly NOT protecting.
        if ot == "SIGNAL_ONLY":
            est = row.get("est_trail_trigger")
            tp = row.get("trail_pct")
            if est is not None:
                tail = f" {float(tp):.1f}%" if tp else ""
                return f"~{cur}{float(est):,.2f} est.{tail} — NOT live"
            return "— NOT live (no estimate)"
        if row.get("stop_price") is not None:
            return f"{_money_at(row, 'stop_price')} (live)"
        return "—"

    def _signal_when(row) -> str:
        sa = row.get("signal_at") or row.get("armed_at")
        return str(sa)[:16].replace("T", " ") if sa else "—"

    def _peak_at(row) -> str:
        # Durable high-water mark the trail ratchets off. The stop holds at
        # peak × (1 - trail%) and does NOT drop when price pulls back below it.
        pk = row.get("peak_price")
        if pk is None:
            return "—"
        return _money_at(row, "peak_price")

    df_at["_sig_px"]    = df_at.apply(lambda r: _money_at(r, "signal_price"), axis=1)
    df_at["_last"]      = df_at.apply(lambda r: _money_at(r, "last_price"), axis=1)
    df_at["_peak"]      = df_at.apply(_peak_at, axis=1)
    df_at["_upnl"]      = df_at.apply(lambda r: _money_at(r, "unrealized_pnl", signed=True), axis=1)
    df_at["_trigger"]   = df_at.apply(_trigger_at, axis=1)
    df_at["_sig_when"]  = df_at.apply(_signal_when, axis=1)
    df_at["_protection"] = df_at.apply(_protection, axis=1)
    df_at["_status"]    = df_at["trail_helping"].apply(
        lambda v: "🟢 Riding" if v is True else ("🔴 Below signal" if v is False else "—")
    )

    df_at_show = df_at[[
        "symbol", "quantity", "signal_strategy", "_sig_when", "_sig_px",
        "_last", "_peak", "move_since_signal_pct", "_protection", "_trigger", "_upnl",
        "_status", "days_armed", "broker",
    ]].copy()
    df_at_show.columns = [
        "Symbol", "Qty", "Strategy", "Signal fired", "Signal price",
        "Last", "Peak", "Move since signal", "Protection", "Trail trigger", "Unrealized",
        "Status", "Days since signal", "Broker",
    ]
    st.dataframe(
        df_at_show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Qty": st.column_config.NumberColumn(format="%.4f"),
            "Move since signal": st.column_config.NumberColumn(format="%+.2f%%"),
            "Days since signal": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        f"{len(df_at_show)} position(s) · {_n_armed} protected, {_n_unarmed} unprotected. "
        "Once a trailing stop is ARMED, the broker auto-ratchets it up and auto-closes the "
        "position when price hits it — the trade then moves to the Trail Stop Audit below. "
        "A 🔴 NOT ARMED row has no live stop and will not auto-close until the scheduler arms it. "
        "India (Zerodha) positions show **static stop · Chandelier ratchets** — Kite has no "
        "native trailing-stop order, so the bot places a static STOP and ratchets it up every "
        "15 min during market hours. That's the expected, fully-protected state for India names."
    )

# ── Trail Stop Audit (Approach C peak-capture) ───────────────────────────────
st.divider()
section(
    "Trail Stop Audit",
    "Approach C: after a SELL signal the bot arms a floored trailing stop and "
    "lets the position ride. This audits how much of the post-signal run-up the "
    "trail actually captured — comparing the signal price, the PEAK reached "
    "after it, and where the trail finally exited.",
)
st.caption(
    "**Signal price** = price when the SELL signal fired. "
    "**Peak after signal** = the high-water mark the trail ratcheted off (recorded live). "
    "**Exit** = where the trail filled. "
    "**Capture efficiency** = (exit − signal) / (peak − signal) — of the upside that was "
    "actually available after the signal, the share the trail kept. 100% = exited at the "
    "peak; 0% = no better than selling at the signal; below 0% = exited under the signal."
)

# Re-use the closed trades already fetched above (no extra API call). The audit
# needs a recorded PEAK (trail_peaks) — only trades the bot trailed have one.
_trail_rows = [
    r for r in (closed or [])
    if r.get("signal_price") is not None and r.get("peak_price") is not None
]

if not _trail_rows:
    st.info(
        "No peak-capture data yet. This fills in as the bot arms floored trailing "
        "stops (Approach C) and those positions close — each records its signal "
        "price and the peak it rode to. Older closes (before peak tracking, or "
        "reconciled from the broker) won't have a peak and are shown in the "
        "Closed Trades table above instead."
    )
else:
    df_tr = pd.DataFrame(_trail_rows)
    eff = df_tr["capture_efficiency_pct"].dropna()

    # ── Summary ──────────────────────────────────────────────────────────────
    n_audited = len(df_tr)
    n_trail_exit = int((df_tr["exit_type"] == "trail").sum())
    avg_eff = eff.mean() if len(eff) else 0.0
    # "Beat the signal" = exited above the signal price (trail added net value).
    beat_signal = int((df_tr["trail_captured_pct"] > 0).sum())
    # Extra $ the trail added vs selling at the signal price, summed.
    extra_dollars = sum(
        ((r.get("sell_price") or 0) - (r.get("signal_price") or 0)) * (r.get("quantity") or 0)
        for r in _trail_rows
    )

    ta1, ta2, ta3, ta4 = st.columns(4)
    ta1.metric("Trades audited", f"{n_audited}",
               help="Closed trades with a recorded signal price AND peak (bot-trailed).")
    ta2.metric("Avg capture efficiency", f"{avg_eff:.0f}%",
               help="Average share of the available run-up (signal → peak) the trail kept. "
                    "100% = exited at the peak; 0% = no better than the signal price.")
    ta3.metric("Beat the signal", f"{beat_signal} / {n_audited}",
               help="Trades where the trail exited ABOVE the signal price — i.e. riding "
                    "past the signal added net value vs selling immediately.")
    ta4.metric("Extra captured ($)", f"${extra_dollars:+,.2f}",
               help="Total dollars the trail added vs selling everything at the signal "
                    "price: Σ (exit − signal) × qty.")

    # ── Per-trade breakdown (signal → peak → exit) ───────────────────────────
    section("Per-trade breakdown")

    def _bar(eff_val) -> str:
        # Tiny text gauge so capture efficiency reads at a glance.
        if eff_val is None:
            return "—"
        filled = max(0, min(10, round(eff_val / 10)))
        return "█" * filled + "░" * (10 - filled)

    def _trail_row_fmt(row):
        sig_p = row.get("signal_price")
        peak_p = row.get("peak_price")
        exit_p = row.get("sell_price")
        eff_val = row.get("capture_efficiency_pct")
        return {
            "Symbol": row.get("symbol", ""),
            "Signal fired": (row.get("signal_at") or "")[:16].replace("T", " "),
            "Signal $": f"${sig_p:.2f}" if sig_p else "—",
            "Peak $": f"${peak_p:.2f}" if peak_p else "—",
            "Exit $": f"${exit_p:.2f}" if exit_p else "—",
            "Ran up": f"{((peak_p - sig_p) / sig_p * 100):+.2f}%" if (sig_p and peak_p) else "—",
            "Capture": f"{eff_val:.0f}%" if eff_val is not None else "—",
            "": _bar(eff_val),
            "Total P/L %": f"{row.get('realized_pct', 0):+.2f}%",
            "Exit": row.get("exit_type", "?"),
        }

    df_audit = pd.DataFrame([_trail_row_fmt(r) for r in _trail_rows])

    def _highlight_capture(val):
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
            return "color: #2ec4b6" if v >= 50 else ("color: #e8a23d" if v >= 0 else "color: #e84545")
        except Exception:
            return ""

    st.dataframe(
        df_audit.style.map(_highlight_capture, subset=["Capture"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "**Capture** colour: 🟢 ≥50% of the run-up kept · 🟠 0–50% · 🔴 exited below the "
        "signal. A low capture on a big **Ran up** means the trail width may be too wide "
        "(giving back too much from the peak); consistently 100% means it could ride looser."
    )
