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
    df_eq["at"] = pd.to_datetime(df_eq["at"])

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

# ── Trail Stop Audit ─────────────────────────────────────────────────────────
st.divider()
section(
    "Trail Stop Audit",
    "For each closed trade: when the SELL signal fired, the price at that moment, "
    "where the trailing stop actually exited, and whether the trail added or cost gains. "
    "Only trades with a recorded signal price are shown.",
)
st.caption(
    "**Signal price** = price when strategy SELL signal fired (RSI/extension threshold hit). "
    "**Exit price** = where the trailing stop filled. "
    "**Trail captured %** = (exit − signal) / signal — positive means the trail let you "
    "ride extra upside after the signal; negative means the stock reversed before the signal "
    "price and the trail exited below it."
)

# Re-use the same closed trades already fetched above (no extra API call)
_trail_rows = [r for r in (closed or []) if r.get("signal_price") is not None]

if not _trail_rows:
    st.info(
        "No trail audit data yet — signal prices are recorded starting from when the "
        "trailing stop feature was enabled. Run a live cycle with auto_protective_stop "
        "enabled to populate this section."
    )
else:
    df_tr = pd.DataFrame(_trail_rows)

    # ── Summary stats ────────────────────────────────────────────────────────
    trail_only = df_tr[df_tr["exit_type"] == "trailing_stop"]
    n_trail = len(trail_only)
    n_market = len(df_tr[df_tr["exit_type"] == "market"])
    n_total = len(df_tr)

    captured = trail_only["trail_captured_pct"].dropna()
    avg_captured = captured.mean() if len(captured) else 0.0
    positive_trails = (captured > 0).sum()
    negative_trails = (captured <= 0).sum()

    ta1, ta2, ta3, ta4, ta5 = st.columns(5)
    ta1.metric("Trail exits", n_trail)
    ta2.metric("Market exits", n_market)
    ta3.metric("Avg extra captured", f"{avg_captured:+.2f}%",
               help="Average (exit price − signal price) / signal price across trail exits. "
                    "Positive = trail let you ride more upside after the signal.")
    ta4.metric("Trail helped", f"{positive_trails} / {n_trail}",
               help="Trades where exit price > signal price (trail captured extra upside).")
    ta5.metric("Trail hurt", f"{negative_trails} / {n_trail}",
               help="Trades where exit price < signal price (stock reversed before signal "
                    "price and trail exited below it — usually a fast drop).")

    # ── Scatter: signal price vs exit price ──────────────────────────────────
    if n_trail > 0:
        import plotly.graph_objects as _go

        fig_scatter = _go.Figure()

        # Reference line: exit = signal (no extra gain)
        all_prices = pd.concat([
            trail_only["signal_price"].dropna(),
            trail_only["sell_price"].dropna(),
        ])
        pmin, pmax = float(all_prices.min()), float(all_prices.max())
        fig_scatter.add_trace(_go.Scatter(
            x=[pmin, pmax], y=[pmin, pmax],
            mode="lines",
            line=dict(color="#555", dash="dash", width=1),
            name="Exit = Signal (no extra gain)",
            hoverinfo="skip",
        ))

        # Dots coloured by trail_captured_pct
        col_vals = trail_only["trail_captured_pct"].fillna(0)
        colours = ["#2ec4b6" if v >= 0 else "#e84545" for v in col_vals]
        hover = [
            f"<b>{row['symbol']}</b><br>"
            f"Signal @ {row.get('signal_at','')[:10] if row.get('signal_at') else '?'}<br>"
            f"Signal price: ${row['signal_price']:.2f}<br>"
            f"Exit price: ${row['sell_price']:.2f}<br>"
            f"Trail captured: {row.get('trail_captured_pct', 0):+.2f}%<br>"
            f"Trail width: {row.get('trail_pct', '?')}%<br>"
            f"P/L: {row.get('realized_pct', 0):+.2f}%"
            for _, row in trail_only.iterrows()
        ]
        fig_scatter.add_trace(_go.Scatter(
            x=trail_only["signal_price"].tolist(),
            y=trail_only["sell_price"].tolist(),
            mode="markers+text",
            text=trail_only["symbol"].tolist(),
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(color=colours, size=10, line=dict(color="#1a1a2e", width=1)),
            hovertext=hover,
            hoverinfo="text",
            name="Trail exits",
        ))

        fig_scatter.update_layout(
            title="Signal price vs Exit price (trailing stop exits)",
            xaxis_title="Signal price (SELL signal fired here)",
            yaxis_title="Exit price (trailing stop filled here)",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#e0e0e0"),
            height=420,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            "Dots **above** the dashed line = exit price > signal price (trail captured extra upside). "
            "Dots **below** = exit below signal price (fast reversal, trail couldn't hold)."
        )

    # ── Per-trade audit table ────────────────────────────────────────────────
    section("Per-trade breakdown")

    def _trail_row_fmt(row):
        sym = row.get("symbol", "")
        sig_p = row.get("signal_price")
        sig_at = (row.get("signal_at") or "")[:16]
        exit_p = row.get("sell_price")
        trail_pct = row.get("trail_pct")
        captured = row.get("trail_captured_pct")
        pnl_pct = row.get("realized_pct", 0)
        exit_type = row.get("exit_type", "?")
        return {
            "Symbol": sym,
            "Signal fired": sig_at,
            "Signal price": f"${sig_p:.2f}" if sig_p else "—",
            "Exit price": f"${exit_p:.2f}" if exit_p else "—",
            "Trail width": f"{trail_pct:.1f}%" if trail_pct else "—",
            "Trail captured": f"{captured:+.2f}%" if captured is not None else "—",
            "Total P/L %": f"{pnl_pct:+.2f}%",
            "Exit type": exit_type,
        }

    audit_rows = [_trail_row_fmt(r) for r in _trail_rows]
    df_audit = pd.DataFrame(audit_rows)

    # Colour-code the "Trail captured" column header via styling
    def _highlight_captured(val):
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
            return "color: #2ec4b6" if v > 0 else ("color: #e84545" if v < 0 else "")
        except Exception:
            return ""

    st.dataframe(
        # Styler.applymap was removed in pandas 2.1+; .map is the drop-in
        # replacement with the same signature.
        df_audit.style.map(_highlight_captured, subset=["Trail captured"]),
        use_container_width=True,
        hide_index=True,
    )
