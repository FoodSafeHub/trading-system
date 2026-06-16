"""Reusable advanced chart components for the trading dashboard.

Built for traders: every chart is a candlestick by default, with volume on a
subplot, optional indicator overlays (EMA9/21/50/200, VWAP, Bollinger,
Supertrend), and BUY/SELL signal markers anchored to the actual fill price on
the candle. Equity curves are rendered as OHLC candles built from the daily
equity track plus markers for each trade.

Two flavors:
  - `price_candles(...)`   → full price/volume/RSI/MACD stacked chart with
                              indicator overlays + BUY/SELL trade markers.
  - `equity_candles(...)`  → equity curve as candles + signal markers + DD ribbon.
  - `tradingview_embed(...)` → drops the full TradingView advanced widget into
                                any page (used by Charts page + Scanner detail).

All charts default to a Schwab-/TradingView-like dark theme with green/red
candles and a subtle grid.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


# ── Theme — aligned with the "Midnight" dashboard palette (_theme.py) ───────────
GREEN = "#34d399"   # bullish candle / BUY marker (emerald, matches --pos)
RED   = "#fb7185"   # bearish candle / SELL marker (rose, matches --neg)
GRID  = "rgba(255,255,255,0.06)"
BG    = "rgba(20,22,34,1)"   # matches --panel-solid (#141622) for seamless cards
TEXT  = "rgba(238,240,246,0.9)"

_OVERLAY_COLORS = {
    "ema9":   "#26C6DA",
    "ema21":  "#FF9800",
    "ema50":  "#AB47BC",
    "ema200": "#EF5350",
    "sma50":  "#FFD54F",
    "sma200": "#90A4AE",
    "vwap":   "#42A5F5",
    "bb_upper":  "rgba(176, 190, 197, 0.45)",
    "bb_middle": "rgba(176, 190, 197, 0.30)",
    "bb_lower":  "rgba(176, 190, 197, 0.45)",
    "supertrend": "#FFCA28",
}

_OVERLAY_LABELS = {
    "ema9": "EMA 9", "ema21": "EMA 21", "ema50": "EMA 50", "ema200": "EMA 200",
    "sma50": "SMA 50", "sma200": "SMA 200",
    "vwap": "VWAP",
    "bb_upper": "BB Upper", "bb_middle": "BB Mid", "bb_lower": "BB Lower",
    "supertrend": "Supertrend",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _layout(fig: go.Figure, height: int, title: str | None = None, *, range_selector: bool = True) -> None:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=8, r=8, t=40 if title else 12, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=TEXT),
        ),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        title=dict(text=title, x=0.01, y=0.97, font=dict(size=14, color=TEXT)) if title else None,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikemode="across",
                     spikethickness=1, spikecolor="rgba(255,255,255,0.25)")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikethickness=1,
                     spikecolor="rgba(255,255,255,0.25)")
    if range_selector:
        fig.update_xaxes(
            rangeselector=dict(
                bgcolor="rgba(255,255,255,0.05)",
                activecolor="rgba(124,92,255,0.55)",
                bordercolor="rgba(255,255,255,0.08)",
                font=dict(color=TEXT, size=10),
                buttons=[
                    dict(count=5,  label="5D",  step="day",   stepmode="backward"),
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=3,  label="3M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
            type="date",
        )


def _candle_trace(dates, opens, highs, lows, closes, name: str = "Price") -> go.Candlestick:
    return go.Candlestick(
        x=dates, open=opens, high=highs, low=lows, close=closes,
        name=name,
        increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED,   width=1), fillcolor=RED),
        whiskerwidth=0.4,
        hoverlabel=dict(bgcolor="rgba(20,22,34,0.95)", font=dict(family="monospace", size=11)),
    )


def _volume_trace(dates, volumes, opens, closes) -> go.Bar:
    colors = [
        "rgba(52,211,153,0.55)" if (c is not None and o is not None and c >= o) else "rgba(251,113,133,0.55)"
        for o, c in zip(opens, closes)
    ]
    return go.Bar(x=dates, y=volumes, marker_color=colors, name="Volume",
                  hovertemplate="Vol %{y:,.0f}<extra></extra>")


def _signal_markers(
    fig: go.Figure,
    *,
    buy_dates: Sequence,
    buy_prices: Sequence,
    sell_dates: Sequence,
    sell_prices: Sequence,
    row: int = 1,
    col: int = 1,
    buy_text: Sequence | None = None,
    sell_text: Sequence | None = None,
) -> None:
    if buy_dates:
        fig.add_trace(
            go.Scatter(
                x=buy_dates, y=buy_prices,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color=GREEN,
                            line=dict(color="rgba(0,0,0,0.6)", width=1)),
                name="BUY",
                text=buy_text if buy_text else [f"BUY @ ${p:,.2f}" for p in buy_prices],
                hovertemplate="%{text}<br>%{x}<extra></extra>",
            ),
            row=row, col=col,
        )
    if sell_dates:
        fig.add_trace(
            go.Scatter(
                x=sell_dates, y=sell_prices,
                mode="markers",
                marker=dict(symbol="triangle-down", size=14, color=RED,
                            line=dict(color="rgba(0,0,0,0.6)", width=1)),
                name="SELL",
                text=sell_text if sell_text else [f"SELL @ ${p:,.2f}" for p in sell_prices],
                hovertemplate="%{text}<br>%{x}<extra></extra>",
            ),
            row=row, col=col,
        )


def _to_series(values: Iterable | None, n: int) -> list[Any]:
    if not values:
        return [None] * n
    out = list(values)
    if len(out) < n:
        out = [None] * (n - len(out)) + out
    return out[:n]


# ── Price candlestick chart with full indicator stack ──────────────────────────
def price_candles(
    payload: dict,
    *,
    trades: list[dict] | None = None,
    overlays: Sequence[str] = ("ema9", "ema21", "ema50", "vwap"),
    include_volume: bool = True,
    include_rsi: bool = True,
    include_macd: bool = True,
    height: int | None = None,
    title: str | None = None,
) -> go.Figure:
    """Render a multi-pane candlestick chart.

    `payload` is the response from `/strategy/chart/{symbol}` (keys: dates,
    open/high/low/close/volume, indicators dict).

    `trades` is a list of `{date, side, price}` dicts; markers are anchored to
    `price` so they sit on top of the actual candle bodies.
    """
    dates  = payload.get("dates", [])
    opens  = payload.get("open",  [])
    highs  = payload.get("high",  [])
    lows   = payload.get("low",   [])
    closes = payload.get("close", [])
    vols   = payload.get("volume", []) or []
    inds   = payload.get("indicators", {}) or {}
    n = len(dates)

    # Subplot grid: price always present; volume + RSI + MACD are stacked below.
    rows: list[tuple[str, float]] = [("price", 0.62)]
    if include_volume: rows.append(("volume", 0.12))
    if include_rsi:    rows.append(("rsi",    0.13))
    if include_macd:   rows.append(("macd",   0.13))

    # Re-normalise heights to sum 1.
    total = sum(h for _, h in rows)
    row_heights = [h / total for _, h in rows]
    row_index = {name: i + 1 for i, (name, _) in enumerate(rows)}

    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights,
    )

    fig.add_trace(_candle_trace(dates, opens, highs, lows, closes), row=row_index["price"], col=1)

    # Overlays on price pane.
    for key in overlays:
        series = inds.get(key)
        if not series:
            continue
        fig.add_trace(
            go.Scatter(
                x=dates, y=_to_series(series, n),
                mode="lines", line=dict(width=1.3, color=_OVERLAY_COLORS.get(key, "#aaa")),
                name=_OVERLAY_LABELS.get(key, key),
                hovertemplate=f"{_OVERLAY_LABELS.get(key, key)} %{{y:.2f}}<extra></extra>",
            ),
            row=row_index["price"], col=1,
        )

    # Bollinger bands (if requested via overlays containing 'bb_').
    if "bb_upper" in overlays or "bb_lower" in overlays:
        upper = inds.get("bb_upper")
        lower = inds.get("bb_lower")
        middle = inds.get("bb_middle")
        if upper and lower:
            fig.add_trace(
                go.Scatter(x=dates, y=upper, mode="lines",
                           line=dict(width=1, color=_OVERLAY_COLORS["bb_upper"]),
                           name="BB Upper", showlegend=False),
                row=row_index["price"], col=1,
            )
            fig.add_trace(
                go.Scatter(x=dates, y=lower, mode="lines",
                           line=dict(width=1, color=_OVERLAY_COLORS["bb_lower"]),
                           fill="tonexty", fillcolor="rgba(176,190,197,0.06)",
                           name="BB Lower", showlegend=False),
                row=row_index["price"], col=1,
            )
        if middle:
            fig.add_trace(
                go.Scatter(x=dates, y=middle, mode="lines",
                           line=dict(width=1, dash="dot", color=_OVERLAY_COLORS["bb_middle"]),
                           name="BB Mid", showlegend=False),
                row=row_index["price"], col=1,
            )

    # Supertrend
    if "supertrend" in overlays and inds.get("supertrend"):
        fig.add_trace(
            go.Scatter(x=dates, y=inds["supertrend"], mode="lines",
                       line=dict(width=1.5, color=_OVERLAY_COLORS["supertrend"], dash="dash"),
                       name="Supertrend"),
            row=row_index["price"], col=1,
        )

    # Trade markers anchored to fill price.
    if trades:
        buys  = [t for t in trades if str(t.get("side", "")).upper() == "BUY"]
        sells = [t for t in trades if "SELL" in str(t.get("side", "")).upper()]
        _signal_markers(
            fig,
            buy_dates=[t.get("date") for t in buys],
            buy_prices=[t.get("price") for t in buys],
            sell_dates=[t.get("date") for t in sells],
            sell_prices=[t.get("price") for t in sells],
            row=row_index["price"],
        )

    # Volume pane.
    if include_volume and vols:
        fig.add_trace(_volume_trace(dates, vols, opens, closes),
                      row=row_index["volume"], col=1)

    # RSI pane with 30/70 guides.
    if include_rsi:
        rsi = inds.get("rsi14")
        if rsi:
            fig.add_trace(
                go.Scatter(x=dates, y=rsi, mode="lines",
                           line=dict(width=1.2, color="#BA68C8"), name="RSI 14"),
                row=row_index["rsi"], col=1,
            )
            for level, color in [(30, "rgba(38,166,154,0.4)"), (70, "rgba(239,83,80,0.4)")]:
                fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"),
                              row=row_index["rsi"], col=1)
            fig.update_yaxes(range=[0, 100], row=row_index["rsi"], col=1)

    # MACD pane (line, signal, histogram).
    if include_macd:
        macd = inds.get("macd")
        sig  = inds.get("macd_signal")
        hist = inds.get("macd_hist")
        if macd:
            fig.add_trace(
                go.Scatter(x=dates, y=macd, mode="lines",
                           line=dict(width=1.2, color="#26C6DA"), name="MACD"),
                row=row_index["macd"], col=1,
            )
        if sig:
            fig.add_trace(
                go.Scatter(x=dates, y=sig, mode="lines",
                           line=dict(width=1.2, color="#FFA726"), name="Signal"),
                row=row_index["macd"], col=1,
            )
        if hist:
            colors = ["rgba(38,166,154,0.55)" if (h or 0) >= 0 else "rgba(239,83,80,0.55)" for h in hist]
            fig.add_trace(
                go.Bar(x=dates, y=hist, marker_color=colors, name="Hist",
                       hovertemplate="Hist %{y:.3f}<extra></extra>"),
                row=row_index["macd"], col=1,
            )

    # Axis labels per pane.
    fig.update_yaxes(title_text="Price", row=row_index["price"], col=1)
    if include_volume: fig.update_yaxes(title_text="Vol", row=row_index["volume"], col=1)
    if include_rsi:    fig.update_yaxes(title_text="RSI", row=row_index["rsi"], col=1)
    if include_macd:   fig.update_yaxes(title_text="MACD", row=row_index["macd"], col=1)

    auto_height = 720 if (include_rsi and include_macd) else 560
    _layout(fig, height or auto_height, title=title, range_selector=True)
    return fig


# ── Equity curve as candles ────────────────────────────────────────────────────
def equity_candles(
    equity_curve: list[dict],
    *,
    trades: list[dict] | None = None,
    initial_capital: float | None = None,
    title: str = "Equity Curve",
    height: int = 460,
    bucket: str = "W",  # 'D' (daily candles), 'W' (weekly), 'M' (monthly)
) -> go.Figure:
    """Render the equity track as OHLC candles plus BUY/SELL markers.

    The equity curve from the backtest is a daily mark-to-market series, so we
    resample it into the chosen bucket (default weekly) to build proper OHLC
    candles — same shape a trader expects to read.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.04,
    )

    if not equity_curve:
        fig.add_annotation(text="No equity data", showarrow=False,
                           font=dict(color=TEXT, size=12))
        _layout(fig, height, title=title)
        return fig

    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Resample to OHLC candles. If user picked daily and the series IS daily,
    # each candle is a flat-line — fall back to weekly automatically when
    # daily would produce zero range candles.
    resampled = df["equity"].resample(bucket).ohlc().dropna()
    if resampled.empty or (bucket == "D" and (resampled["high"] - resampled["low"]).abs().sum() == 0):
        resampled = df["equity"].resample("W").ohlc().dropna()

    fig.add_trace(
        _candle_trace(
            resampled.index, resampled["open"], resampled["high"],
            resampled["low"], resampled["close"], name="Equity",
        ),
        row=1, col=1,
    )

    # Continuous line so users can see the fine-grained track behind the candles.
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["equity"], mode="lines",
            line=dict(color="rgba(38,166,154,0.55)", width=1.2),
            name="Equity (daily)",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Starting-capital guide.
    if initial_capital:
        fig.add_hline(
            y=initial_capital, line=dict(color="rgba(255,255,255,0.35)", dash="dash", width=1),
            annotation_text=f"Start ${initial_capital:,.0f}", annotation_position="right",
            row=1, col=1,
        )

    # BUY/SELL markers anchored to the equity track at the trade date.
    if trades:
        equity_lookup = df["equity"].to_dict()
        def _eq_at(d):
            ts = pd.to_datetime(d)
            if ts in equity_lookup:
                return equity_lookup[ts]
            # nearest preceding day (forward-fill)
            sub = df.loc[:ts]
            return float(sub["equity"].iloc[-1]) if not sub.empty else None

        buys  = [t for t in trades if str(t.get("side", "")).upper() == "BUY"]
        sells = [t for t in trades if "SELL" in str(t.get("side", "")).upper()]
        _signal_markers(
            fig,
            buy_dates=[t["date"] for t in buys],
            buy_prices=[_eq_at(t["date"]) for t in buys],
            sell_dates=[t["date"] for t in sells],
            sell_prices=[_eq_at(t["date"]) for t in sells],
            buy_text=[
                f"BUY {t.get('quantity',0):.2f}u @ ${t.get('price',0):,.2f}<br>Equity ${_eq_at(t['date']) or 0:,.0f}"
                for t in buys
            ],
            sell_text=[
                f"SELL {t.get('quantity',0):.2f}u @ ${t.get('price',0):,.2f}<br>P&L: "
                + ("+$%.2f" % t["pnl"] if t.get("pnl") not in (None,) else "—")
                for t in sells
            ],
            row=1, col=1,
        )

    # Drawdown pane.
    running_max = df["equity"].cummax()
    dd_pct = (df["equity"] / running_max - 1.0) * 100
    fig.add_trace(
        go.Scatter(
            x=df.index, y=dd_pct, mode="lines", fill="tozeroy",
            line=dict(color="#ef5350", width=1),
            fillcolor="rgba(239,83,80,0.18)",
            name="Drawdown %",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Equity ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="DD %", ticksuffix="%", row=2, col=1)
    _layout(fig, height, title=title, range_selector=True)
    return fig


# ── Simple intraday candle chart (for daytrading 5m/15m views) ─────────────────
def intraday_candles(
    df: pd.DataFrame,
    *,
    signals: list[dict] | None = None,
    title: str = "Intraday",
    height: int = 520,
) -> go.Figure:
    """Render an intraday OHLC dataframe (yfinance shape) with signal markers."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )
    if df is None or df.empty:
        fig.add_annotation(text="No intraday data", showarrow=False, font=dict(color=TEXT))
        _layout(fig, height, title=title, range_selector=False)
        return fig

    idx = df.index
    fig.add_trace(_candle_trace(idx, df["Open"], df["High"], df["Low"], df["Close"]),
                  row=1, col=1)

    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(x=idx, y=df["VWAP"], mode="lines",
                                  line=dict(color="#42A5F5", width=1.2), name="VWAP"),
                      row=1, col=1)

    if signals:
        buys  = [s for s in signals if str(s.get("direction", s.get("side", ""))).upper() == "BUY"]
        sells = [s for s in signals if "SELL" in str(s.get("direction", s.get("side", ""))).upper()]
        _signal_markers(
            fig,
            buy_dates=[s.get("timestamp") or s.get("date") for s in buys],
            buy_prices=[s.get("entry_price") or s.get("price") for s in buys],
            sell_dates=[s.get("timestamp") or s.get("date") for s in sells],
            sell_prices=[s.get("entry_price") or s.get("price") for s in sells],
            row=1,
        )

    if "Volume" in df.columns:
        fig.add_trace(_volume_trace(idx, df["Volume"], df["Open"], df["Close"]),
                      row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    _layout(fig, height, title=title, range_selector=False)
    return fig


# ── Shared TradingView symbol routing ─────────────────────────────────────────

_TV_EXCHANGE_GUESS = {"SPY": "AMEX", "QQQ": "NASDAQ", "IWM": "AMEX"}
_TV_NYSE_HINTS = {"JPM", "BAC", "GS", "MS", "WFC", "XOM", "CVX",
                  "JNJ", "UNH", "V", "MA"}


def tv_symbol(sym: str) -> str:
    """Format a ticker for any TradingView widget URL ('NSE:RELIANCE', 'NASDAQ:AAPL', ...).

    * Already-prefixed inputs (NSE:..., NASDAQ:..., BSE:...) pass through.
    * India symbols (Nifty 200 pool OR Upstox NSE map) → NSE:; .BO → BSE:.
    * US fallback: NYSE for known NYSE names; AMEX for SPY/IWM; NASDAQ otherwise.
    """
    s = (sym or "").upper().strip()
    if ":" in s:
        return s
    try:
        from app.services.markets import is_india_symbol
        if is_india_symbol(s):
            if s.endswith(".BO"):
                return f"BSE:{s[:-3]}"
            if s.endswith(".NS"):
                return f"NSE:{s[:-3]}"
            return f"NSE:{s}"
    except Exception:
        pass
    if s in _TV_EXCHANGE_GUESS:
        return f"{_TV_EXCHANGE_GUESS[s]}:{s}"
    return f"NYSE:{s}" if s in _TV_NYSE_HINTS else f"NASDAQ:{s}"


def tradingview_url(sym: str) -> str:
    """Public TradingView chart URL for opening a symbol in a new tab."""
    return f"https://www.tradingview.com/chart/?symbol={tv_symbol(sym)}"


# ── TradingView Advanced Chart embed ───────────────────────────────────────────
def tradingview_embed(
    symbol: str,
    *,
    interval: str = "D",
    studies: Sequence[dict] | None = None,
    watchlist: Sequence[str] | None = None,
    height: int = 720,
) -> None:
    """Drop the official TradingView Advanced Chart widget into the page.

    Pre-loads the standard trader study stack (EMA 9/21/50/200, VWAP,
    Bollinger, Supertrend, RSI, MACD, ATR, Volume) so traders always start
    from a useful baseline.
    """
    default_studies = [
        {"name": "Moving Average Exponential", "override": {"length": 9,   "linecolor": "#26C6DA", "linewidth": 1}},
        {"name": "Moving Average Exponential", "override": {"length": 21,  "linecolor": "#FF9800", "linewidth": 1}},
        {"name": "Moving Average Exponential", "override": {"length": 50,  "linecolor": "#AB47BC", "linewidth": 1}},
        {"name": "Moving Average Exponential", "override": {"length": 200, "linecolor": "#EF5350", "linewidth": 2}},
        {"name": "Bollinger Bands",            "override": {"length": 20, "mult": 2}},
        {"name": "VWAP",                       "override": {}},
        {"name": "Supertrend",                 "override": {"Factor": 3, "ATR Length": 10}},
        {"name": "Relative Strength Index",    "override": {"length": 14}},
        {"name": "MACD",                       "override": {"fast length": 12, "slow length": 26, "signal smoothing": 9}},
        {"name": "Volume",                     "override": {}},
    ]
    cfg = {
        "autosize": True,
        "symbol": symbol,
        "interval": interval,
        "timezone": "America/New_York",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "backgroundColor": BG,
        "gridColor": GRID,
        "hide_top_toolbar": False,
        "hide_legend": False,
        "range": "3M",
        "allow_symbol_change": True,
        "save_image": True,
        "watchlist": list(watchlist) if watchlist else [],
        "studies": list(studies) if studies else default_studies,
        "support_host": "https://www.tradingview.com",
    }
    cfg_json = json.dumps(cfg)
    components.html(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>html,body{{margin:0;padding:0;height:100%;overflow:hidden;background:transparent;}}</style>
</head><body>
  <div class="tradingview-widget-container" style="height:{height}px;width:100%">
    <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>
    <div class="tradingview-widget-copyright">
      <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">
        <span class="blue-text">Track all markets on TradingView</span>
      </a>
    </div>
  </div>
  <script type="text/javascript"
    src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {cfg_json}
  </script>
</body></html>""",
        height=height + 30, scrolling=False,
    )


def tradingview_mini(symbol: str, *, height: int = 220) -> None:
    """Small symbol-overview widget — used for inline scanner candidate previews."""
    cfg = {
        "symbol": symbol,
        "width": "100%",
        "height": height,
        "locale": "en",
        "dateRange": "3M",
        "colorTheme": "dark",
        "trendLineColor": "rgba(38, 166, 154, 1)",
        "underLineColor": "rgba(38, 166, 154, 0.15)",
        "underLineBottomColor": "rgba(38, 166, 154, 0)",
        "isTransparent": True,
        "autosize": True,
        "largeChartUrl": "",
        "chartOnly": False,
        "noTimeScale": False,
    }
    cfg_json = json.dumps(cfg)
    components.html(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>html,body{{margin:0;padding:0;background:transparent;}}</style></head><body>
  <div class="tradingview-widget-container">
    <div class="tradingview-widget-container__widget"></div>
    <script type="text/javascript"
      src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>
    {cfg_json}
    </script>
  </div>
</body></html>""",
        height=height + 20, scrolling=False,
    )


# ── Streamlit one-liner helpers ────────────────────────────────────────────────
def render_price_chart(payload: dict, **kwargs) -> None:
    fig = price_candles(payload, **kwargs)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_equity_chart(equity_curve: list[dict], **kwargs) -> None:
    fig = equity_candles(equity_curve, **kwargs)
    st.plotly_chart(fig, use_container_width=True, theme=None)
