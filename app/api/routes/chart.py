"""
Unified chart payload for the live strategy-overlay chart.

The frontend (lightweight-charts in the Streamlit dashboard) is a *renderer
only* — it does not compute trade decisions. This endpoint bundles, for one
(symbol, timeframe) pair:

  * intraday candles (Twelve Data → yfinance fallback, same source as the
    daytrading runner — so the chart's bars match what the strategies see),
  * overlay series (EMA9/21, VWAP, Bollinger, Supertrend) computed from the
    same indicator code the strategies use, so what the trader sees on screen
    is what the strategy is reading,
  * BUY/SELL markers from the backend strategy pipeline (run_signals),
    each with entry/stop/target/confidence/reason,
  * rejected-setup markers carrying the brain's rejection_reason so the
    user can inspect *why* a setup didn't trade.

Schema is deliberately flat and lightweight-charts-friendly: time fields are
unix seconds (UTC), and every marker carries the full payload needed to
populate the side-panel explanation card.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.ema import compute_ema
from app.services.strategy.daytrading.market_open import compute_vwap
from app.services.strategy.daytrading.runner import (
    _fetch_twelvedata, _normalise_df, fetch_intraday, run_signals,
)
from app.services.strategy.daytrading.strategies import ALL_STRATEGIES, STRATEGY_MAP


router = APIRouter(prefix="/chart", tags=["chart"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _r(v: Any, digits: int = 4) -> float | None:
    """Round-or-None, NaN-safe."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def _to_unix(ts: pd.Timestamp) -> int:
    """lightweight-charts wants UTC seconds-since-epoch as an integer."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.tz_convert("UTC").timestamp())


def _signal_time_to_unix(signal_time: str) -> int | None:
    """Parse the runner's ISO-ish 'signal_time' into unix seconds."""
    if not signal_time:
        return None
    try:
        ts = pd.to_datetime(signal_time)
        if ts.tzinfo is None:
            # signal_time is wall-clock ET in the runner; treat as ET.
            ts = ts.tz_localize("America/New_York")
        return int(ts.tz_convert("UTC").timestamp())
    except Exception:
        return None


def _supertrend_line(df: pd.DataFrame, period: int = 10, factor: float = 3.0) -> list[float | None]:
    """ATR-based Supertrend line — same construction as /strategy/chart, but
    returned as a flat list aligned to df.index."""
    n = len(df)
    if n < period + 2:
        return [None] * n
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    hl2 = (high + low) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    line: list[float | None] = [None] * n
    trend = [1] * n
    for i in range(1, n):
        upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1]) if close.iloc[i - 1] > lower.iloc[i - 1] else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1]) if close.iloc[i - 1] < upper.iloc[i - 1] else lower.iloc[i]
        if trend[i - 1] == -1:
            trend[i] = 1 if close.iloc[i] > upper.iloc[i - 1] else -1
        else:
            trend[i] = -1 if close.iloc[i] < lower.iloc[i - 1] else 1
        line[i] = _r(lower.iloc[i]) if trend[i] == 1 else _r(upper.iloc[i])
    line[0] = _r(lower.iloc[0])
    return line


def _series_to_list(s: pd.Series) -> list[float | None]:
    return [_r(v) for v in s.values.tolist()]


def _build_overlays(df: pd.DataFrame, times: list[int]) -> dict[str, list[dict[str, Any]]]:
    """Build lightweight-charts line-series payloads keyed by overlay name.
    Each value is a list of {time, value} dicts with None values dropped."""
    closes = df["Close"]
    overlays: dict[str, list[float | None]] = {}

    try:
        overlays["ema9"] = _series_to_list(compute_ema(closes, 9).values)
    except Exception:
        overlays["ema9"] = [None] * len(df)
    try:
        overlays["ema21"] = _series_to_list(compute_ema(closes, 21).values)
    except Exception:
        overlays["ema21"] = [None] * len(df)
    try:
        overlays["ema50"] = _series_to_list(compute_ema(closes, 50).values)
    except Exception:
        overlays["ema50"] = [None] * len(df)

    try:
        overlays["vwap"] = _series_to_list(compute_vwap(df))
    except Exception:
        overlays["vwap"] = [None] * len(df)

    try:
        bb = compute_bollinger(closes, 20, 2.0)
        overlays["bb_upper"] = _series_to_list(bb.upper.values)
        overlays["bb_middle"] = _series_to_list(bb.middle.values)
        overlays["bb_lower"] = _series_to_list(bb.lower.values)
    except Exception:
        overlays["bb_upper"] = overlays["bb_middle"] = overlays["bb_lower"] = [None] * len(df)

    overlays["supertrend"] = _supertrend_line(df)

    # Convert to lightweight-charts {time,value} pairs, dropping None gaps.
    return {
        key: [{"time": t, "value": v} for t, v in zip(times, vals) if v is not None]
        for key, vals in overlays.items()
    }


def _anchor_to_bar(
    bar_times: list[int],
    closes: list[float],
    signal_unix: int | None,
) -> tuple[int | None, float | None]:
    """Find the bar at-or-before signal_unix; return (bar_time, bar_close)."""
    if signal_unix is None or not bar_times:
        return None, None
    # bar_times is monotonic; binary search would be nicer but list is small.
    chosen_t, chosen_c = None, None
    for t, c in zip(bar_times, closes):
        if t <= signal_unix:
            chosen_t, chosen_c = t, c
        else:
            break
    return chosen_t, chosen_c


# ── endpoint ──────────────────────────────────────────────────────────────────

_VALID_TIMEFRAMES = {"5m": ("5d", "5m"), "15m": ("60d", "15m"), "1m": ("2d", "1m")}


def _fetch_with_source(symbol: str, interval: str, period: str) -> tuple[pd.DataFrame, str]:
    """Fetch intraday bars and report which source produced them.

    Mirrors runner.fetch_intraday's routing:
      - India (NSE) symbols → Upstox ONLY (no fall-through to US providers,
        which return 429 / "possibly delisted" for NSE names).
      - US symbols → Twelve Data → yfinance fallback.

    Returns (df, "upstox"|"twelvedata"|"yfinance"|"none").
    """
    # India (NSE): route to Upstox and short-circuit the US provider chain.
    try:
        from app.services.markets import is_india_symbol
        if is_india_symbol(symbol):
            from app.services.marketdata import upstox_data
            ind = upstox_data.fetch_bars(symbol, interval=interval, period=period)
            return (ind, "upstox") if not ind.empty else (ind, "none")
    except Exception:
        # If India detection / Upstox import fails, fall through to US chain
        # rather than erroring the whole chart request.
        pass

    df = _fetch_twelvedata(symbol, interval, period)
    if not df.empty:
        return df, "twelvedata"
    import yfinance as yf
    raw = yf.download(symbol, period=period, interval=interval, progress=False)
    if raw.empty:
        return raw, "none"
    return _normalise_df(raw), "yfinance"


@router.get("/intraday/{symbol}")
def intraday_chart(
    symbol: str,
    timeframe: Literal["1m", "5m", "15m"] = Query("5m"),
    strategies: str = Query("all", description="Comma-separated strategy names, or 'all'"),
    include_rejected: bool = Query(True),
) -> dict[str, Any]:
    """Unified chart payload: candles + overlays + strategy markers.

    The frontend renders this exactly as returned — *no* strategy logic on
    the client side. Markers carry the full explanation payload the user
    needs to inspect a setup.
    """
    sym = symbol.upper().strip()
    if timeframe not in _VALID_TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"unsupported timeframe {timeframe!r}")

    period, interval = _VALID_TIMEFRAMES[timeframe]
    df, data_source = _fetch_with_source(sym, interval, period)
    if df.empty:
        return {
            "symbol": sym,
            "timeframe": timeframe,
            "candles": [],
            "volume": [],
            "overlays": {},
            "markers": [],
            "rejected_markers": [],
            "data_source": data_source,
            "warning": "no bars available for this symbol/timeframe",
        }

    # Candles + volume in lightweight-charts shape.
    times = [_to_unix(ts) for ts in df.index]
    candles = [
        {
            "time": t,
            "open": _r(o, 2),
            "high": _r(h, 2),
            "low": _r(l, 2),
            "close": _r(c, 2),
        }
        for t, o, h, l, c in zip(
            times, df["Open"].tolist(), df["High"].tolist(),
            df["Low"].tolist(), df["Close"].tolist(),
        )
    ]
    volume = [
        {
            "time": t,
            "value": int(v) if v == v else 0,
            "color": "rgba(38,166,154,0.55)" if c >= o else "rgba(239,83,80,0.55)",
        }
        for t, v, o, c in zip(
            times, df["Volume"].tolist(), df["Open"].tolist(), df["Close"].tolist(),
        )
    ]

    overlays = _build_overlays(df, times)

    # Strategy markers — fully backend-driven. We re-use the same run_signals
    # path the autotrader sees so what's on the chart matches what the
    # platform actually decided.
    if strategies == "all" or not strategies:
        enabled = [s.name for s in ALL_STRATEGIES]
    else:
        requested = [s.strip() for s in strategies.split(",") if s.strip()]
        enabled = [s for s in requested if s in STRATEGY_MAP]
        if not enabled:
            raise HTTPException(
                status_code=400,
                detail=f"no recognised strategy in {requested!r}",
            )

    sig_payload: dict[str, Any] = {}
    try:
        sig_payload = run_signals(sym, enabled_strategies=enabled, apply_brain=True)
    except Exception as exc:  # never let the chart 500 because the runner blew up
        sig_payload = {"signals": [], "rejected_signals": [], "error": str(exc)}

    closes_list = [c["close"] for c in candles]

    def _marker_from_signal(sig: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
        raw_signal_time = str(sig.get("signal_time") or "")
        signal_unix = _signal_time_to_unix(raw_signal_time)
        bar_time, bar_close = _anchor_to_bar(times, closes_list, signal_unix)
        # Fall back to the latest bar so a marker without a time still shows up
        # at the right-hand edge — better than dropping it silently.
        anchor_fallback = False
        if bar_time is None:
            bar_time = times[-1] if times else None
            bar_close = closes_list[-1] if closes_list else None
            # Only flag as a true fallback when we had a signal_time string but
            # couldn't parse/anchor it — empty signal_time is a different shape.
            anchor_fallback = bool(raw_signal_time)
        direction = str(sig.get("direction") or "").upper()
        side = "BUY" if direction == "BUY" else ("SELL" if "SELL" in direction else direction)
        if not accepted:
            reason = sig.get("brain_reason") or sig.get("reason") or ""
        else:
            reason = sig.get("reason") or ""
        return {
            "time": bar_time,
            "anchor_price": _r(sig.get("entry_price") or bar_close, 2),
            "side": side,
            "accepted": accepted,
            "strategy": sig.get("strategy") or "",
            "timeframe": sig.get("timeframe") or "",
            "entry_price": _r(sig.get("entry_price"), 2),
            "stop_price": _r(sig.get("stop_price"), 2),
            "target_price": _r(sig.get("target_price"), 2),
            "confidence": _r(sig.get("confidence"), 3),
            "r_multiple": _r(sig.get("r_multiple"), 2),
            "regime": sig.get("regime") or sig.get("brain_market_state") or "",
            "reason": reason,
            "explanation": sig.get("brain_reason") or sig.get("reason") or "",
            "anchor_fallback": anchor_fallback,
            "signal_time_raw": raw_signal_time,
        }

    accepted_markers = [
        _marker_from_signal(s, accepted=True) for s in sig_payload.get("signals", [])
    ]
    rejected_markers = []
    if include_rejected:
        rejected_markers = [
            _marker_from_signal(s, accepted=False)
            for s in sig_payload.get("rejected_signals", [])
        ]
    fallback_anchored = sum(
        1 for m in accepted_markers + rejected_markers if m.get("anchor_fallback")
    )

    # Market timezone for display — IST for NSE symbols, ET for US.
    try:
        from app.services.strategy.daytrading.market_open import market_session, IST
        _sess = market_session(sym)
        _market_tz = "Asia/Kolkata" if _sess.tz is IST else "America/New_York"
        _tz_label = "IST" if _sess.tz is IST else "ET"
    except Exception:
        _market_tz, _tz_label = "America/New_York", "ET"

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "candles": candles,
        "volume": volume,
        "overlays": overlays,
        "markers": accepted_markers,
        "rejected_markers": rejected_markers,
        "regime": sig_payload.get("regime"),
        "brain_status": sig_payload.get("brain_status"),
        "market_status": sig_payload.get("market_status"),
        "diagnostics": sig_payload.get("diagnostics"),
        "policy_blocked": sig_payload.get("policy_blocked", False),
        "policy_reason": sig_payload.get("policy_reason"),
        "data_source": data_source,
        "fallback_anchored": fallback_anchored,
        "strategy_timeframes": ["5m", "15m"],
        "market_tz": _market_tz,
        "tz_label": _tz_label,
    }


@router.get("/strategies")
def list_chart_strategies() -> list[dict[str, Any]]:
    """Strategy names the chart's filter dropdown can offer."""
    return [{"name": s.name, "timeframe": s.timeframe} for s in ALL_STRATEGIES]
