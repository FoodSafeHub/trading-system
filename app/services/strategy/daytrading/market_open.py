"""
Market hours utilities and pre-market regime detection for day trading.

Regime is evaluated at 9:45 AM ET (after first 15 minutes complete).
  BULL_OPEN  = SPY opened above prior close AND current price > VWAP
  BEAR_OPEN  = SPY opened below prior close AND current price < VWAP
  CHOPPY     = neither condition clearly met
  PRE_MARKET = before 9:30 AM ET
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Tuple

_log = logging.getLogger(__name__)

import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

MARKET_OPEN_TIME = time(9, 30)
MARKET_CLOSE_TIME = time(15, 45)
REGIME_EVAL_TIME = time(9, 45)
GAP_FADE_CUTOFF = time(11, 0)
LAST_ENTRY_TIME = time(15, 15)


def now_et() -> datetime:
    return datetime.now(ET)


def is_market_open() -> bool:
    t = now_et().time()
    return MARKET_OPEN_TIME <= t <= MARKET_CLOSE_TIME


def is_pre_market() -> bool:
    t = now_et().time()
    return t < MARKET_OPEN_TIME


def is_past_last_entry() -> bool:
    return now_et().time() >= LAST_ENTRY_TIME


def market_status() -> dict:
    now = now_et()
    t = now.time()
    if t < MARKET_OPEN_TIME:
        status = "PRE_MARKET"
    elif t > time(16, 0):
        status = "CLOSED"
    elif t > MARKET_CLOSE_TIME:
        status = "CLOSING"
    else:
        status = "OPEN"
    return {
        "status": status,
        "is_open": status == "OPEN",
        "time_et": now.strftime("%H:%M:%S %Z"),
        "date": now.strftime("%Y-%m-%d"),
    }


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP that resets at 9:30 AM ET every trading day.
    Grouped by calendar date so overnight data never bleeds into today's VWAP.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    tpv = typical * df["Volume"]
    # group by date so VWAP resets each session
    dates = df.index.normalize()
    cum_tpv = tpv.groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()
    return cum_tpv / cum_vol


def validate_vwap_resets(df: pd.DataFrame) -> dict:
    """
    Verify that VWAP resets correctly at each session boundary.

    Returns a dict with:
      ok: bool — True if all dates show a proper reset
      dates_checked: int
      issues: list of (date, reason) for any anomalies
      open_vwap_by_date: {date: vwap_at_9:35} — sanity snapshot
    """
    if df.empty:
        return {"ok": False, "dates_checked": 0, "issues": ["Empty dataframe"], "open_vwap_by_date": {}}

    vwap = compute_vwap(df)
    dates = sorted(set(df.index.date))
    issues: list[str] = []
    open_vwap: dict[str, float] = {}

    for date in dates:
        day = df[df.index.date == date]
        day_vwap = vwap[df.index.date == date]
        if day.empty:
            continue

        first_bar_vwap = float(day_vwap.iloc[0])
        first_bar_close = float(day["Close"].iloc[0])

        # VWAP at open should equal typical price of first bar (cumsum of 1 bar)
        expected_vwap_open = (float(day["High"].iloc[0]) + float(day["Low"].iloc[0]) + first_bar_close) / 3
        diff = abs(first_bar_vwap - expected_vwap_open)

        if diff > expected_vwap_open * 0.001:   # >0.1% off means bleed from prior day
            issues.append(f"{date}: VWAP at open={first_bar_vwap:.4f} expected={expected_vwap_open:.4f} — possible daily reset failure")
            _log.warning("VWAP reset issue on %s: got %.4f expected %.4f", date, first_bar_vwap, expected_vwap_open)
        else:
            _log.debug("VWAP reset OK on %s: %.4f", date, first_bar_vwap)

        # Log VWAP at ~9:35 (second bar) and last bar
        if len(day_vwap) >= 2:
            vwap_935 = float(day_vwap.iloc[1])
            open_vwap[str(date)] = round(vwap_935, 4)
            _log.debug("VWAP at 9:35 on %s: %.4f | EOD: %.4f", date, vwap_935, float(day_vwap.iloc[-1]))

    return {
        "ok": len(issues) == 0,
        "dates_checked": len(dates),
        "issues": issues,
        "open_vwap_by_date": open_vwap,
    }


def get_spy_regime(df_spy_5m: pd.DataFrame | None = None) -> Tuple[str, float, float, str]:
    """
    Return (regime, spy_vs_vwap_pct).
    Downloads SPY 5m data if not provided.
    """
    if df_spy_5m is None:
        try:
            df_spy_5m = yf.download("SPY", period="2d", interval="5m", progress=False)
            if df_spy_5m.empty:
                return "CHOPPY", 0.0, 0.0, "unknown"
            if isinstance(df_spy_5m.columns, pd.MultiIndex):
                df_spy_5m.columns = df_spy_5m.columns.get_level_values(0)
            df_spy_5m.index = pd.to_datetime(df_spy_5m.index).tz_convert(ET)
        except Exception:
            return "CHOPPY", 0.0, 0.0, "unknown"

    today = now_et().date()
    today_bars = df_spy_5m[df_spy_5m.index.date == today]
    if today_bars.empty:
        return "CHOPPY", 0.0, 0.0, "unknown"

    # prior close = last bar from previous session
    prev_bars = df_spy_5m[df_spy_5m.index.date < today]
    if prev_bars.empty:
        return "CHOPPY", 0.0, 0.0, "unknown"
    prior_close = float(prev_bars["Close"].iloc[-1])

    open_price = float(today_bars["Open"].iloc[0])
    current_price = float(today_bars["Close"].iloc[-1])

    vwap_series = compute_vwap(today_bars)
    current_vwap = float(vwap_series.iloc[-1])
    spy_vs_vwap_pct = round((current_price - current_vwap) / current_vwap * 100, 3)

    opened_above = open_price > prior_close
    above_vwap = current_price > current_vwap

    if opened_above and above_vwap:
        regime = "BULL_OPEN"
    elif (not opened_above) and (not above_vwap):
        regime = "BEAR_OPEN"
    else:
        regime = "CHOPPY"

    # Compute gap using prior close already computed above
    gap_pct = round((open_price - prior_close) / prior_close * 100, 3) if prior_close > 0 else 0.0
    if gap_pct > 1.0:
        gap_type = "up_gap"
    elif gap_pct < -1.0:
        gap_type = "down_gap"
    else:
        gap_type = "normal"

    return regime, spy_vs_vwap_pct, gap_pct, gap_type


def get_prior_close(symbol: str) -> float | None:
    """
    Fetch the most recent prior session's closing price from daily bars.
    Uses yfinance daily data so it correctly handles weekends and holidays.
    Returns None if data is unavailable.
    """
    try:
        df = yf.download(symbol, period="5d", interval="1d", progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        today = now_et().date()
        prior = df[df.index.date < today]
        if prior.empty:
            return None
        return float(prior["Close"].iloc[-1])
    except Exception:
        return None


def get_premarket_gap(symbol: str, open_price: float | None = None) -> dict:
    """
    Compute the pre-market gap for a symbol.

    Gap % = (today_open - prior_close) / prior_close * 100
    Classification:
      > +1.0% → "up_gap"
      < -1.0% → "down_gap"
      else    → "normal"

    If open_price is None, fetches today's first 5m bar open from yfinance.
    Returns dict: {gap_pct, gap_type, open_price, prior_close, symbol}
    """
    prior_close = get_prior_close(symbol)
    if prior_close is None or prior_close <= 0:
        return {"gap_pct": 0.0, "gap_type": "unknown", "open_price": None, "prior_close": None, "symbol": symbol}

    if open_price is None:
        try:
            df = yf.download(symbol, period="1d", interval="5m", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_convert(ET) if df.index.tzinfo else pd.to_datetime(df.index).tz_localize("UTC").tz_convert(ET)
                today = now_et().date()
                today_bars = df[df.index.date == today]
                if not today_bars.empty:
                    open_price = float(today_bars["Open"].iloc[0])
        except Exception:
            pass

    if open_price is None:
        return {"gap_pct": 0.0, "gap_type": "unknown", "open_price": None, "prior_close": round(prior_close, 4), "symbol": symbol}

    gap_pct = (open_price - prior_close) / prior_close * 100

    if gap_pct > 1.0:
        gap_type = "up_gap"
    elif gap_pct < -1.0:
        gap_type = "down_gap"
    else:
        gap_type = "normal"

    _log.info("%s gap: %.2f%% (%s) open=%.2f prior_close=%.2f", symbol, gap_pct, gap_type, open_price, prior_close)

    return {
        "gap_pct": round(gap_pct, 3),
        "gap_type": gap_type,
        "open_price": round(open_price, 4),
        "prior_close": round(prior_close, 4),
        "symbol": symbol,
    }


def regime_allows_strategy(regime: str, strategy_name: str) -> bool:
    """
    BEAR_OPEN: only strategies that explicitly support shorting are allowed.
    BollingerMomentum and SupertrendTrend have built-in short logic and
    self-filter to shorts-only when BEAR_OPEN; pass them through.
    See docs/strategies_spec.md for the full regime × strategy matrix.
    """
    bear_open_allowed = {
        "EMAMomentum",
        "OpeningGapFade",
        "VolumeSpikeReversal",
        "BollingerMomentum",   # shorts allowed; strategy returns [] for BULL_OPEN longs
        "SupertrendTrend",     # shorts allowed; strategy uses 15m ST for direction
    }
    if regime == "BEAR_OPEN" and strategy_name not in bear_open_allowed:
        return False
    return True


def apply_choppy_penalty(confidence: float, regime: str) -> float:
    if regime == "CHOPPY":
        return max(0.0, confidence - 0.2)
    return confidence
