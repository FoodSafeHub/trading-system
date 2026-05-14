"""
Intrabar signal checks on 1m bars.

Each function returns a signal dict or None.
These mirror the 5m strategy logic but operate on 1m bars so signals
are detected AS THEY FORM rather than waiting for bar close.

Key differences from 5m strategies:
  - Only the LATEST bar is checked (no loop — called every minute)
  - Volume thresholds are 1/5 of the 5m threshold (1m bar = 1/5 of a 5m bar's vol)
  - RSI uses a 5-bar window (shorter = more responsive on 1m)
  - ORB is defined as the first 15 1m bars (9:30–9:44)
"""
from __future__ import annotations

from datetime import time
from typing import Any

import pandas as pd

try:
    import ta.momentum as tam
    import ta.trend as tat
    import ta.volatility as tav
    _TA_OK = True
except ImportError:
    _TA_OK = False

from app.services.strategy.daytrading.market_open import ET, compute_vwap

_ORB_BARS_1M = 15      # first 15 1m bars = 9:30–9:44
_ORB_CUTOFF  = time(11, 30)
_VWAP_CUTOFF = time(14, 30)
_EMA_CUTOFF  = time(14, 0)
_VSR_CUTOFF  = time(15, 0)
_VSR_START   = time(9, 45)


def check_orb_intrabar(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    symbol: str,
) -> dict[str, Any] | None:
    """ORB Breakout on 1m bars. Returns signal dict or None."""
    if not _TA_OK or df_1m.empty or len(df_1m) < _ORB_BARS_1M + 3:
        return None

    today = _today(df_1m)
    if len(today) < _ORB_BARS_1M + 1:
        return None

    bar_time = today.index[-1].time()
    if bar_time >= _ORB_CUTOFF:
        return None

    # ORB range
    orb = today.iloc[:_ORB_BARS_1M]
    orb_high = float(orb["High"].max())
    orb_low  = float(orb["Low"].min())
    orb_height = orb_high - orb_low
    if orb_height <= 0:
        return None

    bar = today.iloc[-1]
    close = float(bar["Close"])
    volume = float(bar["Volume"])

    # Must close above ORB high + 0.1% buffer
    buffer = orb_high * 0.001
    if close <= orb_high + buffer:
        return None

    # ATR check
    if not _TA_OK:
        return None
    atr_val = _atr(today, 14)
    if atr_val <= 0:
        return None

    # ORB too wide
    if orb_height / atr_val > 2.5:
        return None

    # Volume must be elevated
    vol_avg = float(today["Volume"].rolling(20).mean().iloc[-1]) if len(today) >= 20 else volume
    if vol_avg > 0 and volume < vol_avg * 1.5:
        return None

    # RSI > 50
    rsi = _rsi(today, 10)
    if rsi < 50:
        return None

    stop = close - atr_val * 1.0
    target = close + orb_height * 2.0
    rr = (target - close) / (close - stop) if close > stop else 0
    if rr < 1.5:
        return None

    return {
        "direction": "BUY",
        "entry_price": round(close, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "confidence": 0.65,
        "reason": f"ORB intrabar breakout above {orb_high:.2f} at {bar_time}",
        "indicators": {
            "orb_high": round(orb_high, 4),
            "orb_low": round(orb_low, 4),
            "atr": round(atr_val, 4),
            "rsi": round(rsi, 1),
            "vol_ratio": round(volume / vol_avg, 2) if vol_avg > 0 else 0,
        },
    }


def check_vwap_reversion_intrabar(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    symbol: str,
) -> dict[str, Any] | None:
    """VWAP mean reversion on 1m bars."""
    if not _TA_OK or df_1m.empty or len(df_1m) < 20:
        return None

    today = _today(df_1m)
    if len(today) < 10:
        return None

    bar_time = today.index[-1].time()
    if bar_time >= _VWAP_CUTOFF:
        return None

    vwap = compute_vwap(today)
    bar = today.iloc[-1]
    prev = today.iloc[-2]

    close = float(bar["Close"])
    vwap_val = float(vwap.iloc[-1])
    prev_close = float(prev["Close"])

    atr_val = _atr(today, 14)
    if atr_val <= 0:
        return None

    dist = (vwap_val - close) / atr_val
    if dist < 0.3:
        return None

    # Both bars below VWAP
    if close >= vwap_val or prev_close >= vwap_val:
        return None

    # RSI oversold
    rsi = _rsi(today, 8)
    if rsi >= 42:
        return None

    # Entry bar must be green (close > open)
    if close <= float(bar["Open"]):
        return None

    stop = close - atr_val * 1.2
    target = vwap_val
    rr = (target - close) / (close - stop) if close > stop else 0
    if rr < 1.2:
        return None

    return {
        "direction": "BUY",
        "entry_price": round(close, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "confidence": 0.60,
        "reason": f"VWAP intrabar pullback {dist:.2f}×ATR below VWAP, RSI {rsi:.0f}",
        "indicators": {
            "vwap": round(vwap_val, 4),
            "atr_dist_mult": round(dist, 2),
            "rsi": round(rsi, 1),
        },
    }


def check_ema_momentum_intrabar(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    symbol: str,
) -> dict[str, Any] | None:
    """EMA momentum crossover on 1m bars."""
    if not _TA_OK or df_1m.empty or len(df_1m) < 30:
        return None

    today = _today(df_1m)
    if len(today) < 25:
        return None

    bar_time = today.index[-1].time()
    if bar_time < time(10, 0) or bar_time >= _EMA_CUTOFF:
        return None

    closes = today["Close"]
    ema9  = float(tat.EMAIndicator(closes, window=9).ema_indicator().iloc[-1])
    ema21 = float(tat.EMAIndicator(closes, window=21).ema_indicator().iloc[-1])

    prev_ema9  = float(tat.EMAIndicator(closes, window=9).ema_indicator().iloc[-2])
    prev_ema21 = float(tat.EMAIndicator(closes, window=21).ema_indicator().iloc[-2])

    bar  = today.iloc[-1]
    close = float(bar["Close"])
    atr_val = _atr(today, 14)

    # Fresh bullish crossover
    bullish_cross = prev_ema9 <= prev_ema21 and ema9 > ema21

    # Bounce setup (already bullish, low touches EMA9)
    already_bullish = ema9 > ema21
    low = float(bar["Low"])
    bounce = already_bullish and low <= ema9 + atr_val * 0.3 and close > ema9

    if not bullish_cross and not bounce:
        return None

    stop  = close - atr_val * 1.2
    target = close + atr_val * 2.5
    rr = (target - close) / (close - stop) if close > stop else 0
    if rr < 1.5:
        return None

    setup = "EMA crossover" if bullish_cross else "EMA bounce"
    return {
        "direction": "BUY",
        "entry_price": round(close, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "confidence": 0.62,
        "reason": f"EMA intrabar {setup}: EMA9={ema9:.2f} EMA21={ema21:.2f}",
        "indicators": {
            "ema9": round(ema9, 4),
            "ema21": round(ema21, 4),
            "atr": round(atr_val, 4),
            "setup": setup,
        },
    }


def check_volume_spike_intrabar(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame | None,
    symbol: str,
) -> dict[str, Any] | None:
    """Volume spike reversal on 1m bars."""
    if not _TA_OK or df_1m.empty or len(df_1m) < 25:
        return None

    today = _today(df_1m)
    if len(today) < 10:
        return None

    bar_time = today.index[-1].time()
    if bar_time < _VSR_START or bar_time >= _VSR_CUTOFF:
        return None

    bar = today.iloc[-1]
    volume = float(bar["Volume"])
    close = float(bar["Close"])
    low   = float(bar["Low"])
    high  = float(bar["High"])

    vol_avg = float(today["Volume"].rolling(20).mean().iloc[-1]) if len(today) >= 20 else volume
    if vol_avg <= 0:
        return None

    spike_mult = volume / vol_avg
    if spike_mult < 2.5:
        return None

    # Must be the largest volume bar in last 10 bars
    if volume < float(today["Volume"].iloc[-10:].max()):
        return None

    atr_val = _atr(today, 14)
    if atr_val <= 0:
        return None

    bar_range = high - low
    if bar_range < atr_val * 0.6:
        return None

    rsi = _rsi(today, 8)
    if rsi >= 42:
        return None

    stop  = low - atr_val * 0.5
    target = close + atr_val * 2.0
    rr = (target - close) / (close - stop) if close > stop else 0
    if rr < 1.5:
        return None

    return {
        "direction": "BUY",
        "entry_price": round(close, 4),
        "stop_price": round(stop, 4),
        "target_price": round(target, 4),
        "confidence": 0.63,
        "reason": f"Volume spike {spike_mult:.1f}× avg, RSI {rsi:.0f} — intrabar reversal",
        "indicators": {
            "spike_multiple": round(spike_mult, 2),
            "rsi": round(rsi, 1),
            "atr": round(atr_val, 4),
        },
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _today(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to today's bars only."""
    if df.empty:
        return df
    idx = df.index
    if idx.tzinfo is None:
        idx = idx.tz_localize(ET)
    latest_date = idx[-1].date()
    return df[idx.date == latest_date]


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if not _TA_OK or len(df) < period:
        return 0.0
    try:
        atr = tav.AverageTrueRange(df["High"], df["Low"], df["Close"], window=period)
        val = atr.average_true_range().iloc[-1]
        return float(val) if not pd.isna(val) else 0.0
    except Exception:
        return 0.0


def _rsi(df: pd.DataFrame, period: int = 14) -> float:
    if not _TA_OK or len(df) < period + 1:
        return 50.0
    try:
        rsi = tam.RSIIndicator(df["Close"], window=period).rsi().iloc[-1]
        return float(rsi) if not pd.isna(rsi) else 50.0
    except Exception:
        return 50.0
