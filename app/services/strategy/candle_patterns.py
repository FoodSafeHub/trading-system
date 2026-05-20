"""Reusable candlestick pattern detectors + structure-aware exit helpers.

Pure functions over a daily/intraday OHLCV ``pd.DataFrame`` (columns
``Open / High / Low / Close / Volume``, datetime index). No I/O, no side
effects — the same primitives drive both the Perplexity (daily) and
DayTrading (5m/15m) momentum strategies and the tiered exit manager.

Pattern set
-----------
* :func:`is_bullish_engulfing` / :func:`is_bearish_engulfing` — body of bar -1
  fully engulfs body of bar -2 in the opposite direction.
* :func:`is_hammer` / :func:`is_shooting_star` — long single-tail reversal with
  small body on the opposite side.
* :func:`is_inside_bar` / :func:`is_nr4` / :func:`is_nr7` — range compression
  setups that often precede expansion.
* :func:`is_three_bar_push` — three consecutive same-direction bars with
  expanding range and closes near the extreme (momentum thrust).

Exit helpers
------------
* :func:`chandelier_stop_long` / :func:`chandelier_stop_short` — Chuck LeBeau's
  trailing stop: ``highest_high_since_entry − N × ATR`` (long).
* :func:`structure_swing_low` / :func:`structure_swing_high` — most recent
  pivot the trade is structurally hanging off. A close beyond it means the
  setup that justified the entry is no longer intact.
* :func:`atr_series` — Wilder ATR used everywhere here so callers don't pull
  in ``ta`` for a single calculation.

Conventions
-----------
All ``is_*`` detectors accept an integer ``i`` index (default ``-1``) so the
same code works in backtests (iterate over bars) and live evaluation (most
recent close). They return plain ``bool`` and tolerate missing history by
returning ``False`` rather than raising.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── ATR (Wilder) ──────────────────────────────────────────────────────────────

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def current_atr(df: pd.DataFrame, period: int = 14) -> float:
    s = atr_series(df, period)
    if s.empty or pd.isna(s.iloc[-1]):
        # Fall back to 2% of close — never let a strategy size off NaN.
        return float(df["Close"].iloc[-1]) * 0.02 if not df.empty else 0.0
    return float(s.iloc[-1])


# ── Bar geometry helpers ──────────────────────────────────────────────────────

def _body(bar) -> float:
    return abs(float(bar["Close"]) - float(bar["Open"]))


def _range(bar) -> float:
    return float(bar["High"]) - float(bar["Low"])


def _upper_wick(bar) -> float:
    return float(bar["High"]) - max(float(bar["Open"]), float(bar["Close"]))


def _lower_wick(bar) -> float:
    return min(float(bar["Open"]), float(bar["Close"])) - float(bar["Low"])


def _is_green(bar) -> bool:
    return float(bar["Close"]) > float(bar["Open"])


def _is_red(bar) -> bool:
    return float(bar["Close"]) < float(bar["Open"])


# ── Engulfing ─────────────────────────────────────────────────────────────────

def is_bullish_engulfing(df: pd.DataFrame, i: int = -1, *, min_body_ratio: float = 1.1) -> bool:
    """Bar ``i`` is a green bar whose body fully engulfs bar ``i-1``'s red body.

    ``min_body_ratio`` lets us require the engulfing body be ~10% larger than
    the body being engulfed — guards against two tiny dojis being called a
    pattern.
    """
    if len(df) < abs(i) + 1:
        return False
    cur, prev = df.iloc[i], df.iloc[i - 1]
    if not (_is_red(prev) and _is_green(cur)):
        return False
    if _body(prev) <= 0 or _body(cur) < _body(prev) * min_body_ratio:
        return False
    # Body of current must cover the body of previous.
    return float(cur["Open"]) <= float(prev["Close"]) and float(cur["Close"]) >= float(prev["Open"])


def is_bearish_engulfing(df: pd.DataFrame, i: int = -1, *, min_body_ratio: float = 1.1) -> bool:
    if len(df) < abs(i) + 1:
        return False
    cur, prev = df.iloc[i], df.iloc[i - 1]
    if not (_is_green(prev) and _is_red(cur)):
        return False
    if _body(prev) <= 0 or _body(cur) < _body(prev) * min_body_ratio:
        return False
    return float(cur["Open"]) >= float(prev["Close"]) and float(cur["Close"]) <= float(prev["Open"])


# ── Hammer / Shooting Star ────────────────────────────────────────────────────

def is_hammer(df: pd.DataFrame, i: int = -1, *, tail_to_body: float = 2.0, max_upper_wick_pct: float = 0.2) -> bool:
    """Classic hammer: small body at top, lower wick >= ``tail_to_body`` × body,
    minimal upper wick. Must come at the *bottom* of a downtrend — the caller
    is expected to gate on context (oversold RSI, prior down-bars, etc.).
    """
    if len(df) < abs(i) + 1:
        return False
    bar = df.iloc[i]
    body = _body(bar)
    rng = _range(bar)
    if rng <= 0 or body <= 0:
        return False
    lower = _lower_wick(bar)
    upper = _upper_wick(bar)
    if lower < tail_to_body * body:
        return False
    if upper > max_upper_wick_pct * rng:
        return False
    return True


def is_shooting_star(df: pd.DataFrame, i: int = -1, *, tail_to_body: float = 2.0, max_lower_wick_pct: float = 0.2) -> bool:
    if len(df) < abs(i) + 1:
        return False
    bar = df.iloc[i]
    body = _body(bar)
    rng = _range(bar)
    if rng <= 0 or body <= 0:
        return False
    upper = _upper_wick(bar)
    lower = _lower_wick(bar)
    if upper < tail_to_body * body:
        return False
    if lower > max_lower_wick_pct * rng:
        return False
    return True


# ── Inside bar / NR4 / NR7 ────────────────────────────────────────────────────

def is_inside_bar(df: pd.DataFrame, i: int = -1) -> bool:
    """Bar ``i``'s entire range is inside bar ``i-1``'s range."""
    if len(df) < abs(i) + 1:
        return False
    cur, prev = df.iloc[i], df.iloc[i - 1]
    return float(cur["High"]) <= float(prev["High"]) and float(cur["Low"]) >= float(prev["Low"])


def is_nr4(df: pd.DataFrame, i: int = -1) -> bool:
    """Bar ``i`` has the smallest range of the last 4 bars (incl. itself)."""
    if len(df) < 4:
        return False
    window = df.iloc[i - 3 : i + 1] if i != -1 else df.iloc[-4:]
    if len(window) < 4:
        return False
    ranges = window["High"] - window["Low"]
    return float(ranges.iloc[-1]) == float(ranges.min()) and len(ranges.unique()) > 1


def is_nr7(df: pd.DataFrame, i: int = -1) -> bool:
    if len(df) < 7:
        return False
    window = df.iloc[i - 6 : i + 1] if i != -1 else df.iloc[-7:]
    if len(window) < 7:
        return False
    ranges = window["High"] - window["Low"]
    return float(ranges.iloc[-1]) == float(ranges.min()) and len(ranges.unique()) > 1


# ── 3-bar momentum push ───────────────────────────────────────────────────────

def is_three_bar_push_up(df: pd.DataFrame, i: int = -1, *, min_close_pct: float = 0.6) -> bool:
    """Three consecutive higher closes with expanding ranges, each closing in
    the upper ``min_close_pct`` of its own range (closes near highs = sellers
    failed to push back).
    """
    if len(df) < abs(i) + 3:
        return False
    bars = [df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]]
    closes = [float(b["Close"]) for b in bars]
    if not (closes[0] < closes[1] < closes[2]):
        return False
    ranges = [_range(b) for b in bars]
    if not all(r > 0 for r in ranges):
        return False
    if not (ranges[1] >= ranges[0] * 0.9 and ranges[2] >= ranges[1] * 0.9):
        return False  # ranges should be holding or expanding
    for b, r in zip(bars, ranges):
        close = float(b["Close"]); low = float(b["Low"])
        if (close - low) / r < min_close_pct:
            return False
    return True


def is_three_bar_push_down(df: pd.DataFrame, i: int = -1, *, min_close_pct: float = 0.6) -> bool:
    if len(df) < abs(i) + 3:
        return False
    bars = [df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]]
    closes = [float(b["Close"]) for b in bars]
    if not (closes[0] > closes[1] > closes[2]):
        return False
    ranges = [_range(b) for b in bars]
    if not all(r > 0 for r in ranges):
        return False
    if not (ranges[1] >= ranges[0] * 0.9 and ranges[2] >= ranges[1] * 0.9):
        return False
    for b, r in zip(bars, ranges):
        close = float(b["Close"]); high = float(b["High"])
        if (high - close) / r < min_close_pct:
            return False
    return True


# ── Volume helpers ────────────────────────────────────────────────────────────

def volume_surge_ratio(df: pd.DataFrame, i: int = -1, lookback: int = 20) -> float:
    """Returns volume[i] / mean(volume[i-lookback:i]). 0.0 if not enough history."""
    if len(df) < lookback + abs(i):
        return 0.0
    avg = float(df["Volume"].iloc[i - lookback : i].mean()) if i != -1 else float(df["Volume"].iloc[-lookback - 1 : -1].mean())
    if avg <= 0:
        return 0.0
    cur = float(df["Volume"].iloc[i])
    return cur / avg


# ── Swing pivots (structure) ──────────────────────────────────────────────────

def structure_swing_low(df: pd.DataFrame, lookback: int = 10, left: int = 2, right: int = 2) -> Optional[float]:
    """Most recent confirmed swing low in the last ``lookback`` bars.

    A bar is a swing low if its low is the minimum of the surrounding
    ``left + right`` bars. ``right`` requires forward bars, so the most
    recent ``right`` bars are never themselves pivots — we walk backwards
    looking for the last confirmed pivot.
    """
    if len(df) < lookback + left + right:
        return None
    lows = df["Low"].to_numpy()
    n = len(lows)
    start = max(left, n - lookback)
    end = n - right
    pivot: Optional[float] = None
    for idx in range(start, end):
        window = lows[idx - left : idx + right + 1]
        if lows[idx] == window.min():
            pivot = float(lows[idx])
    return pivot


def structure_swing_high(df: pd.DataFrame, lookback: int = 10, left: int = 2, right: int = 2) -> Optional[float]:
    if len(df) < lookback + left + right:
        return None
    highs = df["High"].to_numpy()
    n = len(highs)
    start = max(left, n - lookback)
    end = n - right
    pivot: Optional[float] = None
    for idx in range(start, end):
        window = highs[idx - left : idx + right + 1]
        if highs[idx] == window.max():
            pivot = float(highs[idx])
    return pivot


# ── Chandelier trailing stop ──────────────────────────────────────────────────

@dataclass
class ChandelierState:
    """Persisted between bars by the exit manager."""
    highest_high: float
    lowest_low: float
    atr_mult: float = 3.0
    atr_period: int = 14


def chandelier_stop_long(df: pd.DataFrame, *, atr_mult: float = 3.0, atr_period: int = 14, since: int = 0) -> float:
    """Long Chandelier exit: ``max(High[since:]) − atr_mult × ATR``.

    ``since`` is a positive integer offset from the start of the trade. The
    caller passes the number of bars elapsed since entry; we look at the
    highest high in that window so the stop only ratchets up.
    """
    if df.empty:
        return 0.0
    window = df.iloc[-max(since, 1):]
    highest = float(window["High"].max())
    atr = current_atr(df, atr_period)
    return highest - atr_mult * atr


def chandelier_stop_short(df: pd.DataFrame, *, atr_mult: float = 3.0, atr_period: int = 14, since: int = 0) -> float:
    if df.empty:
        return 0.0
    window = df.iloc[-max(since, 1):]
    lowest = float(window["Low"].min())
    atr = current_atr(df, atr_period)
    return lowest + atr_mult * atr


# ── Composite: structure-aware exit decision ──────────────────────────────────

@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    stop_price: Optional[float] = None


def evaluate_tiered_exit_long(
    df: pd.DataFrame,
    *,
    entry_price: float,
    bars_since_entry: int,
    atr_mult: float = 3.0,
    atr_period: int = 14,
    structure_lookback: int = 20,
) -> ExitDecision:
    """Combined Chandelier + structure-break exit for a long position.

    Exits when EITHER of these trips on the most recent bar's *close*:
      1. Close < chandelier_stop_long (trailing volatility stop)
      2. Close < most recent confirmed swing low (structure break — the setup
         that justified the trade is broken)

    Returns ``ExitDecision(should_exit=False)`` if neither trips, otherwise the
    reason and the active stop price. Never exits before at least 2 bars of
    holding time so a fresh trade isn't stopped by its own entry bar's wick.
    """
    if df.empty or bars_since_entry < 2:
        return ExitDecision(False, "")

    close = float(df["Close"].iloc[-1])
    chand = chandelier_stop_long(df, atr_mult=atr_mult, atr_period=atr_period, since=bars_since_entry)
    # Never let the trailing stop be wider than entry — once the trade is open
    # we accept giving back some open profit but not increasing initial risk.
    chand = max(chand, entry_price - atr_mult * current_atr(df, atr_period))

    if close < chand:
        return ExitDecision(True, f"Chandelier stop hit: close {close:.2f} < {chand:.2f}", chand)

    swing = structure_swing_low(df, lookback=structure_lookback)
    if swing is not None and close < swing:
        return ExitDecision(True, f"Structure break: close {close:.2f} < swing low {swing:.2f}", swing)

    return ExitDecision(False, "", chand)


def evaluate_tiered_exit_short(
    df: pd.DataFrame,
    *,
    entry_price: float,
    bars_since_entry: int,
    atr_mult: float = 3.0,
    atr_period: int = 14,
    structure_lookback: int = 20,
) -> ExitDecision:
    if df.empty or bars_since_entry < 2:
        return ExitDecision(False, "")

    close = float(df["Close"].iloc[-1])
    chand = chandelier_stop_short(df, atr_mult=atr_mult, atr_period=atr_period, since=bars_since_entry)
    chand = min(chand, entry_price + atr_mult * current_atr(df, atr_period))

    if close > chand:
        return ExitDecision(True, f"Chandelier stop hit: close {close:.2f} > {chand:.2f}", chand)

    swing = structure_swing_high(df, lookback=structure_lookback)
    if swing is not None and close > swing:
        return ExitDecision(True, f"Structure break: close {close:.2f} > swing high {swing:.2f}", swing)

    return ExitDecision(False, "", chand)


__all__ = [
    "atr_series",
    "current_atr",
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_hammer",
    "is_shooting_star",
    "is_inside_bar",
    "is_nr4",
    "is_nr7",
    "is_three_bar_push_up",
    "is_three_bar_push_down",
    "volume_surge_ratio",
    "structure_swing_low",
    "structure_swing_high",
    "chandelier_stop_long",
    "chandelier_stop_short",
    "ChandelierState",
    "ExitDecision",
    "evaluate_tiered_exit_long",
    "evaluate_tiered_exit_short",
]
