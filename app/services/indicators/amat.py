from __future__ import annotations

"""
Adaptive Momentum Acceleration Trend (AMAT).

Custom composite indicator combining:
  1. Trend Spine   — ATR-based adaptive trailing support/resistance line
                     (AlphaTrend-style ratchet, isolated here as its own function).
  2. Acceleration  — second derivative of RSI/MFI momentum over an N-bar step,
                     min-max normalized to [-100, +100] over a rolling window.
  3. Conviction    — relative-volume multiplier clipped to [0.5, 2.0].
  4. Divergence    — price new-high/low without a matching weighted-acceleration
                     extreme applies a penalty (default x0.7) to cut false confirmations.
  5. AMAT Score    — penalized weighted acceleration rescaled to [0, 100] over a
                     rolling window.

Signals: BUY when score crosses above buy_threshold AND close > TrendSpine;
         SELL when score crosses below sell_threshold AND close < TrendSpine.

All series (score, spine, weighted acceleration, divergence flags, signals) are
returned per-bar so they can be plotted/logged for backtesting.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AMATParams:
    """All tunables for AMAT. Defaults match the reference spec."""
    atr_period: int = 14            # SMA(True Range) length for the Trend Spine
    multiplier: float = 1.0         # ATR multiplier for the spine bands
    momentum_period: int = 14       # RSI/MFI period
    accel_step: int = 3             # bar step for the second derivative (3/6 by default)
    accel_norm_window: int = 50     # rolling window for accel min-max scaling
    rel_volume_period: int = 20     # SMA(volume) length for relative volume
    conviction_min: float = 0.5     # RelVolume clip floor
    conviction_max: float = 2.0     # RelVolume clip ceiling
    divergence_lookback: int = 10   # N-bar high/low window for divergence detection
    divergence_penalty: float = 0.7 # multiplier applied on divergent bars
    score_window: int = 100         # rolling window for the 0-100 score rescale
    buy_threshold: float = 65.0     # score cross-above level for BUY
    sell_threshold: float = 35.0    # score cross-below level for SELL


@dataclass
class AMATResult:
    trend_spine: pd.Series           # adaptive support/resistance line
    momentum: pd.Series              # raw RSI or MFI series
    acceleration: pd.Series          # normalized acceleration, [-100, +100]
    conviction: pd.Series            # relative-volume multiplier, [0.5, 2.0]
    weighted_acceleration: pd.Series # acceleration * conviction (pre-penalty)
    divergence: pd.Series            # bool — penalty applied on this bar
    score: pd.Series                 # composite AMAT score, [0, 100]
    signal: pd.Series                # "BUY" / "SELL" / "HOLD" per bar
    momentum_source: str = "rsi"     # "rsi" or "mfi"
    params: AMATParams = field(default_factory=AMATParams)

    @property
    def latest_score(self) -> Optional[float]:
        valid = self.score.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None

    @property
    def latest_spine(self) -> Optional[float]:
        valid = self.trend_spine.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None


# ──────────────────────────────────────────────────────────────────────────────
# 1. Trend Spine
# ──────────────────────────────────────────────────────────────────────────────

def compute_trend_spine(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    multiplier: float = 1.0,
) -> pd.Series:
    """ATR-based adaptive trailing line.

    ATR = SMA(True Range, atr_period); upT = low - ATR*mult; downT = high + ATR*mult.
    Spine ratchets: if close > spine[t-1] take max(upT, spine[t-1]),
    else min(downT, spine[t-1]).
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    up_t = (low - atr * multiplier).to_numpy(dtype=float)
    down_t = (high + atr * multiplier).to_numpy(dtype=float)
    close_v = close.to_numpy(dtype=float)

    spine = np.full(len(close_v), np.nan)
    prev = np.nan
    for i in range(len(close_v)):
        if np.isnan(up_t[i]) or np.isnan(down_t[i]):
            continue  # ATR warm-up
        if np.isnan(prev):
            prev = up_t[i]
        elif close_v[i] > prev:
            prev = max(up_t[i], prev)
        else:
            prev = min(down_t[i], prev)
        spine[i] = prev
    return pd.Series(spine, index=close.index)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Momentum + acceleration
# ──────────────────────────────────────────────────────────────────────────────

def _rsi_series(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0).where(~close.isna())


def _mfi_series(high: pd.Series, low: pd.Series, close: pd.Series,
                volume: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    raw_flow = tp * volume
    up = raw_flow.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    down = raw_flow.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    ratio = up / down.replace(0, np.nan)
    mfi = 100 - 100 / (1 + ratio)
    return mfi.fillna(50.0).where(~close.isna())


def _minmax_scale(series: pd.Series, window: int, out_min: float, out_max: float) -> pd.Series:
    """Rolling min-max rescale; zero-range windows map to the output midpoint."""
    roll_min = series.rolling(window, min_periods=max(2, window // 5)).min()
    roll_max = series.rolling(window, min_periods=max(2, window // 5)).max()
    rng = roll_max - roll_min
    mid = (out_min + out_max) / 2.0
    scaled = (series - roll_min) / rng * (out_max - out_min) + out_min
    return scaled.where(rng > 0, mid)


# ──────────────────────────────────────────────────────────────────────────────
# Composite
# ──────────────────────────────────────────────────────────────────────────────

def compute_amat(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None = None,
    params: AMATParams | None = None,
    log_context: str = "",
) -> AMATResult:
    """Full AMAT computation. `volume=None` (or all-zero/NaN volume) falls back
    to RSI momentum and a neutral conviction multiplier of 1.0.

    `log_context` (e.g. the symbol) is prefixed to divergence-penalty log lines.
    """
    p = params or AMATParams()
    step = p.accel_step

    spine = compute_trend_spine(high, low, close, p.atr_period, p.multiplier)

    has_volume = (
        volume is not None
        and volume.notna().any()
        and float(volume.fillna(0).abs().sum()) > 0
    )
    if has_volume:
        momentum = _mfi_series(high, low, close, volume.fillna(0), p.momentum_period)
        source = "mfi"
    else:
        momentum = _rsi_series(close, p.momentum_period)
        source = "rsi"

    # Second derivative: (M[t] - M[t-step]) - (M[t-step] - M[t-2*step])
    raw_accel = (momentum - momentum.shift(step)) - (momentum.shift(step) - momentum.shift(2 * step))
    acceleration = _minmax_scale(raw_accel, p.accel_norm_window, -100.0, 100.0)

    if has_volume:
        vol_sma = volume.rolling(p.rel_volume_period).mean()
        rel_vol = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)
        conviction = rel_vol.clip(p.conviction_min, p.conviction_max)
    else:
        conviction = pd.Series(1.0, index=close.index)

    weighted = acceleration * conviction

    # Divergence: price prints a new N-bar high (low) while weighted acceleration
    # does NOT print a corresponding N-bar high (low).
    n = p.divergence_lookback
    price_new_high = close >= close.rolling(n).max()
    price_new_low = close <= close.rolling(n).min()
    accel_new_high = weighted >= weighted.rolling(n).max()
    accel_new_low = weighted <= weighted.rolling(n).min()
    divergence = ((price_new_high & ~accel_new_high) | (price_new_low & ~accel_new_low)).fillna(False)

    penalized = weighted.where(~divergence, weighted * p.divergence_penalty)

    if bool(divergence.iloc[-1]) if len(divergence) else False:
        kind = "bearish (new price high, no accel high)" if bool(price_new_high.iloc[-1]) \
            else "bullish (new price low, no accel low)"
        logger.info(
            "AMAT divergence penalty applied%s: %s — weighted accel %.2f -> %.2f (x%.2f)",
            f" [{log_context}]" if log_context else "",
            kind, float(weighted.iloc[-1]), float(penalized.iloc[-1]), p.divergence_penalty,
        )

    score = _minmax_scale(penalized, p.score_window, 0.0, 100.0)

    # Signal series: threshold cross + spine confirmation
    prev_score = score.shift(1)
    buy = (prev_score <= p.buy_threshold) & (score > p.buy_threshold) & (close > spine)
    sell = (prev_score >= p.sell_threshold) & (score < p.sell_threshold) & (close < spine)
    signal = pd.Series("HOLD", index=close.index)
    signal[buy.fillna(False)] = "BUY"
    signal[sell.fillna(False)] = "SELL"

    return AMATResult(
        trend_spine=spine,
        momentum=momentum,
        acceleration=acceleration,
        conviction=conviction,
        weighted_acceleration=weighted,
        divergence=divergence,
        score=score,
        signal=signal,
        momentum_source=source,
        params=p,
    )
