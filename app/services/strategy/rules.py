from __future__ import annotations

"""
Strategy rule implementations.
Each rule takes a price Series and parameters, returns a StrategySignal.

Rules are trend-aware: all use SMA(200) as an uptrend filter so they only
issue BUY signals when the broader trend is up.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.ema import compute_ema, ema_crossover_signal
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma, sma_crossover_signal
from app.services.indicators.supertrend import compute_supertrend
from app.services.strategy.models import StrategySignal

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """Open-position context the caller threads in so a stateless rule can run
    a trailing-stop overlay. entry_price = our fill; highest_close = the highest
    close seen since entry (caller maintains it bar-by-bar)."""
    entry_price: float
    highest_close: float
    bars_held: int = 0


def _apply_chandelier_overlay(
    signal: StrategySignal,
    prices: pd.Series,
    ohlcv: Optional[pd.DataFrame],
    params: Dict[str, Any],
    position: Optional[PositionState],
) -> StrategySignal:
    """ATR-Chandelier trailing-stop overlay on the exit decision.

    Opt-in via params['trail_enabled']. Default OFF -> returns signal unchanged,
    so any symbol without the param keeps its current fixed-band exit exactly.

    Once an open position is up >= trail_trigger_pct, we stop honouring the
    rule's fixed-band SELL and instead ride the trend until the close breaks
    below the Chandelier line (highest_close_since_entry - atr_mult*ATR). Below
    the trigger (failed bounce / loser), the rule's SELL is honoured as-is, so
    hard stops and quick mean-reversion exits are untouched. Entries are never
    affected — overlay only runs when a position is open.
    """
    if not params.get("trail_enabled") or position is None:
        return signal

    trigger_pct = float(params.get("trail_trigger_pct", 3.0))
    atr_mult    = float(params.get("atr_trail_mult", 3.0))
    atr_period  = int(params.get("atr_trail_period", 22))

    if ohlcv is None or "High" not in ohlcv.columns or len(ohlcv) < atr_period + 1:
        return signal  # can't compute ATR -> leave the rule's decision alone

    c_now = float(prices.iloc[-1])
    if position.entry_price <= 0:
        return signal
    unreal_pct = (c_now - position.entry_price) / position.entry_price * 100.0
    if unreal_pct < trigger_pct:
        return signal  # not yet in profit-runway zone — honour the rule's exit

    atr_v = float(_atr_raw(ohlcv, atr_period).iloc[-1])
    if pd.isna(atr_v) or atr_v <= 0:
        return signal
    chandelier = position.highest_close - atr_mult * atr_v

    if c_now <= chandelier:
        # Trend broke -> force the exit even if the rule said HOLD.
        return StrategySignal(
            symbol=signal.symbol, direction="SELL", strength=signal.strength,
            price_at_signal=c_now, indicators={**signal.indicators, "trail_exit": True,
                                               "chandelier": round(chandelier, 2)},
            strategy_name=signal.strategy_name,
        )
    if signal.direction == "SELL":
        # Still above the trail -> suppress the fixed-band SELL, ride the winner.
        return StrategySignal(
            symbol=signal.symbol, direction="HOLD", strength=signal.strength,
            price_at_signal=c_now, indicators={**signal.indicators, "trail_hold": True,
                                               "chandelier": round(chandelier, 2)},
            strategy_name=signal.strategy_name,
        )
    return signal


def _above_sma200(prices: pd.Series) -> bool:
    """True when the latest close is above the 200-bar SMA (uptrend filter)."""
    if len(prices) < 200:
        return False
    sma200 = prices.rolling(200).mean().iloc[-1]
    return float(prices.iloc[-1]) > float(sma200)


def rule_sma_rsi(symbol: str, prices: pd.Series, params: Dict[str, Any], **_) -> StrategySignal:
    """
    BUY  when fast SMA is above slow SMA AND RSI is in momentum zone (45-70)
         AND price is above SMA(200) — trend-following entry.
    SELL when fast SMA crosses below slow SMA OR RSI > overbought threshold.

    Fixed: old version required SMA crossover UP + RSI < 30 simultaneously —
    physically contradictory conditions that never fired a BUY in 2 years.
    """
    fast = params.get("sma_fast", 10)
    slow = params.get("sma_slow", 30)
    rsi_period = params.get("rsi_period", 14)
    rsi_momentum_low = params.get("rsi_momentum_low", 45)
    overbought = params.get("rsi_overbought", 75)

    fast_sma_vals = compute_sma(prices, fast).values
    slow_sma_vals = compute_sma(prices, slow).values
    rsi_result = compute_rsi(prices, rsi_period)
    rsi_now = rsi_result.latest

    fast_now  = float(fast_sma_vals.iloc[-1])
    slow_now  = float(slow_sma_vals.iloc[-1])
    fast_prev = float(fast_sma_vals.iloc[-2])
    slow_prev = float(slow_sma_vals.iloc[-2])

    uptrend = _above_sma200(prices)
    fast_above_slow = fast_now > slow_now
    bearish_cross = fast_prev >= slow_prev and fast_now < slow_now

    indicators = {
        "sma_fast": round(fast_now, 2),
        "sma_slow": round(slow_now, 2),
        "rsi": round(rsi_now, 1) if rsi_now else None,
        "above_sma200": uptrend,
    }

    direction = "HOLD"

    # SELL: fast SMA crosses below slow, or RSI overbought
    if bearish_cross or (rsi_now is not None and rsi_now > overbought):
        direction = "SELL"
    # BUY: uptrend, fast SMA above slow, RSI in healthy momentum zone
    elif uptrend and fast_above_slow and rsi_now is not None and rsi_momentum_low <= rsi_now <= overbought:
        direction = "BUY"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        strength=round(min(rsi_now / 100, 1.0), 2) if rsi_now else 0.5,
        price_at_signal=float(prices.iloc[-1]) if not prices.empty else None,
        indicators=indicators,
        strategy_name="sma_rsi",
    )


def rule_ema_crossover(symbol: str, prices: pd.Series, params: Dict[str, Any], **_) -> StrategySignal:
    """
    EMA crossover with trend filter.
    BUY  when fast EMA crosses above slow EMA AND price > SMA(200).
    SELL when fast EMA crosses below slow EMA.
    """
    fast = params.get("ema_fast", 9)
    slow = params.get("ema_slow", 21)

    raw_signal = ema_crossover_signal(prices, fast, slow)
    fast_ema = compute_ema(prices, fast).latest
    slow_ema = compute_ema(prices, slow).latest
    uptrend = _above_sma200(prices)

    direction = raw_signal
    if raw_signal == "BUY" and not uptrend:
        direction = "HOLD"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=float(prices.iloc[-1]) if not prices.empty else None,
        indicators={"ema_fast": round(fast_ema, 2) if fast_ema else None,
                    "ema_slow": round(slow_ema, 2) if slow_ema else None,
                    "above_sma200": uptrend},
        strategy_name="ema_crossover",
    )


def rule_macd(symbol: str, prices: pd.Series, params: Dict[str, Any], **_) -> StrategySignal:
    """
    MACD crossover with RSI confirmation and trend filter.
    BUY  on bullish MACD crossover AND RSI > 45 AND price > SMA(200).
    SELL on bearish MACD crossover OR RSI > 75.
    """
    fast = params.get("macd_fast", 12)
    slow = params.get("macd_slow", 26)
    signal_period = params.get("macd_signal", 9)

    result = compute_macd(prices, fast, slow, signal_period)
    rsi_now = compute_rsi(prices, 14).latest
    uptrend = _above_sma200(prices)
    raw_signal = result.crossover_signal()

    direction = "HOLD"
    if raw_signal == "BUY" and uptrend and rsi_now is not None and rsi_now > 45:
        direction = "BUY"
    elif raw_signal == "SELL" or (rsi_now is not None and rsi_now > 75):
        direction = "SELL"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=float(prices.iloc[-1]) if not prices.empty else None,
        indicators={
            "macd": round(result.latest_macd, 4) if result.latest_macd else None,
            "signal": round(result.latest_signal, 4) if result.latest_signal else None,
            "histogram": round(result.latest_histogram, 4) if result.latest_histogram else None,
            "rsi": round(rsi_now, 1) if rsi_now else None,
            "above_sma200": uptrend,
        },
        strategy_name="macd",
    )


def rule_bollinger(symbol: str, prices: pd.Series, params: Dict[str, Any], **_) -> StrategySignal:
    """
    Bollinger Band mean-reversion WITH trend filter and RSI confirmation.

    BUY  when price closes back above lower band after touching it (reversion),
         RSI in rsi_low–rsi_high range, AND price > SMA(200).
    SELL when price reaches upper band AND RSI > 70 (confirmed overbought).
    """
    period = params.get("bb_period", 20)
    std_dev = params.get("bb_std", 2.0)
    rsi_buy_low  = params.get("rsi_low", 30)
    rsi_buy_high = params.get("rsi_high", 55)

    result = compute_bollinger(prices, period, std_dev)
    current_price = float(prices.iloc[-1])
    prev_price = float(prices.iloc[-2]) if len(prices) >= 2 else current_price

    lower_now  = float(result.lower.values.iloc[-1])
    lower_prev = float(result.lower.values.iloc[-2]) if len(prices) >= 2 else lower_now
    upper_now  = float(result.upper.values.iloc[-1])
    middle_now = float(result.middle.values.iloc[-1])

    rsi_now = compute_rsi(prices, 14).latest
    uptrend = _above_sma200(prices)

    # Reversion signal: prev bar closed below lower band, current closed back above
    reverted = (prev_price < lower_prev) and (current_price > lower_now)

    direction = "HOLD"

    # SELL: price at upper band AND RSI confirms overbought
    if current_price >= upper_now and rsi_now is not None and rsi_now > 70:
        direction = "SELL"
    # SELL: price drops back below middle when not in uptrend (trend reversal)
    elif prev_price >= middle_now and current_price < middle_now and not uptrend:
        direction = "SELL"
    # BUY: lower-band reversion in uptrend with RSI in configured range
    elif reverted and uptrend and rsi_now is not None and rsi_buy_low <= rsi_now <= rsi_buy_high:
        direction = "BUY"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=current_price,
        indicators={
            "bb_upper": round(upper_now, 2),
            "bb_middle": round(middle_now, 2),
            "bb_lower": round(lower_now, 2),
            "bb_pct_b": round(result.percent_b.latest, 3) if result.percent_b.latest else None,
            "rsi": round(rsi_now, 1) if rsi_now else None,
            "above_sma200": uptrend,
        },
        strategy_name="bollinger",
    )


def rule_supertrend(symbol: str, prices: pd.Series, params: Dict[str, Any], ohlcv: pd.DataFrame | None = None, **_) -> StrategySignal:
    """
    Supertrend trend-following strategy.
    BUY  when Supertrend direction is bullish (1) and RSI > 40 (trend confirmed).
    SELL when Supertrend direction flips bearish (-1).
    Holds position while in bullish direction rather than waiting for re-flip.
    """
    period = params.get("st_period", 10)
    multiplier = params.get("st_multiplier", 3.0)

    if len(prices) < max(period + 5, 50):
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="supertrend")

    if ohlcv is not None and "High" in ohlcv.columns and "Low" in ohlcv.columns:
        high_proxy = ohlcv["High"].reindex(prices.index).fillna(prices)
        low_proxy  = ohlcv["Low"].reindex(prices.index).fillna(prices)
    else:
        daily_move = prices.diff().abs()
        atr_proxy = daily_move.ewm(span=period, adjust=False).mean().fillna(daily_move.mean())
        high_proxy = prices + atr_proxy
        low_proxy  = prices - atr_proxy

    result = compute_supertrend(high_proxy, low_proxy, prices, period=period, multiplier=multiplier)

    dir_now  = int(result.direction.iloc[-1]) if not pd.isna(result.direction.iloc[-1]) else 0
    dir_prev = int(result.direction.iloc[-2]) if len(result.direction) >= 2 and not pd.isna(result.direction.iloc[-2]) else dir_now
    st_val   = float(result.values.iloc[-1]) if not pd.isna(result.values.iloc[-1]) else None
    rsi_now  = compute_rsi(prices, 14).latest

    direction = "HOLD"
    # BUY on flip to bullish or first bullish bar with RSI confirmation
    if dir_now == 1 and dir_prev == -1:
        direction = "BUY"
    # SELL on flip to bearish
    elif dir_now == -1 and dir_prev == 1:
        direction = "SELL"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=float(prices.iloc[-1]),
        indicators={
            "st_direction": dir_now,
            "st_value": round(st_val, 2) if st_val else None,
            "rsi": round(rsi_now, 1) if rsi_now else None,
        },
        strategy_name="supertrend",
    )


def rule_vwap_rsi(symbol: str, prices: pd.Series, params: Dict[str, Any], **_) -> StrategySignal:
    """
    VWAP + RSI momentum/mean-reversion strategy.
    BUY  when RSI rises from oversold (crosses above 35) AND price is within 3% of VWAP proxy.
    SELL when RSI drops from overbought (crosses below 65) OR price >4% above VWAP.
    No SMA200 filter — works in both uptrends and consolidations.
    """
    vwap_period = params.get("vwap_period", 20)
    rsi_period  = params.get("rsi_period", 14)
    rsi_oversold = params.get("rsi_oversold", 35)
    rsi_overbought = params.get("rsi_overbought", 65)

    if len(prices) < max(vwap_period, rsi_period) + 5:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="vwap_rsi")

    vwap_proxy = prices.ewm(span=vwap_period, adjust=False).mean()
    vwap_now   = float(vwap_proxy.iloc[-1])
    price_now  = float(prices.iloc[-1])

    rsi_now  = compute_rsi(prices, rsi_period).latest
    rsi_prev = compute_rsi(prices.iloc[:-1], rsi_period).latest if len(prices) >= rsi_period + 2 else (rsi_now or 50)

    vwap_pct = (price_now - vwap_now) / vwap_now * 100 if vwap_now else 0

    direction = "HOLD"
    if rsi_now is not None and rsi_prev is not None:
        # SELL: RSI crosses below overbought, or price far extended above VWAP
        if (rsi_prev >= rsi_overbought and rsi_now < rsi_overbought) or vwap_pct > 4.0:
            direction = "SELL"
        # BUY: RSI crosses above oversold threshold and price near VWAP (not overextended)
        elif rsi_prev < rsi_oversold and rsi_now >= rsi_oversold and -3.0 <= vwap_pct <= 2.0:
            direction = "BUY"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=price_now,
        indicators={
            "vwap_proxy": round(vwap_now, 2),
            "vwap_pct": round(vwap_pct, 2),
            "rsi": round(rsi_now, 1) if rsi_now else None,
            "rsi_prev": round(rsi_prev, 1) if rsi_prev else None,
        },
        strategy_name="vwap_rsi",
    )


def rule_ema_ribbon(symbol: str, prices: pd.Series, params: Dict[str, Any], **_) -> StrategySignal:
    """
    EMA Ribbon trend-following strategy.
    Uses 3 EMAs (fast/mid/slow). All aligned (fast>mid>slow) = strong uptrend.
    BUY  when fast EMA crosses above mid AND mid > slow AND price > SMA(200) AND RSI 45-70.
    SELL when fast EMA crosses below mid OR RSI > 75.
    """
    ema_fast   = params.get("ema_fast", 8)
    ema_mid    = params.get("ema_mid", 21)
    ema_slow   = params.get("ema_slow", 50)
    rsi_period = params.get("rsi_period", 14)
    rsi_low    = params.get("rsi_low", 40)
    rsi_high   = params.get("rsi_high", 78)

    if len(prices) < ema_slow + 5:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="ema_ribbon")

    fast_ema = prices.ewm(span=ema_fast, adjust=False).mean()
    mid_ema  = prices.ewm(span=ema_mid, adjust=False).mean()
    slow_ema = prices.ewm(span=ema_slow, adjust=False).mean()

    fast_now  = float(fast_ema.iloc[-1])
    fast_prev = float(fast_ema.iloc[-2])
    mid_now   = float(mid_ema.iloc[-1])
    mid_prev  = float(mid_ema.iloc[-2])
    slow_now  = float(slow_ema.iloc[-1])

    rsi_now = compute_rsi(prices, rsi_period).latest
    uptrend = _above_sma200(prices)

    fast_crossed_above = fast_prev <= mid_prev and fast_now > mid_now
    fast_crossed_below = fast_prev >= mid_prev and fast_now < mid_now
    ribbon_aligned = fast_now > mid_now > slow_now

    direction = "HOLD"
    # SELL: fast crosses below mid, or RSI above configured ceiling
    if fast_crossed_below or (rsi_now is not None and rsi_now > rsi_high):
        direction = "SELL"
    # BUY: fast crosses above mid while ribbon aligned; RSI in configured range
    elif (fast_crossed_above and mid_now > slow_now
          and rsi_now is not None and rsi_low <= rsi_now <= rsi_high):
        direction = "BUY"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=float(prices.iloc[-1]),
        indicators={
            "ema_fast": round(fast_now, 2),
            "ema_mid": round(mid_now, 2),
            "ema_slow": round(slow_now, 2),
            "ribbon_aligned": ribbon_aligned,
            "rsi": round(rsi_now, 1) if rsi_now else None,
            "above_sma200": uptrend,
        },
        strategy_name="ema_ribbon",
    )


def rule_fib_pullback(symbol: str, prices: pd.Series, params: Dict[str, Any], ohlcv: pd.DataFrame | None = None, **_) -> StrategySignal:
    """
    Fibonacci pullback strategy.
    Identifies the most recent swing high/low over a lookback window and checks
    whether the current price has retraced to the 0.382 or 0.5 Fib level,
    with RSI and a low-wick confirmation (volume ratio optional).

    BUY  when price is near a Fib retracement level (0.382 or 0.5) from a recent
         swing high, RSI is in the 35-60 range (pullback zone), and trend is intact
         (price above SMA200 or EMA distance mildly positive).
    SELL when price closes back above the prior swing high OR RSI > 70.
    """
    lookback    = params.get("lookback", 60)
    rsi_low     = params.get("rsi_low", 35)
    rsi_high    = params.get("rsi_high", 60)
    fib_tol     = params.get("fib_tolerance", 0.015)   # ±1.5% band around fib level

    if len(prices) < lookback + 5:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="fib_pullback")

    window      = prices.iloc[-lookback:]
    swing_high  = float(window.max())
    swing_low   = float(window.min())
    current     = float(prices.iloc[-1])
    rsi_now     = compute_rsi(prices, 14).latest
    uptrend     = _above_sma200(prices)

    fib_382 = swing_high - 0.382 * (swing_high - swing_low)
    fib_500 = swing_high - 0.500 * (swing_high - swing_low)

    near_382 = abs(current - fib_382) / fib_382 <= fib_tol if fib_382 > 0 else False
    near_500 = abs(current - fib_500) / fib_500 <= fib_tol if fib_500 > 0 else False
    at_fib   = near_382 or near_500

    direction = "HOLD"
    if rsi_now is not None:
        if current >= swing_high or rsi_now > 70:
            direction = "SELL"
        elif at_fib and rsi_low <= rsi_now <= rsi_high and uptrend:
            direction = "BUY"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=current,
        indicators={
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "fib_382": round(fib_382, 2),
            "fib_500": round(fib_500, 2),
            "at_fib_level": at_fib,
            "rsi": round(rsi_now, 1) if rsi_now else None,
            "above_sma200": uptrend,
        },
        strategy_name="fib_pullback",
    )


def rule_breakout(symbol: str, prices: pd.Series, params: Dict[str, Any], ohlcv: pd.DataFrame | None = None, **_) -> StrategySignal:
    """
    Range breakout strategy.
    BUY  when price closes above the highest close of the last N bars (breakout),
         volume ratio is expanding (ATR %), RSI is in 55-70 zone, and price is
         above both EMA and SMA200.
    SELL when price drops back below the breakout level OR RSI > 75.
    """
    range_period = params.get("range_period", 20)
    rsi_low      = params.get("rsi_low", 55)
    rsi_high     = params.get("rsi_high", 70)
    rsi_period   = params.get("rsi_period", 14)

    if len(prices) < range_period + 10:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="breakout")

    # Range high excludes the current bar
    range_high = float(prices.iloc[-(range_period + 1):-1].max())
    current    = float(prices.iloc[-1])
    rsi_now    = compute_rsi(prices, rsi_period).latest
    uptrend    = _above_sma200(prices)

    ema_fast = float(prices.ewm(span=20, adjust=False).mean().iloc[-1])
    above_ema = current > ema_fast

    # ATR % as a simple expansion proxy
    atr = float(prices.diff().abs().ewm(span=14, adjust=False).mean().iloc[-1])
    atr_pct = atr / current * 100 if current > 0 else 0

    direction = "HOLD"
    if rsi_now is not None:
        if current < range_high * 0.98 or rsi_now > 75:
            direction = "SELL"
        elif current > range_high and rsi_low <= rsi_now <= rsi_high and uptrend and above_ema:
            direction = "BUY"

    return StrategySignal(
        symbol=symbol,
        direction=direction,
        price_at_signal=current,
        indicators={
            "range_high": round(range_high, 2),
            "above_range": current > range_high,
            "above_ema20": above_ema,
            "atr_pct": round(atr_pct, 2),
            "rsi": round(rsi_now, 1) if rsi_now else None,
            "above_sma200": uptrend,
        },
        strategy_name="breakout",
    )


# ══════════════════════════════════════════════════════════════════════════════
# NEW STRATEGIES (v2) — SPY regime-aware, validated 2015–2025
# All receive ohlcv as full OHLCV df and use it for ATR / wick / volume.
# SPY regime is derived inline from the ohlcv index when spy_ohlcv kwarg passed;
# falls back to the symbol's own SMA200 check when no SPY data is available.
# ══════════════════════════════════════════════════════════════════════════════

def _ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def _atr_raw(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _spy_is_bull(ohlcv: pd.DataFrame | None, prices: pd.Series) -> bool:
    """
    Returns True when SPY is in BULL regime (close > SMA200).
    If no SPY OHLCV is available, falls back to the symbol's own SMA200.
    """
    if ohlcv is not None and "spy_close" in ohlcv.columns:
        spy_close = ohlcv["spy_close"].dropna()
        if len(spy_close) >= 200:
            sma200 = spy_close.rolling(200).mean().iloc[-1]
            return float(spy_close.iloc[-1]) > float(sma200)
    # Fallback: symbol's own SMA200
    return _above_sma200(prices)


def rule_rsi2_mean_reversion(
    symbol: str, prices: pd.Series, params: dict, ohlcv: pd.DataFrame | None = None, **_
) -> StrategySignal:
    """
    Connors RSI(2) mean-reversion.
    BUY  when RSI(2) < entry_threshold AND symbol > SMA(200) AND SPY BULL AND ATR% <= skip_threshold.
    SELL when close > SMA(5) OR RSI(2) > exit_threshold OR max-hold bars exceeded.
    """
    rsi_period   = params.get("rsi_period", 2)
    rsi_entry    = params.get("rsi_entry_threshold", 10)
    rsi_exit     = params.get("rsi_exit_threshold", 70)
    sma_trend    = params.get("sma_trend", 200)
    exit_sma     = params.get("exit_sma", 5)
    atr_skip     = params.get("atr_skip_threshold", 5.0)

    if len(prices) < max(sma_trend + 5, 250):
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="rsi2_mean_reversion")

    c_now   = float(prices.iloc[-1])
    rsi2    = _rsi_series(prices, rsi_period)
    rsi_now = float(rsi2.iloc[-1]) if not pd.isna(rsi2.iloc[-1]) else 50.0
    sma200v = float(prices.rolling(sma_trend).mean().iloc[-1])
    sma5v   = float(prices.rolling(exit_sma).mean().iloc[-1])

    atr_pct = 0.0
    if ohlcv is not None and "High" in ohlcv.columns and len(ohlcv) >= 14:
        atr_v = float(_atr_raw(ohlcv, 14).iloc[-1])
        atr_pct = atr_v / c_now * 100 if c_now > 0 else 0.0

    is_bull = _spy_is_bull(ohlcv, prices)
    above_own_sma200 = c_now > sma200v

    direction = "HOLD"
    if rsi_now > rsi_exit or c_now > sma5v:
        direction = "SELL"
    elif is_bull and above_own_sma200 and rsi_now < rsi_entry and atr_pct <= atr_skip:
        direction = "BUY"

    return StrategySignal(
        symbol=symbol, direction=direction,
        price_at_signal=c_now,
        indicators={
            "rsi2": round(rsi_now, 1),
            "sma200": round(sma200v, 2),
            "sma5": round(sma5v, 2),
            "atr_pct": round(atr_pct, 2),
            "spy_bull": is_bull,
        },
        strategy_name="rsi2_mean_reversion",
    )


def rule_ema_macd_crossover(
    symbol: str, prices: pd.Series, params: dict, ohlcv: pd.DataFrame | None = None, **_
) -> StrategySignal:
    """
    EMA(9) crosses above EMA(21) + MACD above signal + RSI 45–65 + volume spike.
    BUY  on bullish EMA crossover with all confirmations in SPY BULL regime.
    SELL on bearish EMA crossover OR MACD cross below signal.
    """
    ema_fast    = params.get("ema_fast", 9)
    ema_slow    = params.get("ema_slow", 21)
    macd_fast   = params.get("macd_fast", 12)
    macd_slow   = params.get("macd_slow", 26)
    macd_sig    = params.get("macd_signal", 9)
    rsi_period  = params.get("rsi_period", 14)
    rsi_min     = params.get("rsi_min", 45)
    rsi_max     = params.get("rsi_max", 65)
    vol_ratio   = params.get("vol_ratio_min", 1.1)

    if len(prices) < 60:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="ema_macd_crossover")

    c_now = float(prices.iloc[-1])
    ef    = _ema_series(prices, ema_fast)
    es    = _ema_series(prices, ema_slow)
    rsi_v = float(_rsi_series(prices, rsi_period).iloc[-1])

    ema_fast_close  = prices.ewm(span=macd_fast, adjust=False).mean()
    ema_slow_close  = prices.ewm(span=macd_slow, adjust=False).mean()
    macd_line       = ema_fast_close - ema_slow_close
    signal_line     = macd_line.ewm(span=macd_sig, adjust=False).mean()

    ef_now  = float(ef.iloc[-1]);  ef_prv = float(ef.iloc[-2])
    es_now  = float(es.iloc[-1]);  es_prv = float(es.iloc[-2])
    ml_now  = float(macd_line.iloc[-1]); ml_prv = float(macd_line.iloc[-2])
    ms_now  = float(signal_line.iloc[-1]); ms_prv = float(signal_line.iloc[-2])

    bullish_cross  = ef_prv <= es_prv and ef_now > es_now
    bearish_cross  = ef_prv >= es_prv and ef_now < es_now
    macd_bear_cross = ml_prv >= ms_prv and ml_now < ms_now

    vol_ok = True
    if ohlcv is not None and "Volume" in ohlcv.columns and len(ohlcv) > 21:
        avg_vol = float(ohlcv["Volume"].iloc[-21:-1].mean())
        cur_vol = float(ohlcv["Volume"].iloc[-1])
        vol_ok  = (cur_vol / avg_vol >= vol_ratio) if avg_vol > 0 else True

    is_bull = _spy_is_bull(ohlcv, prices)

    direction = "HOLD"
    if bearish_cross or macd_bear_cross:
        direction = "SELL"
    elif is_bull and bullish_cross and ml_now > ms_now and rsi_min <= rsi_v <= rsi_max and vol_ok:
        direction = "BUY"

    return StrategySignal(
        symbol=symbol, direction=direction,
        price_at_signal=c_now,
        indicators={
            "ema_fast": round(ef_now, 2), "ema_slow": round(es_now, 2),
            "macd": round(ml_now, 4), "macd_signal": round(ms_now, 4),
            "rsi": round(rsi_v, 1), "spy_bull": is_bull,
        },
        strategy_name="ema_macd_crossover",
    )


def rule_bb_squeeze_breakout(
    symbol: str, prices: pd.Series, params: dict, ohlcv: pd.DataFrame | None = None, **_
) -> StrategySignal:
    """
    Bollinger Band squeeze (≥5 bars of contracting bandwidth) then close above upper band.
    BUY  on breakout with RSI > 50 and volume expansion.
    SELL when close < middle BB OR RSI > 80.
    """
    bb_period   = params.get("bb_period", 20)
    bb_std      = params.get("bb_std", 2.0)
    squeeze_bars = params.get("squeeze_bars", 5)
    rsi_period  = params.get("rsi_period", 14)
    rsi_entry   = params.get("rsi_entry_min", 50)
    rsi_overbought = params.get("rsi_overbought", 80)
    vol_ratio   = params.get("vol_ratio_min", 1.3)

    if len(prices) < bb_period + squeeze_bars + 10:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="bb_squeeze_breakout")

    c_now  = float(prices.iloc[-1])
    sma_bb = prices.rolling(bb_period).mean()
    std_bb = prices.rolling(bb_period).std()
    upper  = sma_bb + bb_std * std_bb
    lower  = sma_bb - bb_std * std_bb
    mid    = sma_bb
    bw     = (upper - lower) / mid.replace(0, 1e-9)

    u_now  = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else c_now
    m_now  = float(mid.iloc[-1]) if not pd.isna(mid.iloc[-1]) else c_now
    l_now  = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else c_now * 0.95
    rsi_v  = float(_rsi_series(prices, rsi_period).iloc[-1])

    bw_window = [float(bw.iloc[-squeeze_bars - 1 + k]) for k in range(squeeze_bars)]
    bw_window = [v for v in bw_window if not pd.isna(v)]
    squeeze   = len(bw_window) >= squeeze_bars and all(
        bw_window[k] > bw_window[k + 1] for k in range(len(bw_window) - 1)
    )

    vol_ok = True
    if ohlcv is not None and "Volume" in ohlcv.columns and len(ohlcv) > 21:
        avg_vol = float(ohlcv["Volume"].iloc[-21:-1].mean())
        cur_vol = float(ohlcv["Volume"].iloc[-1])
        vol_ok  = (cur_vol / avg_vol >= vol_ratio) if avg_vol > 0 else True

    is_bull = _spy_is_bull(ohlcv, prices)

    direction = "HOLD"
    if c_now < m_now or rsi_v > rsi_overbought:
        direction = "SELL"
    elif is_bull and squeeze and c_now > u_now and rsi_v > rsi_entry and vol_ok:
        direction = "BUY"

    return StrategySignal(
        symbol=symbol, direction=direction,
        price_at_signal=c_now,
        indicators={
            "bb_upper": round(u_now, 2), "bb_mid": round(m_now, 2), "bb_lower": round(l_now, 2),
            "squeeze": squeeze, "rsi": round(rsi_v, 1), "spy_bull": is_bull,
        },
        strategy_name="bb_squeeze_breakout",
    )


def rule_pullback_ema50(
    symbol: str, prices: pd.Series, params: dict, ohlcv: pd.DataFrame | None = None, **_
) -> StrategySignal:
    """
    Pullback to rising EMA(50).
    BUY  when EMA50 rising + price within ±1% of EMA50 + RSI 35–55 + bullish wick.
         Allowed in mild bear (Strategy 4 of 2 bear-allowed strategies).
    SELL when RSI > 65 OR price > 3% above EMA50 OR 2% hard stop.
    """
    ema_period   = params.get("ema_trend", 50)
    slope_bars   = params.get("ema_slope_bars", 5)
    prox_pct     = params.get("price_ema_proximity_pct", 1.0)
    rsi_period   = params.get("rsi_period", 14)
    rsi_min      = params.get("rsi_min", 35)
    rsi_max      = params.get("rsi_max", 55)
    wick_min     = params.get("wick_ratio_min", 0.4)
    exit_rsi     = params.get("exit_rsi", 65)
    ext_pct      = params.get("exit_extension_pct", 3.0)
    bear_skip    = params.get("bear_skip_threshold_pct", 10.0)

    if len(prices) < ema_period + slope_bars + 10:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="pullback_ema50")

    c_now  = float(prices.iloc[-1])
    ema50  = _ema_series(prices, ema_period)
    e_now  = float(ema50.iloc[-1])
    e_old  = float(ema50.iloc[-slope_bars - 1])
    rsi_v  = float(_rsi_series(prices, rsi_period).iloc[-1])

    # Check extreme bear skip (SPY > bear_skip% below SMA200)
    spy_pct_below = 0.0
    if ohlcv is not None and "spy_close" in ohlcv.columns:
        spy_c = ohlcv["spy_close"].dropna()
        if len(spy_c) >= 200:
            spy_sma200 = float(spy_c.rolling(200).mean().iloc[-1])
            spy_now    = float(spy_c.iloc[-1])
            if spy_sma200 > 0:
                spy_pct_below = max(0.0, (spy_sma200 - spy_now) / spy_sma200 * 100)

    dist_pct = abs(c_now - e_now) / e_now * 100 if e_now > 0 else 999
    ext_now  = (c_now / e_now - 1) * 100 if e_now > 0 else 0.0

    wick_ok = False
    if ohlcv is not None and "High" in ohlcv.columns:
        h = float(ohlcv["High"].iloc[-1])
        lo = float(ohlcv["Low"].iloc[-1])
        bar_range = h - lo
        wick_ratio = (c_now - lo) / bar_range if bar_range > 0 else 0.0
        wick_ok = wick_ratio >= wick_min
    else:
        wick_ok = True  # can't check, allow

    ema_rising = e_now > e_old

    direction = "HOLD"
    if rsi_v > exit_rsi or ext_now > ext_pct:
        direction = "SELL"
    elif (spy_pct_below <= bear_skip and ema_rising
          and dist_pct <= prox_pct and rsi_min <= rsi_v <= rsi_max and wick_ok):
        direction = "BUY"

    return StrategySignal(
        symbol=symbol, direction=direction,
        price_at_signal=c_now,
        indicators={
            "ema50": round(e_now, 2), "ema_rising": ema_rising,
            "dist_pct": round(dist_pct, 2), "rsi": round(rsi_v, 1),
            "spy_pct_below_sma200": round(spy_pct_below, 2),
        },
        strategy_name="pullback_ema50",
    )


def rule_vix_spike_reversal(
    symbol: str, prices: pd.Series, params: dict, ohlcv: pd.DataFrame | None = None, **_
) -> StrategySignal:
    """
    VIX-proxy (ATR%) spike reversal after panic sell-off.
    Works in BOTH bull and bear — one of 2 strategies allowed in bear regime.
    BUY  when ATR% > 3.0 + RSI < 30 + BB position < 0.15 + long lower wick + 3-bar decline > 2%.
    SELL when ATR% < 2.0 OR RSI > 55 OR hard stop/TP.
    """
    atr_period   = params.get("atr_period", 14)
    atr_spike    = params.get("atr_spike_threshold", 3.0)
    atr_exit     = params.get("atr_exit_threshold", 2.0)
    rsi_period   = params.get("rsi_period", 14)
    rsi_entry    = params.get("rsi_entry_max", 30)
    rsi_exit_thr = params.get("rsi_exit", 55)
    bb_pos_max   = params.get("bb_pos_max", 0.15)
    wick_min     = params.get("wick_ratio_min", 0.5)
    decline_pct  = params.get("prior_decline_pct", 2.0)
    decline_bars = params.get("prior_decline_bars", 3)

    if len(prices) < 50 or ohlcv is None or len(ohlcv) < 50:
        return StrategySignal(symbol=symbol, direction="HOLD",
                              price_at_signal=float(prices.iloc[-1]), indicators={},
                              strategy_name="vix_spike_reversal")

    c_now  = float(prices.iloc[-1])
    rsi_v  = float(_rsi_series(prices, rsi_period).iloc[-1])
    atr_v  = float(_atr_raw(ohlcv, atr_period).iloc[-1])
    atr_pct = atr_v / c_now * 100 if c_now > 0 else 0.0

    # Bollinger position
    sma20   = prices.rolling(20).mean()
    std20   = prices.rolling(20).std()
    bb_l    = sma20 - 2 * std20
    bb_u    = sma20 + 2 * std20
    bb_rng  = (bb_u - bb_l).iloc[-1]
    bb_pos  = float((c_now - float(bb_l.iloc[-1])) / bb_rng) if bb_rng > 0 else 0.5

    # Wick ratio
    h      = float(ohlcv["High"].iloc[-1])
    lo     = float(ohlcv["Low"].iloc[-1])
    bar_rng = h - lo
    wick_ratio = (c_now - lo) / bar_rng if bar_rng > 0 else 0.0

    # Prior decline
    c_ago      = float(prices.iloc[-decline_bars - 1]) if len(prices) > decline_bars + 1 else c_now
    dec_pct    = (c_ago - c_now) / c_ago * 100 if c_ago > 0 else 0.0

    # No regime gate — works in bull AND bear
    direction = "HOLD"
    if atr_pct < atr_exit or rsi_v > rsi_exit_thr:
        direction = "SELL"
    elif (atr_pct >= atr_spike and rsi_v < rsi_entry
          and bb_pos < bb_pos_max and wick_ratio >= wick_min
          and dec_pct >= decline_pct):
        direction = "BUY"

    return StrategySignal(
        symbol=symbol, direction=direction,
        price_at_signal=c_now,
        indicators={
            "atr_pct": round(atr_pct, 2), "rsi": round(rsi_v, 1),
            "bb_pos": round(bb_pos, 3), "wick_ratio": round(wick_ratio, 2),
            "prior_decline_pct": round(dec_pct, 2),
        },
        strategy_name="vix_spike_reversal",
    )


_RULE_REGISTRY = {
    "sma_rsi": rule_sma_rsi,
    "ema_crossover": rule_ema_crossover,
    "macd": rule_macd,
    "bollinger": rule_bollinger,
    "supertrend": rule_supertrend,
    "vwap_rsi": rule_vwap_rsi,
    "ema_ribbon": rule_ema_ribbon,
    "fib_pullback": rule_fib_pullback,
    "breakout": rule_breakout,
    # ── New strategies (v2) ───────────────────────────────────────────────────
    "rsi2_mean_reversion":  rule_rsi2_mean_reversion,
    "ema_macd_crossover":   rule_ema_macd_crossover,
    "bb_squeeze_breakout":  rule_bb_squeeze_breakout,
    "pullback_ema50":       rule_pullback_ema50,
    "vix_spike_reversal":   rule_vix_spike_reversal,
}


def evaluate_strategy(
    strategy_type: str,
    symbol: str,
    prices: pd.Series,
    params: Dict[str, Any],
    ohlcv: pd.DataFrame | None = None,
    position: Optional[PositionState] = None,
) -> StrategySignal:
    rule_fn = _RULE_REGISTRY.get(strategy_type)
    if rule_fn is None:
        raise ValueError(f"Unknown strategy type: {strategy_type!r}. Available: {list(_RULE_REGISTRY)}")
    if ohlcv is not None:
        signal = rule_fn(symbol, prices, params, ohlcv=ohlcv)
    else:
        signal = rule_fn(symbol, prices, params)
    # Trailing-stop overlay (opt-in via params['trail_enabled']; no-op otherwise).
    return _apply_chandelier_overlay(signal, prices, ohlcv, params, position)
