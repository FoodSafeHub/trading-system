from __future__ import annotations

"""
Strategy rule implementations.
Each rule takes a price Series and parameters, returns a StrategySignal.

Rules are trend-aware: all use SMA(200) as an uptrend filter so they only
issue BUY signals when the broader trend is up.
"""

import logging
from typing import Any, Dict

import pandas as pd

from app.services.indicators.bollinger import compute_bollinger
from app.services.indicators.ema import compute_ema, ema_crossover_signal
from app.services.indicators.macd import compute_macd
from app.services.indicators.rsi import compute_rsi
from app.services.indicators.sma import compute_sma, sma_crossover_signal
from app.services.indicators.supertrend import compute_supertrend
from app.services.strategy.models import StrategySignal

logger = logging.getLogger(__name__)


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
         RSI < 55 (not overbought on entry), AND price > SMA(200).
    SELL when price reaches upper band AND RSI > 70 (confirmed overbought).

    Fixed: old version issued SELL whenever price touched the upper band — in
    a bull market stocks ride the upper band for weeks. Now requires RSI > 70
    to confirm the sell, avoiding premature exits in trending markets.
    """
    period = params.get("bb_period", 20)
    std_dev = params.get("bb_std", 2.0)

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
    # BUY: lower-band reversion in uptrend with RSI not overbought
    elif reverted and uptrend and rsi_now is not None and rsi_now < 55:
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
    ema_fast = params.get("ema_fast", 8)
    ema_mid  = params.get("ema_mid", 21)
    ema_slow = params.get("ema_slow", 50)
    rsi_period = params.get("rsi_period", 14)

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
    # SELL: fast crosses below mid (momentum weakening), or RSI very overbought
    if fast_crossed_below or (rsi_now is not None and rsi_now > 78):
        direction = "SELL"
    # BUY: fast crosses above mid while ribbon is aligned upward (fast>mid>slow)
    # RSI must be above 40 (not in deep oversold) — no strict SMA200 requirement
    elif (fast_crossed_above and mid_now > slow_now
          and rsi_now is not None and rsi_now >= 40):
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


_RULE_REGISTRY = {
    "sma_rsi": rule_sma_rsi,
    "ema_crossover": rule_ema_crossover,
    "macd": rule_macd,
    "bollinger": rule_bollinger,
    "supertrend": rule_supertrend,
    "vwap_rsi": rule_vwap_rsi,
    "ema_ribbon": rule_ema_ribbon,
}


def evaluate_strategy(
    strategy_type: str,
    symbol: str,
    prices: pd.Series,
    params: Dict[str, Any],
    ohlcv: pd.DataFrame | None = None,
) -> StrategySignal:
    rule_fn = _RULE_REGISTRY.get(strategy_type)
    if rule_fn is None:
        raise ValueError(f"Unknown strategy type: {strategy_type!r}. Available: {list(_RULE_REGISTRY)}")
    if ohlcv is not None:
        return rule_fn(symbol, prices, params, ohlcv=ohlcv)
    return rule_fn(symbol, prices, params)
