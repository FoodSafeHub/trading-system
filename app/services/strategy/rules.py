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
from app.services.strategy.models import StrategySignal

logger = logging.getLogger(__name__)


def _above_sma200(prices: pd.Series) -> bool:
    """True when the latest close is above the 200-bar SMA (uptrend filter)."""
    if len(prices) < 200:
        return False
    sma200 = prices.rolling(200).mean().iloc[-1]
    return float(prices.iloc[-1]) > float(sma200)


def rule_sma_rsi(symbol: str, prices: pd.Series, params: Dict[str, Any]) -> StrategySignal:
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


def rule_ema_crossover(symbol: str, prices: pd.Series, params: Dict[str, Any]) -> StrategySignal:
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


def rule_macd(symbol: str, prices: pd.Series, params: Dict[str, Any]) -> StrategySignal:
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


def rule_bollinger(symbol: str, prices: pd.Series, params: Dict[str, Any]) -> StrategySignal:
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


_RULE_REGISTRY = {
    "sma_rsi": rule_sma_rsi,
    "ema_crossover": rule_ema_crossover,
    "macd": rule_macd,
    "bollinger": rule_bollinger,
}


def evaluate_strategy(
    strategy_type: str,
    symbol: str,
    prices: pd.Series,
    params: Dict[str, Any],
) -> StrategySignal:
    rule_fn = _RULE_REGISTRY.get(strategy_type)
    if rule_fn is None:
        raise ValueError(f"Unknown strategy type: {strategy_type!r}. Available: {list(_RULE_REGISTRY)}")
    return rule_fn(symbol, prices, params)
