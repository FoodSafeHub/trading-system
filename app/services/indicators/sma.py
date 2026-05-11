from __future__ import annotations

import pandas as pd

from app.services.indicators.base import IndicatorResult


def compute_sma(prices: pd.Series, period: int) -> IndicatorResult:
    """Simple Moving Average over `period` bars."""
    if len(prices) < period:
        raise ValueError(f"SMA({period}) requires at least {period} data points, got {len(prices)}")
    values = prices.rolling(window=period).mean()
    return IndicatorResult(name=f"SMA_{period}", values=values, metadata={"period": period})


def sma_crossover_signal(prices: pd.Series, fast: int, slow: int) -> str:
    """
    Return 'BUY', 'SELL', or 'HOLD' based on SMA crossover.
    BUY  when fast SMA crosses above slow SMA on the last bar.
    SELL when fast SMA crosses below slow SMA on the last bar.
    """
    if len(prices) < slow + 1:
        return "HOLD"

    fast_sma = compute_sma(prices, fast).values
    slow_sma = compute_sma(prices, slow).values

    # Current bar
    fast_now = fast_sma.iloc[-1]
    slow_now = slow_sma.iloc[-1]
    # Previous bar
    fast_prev = fast_sma.iloc[-2]
    slow_prev = slow_sma.iloc[-2]

    if pd.isna(fast_now) or pd.isna(slow_now) or pd.isna(fast_prev) or pd.isna(slow_prev):
        return "HOLD"

    if fast_prev <= slow_prev and fast_now > slow_now:
        return "BUY"
    if fast_prev >= slow_prev and fast_now < slow_now:
        return "SELL"
    return "HOLD"
