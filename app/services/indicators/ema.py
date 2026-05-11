from __future__ import annotations

import pandas as pd

from app.services.indicators.base import IndicatorResult


def compute_ema(prices: pd.Series, period: int) -> IndicatorResult:
    """Exponential Moving Average over `period` bars."""
    if len(prices) < period:
        raise ValueError(f"EMA({period}) requires at least {period} data points, got {len(prices)}")
    values = prices.ewm(span=period, adjust=False).mean()
    return IndicatorResult(name=f"EMA_{period}", values=values, metadata={"period": period})


def ema_crossover_signal(prices: pd.Series, fast: int, slow: int) -> str:
    """Return 'BUY', 'SELL', or 'HOLD' based on EMA crossover."""
    if len(prices) < slow + 1:
        return "HOLD"

    fast_ema = compute_ema(prices, fast).values
    slow_ema = compute_ema(prices, slow).values

    fast_now, slow_now = fast_ema.iloc[-1], slow_ema.iloc[-1]
    fast_prev, slow_prev = fast_ema.iloc[-2], slow_ema.iloc[-2]

    if any(pd.isna(v) for v in [fast_now, slow_now, fast_prev, slow_prev]):
        return "HOLD"

    if fast_prev <= slow_prev and fast_now > slow_now:
        return "BUY"
    if fast_prev >= slow_prev and fast_now < slow_now:
        return "SELL"
    return "HOLD"
