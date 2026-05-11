from __future__ import annotations

import pandas as pd

from app.services.indicators.base import IndicatorResult


def compute_rsi(prices: pd.Series, period: int = 14) -> IndicatorResult:
    """
    Relative Strength Index using Wilder's smoothing method.
    Returns values in [0, 100].
    """
    if len(prices) < period + 1:
        raise ValueError(f"RSI({period}) requires at least {period + 1} data points")

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.where(avg_loss != 0, other=float("nan"))
    rs = rs.fillna(float("inf"))
    rsi = 100 - (100 / (1 + rs))

    return IndicatorResult(name=f"RSI_{period}", values=rsi, metadata={"period": period})
