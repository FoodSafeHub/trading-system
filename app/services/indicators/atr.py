from __future__ import annotations

import pandas as pd

from app.services.indicators.base import IndicatorResult


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> IndicatorResult:
    """Average True Range — measures volatility for stop placement."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return IndicatorResult("ATR", atr, {"period": period})
