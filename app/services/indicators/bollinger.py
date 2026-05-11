from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.indicators.base import IndicatorResult


@dataclass
class BollingerResult:
    middle: IndicatorResult  # SMA
    upper: IndicatorResult
    lower: IndicatorResult
    bandwidth: IndicatorResult
    percent_b: IndicatorResult  # position of price within the bands

    def signal(self, current_price: float) -> str:
        """
        Simple mean-reversion signal:
        BUY when price touches/crosses below lower band,
        SELL when price touches/crosses above upper band.
        """
        lower = self.lower.latest
        upper = self.upper.latest
        if lower is None or upper is None:
            return "HOLD"
        if current_price <= lower:
            return "BUY"
        if current_price >= upper:
            return "SELL"
        return "HOLD"


def compute_bollinger(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> BollingerResult:
    if len(prices) < period:
        raise ValueError(f"Bollinger({period}) requires at least {period} data points")

    middle = prices.rolling(window=period).mean()
    rolling_std = prices.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    bandwidth = (upper - lower) / middle
    percent_b = (prices - lower) / (upper - lower)

    return BollingerResult(
        middle=IndicatorResult("BB_middle", middle, {"period": period, "std": std_dev}),
        upper=IndicatorResult("BB_upper", upper),
        lower=IndicatorResult("BB_lower", lower),
        bandwidth=IndicatorResult("BB_bandwidth", bandwidth),
        percent_b=IndicatorResult("BB_pct_b", percent_b),
    )
