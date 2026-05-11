from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.services.indicators.base import IndicatorResult


@dataclass
class MACDResult:
    macd: IndicatorResult
    signal: IndicatorResult
    histogram: IndicatorResult

    @property
    def latest_macd(self) -> float | None:
        return self.macd.latest

    @property
    def latest_signal(self) -> float | None:
        return self.signal.latest

    @property
    def latest_histogram(self) -> float | None:
        return self.histogram.latest

    def crossover_signal(self) -> str:
        """BUY on bullish crossover (MACD crosses above signal), SELL on bearish."""
        macd_vals = self.macd.values.dropna()
        sig_vals = self.signal.values.dropna()
        if len(macd_vals) < 2 or len(sig_vals) < 2:
            return "HOLD"

        macd_now, macd_prev = float(macd_vals.iloc[-1]), float(macd_vals.iloc[-2])
        sig_now, sig_prev = float(sig_vals.iloc[-1]), float(sig_vals.iloc[-2])

        if macd_prev <= sig_prev and macd_now > sig_now:
            return "BUY"
        if macd_prev >= sig_prev and macd_now < sig_now:
            return "SELL"
        return "HOLD"


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    if len(prices) < slow + signal:
        raise ValueError(f"MACD({fast},{slow},{signal}) requires at least {slow + signal} data points")

    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return MACDResult(
        macd=IndicatorResult("MACD", macd_line, {"fast": fast, "slow": slow}),
        signal=IndicatorResult("MACD_signal", signal_line, {"signal": signal}),
        histogram=IndicatorResult("MACD_hist", histogram),
    )
