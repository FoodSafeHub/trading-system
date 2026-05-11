from __future__ import annotations

import pandas as pd
from dataclasses import dataclass


@dataclass
class SupertrendResult:
    values: pd.Series      # supertrend line values
    direction: pd.Series   # 1 = bullish (price above), -1 = bearish (price below)


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> SupertrendResult:
    """
    Supertrend indicator.
    Returns the supertrend line and direction (1=up/bullish, -1=down/bearish).
    BUY when direction flips from -1 to 1. SELL when flips from 1 to -1.
    """
    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=close.index, dtype=float)
    direction  = pd.Series(index=close.index, dtype=int)

    for i in range(1, len(close)):
        # Lower band: never decreases when in uptrend
        if lower_band.iloc[i] > lower_band.iloc[i - 1] or close.iloc[i - 1] < supertrend.iloc[i - 1]:
            lb = lower_band.iloc[i]
        else:
            lb = lower_band.iloc[i - 1]

        # Upper band: never increases when in downtrend
        if upper_band.iloc[i] < upper_band.iloc[i - 1] or close.iloc[i - 1] > supertrend.iloc[i - 1]:
            ub = upper_band.iloc[i]
        else:
            ub = upper_band.iloc[i - 1]

        if pd.isna(supertrend.iloc[i - 1]):
            supertrend.iloc[i] = lb
            direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
            # Was in downtrend
            if close.iloc[i] > ub:
                supertrend.iloc[i] = lb
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = ub
                direction.iloc[i] = -1
        else:
            # Was in uptrend
            if close.iloc[i] < lb:
                supertrend.iloc[i] = ub
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lb
                direction.iloc[i] = 1

    return SupertrendResult(values=supertrend, direction=direction)
