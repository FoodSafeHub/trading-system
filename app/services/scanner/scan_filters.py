from __future__ import annotations

"""
Liquidity and price filters applied before running strategies.
Rejects symbols that are too cheap, too thinly traded, or missing data.
"""

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    passed: bool
    symbol: str
    price: float | None = None
    avg_volume: float | None = None
    reason: str = ""


def apply_filters(
    symbol: str,
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_avg_volume: float = 500_000.0,
    min_bars: int = 60,
) -> FilterResult:
    """
    Run all pre-strategy filters on an OHLCV DataFrame.
    Returns FilterResult indicating pass/fail and why.
    """
    if df is None or df.empty:
        return FilterResult(passed=False, symbol=symbol, reason="No price data returned")

    if len(df) < min_bars:
        return FilterResult(
            passed=False, symbol=symbol,
            reason=f"Insufficient history: {len(df)} bars (need {min_bars})",
        )

    last_price = float(df["Close"].iloc[-1])
    if last_price < min_price:
        return FilterResult(
            passed=False, symbol=symbol, price=last_price,
            reason=f"Price ${last_price:.2f} below minimum ${min_price:.2f}",
        )

    # Average daily volume over last 20 trading days
    vol_col = "Volume" if "Volume" in df.columns else None
    avg_vol: float | None = None
    if vol_col:
        avg_vol = float(df[vol_col].iloc[-20:].mean())
        if avg_vol < min_avg_volume:
            return FilterResult(
                passed=False, symbol=symbol, price=last_price, avg_volume=avg_vol,
                reason=f"Avg volume {avg_vol:,.0f} below minimum {min_avg_volume:,.0f}",
            )

    return FilterResult(passed=True, symbol=symbol, price=last_price, avg_volume=avg_vol)
