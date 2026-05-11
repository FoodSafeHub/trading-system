from __future__ import annotations

"""
Market data provider for historical OHLCV data used by the strategy engine.

Currently wraps yfinance for local development / paper trading.
In live mode, the active broker's get_quotes() method should be preferred
for real-time quotes, while yfinance provides historical bars for indicators.

TODO: Add a broker-native historical data endpoint when Schwab makes one available.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_cache: dict[str, pd.DataFrame] = {}


def get_price_series(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch historical closing prices for a symbol via yfinance.
    Returns a pd.Series of closing prices indexed by date.
    """
    cache_key = f"{symbol}:{period}:{interval}"
    if use_cache and cache_key in _cache:
        df = _cache[cache_key]
    else:
        logger.debug("[market_data] Fetching %s period=%s interval=%s", symbol, period, interval)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No price data returned for {symbol!r}")
        _cache[cache_key] = df

    return df["Close"].dropna()


def get_ohlcv(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Return full OHLCV DataFrame for a symbol."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol!r}")
    return df


def clear_cache() -> None:
    _cache.clear()
