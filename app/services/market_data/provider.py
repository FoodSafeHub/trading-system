from __future__ import annotations

"""
Market data provider for historical OHLCV data used by the strategy engine.

Currently wraps yfinance for local development / paper trading.
In live mode, the active broker's get_quotes() method should be preferred
for real-time quotes, while yfinance provides historical bars for indicators.

TODO: Add a broker-native historical data endpoint when Schwab makes one available.
"""

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from app.services.markets import is_india_symbol, yf_symbol

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 3600   # 1 hour
_CACHE_MAX_ENTRIES = 150

_cache: dict[str, pd.DataFrame] = {}
_cache_ts: dict[str, float] = {}   # timestamp each entry was populated


def _cache_get(key: str) -> Optional[pd.DataFrame]:
    if key not in _cache:
        return None
    if time.time() - _cache_ts.get(key, 0) > _CACHE_TTL_SECONDS:
        del _cache[key]
        del _cache_ts[key]
        return None
    return _cache[key]


def _cache_set(key: str, df: pd.DataFrame) -> None:
    if len(_cache) >= _CACHE_MAX_ENTRIES:
        # Evict the oldest entry
        oldest = min(_cache_ts, key=_cache_ts.get)
        del _cache[oldest]
        del _cache_ts[oldest]
    _cache[key] = df
    _cache_ts[key] = time.time()


def _fetch_india(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """OHLCV for an India (NSE) symbol.

    Prefers the Upstox feed when configured; otherwise falls back to yfinance
    with the '.NS' suffix so India scans/backtests work even before an Upstox
    app is wired up. yfinance India data is delayed/EOD-grade, not a live feed.
    """
    # 1) Upstox (live-grade, when keys + a daily token are present).
    try:
        from app.services.marketdata import upstox_data
        if upstox_data.is_configured():
            df = upstox_data.fetch_bars(symbol, interval=interval, period=period)
            if df is not None and not df.empty:
                return df
    except Exception as exc:  # never let the data feed take down a scan
        logger.debug("[market_data] Upstox fetch failed for %s: %s", symbol, exc)

    # 2) yfinance '.NS' fallback.
    ticker = yf.Ticker(yf_symbol(symbol))
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No India price data for {symbol!r} (Upstox + yfinance.NS both empty)")
    # yfinance often appends a placeholder row for the in-progress/just-closed
    # IST session whose OHLC are all NaN. Backtest/scan engines choke on it
    # ("inf or nan" / "single positional indexer is out-of-bounds"), so drop
    # any row missing a Close before handing the frame off.
    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"No valid India bars for {symbol!r} after dropping NaN rows")
    df.attrs["source"] = "yfinance.NS"
    return df


def get_price_series(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch historical closing prices for a symbol.
    US symbols come from yfinance; India (NSE) symbols from Upstox/yfinance.NS.
    Returns a pd.Series of closing prices indexed by date.
    """
    cache_key = f"{symbol}:{period}:{interval}"
    cached = _cache_get(cache_key) if use_cache else None
    if cached is not None:
        df = cached
    else:
        logger.debug("[market_data] Fetching %s period=%s interval=%s", symbol, period, interval)
        if is_india_symbol(symbol):
            df = _fetch_india(symbol, period, interval)
        else:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No price data returned for {symbol!r}")
        _cache_set(cache_key, df)

    return df["Close"].dropna()


def get_ohlcv(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Return full OHLCV DataFrame for a symbol. Results cached for 1 hour.

    US symbols come from yfinance; India (NSE) symbols from Upstox/yfinance.NS.
    """
    cache_key = f"ohlcv:{symbol}:{period}:{interval}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    logger.debug("[market_data] Fetching OHLCV %s period=%s interval=%s", symbol, period, interval)
    if is_india_symbol(symbol):
        df = _fetch_india(symbol, period, interval)
    else:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No OHLCV data for {symbol!r}")
    _cache_set(cache_key, df)
    return df


def clear_cache() -> None:
    _cache.clear()
    _cache_ts.clear()
