"""
MarketDataPoller — background thread that keeps BarCache fresh during market hours.

Polls yfinance every 30 seconds for each registered symbol.
Emits "new_bar_close" events when a new complete bar is detected.
Logs data staleness warnings automatically.

Usage:
    poller = MarketDataPoller(symbols=["SPY", "AAPL", "TSLA"])
    poller.start()
    # ... later:
    poller.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable

import pandas as pd
import yfinance as yf

from app.services.strategy.daytrading.cache_manager import BarCache, get_bar_cache
from app.services.strategy.daytrading.market_open import ET

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 30
_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MINUTE = 30
_MARKET_CLOSE_HOUR = 16


class MarketDataPoller:
    """
    Runs a background thread that polls yfinance every 30 seconds and
    writes fresh bars into BarCache.

    Callbacks fire when a new complete bar is confirmed.
    """

    TIMEFRAMES: list[tuple[str, str, str]] = [
        # (cache_key, yfinance_interval, yfinance_period)
        ("1m",  "1m",  "1d"),
        ("5m",  "5m",  "5d"),
        ("15m", "15m", "5d"),
    ]

    def __init__(
        self,
        symbols: list[str],
        cache: BarCache | None = None,
        poll_interval: int = _POLL_INTERVAL_SECONDS,
    ):
        self._symbols = [s.upper() for s in symbols]
        self._cache = cache or get_bar_cache()
        self._poll_interval = poll_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list[Callable[[str, str, pd.DataFrame], None]] = []
        # Track last bar index per symbol/timeframe to detect new bars
        self._last_bar_time: dict[tuple[str, str], pd.Timestamp | None] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("MarketDataPoller already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="MarketDataPoller",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "MarketDataPoller started — %d symbols, poll every %ds",
            len(self._symbols), self._poll_interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("MarketDataPoller stopped.")

    def add_symbol(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym not in self._symbols:
            self._symbols.append(sym)
            logger.info("Added %s to polling list.", sym)

    def remove_symbol(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym in self._symbols:
            self._symbols.remove(sym)

    def on_new_bar(self, callback: Callable[[str, str, pd.DataFrame], None]) -> None:
        """
        Register callback(symbol, timeframe, df_latest_bars).
        Called whenever a new complete bar is detected.
        """
        self._callbacks.append(callback)

    def force_refresh(self, symbol: str | None = None) -> None:
        """Trigger an immediate poll (called from UI refresh buttons)."""
        symbols = [symbol.upper()] if symbol else self._symbols
        for sym in symbols:
            self._poll_symbol(sym)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if self._is_market_hours():
                self._poll_all()
            else:
                # Outside market hours: poll daily bars only, less frequently
                self._poll_daily()

            # Sleep in short chunks to allow clean shutdown
            for _ in range(self._poll_interval * 2):
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

    def _poll_all(self) -> None:
        for sym in self._symbols:
            try:
                self._poll_symbol(sym)
            except Exception as e:
                logger.warning("Poll failed for %s: %s", sym, e)
                self._cache.record_error(sym)

    def _poll_symbol(self, symbol: str) -> None:
        for tf_key, yf_interval, yf_period in self.TIMEFRAMES:
            try:
                df = self._fetch(symbol, yf_interval, yf_period)
                if df.empty:
                    continue

                self._cache.update(symbol, tf_key, df)

                # Check for new bar
                last_bar_ts = df.index[-1]
                cache_key = (symbol, tf_key)
                prev = self._last_bar_time.get(cache_key)
                if prev is not None and last_bar_ts > prev:
                    self._fire_new_bar(symbol, tf_key, df)
                self._last_bar_time[cache_key] = last_bar_ts

            except Exception as e:
                logger.debug("Fetch error %s/%s: %s", symbol, tf_key, e)

        age = self._cache.get_age(symbol)
        if age > 120:
            logger.warning("DATA STALE: %s — last update %.0fs ago", symbol, age)

    def _poll_daily(self) -> None:
        for sym in self._symbols:
            try:
                df = self._fetch(sym, "1d", "5d")
                if not df.empty:
                    self._cache.update(sym, "1d", df)
            except Exception:
                pass

    @staticmethod
    def _fetch(symbol: str, interval: str, period: str) -> pd.DataFrame:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.to_datetime(df.index)
        if idx.tzinfo is None:
            idx = idx.tz_localize("UTC").tz_convert(ET)
        else:
            idx = idx.tz_convert(ET)
        df.index = idx
        return df

    def _fire_new_bar(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        for cb in self._callbacks:
            try:
                cb(symbol, timeframe, df)
            except Exception as e:
                logger.warning("new_bar callback error: %s", e)

    @staticmethod
    def _is_market_hours() -> bool:
        now = datetime.now(ET)
        if now.weekday() >= 5:   # Saturday / Sunday
            return False
        t = now.time()
        from datetime import time as dt_time
        return dt_time(_MARKET_OPEN_HOUR, _MARKET_OPEN_MINUTE) <= t <= dt_time(_MARKET_CLOSE_HOUR, 0)


# Module-level singleton (created on first use)
_poller: MarketDataPoller | None = None


def get_poller(symbols: list[str] | None = None) -> MarketDataPoller:
    global _poller
    if _poller is None:
        _poller = MarketDataPoller(symbols=symbols or [])
    elif symbols:
        for s in symbols:
            _poller.add_symbol(s)
    return _poller
