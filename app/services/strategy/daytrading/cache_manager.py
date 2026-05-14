"""
BarCache — thread-safe in-memory bar cache for live trading.

Stores the latest N bars per symbol per timeframe, updated by MarketDataPoller.
Any strategy or UI component can read from here without triggering a network call.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SymbolCache:
    symbol: str
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)   # "1m"/"5m"/"15m" → df
    last_update: float = 0.0      # unix timestamp
    fetch_errors: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_update if self.last_update else float("inf")

    @property
    def is_stale(self) -> bool:
        return self.age_seconds > 120    # stale after 2 minutes

    def to_status(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframes": list(self.bars.keys()),
            "last_update": self.last_update,
            "age_seconds": round(self.age_seconds, 1),
            "stale": self.is_stale,
            "fetch_errors": self.fetch_errors,
        }


class BarCache:
    """
    Thread-safe multi-symbol bar cache.
    Read with get_bars(symbol, timeframe).
    Write with update(symbol, timeframe, df).
    """

    # Maximum bars retained per timeframe (memory control)
    MAX_BARS: dict[str, int] = {"1m": 390, "5m": 120, "15m": 60, "1d": 252}

    def __init__(self):
        self._cache: dict[str, SymbolCache] = {}
        self._lock = threading.RLock()

    def update(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        sym = symbol.upper()
        max_bars = self.MAX_BARS.get(timeframe, 100)
        df = df.tail(max_bars).copy()

        with self._lock:
            if sym not in self._cache:
                self._cache[sym] = SymbolCache(symbol=sym)
            sc = self._cache[sym]
            sc.bars[timeframe] = df
            sc.last_update = time.time()
            sc.fetch_errors = 0

    def get_bars(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        with self._lock:
            sc = self._cache.get(symbol.upper())
            if sc is None:
                return None
            return sc.bars.get(timeframe)

    def get_age(self, symbol: str) -> float:
        with self._lock:
            sc = self._cache.get(symbol.upper())
            return sc.age_seconds if sc else float("inf")

    def is_stale(self, symbol: str) -> bool:
        with self._lock:
            sc = self._cache.get(symbol.upper())
            return sc.is_stale if sc else True

    def record_error(self, symbol: str) -> None:
        with self._lock:
            sc = self._cache.get(symbol.upper())
            if sc:
                sc.fetch_errors += 1

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {sym: sc.to_status() for sym, sc in self._cache.items()}

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())


# Module-level singleton
_bar_cache = BarCache()


def get_bar_cache() -> BarCache:
    return _bar_cache
