"""
IntrabarMonitor — detects signals AS THEY FORM on 1m bars, not at 5m close.

Problem solved: ORB setup triggers at 9:45:30 AM (30 seconds into the bar).
Waiting for the 9:50 bar close to detect it loses 4.5 minutes of the move.

How it works:
  1. Polls the BarCache every 60 seconds for fresh 1m bars
  2. Runs each strategy's intrabar check against the latest 1m bar
  3. Emits signals immediately with exact formation time + time_to_close_bar
  4. Deduplicates — same setup won't re-emit within the same 5m window

Usage:
    monitor = IntrabarMonitor(symbols=["SPY", "TSLA"], cache=cache)
    monitor.on_signal(lambda sig: print(sig))
    monitor.start()
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from app.services.strategy.daytrading.cache_manager import BarCache, get_bar_cache
from app.services.strategy.daytrading.intrabar.strategies_1m import (
    check_ema_momentum_intrabar,
    check_orb_intrabar,
    check_volume_spike_intrabar,
    check_vwap_reversion_intrabar,
)
from app.services.strategy.daytrading.market_open import ET

logger = logging.getLogger(__name__)


@dataclass
class IntrabarSignal:
    symbol: str
    strategy: str
    direction: str          # "BUY" | "SELL"
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float
    signal_time: str        # exact 1m bar open time
    bar_close_time: str     # when the current 1m bar closes
    time_to_bar_close_sec: float   # seconds left in the 1m bar
    signal_bar_index: int   # index in 1m df where signal formed
    reason: str
    indicators: dict = field(default_factory=dict)
    market_state: str = "UNKNOWN"

    @property
    def is_current(self) -> bool:
        """True if signal is < 5 minutes old."""
        try:
            sig_ts = pd.Timestamp(self.signal_time)
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.tz_localize(ET)
            age = (datetime.now(ET) - sig_ts.to_pydatetime()).total_seconds()
            return age < 300
        except Exception:
            return True

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "confidence": self.confidence,
            "signal_time": self.signal_time,
            "bar_close_time": self.bar_close_time,
            "time_to_bar_close_sec": self.time_to_bar_close_sec,
            "signal_bar_index": self.signal_bar_index,
            "reason": self.reason,
            "indicators": self.indicators,
            "market_state": self.market_state,
            "is_current": self.is_current,
        }


class IntrabarMonitor:
    """
    Continuously scans 1m bars for intrabar signal formation.
    Does NOT wait for bar close — fires the moment conditions are met.
    """

    STRATEGIES = [
        ("ORBBreakout",       check_orb_intrabar),
        ("VWAPMeanReversion", check_vwap_reversion_intrabar),
        ("EMAMomentum",       check_ema_momentum_intrabar),
        ("VolumeSpikeReversal", check_volume_spike_intrabar),
    ]

    def __init__(
        self,
        symbols: list[str],
        cache: BarCache | None = None,
        check_interval: int = 60,
    ):
        self._symbols = [s.upper() for s in symbols]
        self._cache = cache or get_bar_cache()
        self._check_interval = check_interval
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list[Callable[[IntrabarSignal], None]] = []
        # Dedup: (symbol, strategy, 5m_window_start) → True
        self._emitted: dict[tuple, bool] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="IntrabarMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("IntrabarMonitor started for %s", self._symbols)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def on_signal(self, callback: Callable[[IntrabarSignal], None]) -> None:
        self._callbacks.append(callback)

    def scan_now(self) -> list[IntrabarSignal]:
        """Force an immediate scan and return all signals. Does not emit callbacks."""
        signals = []
        for sym in self._symbols:
            signals.extend(self._scan_symbol(sym, emit=False))
        return signals

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for sym in self._symbols:
                try:
                    self._scan_symbol(sym, emit=True)
                except Exception as e:
                    logger.warning("IntrabarMonitor scan error %s: %s", sym, e)

            # Clean up dedup cache older than 30 min
            self._cleanup_dedup()

            for _ in range(self._check_interval * 2):
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

    def _scan_symbol(self, symbol: str, emit: bool) -> list[IntrabarSignal]:
        df_1m = self._cache.get_bars(symbol, "1m")
        df_5m = self._cache.get_bars(symbol, "5m")

        if df_1m is None or df_1m.empty or len(df_1m) < 10:
            return []

        signals: list[IntrabarSignal] = []
        now_ts = df_1m.index[-1]

        # Current bar's time_to_close
        bar_open = now_ts
        # 1m bars close 1 minute after open
        bar_close = bar_open + pd.Timedelta(minutes=1)
        now_dt = datetime.now(ET)
        secs_to_close = max(0, (bar_close.to_pydatetime() - now_dt).total_seconds())

        for strat_name, check_fn in self.STRATEGIES:
            try:
                result = check_fn(df_1m, df_5m, symbol)
                if result is None:
                    continue

                # Dedup: don't re-emit same strategy on same 5m window
                five_min_window = _floor_to_5m(now_ts)
                dedup_key = (symbol, strat_name, five_min_window)
                if dedup_key in self._emitted:
                    continue

                sig = IntrabarSignal(
                    symbol=symbol,
                    strategy=strat_name,
                    direction=result["direction"],
                    entry_price=result["entry_price"],
                    stop_price=result["stop_price"],
                    target_price=result["target_price"],
                    confidence=result.get("confidence", 0.6),
                    signal_time=str(now_ts),
                    bar_close_time=str(bar_close),
                    time_to_bar_close_sec=secs_to_close,
                    signal_bar_index=len(df_1m) - 1,
                    reason=result.get("reason", ""),
                    indicators=result.get("indicators", {}),
                )

                self._emitted[dedup_key] = True
                signals.append(sig)

                if emit:
                    logger.info(
                        "INTRABAR SIGNAL: %s %s %s @ %.2f (%.0fs to bar close)",
                        sig.strategy, sig.symbol, sig.direction,
                        sig.entry_price, secs_to_close,
                    )
                    self._fire(sig)

            except Exception as e:
                logger.debug("Intrabar check error %s/%s: %s", strat_name, symbol, e)

        return signals

    def _fire(self, sig: IntrabarSignal) -> None:
        for cb in self._callbacks:
            try:
                cb(sig)
            except Exception as e:
                logger.warning("Intrabar signal callback error: %s", e)

    def _cleanup_dedup(self) -> None:
        """Remove dedup entries older than 30 minutes."""
        now = pd.Timestamp.now(tz=ET)
        stale_keys = [
            k for k in self._emitted
            if (now - pd.Timestamp(k[2]).tz_localize(ET) if pd.Timestamp(k[2]).tzinfo is None
                else now - pd.Timestamp(k[2])).total_seconds() > 1800
        ]
        for k in stale_keys:
            del self._emitted[k]


def _floor_to_5m(ts: pd.Timestamp) -> pd.Timestamp:
    """Floor a timestamp to the start of its 5m bar."""
    minute = (ts.minute // 5) * 5
    return ts.replace(minute=minute, second=0, microsecond=0)
