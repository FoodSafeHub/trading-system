"""Native-signal pre-check for the day-trading scanner.

Runs the 4 audited intraday strategies (BollingerMomentum, SupertrendTrend,
EMAMomentum, ORBBreakout) against a small list of *already-ranked* scanner
candidates and reports which strategies, if any, have an accepted signal
right now.

This is intentionally a thin wrapper — it does NOT replicate the autotrader's
NativeStrategyEntry filter stack (RR floor, stale-bar gate, confidence floor,
direction_mode). The scanner's job is "would I trade this if asked?", not
"would I trade this with my exact risk knobs?". The autotrader applies its own
filters at flip_on time, so a `native_signal_active=True` here is a strong hint,
not a promise of execution.

Cost-shape: 2 yfinance intraday fetches (5m + 15m) per symbol, plus running the
4 strategies' generate_signals(). The full pre-check on K=10 candidates is
typically 10-20s — small relative to the scanner itself.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.services.strategy.daytrading.autotrader.native_entry import (
    SUPPORTED_NATIVE_STRATEGIES,
    map_brain_state_to_regime,
)
from app.services.strategy.daytrading.strategies import STRATEGY_MAP

logger = logging.getLogger(__name__)


@dataclass
class NativeSignalCheck:
    """Per-symbol pre-check result."""

    symbol: str
    native_signal_active: bool = False
    active_native_strategies: list[str] = field(default_factory=list)
    best_native_strategy: str | None = None
    best_native_side: str | None = None          # "BUY" | "SELL" | None
    best_native_confidence: float | None = None
    regime: str | None = None
    error: str | None = None                     # "no_bars" | "exception:..." | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_signal_active": self.native_signal_active,
            "active_native_strategies": list(self.active_native_strategies),
            "best_native_strategy": self.best_native_strategy,
            "best_native_side": self.best_native_side,
            "best_native_confidence": (
                round(self.best_native_confidence, 3)
                if self.best_native_confidence is not None else None
            ),
            "native_precheck_regime": self.regime,
            "native_precheck_error": self.error,
        }


def _check_one(symbol: str, regime: str) -> NativeSignalCheck:
    """Fetch 5m+15m bars and run the 4 audited strategies for a single symbol."""
    # Import inside the worker so the scanner module's top-level import surface
    # stays small and we don't pay for runner.py until pre-check actually fires.
    from app.services.strategy.daytrading.runner import fetch_intraday

    try:
        df_5m = fetch_intraday(symbol, interval="5m", period="5d")
        df_15m = fetch_intraday(symbol, interval="15m", period="60d")
    except Exception as e:
        return NativeSignalCheck(symbol=symbol, regime=regime, error=f"fetch:{type(e).__name__}")

    if df_5m is None or len(df_5m) < 15:
        return NativeSignalCheck(symbol=symbol, regime=regime, error="no_bars")
    if df_15m is None:
        df_15m = pd.DataFrame()

    active: list[tuple[str, str, float]] = []   # (strategy, side, confidence)
    for sname in SUPPORTED_NATIVE_STRATEGIES:
        strat = STRATEGY_MAP.get(sname)
        if strat is None:
            continue
        try:
            sigs = strat.generate_signals(
                df_5m=df_5m,
                df_15m=df_15m,
                symbol=symbol,
                config=None,
                regime=regime,
            )
        except Exception as e:
            logger.debug("[native_precheck] %s %s raised: %s", symbol, sname, e)
            continue
        if not sigs:
            continue
        # Only consider the most-recent emitted signal — older bars are stale.
        sig = sigs[-1]
        side = (sig.direction or "").upper()
        if side not in ("BUY", "SELL"):
            continue
        conf = float(getattr(sig, "confidence", 0.0) or 0.0)
        active.append((sname, side, conf))

    if not active:
        return NativeSignalCheck(symbol=symbol, regime=regime)

    # Winner = highest-confidence accepted signal, matching NativeStrategyEntry.
    winner = max(active, key=lambda t: t[2])
    return NativeSignalCheck(
        symbol=symbol,
        native_signal_active=True,
        active_native_strategies=[name for name, _, _ in active],
        best_native_strategy=winner[0],
        best_native_side=winner[1],
        best_native_confidence=winner[2],
        regime=regime,
    )


def run_native_precheck(
    symbols: list[str],
    market_state: str | None,
    *,
    max_workers: int = 4,
) -> dict[str, NativeSignalCheck]:
    """Run the native-signal pre-check on a small symbol list.

    Parameters
    ----------
    symbols : list of tickers — should already be the scanner's top-K, not the
        full universe. The cost is linear in len(symbols).
    market_state : the brain's MarketStateResult.state (TREND_UP / TREND_DOWN /
        CHOPPY / HIGH_VOL / NEWS_RISK / UNKNOWN) or None. Mapped to the legacy
        regime string (BULL_OPEN / BEAR_OPEN / CHOPPY) that the strategies'
        generate_signals() expects.
    max_workers : parallel fetchers. Kept small (4) — yfinance's intraday
        endpoint rate-limits aggressively under higher concurrency.

    Returns
    -------
    dict mapping symbol → NativeSignalCheck. Symbols that error out get a
    NativeSignalCheck with native_signal_active=False and `error` populated.
    """
    if not symbols:
        return {}

    regime = map_brain_state_to_regime(market_state)
    workers = max(1, min(max_workers, len(symbols)))
    results: dict[str, NativeSignalCheck] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_check_one, sym, regime): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                results[sym] = fut.result()
            except Exception as e:
                logger.warning("[native_precheck] worker for %s failed: %s", sym, e)
                results[sym] = NativeSignalCheck(
                    symbol=sym, regime=regime, error=f"worker:{type(e).__name__}",
                )

    n_active = sum(1 for r in results.values() if r.native_signal_active)
    logger.info(
        "[native_precheck] %d/%d symbols have active native signals (regime=%s)",
        n_active, len(results), regime,
    )
    return results
